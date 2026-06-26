use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use base64::Engine;
use hipfire_config::models_dir;
use hipfire_diffusion::{
    encode_rgb_batch_png_base64, inspect_hfq, inspect_hfq_with_runtime_support,
    resize_rgb_batch_nearest, resize_rgb_batch_to_contain_fill_nearest,
    resize_rgb_batch_to_cover_nearest, DiffusionBatchOutput, DiffusionBatchRequest, DiffusionError,
    DiffusionGenerationRuntimeOptions, DiffusionHfqInspection, DiffusionImg2ImgRequest,
    DiffusionImg2ImgResizeMode, DiffusionPipeline, DiffusionProgress, DiffusionPrompt,
    RgbImageBatch,
};
use image::{ImageEncoder, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::discovery::{find_model, list_local_models, local_llm_registry};
use crate::routes::chat::{execute_blocking_chat, ChatMessage, ChatRequest};
use crate::state::{SdapiProgressState, SharedState};

const COMPAT_SAMPLER: &str = "Hipfire";

#[derive(Debug, Clone)]
struct SdapiInpaintFullResPlan {
    base_image: RgbImageBatch,
    overlay_mask: RgbImageBatch,
    paste_region: SdapiCropRegion,
    padding: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SdapiCropRegion {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[derive(Debug)]
struct SdapiPreparedImg2Img {
    init_image: RgbImageBatch,
    mask: Option<RgbImageBatch>,
    processing_dimensions: (u32, u32),
    full_res_plan: Option<SdapiInpaintFullResPlan>,
    image_fill_applied: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdGenerationRequest {
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub model: Option<String>,
    pub sampler_name: Option<String>,
    pub sampler_index: Option<String>,
    pub scheduler: Option<String>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f64>,
    pub seed: Option<i64>,
    pub subseed: Option<i64>,
    pub subseed_strength: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub original_width: Option<u32>,
    pub original_height: Option<u32>,
    pub target_width: Option<u32>,
    pub target_height: Option<u32>,
    pub crop_x: Option<u32>,
    pub crop_y: Option<u32>,
    pub batch_size: Option<u32>,
    pub n_iter: Option<u32>,
    pub send_images: Option<bool>,
    pub save_images: Option<bool>,
    pub force_task_id: Option<String>,
    pub infotext: Option<String>,
    pub init_images: Option<Vec<String>>,
    pub mask: Option<String>,
    pub mask_blur: Option<Value>,
    pub mask_blur_x: Option<Value>,
    pub mask_blur_y: Option<Value>,
    pub mask_round: Option<Value>,
    pub inpainting_mask_invert: Option<Value>,
    pub inpainting_fill: Option<Value>,
    pub inpaint_full_res: Option<Value>,
    pub inpaint_full_res_padding: Option<Value>,
    pub resize_mode: Option<u32>,
    pub include_init_images: Option<bool>,
    pub denoising_strength: Option<f64>,
    pub enable_hr: Option<bool>,
    pub firstphase_width: Option<u32>,
    pub firstphase_height: Option<u32>,
    pub hr_scale: Option<f64>,
    pub hr_upscaler: Option<String>,
    pub hr_resize_x: Option<u32>,
    pub hr_resize_y: Option<u32>,
    pub hr_second_pass_steps: Option<u32>,
    pub hr_checkpoint_name: Option<String>,
    pub hr_sampler_name: Option<String>,
    pub hr_scheduler: Option<String>,
    pub hr_prompt: Option<String>,
    pub hr_negative_prompt: Option<String>,
    pub rocm_device_id: Option<i32>,
    pub hipfire_rocm_device_id: Option<i32>,
    pub override_settings: Option<Value>,
    pub script_name: Option<String>,
    pub script_args: Option<Value>,
    pub alwayson_scripts: Option<Value>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Value>,
}

#[derive(Debug, Serialize)]
struct SdGenerationResponse {
    images: Vec<String>,
    parameters: SdGenerationRequest,
    info: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdExtrasSingleImageRequest {
    #[serde(default)]
    pub image: String,
    pub show_extras_results: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdExtrasBatchImagesRequest {
    #[serde(default, rename = "imageList")]
    pub image_list: Vec<SdExtrasFileData>,
    pub show_extras_results: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdExtrasFileData {
    #[serde(default)]
    pub data: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdInterrogateRequest {
    #[serde(default)]
    pub image: String,
    #[serde(default = "default_interrogate_model")]
    pub model: String,
}

#[derive(Debug, Serialize)]
struct SdExtrasSingleImageResponse {
    image: String,
    html_info: String,
}

#[derive(Debug, Serialize)]
struct SdExtrasBatchImagesResponse {
    images: Vec<String>,
    html_info: String,
}

#[derive(Debug, Serialize)]
struct SdInterrogateResponse {
    caption: String,
}

fn default_interrogate_model() -> String {
    "clip".to_string()
}

pub async fn post_txt2img(
    State(state): State<SharedState>,
    Json(body): Json<SdGenerationRequest>,
) -> Response {
    let body = sdapi_apply_infotext_defaults(body);
    if let Err(error) = sdapi_validate_supported_scripts(&body) {
        return diffusion_error_response(error);
    }
    execute_sd_generation(state, body, None).await
}

pub async fn post_img2img(
    State(state): State<SharedState>,
    Json(body): Json<SdGenerationRequest>,
) -> Response {
    let body = sdapi_apply_infotext_defaults(body);
    if let Err(error) = sdapi_validate_supported_scripts(&body) {
        return diffusion_error_response(error);
    }
    let images = body.init_images.clone().filter(|images| !images.is_empty());
    execute_sd_generation(state, body, images).await
}

async fn execute_sd_generation(
    state: SharedState,
    body: SdGenerationRequest,
    init_images_base64: Option<Vec<String>>,
) -> Response {
    let requested_model = sd_requested_model(&body);
    if let Some(diffusion_path) =
        resolve_diffusion_hfq_for_request(&state, requested_model.as_deref()).await
    {
        return match init_images_base64 {
            Some(images) => {
                execute_hfq_diffusion_img2img(state, diffusion_path, body, images).await
            }
            None => execute_hfq_diffusion_txt2img(state, diffusion_path, body).await,
        };
    }

    if requested_model_is_diffusers_pipeline(requested_model.as_deref()) {
        return diffusion_backend_missing_response();
    }

    let response_mode = if init_images_base64.is_some() {
        "img2img"
    } else {
        "txt2img"
    };
    let chat = sd_request_to_chat_request(
        &body,
        init_images_base64.and_then(|images| images.into_iter().next()),
    );
    match execute_blocking_chat(state, chat).await {
        Ok(result) => {
            let info = json!({
                "compat": "stable-diffusion-webui",
                "backend": "hipfire",
                "mode": "text-generation",
                "generated_text": result.text,
                "finish_reason": result.done.finish_reason.unwrap_or_else(|| "stop".to_string()),
                "tokens": result.done.tokens,
                "model": result.model,
                "request_id": result.req_id,
                "images": [],
                "notice": "Hipfire implements this SD API route as prompt-compatible text generation; no diffusion image backend is attached.",
            });
            Json(SdGenerationResponse {
                images: Vec::new(),
                parameters: sdapi_response_parameters(body, response_mode),
                info: info.to_string(),
            })
            .into_response()
        }
        Err(error) => (error_status(&error), Json(error)).into_response(),
    }
}

pub async fn post_extra_single_image(Json(body): Json<SdExtrasSingleImageRequest>) -> Response {
    let image = match normalize_sdapi_base64_image_to_png(&body.image) {
        Ok(image) => image,
        Err(error) => return diffusion_error_response(error),
    };
    Json(SdExtrasSingleImageResponse {
        image: if body.show_extras_results.unwrap_or(true) {
            image
        } else {
            String::new()
        },
        html_info: sdapi_extras_noop_html_info(),
    })
    .into_response()
}

pub async fn post_extra_batch_images(Json(body): Json<SdExtrasBatchImagesRequest>) -> Response {
    let mut images = Vec::with_capacity(body.image_list.len());
    for (idx, image) in body.image_list.iter().enumerate() {
        match normalize_sdapi_base64_image_to_png(&image.data) {
            Ok(image) => images.push(image),
            Err(error) => {
                return diffusion_error_response(DiffusionError::InvalidRequest(format!(
                    "extra-batch image {idx} is invalid: {error}"
                )));
            }
        }
    }
    Json(SdExtrasBatchImagesResponse {
        images: if body.show_extras_results.unwrap_or(true) {
            images
        } else {
            Vec::new()
        },
        html_info: sdapi_extras_noop_html_info(),
    })
    .into_response()
}

fn sdapi_extras_noop_html_info() -> String {
    "<p>Hipfire extras compatibility: no post-processing was applied.</p>".to_string()
}

pub async fn post_interrogate(Json(body): Json<SdInterrogateRequest>) -> Response {
    let model = body.model.trim().to_ascii_lowercase();
    if model != "clip" && model != "deepdanbooru" {
        return diffusion_error_response(DiffusionError::InvalidRequest(format!(
            "unsupported interrogate model {:?}; supported models are clip and deepdanbooru",
            body.model
        )));
    }
    if let Err(error) = normalize_sdapi_base64_image_to_png(&body.image) {
        return diffusion_error_response(error);
    }
    Json(SdInterrogateResponse {
        caption: format!(
            "Hipfire {model} interrogation compatibility response: no caption model is loaded."
        ),
    })
    .into_response()
}

fn normalize_sdapi_base64_image_to_png(image: &str) -> Result<String, DiffusionError> {
    if image.trim().is_empty() {
        return Err(DiffusionError::InvalidRequest(
            "extras image is required".to_string(),
        ));
    }
    let bytes = decode_base64_image_payload(image).map_err(|error| {
        DiffusionError::InvalidRequest(format!("extras image is not valid base64: {error}"))
    })?;
    let decoded = image::load_from_memory(&bytes)
        .map_err(|error| {
            DiffusionError::InvalidRequest(format!("extras image is invalid: {error}"))
        })?
        .to_rgb8();
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(
            decoded.as_raw(),
            decoded.width(),
            decoded.height(),
            image::ColorType::Rgb8.into(),
        )
        .map_err(|error| DiffusionError::Io(format!("failed to encode extras PNG: {error}")))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png))
}

async fn resolve_diffusion_hfq_for_request(
    state: &SharedState,
    requested_model: Option<&str>,
) -> Option<PathBuf> {
    let candidate = match requested_model.filter(|model| !model.is_empty()) {
        Some(model) => Some(model.to_string()),
        None => {
            let cfg = state.config.lock().await;
            cfg.default_model.clone()
        }
    }?;
    resolve_diffusion_hfq_candidate(&candidate)
}

fn resolve_diffusion_hfq_candidate(candidate: &str) -> Option<PathBuf> {
    if candidate.is_empty() {
        return None;
    }
    if let Some(path) = find_model(candidate) {
        if inspect_hfq(&path).is_ok() {
            return Some(path);
        }
    }
    discover_diffusion_hfq_models()
        .into_iter()
        .find(|inspection| diffusion_summary_matches_candidate(&inspection.summary, candidate))
        .map(|inspection| inspection.summary.path)
}

fn diffusion_summary_matches_candidate(
    summary: &hipfire_diffusion::DiffusionModelSummary,
    candidate: &str,
) -> bool {
    if candidate == summary.title
        || candidate == summary.model_name
        || candidate == summary.path.to_string_lossy()
    {
        return true;
    }
    summary
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| candidate == name)
}

async fn execute_hfq_diffusion_txt2img(
    state: SharedState,
    path: PathBuf,
    body: SdGenerationRequest,
) -> Response {
    let sdapi_options = state.sdapi_options.lock().await.clone();
    let body = sdapi_apply_stored_generation_defaults(body, &sdapi_options);
    let original_send_images = body.send_images.unwrap_or(true);
    let first_pass_body = match sdapi_txt2img_first_pass_body(&body) {
        Ok(body) => body,
        Err(error) => return diffusion_error_response(error),
    };
    let request = match sd_request_to_diffusion_batch_request(&first_pass_body, None, 0) {
        Ok(request) => request,
        Err(error) => return diffusion_error_response(error),
    };
    let pipeline = match cached_diffusion_pipeline(&state, path).await {
        Ok(pipeline) => pipeline,
        Err(error) => return diffusion_error_response(error),
    };
    let n_iter = sd_request_n_iter(&body);
    let save_images = body.save_images.unwrap_or(false);
    let highres_target = match sdapi_highres_target_dimensions(&first_pass_body) {
        Ok(target) => target,
        Err(error) => return diffusion_error_response(error),
    };
    let highres_pipeline = match highres_diffusion_pipeline_for_request(
        &state,
        &body,
        highres_target,
        &pipeline,
    )
    .await
    {
        Ok(pipeline) => pipeline,
        Err(error) => return diffusion_error_response(error),
    };
    let iteration_steps =
        request.steps as usize + sdapi_txt2img_highres_steps(&body, highres_target);
    let runtime_options = sd_request_generation_runtime_options(&body, &sdapi_options);
    let progress_state = state.sdapi_progress.clone();
    start_sdapi_progress(
        &progress_state,
        &body,
        "txt2img",
        iteration_steps.saturating_mul(n_iter as usize),
    );
    let worker_progress_state = progress_state.clone();
    let worker_body = body.clone();
    let worker_first_pass_body = first_pass_body.clone();
    let first_pass_pipeline = pipeline.clone();
    let output = match tokio::task::spawn_blocking(move || {
        let mut outputs = Vec::with_capacity(n_iter as usize);
        for iter in 0..n_iter {
            let iter_seed_offset = iter.saturating_mul(batch_size_for_body(&worker_body));
            let mut iter_request = sd_request_to_diffusion_batch_request(
                &worker_first_pass_body,
                None,
                iter_seed_offset,
            )?;
            if save_images || highres_target.is_some() {
                iter_request.send_images = true;
            }
            let base_step_offset = iter as usize * iteration_steps;
            let total_steps = iteration_steps * n_iter as usize;
            let mut progress = |progress: DiffusionProgress| {
                let progress = DiffusionProgress {
                    completed_steps: base_step_offset.saturating_add(progress.completed_steps),
                    total_steps,
                    timestep: progress.timestep,
                    preview_latents: progress.preview_latents,
                };
                let current_image = sdapi_preview_image_from_progress(
                    first_pass_pipeline.as_ref(),
                    runtime_options,
                    &progress,
                )?;
                update_sdapi_progress(&worker_progress_state, progress, current_image)
            };
            let first_output = first_pass_pipeline
                .generate_batch_with_progress_and_runtime_options(
                    iter_request,
                    runtime_options,
                    &mut progress,
                )?;
            let Some(target_dimensions) = highres_target else {
                outputs.push(first_output);
                continue;
            };

            let first_pass_images = sdapi_highres_second_pass_init_images(
                &worker_body,
                decode_sd_init_images(&first_output.images)?,
                target_dimensions,
            )?;
            let highres_body = sdapi_highres_second_pass_body(&worker_body, target_dimensions);
            let mut highres_batch = sd_request_to_diffusion_batch_request(
                &highres_body,
                Some(target_dimensions),
                iter_seed_offset,
            )?;
            if save_images {
                highres_batch.send_images = true;
            }
            let highres_request = DiffusionImg2ImgRequest {
                batch: highres_batch,
                init_image: first_pass_images,
                mask: None,
                inpainting_fill: None,
                resize_mode: DiffusionImg2ImgResizeMode::Image,
                denoising_strength: worker_body.denoising_strength.unwrap_or(0.75) as f32,
            };
            let highres_step_offset =
                base_step_offset.saturating_add(first_output_steps(&worker_first_pass_body));
            let mut progress = |progress: DiffusionProgress| {
                let progress = DiffusionProgress {
                    completed_steps: highres_step_offset.saturating_add(progress.completed_steps),
                    total_steps,
                    timestep: progress.timestep,
                    preview_latents: progress.preview_latents,
                };
                let current_image = sdapi_preview_image_from_progress(
                    highres_pipeline.as_ref(),
                    runtime_options,
                    &progress,
                )?;
                update_sdapi_progress(&worker_progress_state, progress, current_image)
            };
            let mut highres_output = highres_pipeline
                .generate_img2img_batch_with_progress_and_runtime_options(
                    highres_request,
                    runtime_options,
                    &mut progress,
                )?;
            annotate_highres_txt2img_info(
                &mut highres_output.info,
                &worker_body,
                (
                    worker_first_pass_body.width.unwrap_or(512),
                    worker_first_pass_body.height.unwrap_or(512),
                ),
                target_dimensions,
            );
            outputs.push(highres_output);
        }
        merge_diffusion_outputs(outputs)
    })
    .await
    {
        Ok(result) => result,
        Err(error) => Err(DiffusionError::Io(format!(
            "diffusion worker task failed: {error}"
        ))),
    };
    let current_image = output
        .as_ref()
        .ok()
        .and_then(|output| output.images.first().cloned());
    finish_sdapi_progress(&progress_state, output.as_ref().err(), current_image);
    match output {
        Ok(output) => {
            finalize_hfq_diffusion_response(body, output, "txt2img", original_send_images)
        }
        Err(error) => diffusion_error_response(error),
    }
}

async fn execute_hfq_diffusion_img2img(
    state: SharedState,
    path: PathBuf,
    body: SdGenerationRequest,
    images_base64: Vec<String>,
) -> Response {
    let sdapi_options = state.sdapi_options.lock().await.clone();
    let body = sdapi_apply_stored_generation_defaults(body, &sdapi_options);
    let original_send_images = body.send_images.unwrap_or(true);
    let init_image = match decode_sd_init_images(&images_base64) {
        Ok(image) => image,
        Err(error) => return diffusion_error_response(error),
    };
    let target_dimensions = match sdapi_img2img_target_dimensions(&body, &init_image) {
        Ok(dimensions) => dimensions,
        Err(error) => return diffusion_error_response(error),
    };
    let mask = match body
        .mask
        .as_ref()
        .map(|mask| decode_sd_init_image(mask))
        .transpose()
    {
        Ok(mask) => mask,
        Err(error) => return diffusion_error_response(error),
    };
    let prepared = match sdapi_prepare_img2img_inputs(&body, init_image, mask, target_dimensions) {
        Ok(prepared) => prepared,
        Err(error) => return diffusion_error_response(error),
    };
    let default_dimensions = Some(prepared.processing_dimensions);
    let _first_batch = match sd_request_to_diffusion_batch_request(&body, default_dimensions, 0) {
        Ok(request) => request,
        Err(error) => return diffusion_error_response(error),
    };
    let pipeline = match cached_diffusion_pipeline(&state, path).await {
        Ok(pipeline) => pipeline,
        Err(error) => return diffusion_error_response(error),
    };
    let n_iter = sd_request_n_iter(&body);
    let save_images = body.save_images.unwrap_or(false);
    let denoising_strength = body.denoising_strength.unwrap_or(0.75) as f32;
    let inpainting_fill = match sdapi_inpainting_fill(&body) {
        Ok(inpainting_fill) => inpainting_fill,
        Err(error) => return diffusion_error_response(error),
    };
    let runtime_options = sd_request_generation_runtime_options(&body, &sdapi_options);
    let progress_state = state.sdapi_progress.clone();
    start_sdapi_progress(
        &progress_state,
        &body,
        "img2img",
        sdapi_img2img_denoise_steps(&body).saturating_mul(n_iter as usize),
    );
    let worker_progress_state = progress_state.clone();
    let worker_body = body.clone();
    let worker_init_image = prepared.init_image.clone();
    let worker_mask = prepared.mask.clone();
    let output = match tokio::task::spawn_blocking(move || {
        let mut outputs = Vec::with_capacity(n_iter as usize);
        for iter in 0..n_iter {
            let mut iter_batch = sd_request_to_diffusion_batch_request(
                &worker_body,
                default_dimensions,
                iter.saturating_mul(batch_size_for_body(&worker_body)),
            )?;
            if save_images {
                iter_batch.send_images = true;
            }
            let request = DiffusionImg2ImgRequest {
                batch: iter_batch,
                init_image: worker_init_image.clone(),
                mask: worker_mask.clone(),
                inpainting_fill,
                resize_mode: sdapi_img2img_diffusion_resize_mode(&worker_body),
                denoising_strength,
            };
            let step_offset = iter as usize * sdapi_img2img_denoise_steps(&worker_body);
            let total_steps = sdapi_img2img_denoise_steps(&worker_body) * n_iter as usize;
            let mut progress = |progress: DiffusionProgress| {
                let progress = DiffusionProgress {
                    completed_steps: step_offset.saturating_add(progress.completed_steps),
                    total_steps,
                    timestep: progress.timestep,
                    preview_latents: progress.preview_latents,
                };
                let current_image = sdapi_preview_image_from_progress(
                    pipeline.as_ref(),
                    runtime_options,
                    &progress,
                )?;
                update_sdapi_progress(&worker_progress_state, progress, current_image)
            };
            outputs.push(
                pipeline.generate_img2img_batch_with_progress_and_runtime_options(
                    request,
                    runtime_options,
                    &mut progress,
                )?,
            );
        }
        merge_diffusion_outputs(outputs)
    })
    .await
    {
        Ok(result) => result,
        Err(error) => Err(DiffusionError::Io(format!(
            "diffusion worker task failed: {error}"
        ))),
    };
    let current_image = output
        .as_ref()
        .ok()
        .and_then(|output| output.images.first().cloned());
    let output = output.and_then(|mut output| {
        sdapi_annotate_img2img_inpainting_info(
            &mut output,
            prepared.image_fill_applied,
            inpainting_fill,
        );
        if let Some(plan) = prepared.full_res_plan.as_ref() {
            apply_sdapi_inpaint_full_res_output(&mut output, plan)?;
        }
        Ok(output)
    });
    let current_image = output
        .as_ref()
        .ok()
        .and_then(|output| output.images.first().cloned())
        .or(current_image);
    finish_sdapi_progress(&progress_state, output.as_ref().err(), current_image);
    match output {
        Ok(output) => {
            finalize_hfq_diffusion_response(body, output, "img2img", original_send_images)
        }
        Err(error) => diffusion_error_response(error),
    }
}

fn finalize_hfq_diffusion_response(
    body: SdGenerationRequest,
    mut output: hipfire_diffusion::DiffusionBatchOutput,
    mode: &str,
    original_send_images: bool,
) -> Response {
    let infotext = sdapi_parameters_text(&body, mode, &output.info);
    if !output.images.is_empty() {
        match annotate_sdapi_images(&output.images, &infotext) {
            Ok(images) => output.images = images,
            Err(error) => return diffusion_error_response(error),
        }
    }
    if let Value::Object(map) = &mut output.info {
        map.insert(
            "infotexts".to_string(),
            json!(vec![infotext.clone(); output.images.len()]),
        );
    }
    if body.save_images.unwrap_or(false) {
        match save_sdapi_images(&body, mode, &output.images) {
            Ok(paths) => {
                if let Value::Object(map) = &mut output.info {
                    map.insert("saved_images".to_string(), json!(paths));
                    map.insert("save_images".to_string(), json!(true));
                }
            }
            Err(error) => return diffusion_error_response(error),
        }
    }
    let images = if original_send_images {
        output.images
    } else {
        Vec::new()
    };
    let parameters = sdapi_response_parameters(body, mode);
    Json(SdGenerationResponse {
        images,
        parameters,
        info: output.info.to_string(),
    })
    .into_response()
}

fn sdapi_response_parameters(mut body: SdGenerationRequest, mode: &str) -> SdGenerationRequest {
    if mode == "img2img" && !body.include_init_images.unwrap_or(false) {
        body.init_images = None;
    }
    body
}

fn sdapi_validate_supported_scripts(body: &SdGenerationRequest) -> Result<(), DiffusionError> {
    if body
        .script_name
        .as_deref()
        .is_some_and(|name| !name.trim().is_empty())
    {
        return Err(DiffusionError::InvalidRequest(
            "script_name is not supported because Hipfire does not expose SDAPI selectable scripts"
                .to_string(),
        ));
    }
    if body
        .script_args
        .as_ref()
        .is_some_and(|value| !sdapi_script_value_is_empty(value))
    {
        return Err(DiffusionError::InvalidRequest(
            "script_args is not supported because Hipfire does not expose SDAPI selectable scripts"
                .to_string(),
        ));
    }
    if body
        .alwayson_scripts
        .as_ref()
        .is_some_and(|value| !sdapi_alwayson_scripts_are_noop(value))
    {
        return Err(DiffusionError::InvalidRequest(
            "alwayson_scripts contains active or unsupported script payloads; Hipfire accepts only empty or disabled always-on script defaults"
                .to_string(),
        ));
    }
    Ok(())
}

fn sdapi_script_value_is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Array(values) => values.is_empty(),
        Value::Object(values) => values.is_empty(),
        Value::String(value) => value.trim().is_empty(),
        _ => false,
    }
}

fn sdapi_alwayson_scripts_are_noop(value: &Value) -> bool {
    if sdapi_script_value_is_empty(value) {
        return true;
    }
    let Value::Object(scripts) = value else {
        return false;
    };
    scripts.values().all(sdapi_alwayson_script_payload_is_noop)
}

fn sdapi_alwayson_script_payload_is_noop(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Array(values) => sdapi_alwayson_script_args_are_noop(values),
        Value::Object(payload) => {
            if payload.is_empty() {
                return true;
            }
            let args_are_noop = payload
                .get("args")
                .is_none_or(sdapi_alwayson_script_args_value_is_noop);
            args_are_noop
                && payload.iter().all(|(key, value)| {
                    key == "args" || sdapi_alwayson_script_control_value_is_noop(key, value)
                })
        }
        _ => sdapi_script_value_is_empty(value),
    }
}

fn sdapi_alwayson_script_args_value_is_noop(value: &Value) -> bool {
    match value {
        Value::Array(values) => sdapi_alwayson_script_args_are_noop(values),
        Value::Object(_) => sdapi_alwayson_script_arg_is_disabled(value),
        _ => sdapi_script_value_is_empty(value),
    }
}

fn sdapi_alwayson_script_args_are_noop(values: &[Value]) -> bool {
    if values.is_empty() {
        return true;
    }
    if values
        .first()
        .is_some_and(sdapi_value_is_explicit_disable_flag)
    {
        return true;
    }
    values.iter().all(sdapi_alwayson_script_arg_is_disabled)
}

fn sdapi_alwayson_script_arg_is_disabled(value: &Value) -> bool {
    if sdapi_script_value_is_empty(value) || sdapi_value_is_explicit_disable_flag(value) {
        return true;
    }
    let Value::Object(arg) = value else {
        return false;
    };
    if let Some((_, flag)) = arg
        .iter()
        .find(|(key, _)| sdapi_alwayson_script_enable_key(key))
    {
        return !sdapi_value_is_truthy(flag);
    }
    arg.values().all(sdapi_script_value_is_empty)
}

fn sdapi_alwayson_script_control_value_is_noop(key: &str, value: &Value) -> bool {
    if sdapi_alwayson_script_enable_key(key) {
        return !sdapi_value_is_truthy(value);
    }
    sdapi_script_value_is_empty(value)
}

fn sdapi_alwayson_script_enable_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "enabled"
            | "enable"
            | "is_enabled"
            | "active"
            | "ad_enabled"
            | "ad_enable"
            | "controlnet_enabled"
    )
}

fn sdapi_value_is_explicit_disable_flag(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::Number(_) | Value::String(_))
        && !sdapi_value_is_truthy(value)
}

fn sdapi_apply_stored_generation_defaults(
    mut body: SdGenerationRequest,
    stored_options: &std::collections::HashMap<String, Value>,
) -> SdGenerationRequest {
    if body.send_images.is_none() {
        body.send_images = sd_stored_bool(stored_options, "send_images");
    }
    if body.save_images.is_none() {
        body.save_images = sd_stored_bool(stored_options, "save_images");
    }
    for key in [
        "outdir_samples",
        "outdir_txt2img_samples",
        "outdir_img2img_samples",
    ] {
        sdapi_apply_stored_override_default(&mut body, stored_options, key);
    }
    body
}

fn sdapi_apply_stored_override_default(
    body: &mut SdGenerationRequest,
    stored_options: &std::collections::HashMap<String, Value>,
    key: &str,
) {
    let Some(value) = stored_options.get(key).cloned() else {
        return;
    };
    if body.override_settings.is_none() {
        body.override_settings = Some(json!({}));
    }
    let Some(settings) = body
        .override_settings
        .as_mut()
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    settings.entry(key.to_string()).or_insert(value);
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SdapiParsedInfotext {
    prompt: String,
    negative_prompt: String,
    params: Vec<(String, String)>,
}

fn sdapi_apply_infotext_defaults(mut body: SdGenerationRequest) -> SdGenerationRequest {
    let Some(parsed) = body.infotext.as_deref().and_then(sdapi_parse_infotext) else {
        return body;
    };
    if body.prompt.is_empty() && !parsed.prompt.is_empty() {
        body.prompt = parsed.prompt.clone();
    }
    if body.negative_prompt.is_empty() && !parsed.negative_prompt.is_empty() {
        body.negative_prompt = parsed.negative_prompt.clone();
    }
    if body.steps.is_none() {
        body.steps = sdapi_infotext_u32(&parsed, "Steps");
    }
    if body.cfg_scale.is_none() {
        body.cfg_scale = sdapi_infotext_f64(&parsed, "CFG scale");
    }
    if body.seed.is_none() {
        body.seed = sdapi_infotext_i64(&parsed, "Seed");
    }
    if body.width.is_none() || body.height.is_none() {
        if let Some((width, height)) = sdapi_infotext_dimensions(&parsed, "Size") {
            body.width.get_or_insert(width);
            body.height.get_or_insert(height);
        }
    }
    if body.sampler_name.is_none() && body.sampler_index.is_none() {
        body.sampler_name = sdapi_infotext_string(&parsed, "Sampler");
    }
    if body.scheduler.is_none() {
        if let Some(scheduler) = sdapi_infotext_string(&parsed, "Schedule type")
            .filter(|scheduler| scheduler != "Automatic")
        {
            body.scheduler = Some(scheduler);
        }
    }
    if body.denoising_strength.is_none() {
        body.denoising_strength = sdapi_infotext_f64(&parsed, "Denoising strength");
    }
    let has_highres = parsed
        .params
        .iter()
        .any(|(key, _)| key.starts_with("Hires "));
    if has_highres && body.enable_hr.is_none() {
        body.enable_hr = Some(true);
    }
    if body.hr_scale.is_none() {
        body.hr_scale = sdapi_infotext_f64(&parsed, "Hires upscale");
    }
    if body.hr_resize_x.is_none() || body.hr_resize_y.is_none() {
        if let Some((width, height)) = sdapi_infotext_dimensions(&parsed, "Hires resize") {
            body.hr_resize_x.get_or_insert(width);
            body.hr_resize_y.get_or_insert(height);
        }
    }
    if body.hr_second_pass_steps.is_none() {
        body.hr_second_pass_steps = sdapi_infotext_u32(&parsed, "Hires steps");
    }
    if body.hr_upscaler.is_none() {
        body.hr_upscaler = sdapi_infotext_string(&parsed, "Hires upscaler");
    }
    if body.hr_checkpoint_name.is_none() {
        body.hr_checkpoint_name = sdapi_infotext_string(&parsed, "Hires checkpoint");
    }
    if body.hr_prompt.is_none() {
        body.hr_prompt = sdapi_infotext_string(&parsed, "Hires prompt");
    }
    if body.hr_negative_prompt.is_none() {
        body.hr_negative_prompt = sdapi_infotext_string(&parsed, "Hires negative prompt");
    }
    if body.hr_sampler_name.is_none() {
        body.hr_sampler_name = sdapi_infotext_string(&parsed, "Hires sampler");
    }
    if body.hr_scheduler.is_none() {
        body.hr_scheduler = sdapi_infotext_string(&parsed, "Hires schedule type");
    }
    if body.inpainting_fill.is_none() {
        body.inpainting_fill = sdapi_infotext_string(&parsed, "Masked content").and_then(|value| {
            let mode = match value.to_ascii_lowercase().as_str() {
                "fill" => 0,
                "original" => 1,
                "latent noise" => 2,
                "latent nothing" => 3,
                _ => return None,
            };
            Some(json!(mode))
        });
    }
    if body.inpainting_mask_invert.is_none() {
        body.inpainting_mask_invert =
            sdapi_infotext_string(&parsed, "Mask mode").and_then(|value| {
                match value.to_ascii_lowercase().as_str() {
                    "inpaint masked" => Some(json!(false)),
                    "inpaint not masked" => Some(json!(true)),
                    _ => None,
                }
            });
    }
    if body.inpaint_full_res.is_none() {
        body.inpaint_full_res = sdapi_infotext_string(&parsed, "Inpaint area").and_then(|value| {
            match value.to_ascii_lowercase().as_str() {
                "whole picture" => Some(json!(false)),
                "only masked" => Some(json!(true)),
                _ => None,
            }
        });
    }
    if body.inpaint_full_res_padding.is_none() {
        body.inpaint_full_res_padding =
            sdapi_infotext_u32(&parsed, "Masked area padding").map(|value| json!(value));
    }
    body
}

fn sdapi_parse_infotext(infotext: &str) -> Option<SdapiParsedInfotext> {
    let mut lines = infotext.trim().lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return None;
    }
    let mut params = Vec::new();
    let last_line = lines.last().map(|line| line.trim()).unwrap_or_default();
    if sdapi_parse_infotext_params(last_line).len() >= 3 {
        params = sdapi_parse_infotext_params(last_line);
        lines.pop();
    }

    let mut prompt = String::new();
    let mut negative_prompt = String::new();
    let mut in_negative_prompt = false;
    for line in lines {
        let mut line = line.trim();
        if let Some(rest) = line.strip_prefix("Negative prompt:") {
            in_negative_prompt = true;
            line = rest.trim();
        }
        let target = if in_negative_prompt {
            &mut negative_prompt
        } else {
            &mut prompt
        };
        if !target.is_empty() && !line.is_empty() {
            target.push('\n');
        }
        target.push_str(line);
    }
    Some(SdapiParsedInfotext {
        prompt,
        negative_prompt,
        params,
    })
}

fn sdapi_parse_infotext_params(line: &str) -> Vec<(String, String)> {
    let mut params = Vec::new();
    for segment in sdapi_split_infotext_param_segments(line) {
        let Some((key, value)) = segment.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if !key.is_empty() && !value.is_empty() {
            params.push((key.to_string(), sdapi_unquote_infotext_value(value)));
        }
    }
    params
}

fn sdapi_split_infotext_param_segments(line: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut escaped = false;
    for ch in line.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_quotes => {
                current.push(ch);
                escaped = true;
            }
            '"' => {
                current.push(ch);
                in_quotes = !in_quotes;
            }
            ',' if !in_quotes => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    segments.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }
    segments
}

fn sdapi_unquote_infotext_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() < 2 || !trimmed.starts_with('"') || !trimmed.ends_with('"') {
        return trimmed.to_string();
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut out = String::new();
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped {
        out.push('\\');
    }
    out
}

fn sdapi_infotext_value<'a>(parsed: &'a SdapiParsedInfotext, key: &str) -> Option<&'a str> {
    parsed
        .params
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value.trim())
        .filter(|value| !value.is_empty())
}

fn sdapi_infotext_string(parsed: &SdapiParsedInfotext, key: &str) -> Option<String> {
    sdapi_infotext_value(parsed, key).map(str::to_string)
}

fn sdapi_infotext_u32(parsed: &SdapiParsedInfotext, key: &str) -> Option<u32> {
    sdapi_infotext_value(parsed, key)?.parse::<u32>().ok()
}

fn sdapi_infotext_i64(parsed: &SdapiParsedInfotext, key: &str) -> Option<i64> {
    sdapi_infotext_value(parsed, key)?.parse::<i64>().ok()
}

fn sdapi_infotext_f64(parsed: &SdapiParsedInfotext, key: &str) -> Option<f64> {
    let value = sdapi_infotext_value(parsed, key)?.parse::<f64>().ok()?;
    value.is_finite().then_some(value)
}

fn sdapi_infotext_dimensions(parsed: &SdapiParsedInfotext, key: &str) -> Option<(u32, u32)> {
    let value = sdapi_infotext_value(parsed, key)?;
    sdapi_infotext_split_dimensions(value)
}

fn sdapi_infotext_split_dimensions(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value.split_once('x').or_else(|| value.split_once('X'))?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
}

fn sdapi_parameters_text(body: &SdGenerationRequest, mode: &str, info: &Value) -> String {
    let mut lines = Vec::new();
    let prompt = if body.prompt.is_empty() {
        body.infotext.clone().unwrap_or_default()
    } else {
        body.prompt.clone()
    };
    lines.push(prompt);
    if !body.negative_prompt.is_empty() {
        lines.push(format!("Negative prompt: {}", body.negative_prompt));
    }
    let seeds = info
        .get("seeds")
        .and_then(Value::as_array)
        .and_then(|seeds| seeds.first())
        .and_then(Value::as_i64)
        .or(body.seed);
    lines.push(format!(
        "Steps: {}, Sampler: {}, CFG scale: {}, Seed: {}, Size: {}x{}, Model: {}, Mode: {}",
        body.steps.unwrap_or(20),
        sdapi_effective_scheduler(body),
        body.cfg_scale.unwrap_or(7.0),
        seeds.unwrap_or(-1),
        info.get("width")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .or(body.width)
            .unwrap_or(512),
        info.get("height")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .or(body.height)
            .unwrap_or(512),
        body.model.as_deref().unwrap_or(""),
        mode,
    ));
    lines.join("\n")
}

fn annotate_sdapi_images(images: &[String], infotext: &str) -> Result<Vec<String>, DiffusionError> {
    images
        .iter()
        .enumerate()
        .map(|(idx, image)| {
            let bytes = decode_base64_image_payload(image).map_err(|error| {
                DiffusionError::Io(format!(
                    "generated image {idx} is not valid base64: {error}"
                ))
            })?;
            let annotated = insert_png_text_chunk(&bytes, "parameters", infotext)?;
            Ok(base64::engine::general_purpose::STANDARD.encode(annotated))
        })
        .collect()
}

fn save_sdapi_images(
    body: &SdGenerationRequest,
    mode: &str,
    images: &[String],
) -> Result<Vec<String>, DiffusionError> {
    let output_dir = sdapi_output_dir(body, mode);
    fs::create_dir_all(&output_dir).map_err(|error| {
        DiffusionError::Io(format!(
            "failed to create SDAPI output directory {}: {error}",
            output_dir.display()
        ))
    })?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| DiffusionError::Io(format!("system clock before unix epoch: {error}")))?
        .as_millis();
    let mut paths = Vec::with_capacity(images.len());
    for (idx, image) in images.iter().enumerate() {
        let bytes = decode_base64_image_payload(image).map_err(|error| {
            DiffusionError::Io(format!(
                "generated image {idx} is not valid base64: {error}"
            ))
        })?;
        if bytes.get(..8) != Some(b"\x89PNG\r\n\x1a\n") {
            return Err(DiffusionError::Io(format!(
                "generated image {idx} is not a PNG"
            )));
        }
        let path = output_dir.join(format!(
            "hipfire-{mode}-{timestamp}-{}-{idx}.png",
            std::process::id()
        ));
        let mut file = fs::File::create(&path).map_err(|error| {
            DiffusionError::Io(format!("failed to create {}: {error}", path.display()))
        })?;
        file.write_all(&bytes).map_err(|error| {
            DiffusionError::Io(format!("failed to write {}: {error}", path.display()))
        })?;
        paths.push(path.to_string_lossy().into_owned());
    }
    Ok(paths)
}

fn decode_base64_image_payload(image: &str) -> Result<Vec<u8>, base64::DecodeError> {
    let payload = image
        .split_once(',')
        .map(|(_, payload)| payload)
        .unwrap_or(image);
    base64::engine::general_purpose::STANDARD.decode(payload)
}

fn insert_png_text_chunk(png: &[u8], keyword: &str, text: &str) -> Result<Vec<u8>, DiffusionError> {
    let iend_offset = find_png_iend_offset(png)?;
    let mut chunk_data = Vec::with_capacity(keyword.len() + 1 + text.len());
    chunk_data.extend_from_slice(keyword.as_bytes());
    chunk_data.push(0);
    chunk_data.extend_from_slice(text.as_bytes());
    let mut chunk = Vec::with_capacity(12 + chunk_data.len());
    chunk.extend_from_slice(&(chunk_data.len() as u32).to_be_bytes());
    chunk.extend_from_slice(b"tEXt");
    chunk.extend_from_slice(&chunk_data);
    let mut crc_input = Vec::with_capacity(4 + chunk_data.len());
    crc_input.extend_from_slice(b"tEXt");
    crc_input.extend_from_slice(&chunk_data);
    chunk.extend_from_slice(&png_crc32(&crc_input).to_be_bytes());

    let mut out = Vec::with_capacity(png.len() + chunk.len());
    out.extend_from_slice(&png[..iend_offset]);
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&png[iend_offset..]);
    Ok(out)
}

fn extract_png_text_chunk(png: &[u8], keyword: &str) -> Result<Option<String>, DiffusionError> {
    if png.get(..8) != Some(b"\x89PNG\r\n\x1a\n") {
        return Err(DiffusionError::InvalidRequest(
            "image is not a PNG".to_string(),
        ));
    }
    let mut offset = 8usize;
    while offset + 12 <= png.len() {
        let len = u32::from_be_bytes(
            png[offset..offset + 4]
                .try_into()
                .expect("slice length checked"),
        ) as usize;
        let kind_start = offset + 4;
        let data_start = offset + 8;
        let data_end = data_start.checked_add(len).ok_or_else(|| {
            DiffusionError::InvalidRequest("PNG chunk length overflow".to_string())
        })?;
        let next = data_end.checked_add(4).ok_or_else(|| {
            DiffusionError::InvalidRequest("PNG chunk CRC offset overflow".to_string())
        })?;
        if next > png.len() {
            return Err(DiffusionError::InvalidRequest(
                "PNG chunk extends past end of image".to_string(),
            ));
        }
        let kind = &png[kind_start..kind_start + 4];
        if kind == b"tEXt" {
            let data = &png[data_start..data_end];
            if let Some(nul) = data.iter().position(|byte| *byte == 0) {
                if &data[..nul] == keyword.as_bytes() {
                    return Ok(Some(String::from_utf8_lossy(&data[nul + 1..]).into_owned()));
                }
            }
        }
        if kind == b"IEND" {
            return Ok(None);
        }
        offset = next;
    }
    Err(DiffusionError::InvalidRequest(
        "PNG is missing IEND chunk".to_string(),
    ))
}

fn find_png_iend_offset(png: &[u8]) -> Result<usize, DiffusionError> {
    if png.get(..8) != Some(b"\x89PNG\r\n\x1a\n") {
        return Err(DiffusionError::Io(
            "generated image is not a PNG".to_string(),
        ));
    }
    let mut offset = 8usize;
    while offset + 12 <= png.len() {
        let len = u32::from_be_bytes(
            png[offset..offset + 4]
                .try_into()
                .expect("slice length checked"),
        ) as usize;
        let kind_start = offset + 4;
        let data_start = offset + 8;
        let data_end = data_start
            .checked_add(len)
            .ok_or_else(|| DiffusionError::Io("PNG chunk length overflow".to_string()))?;
        let next = data_end
            .checked_add(4)
            .ok_or_else(|| DiffusionError::Io("PNG chunk CRC offset overflow".to_string()))?;
        if next > png.len() {
            return Err(DiffusionError::Io(
                "PNG chunk extends past end of image".to_string(),
            ));
        }
        if &png[kind_start..kind_start + 4] == b"IEND" {
            return Ok(offset);
        }
        offset = next;
    }
    Err(DiffusionError::Io(
        "generated PNG is missing IEND chunk".to_string(),
    ))
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn sdapi_output_dir(body: &SdGenerationRequest, mode: &str) -> PathBuf {
    let mode_key = match mode {
        "img2img" => "outdir_img2img_samples",
        _ => "outdir_txt2img_samples",
    };
    body.override_settings
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|settings| {
            settings
                .get(mode_key)
                .or_else(|| settings.get("outdir_samples"))
                .and_then(Value::as_str)
        })
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/hipfire-sdapi").join(mode))
}

async fn cached_diffusion_pipeline(
    state: &SharedState,
    path: PathBuf,
) -> Result<Arc<DiffusionPipeline>, DiffusionError> {
    if let Some(pipeline) = state.diffusion_pipelines.lock().await.get(&path).cloned() {
        return Ok(pipeline);
    }

    let load_path = path.clone();
    let pipeline =
        match tokio::task::spawn_blocking(move || DiffusionPipeline::open_hfq(load_path)).await {
            Ok(result) => Arc::new(result?),
            Err(error) => {
                return Err(DiffusionError::Io(format!(
                    "diffusion loader task failed: {error}"
                )));
            }
        };

    let mut cache = state.diffusion_pipelines.lock().await;
    Ok(cache.entry(path).or_insert_with(|| pipeline).clone())
}

fn sd_request_to_diffusion_batch_request(
    body: &SdGenerationRequest,
    default_dimensions: Option<(u32, u32)>,
    seed_offset: u32,
) -> Result<DiffusionBatchRequest, DiffusionError> {
    let batch_size = body.batch_size.unwrap_or(1).max(1);
    let base_seed = body.seed.unwrap_or(-1);
    let prompt = if body.prompt.is_empty() {
        body.infotext.clone().unwrap_or_default()
    } else {
        body.prompt.clone()
    };
    let prompts = (0..batch_size)
        .map(|idx| DiffusionPrompt {
            prompt: prompt.clone(),
            negative_prompt: body.negative_prompt.clone(),
            seed: if base_seed < 0 {
                base_seed
            } else {
                base_seed.saturating_add(seed_offset.saturating_add(idx) as i64)
            },
            subseed: body.subseed,
        })
        .collect();
    let width = body
        .width
        .or_else(|| default_dimensions.map(|dimensions| dimensions.0))
        .unwrap_or(512);
    let height = body
        .height
        .or_else(|| default_dimensions.map(|dimensions| dimensions.1))
        .unwrap_or(512);
    Ok(DiffusionBatchRequest {
        prompts,
        width,
        height,
        original_width: body.original_width,
        original_height: body.original_height,
        target_width: body.target_width,
        target_height: body.target_height,
        crop_x: body.crop_x.unwrap_or(0),
        crop_y: body.crop_y.unwrap_or(0),
        steps: body.steps.unwrap_or(20),
        cfg_scale: body.cfg_scale.unwrap_or(7.0) as f32,
        scheduler: sdapi_effective_scheduler(body),
        subseed_strength: body.subseed_strength.unwrap_or(0.0) as f32,
        send_images: body.send_images.unwrap_or(true),
        save_images: body.save_images.unwrap_or(false),
    })
}

fn sdapi_effective_scheduler(body: &SdGenerationRequest) -> String {
    let sampler = body
        .sampler_name
        .as_deref()
        .or(body.sampler_index.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let scheduler = body
        .scheduler
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(scheduler) = scheduler else {
        return sampler.unwrap_or("DPM++ 2M").to_string();
    };
    if sdapi_scheduler_is_automatic(scheduler) {
        return sampler.unwrap_or("DPM++ 2M").to_string();
    }
    if sdapi_scheduler_is_karras(scheduler) {
        let sampler = sampler.unwrap_or("DPM++ 2M");
        if normalize_scheduler_name_for_sdapi(sampler).contains("karras") {
            sampler.to_string()
        } else {
            format!("{sampler} Karras")
        }
    } else {
        scheduler.to_string()
    }
}

fn sdapi_scheduler_is_automatic(value: &str) -> bool {
    matches!(
        normalize_scheduler_name_for_sdapi(value).as_str(),
        "automatic" | "auto" | "use same scheduler"
    )
}

fn sdapi_scheduler_is_karras(value: &str) -> bool {
    normalize_scheduler_name_for_sdapi(value) == "karras"
}

fn sdapi_scheduler_is_schedule_modifier(value: &str) -> bool {
    sdapi_scheduler_is_automatic(value) || sdapi_scheduler_is_karras(value)
}

fn normalize_scheduler_name_for_sdapi(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn sdapi_img2img_target_dimensions(
    body: &SdGenerationRequest,
    init_image: &RgbImageBatch,
) -> Result<(u32, u32), DiffusionError> {
    let init_width = u32::try_from(init_image.width).map_err(|_| {
        DiffusionError::InvalidRequest("init image width is out of range".to_string())
    })?;
    let init_height = u32::try_from(init_image.height).map_err(|_| {
        DiffusionError::InvalidRequest("init image height is out of range".to_string())
    })?;
    Ok((
        body.width.unwrap_or(init_width),
        body.height.unwrap_or(init_height),
    ))
}

fn sdapi_img2img_resize_image(
    body: &SdGenerationRequest,
    image: RgbImageBatch,
    target_dimensions: (u32, u32),
) -> Result<RgbImageBatch, DiffusionError> {
    match body.resize_mode.unwrap_or(0) {
        0 => resize_rgb_batch_nearest(&image, target_dimensions.0, target_dimensions.1),
        1 => resize_rgb_batch_to_cover_nearest(&image, target_dimensions.0, target_dimensions.1),
        2 => resize_rgb_batch_to_contain_fill_nearest(
            &image,
            target_dimensions.0,
            target_dimensions.1,
        ),
        3 => Ok(image),
        mode => Err(DiffusionError::InvalidRequest(format!(
            "unsupported img2img resize_mode {mode}; supported values are 0, 1, 2, and 3"
        ))),
    }
}

fn sdapi_img2img_diffusion_resize_mode(body: &SdGenerationRequest) -> DiffusionImg2ImgResizeMode {
    if body.resize_mode == Some(3) {
        DiffusionImg2ImgResizeMode::Latent
    } else {
        DiffusionImg2ImgResizeMode::Image
    }
}

fn sdapi_prepare_img2img_inputs(
    body: &SdGenerationRequest,
    init_image: RgbImageBatch,
    mask: Option<RgbImageBatch>,
    target_dimensions: (u32, u32),
) -> Result<SdapiPreparedImg2Img, DiffusionError> {
    let Some(mask) = mask else {
        return Ok(SdapiPreparedImg2Img {
            init_image: sdapi_img2img_resize_image(body, init_image, target_dimensions)?,
            mask: None,
            processing_dimensions: target_dimensions,
            full_res_plan: None,
            image_fill_applied: false,
        });
    };
    let inpainting_fill = sdapi_inpainting_fill(body)?;
    let init_dimensions = (
        u32::try_from(init_image.width).map_err(|_| {
            DiffusionError::InvalidRequest("init image width is out of range".to_string())
        })?,
        u32::try_from(init_image.height).map_err(|_| {
            DiffusionError::InvalidRequest("init image height is out of range".to_string())
        })?,
    );
    let mask = sdapi_apply_inpainting_mask_options(body, mask)?;
    let mask = if mask.width != init_image.width || mask.height != init_image.height {
        resize_rgb_batch_nearest(&mask, init_dimensions.0, init_dimensions.1)?
    } else {
        mask
    };
    if sdapi_inpaint_full_res(body) {
        let padding = sdapi_inpaint_full_res_padding(body)?;
        if let Some(crop_region) = sdapi_mask_crop_region(&mask, padding)? {
            let crop_region = sdapi_expand_crop_region(
                crop_region,
                target_dimensions,
                (init_image.width, init_image.height),
            )?;
            let cropped_init = sdapi_crop_rgb_batch(&init_image, crop_region)?;
            let cropped_mask = sdapi_crop_rgb_batch(&mask, crop_region)?;
            let prepared_mask = resize_rgb_batch_to_contain_fill_nearest(
                &cropped_mask,
                target_dimensions.0,
                target_dimensions.1,
            )?;
            let (prepared_init, image_fill_applied) = sdapi_apply_inpainting_fill_to_init_image(
                resize_rgb_batch_to_contain_fill_nearest(
                    &cropped_init,
                    target_dimensions.0,
                    target_dimensions.1,
                )?,
                &prepared_mask,
                inpainting_fill,
            )?;
            return Ok(SdapiPreparedImg2Img {
                init_image: prepared_init,
                mask: Some(prepared_mask),
                processing_dimensions: target_dimensions,
                full_res_plan: Some(SdapiInpaintFullResPlan {
                    base_image: init_image,
                    overlay_mask: mask,
                    paste_region: crop_region,
                    padding,
                }),
                image_fill_applied,
            });
        }
        return Ok(SdapiPreparedImg2Img {
            init_image: sdapi_img2img_resize_image(body, init_image, target_dimensions)?,
            mask: None,
            processing_dimensions: target_dimensions,
            full_res_plan: None,
            image_fill_applied: false,
        });
    }
    let prepared_mask = sdapi_img2img_resize_image(body, mask, target_dimensions)?;
    let (prepared_init, image_fill_applied) = sdapi_apply_inpainting_fill_to_init_image(
        sdapi_img2img_resize_image(body, init_image, target_dimensions)?,
        &prepared_mask,
        inpainting_fill,
    )?;
    Ok(SdapiPreparedImg2Img {
        init_image: prepared_init,
        mask: Some(prepared_mask),
        processing_dimensions: target_dimensions,
        full_res_plan: None,
        image_fill_applied,
    })
}

fn apply_sdapi_inpaint_full_res_output(
    output: &mut DiffusionBatchOutput,
    plan: &SdapiInpaintFullResPlan,
) -> Result<(), DiffusionError> {
    if output.images.is_empty() {
        if let Value::Object(map) = &mut output.info {
            map.insert("inpaint_full_res".to_string(), json!(true));
            map.insert("inpaint_full_res_padding".to_string(), json!(plan.padding));
            map.insert(
                "inpaint_full_res_crop".to_string(),
                json!([
                    plan.paste_region.x,
                    plan.paste_region.y,
                    plan.paste_region.width,
                    plan.paste_region.height
                ]),
            );
        }
        return Ok(());
    }
    let generated = decode_sd_init_images(&output.images)?;
    let composited = sdapi_composite_inpaint_full_res(&generated, plan)?;
    output.images = encode_rgb_batch_png_base64(&composited)?;
    if let Value::Object(map) = &mut output.info {
        map.insert("width".to_string(), json!(plan.base_image.width));
        map.insert("height".to_string(), json!(plan.base_image.height));
        map.insert("inpaint_full_res".to_string(), json!(true));
        map.insert("inpaint_full_res_padding".to_string(), json!(plan.padding));
        map.insert(
            "inpaint_full_res_crop".to_string(),
            json!([
                plan.paste_region.x,
                plan.paste_region.y,
                plan.paste_region.width,
                plan.paste_region.height
            ]),
        );
    }
    Ok(())
}

fn sdapi_composite_inpaint_full_res(
    generated: &RgbImageBatch,
    plan: &SdapiInpaintFullResPlan,
) -> Result<RgbImageBatch, DiffusionError> {
    sdapi_validate_rgb_batch_len(generated, "generated image")?;
    sdapi_validate_rgb_batch_len(&plan.base_image, "inpaint base image")?;
    sdapi_validate_rgb_batch_len(&plan.overlay_mask, "inpaint overlay mask")?;
    if plan.overlay_mask.batch != 1 && plan.overlay_mask.batch != plan.base_image.batch {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint overlay mask batch {} must be 1 or match base image batch {}",
            plan.overlay_mask.batch, plan.base_image.batch
        )));
    }
    let paste_width = u32::try_from(plan.paste_region.width).map_err(|_| {
        DiffusionError::InvalidRequest("inpaint crop width is out of range".to_string())
    })?;
    let paste_height = u32::try_from(plan.paste_region.height).map_err(|_| {
        DiffusionError::InvalidRequest("inpaint crop height is out of range".to_string())
    })?;
    let resized_generated =
        resize_rgb_batch_to_cover_nearest(generated, paste_width, paste_height)?;
    let base_image_bytes = plan
        .base_image
        .width
        .checked_mul(plan.base_image.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("inpaint base image dimensions overflow".to_string())
        })?;
    let crop_image_bytes = plan
        .paste_region
        .width
        .checked_mul(plan.paste_region.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("inpaint crop dimensions overflow".to_string())
        })?;
    let mut data = vec![0u8; generated.batch * base_image_bytes];
    for image_idx in 0..generated.batch {
        let base_idx = if plan.base_image.batch == 1 {
            0
        } else {
            image_idx % plan.base_image.batch
        };
        let mask_idx = if plan.overlay_mask.batch == 1 {
            0
        } else {
            image_idx % plan.overlay_mask.batch
        };
        let output_offset = image_idx * base_image_bytes;
        let base_offset = base_idx * base_image_bytes;
        let mask_offset = mask_idx * base_image_bytes;
        let crop_offset = image_idx * crop_image_bytes;
        data[output_offset..output_offset + base_image_bytes]
            .copy_from_slice(&plan.base_image.data[base_offset..base_offset + base_image_bytes]);
        for y in 0..plan.paste_region.height {
            let target_y = plan.paste_region.y + y;
            for x in 0..plan.paste_region.width {
                let target_x = plan.paste_region.x + x;
                let full_pixel = (target_y * plan.base_image.width + target_x) * 3;
                let crop_pixel = (y * plan.paste_region.width + x) * 3;
                let mask_luma = (plan.overlay_mask.data[mask_offset + full_pixel] as f32
                    + plan.overlay_mask.data[mask_offset + full_pixel + 1] as f32
                    + plan.overlay_mask.data[mask_offset + full_pixel + 2] as f32)
                    / (3.0 * 255.0);
                let weight = mask_luma.clamp(0.0, 1.0);
                if weight == 0.0 {
                    continue;
                }
                for channel in 0..3 {
                    let dst = output_offset + full_pixel + channel;
                    let base = plan.base_image.data[base_offset + full_pixel + channel] as f32;
                    let generated =
                        resized_generated.data[crop_offset + crop_pixel + channel] as f32;
                    data[dst] = (base * (1.0 - weight) + generated * weight)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(RgbImageBatch {
        batch: generated.batch,
        width: plan.base_image.width,
        height: plan.base_image.height,
        data,
    })
}

fn sdapi_mask_crop_region(
    mask: &RgbImageBatch,
    padding: u32,
) -> Result<Option<SdapiCropRegion>, DiffusionError> {
    sdapi_validate_rgb_batch_len(mask, "mask")?;
    let mut min_x = mask.width;
    let mut min_y = mask.height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;
    let image_bytes = mask.width * mask.height * 3;
    for batch_idx in 0..mask.batch {
        let batch_offset = batch_idx * image_bytes;
        for y in 0..mask.height {
            for x in 0..mask.width {
                let idx = batch_offset + (y * mask.width + x) * 3;
                if mask.data[idx] != 0 || mask.data[idx + 1] != 0 || mask.data[idx + 2] != 0 {
                    found = true;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x + 1);
                    max_y = max_y.max(y + 1);
                }
            }
        }
    }
    if !found {
        return Ok(None);
    }
    let padding = usize::try_from(padding).map_err(|_| {
        DiffusionError::InvalidRequest("inpaint_full_res_padding is out of range".to_string())
    })?;
    let x1 = min_x.saturating_sub(padding);
    let y1 = min_y.saturating_sub(padding);
    let x2 = max_x.saturating_add(padding).min(mask.width);
    let y2 = max_y.saturating_add(padding).min(mask.height);
    Ok(Some(SdapiCropRegion {
        x: x1,
        y: y1,
        width: x2.saturating_sub(x1),
        height: y2.saturating_sub(y1),
    }))
}

fn sdapi_expand_crop_region(
    region: SdapiCropRegion,
    processing_dimensions: (u32, u32),
    image_dimensions: (usize, usize),
) -> Result<SdapiCropRegion, DiffusionError> {
    if region.width == 0 || region.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "inpaint crop region dimensions must be positive".to_string(),
        ));
    }
    if processing_dimensions.0 == 0 || processing_dimensions.1 == 0 {
        return Err(DiffusionError::InvalidRequest(
            "inpaint processing dimensions must be positive".to_string(),
        ));
    }
    let image_width = i64::try_from(image_dimensions.0)
        .map_err(|_| DiffusionError::InvalidRequest("image width is out of range".to_string()))?;
    let image_height = i64::try_from(image_dimensions.1)
        .map_err(|_| DiffusionError::InvalidRequest("image height is out of range".to_string()))?;
    let mut x1 = i64::try_from(region.x)
        .map_err(|_| DiffusionError::InvalidRequest("crop x is out of range".to_string()))?;
    let mut y1 = i64::try_from(region.y)
        .map_err(|_| DiffusionError::InvalidRequest("crop y is out of range".to_string()))?;
    let mut x2 = i64::try_from(region.x + region.width)
        .map_err(|_| DiffusionError::InvalidRequest("crop width is out of range".to_string()))?;
    let mut y2 = i64::try_from(region.y + region.height)
        .map_err(|_| DiffusionError::InvalidRequest("crop height is out of range".to_string()))?;
    let ratio_crop_region = (x2 - x1) as f64 / (y2 - y1) as f64;
    let ratio_processing = processing_dimensions.0 as f64 / processing_dimensions.1 as f64;
    if ratio_crop_region > ratio_processing {
        let desired_height = (x2 - x1) as f64 / ratio_processing;
        let desired_height_diff = (desired_height as i64) - (y2 - y1);
        y1 -= desired_height_diff / 2;
        y2 += desired_height_diff - desired_height_diff / 2;
        if y2 >= image_height {
            let diff = y2 - image_height;
            y2 -= diff;
            y1 -= diff;
        }
        if y1 < 0 {
            y2 -= y1;
            y1 = 0;
        }
        if y2 >= image_height {
            y2 = image_height;
        }
    } else {
        let desired_width = (y2 - y1) as f64 * ratio_processing;
        let desired_width_diff = (desired_width as i64) - (x2 - x1);
        x1 -= desired_width_diff / 2;
        x2 += desired_width_diff - desired_width_diff / 2;
        if x2 >= image_width {
            let diff = x2 - image_width;
            x2 -= diff;
            x1 -= diff;
        }
        if x1 < 0 {
            x2 -= x1;
            x1 = 0;
        }
        if x2 >= image_width {
            x2 = image_width;
        }
    }
    if x2 <= x1 || y2 <= y1 {
        return Err(DiffusionError::InvalidRequest(
            "expanded inpaint crop region is empty".to_string(),
        ));
    }
    Ok(SdapiCropRegion {
        x: usize::try_from(x1)
            .map_err(|_| DiffusionError::InvalidRequest("crop x is negative".to_string()))?,
        y: usize::try_from(y1)
            .map_err(|_| DiffusionError::InvalidRequest("crop y is negative".to_string()))?,
        width: usize::try_from(x2 - x1).map_err(|_| {
            DiffusionError::InvalidRequest("crop width is out of range".to_string())
        })?,
        height: usize::try_from(y2 - y1).map_err(|_| {
            DiffusionError::InvalidRequest("crop height is out of range".to_string())
        })?,
    })
}

fn sdapi_crop_rgb_batch(
    image: &RgbImageBatch,
    region: SdapiCropRegion,
) -> Result<RgbImageBatch, DiffusionError> {
    sdapi_validate_rgb_batch_len(image, "image")?;
    if region.width == 0 || region.height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "crop dimensions must be positive".to_string(),
        ));
    }
    if region.x + region.width > image.width || region.y + region.height > image.height {
        return Err(DiffusionError::InvalidRequest(format!(
            "crop region {}x{}+{}+{} exceeds image {}x{}",
            region.width, region.height, region.x, region.y, image.width, image.height
        )));
    }
    let source_image_bytes = image.width * image.height * 3;
    let target_image_bytes = region
        .width
        .checked_mul(region.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("crop dimensions overflow".to_string()))?;
    let mut data = vec![0u8; image.batch * target_image_bytes];
    for batch_idx in 0..image.batch {
        let source_offset = batch_idx * source_image_bytes;
        let target_offset = batch_idx * target_image_bytes;
        for y in 0..region.height {
            let source_row = source_offset + ((region.y + y) * image.width + region.x) * 3;
            let target_row = target_offset + y * region.width * 3;
            let row_bytes = region.width * 3;
            data[target_row..target_row + row_bytes]
                .copy_from_slice(&image.data[source_row..source_row + row_bytes]);
        }
    }
    Ok(RgbImageBatch {
        batch: image.batch,
        width: region.width,
        height: region.height,
        data,
    })
}

fn sdapi_apply_inpainting_mask_options(
    body: &SdGenerationRequest,
    mask: RgbImageBatch,
) -> Result<RgbImageBatch, DiffusionError> {
    let mask = if sdapi_mask_round(body) {
        sdapi_round_mask(mask)?
    } else {
        mask
    };
    let mut mask = if sdapi_inpainting_mask_invert(body) {
        RgbImageBatch {
            data: mask
                .data
                .into_iter()
                .map(|byte| 255u8.saturating_sub(byte))
                .collect(),
            ..mask
        }
    } else {
        mask
    };
    let (blur_x, blur_y) = sdapi_mask_blur_axes(body)?;
    if blur_x > 0.0 || blur_y > 0.0 {
        mask = sdapi_blur_mask(mask, blur_x, blur_y)?;
    }
    Ok(mask)
}

fn sdapi_apply_inpainting_fill_to_init_image(
    mut init_image: RgbImageBatch,
    mask: &RgbImageBatch,
    inpainting_fill: Option<u32>,
) -> Result<(RgbImageBatch, bool), DiffusionError> {
    sdapi_validate_rgb_batch_len(&init_image, "init image")?;
    sdapi_validate_rgb_batch_len(mask, "mask")?;
    if init_image.width != mask.width || init_image.height != mask.height {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpainting_fill mask dimensions {}x{} do not match init image {}x{}",
            mask.width, mask.height, init_image.width, init_image.height
        )));
    }
    if mask.batch != 1 && mask.batch != init_image.batch {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpainting_fill mask batch {} must be 1 or match init image batch {}",
            mask.batch, init_image.batch
        )));
    }
    let mode = inpainting_fill.unwrap_or(0);
    if mode == 1 {
        return Ok((init_image, false));
    }

    let width = u32::try_from(init_image.width).map_err(|_| {
        DiffusionError::InvalidRequest("init image width is out of range".to_string())
    })?;
    let height = u32::try_from(init_image.height).map_err(|_| {
        DiffusionError::InvalidRequest("init image height is out of range".to_string())
    })?;
    for batch in 0..init_image.batch {
        let mask_batch = if mask.batch == 1 { 0 } else { batch };
        let mut masked = RgbaImage::new(width, height);
        for y in 0..init_image.height {
            for x in 0..init_image.width {
                let mask_luma = sdapi_mask_luma(mask, mask_batch, x, y);
                let alpha = 255u8.saturating_sub(mask_luma);
                let mut rgba = [0u8; 4];
                for (channel, value) in rgba.iter_mut().take(3).enumerate() {
                    let source =
                        init_image.data[sdapi_rgb_index(&init_image, batch, x, y, channel)];
                    *value = ((source as u16 * alpha as u16 + 127) / 255) as u8;
                }
                rgba[3] = alpha;
                masked.put_pixel(x as u32, y as u32, Rgba(rgba));
            }
        }

        let mut filled = RgbaImage::new(width, height);
        for (sigma, repeats) in [
            (256.0f32, 1usize),
            (64.0, 1),
            (16.0, 2),
            (4.0, 4),
            (2.0, 2),
            (0.0, 1),
        ] {
            let blurred = if sigma == 0.0 {
                masked.clone()
            } else {
                image::imageops::blur(&masked, sigma)
            };
            for _ in 0..repeats {
                sdapi_alpha_composite_premul(&mut filled, &blurred);
            }
        }

        for y in 0..init_image.height {
            for x in 0..init_image.width {
                let pixel = filled.get_pixel(x as u32, y as u32).0;
                let alpha = pixel[3] as u16;
                for (channel, value) in pixel.iter().take(3).enumerate() {
                    let target = sdapi_rgb_index(&init_image, batch, x, y, channel);
                    if alpha == 0 {
                        continue;
                    }
                    init_image.data[target] =
                        ((*value as u16 * 255 + alpha / 2) / alpha).min(255) as u8;
                }
            }
        }
    }

    Ok((init_image, true))
}

fn sdapi_alpha_composite_premul(target: &mut RgbaImage, source: &RgbaImage) {
    for (target_pixel, source_pixel) in target.pixels_mut().zip(source.pixels()) {
        let source = source_pixel.0;
        let target = &mut target_pixel.0;
        let inverse_alpha = 255u16.saturating_sub(source[3] as u16);
        for channel in 0..3 {
            target[channel] = (source[channel] as u16
                + ((target[channel] as u16 * inverse_alpha + 127) / 255))
                .min(255) as u8;
        }
        target[3] =
            (source[3] as u16 + ((target[3] as u16 * inverse_alpha + 127) / 255)).min(255) as u8;
    }
}

fn sdapi_mask_luma(mask: &RgbImageBatch, batch: usize, x: usize, y: usize) -> u8 {
    let base = sdapi_rgb_index(mask, batch, x, y, 0);
    ((mask.data[base] as u16 + mask.data[base + 1] as u16 + mask.data[base + 2] as u16 + 1) / 3)
        as u8
}

fn sdapi_annotate_img2img_inpainting_info(
    output: &mut DiffusionBatchOutput,
    image_fill_applied: bool,
    inpainting_fill: Option<u32>,
) {
    if !image_fill_applied {
        return;
    }
    let Value::Object(map) = &mut output.info else {
        return;
    };
    let mode = inpainting_fill.unwrap_or(0);
    map.insert("inpainting_fill".to_string(), json!(mode));
    if mode == 0 {
        map.insert(
            "masked_content".to_string(),
            Value::String("fill".to_string()),
        );
    }
}

fn sdapi_mask_round(body: &SdGenerationRequest) -> bool {
    body.mask_round.as_ref().map_or(true, sdapi_value_is_truthy)
}

fn sdapi_round_mask(mask: RgbImageBatch) -> Result<RgbImageBatch, DiffusionError> {
    sdapi_validate_rgb_batch_len(&mask, "mask")?;
    Ok(RgbImageBatch {
        data: mask
            .data
            .into_iter()
            .map(|value| if value > 128 { 255 } else { 0 })
            .collect(),
        ..mask
    })
}

fn sdapi_mask_blur_axes(body: &SdGenerationRequest) -> Result<(f32, f32), DiffusionError> {
    let shared = sdapi_optional_nonnegative_f32(body.mask_blur.as_ref(), "mask_blur")?;
    let blur_x =
        sdapi_optional_nonnegative_f32(body.mask_blur_x.as_ref(), "mask_blur_x")?.or(shared);
    let blur_y =
        sdapi_optional_nonnegative_f32(body.mask_blur_y.as_ref(), "mask_blur_y")?.or(shared);
    Ok((blur_x.unwrap_or(0.0), blur_y.unwrap_or(0.0)))
}

fn sdapi_optional_nonnegative_f32(
    value: Option<&Value>,
    label: &str,
) -> Result<Option<f32>, DiffusionError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed = match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.trim().parse::<f64>().ok(),
        _ => None,
    }
    .ok_or_else(|| {
        DiffusionError::InvalidRequest(format!("{label} must be a non-negative number"))
    })?;
    if !parsed.is_finite() || parsed < 0.0 || parsed > f32::MAX as f64 {
        return Err(DiffusionError::InvalidRequest(format!(
            "{label} must be a finite non-negative number"
        )));
    }
    Ok(Some(parsed as f32))
}

fn sdapi_blur_mask(
    mask: RgbImageBatch,
    sigma_x: f32,
    sigma_y: f32,
) -> Result<RgbImageBatch, DiffusionError> {
    sdapi_validate_rgb_batch_len(&mask, "mask")?;
    let mut data = mask
        .data
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    if sigma_x > 0.0 {
        data = sdapi_blur_mask_axis(&data, &mask, &sdapi_gaussian_kernel(sigma_x), true);
    }
    if sigma_y > 0.0 {
        data = sdapi_blur_mask_axis(&data, &mask, &sdapi_gaussian_kernel(sigma_y), false);
    }
    Ok(RgbImageBatch {
        data: data
            .into_iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect(),
        ..mask
    })
}

fn sdapi_validate_rgb_batch_len(image: &RgbImageBatch, label: &str) -> Result<(), DiffusionError> {
    let expected = image
        .batch
        .checked_mul(image.width)
        .and_then(|value| value.checked_mul(image.height))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest(format!("{label} dimensions overflow")))?;
    if image.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "{label} data length {} does not match dimensions {}x{}x{}x3",
            image.data.len(),
            image.batch,
            image.width,
            image.height
        )));
    }
    Ok(())
}

fn sdapi_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (2.5 * sigma + 0.5) as usize;
    if radius == 0 {
        return vec![1.0];
    }
    let sigma_sq = 2.0 * sigma * sigma;
    let mut kernel = (0..=(radius * 2))
        .map(|idx| {
            let offset = idx as isize - radius as isize;
            (-(offset * offset) as f32 / sigma_sq).exp()
        })
        .collect::<Vec<_>>();
    let sum: f32 = kernel.iter().sum();
    if sum > 0.0 {
        for value in &mut kernel {
            *value /= sum;
        }
    }
    kernel
}

fn sdapi_blur_mask_axis(
    input: &[f32],
    mask: &RgbImageBatch,
    kernel: &[f32],
    horizontal: bool,
) -> Vec<f32> {
    if kernel.len() <= 1 {
        return input.to_vec();
    }
    let radius = kernel.len() / 2;
    let mut output = vec![0.0; input.len()];
    for batch in 0..mask.batch {
        for y in 0..mask.height {
            for x in 0..mask.width {
                for channel in 0..3 {
                    let mut acc = 0.0;
                    for (kernel_idx, weight) in kernel.iter().enumerate() {
                        let offset = kernel_idx as isize - radius as isize;
                        let source_x = if horizontal {
                            (x as isize + offset).clamp(0, mask.width.saturating_sub(1) as isize)
                                as usize
                        } else {
                            x
                        };
                        let source_y = if horizontal {
                            y
                        } else {
                            (y as isize + offset).clamp(0, mask.height.saturating_sub(1) as isize)
                                as usize
                        };
                        acc += input[sdapi_rgb_index(mask, batch, source_x, source_y, channel)]
                            * weight;
                    }
                    output[sdapi_rgb_index(mask, batch, x, y, channel)] = acc;
                }
            }
        }
    }
    output
}

fn sdapi_rgb_index(
    image: &RgbImageBatch,
    batch: usize,
    x: usize,
    y: usize,
    channel: usize,
) -> usize {
    ((batch * image.height + y) * image.width + x) * 3 + channel
}

fn sdapi_inpainting_mask_invert(body: &SdGenerationRequest) -> bool {
    body.inpainting_mask_invert
        .as_ref()
        .is_some_and(sdapi_value_is_truthy)
}

fn sdapi_inpainting_fill(body: &SdGenerationRequest) -> Result<Option<u32>, DiffusionError> {
    let Some(value) = body.inpainting_fill.as_ref() else {
        return Ok(None);
    };
    let mode = match value {
        Value::Number(value) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        Value::String(value) => value.trim().parse::<u32>().ok(),
        _ => None,
    }
    .ok_or_else(|| {
        DiffusionError::InvalidRequest("inpainting_fill must be 0, 1, 2, or 3".to_string())
    })?;
    if mode > 3 {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpainting_fill {mode} must be 0, 1, 2, or 3"
        )));
    }
    Ok(Some(mode))
}

fn sdapi_inpaint_full_res(body: &SdGenerationRequest) -> bool {
    body.inpaint_full_res
        .as_ref()
        .map_or(true, sdapi_value_is_truthy)
}

fn sdapi_inpaint_full_res_padding(body: &SdGenerationRequest) -> Result<u32, DiffusionError> {
    Ok(sdapi_optional_nonnegative_u32(
        body.inpaint_full_res_padding.as_ref(),
        "inpaint_full_res_padding",
    )?
    .unwrap_or(0))
}

fn sdapi_optional_nonnegative_u32(
    value: Option<&Value>,
    label: &str,
) -> Result<Option<u32>, DiffusionError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed = match value {
        Value::Number(value) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        Value::String(value) => value.trim().parse::<u32>().ok(),
        _ => None,
    }
    .ok_or_else(|| {
        DiffusionError::InvalidRequest(format!("{label} must be a non-negative integer"))
    })?;
    Ok(Some(parsed))
}

fn sdapi_value_is_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Number(value) => value
            .as_i64()
            .map(|value| value != 0)
            .or_else(|| value.as_u64().map(|value| value != 0))
            .or_else(|| value.as_f64().map(|value| value != 0.0))
            .unwrap_or(false),
        Value::String(value) => {
            let value = value.trim();
            value
                .parse::<f64>()
                .map(|value| value != 0.0)
                .unwrap_or_else(|_| {
                    matches!(value.to_ascii_lowercase().as_str(), "true" | "yes" | "on")
                })
        }
        _ => false,
    }
}

fn sd_request_generation_runtime_options(
    body: &SdGenerationRequest,
    stored_options: &std::collections::HashMap<String, Value>,
) -> DiffusionGenerationRuntimeOptions {
    let rocm_device_id = body
        .rocm_device_id
        .or(body.hipfire_rocm_device_id)
        .or_else(|| sd_override_i32(body, "rocm_device_id"))
        .or_else(|| sd_override_i32(body, "hipfire_rocm_device_id"))
        .or_else(|| sd_stored_i32(stored_options, "rocm_device_id"))
        .or_else(|| sd_stored_i32(stored_options, "hipfire_rocm_device_id"));
    rocm_device_id.map_or_else(
        DiffusionGenerationRuntimeOptions::cpu_reference,
        DiffusionGenerationRuntimeOptions::rocm_hybrid,
    )
}

fn sd_override_i32(body: &SdGenerationRequest, key: &str) -> Option<i32> {
    body.override_settings
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|settings| settings.get(key))
        .and_then(value_to_i32)
}

fn sd_stored_i32(
    stored_options: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Option<i32> {
    stored_options.get(key).and_then(value_to_i32)
}

fn sd_stored_bool(
    stored_options: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Option<bool> {
    stored_options.get(key).and_then(value_to_bool)
}

fn value_to_i32(value: &Value) -> Option<i32> {
    value.as_i64().and_then(|value| i32::try_from(value).ok())
}

fn value_to_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::Number(value) => value
            .as_i64()
            .map(|value| value != 0)
            .or_else(|| value.as_u64().map(|value| value != 0))
            .or_else(|| value.as_f64().map(|value| value != 0.0)),
        Value::String(value) => {
            let value = value.trim();
            if value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("no")
                || value.eq_ignore_ascii_case("off")
            {
                Some(false)
            } else if value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("on")
            {
                Some(true)
            } else {
                value.parse::<f64>().ok().map(|value| value != 0.0)
            }
        }
        _ => None,
    }
}

fn batch_size_for_body(body: &SdGenerationRequest) -> u32 {
    body.batch_size.unwrap_or(1).max(1)
}

fn sd_request_n_iter(body: &SdGenerationRequest) -> u32 {
    body.n_iter.unwrap_or(1).max(1)
}

fn merge_diffusion_outputs(
    outputs: Vec<DiffusionBatchOutput>,
) -> Result<DiffusionBatchOutput, DiffusionError> {
    let mut iter = outputs.into_iter();
    let Some(mut merged) = iter.next() else {
        return Err(DiffusionError::InvalidRequest(
            "n_iter must produce at least one diffusion output".to_string(),
        ));
    };
    for output in iter {
        merged.images.extend(output.images);
        merge_generation_info(&mut merged.info, output.info);
    }
    Ok(merged)
}

fn merge_generation_info(merged: &mut Value, next: Value) {
    let (Value::Object(merged_map), Value::Object(next_map)) = (merged, next) else {
        return;
    };
    for key in ["seeds", "subseeds", "infotexts", "saved_images"] {
        if let Some(Value::Array(next_values)) = next_map.get(key) {
            match merged_map.get_mut(key) {
                Some(Value::Array(values)) => values.extend(next_values.clone()),
                _ => {
                    merged_map.insert(key.to_string(), Value::Array(next_values.clone()));
                }
            }
        }
    }
    if let (Some(Value::Number(left)), Some(Value::Number(right))) =
        (merged_map.get("batch_size"), next_map.get("batch_size"))
    {
        if let (Some(left), Some(right)) = (left.as_u64(), right.as_u64()) {
            merged_map.insert("batch_size".to_string(), json!(left.saturating_add(right)));
        }
    }
}

fn decode_sd_init_images(images: &[String]) -> Result<RgbImageBatch, DiffusionError> {
    if images.is_empty() {
        return Err(DiffusionError::InvalidRequest(
            "img2img requires at least one init image".to_string(),
        ));
    }
    let mut decoded = Vec::with_capacity(images.len());
    for image in images {
        decoded.push(decode_sd_init_image(image)?);
    }
    let width = decoded[0].width;
    let height = decoded[0].height;
    let bytes_per_image = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("init image dimensions overflow".to_string())
        })?;
    let mut data = Vec::with_capacity(bytes_per_image * decoded.len());
    for (idx, image) in decoded.into_iter().enumerate() {
        if image.width != width || image.height != height {
            return Err(DiffusionError::InvalidRequest(format!(
                "init image {idx} dimensions {}x{} do not match first init image {width}x{height}",
                image.width, image.height
            )));
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

fn decode_sd_init_image(image: &str) -> Result<RgbImageBatch, DiffusionError> {
    let payload = image
        .split_once(',')
        .map(|(_, payload)| payload)
        .unwrap_or(image);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|error| {
            DiffusionError::InvalidRequest(format!("invalid init image base64: {error}"))
        })?;
    let image = image::load_from_memory(&bytes)
        .map_err(|error| DiffusionError::InvalidRequest(format!("invalid init image: {error}")))?
        .to_rgb8();
    let width = usize::try_from(image.width()).map_err(|_| {
        DiffusionError::InvalidRequest("init image width does not fit usize".to_string())
    })?;
    let height = usize::try_from(image.height()).map_err(|_| {
        DiffusionError::InvalidRequest("init image height does not fit usize".to_string())
    })?;
    Ok(RgbImageBatch {
        batch: 1,
        width,
        height,
        data: image.into_raw(),
    })
}

fn diffusion_error_response(error: DiffusionError) -> Response {
    let (status, error_type) = match error {
        DiffusionError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        DiffusionError::InvalidMetadata(_) => (StatusCode::BAD_REQUEST, "invalid_model_error"),
        DiffusionError::BackendUnavailable(_) => {
            (StatusCode::NOT_IMPLEMENTED, "not_implemented_error")
        }
        DiffusionError::Interrupted(_) => (StatusCode::CONFLICT, "interrupted_error"),
        DiffusionError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
    };
    (
        status,
        Json(json!({
            "error": {
                "message": error.to_string(),
                "type": error_type
            }
        })),
    )
        .into_response()
}

fn sd_request_to_chat_request(
    body: &SdGenerationRequest,
    image_base64: Option<String>,
) -> ChatRequest {
    let mut prompt = body.prompt.clone();
    if prompt.is_empty() {
        prompt = body.infotext.clone().unwrap_or_default();
    }
    if !body.negative_prompt.is_empty() {
        prompt.push_str("\n\nNegative prompt: ");
        prompt.push_str(&body.negative_prompt);
    }

    let content = match image_base64 {
        Some(image) => Value::Array(vec![
            json!({"type": "text", "text": prompt}),
            json!({"type": "image_url", "image_url": {"url": normalize_sd_image_data_url(&image)}}),
        ]),
        None => Value::String(prompt),
    };

    ChatRequest {
        model: sd_requested_model(body),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
        }],
        stream: false,
        temperature: body.temperature,
        top_p: body.top_p,
        repeat_penalty: body.repeat_penalty,
        presence_penalty: None,
        frequency_penalty: None,
        max_tokens: body.max_tokens.or(body.steps),
        stop: body.stop.clone(),
        priority: None,
        tools: None,
        system: None,
        reasoning_effort: None,
        reasoning: None,
        stream_options: None,
        chat_template_kwargs: None,
    }
}

fn sd_requested_model(body: &SdGenerationRequest) -> Option<String> {
    body.model
        .as_deref()
        .filter(|model| !model.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            body.override_settings
                .as_ref()
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("sd_model_checkpoint"))
                .and_then(Value::as_str)
                .filter(|model| !model.trim().is_empty())
                .map(str::to_string)
        })
}

fn normalize_sd_image_data_url(image: &str) -> String {
    if image.starts_with("data:image/") {
        image.to_string()
    } else {
        format!("data:image/png;base64,{image}")
    }
}

fn start_sdapi_progress(
    progress_state: &Arc<std::sync::Mutex<SdapiProgressState>>,
    body: &SdGenerationRequest,
    mode: &str,
    sampling_steps: usize,
) {
    if let Ok(mut progress) = progress_state.lock() {
        *progress = SdapiProgressState {
            active: true,
            interrupted: false,
            task_id: body
                .force_task_id
                .clone()
                .or_else(|| Some(format!("hipfire-{mode}-{}", sdapi_now_secs()))),
            mode: Some(mode.to_string()),
            prompt: Some(body.prompt.clone()),
            sampling_step: 0,
            sampling_steps,
            current_image: None,
            textinfo: Some(format!("{mode} running")),
            started_at_unix_secs: Some(sdapi_now_secs()),
            completed_at_unix_secs: None,
        };
    }
}

fn sdapi_preview_image_from_progress(
    pipeline: &DiffusionPipeline,
    runtime_options: DiffusionGenerationRuntimeOptions,
    event: &DiffusionProgress,
) -> Result<Option<String>, DiffusionError> {
    event
        .preview_latents
        .as_ref()
        .map(|latents| {
            pipeline
                .decode_preview_latents_png_base64_with_runtime_options(latents, runtime_options)
        })
        .transpose()
}

fn update_sdapi_progress(
    progress_state: &Arc<std::sync::Mutex<SdapiProgressState>>,
    event: DiffusionProgress,
    current_image: Option<String>,
) -> Result<(), DiffusionError> {
    let mut progress = progress_state
        .lock()
        .map_err(|error| DiffusionError::Io(format!("SDAPI progress lock poisoned: {error}")))?;
    progress.sampling_step = event.completed_steps;
    progress.sampling_steps = event.total_steps;
    if current_image.is_some() {
        progress.current_image = current_image;
    }
    if progress.interrupted {
        progress.active = false;
        progress.textinfo = Some("interrupted".to_string());
        progress.completed_at_unix_secs = Some(sdapi_now_secs());
        return Err(DiffusionError::Interrupted(
            "SDAPI generation interrupted".to_string(),
        ));
    }
    progress.textinfo = Some(format!(
        "sampling step {}/{}",
        event.completed_steps, event.total_steps
    ));
    Ok(())
}

fn finish_sdapi_progress(
    progress_state: &Arc<std::sync::Mutex<SdapiProgressState>>,
    error: Option<&DiffusionError>,
    current_image: Option<String>,
) {
    if let Ok(mut progress) = progress_state.lock() {
        progress.active = false;
        progress.completed_at_unix_secs = Some(sdapi_now_secs());
        match error {
            Some(DiffusionError::Interrupted(_)) => {
                progress.interrupted = true;
                progress.textinfo = Some("interrupted".to_string());
            }
            Some(error) => {
                progress.textinfo = Some(error.to_string());
            }
            None => {
                progress.sampling_step = progress.sampling_steps;
                progress.current_image = current_image;
                progress.textinfo = Some("complete".to_string());
            }
        }
    }
}

fn interrupt_sdapi_progress(progress_state: &Arc<std::sync::Mutex<SdapiProgressState>>) {
    if let Ok(mut progress) = progress_state.lock() {
        progress.interrupted = true;
        progress.textinfo = Some("interrupt requested".to_string());
    }
}

fn sdapi_progress_json(progress: &SdapiProgressState) -> Value {
    let ratio = if progress.sampling_steps == 0 {
        0.0
    } else {
        (progress.sampling_step as f64 / progress.sampling_steps as f64).clamp(0.0, 1.0)
    };
    json!({
        "progress": ratio,
        "eta_relative": 0.0,
        "state": {
            "skipped": false,
            "interrupted": progress.interrupted,
            "job": progress.mode,
            "job_count": 1,
            "job_no": if progress.active { 0 } else { 1 },
            "sampling_step": progress.sampling_step,
            "sampling_steps": progress.sampling_steps,
        },
        "current_image": progress.current_image,
        "textinfo": progress.textinfo,
        "current_task": progress.task_id,
    })
}

fn sdapi_img2img_denoise_steps(body: &SdGenerationRequest) -> usize {
    let steps = body.steps.unwrap_or(20).max(1) as f64;
    let strength = body.denoising_strength.unwrap_or(0.75).clamp(0.0, 1.0);
    (steps * strength).ceil() as usize
}

fn first_output_steps(body: &SdGenerationRequest) -> usize {
    body.steps.unwrap_or(20) as usize
}

fn sdapi_txt2img_highres_steps(
    body: &SdGenerationRequest,
    highres_target: Option<(u32, u32)>,
) -> usize {
    if highres_target.is_some() {
        sdapi_img2img_denoise_steps(&sdapi_highres_second_pass_body(body, (1, 1)))
    } else {
        0
    }
}

fn sdapi_txt2img_first_pass_body(
    body: &SdGenerationRequest,
) -> Result<SdGenerationRequest, DiffusionError> {
    let Some((width, height)) = sdapi_highres_first_pass_dimensions(body)? else {
        return Ok(body.clone());
    };
    let mut first_pass_body = body.clone();
    first_pass_body.width = Some(width);
    first_pass_body.height = Some(height);
    Ok(first_pass_body)
}

fn sdapi_highres_first_pass_dimensions(
    body: &SdGenerationRequest,
) -> Result<Option<(u32, u32)>, DiffusionError> {
    if !body.enable_hr.unwrap_or(false) {
        return Ok(None);
    }
    let firstphase_width = body.firstphase_width.unwrap_or(0);
    let firstphase_height = body.firstphase_height.unwrap_or(0);
    match (firstphase_width, firstphase_height) {
        (0, 0) => Ok(None),
        (width, height) if width > 0 && height > 0 => Ok(Some((width, height))),
        (width, 0) => {
            let base_width = body.width.unwrap_or(512);
            let base_height = body.height.unwrap_or(512);
            if base_width == 0 || base_height == 0 {
                return Err(DiffusionError::InvalidRequest(
                    "highres txt2img requires non-zero base width and height for firstphase_width"
                        .to_string(),
                ));
            }
            Ok(Some((
                width,
                aspect_scaled_dimension(width, base_height, base_width, "first-pass height")?,
            )))
        }
        (0, height) => {
            let base_width = body.width.unwrap_or(512);
            let base_height = body.height.unwrap_or(512);
            if base_width == 0 || base_height == 0 {
                return Err(DiffusionError::InvalidRequest(
                    "highres txt2img requires non-zero base width and height for firstphase_height"
                        .to_string(),
                ));
            }
            Ok(Some((
                aspect_scaled_dimension(height, base_width, base_height, "first-pass width")?,
                height,
            )))
        }
        _ => unreachable!("zero firstphase dimensions are handled by earlier match arms"),
    }
}

fn sdapi_highres_second_pass_body(
    body: &SdGenerationRequest,
    target_dimensions: (u32, u32),
) -> SdGenerationRequest {
    let mut highres_body = body.clone();
    highres_body.width = Some(target_dimensions.0);
    highres_body.height = Some(target_dimensions.1);
    highres_body.steps = Some(
        body.hr_second_pass_steps
            .unwrap_or_else(|| body.steps.unwrap_or(20))
            .max(1),
    );
    highres_body.enable_hr = Some(false);
    highres_body.init_images = None;
    highres_body.mask = None;
    if let Some(prompt) = highres_override_text(body.hr_prompt.as_deref(), "") {
        highres_body.prompt = prompt.to_string();
    }
    if let Some(negative_prompt) = highres_override_text(body.hr_negative_prompt.as_deref(), "") {
        highres_body.negative_prompt = negative_prompt.to_string();
    }
    if let Some(sampler) =
        highres_override_text(body.hr_sampler_name.as_deref(), "Use same sampler")
    {
        highres_body.sampler_name = Some(sampler.to_string());
        highres_body.sampler_index = None;
        if highres_body
            .scheduler
            .as_deref()
            .is_some_and(|scheduler| !sdapi_scheduler_is_schedule_modifier(scheduler))
        {
            highres_body.scheduler = None;
        }
    }
    if let Some(scheduler) =
        highres_override_text(body.hr_scheduler.as_deref(), "Use same scheduler")
    {
        highres_body.scheduler = Some(scheduler.to_string());
    }
    highres_body
}

fn sdapi_highres_second_pass_init_images(
    body: &SdGenerationRequest,
    init_images: RgbImageBatch,
    target_dimensions: (u32, u32),
) -> Result<RgbImageBatch, DiffusionError> {
    if body.hr_resize_x.unwrap_or(0) > 0 && body.hr_resize_y.unwrap_or(0) > 0 {
        resize_rgb_batch_to_cover_nearest(&init_images, target_dimensions.0, target_dimensions.1)
    } else {
        Ok(init_images)
    }
}

async fn highres_diffusion_pipeline_for_request(
    state: &SharedState,
    body: &SdGenerationRequest,
    highres_target: Option<(u32, u32)>,
    first_pass_pipeline: &Arc<DiffusionPipeline>,
) -> Result<Arc<DiffusionPipeline>, DiffusionError> {
    if highres_target.is_none() || !body.enable_hr.unwrap_or(false) {
        return Ok(first_pass_pipeline.clone());
    }
    let Some(checkpoint) =
        highres_override_text(body.hr_checkpoint_name.as_deref(), "Use same checkpoint")
    else {
        return Ok(first_pass_pipeline.clone());
    };
    if diffusion_summary_matches_candidate(first_pass_pipeline.summary(), checkpoint) {
        return Ok(first_pass_pipeline.clone());
    }
    let Some(path) = resolve_diffusion_hfq_for_request(state, Some(checkpoint)).await else {
        return Err(DiffusionError::InvalidRequest(format!(
            "hr_checkpoint_name {checkpoint:?} could not be resolved to a diffusion HFQ artifact"
        )));
    };
    cached_diffusion_pipeline(state, path).await
}

fn highres_override_text<'a>(value: Option<&'a str>, same_label: &str) -> Option<&'a str> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != same_label)
}

fn sdapi_highres_target_dimensions(
    body: &SdGenerationRequest,
) -> Result<Option<(u32, u32)>, DiffusionError> {
    if !body.enable_hr.unwrap_or(false) {
        return Ok(None);
    }
    let base_width = body.width.unwrap_or(512);
    let base_height = body.height.unwrap_or(512);
    if base_width == 0 || base_height == 0 {
        return Err(DiffusionError::InvalidRequest(
            "highres txt2img requires non-zero base width and height".to_string(),
        ));
    }
    let resize_x = body.hr_resize_x.unwrap_or(0);
    let resize_y = body.hr_resize_y.unwrap_or(0);
    let target = match (resize_x, resize_y) {
        (0, 0) => {
            let scale = body.hr_scale.unwrap_or(2.0);
            if !scale.is_finite() || scale <= 0.0 {
                return Err(DiffusionError::InvalidRequest(
                    "highres txt2img requires a positive finite hr_scale".to_string(),
                ));
            }
            (
                scaled_highres_dimension(base_width, scale, "width")?,
                scaled_highres_dimension(base_height, scale, "height")?,
            )
        }
        (width, 0) => {
            if width == 0 {
                unreachable!("zero width handled by match arm");
            }
            (
                width,
                aspect_scaled_dimension(width, base_height, base_width, "height")?,
            )
        }
        (0, height) => (
            aspect_scaled_dimension(height, base_width, base_height, "width")?,
            height,
        ),
        (width, height) => (width, height),
    };
    Ok(Some(target))
}

fn scaled_highres_dimension(
    dimension: u32,
    scale: f64,
    label: &str,
) -> Result<u32, DiffusionError> {
    let scaled = (dimension as f64 * scale).round();
    if scaled < 1.0 || scaled > u32::MAX as f64 {
        return Err(DiffusionError::InvalidRequest(format!(
            "highres txt2img target {label} is out of range"
        )));
    }
    Ok(scaled as u32)
}

fn aspect_scaled_dimension(
    fixed_dimension: u32,
    scaled_dimension: u32,
    base_dimension: u32,
    label: &str,
) -> Result<u32, DiffusionError> {
    let value = (fixed_dimension as u64)
        .saturating_mul(scaled_dimension as u64)
        .checked_div(base_dimension as u64)
        .unwrap_or(0)
        .max(1);
    u32::try_from(value).map_err(|_| {
        DiffusionError::InvalidRequest(format!("highres txt2img target {label} is out of range"))
    })
}

fn annotate_highres_txt2img_info(
    info: &mut Value,
    body: &SdGenerationRequest,
    first_pass_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) {
    if let Value::Object(map) = info {
        map.insert("mode".to_string(), json!("txt2img-hires"));
        map.insert("highres".to_string(), json!(true));
        map.insert(
            "firstpass_width".to_string(),
            json!(first_pass_dimensions.0),
        );
        map.insert(
            "firstpass_height".to_string(),
            json!(first_pass_dimensions.1),
        );
        map.insert("hr_width".to_string(), json!(target_dimensions.0));
        map.insert("hr_height".to_string(), json!(target_dimensions.1));
        map.insert(
            "hr_second_pass_steps".to_string(),
            json!(body
                .hr_second_pass_steps
                .unwrap_or_else(|| body.steps.unwrap_or(20))
                .max(1)),
        );
        map.insert(
            "denoising_strength".to_string(),
            json!(body.denoising_strength.unwrap_or(0.75)),
        );
        if let Some(upscaler) = highres_override_text(body.hr_upscaler.as_deref(), "") {
            map.insert("hr_upscaler".to_string(), json!(upscaler));
        }
        if let Some(checkpoint) =
            highres_override_text(body.hr_checkpoint_name.as_deref(), "Use same checkpoint")
        {
            map.insert("hr_checkpoint_name".to_string(), json!(checkpoint));
        }
        if let Some(prompt) = highres_override_text(body.hr_prompt.as_deref(), "") {
            map.insert("hr_prompt".to_string(), json!(prompt));
        }
        if let Some(negative_prompt) = highres_override_text(body.hr_negative_prompt.as_deref(), "")
        {
            map.insert("hr_negative_prompt".to_string(), json!(negative_prompt));
        }
        if let Some(sampler) =
            highres_override_text(body.hr_sampler_name.as_deref(), "Use same sampler")
        {
            map.insert("hr_sampler_name".to_string(), json!(sampler));
        }
        if let Some(scheduler) =
            highres_override_text(body.hr_scheduler.as_deref(), "Use same scheduler")
        {
            map.insert("hr_scheduler".to_string(), json!(scheduler));
        }
    }
}

fn sdapi_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub async fn post_png_info(Json(body): Json<Value>) -> Json<Value> {
    let info = body
        .get("image")
        .and_then(Value::as_str)
        .and_then(|image| decode_base64_image_payload(image).ok())
        .and_then(|bytes| extract_png_text_chunk(&bytes, "parameters").ok().flatten())
        .unwrap_or_default();
    let parameters = sdapi_png_info_parameters(&info);
    Json(json!({
        "info": info,
        "items": {},
        "parameters": parameters,
    }))
}

fn sdapi_png_info_parameters(info: &str) -> Value {
    let Some(parsed) = sdapi_parse_infotext(info) else {
        return json!({});
    };
    let mut parameters = serde_json::Map::new();
    parameters.insert("Prompt".to_string(), json!(parsed.prompt));
    parameters.insert("Negative prompt".to_string(), json!(parsed.negative_prompt));
    for (key, value) in parsed.params {
        if let Some((width, height)) = sdapi_infotext_split_dimensions(&value) {
            parameters.insert(format!("{key}-1"), json!(width));
            parameters.insert(format!("{key}-2"), json!(height));
        } else {
            parameters.insert(key, json!(value));
        }
    }
    parameters
        .entry("Clip skip".to_string())
        .or_insert_with(|| json!("1"));
    parameters
        .entry("Hires resize-1".to_string())
        .or_insert_with(|| json!(0));
    parameters
        .entry("Hires resize-2".to_string())
        .or_insert_with(|| json!(0));
    parameters
        .entry("Hires sampler".to_string())
        .or_insert_with(|| json!("Use same sampler"));
    parameters
        .entry("Hires schedule type".to_string())
        .or_insert_with(|| json!("Use same scheduler"));
    parameters
        .entry("Hires checkpoint".to_string())
        .or_insert_with(|| json!("Use same checkpoint"));
    parameters
        .entry("Hires prompt".to_string())
        .or_insert_with(|| json!(""));
    parameters
        .entry("Hires negative prompt".to_string())
        .or_insert_with(|| json!(""));
    parameters
        .entry("Mask mode".to_string())
        .or_insert_with(|| json!("Inpaint masked"));
    parameters
        .entry("Masked content".to_string())
        .or_insert_with(|| json!("original"));
    parameters
        .entry("Inpaint area".to_string())
        .or_insert_with(|| json!("Whole picture"));
    parameters
        .entry("Masked area padding".to_string())
        .or_insert_with(|| json!(32));
    parameters
        .entry("RNG".to_string())
        .or_insert_with(|| json!("GPU"));
    parameters
        .entry("Schedule type".to_string())
        .or_insert_with(|| json!("Automatic"));
    parameters
        .entry("Schedule max sigma".to_string())
        .or_insert_with(|| json!(0));
    parameters
        .entry("Schedule min sigma".to_string())
        .or_insert_with(|| json!(0));
    parameters
        .entry("Schedule rho".to_string())
        .or_insert_with(|| json!(0));
    parameters
        .entry("VAE Encoder".to_string())
        .or_insert_with(|| json!("Full"));
    parameters
        .entry("VAE Decoder".to_string())
        .or_insert_with(|| json!("Full"));
    parameters
        .entry("FP8 weight".to_string())
        .or_insert_with(|| json!("Disable"));
    Value::Object(parameters)
}

pub async fn get_progress(State(state): State<SharedState>) -> Json<Value> {
    let progress = state
        .sdapi_progress
        .lock()
        .map(|progress| progress.clone())
        .unwrap_or_default();
    Json(sdapi_progress_json(&progress))
}

pub async fn get_options(State(state): State<SharedState>) -> Json<Value> {
    let cfg = state.config.lock().await;
    let options = state.sdapi_options.lock().await;
    Json(sdapi_options_json(cfg.default_model.clone(), &options))
}

fn sdapi_options_json(
    default_model: Option<String>,
    stored_options: &std::collections::HashMap<String, Value>,
) -> Value {
    let mut options = json!({
        "sd_model_checkpoint": default_model,
        "samples_format": "png",
        "send_images": true,
        "send_seed": true,
        "save_images": false,
        "outdir_samples": "/tmp/hipfire-sdapi",
        "outdir_txt2img_samples": "/tmp/hipfire-sdapi/txt2img",
        "outdir_img2img_samples": "/tmp/hipfire-sdapi/img2img",
        "hipfire_backend": "diffusion-hfq-or-text-fallback",
        "hipfire_rocm_device_id": null,
        "hipfire_sdapi_save_images_supported": true,
        "hipfire_notice": "SD API compatibility routes generate PNG images for diffusion HFQ models and fall back to text generation for non-diffusion models.",
    });
    let Value::Object(map) = &mut options else {
        return options;
    };
    for (key, value) in stored_options {
        if key != "sd_model_checkpoint" {
            map.insert(key.clone(), value.clone());
        }
    }
    map.insert(
        "sd_model_checkpoint".to_string(),
        default_model.map(Value::String).unwrap_or(Value::Null),
    );
    options
}

pub async fn post_options(
    State(state): State<SharedState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let mut cfg = state.config.lock().await;
    if let Some(settings) = body.as_object() {
        let mut options = state.sdapi_options.lock().await;
        if let Some(checkpoint) = settings.get("sd_model_checkpoint") {
            cfg.default_model = match checkpoint {
                Value::String(model) if !model.is_empty() => Some(model.clone()),
                Value::Null => None,
                _ => cfg.default_model.clone(),
            };
        }
        for (key, value) in settings {
            if key != "sd_model_checkpoint" {
                options.insert(key.clone(), value.clone());
            }
        }
    }
    let options = state.sdapi_options.lock().await;
    Json(sdapi_options_json(cfg.default_model.clone(), &options))
}

pub async fn get_cmd_flags() -> Json<Value> {
    Json(json!({
        "api": true,
        "nowebui": true,
        "api_auth": null,
        "api_log": false,
        "api_server_stop": false,
        "compatibility": "stable-diffusion-webui-sdapi",
        "backend": "hipfire",
    }))
}

pub async fn get_samplers() -> Json<Value> {
    Json(json!([
        {
            "name": COMPAT_SAMPLER,
            "aliases": ["Euler", "Euler a", "Euler Karras", "DDIM", "DPM++ 2M", "DPM++ 2M Karras"],
            "options": {},
        }
    ]))
}

pub async fn get_schedulers() -> Json<Value> {
    Json(json!([
        {
            "name": "Automatic",
            "label": "Automatic",
            "aliases": [],
            "default_rho": null,
            "need_inner_model": false,
        },
        {
            "name": "Karras",
            "label": "Karras",
            "aliases": ["karras"],
            "default_rho": null,
            "need_inner_model": false,
        }
    ]))
}

pub async fn get_upscalers() -> Json<Value> {
    Json(
        json!([{"name": "None", "model_name": null, "model_path": null, "model_url": null, "scale": 1.0}]),
    )
}

pub async fn get_latent_upscale_modes() -> Json<Value> {
    Json(json!([]))
}

pub async fn get_sd_models() -> Json<Value> {
    let mut models = discover_diffusion_hfq_models()
        .into_iter()
        .map(|inspection| {
            let model = inspection.summary;
            let runtime_kind = inspection
                .runtime_support
                .runtime_kind
                .map(|kind| kind.as_str().to_string());
            json!({
                "title": model.title,
                "model_name": model.model_name,
                "hash": null,
                "sha256": null,
                "filename": model.path,
                "config": model.pipeline_class,
                "max_batch": model.max_batch,
                "weight_format": model.weight_format,
                "runtime_support": {
                    "metadata_supported": inspection.runtime_support.supported,
                    "runtime": runtime_kind,
                    "reason": inspection.runtime_support.reason,
                },
            })
        })
        .collect::<Vec<_>>();

    models.extend(discover_diffusers_models().into_iter().map(|model| {
        json!({
            "title": model.title,
            "model_name": model.model_name,
            "hash": null,
            "sha256": null,
            "filename": model.path,
            "config": model.pipeline_class,
        })
    }));

    models.extend(
        discover_diffusion_checkpoint_models()
            .into_iter()
            .map(|model| {
                json!({
                    "title": model.title,
                    "model_name": model.model_name,
                    "hash": null,
                    "sha256": null,
                    "filename": model.path,
                    "config": model.pipeline_class,
                })
            }),
    );

    models.extend(local_llm_registry().models.into_iter().map(|model| {
        json!({
            "title": model.id,
            "model_name": model.id,
            "hash": null,
            "sha256": null,
            "filename": model.path,
            "config": null,
        })
    }));
    Json(Value::Array(models))
}

pub async fn get_sd_vae() -> Json<Value> {
    Json(json!([]))
}

pub async fn get_hypernetworks() -> Json<Value> {
    Json(json!([]))
}

pub async fn get_face_restorers() -> Json<Value> {
    Json(json!([]))
}

pub async fn get_realesrgan_models() -> Json<Value> {
    Json(json!([]))
}

pub async fn get_prompt_styles() -> Json<Value> {
    Json(json!([]))
}

pub async fn get_embeddings() -> Json<Value> {
    Json(json!({"loaded": {}, "skipped": {}}))
}

pub async fn get_scripts() -> Json<Value> {
    Json(json!({"txt2img": [], "img2img": []}))
}

pub async fn get_script_info() -> Json<Value> {
    Json(json!([]))
}

pub async fn get_extensions() -> Json<Value> {
    Json(json!([]))
}

pub async fn post_interrupt(State(state): State<SharedState>) -> Json<Value> {
    interrupt_sdapi_progress(&state.sdapi_progress);
    Json(json!({}))
}

pub async fn post_reload_checkpoint(State(state): State<SharedState>) -> Response {
    let requested_model = {
        let cfg = state.config.lock().await;
        cfg.default_model.clone()
    };
    let Some(requested_model) = requested_model.filter(|model| !model.is_empty()) else {
        return Json(json!({
            "reloaded": false,
            "loaded": false,
            "model": null,
            "reason": "sd_model_checkpoint is not configured",
        }))
        .into_response();
    };
    let Some(path) = resolve_diffusion_hfq_candidate(&requested_model) else {
        return diffusion_error_response(DiffusionError::InvalidRequest(format!(
            "sd_model_checkpoint {requested_model:?} could not be resolved"
        )));
    };

    state.diffusion_pipelines.lock().await.remove(&path);
    let pipeline = match cached_diffusion_pipeline(&state, path.clone()).await {
        Ok(pipeline) => pipeline,
        Err(error) => return diffusion_error_response(error),
    };
    let summary = pipeline.summary();
    Json(json!({
        "reloaded": true,
        "loaded": true,
        "model": requested_model,
        "title": summary.title,
        "model_name": summary.model_name,
        "filename": path,
        "pipeline": summary.pipeline_class,
        "weight_format": summary.weight_format,
    }))
    .into_response()
}

pub async fn post_unload_checkpoint(State(state): State<SharedState>) -> Json<Value> {
    let mut cache = state.diffusion_pipelines.lock().await;
    let unloaded = cache.len();
    cache.clear();
    Json(json!({
        "unloaded": unloaded,
        "loaded": false,
    }))
}

pub async fn post_control_noop() -> Json<Value> {
    Json(json!({}))
}

pub async fn post_unsupported() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "this stable-diffusion-webui endpoint is not implemented by Hipfire's SDAPI compatibility layer",
                "type": "not_implemented_error"
            }
        })),
    )
        .into_response()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiffusersModel {
    title: String,
    model_name: String,
    path: String,
    pipeline_class: String,
}

fn requested_model_is_diffusers_pipeline(model: Option<&str>) -> bool {
    let Some(model) = model.filter(|model| !model.is_empty()) else {
        return false;
    };
    if is_single_file_checkpoint_path(Path::new(model)) {
        return true;
    }
    discover_diffusers_models().into_iter().any(|entry| {
        model == entry.title
            || model == entry.model_name
            || model == entry.path
            || model == entry.pipeline_class
    }) || discover_diffusion_checkpoint_models()
        .into_iter()
        .any(|entry| model == entry.title || model == entry.model_name || model == entry.path)
}

fn diffusion_backend_missing_response() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "diffusers Stable Diffusion models are discoverable, but hipfire serve runs image generation from diffusion HFQ artifacts; import or convert this model to .hfq before serving",
                "type": "not_implemented_error"
            }
        })),
    )
        .into_response()
}

fn discover_diffusion_hfq_models() -> Vec<DiffusionHfqInspection> {
    let mut models = list_local_models()
        .into_iter()
        .filter_map(|path| inspect_hfq_with_runtime_support(path).ok())
        .collect::<Vec<_>>();
    models.sort_by(|a, b| a.summary.model_name.cmp(&b.summary.model_name));
    models
}

fn discover_diffusers_models() -> Vec<DiffusersModel> {
    let mut roots = vec![PathBuf::from("/srv/huggingface")];
    if let Ok(root) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(root).join("hub"));
    }
    if let Ok(root) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        roots.push(PathBuf::from(root));
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    roots.sort();
    roots.dedup();

    let mut models = Vec::new();
    for root in roots {
        collect_diffusers_models_from_root(&root, &mut models);
    }
    models.sort_by(|a, b| a.model_name.cmp(&b.model_name));
    models.dedup_by(|a, b| a.path == b.path);
    models
}

fn discover_diffusion_checkpoint_models() -> Vec<DiffusersModel> {
    let mut models = Vec::new();
    collect_checkpoint_models_from_root(&models_dir(), &mut models);
    models.sort_by(|a, b| a.model_name.cmp(&b.model_name));
    models.dedup_by(|a, b| a.path == b.path);
    models
}

fn collect_checkpoint_models_from_root(root: &Path, out: &mut Vec<DiffusersModel>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let Ok(children) = fs::read_dir(&path) else {
                continue;
            };
            for child in children.flatten() {
                push_checkpoint_model_if_supported(&child.path(), out);
            }
        } else {
            push_checkpoint_model_if_supported(&path, out);
        }
    }
}

fn push_checkpoint_model_if_supported(path: &Path, out: &mut Vec<DiffusersModel>) {
    if !is_single_file_checkpoint_path(path) {
        return;
    }
    let model_name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("checkpoint")
        .to_string();
    out.push(DiffusersModel {
        title: format!("{model_name}:StableDiffusionCheckpoint"),
        model_name,
        path: path.to_string_lossy().into_owned(),
        pipeline_class: "StableDiffusionCheckpoint".to_string(),
    });
}

fn is_single_file_checkpoint_path(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                extension.eq_ignore_ascii_case("safetensors")
                    || extension.eq_ignore_ascii_case("ckpt")
            })
}

fn collect_diffusers_models_from_root(root: &Path, out: &mut Vec<DiffusersModel>) {
    let Ok(repos) = std::fs::read_dir(root) else {
        return;
    };
    for repo in repos.flatten() {
        let repo_path = repo.path();
        let Some(repo_name) = repo_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !repo_name.starts_with("models--") {
            continue;
        }
        let repo_id = repo_name.trim_start_matches("models--").replace("--", "/");
        let snapshots = repo_path.join("snapshots");
        let Ok(entries) = std::fs::read_dir(snapshots) else {
            continue;
        };
        for snapshot in entries.flatten() {
            let snapshot_path = snapshot.path();
            let index = snapshot_path.join("model_index.json");
            let Some(pipeline_class) = diffusers_pipeline_class(&index) else {
                continue;
            };
            if !is_supported_diffusion_pipeline(&pipeline_class) {
                continue;
            }
            let snapshot_id = snapshot_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            let model_name = repo_id
                .rsplit('/')
                .next()
                .filter(|name| !name.is_empty())
                .unwrap_or(repo_id.as_str())
                .to_string();
            out.push(DiffusersModel {
                title: format!("{repo_id}:{snapshot_id}"),
                model_name,
                path: snapshot_path.to_string_lossy().into_owned(),
                pipeline_class,
            });
        }
    }
}

fn diffusers_pipeline_class(index: &Path) -> Option<String> {
    let text = std::fs::read_to_string(index).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    value
        .get("_class_name")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn is_supported_diffusion_pipeline(class_name: &str) -> bool {
    matches!(
        class_name,
        "StableDiffusionPipeline"
            | "StableDiffusionXLPipeline"
            | "Krea2Pipeline"
            | "FluxPipeline"
            | "DiffusionPipeline"
    )
}

fn error_status(error: &Value) -> StatusCode {
    if error
        .get("error")
        .and_then(|inner| inner.get("type"))
        .and_then(Value::as_str)
        == Some("invalid_request_error")
    {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use base64::Engine;
    use hipfire_diffusion::{
        DiffusionBatchMetadata, DiffusionComponentMetadata, DiffusionHfqMetadata,
        DiffusionPipelineMetadata, DiffusionQuantizationMetadata, DiffusionTokenizerMetadata,
        LatentBatch, DIFFUSION_ARTIFACT_KIND, DIFFUSION_SCHEMA_VERSION, HFQ_ARCH_DIFFUSION,
        QT_DIFFUSION_JSON, QT_DIFFUSION_TENSOR_F32, QT_DIFFUSION_TOKENIZER,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use std::collections::BTreeMap;

    const DEFAULT_TINY_SD_HFQ: &str = "/tmp/hipfire-tiny-sd-diffusion.hfq";

    fn tiny_sd_hfq_path() -> std::path::PathBuf {
        std::env::var_os("HIPFIRE_TINY_SD_HFQ")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_TINY_SD_HFQ))
    }

    fn skip_missing_tiny_sd(path: &std::path::Path) -> bool {
        if path.exists() {
            false
        } else {
            eprintln!(
                "skip: set HIPFIRE_TINY_SD_HFQ or create {}",
                DEFAULT_TINY_SD_HFQ
            );
            true
        }
    }

    #[tokio::test]
    async fn txt2img_route_returns_png_for_direct_diffusion_hfq_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            ..empty_request()
        };

        let response = post_txt2img(State(state.clone()), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let image = images[0].as_str().unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(image)
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let Json(png_info) = post_png_info(Json(json!({"image": image}))).await;
        assert!(png_info["info"].as_str().unwrap().contains("a cat"));
        assert!(png_info["info"].as_str().unwrap().contains("Steps: 1"));
        assert_eq!(png_info["parameters"]["Prompt"], "a cat");
        assert_eq!(png_info["parameters"]["Negative prompt"], "");
        assert_eq!(png_info["parameters"]["Steps"], "1");
        assert_eq!(png_info["parameters"]["CFG scale"], "1");
        assert_eq!(png_info["parameters"]["Size-1"], 2);
        assert_eq!(png_info["parameters"]["Size-2"], 2);
        assert_eq!(png_info["parameters"]["Schedule type"], "Automatic");
        assert_eq!(body["parameters"]["prompt"], "a cat");
        assert_eq!(
            serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap()["backend"],
            "hipfire-diffusion-hfq"
        );
        let Json(progress) = get_progress(State(state.clone())).await;
        assert_eq!(progress["progress"], 1.0);
        assert_eq!(progress["textinfo"], "complete");
        let current_image = progress["current_image"].as_str().unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(current_image)
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        {
            let cache = state.diffusion_pipelines.lock().await;
            assert_eq!(cache.len(), 1);
            assert!(cache.contains_key(&hfq_path));
        }

        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            ..empty_request()
        };
        let response = post_txt2img(State(state.clone()), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.diffusion_pipelines.lock().await.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(feature = "rocm"))]
    #[tokio::test]
    async fn txt2img_route_rejects_rocm_runtime_when_feature_disabled() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-route-rocm-disabled-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-rocm-disabled.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            rocm_device_id: Some(0),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = response_json(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("built without the rocm feature"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_route_uses_override_settings_sd_model_checkpoint_for_hfq_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-override-model-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-override.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            override_settings: Some(json!({
                "sd_model_checkpoint": hfq_path,
            })),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(info["model"], "tiny-route");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_route_applies_webui_infotext_defaults() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-infotext-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-infotext.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            model: Some(hfq_path.to_string_lossy().into_owned()),
            send_images: Some(true),
            save_images: Some(false),
            infotext: Some(
                "a copied prompt\n\
                 Negative prompt: blur\n\
                 Steps: 1, Sampler: Euler, CFG scale: 1, Seed: 77, Size: 2x2"
                    .to_string(),
            ),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        assert_eq!(body["images"].as_array().unwrap().len(), 1);
        assert_eq!(body["parameters"]["prompt"], "a copied prompt");
        assert_eq!(body["parameters"]["negative_prompt"], "blur");
        assert_eq!(body["parameters"]["steps"], 1);
        assert_eq!(body["parameters"]["sampler_name"], "Euler");
        assert_eq!(body["parameters"]["cfg_scale"], 1.0);
        assert_eq!(body["parameters"]["seed"], 77);
        assert_eq!(body["parameters"]["width"], 2);
        assert_eq!(body["parameters"]["height"], 2);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["seeds"], json!([77]));
        assert_eq!(info["width"], 2);
        assert_eq!(info["height"], 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_route_rejects_unsupported_selectable_script_payloads() {
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let response = post_txt2img(
            State(state.clone()),
            Json(SdGenerationRequest {
                prompt: "a cat".to_string(),
                script_name: Some("X/Y/Z plot".to_string()),
                ..empty_request()
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("script_name"));

        let response = post_txt2img(
            State(state),
            Json(SdGenerationRequest {
                prompt: "a cat".to_string(),
                script_args: Some(json!(["arg"])),
                ..empty_request()
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("script_args"));
    }

    #[tokio::test]
    async fn img2img_route_rejects_unsupported_alwayson_scripts_but_allows_empty_defaults() {
        assert!(sdapi_validate_supported_scripts(&SdGenerationRequest {
            script_args: Some(json!([])),
            alwayson_scripts: Some(json!({})),
            ..empty_request()
        })
        .is_ok());
        assert!(sdapi_validate_supported_scripts(&SdGenerationRequest {
            alwayson_scripts: Some(json!({
                "controlnet": {
                    "args": [{
                        "enabled": false,
                        "input_image": tiny_png_base64(),
                        "module": "canny",
                        "model": "control_v11p_sd15_canny",
                        "weight": 1.0
                    }]
                }
            })),
            ..empty_request()
        })
        .is_ok());
        assert!(sdapi_validate_supported_scripts(&SdGenerationRequest {
            alwayson_scripts: Some(json!({
                "ADetailer": {
                    "args": [
                        false,
                        {
                            "ad_model": "face_yolov8n.pt",
                            "ad_prompt": "portrait",
                            "ad_confidence": 0.3
                        }
                    ]
                }
            })),
            ..empty_request()
        })
        .is_ok());

        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let response = post_img2img(
            State(state),
            Json(SdGenerationRequest {
                prompt: "a cat".to_string(),
                init_images: Some(vec![tiny_png_base64()]),
                alwayson_scripts: Some(json!({
                    "controlnet": {
                        "args": [{"enabled": true}]
                    }
                })),
                ..empty_request()
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("alwayson_scripts"));
    }

    #[tokio::test]
    async fn txt2img_route_saves_png_when_save_images_true_and_send_images_false() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-save-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-save.hfq");
        let output_dir = dir.join("outputs");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(false),
            save_images: Some(true),
            override_settings: Some(json!({
                "outdir_txt2img_samples": output_dir,
            })),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        assert!(body["images"].as_array().unwrap().is_empty());
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["save_images"], true);
        let saved = info["saved_images"].as_array().unwrap();
        assert_eq!(saved.len(), 1);
        let saved_path = PathBuf::from(saved[0].as_str().unwrap());
        assert!(saved_path.is_file());
        let bytes = std::fs::read(saved_path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let text = extract_png_text_chunk(&bytes, "parameters")
            .unwrap()
            .unwrap();
        assert!(text.contains("a cat"));
        assert!(text.contains("Mode: txt2img"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_route_runs_highres_second_pass_for_direct_diffusion_hfq_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-highres-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-highres.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a highres cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(4),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            denoising_strength: Some(1.0),
            enable_hr: Some(true),
            firstphase_width: Some(2),
            firstphase_height: Some(2),
            hr_scale: Some(2.0),
            hr_upscaler: Some("Latent".to_string()),
            hr_second_pass_steps: Some(1),
            hr_checkpoint_name: Some("Use same checkpoint".to_string()),
            hr_prompt: Some("a highres dog".to_string()),
            hr_negative_prompt: Some("blur".to_string()),
            hr_sampler_name: Some("Euler".to_string()),
            ..empty_request()
        };

        let response = post_txt2img(State(state.clone()), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (4, 4));
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "txt2img-hires");
        assert_eq!(info["highres"], true);
        assert_eq!(info["firstpass_width"], 2);
        assert_eq!(info["firstpass_height"], 2);
        assert_eq!(info["width"], 4);
        assert_eq!(info["height"], 4);
        assert_eq!(info["hr_width"], 4);
        assert_eq!(info["hr_height"], 4);
        assert_eq!(info["hr_second_pass_steps"], 1);
        assert_eq!(info["scheduler"], "Euler");
        assert_eq!(info["hr_upscaler"], "Latent");
        assert!(info.get("hr_checkpoint_name").is_none());
        assert_eq!(info["hr_prompt"], "a highres dog");
        assert_eq!(info["hr_negative_prompt"], "blur");
        assert_eq!(info["hr_sampler_name"], "Euler");
        let Json(progress) = get_progress(State(state.clone())).await;
        assert_eq!(progress["progress"], 1.0);
        assert_eq!(progress["state"]["sampling_steps"], 2);
        let text = extract_png_text_chunk(&bytes, "parameters")
            .unwrap()
            .unwrap();
        assert!(text.contains("Size: 4x4"));
        assert!(text.contains("Mode: txt2img"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_highres_route_saves_second_pass_when_send_images_false() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-highres-save-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-highres-save.hfq");
        let output_dir = dir.join("highres-outputs");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a highres saved cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(9),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(false),
            save_images: Some(true),
            denoising_strength: Some(1.0),
            enable_hr: Some(true),
            hr_scale: Some(2.0),
            hr_second_pass_steps: Some(1),
            override_settings: Some(json!({
                "outdir_txt2img_samples": output_dir,
            })),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        assert!(body["images"].as_array().unwrap().is_empty());
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "txt2img-hires");
        assert_eq!(info["save_images"], true);
        let saved = info["saved_images"].as_array().unwrap();
        assert_eq!(saved.len(), 1);
        let saved_path = PathBuf::from(saved[0].as_str().unwrap());
        assert!(saved_path.is_file());
        assert!(saved_path.starts_with(output_dir));
        let bytes = std::fs::read(saved_path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (4, 4));
        let text = extract_png_text_chunk(&bytes, "parameters")
            .unwrap()
            .unwrap();
        assert!(text.contains("a highres saved cat"));
        assert!(text.contains("Size: 4x4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_route_switches_highres_checkpoint_for_second_pass() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-highres-checkpoint-switch-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let first_hfq_path = dir.join("tiny-route-diffusion-highres-first.hfq");
        let second_hfq_path = dir.join("tiny-route-diffusion-highres-second.hfq");
        write_tiny_diffusion_hfq(&first_hfq_path);
        write_tiny_diffusion_hfq(&second_hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let checkpoint = second_hfq_path.to_string_lossy().into_owned();
        let body = SdGenerationRequest {
            prompt: "a highres cat".to_string(),
            model: Some(first_hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            denoising_strength: Some(1.0),
            enable_hr: Some(true),
            hr_scale: Some(2.0),
            hr_second_pass_steps: Some(1),
            hr_checkpoint_name: Some(checkpoint.clone()),
            ..empty_request()
        };

        let response = post_txt2img(State(state.clone()), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        assert_eq!(body["images"].as_array().unwrap().len(), 1);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "txt2img-hires");
        assert_eq!(info["highres"], true);
        assert_eq!(info["hr_checkpoint_name"], checkpoint);
        assert_eq!(info["width"], 4);
        assert_eq!(info["height"], 4);
        {
            let cache = state.diffusion_pipelines.lock().await;
            assert_eq!(cache.len(), 2);
            assert!(cache.contains_key(&first_hfq_path));
            assert!(cache.contains_key(&second_hfq_path));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_route_rejects_unresolved_highres_checkpoint_switch() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-highres-checkpoint-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-highres-checkpoint.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a highres cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            enable_hr: Some(true),
            hr_checkpoint_name: Some("different-model".to_string()),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("could not be resolved"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_returns_png_for_direct_diffusion_hfq_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let init_image = tiny_png_base64();
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![init_image]),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["denoise_steps"], 1);
        assert_eq!(body["parameters"]["init_images"], Value::Null);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_preserves_init_images_when_requested() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-include-init-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-include-init.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let init_image = tiny_png_base64();
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(false),
            save_images: Some(false),
            init_images: Some(vec![init_image.clone()]),
            include_init_images: Some(true),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        assert!(body["images"].as_array().unwrap().is_empty());
        assert_eq!(
            body["parameters"]["init_images"][0].as_str().unwrap(),
            init_image
        );
        assert_eq!(body["parameters"]["include_init_images"], true);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_accepts_one_init_image_per_batch_item() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-batch-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-batch.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(21),
            steps: Some(1),
            cfg_scale: Some(1.0),
            batch_size: Some(2),
            n_iter: Some(1),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64(), tiny_png_base64()]),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 2);
        for image in images {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(image.as_str().unwrap())
                .unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["batch_size"], 2);
        assert_eq!(info["width"], 2);
        assert_eq!(info["height"], 2);
        assert_eq!(info["seeds"], json!([21, 22]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_runs_n_iter_as_sequential_batches() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-niter-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-niter.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(30),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            batch_size: Some(1),
            n_iter: Some(3),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64()]),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 3);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["batch_size"], 3);
        assert_eq!(info["seeds"], json!([30, 31, 32]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_applies_mask_for_direct_diffusion_hfq_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-mask-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-mask.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64()]),
            mask: Some(tiny_mask_png_base64(2, 2)),
            inpainting_fill: Some(json!(2)),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["masked"], true);
        assert_eq!(info["inpainting_fill"], 2);
        assert_eq!(info["masked_content"], "latent noise");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_resizes_init_and_mask_to_requested_dimensions() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-resize-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-resize.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64_with_dimensions(1, 1)]),
            mask: Some(tiny_mask_png_base64(1, 1)),
            inpaint_full_res: Some(json!(false)),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["masked"], true);
        assert_eq!(info["width"], 2);
        assert_eq!(info["height"], 2);
        assert_eq!(info["inpainting_fill"], 0);
        assert_eq!(info["masked_content"], "fill");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_accepts_resize_mode_3_latent_upscale() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-latent-upscale-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-latent-upscale.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64_with_dimensions(1, 1)]),
            mask: Some(tiny_mask_png_base64(1, 1)),
            resize_mode: Some(3),
            inpaint_full_res: Some(json!(false)),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 2));
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["masked"], true);
        assert_eq!(info["resize_mode"], "latent");
        assert_eq!(info["latent_resize"], true);
        assert_eq!(info["width"], 2);
        assert_eq!(info["height"], 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_composites_full_res_inpaint_crop() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-full-res-inpaint-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-full-res-inpaint.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64_with_dimensions(4, 4)]),
            mask: Some(rect_mask_png_base64(4, 4, 1, 1, 2, 2)),
            inpaint_full_res: Some(json!(true)),
            inpaint_full_res_padding: Some(json!(0)),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (4, 4));
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["masked"], true);
        assert_eq!(info["width"], 4);
        assert_eq!(info["height"], 4);
        assert_eq!(info["inpaint_full_res"], true);
        assert_eq!(info["inpaint_full_res_padding"], 0);
        assert_eq!(info["inpaint_full_res_crop"], json!([1, 1, 1, 1]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_saves_png_when_save_images_true() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-save-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-save.hfq");
        let output_dir = dir.join("img2img-outputs");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(true),
            init_images: Some(vec![tiny_png_base64()]),
            override_settings: Some(json!({
                "outdir_img2img_samples": output_dir,
            })),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        assert_eq!(body["images"].as_array().unwrap().len(), 1);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        let saved = info["saved_images"].as_array().unwrap();
        assert_eq!(saved.len(), 1);
        let saved_path = PathBuf::from(saved[0].as_str().unwrap());
        assert!(saved_path.is_file());
        assert!(saved_path.starts_with(output_dir));
        let bytes = std::fs::read(saved_path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn img2img_route_resizes_mask_dimension_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-resize-mask-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-resize-mask.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64()]),
            mask: Some(tiny_mask_png_base64(1, 1)),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["masked"], true);
        assert_eq!(info["width"], 2);
        assert_eq!(info["height"], 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn img2img_init_images_must_share_dimensions() {
        let images = vec![tiny_png_base64(), tiny_png_base64_with_dimensions(1, 1)];

        let error = decode_sd_init_images(&images).unwrap_err();

        assert!(error
            .to_string()
            .contains("dimensions 1x1 do not match first init image 2x2"));
    }

    #[tokio::test]
    #[ignore = "real Tiny-SD route smoke; run in release mode under an external timeout"]
    async fn txt2img_route_returns_png_for_real_tiny_sd_hfq_model() {
        let hfq_path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&hfq_path) {
            return;
        }
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a red robot".to_string(),
            negative_prompt: String::new(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(123),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(64),
            height: Some(64),
            send_images: Some(true),
            save_images: Some(false),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(info["width"], 64);
        assert_eq!(info["height"], 64);
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[ignore = "real Tiny-SD ROCm txt2img route smoke; run in release mode under an external timeout"]
    async fn txt2img_route_returns_rocm_runtime_for_real_tiny_sd_hfq_model() {
        let hfq_path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&hfq_path) {
            return;
        }
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a red robot".to_string(),
            negative_prompt: String::new(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(123),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(64),
            height: Some(64),
            send_images: Some(true),
            save_images: Some(false),
            hipfire_rocm_device_id: Some(0),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(info["runtime"], "rocm-hybrid-reference");
        assert_eq!(info["width"], 64);
        assert_eq!(info["height"], 64);
    }

    #[tokio::test]
    #[ignore = "real Tiny-SD img2img route smoke; run in release mode under an external timeout"]
    async fn img2img_route_returns_png_for_real_tiny_sd_hfq_model() {
        let hfq_path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&hfq_path) {
            return;
        }
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a red robot".to_string(),
            negative_prompt: String::new(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(123),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(64),
            height: Some(64),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64_with_dimensions(64, 64)]),
            mask: Some(tiny_mask_png_base64(64, 64)),
            denoising_strength: Some(1.0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["masked"], true);
        assert_eq!(info["width"], 64);
        assert_eq!(info["height"], 64);
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[ignore = "real Tiny-SD ROCm img2img route smoke; run in release mode under an external timeout"]
    async fn img2img_route_returns_rocm_runtime_for_real_tiny_sd_hfq_model() {
        let hfq_path = tiny_sd_hfq_path();
        if skip_missing_tiny_sd(&hfq_path) {
            return;
        }
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a red robot".to_string(),
            negative_prompt: String::new(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(123),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(64),
            height: Some(64),
            send_images: Some(true),
            save_images: Some(false),
            init_images: Some(vec![tiny_png_base64_with_dimensions(64, 64)]),
            mask: Some(tiny_mask_png_base64(64, 64)),
            denoising_strength: Some(1.0),
            hipfire_rocm_device_id: Some(0),
            ..empty_request()
        };

        let response = post_img2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 1);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(images[0].as_str().unwrap())
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["backend"], "hipfire-diffusion-hfq");
        assert_eq!(info["mode"], "img2img");
        assert_eq!(info["runtime"], "rocm-hybrid-reference");
        assert_eq!(info["masked"], true);
        assert_eq!(info["width"], 64);
        assert_eq!(info["height"], 64);
    }

    #[tokio::test]
    async fn txt2img_route_returns_batched_pngs_for_diffusion_hfq_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-batch-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(10),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            batch_size: Some(2),
            n_iter: Some(1),
            send_images: Some(true),
            save_images: Some(false),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 2);
        for image in images {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(image.as_str().unwrap())
                .unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["batch_size"], 2);
        assert_eq!(info["seeds"], json!([10, 11]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn txt2img_route_runs_n_iter_as_sequential_batches() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-niter-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-niter.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            model: Some(hfq_path.to_string_lossy().into_owned()),
            seed: Some(20),
            steps: Some(1),
            cfg_scale: Some(1.0),
            width: Some(2),
            height: Some(2),
            batch_size: Some(1),
            n_iter: Some(3),
            send_images: Some(true),
            save_images: Some(false),
            ..empty_request()
        };

        let response = post_txt2img(State(state), Json(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 3);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        assert_eq!(info["batch_size"], 3);
        assert_eq!(info["seeds"], json!([20, 21, 22]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    async fn response_json(response: Response) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[test]
    fn png_text_chunk_round_trips_parameters() {
        let image = tiny_png_base64();
        let bytes = decode_base64_image_payload(&image).unwrap();

        let annotated = insert_png_text_chunk(&bytes, "parameters", "prompt\nSteps: 1").unwrap();

        assert_eq!(&annotated[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(
            extract_png_text_chunk(&annotated, "parameters").unwrap(),
            Some("prompt\nSteps: 1".to_string())
        );
        assert!(image::load_from_memory(&annotated).is_ok());
    }

    #[tokio::test]
    async fn png_info_returns_structured_generation_parameters() {
        let image = tiny_png_base64();
        let bytes = decode_base64_image_payload(&image).unwrap();
        let infotext = "a cat, cinematic\nNegative prompt: blurry, low quality\nSteps: 8, Sampler: Euler a, CFG scale: 4.5, Seed: 123, Size: 512x768, Hires prompt: \"sharp, detailed\"";
        let annotated = insert_png_text_chunk(&bytes, "parameters", infotext).unwrap();
        let encoded = base64::engine::general_purpose::STANDARD.encode(annotated);

        let Json(response) = post_png_info(Json(json!({"image": encoded}))).await;

        assert_eq!(response["info"], infotext);
        assert_eq!(response["parameters"]["Prompt"], "a cat, cinematic");
        assert_eq!(
            response["parameters"]["Negative prompt"],
            "blurry, low quality"
        );
        assert_eq!(response["parameters"]["Steps"], "8");
        assert_eq!(response["parameters"]["Sampler"], "Euler a");
        assert_eq!(response["parameters"]["CFG scale"], "4.5");
        assert_eq!(response["parameters"]["Seed"], "123");
        assert_eq!(response["parameters"]["Size-1"], 512);
        assert_eq!(response["parameters"]["Size-2"], 768);
        assert_eq!(response["parameters"]["Hires prompt"], "sharp, detailed");
        assert_eq!(response["parameters"]["Clip skip"], "1");
        assert_eq!(
            response["parameters"]["Hires checkpoint"],
            "Use same checkpoint"
        );
        assert_eq!(response["items"], json!({}));
    }

    #[tokio::test]
    async fn extra_single_image_returns_png_and_html_info() {
        let response = post_extra_single_image(Json(SdExtrasSingleImageRequest {
            image: tiny_png_base64_with_dimensions(3, 2),
            show_extras_results: Some(true),
        }))
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert!(body["html_info"]
            .as_str()
            .unwrap()
            .contains("no post-processing"));
        let image = body["image"].as_str().unwrap();
        let bytes = decode_base64_image_payload(image).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (3, 2));
    }

    #[tokio::test]
    async fn extra_batch_images_returns_pngs_and_honors_send_flag() {
        let response = post_extra_batch_images(Json(SdExtrasBatchImagesRequest {
            image_list: vec![
                SdExtrasFileData {
                    data: tiny_png_base64_with_dimensions(2, 2),
                    name: Some("a.png".to_string()),
                },
                SdExtrasFileData {
                    data: tiny_png_base64_with_dimensions(1, 3),
                    name: Some("b.png".to_string()),
                },
            ],
            show_extras_results: Some(true),
        }))
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        let images = body["images"].as_array().unwrap();
        assert_eq!(images.len(), 2);
        for image in images {
            let bytes = decode_base64_image_payload(image.as_str().unwrap()).unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }

        let hidden = post_extra_batch_images(Json(SdExtrasBatchImagesRequest {
            image_list: vec![SdExtrasFileData {
                data: tiny_png_base64(),
                name: None,
            }],
            show_extras_results: Some(false),
        }))
        .await;
        assert_eq!(hidden.status(), StatusCode::OK);
        let body = response_json(hidden).await;
        assert_eq!(body["images"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn extra_single_image_rejects_invalid_image() {
        let response = post_extra_single_image(Json(SdExtrasSingleImageRequest {
            image: "not-base64".to_string(),
            show_extras_results: Some(true),
        }))
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn interrogate_returns_caption_for_valid_image() {
        let response = post_interrogate(Json(SdInterrogateRequest {
            image: tiny_png_base64(),
            model: "clip".to_string(),
        }))
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert!(body["caption"].as_str().unwrap().contains("Hipfire clip"));
    }

    #[tokio::test]
    async fn interrogate_accepts_deepdanbooru_model_alias() {
        let response = post_interrogate(Json(SdInterrogateRequest {
            image: tiny_png_base64(),
            model: "deepdanbooru".to_string(),
        }))
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert!(body["caption"]
            .as_str()
            .unwrap()
            .contains("Hipfire deepdanbooru"));
    }

    #[tokio::test]
    async fn interrogate_rejects_invalid_image() {
        let response = post_interrogate(Json(SdInterrogateRequest {
            image: "not-base64".to_string(),
            model: "clip".to_string(),
        }))
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn interrogate_rejects_unknown_model() {
        let response = post_interrogate(Json(SdInterrogateRequest {
            image: tiny_png_base64(),
            model: "unknown".to_string(),
        }))
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unsupported interrogate model"));
    }

    fn tiny_png_base64() -> String {
        tiny_png_base64_with_dimensions(2, 2)
    }

    fn tiny_png_base64_with_dimensions(width: u32, height: u32) -> String {
        use image::ImageEncoder;

        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for idx in 0..(width * height) {
            let red = if idx % 2 == 0 { 255 } else { 64 };
            pixels.extend_from_slice(&[red, 0, 0]);
        }
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&pixels, width, height, image::ColorType::Rgb8.into())
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(png)
    }

    fn tiny_mask_png_base64(width: u32, height: u32) -> String {
        use image::ImageEncoder;

        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for idx in 0..(width * height) {
            let value = if idx % 2 == 0 { 255 } else { 0 };
            pixels.extend_from_slice(&[value, value, value]);
        }
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&pixels, width, height, image::ColorType::Rgb8.into())
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(png)
    }

    fn rect_mask_png_base64(width: u32, height: u32, x1: u32, y1: u32, x2: u32, y2: u32) -> String {
        use image::ImageEncoder;

        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let value = if x >= x1 && x < x2 && y >= y1 && y < y2 {
                    255
                } else {
                    0
                };
                pixels.extend_from_slice(&[value, value, value]);
            }
        }
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&pixels, width, height, image::ColorType::Rgb8.into())
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(png)
    }

    #[test]
    fn txt2img_request_maps_prompt_controls_to_chat() {
        let body = SdGenerationRequest {
            prompt: "draw a cyberpunk city".to_string(),
            negative_prompt: "low quality".to_string(),
            model: Some("qwen3.5-9b-oq4".to_string()),
            steps: Some(33),
            temperature: Some(0.2),
            top_p: Some(0.8),
            ..empty_request()
        };

        let chat = sd_request_to_chat_request(&body, None);

        assert_eq!(chat.model.as_deref(), Some("qwen3.5-9b-oq4"));
        assert_eq!(chat.max_tokens, Some(33));
        assert_eq!(chat.temperature, Some(0.2));
        assert_eq!(chat.top_p, Some(0.8));
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(
            chat.messages[0].content,
            Some(Value::String(
                "draw a cyberpunk city\n\nNegative prompt: low quality".to_string()
            ))
        );
    }

    #[test]
    fn sd_requested_model_uses_override_checkpoint_and_prefers_explicit_model() {
        let body = SdGenerationRequest {
            prompt: "draw a city".to_string(),
            override_settings: Some(json!({
                "sd_model_checkpoint": "override-diffusion.hfq",
            })),
            ..empty_request()
        };

        assert_eq!(
            sd_requested_model(&body).as_deref(),
            Some("override-diffusion.hfq")
        );
        assert_eq!(
            sd_request_to_chat_request(&body, None).model.as_deref(),
            Some("override-diffusion.hfq")
        );

        let explicit = SdGenerationRequest {
            model: Some("explicit-model.hfq".to_string()),
            ..body
        };

        assert_eq!(
            sd_requested_model(&explicit).as_deref(),
            Some("explicit-model.hfq")
        );
    }

    #[test]
    fn sd_request_generation_runtime_options_select_cpu_or_rocm() {
        let empty_options = std::collections::HashMap::new();
        let default = SdGenerationRequest { ..empty_request() };
        assert_eq!(
            sd_request_generation_runtime_options(&default, &empty_options),
            DiffusionGenerationRuntimeOptions::cpu_reference()
        );

        let direct = SdGenerationRequest {
            rocm_device_id: Some(2),
            ..empty_request()
        };
        assert_eq!(
            sd_request_generation_runtime_options(&direct, &empty_options),
            DiffusionGenerationRuntimeOptions::rocm_hybrid(2)
        );

        let namespaced = SdGenerationRequest {
            hipfire_rocm_device_id: Some(3),
            ..empty_request()
        };
        assert_eq!(
            sd_request_generation_runtime_options(&namespaced, &empty_options),
            DiffusionGenerationRuntimeOptions::rocm_hybrid(3)
        );

        let override_settings = SdGenerationRequest {
            override_settings: Some(json!({
                "hipfire_rocm_device_id": 4,
            })),
            ..empty_request()
        };
        assert_eq!(
            sd_request_generation_runtime_options(&override_settings, &empty_options),
            DiffusionGenerationRuntimeOptions::rocm_hybrid(4)
        );

        let direct_wins = SdGenerationRequest {
            rocm_device_id: Some(5),
            override_settings: Some(json!({
                "hipfire_rocm_device_id": 6,
            })),
            ..empty_request()
        };
        assert_eq!(
            sd_request_generation_runtime_options(&direct_wins, &empty_options),
            DiffusionGenerationRuntimeOptions::rocm_hybrid(5)
        );

        let stored_request = SdGenerationRequest { ..empty_request() };
        let mut stored_options = std::collections::HashMap::new();
        stored_options.insert("hipfire_rocm_device_id".to_string(), json!(7));
        assert_eq!(
            sd_request_generation_runtime_options(&stored_request, &stored_options),
            DiffusionGenerationRuntimeOptions::rocm_hybrid(7)
        );
    }

    #[test]
    fn diffusion_summary_matches_sdapi_model_identifiers() {
        let summary = hipfire_diffusion::DiffusionModelSummary {
            path: PathBuf::from("/tmp/hipfire-models/tiny-route-diffusion.hfq"),
            title: "tiny-route:StableDiffusionPipeline".to_string(),
            model_name: "tiny-route".to_string(),
            pipeline_class: "StableDiffusionPipeline".to_string(),
            max_batch: 2,
            weight_format: "source".to_string(),
        };

        assert!(diffusion_summary_matches_candidate(
            &summary,
            "tiny-route:StableDiffusionPipeline"
        ));
        assert!(diffusion_summary_matches_candidate(&summary, "tiny-route"));
        assert!(diffusion_summary_matches_candidate(
            &summary,
            "/tmp/hipfire-models/tiny-route-diffusion.hfq"
        ));
        assert!(diffusion_summary_matches_candidate(
            &summary,
            "tiny-route-diffusion.hfq"
        ));
        assert!(!diffusion_summary_matches_candidate(
            &summary,
            "other:StableDiffusionPipeline"
        ));
    }

    #[test]
    fn img2img_request_maps_first_image_to_openai_data_url_part() {
        let body = SdGenerationRequest {
            prompt: "describe this".to_string(),
            ..empty_request()
        };

        let chat = sd_request_to_chat_request(&body, Some("AAAA".to_string()));
        let Some(Value::Array(parts)) = chat.messages[0].content.as_ref() else {
            panic!("expected multipart content");
        };

        assert_eq!(parts[0], json!({"type": "text", "text": "describe this"}));
        assert_eq!(
            parts[1],
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})
        );
    }

    #[tokio::test]
    async fn options_advertise_png_diffusion_and_save_support() {
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());

        let Json(options) = get_options(State(state)).await;

        assert_eq!(options["samples_format"], "png");
        assert_eq!(options["send_images"], true);
        assert_eq!(options["send_seed"], true);
        assert_eq!(options["hipfire_rocm_device_id"], Value::Null);
        assert_eq!(options["hipfire_sdapi_save_images_supported"], true);
        assert_eq!(
            options["outdir_txt2img_samples"],
            "/tmp/hipfire-sdapi/txt2img"
        );
        assert_eq!(
            options["outdir_img2img_samples"],
            "/tmp/hipfire-sdapi/img2img"
        );
    }

    #[tokio::test]
    async fn post_options_updates_sd_model_checkpoint_default_model() {
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());

        let Json(updated) = post_options(
            State(state.clone()),
            Json(json!({"sd_model_checkpoint": "/tmp/model-a.hfq"})),
        )
        .await;

        assert_eq!(updated["sd_model_checkpoint"], "/tmp/model-a.hfq");
        assert_eq!(updated["send_seed"], true);
        assert_eq!(
            state.config.lock().await.default_model.as_deref(),
            Some("/tmp/model-a.hfq")
        );

        let Json(ignored) = post_options(
            State(state.clone()),
            Json(json!({"sd_model_checkpoint": 7})),
        )
        .await;
        assert_eq!(ignored["sd_model_checkpoint"], "/tmp/model-a.hfq");

        let Json(cleared) = post_options(
            State(state.clone()),
            Json(json!({"sd_model_checkpoint": null})),
        )
        .await;
        assert_eq!(cleared["sd_model_checkpoint"], Value::Null);
        assert_eq!(state.config.lock().await.default_model, None);
    }

    #[tokio::test]
    async fn post_options_round_trips_webui_compatibility_values() {
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let Json(initial) = get_options(State(state.clone())).await;
        assert_eq!(initial["send_seed"], true);

        let Json(updated) = post_options(
            State(state.clone()),
            Json(json!({
                "send_seed": false,
                "CLIP_stop_at_last_layers": 2,
                "samples_filename_pattern": "[seed]-[prompt_words]",
                "hipfire_rocm_device_id": 0
            })),
        )
        .await;

        assert_eq!(updated["send_seed"], false);
        assert_eq!(updated["CLIP_stop_at_last_layers"], 2);
        assert_eq!(updated["samples_filename_pattern"], "[seed]-[prompt_words]");
        assert_eq!(updated["hipfire_rocm_device_id"], 0);

        let Json(read_back) = get_options(State(state)).await;
        assert_eq!(read_back["send_seed"], false);
        assert_eq!(read_back["CLIP_stop_at_last_layers"], 2);
        assert_eq!(
            read_back["samples_filename_pattern"],
            "[seed]-[prompt_words]"
        );
        assert_eq!(read_back["hipfire_rocm_device_id"], 0);
    }

    #[test]
    fn stored_sdapi_generation_options_apply_as_request_defaults() {
        let mut stored = std::collections::HashMap::new();
        stored.insert("send_images".to_string(), json!(false));
        stored.insert("save_images".to_string(), json!("true"));
        stored.insert(
            "outdir_txt2img_samples".to_string(),
            json!("/tmp/stored-txt2img"),
        );
        stored.insert(
            "outdir_img2img_samples".to_string(),
            json!("/tmp/stored-img2img"),
        );

        let body = sdapi_apply_stored_generation_defaults(empty_request(), &stored);

        assert_eq!(body.send_images, Some(false));
        assert_eq!(body.save_images, Some(true));
        assert_eq!(
            body.override_settings.as_ref().unwrap()["outdir_txt2img_samples"],
            "/tmp/stored-txt2img"
        );

        let explicit = sdapi_apply_stored_generation_defaults(
            SdGenerationRequest {
                send_images: Some(true),
                save_images: Some(false),
                override_settings: Some(json!({
                    "outdir_txt2img_samples": "/tmp/request-txt2img"
                })),
                ..empty_request()
            },
            &stored,
        );

        assert_eq!(explicit.send_images, Some(true));
        assert_eq!(explicit.save_images, Some(false));
        assert_eq!(
            explicit.override_settings.unwrap()["outdir_txt2img_samples"],
            "/tmp/request-txt2img"
        );
    }

    #[tokio::test]
    async fn txt2img_route_uses_stored_send_save_and_output_directory_defaults() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-options-generation-defaults-test-{}",
            std::process::id()
        ));
        let outdir = dir.join("txt2img-out");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-options-defaults-diffusion.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());

        let Json(updated) = post_options(
            State(state.clone()),
            Json(json!({
                "send_images": false,
                "save_images": true,
                "outdir_txt2img_samples": outdir,
            })),
        )
        .await;
        assert_eq!(updated["send_images"], false);
        assert_eq!(updated["save_images"], true);

        let response = post_txt2img(
            State(state.clone()),
            Json(SdGenerationRequest {
                prompt: "a stored options cat".to_string(),
                model: Some(hfq_path.to_string_lossy().into_owned()),
                steps: Some(1),
                cfg_scale: Some(1.0),
                width: Some(2),
                height: Some(2),
                ..empty_request()
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;

        assert_eq!(body["images"].as_array().unwrap().len(), 0);
        assert_eq!(body["parameters"]["send_images"], false);
        assert_eq!(body["parameters"]["save_images"], true);
        let info = serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap();
        let saved = info["saved_images"].as_array().unwrap();
        assert_eq!(saved.len(), 1);
        let saved_path = PathBuf::from(saved[0].as_str().unwrap());
        assert!(saved_path.starts_with(&outdir));
        assert!(saved_path.exists());

        let Json(progress) = get_progress(State(state)).await;
        assert_eq!(progress["textinfo"], "complete");
        assert!(progress["current_image"].as_str().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn reload_and_unload_checkpoint_manage_diffusion_pipeline_cache() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-reload-checkpoint-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-reload-diffusion.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let mut cfg = hipfire_config::HipfireConfig::default();
        cfg.default_model = Some(hfq_path.to_string_lossy().into_owned());
        let state = crate::AppState::new(cfg);

        let response = post_reload_checkpoint(State(state.clone())).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["reloaded"], true);
        assert_eq!(body["loaded"], true);
        assert_eq!(
            body["filename"].as_str().unwrap(),
            hfq_path.to_string_lossy()
        );
        assert_eq!(state.diffusion_pipelines.lock().await.len(), 1);

        let Json(unloaded) = post_unload_checkpoint(State(state.clone())).await;
        assert_eq!(unloaded["unloaded"], 1);
        assert_eq!(unloaded["loaded"], false);
        assert!(state.diffusion_pipelines.lock().await.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn reload_checkpoint_without_default_model_reports_not_loaded() {
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());

        let response = post_reload_checkpoint(State(state)).await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["reloaded"], false);
        assert_eq!(body["loaded"], false);
    }

    #[tokio::test]
    async fn samplers_advertise_runtime_supported_karras_aliases() {
        let Json(samplers) = get_samplers().await;
        let aliases = samplers[0]["aliases"].as_array().unwrap();

        assert!(aliases.contains(&json!("DPM++ 2M Karras")));
        assert!(aliases.contains(&json!("Euler Karras")));
        assert!(aliases.contains(&json!("DDIM")));
    }

    #[tokio::test]
    async fn schedulers_advertise_runtime_supported_schedule_modifiers() {
        let Json(schedulers) = get_schedulers().await;
        let names = schedulers
            .as_array()
            .unwrap()
            .iter()
            .map(|scheduler| scheduler["name"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert!(names.contains(&"Automatic"));
        assert!(names.contains(&"Karras"));
    }

    #[tokio::test]
    async fn progress_endpoint_reports_idle_and_active_sdapi_generation() {
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());

        let Json(idle) = get_progress(State(state.clone())).await;
        assert_eq!(idle["progress"], 0.0);
        assert_eq!(idle["state"]["interrupted"], false);
        assert_eq!(idle["current_task"], Value::Null);

        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            force_task_id: Some("task-123".to_string()),
            ..empty_request()
        };
        start_sdapi_progress(&state.sdapi_progress, &body, "txt2img", 4);
        update_sdapi_progress(
            &state.sdapi_progress,
            DiffusionProgress {
                completed_steps: 2,
                total_steps: 4,
                timestep: 10,
                preview_latents: None,
            },
            Some("preview-b64".to_string()),
        )
        .unwrap();

        let Json(active) = get_progress(State(state.clone())).await;
        assert_eq!(active["progress"], 0.5);
        assert_eq!(active["state"]["job"], "txt2img");
        assert_eq!(active["state"]["sampling_step"], 2);
        assert_eq!(active["state"]["sampling_steps"], 4);
        assert_eq!(active["current_task"], "task-123");
        assert_eq!(active["current_image"], "preview-b64");
        assert_eq!(active["textinfo"], "sampling step 2/4");

        finish_sdapi_progress(&state.sdapi_progress, None, Some("image-b64".to_string()));

        let Json(complete) = get_progress(State(state)).await;
        assert_eq!(complete["progress"], 1.0);
        assert_eq!(complete["state"]["sampling_step"], 4);
        assert_eq!(complete["current_image"], "image-b64");
        assert_eq!(complete["textinfo"], "complete");
    }

    #[tokio::test]
    async fn progress_endpoint_reports_live_preview_image() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-progress-preview-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-progress-preview.hfq");
        write_tiny_diffusion_hfq(&hfq_path);
        let pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            force_task_id: Some("task-preview".to_string()),
            ..empty_request()
        };
        let event = DiffusionProgress {
            completed_steps: 1,
            total_steps: 2,
            timestep: 3,
            preview_latents: Some(LatentBatch {
                batch: 1,
                channels: 1,
                height: 2,
                width: 2,
                data: vec![0.0, 0.25, 0.5, 0.75],
            }),
        };
        let preview = sdapi_preview_image_from_progress(
            &pipeline,
            DiffusionGenerationRuntimeOptions::default(),
            &event,
        )
        .unwrap()
        .unwrap();

        start_sdapi_progress(&state.sdapi_progress, &body, "txt2img", 2);
        update_sdapi_progress(&state.sdapi_progress, event, Some(preview.clone())).unwrap();
        let Json(active) = get_progress(State(state)).await;

        assert_eq!(active["progress"], 0.5);
        assert_eq!(active["textinfo"], "sampling step 1/2");
        assert_eq!(active["current_image"], preview);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(active["current_image"].as_str().unwrap())
            .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn interrupt_endpoint_marks_sdapi_generation_for_cancellation() {
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let body = SdGenerationRequest {
            prompt: "a cat".to_string(),
            force_task_id: Some("task-interrupt".to_string()),
            ..empty_request()
        };
        start_sdapi_progress(&state.sdapi_progress, &body, "txt2img", 4);

        let Json(response) = post_interrupt(State(state.clone())).await;
        assert_eq!(response, json!({}));

        let error = update_sdapi_progress(
            &state.sdapi_progress,
            DiffusionProgress {
                completed_steps: 1,
                total_steps: 4,
                timestep: 3,
                preview_latents: None,
            },
            None,
        )
        .unwrap_err();
        assert!(matches!(error, DiffusionError::Interrupted(_)));

        let Json(progress) = get_progress(State(state)).await;
        assert_eq!(progress["state"]["interrupted"], true);
        assert_eq!(progress["state"]["sampling_step"], 1);
        assert_eq!(progress["textinfo"], "interrupted");
    }

    #[test]
    fn recognizes_known_diffusers_pipeline_classes() {
        assert!(is_supported_diffusion_pipeline("StableDiffusionPipeline"));
        assert!(is_supported_diffusion_pipeline("StableDiffusionXLPipeline"));
        assert!(is_supported_diffusion_pipeline("Krea2Pipeline"));
        assert!(!is_supported_diffusion_pipeline("AutoModelForCausalLM"));
    }

    #[test]
    fn checkpoint_discovery_lists_single_file_models() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-checkpoint-discovery-test-{}",
            std::process::id()
        ));
        let nested = dir.join("subdir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&nested).unwrap();
        let direct = dir.join("dream.safetensors");
        let child = nested.join("paint.ckpt");
        std::fs::write(&direct, b"safe").unwrap();
        std::fs::write(&child, b"ckpt").unwrap();
        std::fs::write(dir.join("notes.txt"), b"ignore").unwrap();

        let mut models = Vec::new();
        collect_checkpoint_models_from_root(&dir, &mut models);
        models.sort_by(|a, b| a.model_name.cmp(&b.model_name));

        assert_eq!(
            models
                .iter()
                .map(|model| model.model_name.as_str())
                .collect::<Vec<_>>(),
            vec!["dream", "paint"]
        );
        assert!(models
            .iter()
            .all(|model| model.pipeline_class == "StableDiffusionCheckpoint"));
        assert!(requested_model_is_diffusers_pipeline(Some(
            direct.to_str().unwrap()
        )));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn txt2img_request_maps_batch_fields_to_diffusion_request() {
        let body = SdGenerationRequest {
            prompt: "a small red robot".to_string(),
            negative_prompt: "blur".to_string(),
            seed: Some(41),
            subseed: Some(99),
            subseed_strength: Some(0.35),
            steps: Some(8),
            cfg_scale: Some(6.5),
            width: Some(512),
            height: Some(512),
            batch_size: Some(2),
            n_iter: Some(2),
            scheduler: Some("DPM++ 2M".to_string()),
            ..empty_request()
        };

        let request = sd_request_to_diffusion_batch_request(&body, None, 0).unwrap();

        assert_eq!(request.prompts.len(), 2);
        assert_eq!(request.prompts[0].seed, 41);
        assert_eq!(request.prompts[1].seed, 42);
        assert_eq!(request.prompts[0].subseed, Some(99));
        assert!((request.subseed_strength - 0.35).abs() < 1e-6);
        assert_eq!(request.prompts[0].prompt, "a small red robot");
        assert_eq!(request.prompts[0].negative_prompt, "blur");
        assert_eq!(request.steps, 8);
        assert_eq!(request.cfg_scale, 6.5);
        assert_eq!(request.scheduler, "DPM++ 2M");
    }

    #[test]
    fn txt2img_request_applies_iteration_seed_offset() {
        let body = SdGenerationRequest {
            prompt: "a small red robot".to_string(),
            seed: Some(41),
            batch_size: Some(2),
            n_iter: Some(2),
            ..empty_request()
        };

        let iter_request = sd_request_to_diffusion_batch_request(&body, None, 2).unwrap();

        assert_eq!(iter_request.prompts.len(), 2);
        assert_eq!(iter_request.prompts[0].seed, 43);
        assert_eq!(iter_request.prompts[1].seed, 44);
    }

    #[test]
    fn txt2img_request_uses_sampler_name_when_scheduler_is_absent() {
        let body = SdGenerationRequest {
            prompt: "a small red robot".to_string(),
            sampler_name: Some("Euler".to_string()),
            ..empty_request()
        };

        let request = sd_request_to_diffusion_batch_request(&body, None, 0).unwrap();

        assert_eq!(request.scheduler, "Euler");
    }

    #[test]
    fn txt2img_request_combines_webui_sampler_and_schedule_type() {
        let body = SdGenerationRequest {
            prompt: "a small red robot".to_string(),
            sampler_name: Some("Euler".to_string()),
            scheduler: Some("Karras".to_string()),
            ..empty_request()
        };

        let request = sd_request_to_diffusion_batch_request(&body, None, 0).unwrap();

        assert_eq!(request.scheduler, "Euler Karras");

        let automatic = SdGenerationRequest {
            sampler_name: Some("DDIM".to_string()),
            scheduler: Some("Automatic".to_string()),
            ..empty_request()
        };
        assert_eq!(
            sd_request_to_diffusion_batch_request(&automatic, None, 0)
                .unwrap()
                .scheduler,
            "DDIM"
        );

        let full_scheduler_wins = SdGenerationRequest {
            sampler_name: Some("Euler".to_string()),
            scheduler: Some("DPM++ 2M".to_string()),
            ..empty_request()
        };
        assert_eq!(
            sd_request_to_diffusion_batch_request(&full_scheduler_wins, None, 0)
                .unwrap()
                .scheduler,
            "DPM++ 2M"
        );
    }

    #[test]
    fn txt2img_request_maps_sdxl_size_conditioning_fields() {
        let body = SdGenerationRequest {
            prompt: "a small red robot".to_string(),
            width: Some(768),
            height: Some(512),
            original_width: Some(1024),
            original_height: Some(768),
            target_width: Some(768),
            target_height: Some(512),
            crop_x: Some(4),
            crop_y: Some(8),
            ..empty_request()
        };

        let request = sd_request_to_diffusion_batch_request(&body, None, 0).unwrap();

        assert_eq!(request.width, 768);
        assert_eq!(request.height, 512);
        assert_eq!(request.original_width, Some(1024));
        assert_eq!(request.original_height, Some(768));
        assert_eq!(request.target_width, Some(768));
        assert_eq!(request.target_height, Some(512));
        assert_eq!(request.crop_x, 4);
        assert_eq!(request.crop_y, 8);
    }

    #[test]
    fn txt2img_highres_target_uses_scale_and_resize_fields() {
        let scaled = SdGenerationRequest {
            width: Some(2),
            height: Some(3),
            enable_hr: Some(true),
            hr_scale: Some(2.0),
            ..empty_request()
        };
        assert_eq!(
            sdapi_highres_target_dimensions(&scaled).unwrap(),
            Some((4, 6))
        );

        let resize_x = SdGenerationRequest {
            width: Some(2),
            height: Some(3),
            enable_hr: Some(true),
            hr_resize_x: Some(8),
            ..empty_request()
        };
        assert_eq!(
            sdapi_highres_target_dimensions(&resize_x).unwrap(),
            Some((8, 12))
        );

        let resize_y = SdGenerationRequest {
            width: Some(2),
            height: Some(3),
            enable_hr: Some(true),
            hr_resize_y: Some(9),
            ..empty_request()
        };
        assert_eq!(
            sdapi_highres_target_dimensions(&resize_y).unwrap(),
            Some((6, 9))
        );

        let exact = SdGenerationRequest {
            width: Some(2),
            height: Some(3),
            enable_hr: Some(true),
            hr_resize_x: Some(7),
            hr_resize_y: Some(5),
            ..empty_request()
        };
        assert_eq!(
            sdapi_highres_target_dimensions(&exact).unwrap(),
            Some((7, 5))
        );
    }

    #[test]
    fn txt2img_highres_first_pass_body_applies_firstphase_dimensions() {
        let explicit = SdGenerationRequest {
            width: Some(4),
            height: Some(2),
            enable_hr: Some(true),
            firstphase_width: Some(2),
            firstphase_height: Some(2),
            ..empty_request()
        };
        let first_pass = sdapi_txt2img_first_pass_body(&explicit).unwrap();
        assert_eq!(first_pass.width, Some(2));
        assert_eq!(first_pass.height, Some(2));
        assert_eq!(
            sdapi_highres_target_dimensions(&first_pass).unwrap(),
            Some((4, 4))
        );

        let width_only = SdGenerationRequest {
            width: Some(4),
            height: Some(2),
            enable_hr: Some(true),
            firstphase_width: Some(8),
            ..empty_request()
        };
        let first_pass = sdapi_txt2img_first_pass_body(&width_only).unwrap();
        assert_eq!(first_pass.width, Some(8));
        assert_eq!(first_pass.height, Some(4));

        let disabled = SdGenerationRequest {
            width: Some(4),
            height: Some(2),
            firstphase_width: Some(8),
            firstphase_height: Some(4),
            ..empty_request()
        };
        let first_pass = sdapi_txt2img_first_pass_body(&disabled).unwrap();
        assert_eq!(first_pass.width, Some(4));
        assert_eq!(first_pass.height, Some(2));
    }

    #[test]
    fn txt2img_highres_second_pass_init_images_cover_crop_exact_resize() {
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
        let exact_resize = SdGenerationRequest {
            enable_hr: Some(true),
            hr_resize_x: Some(4),
            hr_resize_y: Some(4),
            ..empty_request()
        };

        let cropped =
            sdapi_highres_second_pass_init_images(&exact_resize, image.clone(), (4, 4)).unwrap();
        assert_eq!(cropped.width, 4);
        assert_eq!(cropped.height, 4);
        assert_eq!(&cropped.data[..12], &[20u8; 12]);
        assert_eq!(&cropped.data[24..36], &[30u8; 12]);

        let width_only = SdGenerationRequest {
            enable_hr: Some(true),
            hr_resize_x: Some(4),
            ..empty_request()
        };
        let unchanged =
            sdapi_highres_second_pass_init_images(&width_only, image.clone(), (4, 8)).unwrap();
        assert_eq!(unchanged, image);
    }

    #[test]
    fn img2img_resize_image_applies_webui_resize_modes() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 4,
            data: vec![
                10, 10, 10, 11, 11, 11, //
                20, 20, 20, 21, 21, 21, //
                30, 30, 30, 31, 31, 31, //
                40, 40, 40, 41, 41, 41, //
            ],
        };

        let stretch = SdGenerationRequest {
            resize_mode: Some(0),
            ..empty_request()
        };
        let stretched = sdapi_img2img_resize_image(&stretch, image.clone(), (4, 4)).unwrap();
        assert_eq!(stretched.width, 4);
        assert_eq!(stretched.height, 4);
        assert_eq!(
            &stretched.data[..12],
            &[10u8, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11]
        );

        let crop = SdGenerationRequest {
            resize_mode: Some(1),
            ..empty_request()
        };
        let cropped = sdapi_img2img_resize_image(&crop, image.clone(), (4, 4)).unwrap();
        assert_eq!(cropped.width, 4);
        assert_eq!(cropped.height, 4);
        assert_eq!(
            &cropped.data[..12],
            &[20u8, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21]
        );

        let fill = SdGenerationRequest {
            resize_mode: Some(2),
            ..empty_request()
        };
        let filled = sdapi_img2img_resize_image(&fill, image.clone(), (4, 4)).unwrap();
        assert_eq!(filled.width, 4);
        assert_eq!(filled.height, 4);
        assert_eq!(
            &filled.data[..12],
            &[10u8, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11]
        );

        let latent_upscale = SdGenerationRequest {
            resize_mode: Some(3),
            ..empty_request()
        };
        let latent = sdapi_img2img_resize_image(&latent_upscale, image.clone(), (4, 4)).unwrap();
        assert_eq!(latent, image);
        assert_eq!(
            sdapi_img2img_diffusion_resize_mode(&latent_upscale),
            DiffusionImg2ImgResizeMode::Latent
        );
    }

    #[test]
    fn img2img_mask_options_invert_mask_pixels() {
        let mask = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 1,
            data: vec![0, 64, 255, 10, 128, 240],
        };

        let disabled = SdGenerationRequest {
            mask_round: Some(json!(false)),
            inpainting_mask_invert: Some(json!(0)),
            ..empty_request()
        };
        let unchanged = sdapi_apply_inpainting_mask_options(&disabled, mask.clone()).unwrap();
        assert_eq!(unchanged.data, mask.data);

        let enabled = SdGenerationRequest {
            mask_round: Some(json!(false)),
            inpainting_mask_invert: Some(json!(1)),
            ..empty_request()
        };
        let inverted = sdapi_apply_inpainting_mask_options(&enabled, mask).unwrap();

        assert_eq!(inverted.width, 2);
        assert_eq!(inverted.height, 1);
        assert_eq!(inverted.data, vec![255, 191, 0, 245, 127, 15]);
        assert!(sdapi_inpainting_mask_invert(&SdGenerationRequest {
            inpainting_mask_invert: Some(json!(true)),
            ..empty_request()
        }));
        assert!(sdapi_inpainting_mask_invert(&SdGenerationRequest {
            inpainting_mask_invert: Some(json!("2")),
            ..empty_request()
        }));
        assert!(!sdapi_inpainting_mask_invert(&SdGenerationRequest {
            inpainting_mask_invert: Some(json!("0")),
            ..empty_request()
        }));
    }

    #[test]
    fn img2img_inpainting_fill_accepts_webui_modes() {
        assert_eq!(
            sdapi_inpainting_fill(&SdGenerationRequest {
                inpainting_fill: Some(json!(2)),
                ..empty_request()
            })
            .unwrap(),
            Some(2)
        );
        assert_eq!(
            sdapi_inpainting_fill(&SdGenerationRequest {
                inpainting_fill: Some(json!("3")),
                ..empty_request()
            })
            .unwrap(),
            Some(3)
        );
        assert!(sdapi_inpainting_fill(&SdGenerationRequest {
            inpainting_fill: Some(json!(4)),
            ..empty_request()
        })
        .is_err());
    }

    #[test]
    fn img2img_mask_options_round_mask_pixels_by_default() {
        let mask = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 1,
            data: vec![0, 64, 128, 129, 200, 255],
        };

        let rounded = sdapi_apply_inpainting_mask_options(&empty_request(), mask.clone()).unwrap();
        assert_eq!(rounded.data, vec![0, 0, 0, 255, 255, 255]);

        let disabled = SdGenerationRequest {
            mask_round: Some(json!(false)),
            ..empty_request()
        };
        let continuous = sdapi_apply_inpainting_mask_options(&disabled, mask).unwrap();
        assert_eq!(continuous.data, vec![0, 64, 128, 129, 200, 255]);
    }

    #[test]
    fn img2img_mask_options_blur_mask_pixels() {
        let mask = RgbImageBatch {
            batch: 1,
            width: 3,
            height: 3,
            data: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, //
                0, 0, 0, 255, 255, 255, 0, 0, 0, //
                0, 0, 0, 0, 0, 0, 0, 0, 0, //
            ],
        };

        let horizontal = SdGenerationRequest {
            mask_blur_x: Some(json!(1)),
            mask_blur_y: Some(json!(0)),
            ..empty_request()
        };
        let blurred = sdapi_apply_inpainting_mask_options(&horizontal, mask.clone()).unwrap();
        assert_eq!(&blurred.data[0..9], &[0u8; 9]);
        assert!(blurred.data[9] > 0);
        assert!(blurred.data[12] < 255);
        assert!(blurred.data[15] > 0);
        assert_eq!(&blurred.data[18..27], &[0u8; 9]);

        let shared = SdGenerationRequest {
            mask_blur: Some(json!("1")),
            ..empty_request()
        };
        let blurred = sdapi_apply_inpainting_mask_options(&shared, mask).unwrap();
        assert!(blurred.data[3] > 0);
        assert!(blurred.data[9] > 0);
        assert!(blurred.data[12] < 255);
        assert!(blurred.data[21] > 0);
    }

    #[test]
    fn img2img_image_fill_uses_webui_default_and_respects_original_mode() {
        let init_image = RgbImageBatch {
            batch: 1,
            width: 3,
            height: 1,
            data: vec![255, 0, 0, 0, 255, 0, 0, 0, 255],
        };
        let mask = RgbImageBatch {
            batch: 1,
            width: 3,
            height: 1,
            data: vec![0, 0, 0, 255, 255, 255, 0, 0, 0],
        };

        let (filled, applied) =
            sdapi_apply_inpainting_fill_to_init_image(init_image.clone(), &mask, None).unwrap();

        assert!(applied);
        assert_eq!(&filled.data[0..3], &[255, 0, 0]);
        assert_eq!(&filled.data[6..9], &[0, 0, 255]);
        assert_ne!(&filled.data[3..6], &[0, 255, 0]);

        let (original, applied) =
            sdapi_apply_inpainting_fill_to_init_image(init_image.clone(), &mask, Some(1)).unwrap();
        assert!(!applied);
        assert_eq!(original.data, init_image.data);
    }

    #[test]
    fn sdapi_infotext_defaults_populate_core_generation_fields() {
        let body = SdGenerationRequest {
            infotext: Some(
                "a forest cabin\n\
                 Negative prompt: low quality\n\
                 Steps: 13, Sampler: Euler a, Schedule type: Karras, CFG scale: 6.5, Seed: 42, Size: 640x384, Denoising strength: 0.45"
                    .to_string(),
            ),
            ..empty_request()
        };

        let body = sdapi_apply_infotext_defaults(body);

        assert_eq!(body.prompt, "a forest cabin");
        assert_eq!(body.negative_prompt, "low quality");
        assert_eq!(body.steps, Some(13));
        assert_eq!(body.sampler_name.as_deref(), Some("Euler a"));
        assert_eq!(body.scheduler.as_deref(), Some("Karras"));
        assert_eq!(sdapi_effective_scheduler(&body), "Euler a Karras");
        assert_eq!(body.cfg_scale, Some(6.5));
        assert_eq!(body.seed, Some(42));
        assert_eq!(body.width, Some(640));
        assert_eq!(body.height, Some(384));
        assert_eq!(body.denoising_strength, Some(0.45));
    }

    #[test]
    fn sdapi_infotext_defaults_populate_highres_and_inpaint_fields() {
        let body = SdGenerationRequest {
            infotext: Some(
                "prompt\n\
                 Steps: 20, Sampler: Euler, CFG scale: 7, Seed: 1, Size: 256x256, \
                 Hires upscale: 1.5, Hires resize: 768x512, Hires steps: 8, \
                 Hires upscaler: Latent, Hires checkpoint: Use same checkpoint, \
                 Hires prompt: \"sharp, detailed\", Hires negative prompt: blur, \
                 Hires sampler: DDIM, Hires schedule type: Use same scheduler, \
                 Mask mode: Inpaint not masked, Masked content: latent nothing, \
                 Inpaint area: Only masked, Masked area padding: 24"
                    .to_string(),
            ),
            ..empty_request()
        };

        let body = sdapi_apply_infotext_defaults(body);

        assert_eq!(body.enable_hr, Some(true));
        assert_eq!(body.hr_scale, Some(1.5));
        assert_eq!(body.hr_resize_x, Some(768));
        assert_eq!(body.hr_resize_y, Some(512));
        assert_eq!(body.hr_second_pass_steps, Some(8));
        assert_eq!(body.hr_upscaler.as_deref(), Some("Latent"));
        assert_eq!(
            body.hr_checkpoint_name.as_deref(),
            Some("Use same checkpoint")
        );
        assert_eq!(body.hr_prompt.as_deref(), Some("sharp, detailed"));
        assert_eq!(body.hr_negative_prompt.as_deref(), Some("blur"));
        assert_eq!(body.hr_sampler_name.as_deref(), Some("DDIM"));
        assert_eq!(body.hr_scheduler.as_deref(), Some("Use same scheduler"));
        assert_eq!(body.inpainting_mask_invert, Some(json!(true)));
        assert_eq!(body.inpainting_fill, Some(json!(3)));
        assert_eq!(body.inpaint_full_res, Some(json!(true)));
        assert_eq!(body.inpaint_full_res_padding, Some(json!(24)));
    }

    #[test]
    fn sdapi_infotext_defaults_preserve_explicit_request_fields() {
        let body = SdGenerationRequest {
            prompt: "explicit".to_string(),
            steps: Some(3),
            scheduler: Some("DDIM".to_string()),
            width: Some(128),
            infotext: Some(
                "from infotext\nSteps: 30, Sampler: Euler, Schedule type: Karras, CFG scale: 7, Seed: 2, Size: 512x512"
                    .to_string(),
            ),
            ..empty_request()
        };

        let body = sdapi_apply_infotext_defaults(body);

        assert_eq!(body.prompt, "explicit");
        assert_eq!(body.steps, Some(3));
        assert_eq!(body.scheduler.as_deref(), Some("DDIM"));
        assert_eq!(body.sampler_name.as_deref(), Some("Euler"));
        assert_eq!(body.width, Some(128));
        assert_eq!(body.height, Some(512));
        assert_eq!(body.cfg_scale, Some(7.0));
    }

    #[test]
    fn sdapi_infotext_parser_handles_quoted_commas() {
        let parsed = sdapi_parse_infotext(
            "prompt\nSteps: 4, Sampler: Euler, CFG scale: 7, Seed: 9, Size: 64x64, Hires prompt: \"sharp, detailed\"",
        )
        .unwrap();

        assert_eq!(parsed.prompt, "prompt");
        assert_eq!(
            sdapi_infotext_string(&parsed, "Hires prompt").as_deref(),
            Some("sharp, detailed")
        );
    }

    #[test]
    fn txt2img_highres_second_pass_body_applies_prompt_and_sampler_overrides() {
        let body = SdGenerationRequest {
            prompt: "base prompt".to_string(),
            negative_prompt: "base negative".to_string(),
            steps: Some(4),
            scheduler: Some("DPM++ 2M".to_string()),
            sampler_name: Some("Euler a".to_string()),
            enable_hr: Some(true),
            hr_second_pass_steps: Some(2),
            hr_prompt: Some("hires prompt".to_string()),
            hr_negative_prompt: Some("hires negative".to_string()),
            hr_sampler_name: Some("Euler".to_string()),
            hr_scheduler: Some("Use same scheduler".to_string()),
            init_images: Some(vec!["ignored".to_string()]),
            mask: Some("ignored".to_string()),
            ..empty_request()
        };

        let second = sdapi_highres_second_pass_body(&body, (8, 6));

        assert_eq!(second.width, Some(8));
        assert_eq!(second.height, Some(6));
        assert_eq!(second.steps, Some(2));
        assert_eq!(second.enable_hr, Some(false));
        assert_eq!(second.init_images, None);
        assert_eq!(second.mask, None);
        assert_eq!(second.prompt, "hires prompt");
        assert_eq!(second.negative_prompt, "hires negative");
        assert_eq!(second.scheduler, None);
        assert_eq!(second.sampler_name.as_deref(), Some("Euler"));
    }

    #[test]
    fn txt2img_highres_second_pass_body_prefers_hr_scheduler_over_hr_sampler() {
        let body = SdGenerationRequest {
            prompt: "base prompt".to_string(),
            scheduler: Some("DPM++ 2M".to_string()),
            enable_hr: Some(true),
            hr_sampler_name: Some("Euler".to_string()),
            hr_scheduler: Some("DDIM".to_string()),
            ..empty_request()
        };

        let second = sdapi_highres_second_pass_body(&body, (8, 6));
        let request = sd_request_to_diffusion_batch_request(&second, None, 0).unwrap();

        assert_eq!(second.scheduler.as_deref(), Some("DDIM"));
        assert_eq!(second.sampler_name.as_deref(), Some("Euler"));
        assert_eq!(request.scheduler, "DDIM");
    }

    #[test]
    fn txt2img_highres_second_pass_body_combines_hr_sampler_and_karras_schedule() {
        let body = SdGenerationRequest {
            prompt: "base prompt".to_string(),
            sampler_name: Some("DPM++ 2M".to_string()),
            scheduler: Some("Karras".to_string()),
            enable_hr: Some(true),
            hr_sampler_name: Some("Euler".to_string()),
            hr_scheduler: Some("Use same scheduler".to_string()),
            ..empty_request()
        };

        let second = sdapi_highres_second_pass_body(&body, (8, 6));
        let request = sd_request_to_diffusion_batch_request(&second, None, 0).unwrap();

        assert_eq!(second.sampler_name.as_deref(), Some("Euler"));
        assert_eq!(second.scheduler.as_deref(), Some("Karras"));
        assert_eq!(request.scheduler, "Euler Karras");
    }

    fn empty_request() -> SdGenerationRequest {
        SdGenerationRequest {
            prompt: String::new(),
            negative_prompt: String::new(),
            model: None,
            sampler_name: None,
            sampler_index: None,
            scheduler: None,
            steps: None,
            cfg_scale: None,
            seed: None,
            subseed: None,
            subseed_strength: None,
            width: None,
            height: None,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            crop_x: None,
            crop_y: None,
            batch_size: None,
            n_iter: None,
            send_images: None,
            save_images: None,
            force_task_id: None,
            infotext: None,
            init_images: None,
            mask: None,
            mask_blur: None,
            mask_blur_x: None,
            mask_blur_y: None,
            mask_round: None,
            inpainting_mask_invert: None,
            inpainting_fill: None,
            inpaint_full_res: None,
            inpaint_full_res_padding: None,
            resize_mode: None,
            include_init_images: None,
            denoising_strength: None,
            enable_hr: None,
            firstphase_width: None,
            firstphase_height: None,
            hr_scale: None,
            hr_upscaler: None,
            hr_resize_x: None,
            hr_resize_y: None,
            hr_second_pass_steps: None,
            hr_checkpoint_name: None,
            hr_sampler_name: None,
            hr_scheduler: None,
            hr_prompt: None,
            hr_negative_prompt: None,
            rocm_device_id: None,
            hipfire_rocm_device_id: None,
            override_settings: None,
            script_name: None,
            script_args: None,
            alwayson_scripts: None,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            max_tokens: None,
            stop: None,
        }
    }

    fn write_tiny_diffusion_hfq(path: &Path) {
        let metadata = tiny_diffusion_metadata();
        write_hfqm_package_mem(
            path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tiny_diffusion_tensors(),
        )
        .unwrap();
    }

    fn tiny_diffusion_metadata() -> DiffusionHfqMetadata {
        let mut components = BTreeMap::new();
        components.insert(
            "text_encoder".into(),
            DiffusionComponentMetadata {
                class_name: Some("CLIPTextModel".into()),
                config_entry: Some("text_encoder/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        components.insert(
            "unet".into(),
            DiffusionComponentMetadata {
                class_name: Some("UNet2DConditionModel".into()),
                config_entry: Some("unet/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        components.insert(
            "vae".into(),
            DiffusionComponentMetadata {
                class_name: Some("AutoencoderKL".into()),
                config_entry: Some("vae/config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        components.insert(
            "scheduler".into(),
            DiffusionComponentMetadata {
                class_name: Some("EulerDiscreteScheduler".into()),
                config_entry: Some("scheduler/scheduler_config.json".into()),
                weight_entries: Vec::new(),
                tensor_roles: Vec::new(),
            },
        );
        DiffusionHfqMetadata {
            artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
            schema_version: DIFFUSION_SCHEMA_VERSION,
            pipeline: DiffusionPipelineMetadata {
                class_name: "StableDiffusionPipeline".into(),
                source: "/tmp/tiny-route".into(),
                model_name: "tiny-route".into(),
                latent_channels: Some(1),
                latent_height: Some(2),
                latent_width: Some(2),
                supported_widths: vec![2],
                supported_heights: vec![2],
            },
            tokenizer: DiffusionTokenizerMetadata::default(),
            tokenizer_2: None,
            batch: DiffusionBatchMetadata {
                max_batch: 2,
                batched_runtime: true,
            },
            quantization: DiffusionQuantizationMetadata::default(),
            components,
        }
    }

    fn tiny_diffusion_tensors() -> Vec<HfqMemTensor> {
        let identity1 = center_identity_conv(1);
        let mut vae_encoder_conv_in = vec![0.0; 1 * 3 * 3 * 3];
        vae_encoder_conv_in[1 * 3 + 1] = 1.0;
        let mut vae_encoder_conv_out = vec![0.0; 2 * 1 * 3 * 3];
        vae_encoder_conv_out[1 * 3 + 1] = 1.0;
        let down_prefix = "unet/tensors/down_blocks.0.resnets.0";
        let up_prefix = "unet/tensors/up_blocks.0.resnets.0";
        let vae_resnet_prefix = "vae/tensors/decoder.up_blocks.0.resnets.0";
        let vae_encoder_resnet_prefix = "vae/tensors/encoder.down_blocks.0.resnets.0";
        vec![
            bytes_mem_tensor(
                "text_encoder/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"CLIPTextModel","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"max_position_embeddings":77,"vocab_size":4}"#,
            ),
            bytes_mem_tensor(
                "unet/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"UNet2DConditionModel","sample_size":2,"in_channels":1,"out_channels":1,"cross_attention_dim":2,"attention_head_dim":[1],"block_out_channels":[1],"down_block_types":["DownBlock2D"],"up_block_types":["UpBlock2D"],"layers_per_block":1,"norm_num_groups":1,"norm_eps":0.00001,"flip_sin_to_cos":true,"freq_shift":0.0}"#,
            ),
            bytes_mem_tensor(
                "vae/config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"AutoencoderKL","latent_channels":1,"scaling_factor":1.0,"block_out_channels":[1],"down_block_types":["DownEncoderBlock2D"],"up_block_types":["UpDecoderBlock2D"],"norm_num_groups":1,"norm_eps":0.000001}"#,
            ),
            bytes_mem_tensor(
                "scheduler/scheduler_config.json",
                QT_DIFFUSION_JSON,
                br#"{"_class_name":"EulerDiscreteScheduler"}"#,
            ),
            bytes_mem_tensor(
                "tokenizer/vocab.json",
                QT_DIFFUSION_TOKENIZER,
                br#"{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2,"cat</w>":3}"#,
            ),
            bytes_mem_tensor("tokenizer/merges.txt", QT_DIFFUSION_TOKENIZER, b"#version: 0.2\n"),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.embeddings.token_embedding.weight",
                &[4, 2],
                &[0.0, 0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.embeddings.position_embedding.weight",
                &[77, 2],
                &[0.0; 154],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.final_layer_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.final_layer_norm.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.q_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.q_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.k_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.k_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.v_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.v_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.out_proj.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.self_attn.out_proj.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm1.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc1.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc2.weight",
                &[2, 2],
                &[0.0; 4],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.mlp.fc2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm2.weight",
                &[2],
                &[1.0, 1.0],
            ),
            f32_mem_tensor(
                "text_encoder/tensors/text_model.encoder.layers.0.layer_norm2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor("unet/tensors/conv_in.weight", &[1, 1, 3, 3], &identity1),
            f32_mem_tensor("unet/tensors/conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_1.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_1.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_2.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor(
                "unet/tensors/time_embedding.linear_2.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{down_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{down_prefix}.conv1.weight"), &[1, 1, 3, 3], &identity1),
            f32_mem_tensor(&format!("{down_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{down_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{down_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{down_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{down_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm1.weight"), &[2], &[1.0, 1.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm1.bias"), &[2], &[0.0, 0.0]),
            f32_mem_tensor(&format!("{up_prefix}.conv1.weight"), &[1, 2, 3, 3], &[0.0; 18]),
            f32_mem_tensor(&format!("{up_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.time_emb_proj.weight"),
                &[1, 2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(&format!("{up_prefix}.time_emb_proj.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{up_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{up_prefix}.conv2.weight"), &[1, 1, 3, 3], &[0.0; 9]),
            f32_mem_tensor(&format!("{up_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{up_prefix}.conv_shortcut.weight"),
                &[1, 2, 1, 1],
                &[1.0, 0.0],
            ),
            f32_mem_tensor(&format!("{up_prefix}.conv_shortcut.bias"), &[1], &[0.0]),
            f32_mem_tensor("unet/tensors/conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("unet/tensors/conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor("unet/tensors/conv_out.weight", &[1, 1, 3, 3], &identity1),
            f32_mem_tensor("unet/tensors/conv_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_in.weight",
                &[1, 3, 3, 3],
                &vae_encoder_conv_in,
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm1.weight"),
                &[1],
                &[1.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm1.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &identity1,
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv1.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm2.weight"),
                &[1],
                &[1.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.norm2.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(
                &format!("{vae_encoder_resnet_prefix}.conv2.bias"),
                &[1],
                &[0.0],
            ),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("vae/tensors/encoder.conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_out.weight",
                &[2, 1, 3, 3],
                &vae_encoder_conv_out,
            ),
            f32_mem_tensor(
                "vae/tensors/encoder.conv_out.bias",
                &[2],
                &[0.0, 0.0],
            ),
            f32_mem_tensor(
                "vae/tensors/quant_conv.weight",
                &[2, 2, 1, 1],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            f32_mem_tensor("vae/tensors/quant_conv.bias", &[2], &[0.0, 0.0]),
            f32_mem_tensor("vae/tensors/post_quant_conv.weight", &[1, 1, 1, 1], &[1.0]),
            f32_mem_tensor("vae/tensors/post_quant_conv.bias", &[1], &[0.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_in.weight", &[1, 1, 3, 3], &identity1),
            f32_mem_tensor("vae/tensors/decoder.conv_in.bias", &[1], &[0.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm1.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm1.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_resnet_prefix}.conv1.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.conv1.bias"), &[1], &[0.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm2.weight"), &[1], &[1.0]),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.norm2.bias"), &[1], &[0.0]),
            f32_mem_tensor(
                &format!("{vae_resnet_prefix}.conv2.weight"),
                &[1, 1, 3, 3],
                &[0.0; 9],
            ),
            f32_mem_tensor(&format!("{vae_resnet_prefix}.conv2.bias"), &[1], &[0.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.weight", &[1], &[1.0]),
            f32_mem_tensor("vae/tensors/decoder.conv_norm_out.bias", &[1], &[0.0]),
            f32_mem_tensor(
                "vae/tensors/decoder.conv_out.weight",
                &[3, 1, 3, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            f32_mem_tensor("vae/tensors/decoder.conv_out.bias", &[3], &[0.0, 0.0, 0.0]),
        ]
    }

    fn f32_mem_tensor(name: &str, shape: &[u32], data: &[f32]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_DIFFUSION_TENSOR_F32,
            shape: shape.to_vec(),
            group_size: 0,
            data: data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        }
    }

    fn bytes_mem_tensor(name: &str, quant_type: u8, data: &[u8]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type,
            shape: vec![data.len() as u32],
            group_size: 0,
            data: data.to_vec(),
        }
    }

    fn center_identity_conv(channels: usize) -> Vec<f32> {
        let mut data = vec![0.0; channels * channels * 3 * 3];
        for channel in 0..channels {
            data[(((channel * channels + channel) * 3 + 1) * 3) + 1] = 1.0;
        }
        data
    }
}
