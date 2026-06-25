use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use base64::Engine;
use hipfire_config::models_dir;
use hipfire_diffusion::{
    inspect_hfq, inspect_hfq_with_runtime_support, DiffusionBatchOutput, DiffusionBatchRequest,
    DiffusionError, DiffusionGenerationRuntimeOptions, DiffusionHfqInspection,
    DiffusionImg2ImgRequest, DiffusionPipeline, DiffusionProgress, DiffusionPrompt, RgbImageBatch,
};
use image::ImageEncoder;
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
    pub include_init_images: Option<bool>,
    pub denoising_strength: Option<f64>,
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
    execute_sd_generation(state, body, None).await
}

pub async fn post_img2img(
    State(state): State<SharedState>,
    Json(body): Json<SdGenerationRequest>,
) -> Response {
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
                parameters: body,
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
    let original_send_images = body.send_images.unwrap_or(true);
    let request = match sd_request_to_diffusion_batch_request(&body, None, 0) {
        Ok(request) => request,
        Err(error) => return diffusion_error_response(error),
    };
    let pipeline = match cached_diffusion_pipeline(&state, path).await {
        Ok(pipeline) => pipeline,
        Err(error) => return diffusion_error_response(error),
    };
    let n_iter = sd_request_n_iter(&body);
    let save_images = body.save_images.unwrap_or(false);
    let sdapi_options = state.sdapi_options.lock().await.clone();
    let runtime_options = sd_request_generation_runtime_options(&body, &sdapi_options);
    let progress_state = state.sdapi_progress.clone();
    start_sdapi_progress(
        &progress_state,
        &body,
        "txt2img",
        (request.steps as usize).saturating_mul(n_iter as usize),
    );
    let worker_progress_state = progress_state.clone();
    let worker_body = body.clone();
    let output = match tokio::task::spawn_blocking(move || {
        let mut outputs = Vec::with_capacity(n_iter as usize);
        for iter in 0..n_iter {
            let mut iter_request = sd_request_to_diffusion_batch_request(
                &worker_body,
                None,
                iter.saturating_mul(batch_size_for_body(&worker_body)),
            )?;
            if save_images {
                iter_request.send_images = true;
            }
            let step_offset = iter as usize * iter_request.steps as usize;
            let total_steps = iter_request.steps as usize * n_iter as usize;
            let mut progress = |progress: DiffusionProgress| {
                update_sdapi_progress(
                    &worker_progress_state,
                    DiffusionProgress {
                        completed_steps: step_offset.saturating_add(progress.completed_steps),
                        total_steps,
                        timestep: progress.timestep,
                    },
                )
            };
            outputs.push(pipeline.generate_batch_with_progress_and_runtime_options(
                iter_request,
                runtime_options,
                &mut progress,
            )?);
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
    finish_sdapi_progress(&progress_state, output.as_ref().err());
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
    let original_send_images = body.send_images.unwrap_or(true);
    let init_image = match decode_sd_init_images(&images_base64) {
        Ok(image) => image,
        Err(error) => return diffusion_error_response(error),
    };
    let default_dimensions = Some((init_image.width as u32, init_image.height as u32));
    let _first_batch = match sd_request_to_diffusion_batch_request(&body, default_dimensions, 0) {
        Ok(request) => request,
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
    let pipeline = match cached_diffusion_pipeline(&state, path).await {
        Ok(pipeline) => pipeline,
        Err(error) => return diffusion_error_response(error),
    };
    let n_iter = sd_request_n_iter(&body);
    let save_images = body.save_images.unwrap_or(false);
    let denoising_strength = body.denoising_strength.unwrap_or(0.75) as f32;
    let sdapi_options = state.sdapi_options.lock().await.clone();
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
                init_image: init_image.clone(),
                mask: mask.clone(),
                denoising_strength,
            };
            let step_offset = iter as usize * sdapi_img2img_denoise_steps(&worker_body);
            let total_steps = sdapi_img2img_denoise_steps(&worker_body) * n_iter as usize;
            let mut progress = |progress: DiffusionProgress| {
                update_sdapi_progress(
                    &worker_progress_state,
                    DiffusionProgress {
                        completed_steps: step_offset.saturating_add(progress.completed_steps),
                        total_steps,
                        timestep: progress.timestep,
                    },
                )
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
    finish_sdapi_progress(&progress_state, output.as_ref().err());
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
    Json(SdGenerationResponse {
        images,
        parameters: body,
        info: output.info.to_string(),
    })
    .into_response()
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
        body.scheduler
            .as_deref()
            .or(body.sampler_name.as_deref())
            .or(body.sampler_index.as_deref())
            .unwrap_or("DPM++ 2M"),
        body.cfg_scale.unwrap_or(7.0),
        seeds.unwrap_or(-1),
        body.width
            .or_else(|| info
                .get("width")
                .and_then(Value::as_u64)
                .map(|value| value as u32))
            .unwrap_or(512),
        body.height
            .or_else(|| info
                .get("height")
                .and_then(Value::as_u64)
                .map(|value| value as u32))
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
        scheduler: body
            .scheduler
            .clone()
            .or_else(|| body.sampler_name.clone())
            .or_else(|| body.sampler_index.clone())
            .unwrap_or_else(|| "DPM++ 2M".to_string()),
        subseed_strength: body.subseed_strength.unwrap_or(0.0) as f32,
        send_images: body.send_images.unwrap_or(true),
        save_images: body.save_images.unwrap_or(false),
    })
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

fn value_to_i32(value: &Value) -> Option<i32> {
    value.as_i64().and_then(|value| i32::try_from(value).ok())
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

fn update_sdapi_progress(
    progress_state: &Arc<std::sync::Mutex<SdapiProgressState>>,
    event: DiffusionProgress,
) -> Result<(), DiffusionError> {
    let mut progress = progress_state
        .lock()
        .map_err(|error| DiffusionError::Io(format!("SDAPI progress lock poisoned: {error}")))?;
    progress.sampling_step = event.completed_steps;
    progress.sampling_steps = event.total_steps;
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
    Json(json!({
        "info": info,
        "items": {},
        "parameters": {},
    }))
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
                "message": "this stable-diffusion-webui endpoint requires an image-processing backend; hipfire serve currently exposes text-generation compatibility only",
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
        DIFFUSION_ARTIFACT_KIND, DIFFUSION_SCHEMA_VERSION, HFQ_ARCH_DIFFUSION, QT_DIFFUSION_JSON,
        QT_DIFFUSION_TENSOR_F32, QT_DIFFUSION_TOKENIZER,
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
        assert_eq!(body["parameters"]["prompt"], "a cat");
        assert_eq!(
            serde_json::from_str::<Value>(body["info"].as_str().unwrap()).unwrap()["backend"],
            "hipfire-diffusion-hfq"
        );
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
    async fn img2img_route_rejects_mask_dimension_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sdapi-diffusion-img2img-bad-mask-route-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("tiny-route-diffusion-img2img-bad-mask.hfq");
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
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mask dimensions 1x1 do not match init image 2x2"));
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
            },
        )
        .unwrap();

        let Json(active) = get_progress(State(state)).await;
        assert_eq!(active["progress"], 0.5);
        assert_eq!(active["state"]["job"], "txt2img");
        assert_eq!(active["state"]["sampling_step"], 2);
        assert_eq!(active["state"]["sampling_steps"], 4);
        assert_eq!(active["current_task"], "task-123");
        assert_eq!(active["textinfo"], "sampling step 2/4");
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
            },
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
            include_init_images: None,
            denoising_strength: None,
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
