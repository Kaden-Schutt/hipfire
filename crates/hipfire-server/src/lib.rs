pub mod admin_ui;
pub mod auth;
pub mod model;
pub mod routes;
pub mod scheduler;
pub mod state;
pub mod telemetry;

pub use state::{AppState, SharedState};

use axum::{
    body::Body,
    http::{HeaderValue, Method, Request},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Router,
};
use hipfire_config::{HipfireConfig, LoadedConfig};
use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

/// Build a CORS layer from the configured allowlist.
///
/// - empty list -> `None` (no CORS layer; same-origin requests only)
/// - `["*"]`    -> allow any origin
/// - otherwise  -> explicit origin allowlist
fn cors_layer(allowed_origins: &[String]) -> Option<CorsLayer> {
    if allowed_origins.is_empty() {
        return None;
    }
    let base = CorsLayer::new().allow_methods(Any).allow_headers(Any);
    if allowed_origins.iter().any(|origin| origin == "*") {
        return Some(base.allow_origin(Any));
    }
    let origins: Vec<HeaderValue> = allowed_origins
        .iter()
        .filter_map(|origin| origin.parse::<HeaderValue>().ok())
        .collect();
    Some(base.allow_origin(AllowOrigin::list(origins)))
}

pub fn build_router(state: SharedState, cors_allowed_origins: &[String]) -> Router {
    // Gated admin data endpoints: require a valid session cookie or the local
    // bearer secret (see `auth::admin_gate`). The `/admin` shell and the
    // login/logout endpoints below stay ungated so the page can load and the
    // user can authenticate.
    let admin_data = Router::new()
        .route(
            "/admin/config/schema",
            get(routes::admin::get_config_schema),
        )
        .route(
            "/admin/config/resolved",
            get(routes::admin::get_resolved_config),
        )
        .route(
            "/admin/diagnostics",
            get(routes::admin::get_admin_diagnostics),
        )
        .route("/admin/logs", get(routes::admin::get_admin_logs))
        .route("/admin/stats", get(routes::admin::get_admin_stats))
        .route(
            "/admin/models/registry",
            get(routes::models::get_model_registry),
        )
        .route(
            "/admin/training/runs",
            get(routes::training::list_training_runs_route),
        )
        .route(
            "/admin/training/runs/{id}",
            get(routes::training::get_training_run),
        )
        .route(
            "/admin/training/runs/{id}/events",
            get(routes::training::get_training_run_events),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::admin_gate,
        ));

    let router = Router::new()
        .route("/", get(routes::chat_ui::get_chat_index))
        .route("/chat", get(routes::chat_ui::get_chat_index))
        .route("/chat/", get(routes::chat_ui::get_chat_index))
        .route("/chat/{*path}", get(routes::chat_ui::get_chat_asset))
        .route("/health", get(routes::health::get_health))
        .route("/admin", get(routes::admin::get_admin_index))
        .route("/admin/", get(routes::admin::get_admin_index))
        .route("/admin/login", post(auth::login))
        .route("/admin/logout", post(auth::logout))
        // New Leptos console (transitional path; shell + assets are not secret,
        // the data endpoints it calls stay gated).
        .route("/admin/ui", get(admin_ui::index))
        .route("/admin/ui/", get(admin_ui::index))
        .route("/admin/ui/{*path}", get(admin_ui::asset))
        .merge(admin_data)
        .route("/v1/models", get(routes::models::get_models))
        .route(
            "/v1/files",
            get(routes::files::list_files).post(routes::files::create_file),
        )
        .route(
            "/v1/files/{id}",
            get(routes::files::get_file).delete(routes::files::delete_file),
        )
        .route(
            "/v1/files/{id}/content",
            get(routes::files::get_file_content),
        )
        .route(
            "/v1/batches",
            get(routes::batches::list_batches).post(routes::batches::create_batch),
        )
        .route("/v1/batches/{id}", get(routes::batches::get_batch))
        .route(
            "/v1/batches/{id}/cancel",
            post(routes::batches::cancel_batch),
        )
        .route(
            "/v1/chat/completions",
            post(routes::chat::post_chat_completions),
        )
        .route("/v1/responses", post(routes::responses::post_responses))
        .route("/sdapi/v1/txt2img", post(routes::sdapi::post_txt2img))
        .route("/sdapi/v1/img2img", post(routes::sdapi::post_img2img))
        .route(
            "/sdapi/v1/extra-single-image",
            post(routes::sdapi::post_extra_single_image),
        )
        .route(
            "/sdapi/v1/extra-batch-images",
            post(routes::sdapi::post_extra_batch_images),
        )
        .route("/sdapi/v1/png-info", post(routes::sdapi::post_png_info))
        .route("/sdapi/v1/progress", get(routes::sdapi::get_progress))
        .route(
            "/sdapi/v1/interrogate",
            post(routes::sdapi::post_interrogate),
        )
        .route("/sdapi/v1/interrupt", post(routes::sdapi::post_interrupt))
        .route("/sdapi/v1/skip", post(routes::sdapi::post_interrupt))
        .route("/sdapi/v1/options", get(routes::sdapi::get_options))
        .route("/sdapi/v1/options", post(routes::sdapi::post_options))
        .route("/sdapi/v1/memory", get(routes::sdapi::get_memory))
        .route("/sdapi/v1/cmd-flags", get(routes::sdapi::get_cmd_flags))
        .route("/sdapi/v1/samplers", get(routes::sdapi::get_samplers))
        .route("/sdapi/v1/schedulers", get(routes::sdapi::get_schedulers))
        .route("/sdapi/v1/upscalers", get(routes::sdapi::get_upscalers))
        .route(
            "/sdapi/v1/latent-upscale-modes",
            get(routes::sdapi::get_latent_upscale_modes),
        )
        .route("/sdapi/v1/sd-models", get(routes::sdapi::get_sd_models))
        .route("/sdapi/v1/sd-vae", get(routes::sdapi::get_sd_vae))
        .route(
            "/sdapi/v1/hypernetworks",
            get(routes::sdapi::get_hypernetworks),
        )
        .route(
            "/sdapi/v1/face-restorers",
            get(routes::sdapi::get_face_restorers),
        )
        .route(
            "/sdapi/v1/realesrgan-models",
            get(routes::sdapi::get_realesrgan_models),
        )
        .route(
            "/sdapi/v1/prompt-styles",
            get(routes::sdapi::get_prompt_styles),
        )
        .route("/sdapi/v1/loras", get(routes::sdapi::get_loras))
        .route(
            "/sdapi/v1/refresh-loras",
            post(routes::sdapi::post_control_noop),
        )
        .route("/sdapi/v1/embeddings", get(routes::sdapi::get_embeddings))
        .route(
            "/sdapi/v1/refresh-embeddings",
            post(routes::sdapi::post_control_noop),
        )
        .route(
            "/sdapi/v1/refresh-checkpoints",
            post(routes::sdapi::post_control_noop),
        )
        .route(
            "/sdapi/v1/reload-checkpoint",
            post(routes::sdapi::post_reload_checkpoint),
        )
        .route(
            "/sdapi/v1/unload-checkpoint",
            post(routes::sdapi::post_unload_checkpoint),
        )
        .route(
            "/sdapi/v1/refresh-vae",
            post(routes::sdapi::post_control_noop),
        )
        .route(
            "/sdapi/v1/create/embedding",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route(
            "/sdapi/v1/create/hypernetwork",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route(
            "/sdapi/v1/train/embedding",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route(
            "/sdapi/v1/train/hypernetwork",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route("/sdapi/v1/scripts", get(routes::sdapi::get_scripts))
        .route("/sdapi/v1/script-info", get(routes::sdapi::get_script_info))
        .route("/sdapi/v1/extensions", get(routes::sdapi::get_extensions));
    let router = match cors_layer(cors_allowed_origins) {
        Some(cors) => router.layer(cors),
        None => router,
    };
    router
        .layer(middleware::from_fn_with_state(
            state.clone(),
            touch_last_request,
        ))
        .with_state(state)
}

pub async fn serve(config: HipfireConfig) -> anyhow::Result<()> {
    serve_loaded(LoadedConfig::from_config(config)).await
}

pub async fn serve_loaded(config: LoadedConfig) -> anyhow::Result<()> {
    let addr = format!("{}:{}", config.config.host, config.config.port);
    let cors_allowed_origins = config.config.cors_allowed_origins.clone();
    let state = AppState::new_loaded(config);

    prewarm_default_model(&state).await;

    let idle_state = state.clone();
    tokio::spawn(async move {
        idle_unload_loop(idle_state).await;
    });

    let app = build_router(state.clone(), &cors_allowed_origins);
    tracing::info!("hipfire listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;
    Ok(())
}

async fn touch_last_request(
    axum::extract::State(state): axum::extract::State<SharedState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if request_counts_for_idle(request.method(), request.uri().path()) {
        *state.last_request_unix_secs.lock().await = now_secs();
    }
    next.run(request).await
}

fn request_counts_for_idle(method: &Method, path: &str) -> bool {
    matches!(
        (method, path),
        (&Method::POST, "/v1/chat/completions")
            | (&Method::POST, "/v1/responses")
            | (&Method::POST, "/v1/batches")
            | (&Method::POST, "/sdapi/v1/txt2img")
            | (&Method::POST, "/sdapi/v1/img2img")
    )
}

async fn idle_unload_loop(state: SharedState) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        interval.tick().await;
        let idle_timeout = {
            let cfg = state.config.lock().await;
            u64::from(cfg.idle_timeout)
        };
        if idle_timeout == 0 {
            continue;
        }
        let last_request = *state.last_request_unix_secs.lock().await;
        if now_secs().saturating_sub(last_request) < idle_timeout {
            continue;
        }
        if state.loaded_model_path.lock().await.is_none() {
            continue;
        }

        let mut engine = state.engine.lock().await;
        let last_request = *state.last_request_unix_secs.lock().await;
        if now_secs().saturating_sub(last_request) < idle_timeout {
            continue;
        }
        if state.loaded_model_path.lock().await.is_none() {
            continue;
        }
        if let Some(engine) = engine.as_mut() {
            tracing::info!("idle timeout reached; unloading daemon model");
            match engine.unload().await {
                Ok(()) => {
                    *state.loaded_model_path.lock().await = None;
                    *state.loaded_model_cache_capable.lock().await = None;
                    *state.loaded_model_max_seq.lock().await = None;
                }
                Err(e) => {
                    tracing::warn!("idle unload failed: {e}");
                    *engine = match hipfire_daemon_adapter::find_daemon_bin_or_error() {
                        Ok(bin) => match hipfire_daemon_adapter::DaemonEngine::spawn(&bin).await {
                            Ok(new_engine) => new_engine,
                            Err(spawn_err) => {
                                tracing::warn!(
                                    "failed to respawn daemon after idle unload error: {spawn_err}"
                                );
                                *state.loaded_model_path.lock().await = None;
                                *state.loaded_model_cache_capable.lock().await = None;
                                *state.loaded_model_max_seq.lock().await = None;
                                continue;
                            }
                        },
                        Err(bin_err) => {
                            tracing::warn!(
                                "failed to locate daemon after idle unload error: {bin_err}"
                            );
                            *state.loaded_model_path.lock().await = None;
                            *state.loaded_model_cache_capable.lock().await = None;
                            *state.loaded_model_max_seq.lock().await = None;
                            continue;
                        }
                    };
                    *state.loaded_model_path.lock().await = None;
                    *state.loaded_model_cache_capable.lock().await = None;
                    *state.loaded_model_max_seq.lock().await = None;
                }
            }
        }
    }
}

async fn shutdown_signal(state: SharedState) {
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate => {}
    }

    tracing::info!("shutdown signal received; unloading daemon");
    let mut engine = state.engine.lock().await;
    if let Some(mut engine) = engine.take() {
        if let Err(e) = engine.unload().await {
            tracing::warn!("daemon unload during shutdown failed: {e}");
        }
    }
    *state.loaded_model_path.lock().await = None;
    *state.loaded_model_cache_capable.lock().await = None;
    *state.loaded_model_max_seq.lock().await = None;
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_diffusion::{
        DiffusionBatchMetadata, DiffusionHfqMetadata, DiffusionPipelineMetadata,
        DiffusionQuantizationMetadata, DiffusionTokenizerMetadata, DIFFUSION_ARTIFACT_KIND,
        DIFFUSION_SCHEMA_VERSION, HFQ_ARCH_DIFFUSION,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use std::collections::BTreeMap;
    use std::path::Path;

    #[test]
    fn cors_layer_disabled_when_no_origins() {
        assert!(cors_layer(&[]).is_none());
    }

    #[test]
    fn cors_layer_present_for_wildcard_and_allowlist() {
        assert!(cors_layer(&["*".to_string()]).is_some());
        assert!(cors_layer(&["http://localhost:8080".to_string()]).is_some());
    }

    #[test]
    fn idle_touch_ignores_probe_routes() {
        assert!(!request_counts_for_idle(&Method::GET, "/health"));
        assert!(!request_counts_for_idle(&Method::GET, "/v1/models"));
        assert!(request_counts_for_idle(
            &Method::POST,
            "/v1/chat/completions"
        ));
        assert!(request_counts_for_idle(&Method::POST, "/v1/responses"));
        assert!(request_counts_for_idle(&Method::POST, "/v1/batches"));
        assert!(request_counts_for_idle(&Method::POST, "/sdapi/v1/txt2img"));
        assert!(request_counts_for_idle(&Method::POST, "/sdapi/v1/img2img"));
    }

    #[tokio::test]
    async fn prewarm_default_model_routes_diffusion_hfq_to_diffusion_cache() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-prewarm-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("metadata-only-diffusion.hfq");
        write_metadata_only_diffusion_hfq(&hfq_path);

        let mut config = HipfireConfig::default();
        config.default_model = Some(hfq_path.to_string_lossy().into_owned());
        let state = AppState::new(config);

        prewarm_default_model(&state).await;

        assert!(state.engine.lock().await.is_none());
        assert!(state.loaded_model_path.lock().await.is_none());
        assert_eq!(state.diffusion_pipelines.lock().await.len(), 1);
    }

    fn write_metadata_only_diffusion_hfq(path: &Path) {
        let metadata = DiffusionHfqMetadata {
            artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
            schema_version: DIFFUSION_SCHEMA_VERSION,
            pipeline: DiffusionPipelineMetadata {
                class_name: "StableDiffusionPipeline".to_string(),
                source: "/tmp/metadata-only-diffusion".to_string(),
                model_name: "metadata-only-diffusion".to_string(),
                latent_channels: Some(4),
                latent_height: Some(64),
                latent_width: Some(64),
                supported_widths: vec![512],
                supported_heights: vec![512],
            },
            tokenizer: DiffusionTokenizerMetadata::default(),
            tokenizer_2: None,
            batch: DiffusionBatchMetadata {
                max_batch: 1,
                batched_runtime: true,
            },
            quantization: DiffusionQuantizationMetadata {
                weight_format: "metadata-only".to_string(),
                activation_format: "fp16".to_string(),
                tensor_roles_version: 1,
            },
            components: BTreeMap::new(),
        };
        let tensors: Vec<HfqMemTensor> = Vec::new();
        write_hfqm_package_mem(
            path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
    }
}

async fn prewarm_default_model(state: &SharedState) {
    let model = {
        let cfg = state.config.lock().await;
        cfg.default_model.clone()
    };
    let Some(model) = model else {
        return;
    };

    tracing::info!("pre-warming {model}");
    if prewarm_diffusion_model(state, &model).await {
        return;
    }

    let required_max_seq = {
        let cfg = state.config.lock().await;
        cfg.max_seq
    };
    match routes::chat::ensure_model_loaded(state, &model, required_max_seq).await {
        Ok(loaded) => {
            let mut engine_guard = state.engine.lock().await;
            let Some(engine) = engine_guard.as_mut() else {
                tracing::warn!("pre-warm loaded model but daemon engine is unavailable");
                return;
            };
            let req = GenerateTextRequest::from_prompt(
                "warmup".to_string(),
                "Hi",
                GenerationSamplingPolicy::greedy(1),
            )
            .with_worker_key_id(loaded.worker_key_id);
            if let Err(e) = engine.generate(req).await {
                tracing::warn!(
                    "pre-warm generate failed: {e}; first request will continue normally"
                );
                return;
            }
            if let Err(e) = engine.reset().await {
                tracing::warn!(
                    "pre-warm reset failed: {e}; first request will reset before generate"
                );
                return;
            }
            tracing::info!("warm-up complete");
        }
        Err(e) => {
            tracing::warn!("pre-warm load failed: {e}; will load on first request");
        }
    }
}

async fn prewarm_diffusion_model(state: &SharedState, model: &str) -> bool {
    let Some(path) = routes::sdapi::resolve_diffusion_hfq_candidate(model) else {
        return false;
    };

    match routes::sdapi::cached_diffusion_pipeline(state, path.clone()).await {
        Ok(pipeline) => {
            let summary = pipeline.summary();
            tracing::info!(
                model = %summary.model_name,
                pipeline = %summary.pipeline_class,
                path = %path.display(),
                "diffusion warm-up complete"
            );
        }
        Err(e) => {
            tracing::warn!(
                "diffusion pre-warm failed for {}: {e}; first SDAPI request will retry",
                path.display()
            );
        }
    }
    true
}
