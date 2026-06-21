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
        .route("/v1/responses", post(routes::responses::post_responses));
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
