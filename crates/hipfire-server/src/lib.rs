pub mod model;
pub mod routes;
pub mod scheduler;
pub mod state;

pub use state::{AppState, SharedState};

use axum::{
    body::Body,
    http::{Method, Request},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Router,
};
use hipfire_config::{HipfireConfig, LoadedConfig};
use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
use tower_http::cors::{Any, CorsLayer};

pub fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(routes::health::get_health))
        .route("/operator", get(routes::operator::get_operator_index))
        .route("/operator/", get(routes::operator::get_operator_index))
        .route(
            "/operator/config/schema",
            get(routes::operator::get_config_schema),
        )
        .route(
            "/operator/config/resolved",
            get(routes::operator::get_resolved_config),
        )
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
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
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
    let state = AppState::new_loaded(config);

    prewarm_default_model(&state).await;

    let idle_state = state.clone();
    tokio::spawn(async move {
        idle_unload_loop(idle_state).await;
    });

    let app = build_router(state.clone());
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
