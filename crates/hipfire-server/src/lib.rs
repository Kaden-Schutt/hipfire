pub mod config;
pub mod daemon;
pub mod model;
pub mod routes;
pub mod state;

pub use config::{load_config, HipfireConfig};
pub use state::{AppState, SharedState};

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};

pub fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(routes::health::get_health))
        .route("/v1/models", get(routes::models::get_models))
        .route(
            "/v1/chat/completions",
            post(routes::chat::post_chat_completions),
        )
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

pub async fn serve(config: HipfireConfig) -> anyhow::Result<()> {
    let addr = format!("{}:{}", config.host, config.port);
    let state = AppState::new(config);
    let app = build_router(state);
    tracing::info!("hipfire listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
