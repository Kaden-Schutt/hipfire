//! Serving for the embedded Leptos admin UI (`hipfire-admin-ui/dist`).
//!
//! Gated behind the `admin-ui-embed` feature. When ON, the prebuilt `trunk`
//! dist is baked into the binary and served at `/admin/ui`. When OFF (default),
//! the routes return a short placeholder so `hipfire-server` builds without a
//! prior `trunk build` — this keeps `cargo check --workspace` and CI green.

use axum::{
    extract::Path as AxumPath,
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[cfg(feature = "admin-ui-embed")]
#[derive(rust_embed::RustEmbed)]
#[folder = "../hipfire-admin-ui/dist"]
struct AdminUiDist;

/// `GET /admin/ui` (and `/admin/ui/`) — the SPA shell.
pub async fn index() -> Response {
    serve("index.html")
}

/// `GET /admin/ui/{*path}` — hashed JS/WASM/CSS assets.
pub async fn asset(AxumPath(path): AxumPath<String>) -> Response {
    serve(&path)
}

#[cfg(feature = "admin-ui-embed")]
fn serve(path: &str) -> Response {
    match AdminUiDist::get(path) {
        Some(file) => (
            [(axum::http::header::CONTENT_TYPE, content_type(path))],
            file.data.into_owned(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

#[cfg(not(feature = "admin-ui-embed"))]
fn serve(_path: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        "admin UI not embedded — build crates/hipfire-admin-ui with `trunk build`, \
         then build hipfire with `--features admin-ui-embed`",
    )
        .into_response()
}

#[cfg(feature = "admin-ui-embed")]
fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("js") => "application/javascript",
        Some("wasm") => "application/wasm",
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    }
}
