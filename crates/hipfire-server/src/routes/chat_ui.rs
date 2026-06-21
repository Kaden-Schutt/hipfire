//! Serving for the embedded Leptos chat UI (`hipfire-chat-ui/dist`).
//!
//! Gated behind the `chat-ui-embed` feature. When ON, the prebuilt `trunk`
//! dist is baked into the binary and served at `/chat`. When OFF (default),
//! the route returns a short placeholder so native cargo checks do not require
//! a wasm toolchain or a prior `trunk build`.

use axum::{
    extract::Path as AxumPath,
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[cfg(feature = "chat-ui-embed")]
#[derive(rust_embed::RustEmbed)]
#[folder = "../hipfire-chat-ui/dist"]
struct ChatUiDist;

pub async fn get_chat_index() -> Response {
    serve("index.html")
}

pub async fn get_chat_asset(AxumPath(path): AxumPath<String>) -> Response {
    serve(&path)
}

#[cfg(feature = "chat-ui-embed")]
fn serve(path: &str) -> Response {
    match ChatUiDist::get(path) {
        Some(file) => (
            [(axum::http::header::CONTENT_TYPE, content_type(path))],
            file.data.into_owned(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

#[cfg(not(feature = "chat-ui-embed"))]
fn serve(_path: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        "chat UI not embedded - build crates/hipfire-chat-ui with `trunk build`, \
         then build hipfire with `--features chat-ui-embed`",
    )
        .into_response()
}

#[cfg(feature = "chat-ui-embed")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "chat-ui-embed"))]
    #[tokio::test]
    async fn chat_index_has_clear_nonembedded_message() {
        let response = get_chat_index().await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(feature = "chat-ui-embed")]
    #[tokio::test]
    async fn chat_index_serves_embedded_shell() {
        let response = get_chat_index().await;
        assert_eq!(response.status(), StatusCode::OK);
    }
}
