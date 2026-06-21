//! Admin-console authentication for the `/admin` surface.
//!
//! Two ways to authenticate, checked by [`admin_gate`]:
//!
//! 1. **Local bearer secret** — same-box CLI/TUI read `~/.hipfire/admin.secret`
//!    and send `Authorization: Bearer <secret>`. "Can read the file ⇒ admin."
//! 2. **Browser session** — `POST /admin/login` with the configured user +
//!    password verifies an argon2id hash and sets an `HttpOnly; SameSite=Strict`
//!    session cookie; subsequent requests carry the cookie.
//!
//! The `/admin` index shell and `/admin/login` itself are intentionally
//! ungated; the data endpoints under `/admin/*` are wrapped with `admin_gate`.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::json;

use crate::SharedState;

const SESSION_COOKIE: &str = "hipfire_admin_session";
const SESSION_TTL_SECS: u64 = 24 * 60 * 60;

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub user: String,
    pub password: String,
}

/// Middleware gating the `/admin/*` data endpoints.
pub async fn admin_gate(
    State(state): State<SharedState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if is_authorized(&state, request.headers()).await {
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": {
                "message": "admin authentication required",
                "type": "unauthorized"
            }
        })),
    )
        .into_response()
}

async fn is_authorized(state: &SharedState, headers: &HeaderMap) -> bool {
    if let Some(bearer) = bearer_token(headers) {
        if hipfire_config::verify_admin_secret(bearer, &state.admin_secret) {
            return true;
        }
    }
    if let Some(token) = cookie_value(headers, SESSION_COOKIE) {
        let mut sessions = state.admin_sessions.lock().await;
        prune_expired(&mut sessions);
        if let Some(expiry) = sessions.get(&token) {
            return *expiry > now_secs();
        }
    }
    false
}

/// `POST /admin/login` — verify credentials, mint a session cookie.
pub async fn login(State(state): State<SharedState>, Json(body): Json<LoginRequest>) -> Response {
    let admin_user = { state.config.lock().await.admin_user.clone() };
    let password_ok = hipfire_config::read_admin_password_hash()
        .map(|hash| hipfire_config::verify_admin_password(&body.password, &hash))
        .unwrap_or(false);

    if body.user != admin_user || !password_ok {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "message": "invalid admin credentials",
                    "type": "unauthorized"
                }
            })),
        )
            .into_response();
    }

    let token = new_session_token();
    {
        let mut sessions = state.admin_sessions.lock().await;
        prune_expired(&mut sessions);
        sessions.insert(token.clone(), now_secs() + SESSION_TTL_SECS);
    }

    let cookie = format!(
        "{SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/admin; Max-Age={SESSION_TTL_SECS}"
    );
    (
        [(header::SET_COOKIE, cookie)],
        Json(json!({ "status": "ok" })),
    )
        .into_response()
}

/// `POST /admin/logout` — drop the session and clear the cookie.
pub async fn logout(State(state): State<SharedState>, headers: HeaderMap) -> Response {
    if let Some(token) = cookie_value(&headers, SESSION_COOKIE) {
        state.admin_sessions.lock().await.remove(&token);
    }
    let cleared = format!("{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/admin; Max-Age=0");
    (
        [(header::SET_COOKIE, cleared)],
        Json(json!({ "status": "ok" })),
    )
        .into_response()
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::trim)
}

fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    let raw = headers.get(header::COOKIE)?.to_str().ok()?;
    for part in raw.split(';') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix(&format!("{name}=")) {
            return Some(value.to_string());
        }
    }
    None
}

fn prune_expired(sessions: &mut std::collections::HashMap<String, u64>) {
    let now = now_secs();
    sessions.retain(|_, expiry| *expiry > now);
}

fn new_session_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers_with(name: header::HeaderName, value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(name, HeaderValue::from_str(value).unwrap());
        headers
    }

    #[test]
    fn parses_bearer_token() {
        let headers = headers_with(header::AUTHORIZATION, "Bearer secret-123");
        assert_eq!(bearer_token(&headers), Some("secret-123"));
    }

    #[test]
    fn missing_bearer_is_none() {
        assert_eq!(bearer_token(&HeaderMap::new()), None);
        let headers = headers_with(header::AUTHORIZATION, "Basic abc");
        assert_eq!(bearer_token(&headers), None);
    }

    #[test]
    fn extracts_named_cookie_among_many() {
        let headers = headers_with(
            header::COOKIE,
            "theme=dark; hipfire_admin_session=tok-xyz; other=1",
        );
        assert_eq!(
            cookie_value(&headers, SESSION_COOKIE),
            Some("tok-xyz".to_string())
        );
        assert_eq!(cookie_value(&headers, "missing"), None);
    }

    #[test]
    fn prune_drops_only_expired() {
        let mut sessions = std::collections::HashMap::new();
        sessions.insert("live".to_string(), now_secs() + 1000);
        sessions.insert("dead".to_string(), now_secs().saturating_sub(1));
        prune_expired(&mut sessions);
        assert!(sessions.contains_key("live"));
        assert!(!sessions.contains_key("dead"));
    }
}
