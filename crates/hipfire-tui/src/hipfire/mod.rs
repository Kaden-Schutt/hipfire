// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

pub mod chat;
pub mod config;
pub mod registry;
pub mod status;
pub mod training;

use std::{env, path::PathBuf};

/// Attach the local admin bearer secret to a request bound for a gated
/// `/admin/*` endpoint, so the TUI authenticates the same way the CLI does.
/// No-op when the secret file is absent (e.g. daemon never started).
pub fn authorize_admin(request: ureq::Request) -> ureq::Request {
    match hipfire_config::read_admin_secret() {
        Some(secret) => request.set("Authorization", &format!("Bearer {secret}")),
        None => request,
    }
}

#[derive(Clone, Debug)]
pub struct HipfirePaths {
    pub root: PathBuf,
    pub models: PathBuf,
    pub config: PathBuf,
    pub host_config: PathBuf,
    pub per_model_config: PathBuf,
    pub serve_pid: PathBuf,
    pub serve_log: PathBuf,
    pub logs: PathBuf,
    pub kernels: PathBuf,
    pub training_runs: PathBuf,
}

impl HipfirePaths {
    pub fn discover() -> Self {
        let home = env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let root = home.join(".hipfire");
        Self {
            models: root.join("models"),
            config: root.join("config.json"),
            host_config: root.join("config.local.json"),
            per_model_config: root.join("per_model_config.json"),
            serve_pid: root.join("serve.pid"),
            serve_log: root.join("serve.log"),
            logs: root.join("logs"),
            kernels: root.join("kernels"),
            training_runs: root.join("training").join("runs"),
            root,
        }
    }
}
