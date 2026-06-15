// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

pub mod chat;
pub mod config;
pub mod registry;
pub mod status;

use std::{
    env,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct HipfirePaths {
    pub root: PathBuf,
    pub models: PathBuf,
    pub config: PathBuf,
    pub per_model_config: PathBuf,
    pub serve_pid: PathBuf,
    pub serve_log: PathBuf,
    pub registry_candidates: Vec<PathBuf>,
}

impl HipfirePaths {
    pub fn discover() -> Self {
        let home = env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let root = home.join(".hipfire");
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            models: root.join("models"),
            config: root.join("config.json"),
            per_model_config: root.join("per_model_config.json"),
            serve_pid: root.join("serve.pid"),
            serve_log: root.join("serve.log"),
            registry_candidates: vec![
                cwd.join("cli/registry.json"),
                root.join("cli/registry.json"),
                root.join("src/cli/registry.json"),
            ],
            root,
        }
    }

    pub fn registry_path(&self) -> Option<&Path> {
        self.registry_candidates
            .iter()
            .map(PathBuf::as_path)
            .find(|p| p.exists())
    }
}
