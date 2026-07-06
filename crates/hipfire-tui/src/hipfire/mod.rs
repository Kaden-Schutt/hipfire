// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

pub mod chat;
pub mod chat_history;
pub mod config;
pub mod dashboard;
pub mod doctor;
pub mod knobs;
pub mod log_tail;
pub mod model_actions;
pub mod registry;
pub mod serve_ctrl;
pub mod status;
pub mod ui_state;
pub mod writer;

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Resolve the bun CLI entrypoint (`cli/index.ts`) across dev + installed
/// layouts: the launcher-provided `HIPFIRE_CLI_SCRIPT`, then
/// `~/.hipfire/cli/index.ts`, `~/.hipfire/src/cli/index.ts`, then
/// `./cli/index.ts` (repo root). `None` if none exists.
pub fn cli_script_path() -> Option<PathBuf> {
    if let Some(p) = env::var_os("HIPFIRE_CLI_SCRIPT") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    [
        home.join(".hipfire/cli/index.ts"),
        home.join(".hipfire/src/cli/index.ts"),
        cwd.join("cli/index.ts"),
    ]
    .into_iter()
    .find(|p| p.exists())
}

/// Build a `bun <cli/index.ts> ...` Command rooted at the CLI's project dir (so
/// the CLI's cwd-relative lookups resolve in both dev and installed layouts), or
/// `None` if the CLI script can't be found. Callers append their own args.
pub fn cli_command() -> Option<Command> {
    let script = cli_script_path()?;
    let root = script.parent().and_then(Path::parent).map(PathBuf::from);
    let mut cmd = Command::new("bun");
    cmd.arg(&script);
    if let Some(root) = root {
        cmd.current_dir(root);
    }
    Some(cmd)
}

#[derive(Clone, Debug)]
pub struct HipfirePaths {
    pub root: PathBuf,
    pub models: PathBuf,
    pub config: PathBuf,
    pub per_model_config: PathBuf,
    pub serve_pid: PathBuf,
    pub serve_log: PathBuf,
    pub ui_state: PathBuf,
    pub chat_history: PathBuf,
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
            ui_state: root.join("ui_state.json"),
            chat_history: root.join("chat_history.json"),
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

#[cfg(test)]
mod tests {
    use super::*;

    // One test owns HIPFIRE_CLI_SCRIPT so two parallel tests can't race on the
    // process-global env var.
    #[test]
    fn cli_script_path_resolves_env_override() {
        let dir = env::temp_dir().join(format!("hipfire-cliscript-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("index.ts");
        std::fs::write(&script, b"// test entrypoint").unwrap();

        // The installed/bare launch sets HIPFIRE_CLI_SCRIPT; it must win from any
        // cwd (the high-severity Phase 2 bug resolved only ./cli/index.ts, which
        // fails outside a checkout).
        env::set_var("HIPFIRE_CLI_SCRIPT", &script);
        assert_eq!(cli_script_path().as_deref(), Some(script.as_path()));

        // A nonexistent override falls through to the file-candidate search — it
        // must not return the bogus path.
        env::set_var("HIPFIRE_CLI_SCRIPT", "/no/such/hipfire/index.ts");
        assert_ne!(
            cli_script_path().as_deref(),
            Some(Path::new("/no/such/hipfire/index.ts"))
        );

        env::remove_var("HIPFIRE_CLI_SCRIPT");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
