// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Locating and shelling out to the hipfire CLI.
//!
//! The CLI (`cli/index.ts`) stays the single source of truth for config
//! validation and persistence; the TUI never writes `config.json` directly.
//! Mutations are applied by spawning `hipfire config set <key> <value>` (or
//! `hipfire config <tag> set <key> <value>` for per-model overrides) and the
//! CLI's stderr is surfaced verbatim when it rejects a value.

use std::{env, path::PathBuf, process::Command};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliInvocation {
    pub program: String,
    pub leading_args: Vec<String>,
    /// Short human-readable label for status lines ("hipfire", "bun cli/index.ts", ...).
    pub label: String,
}

/// Resolution order (first hit wins):
/// 1. `HIPFIRE_TUI_CLI` env override — whitespace-split into program + leading args.
/// 2. Installed `hipfire` wrapper on `PATH`.
/// 3. `bun <cwd>/cli/index.ts` when running from a hipfire checkout.
/// 4. `bun ~/.hipfire/cli/index.ts` (installed CLI without the wrapper).
pub fn resolve_cli() -> Option<CliInvocation> {
    if let Ok(raw) = env::var("HIPFIRE_TUI_CLI") {
        // Explicit override: honor it even if the target does not exist yet so
        // the failure is loud (surfaced by run_cli) instead of silently
        // falling back to a different CLI than the user asked for.
        return parse_cli_override(&raw);
    }
    if let Some(path) = find_on_path("hipfire") {
        return Some(CliInvocation {
            program: path.display().to_string(),
            leading_args: Vec::new(),
            label: "hipfire".into(),
        });
    }
    let mut script_candidates = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        script_candidates.push(cwd.join("cli/index.ts"));
    }
    if let Some(home) = env::var_os("HOME") {
        script_candidates.push(PathBuf::from(home).join(".hipfire/cli/index.ts"));
    }
    for script in script_candidates {
        if script.is_file() {
            return Some(CliInvocation {
                program: "bun".into(),
                leading_args: vec![script.display().to_string()],
                label: format!("bun {}", script.display()),
            });
        }
    }
    None
}

pub fn parse_cli_override(raw: &str) -> Option<CliInvocation> {
    let mut parts = raw.split_whitespace().map(str::to_string);
    let program = parts.next()?;
    Some(CliInvocation {
        label: format!("HIPFIRE_TUI_CLI ({program})"),
        leading_args: parts.collect(),
        program,
    })
}

/// Command line for `hipfire config [tag] set <key> <value>`.
pub fn config_set_args(model_scope: Option<&str>, key: &str, value: &str) -> Vec<String> {
    let mut args = vec!["config".to_string()];
    if let Some(tag) = model_scope {
        args.push(tag.to_string());
    }
    args.push("set".to_string());
    args.push(key.to_string());
    args.push(value.to_string());
    args
}

/// Command line for `hipfire serve -d` (detached daemon).
pub fn serve_detach_args() -> Vec<String> {
    vec!["serve".to_string(), "-d".to_string()]
}

/// Run the CLI to completion. Ok carries trimmed stdout; Err carries the
/// first meaningful stderr line (the CLI prints validation errors there).
pub fn run_cli(cli: &CliInvocation, args: &[String]) -> Result<String, String> {
    let output = Command::new(&cli.program)
        .args(&cli.leading_args)
        .args(args)
        .output()
        .map_err(|err| format!("failed to launch {}: {err}", cli.label))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if output.status.success() {
        return Ok(stdout);
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let msg = if stderr.is_empty() { stdout } else { stderr };
    Err(msg
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("command failed with no output")
        .trim()
        .to_string())
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if !candidate.is_file() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let executable = candidate
                .metadata()
                .map(|m| m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false);
            if !executable {
                continue;
            }
        }
        return Some(candidate);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{config_set_args, parse_cli_override, run_cli, serve_detach_args, CliInvocation};

    #[test]
    fn global_config_set_cmdline() {
        assert_eq!(
            config_set_args(None, "kv_cache", "q8"),
            vec!["config", "set", "kv_cache", "q8"]
        );
    }

    #[test]
    fn per_model_config_set_cmdline() {
        assert_eq!(
            config_set_args(Some("qwen3.5:9b"), "dflash_mode", "on"),
            vec!["config", "qwen3.5:9b", "set", "dflash_mode", "on"]
        );
    }

    #[test]
    fn values_with_spaces_stay_one_argv_entry() {
        let args = config_set_args(None, "kv_adaptive", "advanced:k=fwht4,v=lloyd4");
        assert_eq!(args.len(), 4);
        let args = config_set_args(None, "default_model", "qwen3.5:9b");
        assert_eq!(args[3], "qwen3.5:9b");
    }

    #[test]
    fn serve_cmdline_is_detached() {
        assert_eq!(serve_detach_args(), vec!["serve", "-d"]);
    }

    #[test]
    fn cli_override_splits_program_and_leading_args() {
        let cli = parse_cli_override("bun /repo/cli/index.ts").unwrap();
        assert_eq!(cli.program, "bun");
        assert_eq!(cli.leading_args, vec!["/repo/cli/index.ts"]);
        assert!(parse_cli_override("   ").is_none());
        let bare = parse_cli_override("hipfire").unwrap();
        assert_eq!(bare.program, "hipfire");
        assert!(bare.leading_args.is_empty());
    }

    #[test]
    fn run_cli_surfaces_first_stderr_line_on_failure() {
        let cli = CliInvocation {
            program: "sh".into(),
            leading_args: vec![
                "-c".into(),
                "echo unrelated; echo 'kv_cache must be one of: auto, q8' >&2; exit 1".into(),
            ],
            label: "test-sh".into(),
        };
        let err = run_cli(&cli, &[]).unwrap_err();
        assert_eq!(err, "kv_cache must be one of: auto, q8");

        let ok = CliInvocation {
            program: "sh".into(),
            leading_args: vec!["-c".into(), "echo 'kv_cache = q8'".into()],
            label: "test-sh".into(),
        };
        assert_eq!(run_cli(&ok, &[]).unwrap(), "kv_cache = q8");
    }
}
