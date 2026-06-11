use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use clap::Args;

const EVAL_HELP: &str = r#"hipfire eval - quant admission/model evaluation harness

Usage:
  hipfire eval --model <model> [--tier fast|medium|long|extensive]
  hipfire eval --model <model> --battery smoke,quality,speed
  hipfire eval --model <model> --suite gpqa --fetch-datasets

Build runner:
  cargo build --release -p hipfire-runtime --bin hipfire-eval"#;

const HOST_PROFILE_HELP: &str = r#"hipfire host-profile - measured host capability report

Usage:
  hipfire host-profile [--out <path>] [--models-dir <dir>] [--runs N]

Common options:
  --out <path>              Write report JSON there
  --models-dir <dir>        Model storage directory to test, default ~/.hipfire/models
  --size-mib <N>            CPU/GPU copy test size in MiB, default 128
  --storage-size-mib <N>    Storage test size in MiB, default 128
  --runs <N>                Samples per test, default 3
  --warmup-runs <N>         Unmeasured warmup samples per test, default 1
  --gpu-max-size-mib <N>    Cap largest GPU read/write sweep payload size
  --gpu-sweep-mib-step <N>  Override default GPU MiB payload spacing
  --skip-gpu                Skip HIP copy tests
  --skip-storage            Skip ~/.hipfire/models storage tests
  --json                    Print report JSON to stdout

Build runner:
  cargo build --release -p hipfire-runtime --bin hipfire-host-profile"#;

#[derive(Debug, Args)]
#[command(disable_help_flag = true, trailing_var_arg = true)]
pub struct EvalArgs {
    /// Arguments forwarded to hipfire-eval
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<OsString>,
}

#[derive(Debug, Args)]
#[command(disable_help_flag = true, trailing_var_arg = true)]
pub struct HostProfileArgs {
    /// Arguments forwarded to hipfire-host-profile
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<OsString>,
}

pub fn run_eval(args: EvalArgs) -> anyhow::Result<()> {
    run_forwarded(
        Runner::eval(),
        args.args,
        "HIPFIRE_EVAL_BIN",
        "hipfire-eval",
        EVAL_HELP,
        "cargo build --release -p hipfire-runtime --bin hipfire-eval",
    )
}

pub fn run_host_profile(args: HostProfileArgs) -> anyhow::Result<()> {
    run_forwarded(
        Runner::host_profile(),
        args.args,
        "HIPFIRE_HOST_PROFILE_BIN",
        "hipfire-host-profile",
        HOST_PROFILE_HELP,
        "cargo build --release -p hipfire-runtime --bin hipfire-host-profile",
    )
}

fn run_forwarded(
    runner: Runner,
    args: Vec<OsString>,
    env_var: &str,
    bin_name: &str,
    help: &str,
    build_hint: &str,
) -> anyhow::Result<()> {
    if is_help(&args) {
        println!("{help}");
        return Ok(());
    }

    let bin = resolve_runner_binary(&runner, env_var, bin_name)
        .ok_or_else(|| anyhow::anyhow!("{bin_name} not found.\nBuild it with: {build_hint}"))?;
    let status = Command::new(&bin)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    if let Some(code) = status.code() {
        if code == 0 {
            Ok(())
        } else {
            std::process::exit(code);
        }
    } else {
        anyhow::bail!("{bin_name} terminated by signal")
    }
}

fn is_help(args: &[OsString]) -> bool {
    args.is_empty() || args.iter().any(|arg| arg == "-h" || arg == "--help")
}

#[derive(Debug)]
struct Runner {
    release_name: &'static str,
    debug_name: Option<&'static str>,
}

impl Runner {
    fn eval() -> Self {
        Self {
            release_name: "hipfire-eval",
            debug_name: None,
        }
    }

    fn host_profile() -> Self {
        Self {
            release_name: "hipfire-host-profile",
            debug_name: Some("hipfire-host-profile"),
        }
    }
}

fn resolve_runner_binary(runner: &Runner, env_var: &str, bin_name: &str) -> Option<PathBuf> {
    runner_candidates(runner, env_var, bin_name)
        .into_iter()
        .find(|path| path.exists())
}

fn runner_candidates(runner: &Runner, env_var: &str, bin_name: &str) -> Vec<PathBuf> {
    let exe = std::env::consts::EXE_SUFFIX;
    let mut candidates = Vec::new();

    if let Some(path) = std::env::var_os(env_var).filter(|p| !p.is_empty()) {
        candidates.push(PathBuf::from(path));
    }

    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(dir) = current_exe.parent() {
            candidates.push(dir.join(format!("{}{}", runner.release_name, exe)));
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(format!("target/release/{}{}", runner.release_name, exe)));
        if let Some(debug_name) = runner.debug_name {
            candidates.push(cwd.join(format!("target/debug/{}{}", debug_name, exe)));
        }
    }

    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(
            Path::new(&home)
                .join(".hipfire")
                .join("bin")
                .join(format!("{bin_name}{exe}")),
        );
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_matches_empty_and_help_flags() {
        assert!(is_help(&[]));
        assert!(is_help(&[OsString::from("--help")]));
        assert!(is_help(&[OsString::from("-h")]));
        assert!(!is_help(&[
            OsString::from("--model"),
            OsString::from("qwen")
        ]));
    }

    #[test]
    fn eval_candidates_include_env_release_and_install_locations() {
        let candidates = runner_candidates(&Runner::eval(), "HIPFIRE_EVAL_BIN", "hipfire-eval");
        assert!(candidates
            .iter()
            .any(|p| p.ends_with("target/release/hipfire-eval")));
        assert!(candidates
            .iter()
            .any(|p| p.ends_with(".hipfire/bin/hipfire-eval")));
    }

    #[test]
    fn host_profile_candidates_include_debug_binary() {
        let candidates = runner_candidates(
            &Runner::host_profile(),
            "HIPFIRE_HOST_PROFILE_BIN",
            "hipfire-host-profile",
        );
        assert!(candidates
            .iter()
            .any(|p| p.ends_with("target/debug/hipfire-host-profile")));
    }
}
