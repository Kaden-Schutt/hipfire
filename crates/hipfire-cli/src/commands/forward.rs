use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use clap::Args;

use crate::model::find_model;

const EVAL_HELP: &str = r#"hipfire eval - quant admission/model evaluation harness

Usage:
  hipfire eval --model <model> [--tier fast|medium|long|extensive]
  hipfire eval --model <model> --battery smoke,quality,speed
  hipfire eval --model <model> --suite gpqa --fetch-datasets

Model arguments accept local names, shorthand, aliases, or paths. For example,
lfm2.5:350m resolves to the preferred local quant for lfm2.5-350m.

Build runner:
  cargo build --release -p hipfire-eval"#;

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

const COLLECT_ARTIFACTS_HELP: &str = r#"hipfire collect-artifacts - single-load Tier-1 calibration artifact collector

Loads a bf16 .hfq once and writes a unified <model>.calib.hfq bundling the
per-tensor Hessian + imatrix (full Hessian for dense projections; imatrix-only
for MoE routed experts), the MoE router histogram (MoE models), and optionally
KLDREF.

Usage:
  hipfire collect-artifacts --model <bf16.hfq> --corpus <text> \
      --output <out.calib.hfq> [--max-tokens N] [--kldref]

--model accepts a local name, shorthand, alias, or path.

Build runner:
  cargo build --release -p hipfire-runtime --example collect_artifacts"#;

#[derive(Debug, Args)]
#[command(disable_help_flag = true, trailing_var_arg = true)]
pub struct CollectArtifactsArgs {
    /// Arguments forwarded to the collect_artifacts runner
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<OsString>,
}

const REPACK_HELP: &str = r#"hipfire repack - reshuffle a .hfq into an arch-optimal weight layout

Takes a canonical (general, portable) .hfq and writes an arch-tagged
<model>.<arch>.hfq whose weights are pre-packed into the device layout that
arch's kernels want — so the model loads with no per-load repack. The canonical
file is the source of truth and is never modified.

Currently repacks Opus W4A4 (op4 / op4-4) tensors into the combined interleaved-decode
layout (quant_type 34 -> 37). Other tensors are copied through.

Usage:
  hipfire repack <model.hfq> [--arch <gfx>] [-o <out.hfq>]

  --arch defaults to the live GPU (probed read-only). Default output is
  <model>.<arch>.hfq beside the input, e.g.
    qwen3.5-0.8b-op4.hfq -> qwen3.5-0.8b-op4.gfx1103.hfq

The positional model accepts a local name, shorthand, alias, or path.

Build runner:
  cargo build --release -p hipfire-runtime --example oq4_repack"#;

#[derive(Debug, Args)]
#[command(disable_help_flag = true, trailing_var_arg = true)]
pub struct RepackArgs {
    /// Arguments forwarded to the oq4_repack runner
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<OsString>,
}

pub fn run_eval(args: EvalArgs) -> anyhow::Result<()> {
    run_forwarded(
        Runner::eval(),
        resolve_forwarded_model_args(args.args, false),
        "HIPFIRE_EVAL_BIN",
        "hipfire-eval",
        EVAL_HELP,
        "cargo build --release -p hipfire-eval",
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

pub fn run_collect_artifacts(args: CollectArtifactsArgs) -> anyhow::Result<()> {
    run_forwarded(
        Runner::collect_artifacts(),
        resolve_forwarded_model_args(args.args, false),
        "HIPFIRE_COLLECT_ARTIFACTS_BIN",
        "collect_artifacts",
        COLLECT_ARTIFACTS_HELP,
        "cargo build --release -p hipfire-runtime --example collect_artifacts",
    )
}

pub fn run_repack(args: RepackArgs) -> anyhow::Result<()> {
    run_forwarded(
        Runner::repack(),
        resolve_forwarded_model_args(args.args, true),
        "HIPFIRE_REPACK_BIN",
        "oq4_repack",
        REPACK_HELP,
        "cargo build --release -p hipfire-runtime --example oq4_repack",
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

fn resolve_forwarded_model_args(
    args: Vec<OsString>,
    resolve_first_positional: bool,
) -> Vec<OsString> {
    const MODEL_VALUE_FLAGS: &[&str] = &["--model", "--baseline", "--reference", "--draft"];
    let mut out = Vec::with_capacity(args.len());
    let mut resolve_next = false;
    let mut resolved_positional = false;

    for arg in args {
        if resolve_next {
            out.push(resolve_model_os(arg));
            resolve_next = false;
            continue;
        }

        let Some(s) = arg.to_str() else {
            out.push(arg);
            continue;
        };

        if MODEL_VALUE_FLAGS.contains(&s) {
            out.push(arg);
            resolve_next = true;
            continue;
        }

        if let Some((flag, value)) = s.split_once('=') {
            if MODEL_VALUE_FLAGS.contains(&flag) {
                let resolved = resolve_model_str(value);
                out.push(OsString::from(format!("{flag}={resolved}")));
                continue;
            }
        }

        if resolve_first_positional && !resolved_positional && !s.starts_with('-') {
            out.push(resolve_model_str(s).into());
            resolved_positional = true;
            continue;
        }

        out.push(arg);
    }

    out
}

fn resolve_model_os(arg: OsString) -> OsString {
    arg.to_str()
        .map(resolve_model_str)
        .map(OsString::from)
        .unwrap_or(arg)
}

fn resolve_model_str(value: &str) -> String {
    find_model(value)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| value.to_string())
}

#[derive(Debug)]
struct Runner {
    release_name: &'static str,
    debug_name: Option<&'static str>,
    /// When set, the binary is a cargo example, so it lives under
    /// `target/{profile}/examples/` rather than `target/{profile}/`.
    is_example: bool,
}

impl Runner {
    fn eval() -> Self {
        Self {
            release_name: "hipfire-eval",
            debug_name: None,
            is_example: false,
        }
    }

    fn host_profile() -> Self {
        Self {
            release_name: "hipfire-host-profile",
            debug_name: Some("hipfire-host-profile"),
            is_example: false,
        }
    }

    fn collect_artifacts() -> Self {
        Self {
            release_name: "collect_artifacts",
            debug_name: Some("collect_artifacts"),
            is_example: true,
        }
    }

    fn repack() -> Self {
        Self {
            release_name: "oq4_repack",
            debug_name: Some("oq4_repack"),
            is_example: true,
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

    let sub = if runner.is_example { "examples/" } else { "" };
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(format!("target/release/{sub}{}{exe}", runner.release_name)));
        if let Some(debug_name) = runner.debug_name {
            candidates.push(cwd.join(format!("target/debug/{sub}{debug_name}{exe}")));
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

    #[test]
    fn collect_artifacts_candidates_resolve_the_example_path() {
        let candidates = runner_candidates(
            &Runner::collect_artifacts(),
            "HIPFIRE_COLLECT_ARTIFACTS_BIN",
            "collect_artifacts",
        );
        // The runner is a cargo example, so it lives under examples/.
        assert!(candidates
            .iter()
            .any(|p| p.ends_with("target/release/examples/collect_artifacts")));
        assert!(candidates
            .iter()
            .any(|p| p.ends_with("target/debug/examples/collect_artifacts")));
    }

    #[test]
    fn forwarded_model_args_resolve_model_like_positions() {
        assert_eq!(
            resolve_forwarded_model_args(
                vec![
                    OsString::from("--model=missing-model"),
                    OsString::from("--battery"),
                    OsString::from("speed"),
                ],
                false,
            ),
            vec![
                OsString::from("--model=missing-model"),
                OsString::from("--battery"),
                OsString::from("speed"),
            ]
        );
        assert_eq!(
            resolve_forwarded_model_args(
                vec![
                    OsString::from("missing-model"),
                    OsString::from("--arch"),
                    OsString::from("gfx1151")
                ],
                true,
            ),
            vec![
                OsString::from("missing-model"),
                OsString::from("--arch"),
                OsString::from("gfx1151")
            ]
        );
    }
}
