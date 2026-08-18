// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Machine install / repair wizard (`hipfire setup`).
//!
//! Bash only bootstraps the CLI binary; this module owns ROCm/GPU resolution,
//! runtime builds, atomic binary install, cold-kernel seeding, and install.json.

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::{
    env, fs,
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// Cooperative cancel flag for setup. SIGINT/SIGTERM set this; handlers never
/// call `process::exit` so child cleanup, spinner `Drop`, and binary rollback run.
static SETUP_INTERRUPT: AtomicBool = AtomicBool::new(false);

fn install_setup_interrupt_handler() {
    // `termination` enables SIGTERM alongside SIGINT. Ignore AlreadyExists so a
    // pre-installed process handler does not abort setup.
    let _ = ctrlc::set_handler(|| {
        SETUP_INTERRUPT.store(true, Ordering::SeqCst);
    });
}

fn interrupted() -> bool {
    SETUP_INTERRUPT.load(Ordering::SeqCst)
}

fn ensure_not_interrupted() -> Result<()> {
    if interrupted() {
        bail!("setup interrupted");
    }
    Ok(())
}

/// Result of one interactive stdin line read (including EOF).
#[derive(Debug, Clone, PartialEq, Eq)]
enum PromptRead {
    Eof,
    Line(String),
}

/// Map `Read::read_line` byte count + buffer into [`PromptRead`].
fn prompt_line_from_read(bytes_read: usize, buf: &str) -> PromptRead {
    if bytes_read == 0 {
        PromptRead::Eof
    } else {
        PromptRead::Line(buf.to_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinueDecision {
    Proceed,
    Cancel,
}

/// Continue prompt: EOF and explicit no cancel; anything else proceeds.
fn continue_decision(read: PromptRead) -> ContinueDecision {
    match read {
        PromptRead::Eof => ContinueDecision::Cancel,
        PromptRead::Line(line) => {
            let answer = line.trim().to_ascii_lowercase();
            if matches!(answer.as_str(), "n" | "no") {
                ContinueDecision::Cancel
            } else {
                ContinueDecision::Proceed
            }
        }
    }
}

/// Numbered selection: EOF is a hard error; invalid input yields `Ok(None)` to retry.
fn selection_from_prompt_read(read: PromptRead, count: usize) -> Result<Option<usize>> {
    match read {
        PromptRead::Eof => bail!("EOF while reading selection"),
        PromptRead::Line(line) => {
            if let Ok(n) = line.trim().parse::<usize>() {
                if (1..=count).contains(&n) {
                    return Ok(Some(n - 1));
                }
            }
            Ok(None)
        }
    }
}

pub(crate) fn setup_command(paths: &crate::Paths, args: crate::SetupArgs) -> Result<()> {
    install_setup_interrupt_handler();

    let source_arg = args.source.as_path();
    if !source_arg.join("Cargo.toml").is_file() {
        bail!(
            "source {} has no Cargo.toml; pass --source PATH",
            source_arg.display()
        );
    }
    let source = fs::canonicalize(source_arg)
        .with_context(|| format!("failed to canonicalize source {}", source_arg.display()))?;

    // Resolve git identity before any mutation; failures propagate (no "unknown").
    let commit_short = crate::git_output(&source, &["rev-parse", "--short", "HEAD"])
        .context("failed to resolve short git HEAD for source checkout")?;
    let commit = crate::git_output(&source, &["rev-parse", "HEAD"])
        .context("failed to resolve full git HEAD for source checkout")?;

    // Honour explicit --hipcc and --strict-rocm for this run without mutating
    // process-global env in tests — the pure helpers below take them as params.
    // For the live installer we also export them so child cargo/hipcc processes
    // see the same selection.
    let hipcc_override = args.hipcc.as_deref();
    let strict = args.strict_rocm || hipfire_config::rocm::is_strict_rocm();
    if let Some(hipcc) = hipcc_override {
        if !hipcc.is_file() {
            bail!(
                "--hipcc {} does not exist or is not executable; set --hipcc to an executable hipcc/amdclang++ (also HIPFIRE_HIPCC)",
                hipcc.display()
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(hipcc) {
                if meta.permissions().mode() & 0o111 == 0 {
                    bail!("--hipcc {} is not executable", hipcc.display());
                }
            }
        }
    }
    // Export for child processes (single-threaded installer, safe).
    if let Some(hipcc) = hipcc_override {
        // SAFETY: installer is single-threaded at this point; tests use pure
        // helpers and do not call setup_command.
        unsafe { std::env::set_var("HIPFIRE_HIPCC", hipcc) };
    }
    if strict {
        unsafe { std::env::set_var("HIPFIRE_ROCM_STRICT", "1") };
    }

    // --- 4a. Resolve ROCm root (no mutation) ---
    let rocm_root = resolve_rocm_root_with(args.rocm_root.as_deref(), hipcc_override, strict, args.yes)?;
    ensure_rocm_complete(&rocm_root)?;
    ensure_not_interrupted()?;

    // Print resolved provenance before any heavy work so a failing install
    // report always contains it. Uses the same toolchain resolver as
    // hipfire-rocm-resolve so output stays consistent.
    {
        let toolchain = hipfire_config::rocm::resolve_toolchain_for_explicit(
            Some(&rocm_root),
            hipcc_override,
            strict,
        )
        .or_else(|_| hipfire_config::rocm::resolve_toolchain());
        if let Ok(tc) = toolchain {
            let version = hipfire_config::rocm::version_for_root(&tc.root)
                .or_else(hipfire_config::rocm::version)
                .unwrap_or_else(|| "unknown".to_string());
            let compiler = tc
                .compiler
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not found".to_string());
            let source = tc
                .compiler_source
                .as_ref()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let comp_root = tc
                .compiler_root
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let runtime = hipfire_config::rocm::runtime_library(&tc.root)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not found".to_string());
            eprintln!("ROCm root:       {} (version {version})", tc.root.display());
            eprintln!("HIPCC:           {compiler} (source: {source}, root: {comp_root})");
            eprintln!("HIP runtime:     {runtime}");
            for line in hipfire_config::rocm::toolchain_warnings(&tc) {
                eprintln!("{line}");
            }
        } else {
            // Fallback: at least show the selected root when toolchain
            // resolution failed (e.g. strict cross-root). The error itself will
            // already have been printed via ensure_* failures.
            eprintln!("ROCm root:       {}", rocm_root.display());
        }
    }


    // --- 4b. Resolve GPU arch (no mutation) ---
    let gpu_arch = resolve_gpu_arch(args.gpu_arch.as_deref(), &rocm_root, args.yes)?;
    ensure_not_interrupted()?;

    let profile = args
        .profile
        .as_deref()
        .filter(|p| !p.is_empty())
        .unwrap_or("auto");
    let bin_dir = paths.root.join("bin");

    // --- 4c. Confirm before any mutation ---
    if !args.yes {
        println!("Source:          {}", source.display());
        println!("Commit:          {commit_short}");
        println!("ROCm root:       {}", rocm_root.display());
        println!(
            "GPU arch:        {}",
            gpu_arch.as_deref().unwrap_or("unknown")
        );
        println!("Profile:         {profile}");
        println!("Install prefix:  {}", bin_dir.display());
        eprint!("Continue? [Y/n] ");
        io::stderr().flush()?;
        let mut answer = String::new();
        let n = io::stdin().read_line(&mut answer)?;
        if continue_decision(prompt_line_from_read(n, &answer)) == ContinueDecision::Cancel {
            println!("Installation cancelled.");
            return Ok(());
        }
    }
    ensure_not_interrupted()?;

    run_cargo_required(
        &source,
        &rocm_root,
        &[
            "build",
            "--release",
            "--features",
            "deltanet",
            "--example",
            "daemon",
            "-p",
            "hipfire-runtime",
        ],
        "required runtime (daemon) build",
    )?;
    ensure_not_interrupted()?;

    let tui_built = run_cargo_optional(
        &source,
        &rocm_root,
        &["build", "--release", "-p", "hipfire-tui"],
        "hipfire-tui",
    );
    ensure_not_interrupted()?;
    let quantize_built = run_cargo_optional(
        &source,
        &rocm_root,
        &["build", "--release", "-p", "hipfire-quantize"],
        "hipfire-quantize",
    );
    ensure_not_interrupted()?;

    fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed to create {}", bin_dir.display()))?;
    fs::create_dir_all(&paths.models)
        .with_context(|| format!("failed to create {}", paths.models.display()))?;

    // Always consume `<source>/target` regardless of ambient Cargo config.
    let release = source.join("target").join("release");
    let mut replacements: Vec<BinaryReplacement> = Vec::new();

    let install_one =
        |src: &Path, dest: &Path, replacements: &mut Vec<BinaryReplacement>| -> Result<()> {
            ensure_not_interrupted()?;
            match install_binary_with_backup(src, dest) {
                Ok(rep) => {
                    replacements.push(rep);
                    Ok(())
                }
                Err(err) => {
                    rollback_replacements(replacements);
                    Err(err)
                }
            }
        };

    install_one(
        &release.join("examples").join("daemon"),
        &bin_dir.join("daemon"),
        &mut replacements,
    )?;
    install_one(
        &release.join("hipfire"),
        &bin_dir.join("hipfire"),
        &mut replacements,
    )?;
    if tui_built {
        let src = release.join("hipfire-tui");
        if src.is_file() {
            install_one(&src, &bin_dir.join("hipfire-tui"), &mut replacements)?;
        }
    }
    if quantize_built {
        let src = release.join("hipfire-quantize");
        if src.is_file() {
            install_one(&src, &bin_dir.join("hipfire-quantize"), &mut replacements)?;
        }
    }
    ensure_not_interrupted().map_err(|err| {
        rollback_replacements(&replacements);
        err
    })?;

    if let Some(arch) = gpu_arch.as_deref() {
        let cold = bin_dir.join("kernels").join("compiled").join(arch);
        if let Err(err) = fs::create_dir_all(&cold) {
            rollback_replacements(&replacements);
            return Err(err)
                .with_context(|| format!("failed to create cold kernel dir {}", cold.display()));
        }
    }
    ensure_not_interrupted().map_err(|err| {
        rollback_replacements(&replacements);
        err
    })?;

    let mut precompile = Command::new(bin_dir.join("daemon"));
    precompile.arg("--precompile");
    apply_rocm_env(&mut precompile, &rocm_root);
    match run_capturing(precompile) {
        Ok(out) if out.status.success() && !interrupted() => {}
        Ok(out) if interrupted() => {
            let _ = out;
            rollback_replacements(&replacements);
            bail!("setup interrupted");
        }
        Ok(out) => {
            print_command_output(&out);
            rollback_replacements(&replacements);
            bail!(
                "required kernel precompile failed; previous binaries remain in place ({})",
                out.status
            );
        }
        Err(err) => {
            rollback_replacements(&replacements);
            return Err(err).context("failed to start installed daemon --precompile");
        }
    }

    // Profile can mutate config.toml before install.json is written. Snapshot so
    // a late metadata failure restores both binaries and prior config state.
    let config_snapshot = if matches!(profile, "hip" | "redline") {
        match snapshot_config_toml(&paths.config.config_toml) {
            Ok(snap) => Some(snap),
            Err(err) => {
                rollback_replacements(&replacements);
                return Err(err);
            }
        }
    } else {
        None
    };

    if matches!(profile, "hip" | "redline") {
        if let Err(err) = crate::config_profile_command(
            paths,
            Some(crate::ConfigProfileAction::Set {
                name: profile.to_owned(),
            }),
        ) {
            rollback_replacements(&replacements);
            if let Some(snap) = &config_snapshot {
                restore_config_toml(snap);
            }
            return Err(err);
        }
    }
    ensure_not_interrupted().map_err(|err| {
        rollback_replacements(&replacements);
        if let Some(snap) = &config_snapshot {
            restore_config_toml(snap);
        }
        err
    })?;

    let meta_ref = metadata_ref(&args);
    let installed_at = crate::unix_timestamp();
    let record = json!({
        "commit": commit,
        "ref": meta_ref,
        "rocm_root": rocm_root.to_string_lossy(),
        "gpu_arch": gpu_arch,
        "hipcc": hipcc_override.map(|p| p.to_string_lossy().into_owned()),
        "strict_rocm": strict,
        "profile": profile,
        "installed_at": installed_at,
    });
    if let Err(err) = write_install_json(&paths.root, &record) {
        rollback_replacements(&replacements);
        if let Some(snap) = &config_snapshot {
            restore_config_toml(snap);
        }
        return Err(err);
    }

    // Only drop backups once profile application and install.json both succeed.
    cleanup_backups(&replacements);

    println!("hipfire installed to {}", bin_dir.display());
    if !path_on_path(&bin_dir) {
        println!("export PATH=\"{}:$PATH\"", bin_dir.display());
    }
    Ok(())
}

fn resolve_rocm_root(explicit: Option<&Path>, yes: bool) -> Result<PathBuf> {
    // Env-based wrapper for call sites that do not have explicit hipcc/strict.
    // The live installer passes them explicitly via resolve_rocm_root_with to
    // avoid global mutation in tests (see rocm.rs:723-746 pattern).
    let hipcc = std::env::var_os("HIPFIRE_HIPCC")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let strict = hipfire_config::rocm::is_strict_rocm();
    resolve_rocm_root_with(explicit, hipcc.as_deref(), strict, yes)
}

/// Pure form of [`resolve_rocm_root`] with injected hipcc/strict (no env reads).
///
/// `hipcc` is the `HIPFIRE_HIPCC` / `--hipcc` override if set, `strict`
/// mirrors `HIPFIRE_ROCM_STRICT=1` / `--strict-rocm`.
fn resolve_rocm_root_with(
    explicit: Option<&Path>,
    hipcc: Option<&Path>,
    strict: bool,
    yes: bool,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        // Prefer the toolchain resolver so a libs-only explicit root with a
        // cross-root compiler is accepted when not strict. Preserve the exact
        // invalid-root wording with the path as given (before canonicalize).
        let toolchain = hipfire_config::rocm::resolve_toolchain_for_explicit(Some(path), hipcc, strict);
        if toolchain.is_ok() {
            return Ok(canonicalize_or_keep(path));
        }
        // No usable compiler even with cross-root/override — hard-fail and
        // point at both --rocm-root and --hipcc as remedies.
        let tried = hipfire_config::rocm::resolve_toolchain_for_explicit(Some(path), hipcc, strict)
            .err()
            .unwrap_or_else(|| "no device compiler".to_string());
        bail!(
            "ROCm root {} has no usable device compiler; pass --rocm-root PATH or --hipcc PATH (HIPFIRE_HIPCC)\n{tried}",
            path.display()
        );
    }

    // Gather candidates that are usable either as coherent SDKs or as
    // headers+runtime-only roots with an external compiler when not strict.
    let candidates = usable_rocm_roots_with(hipfire_config::rocm::roots(), hipcc, strict);

    let selected = match candidates.len() {
        0 => bail!("no ROCm installation found; install ROCm or pass --rocm-root PATH or --hipcc PATH (HIPFIRE_HIPCC)"),
        1 => candidates.into_iter().next().unwrap(),
        _ => {
            if io::stdin().is_terminal() && !yes {
                println!("Multiple ROCm installations found:");
                for (i, root) in candidates.iter().enumerate() {
                    // Flag roots that carry only a compiler, so the choice is
                    // informed rather than a coin flip between a usable root
                    // and one that fails at the end of the install.
                    let note = match hipfire_config::rocm::missing_components(root).as_slice() {
                        [] => String::new(),
                        missing => format!(
                            "  [incomplete: missing {}]",
                            missing
                                .iter()
                                .map(|m| m.what)
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    };
                    println!("  {}. {}{note}", i + 1, root.display());
                }
                let idx = read_numbered_selection(candidates.len(), "ROCm root")?;
                candidates.into_iter().nth(idx).unwrap()
            } else {
                let list = candidates
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!(
                    "multiple ROCm installations found: {list}; choose one with --rocm-root PATH"
                );
            }
        }
    };
    Ok(selected)
}

/// Prefer a canonical absolute path; keep `path` if canonicalize fails.
fn canonicalize_or_keep(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Roots with a usable device compiler, canonicalized and deduped (first-seen order).
///
/// Alias paths that resolve to the same filesystem location collapse to one entry
/// (e.g. `/opt/rocm/core`, `core-7`, `core-7.14` → one root). Canonicalization
/// failure keeps the original path rather than discarding a usable candidate.
fn usable_rocm_roots(roots: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let hipcc = std::env::var_os("HIPFIRE_HIPCC")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let strict = hipfire_config::rocm::is_strict_rocm();
    usable_rocm_roots_with(roots, hipcc.as_deref(), strict)
}

/// Pure form of [`usable_rocm_roots`] with injected hipcc/strict.
fn usable_rocm_roots_with(
    roots: impl IntoIterator<Item = PathBuf>,
    hipcc: Option<&Path>,
    strict: bool,
) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    for root in roots {
        // A root is usable if it is coherent, or if it is headers+runtime-only
        // and a cross-root compiler can be supplied (when not strict).
        let usable = if root_has_device_compiler(&root) {
            true
        } else if !strict && hipfire_config::rocm::is_headers_runtime_only_root(&root) {
            // Cross-root compiler available via explicit hipcc or via PATH/other roots.
            if hipcc.is_some_and(|p| p.is_file()) {
                true
            } else {
                // Check PATH / other roots via the toolchain helper — if a
                // toolchain can be resolved for this root alone, it is usable.
                hipfire_config::rocm::resolve_toolchain_for_explicit(Some(&root), hipcc, strict).is_ok()
            }
        } else {
            false
        };
        if !usable {
            continue;
        }
        let key = canonicalize_or_keep(&root);
        if candidates.iter().any(|c| c == &key) {
            continue;
        }
        candidates.push(key);
    }
    candidates
}

/// Refuse a ROCm root that carries a device compiler but not the HIP runtime.
///
/// Runs before any build so the failure costs seconds, not the three cargo
/// builds and 38-kernel precompile it used to sit behind. Without this the
/// first symptom is clang reporting `'hip/hip_runtime.h' file not found` at the
/// very end of a long install, which names neither the real cause nor the fix.
fn ensure_rocm_complete(root: &Path) -> Result<()> {
    // The runtime is resolved at LOAD time across every candidate root and then
    // the dynamic loader, so its absence from this one root is not proof it is
    // missing — warn, never block. The reporting box is exactly that case:
    // `ldconfig -p` knows nothing about libamdhip64, yet the daemon loads it
    // from a resolved root and reports real VRAM. Blocking on a per-root probe
    // would have failed an install that works.
    if hipfire_config::rocm::roots()
        .iter()
        .all(|r| hipfire_config::rocm::runtime_library(r).is_none())
    {
        eprintln!(
            "WARNING: no HIP runtime library found under any known ROCm root.\n\
             \x20        If the daemon later fails to start, install it:"
        );
        for line in hipfire_config::rocm::install_guidance() {
            eprintln!("           {line}");
        }
    }

    // Headers, by contrast, must live under THIS root: they reach the device
    // compiler as --rocm-path=<root> and -I<root>/include, so a copy elsewhere
    // cannot be used and the compile provably cannot succeed. Hard-fail.
    if hipfire_config::rocm::is_complete_root(root) {
        return Ok(());
    }
    let mut msg = format!(
        "ROCm root {} has a device compiler but no HIP headers:\n\n  \
         missing: hip/hip_runtime.h\n  probed:  {}\n\n\
         A working device compiler is not sufficient: ROCm ships the compiler, the\n\
         HIP headers and the HIP runtime as separate packages, so installing only\n\
         the compiler leaves `hipcc --version` working while every kernel compile\n\
         fails on this header.\n\n\
         To install the headers:\n",
        root.display(),
        root.join("include")
            .join("hip")
            .join("hip_runtime.h")
            .display(),
    );
    for line in hipfire_config::rocm::install_guidance() {
        msg.push_str(&format!("  {line}\n"));
    }
    msg.push_str("\nTo use a different ROCm install instead: --rocm-root PATH or --hipcc PATH (HIPFIRE_HIPCC)");
    bail!(msg)
}

fn root_has_device_compiler(root: &Path) -> bool {
    hipfire_config::rocm::DEVICE_COMPILERS
        .iter()
        .any(|name| root.join("bin").join(name).is_file())
}

fn resolve_gpu_arch(explicit: Option<&str>, rocm_root: &Path, yes: bool) -> Result<Option<String>> {
    if let Some(arch) = explicit {
        return Ok(Some(arch.to_ascii_lowercase()));
    }

    let enumerator = rocm_root.join("bin").join("rocm_agent_enumerator");
    let output = match Command::new(&enumerator).output() {
        Ok(o) if o.status.success() => o,
        _ => {
            eprintln!("GPU architecture not detected; kernels will compile on first run");
            return Ok(None);
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut arches = parse_gpu_arches(&stdout);
    match arches.len() {
        0 => {
            eprintln!("GPU architecture not detected; kernels will compile on first run");
            Ok(None)
        }
        1 => Ok(Some(arches.remove(0))),
        _ => {
            if io::stdin().is_terminal() && !yes {
                println!("Multiple GPU architectures found:");
                for (i, arch) in arches.iter().enumerate() {
                    println!("  {}. {arch}", i + 1);
                }
                let idx = read_numbered_selection(arches.len(), "GPU architecture")?;
                Ok(Some(arches.remove(idx)))
            } else {
                let list = arches.join(", ");
                bail!("multiple GPU architectures found: {list}; choose one with --gpu-arch ARCH");
            }
        }
    }
}

/// Parse `rocm_agent_enumerator` stdout into ordered unique lowercase gfx arches.
fn parse_gpu_arches(stdout: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if let Some(arch) = normalize_gfx_arch(line) {
            if arch != "gfx000" && !out.contains(&arch) {
                out.push(arch);
            }
        }
    }
    out
}

fn normalize_gfx_arch(line: &str) -> Option<String> {
    let rest = line.strip_prefix("gfx")?;
    if rest.is_empty() {
        return None;
    }
    if !rest.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("gfx{}", rest.to_ascii_lowercase()))
}

fn metadata_ref(args: &crate::SetupArgs) -> String {
    args.reference
        .as_ref()
        .or(args.branch.as_ref())
        .or(args.tag.as_ref())
        .or(args.commit.as_ref())
        .cloned()
        .unwrap_or_else(|| "local".to_owned())
}

fn read_numbered_selection(count: usize, label: &str) -> Result<usize> {
    loop {
        ensure_not_interrupted()?;
        eprint!("Select {label} [1-{count}]: ");
        io::stderr().flush()?;
        let mut line = String::new();
        let n = io::stdin().read_line(&mut line)?;
        match selection_from_prompt_read(prompt_line_from_read(n, &line), count)? {
            Some(idx) => return Ok(idx),
            None => eprintln!("invalid selection"),
        }
    }
}

fn apply_rocm_env(cmd: &mut Command, root: &Path) {
    cmd.env("ROCM_PATH", root);
    cmd.env("HIP_PATH", root);
    cmd.env("HIPFIRE_ROCM_ROOT", root);
    cmd.env("HIPFIRE_ROCM_PATH", root);
}

/// Captured child process streams and exit status.
struct CapturedOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: ExitStatus,
}

/// Ephemeral ASCII spinner on TTY stderr; no-op otherwise. Always stoppable via
/// [`SpinnerGuard::stop`] or `Drop` so a failed spawn cannot leak the thread.
struct SpinnerGuard {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl SpinnerGuard {
    fn start() -> Self {
        if !io::stderr().is_terminal() {
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
            };
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            const FRAMES: &[u8] = b"|/-\\";
            let mut i = 0usize;
            let mut err = io::stderr();
            while !stop_flag.load(Ordering::Relaxed) {
                let frame = FRAMES[i % FRAMES.len()] as char;
                let _ = write!(err, "\r{frame} ");
                let _ = err.flush();
                i = i.wrapping_add(1);
                thread::sleep(Duration::from_millis(80));
            }
            // Clear the spinner glyphs without relying on ANSI escapes.
            let _ = write!(err, "\r  \r");
            let _ = err.flush();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SpinnerGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn join_captured(handle: Option<thread::JoinHandle<Vec<u8>>>) -> Vec<u8> {
    match handle {
        Some(handle) => match handle.join() {
            Ok(buf) => buf,
            Err(_) => Vec::new(),
        },
        None => Vec::new(),
    }
}

/// Run `cmd` with stdout/stderr captured. On a TTY stderr, show a spinner until
/// the child finishes (or spawn fails). Cooperative interrupt kills the child so
/// spinner `Drop` and caller rollback still run. Does not panic.
fn run_capturing(mut cmd: Command) -> io::Result<CapturedOutput> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let spinner = SpinnerGuard::start();
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            spinner.stop();
            return Err(err);
        }
    };

    let stdout_reader = child.stdout.take().map(|mut pipe| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            buf
        })
    });
    let stderr_reader = child.stderr.take().map(|mut pipe| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            buf
        })
    });

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if interrupted() {
                    let _ = child.kill();
                    match child.wait() {
                        Ok(status) => break status,
                        Err(err) => {
                            let _ = join_captured(stdout_reader);
                            let _ = join_captured(stderr_reader);
                            spinner.stop();
                            return Err(err);
                        }
                    }
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                let _ = join_captured(stdout_reader);
                let _ = join_captured(stderr_reader);
                spinner.stop();
                return Err(err);
            }
        }
    };

    let stdout = join_captured(stdout_reader);
    let stderr = join_captured(stderr_reader);
    spinner.stop();

    Ok(CapturedOutput {
        stdout,
        stderr,
        status,
    })
}

/// Emit complete captured child diagnostics to our stderr.
fn print_command_output(out: &CapturedOutput) {
    let mut err = io::stderr();
    if !out.stdout.is_empty() {
        let _ = err.write_all(&out.stdout);
        if !out.stdout.ends_with(b"\n") {
            let _ = writeln!(err);
        }
    }
    if !out.stderr.is_empty() {
        let _ = err.write_all(&out.stderr);
        if !out.stderr.ends_with(b"\n") {
            let _ = writeln!(err);
        }
    }
}

/// Force every cargo invocation to build into `<source>/target`.
fn pin_cargo_to_source_target(cmd: &mut Command, source: &Path) {
    let target_dir = source.join("target");
    cmd.arg("--target-dir").arg(&target_dir);
    cmd.env("CARGO_TARGET_DIR", &target_dir);
}

fn run_cargo_required(source: &Path, rocm_root: &Path, args: &[&str], label: &str) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.args(args).current_dir(source);
    pin_cargo_to_source_target(&mut cmd, source);
    apply_rocm_env(&mut cmd, rocm_root);
    let out = run_capturing(cmd).with_context(|| format!("failed to start {label}"))?;
    if interrupted() {
        bail!("setup interrupted");
    }
    if !out.status.success() {
        print_command_output(&out);
        bail!("{label} failed with {}", out.status);
    }
    Ok(())
}

fn run_cargo_optional(source: &Path, rocm_root: &Path, args: &[&str], name: &str) -> bool {
    let mut cmd = Command::new("cargo");
    cmd.args(args).current_dir(source);
    pin_cargo_to_source_target(&mut cmd, source);
    apply_rocm_env(&mut cmd, rocm_root);
    match run_capturing(cmd) {
        Ok(out) if out.status.success() && !interrupted() => true,
        Ok(out) if interrupted() => {
            // Treat cancel during optional builds as non-success so the caller
            // hits ensure_not_interrupted and unwinds cleanly.
            let _ = out;
            false
        }
        Ok(out) => {
            eprintln!(
                "WARNING: {name} build failed with {}; continuing",
                out.status
            );
            print_command_output(&out);
            false
        }
        Err(err) => {
            eprintln!("WARNING: {name} build failed; continuing ({err})");
            false
        }
    }
}

/// Record of one destination replaced during install, with optional prior backup.
#[derive(Debug)]
struct BinaryReplacement {
    dest: PathBuf,
    /// Prior destination contents, if any, retained under a unique same-directory backup.
    backup: Option<PathBuf>,
}

/// Unique same-directory backup path for an existing destination.
fn backup_path_for(dest: &Path, nonce: u128) -> Result<PathBuf> {
    let parent = dest
        .parent()
        .ok_or_else(|| anyhow::anyhow!("destination has no parent: {}", dest.display()))?;
    let file_name = dest
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("destination has no file name: {}", dest.display()))?;
    Ok(parent.join(format!(
        ".{}.prev-{}-{}",
        file_name.to_string_lossy(),
        std::process::id(),
        nonce
    )))
}

fn install_nonce() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Stage `src` fully to a same-directory temp (with execute bits), copy any live
/// destination to a backup **without removing it**, then atomically rename the
/// temp over the destination. The live binary never disappears before rename.
fn install_binary_with_backup(src: &Path, dest: &Path) -> Result<BinaryReplacement> {
    if !src.is_file() {
        bail!("build artifact missing: {}", src.display());
    }
    let parent = dest
        .parent()
        .ok_or_else(|| anyhow::anyhow!("destination has no parent: {}", dest.display()))?;
    let file_name = dest
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("destination has no file name: {}", dest.display()))?;

    let tmp = parent.join(format!(
        ".{}.install-{}.tmp",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    let _ = fs::remove_file(&tmp);

    // 1) Fully stage the new binary beside the destination first.
    if let Err(err) = fs::copy(src, &tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(err)
            .with_context(|| format!("failed to stage {} -> {}", src.display(), tmp.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(err) = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)) {
            let _ = fs::remove_file(&tmp);
            return Err(err).with_context(|| {
                format!("failed to set execute permissions on {}", tmp.display())
            });
        }
    }

    // 2) Preserve an existing destination by copying it to backup while it stays live.
    let backup = if dest.is_file() {
        let backup = backup_path_for(dest, install_nonce())?;
        if let Err(err) = fs::copy(dest, &backup) {
            let _ = fs::remove_file(&tmp);
            let _ = fs::remove_file(&backup);
            return Err(err).with_context(|| {
                format!(
                    "failed to backup existing {} -> {}",
                    dest.display(),
                    backup.display()
                )
            });
        }
        Some(backup)
    } else {
        None
    };

    // 3) Atomic same-directory rename over the live destination.
    if let Err(err) = fs::rename(&tmp, dest) {
        let _ = fs::remove_file(&tmp);
        // Destination is still the prior live binary; drop the unused backup copy.
        if let Some(backup) = &backup {
            let _ = fs::remove_file(backup);
        }
        return Err(err)
            .with_context(|| format!("failed to install {} -> {}", src.display(), dest.display()));
    }

    Ok(BinaryReplacement {
        dest: dest.to_path_buf(),
        backup,
    })
}

/// Snapshot of `config.toml` before an explicit profile may rewrite it.
#[derive(Debug, Clone)]
struct ConfigTomlSnapshot {
    path: PathBuf,
    /// `None` means the file was absent before the profile write.
    prior: Option<Vec<u8>>,
}

fn snapshot_config_toml(path: &Path) -> Result<ConfigTomlSnapshot> {
    let prior = if path.is_file() {
        Some(
            fs::read(path)
                .with_context(|| format!("failed to snapshot config {}", path.display()))?,
        )
    } else {
        None
    };
    Ok(ConfigTomlSnapshot {
        path: path.to_path_buf(),
        prior,
    })
}

fn restore_config_toml(snap: &ConfigTomlSnapshot) {
    match &snap.prior {
        Some(bytes) => {
            if let Err(err) = fs::write(&snap.path, bytes) {
                eprintln!(
                    "WARNING: failed to restore config {}: {err}",
                    snap.path.display()
                );
            }
        }
        None => {
            if snap.path.exists() {
                if let Err(err) = fs::remove_file(&snap.path) {
                    eprintln!(
                        "WARNING: failed to remove profile-written config {}: {err}",
                        snap.path.display()
                    );
                }
            }
        }
    }
}

fn rollback_replacements(replacements: &[BinaryReplacement]) {
    for rep in replacements.iter().rev() {
        match &rep.backup {
            Some(backup) => {
                if let Err(err) = fs::rename(backup, &rep.dest) {
                    eprintln!(
                        "WARNING: failed to restore {} from {}: {err}",
                        rep.dest.display(),
                        backup.display()
                    );
                }
            }
            None => {
                if rep.dest.exists() {
                    if let Err(err) = fs::remove_file(&rep.dest) {
                        eprintln!(
                            "WARNING: failed to remove newly installed {}: {err}",
                            rep.dest.display()
                        );
                    }
                }
            }
        }
    }
}

/// Drop successful-install backups; cleanup failure is warning-only.
fn cleanup_backups(replacements: &[BinaryReplacement]) {
    for rep in replacements {
        if let Some(backup) = &rep.backup {
            if let Err(err) = fs::remove_file(backup) {
                eprintln!(
                    "WARNING: failed to remove install backup {}: {err}",
                    backup.display()
                );
            }
        }
    }
}

fn write_install_json(root: &Path, value: &serde_json::Value) -> Result<()> {
    fs::create_dir_all(root).with_context(|| format!("failed to create {}", root.display()))?;
    let dest = root.join("install.json");
    let tmp = root.join(format!(".install.json.{}.tmp", std::process::id()));
    let body = serde_json::to_vec_pretty(value).context("failed to serialize install.json")?;
    if let Err(err) = fs::write(&tmp, body) {
        let _ = fs::remove_file(&tmp);
        return Err(err).with_context(|| format!("failed to write {}", tmp.display()));
    }
    if let Err(err) = fs::rename(&tmp, &dest) {
        let _ = fs::remove_file(&tmp);
        return Err(err).with_context(|| format!("failed to install {}", dest.display()));
    }
    Ok(())
}

fn path_on_path(bin: &Path) -> bool {
    env::var_os("PATH")
        .map(|path| env::split_paths(&path).any(|p| p == bin))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{
        backup_path_for, cleanup_backups, continue_decision, ensure_rocm_complete,
        install_binary_with_backup, metadata_ref_from_parts, parse_gpu_arches,
        pin_cargo_to_source_target, prompt_line_from_read, restore_config_toml,
        rollback_replacements, selection_from_prompt_read, snapshot_config_toml, usable_rocm_roots,
        BinaryReplacement, ContinueDecision, PromptRead,
    };
    use std::{
        env, fs,
        path::{Path, PathBuf},
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn parse_gpu_arches_lowercases_and_accepts_hex() {
        let out = parse_gpu_arches("gfx1201\ngfx90A\ngfx1151\n");
        assert_eq!(out, vec!["gfx1201", "gfx90a", "gfx1151"]);
    }

    #[test]
    fn parse_gpu_arches_excludes_gfx000_and_invalid() {
        let out = parse_gpu_arches(
            "gfx000\n\
             not-an-arch\n\
             gfx\n\
             gfx12zz\n\
             gfx1201\n\
             gfx000\n",
        );
        assert_eq!(out, vec!["gfx1201"]);
    }

    #[test]
    fn parse_gpu_arches_dedups_preserving_order() {
        let out = parse_gpu_arches("gfx1201\ngfx1100\ngfx1201\ngfx1100\ngfx1151\n");
        assert_eq!(out, vec!["gfx1201", "gfx1100", "gfx1151"]);
    }

    /// `/opt/rocm/core`, `core-7`, and `core-7.14` often share one real tree via
    /// symlinks; discovery must collapse them to a single canonical root.
    /// A compiler-only ROCm root must be rejected up front, naming the packages
    /// that supply what is missing. Before this the install ran three cargo
    /// builds and a 38-kernel precompile first, then failed with clang's
    /// "'hip/hip_runtime.h' file not found" — which names neither cause nor fix.
    #[test]
    fn compiler_only_rocm_root_is_rejected_before_any_build() {
        let tmp = env::temp_dir().join(format!("hipfire-setup-rocm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("bin")).unwrap();
        fs::write(tmp.join("bin").join("hipcc"), b"#!/bin/sh\n").unwrap();

        let dirs = hipfire_config::rocm::HIP_RUNTIME_DIRS;
        let libs = hipfire_config::rocm::HIP_RUNTIME_LIBRARIES;

        let err = ensure_rocm_complete(&tmp).unwrap_err().to_string();
        assert!(err.contains("hip/hip_runtime.h"), "{err}");
        assert!(err.contains("--rocm-root"), "{err}");
        // Guidance is present but its wording is host-dependent, so assert that
        // some install advice was emitted rather than a specific distro's.
        assert!(
            hipfire_config::rocm::install_guidance()
                .iter()
                .all(|line| err.contains(line.as_str())),
            "{err}"
        );

        // Headers alone are enough to proceed. The runtime is resolved at load
        // time across all roots and then the loader, so a root without it must
        // NOT block: the reporting box loads libamdhip64 fine while `ldconfig`
        // knows nothing about it.
        fs::create_dir_all(tmp.join("include").join("hip")).unwrap();
        fs::write(tmp.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        assert!(
            ensure_rocm_complete(&tmp).is_ok(),
            "a missing runtime must warn, not block"
        );

        // And a fully populated root is obviously fine.
        fs::create_dir_all(tmp.join(dirs[0])).unwrap();
        fs::write(tmp.join(dirs[0]).join(libs[0]), b"").unwrap();
        assert!(ensure_rocm_complete(&tmp).is_ok());

        fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn usable_rocm_roots_dedups_symlink_aliases_to_one_canonical() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-setup-rocm-alias-{}-{}",
            std::process::id(),
            nonce
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("core-7.14/bin")).unwrap();
        fs::write(root.join("core-7.14/bin/hipcc"), b"#!/bin/sh\n").unwrap();

        let real = root.join("core-7.14");
        let alias_core = root.join("core");
        let alias_core7 = root.join("core-7");
        std::os::unix::fs::symlink(&real, &alias_core).unwrap();
        std::os::unix::fs::symlink(&real, &alias_core7).unwrap();

        let got = usable_rocm_roots([alias_core.clone(), real.clone(), alias_core7.clone()]);
        let expected = fs::canonicalize(&real).unwrap();
        assert_eq!(got, vec![expected]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn metadata_ref_prefers_reference_then_branch_tag_commit() {
        assert_eq!(
            metadata_ref_from_parts(Some("beta"), Some("master"), Some("v1"), Some("abc")),
            "beta"
        );
        assert_eq!(
            metadata_ref_from_parts(None, Some("master"), Some("v1"), Some("abc")),
            "master"
        );
        assert_eq!(
            metadata_ref_from_parts(None, None, Some("v1"), Some("abc")),
            "v1"
        );
        assert_eq!(
            metadata_ref_from_parts(None, None, None, Some("abc")),
            "abc"
        );
        assert_eq!(metadata_ref_from_parts(None, None, None, None), "local");
    }

    #[test]
    fn backup_path_for_is_same_directory_unique() {
        let dest = Path::new("/tmp/hipfire-bin/daemon");
        let backup = backup_path_for(dest, 42).unwrap();
        assert_eq!(backup.parent(), dest.parent());
        let name = backup.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with(".daemon.prev-"));
        assert!(name.contains("-42"));
        assert_ne!(backup, dest);
    }

    #[test]
    fn prompt_line_from_read_maps_eof_and_line() {
        assert_eq!(prompt_line_from_read(0, ""), PromptRead::Eof);
        assert_eq!(
            prompt_line_from_read(2, "y\n"),
            PromptRead::Line("y\n".into())
        );
    }

    #[test]
    fn continue_decision_eof_and_no_cancel() {
        assert_eq!(continue_decision(PromptRead::Eof), ContinueDecision::Cancel);
        assert_eq!(
            continue_decision(PromptRead::Line("n\n".into())),
            ContinueDecision::Cancel
        );
        assert_eq!(
            continue_decision(PromptRead::Line("no\n".into())),
            ContinueDecision::Cancel
        );
        assert_eq!(
            continue_decision(PromptRead::Line("\n".into())),
            ContinueDecision::Proceed
        );
        assert_eq!(
            continue_decision(PromptRead::Line("y\n".into())),
            ContinueDecision::Proceed
        );
    }

    #[test]
    fn selection_eof_is_error_invalid_retries() {
        let err = selection_from_prompt_read(PromptRead::Eof, 3).unwrap_err();
        assert!(err.to_string().contains("EOF"));
        assert_eq!(
            selection_from_prompt_read(PromptRead::Line("2\n".into()), 3).unwrap(),
            Some(1)
        );
        assert_eq!(
            selection_from_prompt_read(PromptRead::Line("9\n".into()), 3).unwrap(),
            None
        );
        assert_eq!(
            selection_from_prompt_read(PromptRead::Line("x\n".into()), 3).unwrap(),
            None
        );
    }

    #[test]
    fn pin_cargo_forces_source_target_dir() {
        let source = Path::new("/tmp/hipfire-src-example");
        let mut cmd = Command::new("cargo");
        pin_cargo_to_source_target(&mut cmd, source);
        let debug = format!("{cmd:?}");
        assert!(
            debug.contains("--target-dir") && debug.contains("hipfire-src-example/target"),
            "expected pinned target-dir in {debug}"
        );
        // Env is set on the Command; Debug may or may not show it depending on
        // std version — flag coverage is the hard requirement.
    }

    #[test]
    fn config_snapshot_restore_rewrites_and_removes() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-setup-cfg-{}-{}",
            std::process::id(),
            nonce
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let path = root.join("config.toml");

        // Absent -> profile wrote file -> restore removes it.
        let snap_absent = snapshot_config_toml(&path).unwrap();
        assert!(snap_absent.prior.is_none());
        fs::write(&path, b"profile-new").unwrap();
        restore_config_toml(&snap_absent);
        assert!(!path.exists());

        // Prior contents restored after overwrite.
        fs::write(&path, b"prior-cfg").unwrap();
        let snap = snapshot_config_toml(&path).unwrap();
        fs::write(&path, b"mutated").unwrap();
        restore_config_toml(&snap);
        assert_eq!(fs::read(&path).unwrap(), b"prior-cfg");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn install_backup_is_copy_while_dest_becomes_new() {
        // Live-destination semantics: prior bytes survive in backup, dest ends as
        // the new binary; backup is an independent file (copy), not the only copy
        // of the prior live path.
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-setup-live-{}-{}",
            std::process::id(),
            nonce
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let src = root.join("src");
        let dest = root.join("daemon");
        fs::write(&src, b"new-bytes").unwrap();
        fs::write(&dest, b"live-prior").unwrap();

        let rep = install_binary_with_backup(&src, &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"new-bytes");
        let backup = rep.backup.as_ref().expect("prior dest must produce backup");
        assert!(backup.is_file());
        assert_eq!(fs::read(backup).unwrap(), b"live-prior");
        // Independent paths: dest and backup both present until cleanup.
        assert_ne!(backup.as_path(), dest.as_path());
        assert!(dest.is_file());

        // Rollback restores prior live content and keeps backup consumed.
        rollback_replacements(std::slice::from_ref(&rep));
        assert_eq!(fs::read(&dest).unwrap(), b"live-prior");
        assert!(!backup.exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rollback_restores_prior_and_removes_new() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-setup-rollback-{}-{}",
            std::process::id(),
            nonce
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let prior = root.join("daemon");
        fs::write(&prior, b"old-daemon").unwrap();
        let new_only = root.join("hipfire-tui");
        // Simulate install: prior moved to backup, new written to dest; new-only has no backup.
        let backup = backup_path_for(&prior, nonce).unwrap();
        fs::rename(&prior, &backup).unwrap();
        fs::write(&prior, b"new-daemon").unwrap();
        fs::write(&new_only, b"new-tui").unwrap();

        let replacements = vec![
            BinaryReplacement {
                dest: prior.clone(),
                backup: Some(backup.clone()),
            },
            BinaryReplacement {
                dest: new_only.clone(),
                backup: None,
            },
        ];
        rollback_replacements(&replacements);

        assert_eq!(fs::read(&prior).unwrap(), b"old-daemon");
        assert!(!new_only.exists());
        assert!(!backup.exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn install_with_backup_then_cleanup_leaves_new_dest() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-setup-install-{}-{}",
            std::process::id(),
            nonce
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let src_old = root.join("src-old");
        let src_new = root.join("src-new");
        let dest = root.join("daemon");
        fs::write(&src_old, b"v1").unwrap();
        fs::write(&src_new, b"v2").unwrap();
        fs::write(&dest, b"v1-installed").unwrap();

        let rep = install_binary_with_backup(&src_new, &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"v2");
        assert!(rep.backup.as_ref().is_some_and(|b| b.is_file()));
        assert_eq!(
            fs::read(rep.backup.as_ref().unwrap()).unwrap(),
            b"v1-installed"
        );

        cleanup_backups(std::slice::from_ref(&rep));
        assert_eq!(fs::read(&dest).unwrap(), b"v2");
        assert!(rep.backup.as_ref().is_some_and(|b| !b.exists()));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn install_new_dest_rollback_removes_file() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-setup-newonly-{}-{}",
            std::process::id(),
            nonce
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let src: PathBuf = root.join("src");
        let dest = root.join("daemon");
        fs::write(&src, b"fresh").unwrap();
        let rep = install_binary_with_backup(&src, &dest).unwrap();
        assert!(rep.backup.is_none());
        assert_eq!(fs::read(&dest).unwrap(), b"fresh");
        rollback_replacements(std::slice::from_ref(&rep));
        assert!(!dest.exists());

        let _ = fs::remove_dir_all(&root);
    }
}

/// Test-only helper mirroring [`metadata_ref`] without requiring a full `SetupArgs`.
#[cfg(test)]
fn metadata_ref_from_parts(
    reference: Option<&str>,
    branch: Option<&str>,
    tag: Option<&str>,
    commit: Option<&str>,
) -> String {
    reference
        .or(branch)
        .or(tag)
        .or(commit)
        .map(str::to_owned)
        .unwrap_or_else(|| "local".to_owned())
}
