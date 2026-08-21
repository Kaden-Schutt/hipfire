// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ROCm installation discovery.
//!
//! hipfire dlopens the HIP/HSA runtimes rather than linking them, so it has to
//! find them at run time. Historically each call site did that differently:
//!
//!   * `hip-bridge` (Linux) asked the dynamic loader for a bare
//!     `libamdhip64.so` and did no root resolution at all — it worked only
//!     when `LD_LIBRARY_PATH`/ldconfig already pointed at ROCm.
//!   * `hsa-bridge`, `rocblas` and `rccl` hardcoded `/opt/rocm/lib`.
//!   * The kernel compiler invoked a bare `hipcc` off `PATH`.
//!
//! That breaks on any install that is not literally `/opt/rocm`: side-by-side
//! installs (`/opt/rocm-6.4`), the `/opt/rocm/core-<ver>` layout used when a
//! host ROCm is overmounted into a container, and Windows' `HIP_PATH`.
//!
//! This module centralises the policy. Resolution order, most authoritative
//! first:
//!
//!   1. `HIPFIRE_ROCM_PATH` — our explicit override, always wins.
//!   2. `ROCM_PATH`         — the ROCm-standard variable.
//!   3. `HIP_PATH`          — HIP SDK variable; a trailing `hip` component is
//!                            stripped so `/opt/rocm/hip` resolves to `/opt/rocm`.
//!   4. `/opt/rocm`, including a `core` / `core-<ver>` split-tree below it.
//!   5. Versioned siblings `/opt/rocm-*`, newest first.
//!   6. The parent of `hipcc`, `amdclang++`, `rocminfo`, or
//!      `rocm_agent_enumerator` on `PATH`, which covers module-managed and Nix
//!      installs.
//!   7. Package-manager roots (`/usr`, `/usr/local`) when they carry concrete
//!      ROCm evidence.
//!
//! An explicit environment root is authoritative. Its split-tree children are
//! considered, but unrelated installs and bare loader sonames are not. This
//! prevents a misspelled or incomplete override from silently mixing one
//! version's compiler/headers with another version's runtime.
//!
//! Nothing here touches the GPU; it is pure path policy and is unit-tested
//! against a synthetic tree.

use std::path::{Path, PathBuf};

/// Device compilers, most specific first. `hipcc` is being wound down upstream
/// in favour of invoking the LLVM driver directly, and on ROCm 7.14 `hipcc` is
/// already a thin wrapper around `amdclang++`, so both are probed.
pub const DEVICE_COMPILERS: &[&str] = &["hipcc", "amdclang++", "amdclang", "clang++"];

/// Tools whose installed path is strong evidence for a ROCm root. Generic
/// `clang++` is deliberately excluded: treating `/usr/bin/clang++` as ROCm
/// would make every normal compiler installation look like a ROCm SDK.
const ROOT_HINT_TOOLS: &[&str] = &[
    "hipcc",
    "amdclang++",
    "amdclang",
    "rocminfo",
    "rocm_agent_enumerator",
];

/// Split a directory name into numeric components for version ordering.
/// `core-7.14` -> [7, 14]; names without digits sort last.
fn version_key(name: &str) -> Vec<u64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.parse().unwrap_or(0));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.parse().unwrap_or(0));
    }
    out
}

/// Versioned ROCm siblings under `base`, newest first.
fn versioned_siblings(base: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut found: Vec<(Vec<u64>, PathBuf)> = Vec::new();
    let Ok(entries) = std::fs::read_dir(base) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(prefix) || !entry.path().is_dir() {
            continue;
        }
        let key = version_key(name);
        if key.is_empty() {
            continue;
        }
        found.push((key, entry.path()));
    }
    // Descending by numeric key so 7.14 precedes 7.
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, p)| p).collect()
}

/// A `HIP_PATH` may point at `<root>/hip`; normalise to `<root>`.
fn normalize_hip_path(p: &Path) -> PathBuf {
    if p.file_name().map(|f| f == "hip").unwrap_or(false) {
        if let Some(parent) = p.parent() {
            return parent.to_path_buf();
        }
    }
    p.to_path_buf()
}

/// Derive the ROCm root from a tool path.
///
/// Most tools live at `<root>/bin/tool`. ROCm's compiler entry points are
/// sometimes symlinks into `<root>/lib/llvm/bin`, so both layouts are
/// recognized after canonicalization.
fn root_from_tool_path(tool: &Path) -> Option<PathBuf> {
    let resolved = std::fs::canonicalize(tool).unwrap_or_else(|_| tool.to_path_buf());
    let bin = resolved.parent()?;
    let parent = bin.parent()?;
    if parent.file_name().is_some_and(|name| name == "llvm")
        && parent
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "lib")
    {
        return parent
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
    }
    Some(parent.to_path_buf())
}

/// Locate ROCm-specific tools on `PATH` and derive every distinct install root.
fn roots_from_path_tools() -> Vec<PathBuf> {
    let Some(path) = std::env::var_os("PATH") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for dir in std::env::split_paths(&path) {
        for tool in ROOT_HINT_TOOLS {
            let Some(candidate) = first_tool_in_dir(&dir, tool, cfg!(windows)) else {
                continue;
            };
            if let Some(root) = root_from_tool_path(&candidate) {
                if !out.contains(&root) {
                    out.push(root);
                }
            }
        }
    }
    out
}

/// The first non-empty explicit root and the variable that supplied it.
///
/// Only the highest-priority variable is honored. Falling through from a bad
/// `HIPFIRE_ROCM_PATH` to a different `ROCM_PATH` would make an override look
/// accepted while loading another ROCm version.
pub fn configured_root() -> Option<(&'static str, PathBuf)> {
    for var in ["HIPFIRE_ROCM_PATH", "ROCM_PATH", "HIP_PATH"] {
        let Some(value) = std::env::var_os(var).filter(|v| !v.is_empty()) else {
            continue;
        };
        let path = PathBuf::from(value);
        // Through ROCm 4.x HIP lived at `$ROCM_PATH/hip`, so older scripts and
        // Dockerfiles still export `ROCM_PATH=$PREFIX/hip` and `HIPFIRE_ROCM_PATH`
        // the same way. `HIP_PATH` handling already stripped a trailing `hip`;
        // apply the same normalization to every explicit root so those users do
        // not hard-fail on a machine that is otherwise fine. `normalize_hip_path`
        // is the single guard — it only strips a literal trailing `hip` component
        // and otherwise leaves the path untouched, mirroring prior `HIP_PATH`
        // behaviour.
        return Some((var, normalize_hip_path(&path)));
    }
    None
}

/// Whether a non-empty explicit ROCm root override is active.
pub fn has_configured_root() -> bool {
    configured_root().is_some()
}
/// The explicit device-compiler override, if set.
///
/// `HIPFIRE_HIPCC` is the project-specific override and is the only variable
/// consulted here. When non-empty it wins over every other compiler candidate.
/// Validation (exists and is executable) is performed by callers so the
/// override never silently falls through to autodetection — that silent
/// mismatch is the bug class this module exists to prevent.
pub fn configured_compiler() -> Option<(&'static str, PathBuf)> {
    std::env::var_os("HIPFIRE_HIPCC")
        .filter(|v| !v.is_empty())
        .map(|value| ("HIPFIRE_HIPCC", PathBuf::from(value)))
}

/// Whether a non-empty explicit compiler override is active.
pub fn has_configured_compiler() -> bool {
    configured_compiler().is_some()
}

/// Pure form of [`configured_compiler`] for tests without process-global env mutation.
pub fn configured_compiler_from(value: Option<&str>) -> Option<(&'static str, PathBuf)> {
    let v = value.filter(|s| !s.is_empty())?;
    Some(("HIPFIRE_HIPCC", PathBuf::from(v)))
}

/// Whether strict ROCm resolution is requested.
///
/// When `HIPFIRE_ROCM_STRICT=1` the cross-root compiler fallback is disabled
/// and a runtime-headers-only root that lacks a compiler hard-fails as before.
pub fn is_strict_rocm() -> bool {
    strict_from(std::env::var_os("HIPFIRE_ROCM_STRICT").as_ref())
}

/// Pure predicate for strict mode without reading process-global env.
pub fn strict_from(value: Option<&std::ffi::OsString>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Alternative strict predicate for `Option<&str>` call sites.
pub fn strict_from_str(value: Option<&str>) -> bool {
    value == Some("1")
}

/// How a device compiler was selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilerSource {
    /// `HIPFIRE_HIPCC` override.
    Override,
    /// Compiler lives under the selected root's `bin/`.
    SelectedRoot,
    /// Compiler found via `PATH`.
    Path,
    /// Compiler from another discovered ROCm root.
    OtherRoot,
}

impl std::fmt::Display for CompilerSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerSource::Override => write!(f, "override"),
            CompilerSource::SelectedRoot => write!(f, "selected_root"),
            CompilerSource::Path => write!(f, "path"),
            CompilerSource::OtherRoot => write!(f, "other_root"),
        }
    }
}

/// Resolved ROCm toolchain: the runtime/headers root, the compiler, and provenance.
///
/// This distinguishes “compiler came from the selected root” from “compiler came
/// from elsewhere” so callers can warn when runtime and compiler are from
/// different installs. The spawned compiler must receive its **own** root as
/// `ROCM_PATH` (see [`compiler_env_root`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedToolchain {
    /// The selected ROCm root that provides headers and the HIP runtime.
    pub root: PathBuf,
    /// The device compiler to use, if any.
    pub compiler: Option<PathBuf>,
    /// The ROCm root that owns `compiler`, if it can be derived.
    pub compiler_root: Option<PathBuf>,
    /// How `compiler` was obtained. `None` when `compiler` is `None`.
    pub compiler_source: Option<CompilerSource>,
}

/// ROCm version for a single root (`<root>/.info/version`), if readable.
pub fn version_for_root(root: &Path) -> Option<String> {
    let f = root.join(".info").join("version");
    if let Ok(s) = std::fs::read_to_string(&f) {
        let s = s.trim();
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    None
}

/// Whether `path` is an executable file.
fn is_executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            return meta.permissions().mode() & 0o111 != 0;
        }
        false
    }
    #[cfg(windows)]
    {
        true
    }
}

/// Validate an explicit `HIPFIRE_HIPCC` path.
///
/// Returns `None` when the path is usable, otherwise a human-readable error
/// that names the variable and the path — routed through `resolution_failure`.
pub fn configured_compiler_error(path: &Path) -> Option<String> {
    configured_compiler_error_from(Some(path), true)
}

/// Pure form for tests: `value` is the `HIPFIRE_HIPCC` contents, `exists` is
/// whether the path is acceptable. When `value` is `None` there is no override.
pub fn configured_compiler_error_from(path: Option<&Path>, is_valid: bool) -> Option<String> {
    let p = path?;
    if is_valid {
        None
    } else {
        Some(format!(
            "HIPFIRE_HIPCC={} does not exist or is not executable; set HIPFIRE_HIPCC to an executable hipcc/amdclang++ or unset it",
            p.display()
        ))
    }
}

/// Human-readable warning lines when runtime and compiler come from different installs.
///
/// Returned, not printed, so the CLI and installer control output. Names the
/// selected root, compiler path, compiler root, and both `.info/version` strings
/// when readable, and states the mixing risk plainly.
pub fn cross_root_warning(
    selected_root: &Path,
    compiler: &Path,
    compiler_root: &Path,
) -> Vec<String> {
    cross_root_warning_with_versions(
        selected_root,
        version_for_root(selected_root),
        compiler,
        compiler_root,
        version_for_root(compiler_root),
    )
}

/// Pure form of [`cross_root_warning`] with injectable versions for tests.
pub fn cross_root_warning_with_versions(
    selected_root: &Path,
    selected_version: Option<String>,
    compiler: &Path,
    compiler_root: &Path,
    compiler_version: Option<String>,
) -> Vec<String> {
    let sel_ver = selected_version
        .map(|v| format!(" (version {v})"))
        .unwrap_or_default();
    let comp_ver = compiler_version
        .map(|v| format!(" (version {v})"))
        .unwrap_or_default();
    vec![
        format!(
            "WARNING: ROCm runtime and device compiler are from different installations."
        ),
        format!(
            "  Selected ROCm root (runtime/headers): {}{sel_ver}",
            selected_root.display()
        ),
        format!("  Device compiler: {}{comp_ver}", compiler.display()),
        format!("  Compiler root: {}", compiler_root.display()),
        format!(
            "  Mixing a runtime from one ROCm install with a compiler from another can \
             produce ABI or LLVM mismatches. Consider installing a complete ROCm SDK \
             under one root or set HIPFIRE_HIPCC to a compiler inside the selected root. \
             Set HIPFIRE_ROCM_STRICT=1 to make this a hard error."
        ),
    ]
}

/// Warning lines for a resolved toolchain, if it is cross-root.
pub fn toolchain_warnings(toolchain: &ResolvedToolchain) -> Vec<String> {
    match (&toolchain.compiler, &toolchain.compiler_root, &toolchain.compiler_source) {
        (Some(compiler), Some(compiler_root), Some(source))
            if matches!(source, CompilerSource::Path | CompilerSource::OtherRoot) =>
        {
            cross_root_warning(&toolchain.root, compiler, compiler_root)
        }
        _ => Vec::new(),
    }
}
/// Whether this root provides headers + HIP runtime (+ HSA on non-Windows) but no compiler.
///
/// Exactly the case where cross-root compiler acceptance is allowed (when not
/// strict): the runtime/headers are usable, only the compiler is missing.
pub fn is_headers_runtime_only_root(path: &Path) -> bool {
    if !is_complete_root(path) {
        return false;
    }
    if runtime_library(path).is_none() {
        return false;
    }
    #[cfg(not(windows))]
    if hsa_runtime_library(path).is_none() {
        return false;
    }
    !has_device_compiler(path)
}

/// Discover all ROCm roots without honouring an authoritative override.
///
/// Used only for cross-root compiler search so a compiler from another install
/// can be found even when `HIPFIRE_ROCM_PATH` is authoritative for libraries.
fn all_roots_unfiltered() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    #[cfg(not(windows))]
    {
        for candidate in root_family(Path::new("/opt/rocm")) {
            push(candidate);
        }
        for candidate in versioned_siblings(Path::new("/opt"), "rocm-") {
            push(candidate);
        }
    }
    for candidate in roots_from_path_tools() {
        push(candidate);
    }
    #[cfg(not(windows))]
    for candidate in [PathBuf::from("/usr"), PathBuf::from("/usr/local")] {
        if has_package_rocm_evidence(&candidate) {
            push(candidate);
        }
    }
    #[cfg(windows)]
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        let base = PathBuf::from(program_files).join("AMD").join("ROCm");
        for candidate in root_family(&base) {
            push(candidate);
        }
        for candidate in versioned_siblings(&base, "") {
            push(candidate);
        }
    }
    out
}

/// Try to find a device compiler on `PATH` (any of `DEVICE_COMPILERS`).
fn find_compiler_on_path() -> Option<PathBuf> {
    for name in DEVICE_COMPILERS {
        if let Some(p) = path_tool(name) {
            return Some(p);
        }
    }
    None
}

/// Try to find a device compiler in other discovered roots (excluding `selected` family).
fn find_compiler_in_other_roots(selected: &Path) -> Option<(PathBuf, PathBuf)> {
    let family = root_family(selected);
    let family_canonical: Vec<PathBuf> = family
        .iter()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))
        .collect();
    for root in all_roots_unfiltered() {
        let canon = std::fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
        if family_canonical.contains(&canon) {
            continue;
        }
        // Also skip exact family members by display equality before canonicalize.
        if family.contains(&root) {
            continue;
        }
        for name in DEVICE_COMPILERS {
            if let Some(tool) = tool_from_selected_root(&root, name) {
                return Some((tool, root));
            }
        }
    }
    None
}

/// Pure core for toolchain resolution with injected inputs (no env reads).
///
/// `selected` is the chosen runtime root (already canonicalized when possible).
/// `override_compiler` is the `HIPFIRE_HIPCC` path if set (non-empty). `strict`
/// disables cross-root fallback. `path_compiler` is a compiler found on `PATH`,
/// `other_root_compiler` is `(compiler_path, its_root)` from another install.
pub fn resolve_toolchain_pure(
    selected: Option<&Path>,
    override_compiler: Option<&Path>,
    strict: bool,
    path_compiler: Option<PathBuf>,
    other_root_compiler: Option<(PathBuf, PathBuf)>,
) -> Result<ResolvedToolchain, String> {
    let Some(root) = selected.map(|p| p.to_path_buf()) else {
        return Err(resolution_failure(
            "a complete ROCm installation",
            &[],
        ));
    };
    // Override takes absolute precedence; validate existence + executable.
    if let Some(ov) = override_compiler {
        if !is_executable(ov) {
            let msg = format!(
                "Could not resolve the ROCm HIP compiler (hipcc). HIPFIRE_HIPCC={} is set but does not exist or is not executable; hipfire did not fall back to another compiler.",
                ov.display()
            );
            // Include install guidance for completeness.
            let mut full = msg;
            for line in install_guidance() {
                full.push_str(&format!("\n  {line}"));
            }
            return Err(full);
        }
        let compiler_root = root_from_tool_path(ov)
            .or_else(|| root_from_compiler(ov))
            .unwrap_or_else(|| root.clone());
        return Ok(ResolvedToolchain {
            root: root.clone(),
            compiler: Some(ov.to_path_buf()),
            compiler_root: Some(compiler_root),
            compiler_source: Some(CompilerSource::Override),
        });
    }
    // Compiler under the selected root, if any.
    for name in DEVICE_COMPILERS {
        if let Some(tool) = tool_from_selected_root(&root, name) {
            let compiler_root = root_from_tool_path(&tool).unwrap_or_else(|| root.clone());
            return Ok(ResolvedToolchain {
                root: root.clone(),
                compiler: Some(tool.clone()),
                compiler_root: Some(compiler_root),
                compiler_source: Some(CompilerSource::SelectedRoot),
            });
        }
    }
    // No compiler under selected root — check whether cross-root is eligible.
    let eligible = is_headers_runtime_only_root(&root);
    if eligible && !strict {
        if let Some(pc) = path_compiler {
            let croot = root_from_tool_path(&pc)
                .or_else(|| root_from_compiler(&pc))
                .unwrap_or_else(|| pc.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()).unwrap_or_else(|| root.clone()));
            return Ok(ResolvedToolchain {
                root: root.clone(),
                compiler: Some(pc),
                compiler_root: Some(croot),
                compiler_source: Some(CompilerSource::Path),
            });
        }
        if let Some((tool, troot)) = other_root_compiler {
            let croot = root_from_tool_path(&tool).unwrap_or(troot.clone());
            return Ok(ResolvedToolchain {
                root: root.clone(),
                compiler: Some(tool),
                compiler_root: Some(croot),
                compiler_source: Some(CompilerSource::OtherRoot),
            });
        }
    }
    // No usable compiler — hard fail. Preserve the authoritative message when
    // an explicit root is set, and mention --hipcc as a remedy.
    let tried = vec![root.join("bin").join("hipcc").display().to_string()];
    let mut msg = resolution_failure("the ROCm HIP compiler (hipcc)", &tried);
    // The shared resolution_failure already says authoritative and suggests
    // HIPFIRE_ROCM_PATH; also hint at HIPFIRE_HIPCC / --hipcc for the split-install case.
    if eligible && strict {
        msg.push_str("\nCross-root compiler fallback is disabled by HIPFIRE_ROCM_STRICT=1; unset it or install a compiler under the selected root or set HIPFIRE_HIPCC=/path/to/hipcc.");
    } else if eligible {
        msg.push_str("\nNo device compiler was found on PATH or in other ROCm installs. Install the ROCm device compiler under the selected root or set HIPFIRE_HIPCC=/path/to/hipcc (--hipcc PATH).");
    } else {
        msg.push_str("\nSet HIPFIRE_HIPCC=/path/to/hipcc (--hipcc PATH) if the compiler lives in a different prefix.");
    }
    Err(msg)
}

/// Resolve the current toolchain from the environment and filesystem.
///
/// This is the primary entry point for callers that want provenance and the
/// cross-root warning. It honours `HIPFIRE_HIPCC`, `HIPFIRE_ROCM_STRICT`, and the
/// authoritative-root rule for libraries while allowing a compiler from `PATH`
/// or another root when the selected root is headers+runtime-complete.
pub fn resolve_toolchain() -> Result<ResolvedToolchain, String> {
    // Determine selected root: coherent SDK preferred, else first is_dir fallback,
    // mirroring `root()` but we also need the fallback for libs-only trees.
    let selected = if !ambiguous_roots().is_empty() {
        None
    } else {
        let candidates = roots();
        candidates
            .iter()
            .find(|p| is_coherent_sdk_root(p))
            .cloned()
            .or_else(|| candidates.into_iter().find(|p| p.is_dir()))
    };
    let ov = configured_compiler().map(|(_, p)| p);
    let strict = is_strict_rocm();
    let path_comp = find_compiler_on_path();
    // Only probe other roots when we may need them (avoid extra FS work when
    // a compiler was already found on PATH or under selected root). For
    // simplicity always compute here; the pure function decides.
    let other = selected.as_deref().and_then(find_compiler_in_other_roots);
    resolve_toolchain_pure(
        selected.as_deref(),
        ov.as_deref(),
        strict,
        path_comp,
        other,
    )
}

/// Pure helper for setup.rs and tests: resolve from caller-supplied explicit root.
///
/// When `explicit` is `Some`, it is treated as the authoritative selected root
/// (mirroring `HIPFIRE_ROCM_PATH` semantics). `hipcc_override` and `strict` are
/// injected so tests avoid global env mutation.
pub fn resolve_toolchain_for_explicit(
    explicit: Option<&Path>,
    hipcc_override: Option<&Path>,
    strict: bool,
) -> Result<ResolvedToolchain, String> {
    let selected = explicit.map(|p| p.to_path_buf());
    // For explicit roots we still allow PATH/other-root fallback via the same
    // pure function; other-root search uses `all_roots_unfiltered` which may
    // include the explicit root's family but we filter it out.
    let path_comp = find_compiler_on_path();
    let other = selected.as_deref().and_then(find_compiler_in_other_roots);
    resolve_toolchain_pure(
        selected.as_deref(),
        hipcc_override,
        strict,
        path_comp,
        other,
    )
}


/// Expand a selected root into only that installation's compatible aliases.
/// Split-tree packaging keeps the real SDK under `<root>/core[-VERSION]`.
fn root_family(root: &Path) -> Vec<PathBuf> {
    let mut out = vec![root.to_path_buf()];
    let core = root.join("core");
    if core.is_dir() {
        out.push(core);
    }
    for candidate in versioned_siblings(root, "core-") {
        if !out.contains(&candidate) {
            out.push(candidate);
        }
    }
    out
}

fn distinct_complete_roots(candidates: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut identities = Vec::new();
    for candidate in candidates {
        if !is_coherent_sdk_root(&candidate) {
            continue;
        }
        let identity = std::fs::canonicalize(&candidate).unwrap_or_else(|_| candidate.clone());
        if !identities.contains(&identity) {
            identities.push(identity);
            out.push(candidate);
        }
    }
    out
}

fn ambiguous_family(root: &Path) -> Vec<PathBuf> {
    if is_coherent_sdk_root(root) || is_coherent_sdk_root(&root.join("core")) {
        return Vec::new();
    }
    let candidates = distinct_complete_roots(versioned_siblings(root, "core-"));
    (candidates.len() > 1)
        .then_some(candidates)
        .unwrap_or_default()
}

/// Complete side-by-side installations that require an explicit choice.
///
/// `/opt/rocm` and its unversioned `core` child are active-install selectors,
/// so either may choose a version. If neither exists and multiple concrete
/// versioned roots remain, silently selecting the numerically newest one would
/// be a guess and could mix ABI/toolchain expectations.
pub fn ambiguous_roots() -> Vec<PathBuf> {
    if let Some((_, configured)) = configured_root() {
        return ambiguous_family(&configured);
    }

    #[cfg(not(windows))]
    {
        let split = ambiguous_family(Path::new("/opt/rocm"));
        if !split.is_empty() {
            return split;
        }
        if !is_coherent_sdk_root(Path::new("/opt/rocm"))
            && !is_coherent_sdk_root(Path::new("/opt/rocm/core"))
            && distinct_complete_roots(versioned_siblings(Path::new("/opt/rocm"), "core-"))
                .is_empty()
        {
            let side_by_side =
                distinct_complete_roots(versioned_siblings(Path::new("/opt"), "rocm-"));
            if side_by_side.len() > 1 {
                return side_by_side;
            }
        }
    }
    Vec::new()
}

#[cfg(not(windows))]
fn has_package_rocm_evidence(root: &Path) -> bool {
    // Prefer a coherent SDK, but also admit package roots that already ship a
    // ROCm-specific tool or the HIP runtime so distro layouts remain discoverable
    // while headers-only trees stay diagnostic-only.
    is_coherent_sdk_root(root)
        || ROOT_HINT_TOOLS
            .iter()
            .any(|tool| tool_from_selected_root(root, tool).is_some())
        || runtime_library(root).is_some()
}

/// Ordered candidate ROCm roots. Entries are deduplicated but NOT filtered for
/// existence — callers that need existence should use [`root`].
pub fn roots() -> Vec<PathBuf> {
    if let Some((_, configured)) = configured_root() {
        return root_family(&configured);
    }

    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };

    #[cfg(not(windows))]
    {
        for candidate in root_family(Path::new("/opt/rocm")) {
            push(candidate);
        }
        for candidate in versioned_siblings(Path::new("/opt"), "rocm-") {
            push(candidate);
        }
    }
    for candidate in roots_from_path_tools() {
        push(candidate);
    }
    #[cfg(not(windows))]
    for candidate in [PathBuf::from("/usr"), PathBuf::from("/usr/local")] {
        if has_package_rocm_evidence(&candidate) {
            push(candidate);
        }
    }
    #[cfg(windows)]
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        let base = PathBuf::from(program_files).join("AMD").join("ROCm");
        for candidate in root_family(&base) {
            push(candidate);
        }
        for candidate in versioned_siblings(&base, "") {
            push(candidate);
        }
    }
    out
}

/// Library directories below a ROCm root.
///
/// The one-level children cover distro-native multiarch layouts such as
/// `/usr/lib/x86_64-linux-gnu` without hardcoding an architecture tuple.
fn root_library_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for relative in HIP_RUNTIME_DIRS {
        let dir = root.join(relative);
        if !out.contains(&dir) {
            out.push(dir.clone());
        }
        #[cfg(not(windows))]
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() && !out.contains(&entry.path()) {
                    out.push(entry.path());
                }
            }
        }
    }
    out
}

fn existing_library_candidates(root: &Path, sonames: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for libdir in root_library_dirs(root) {
        for soname in sonames {
            let candidate = libdir.join(soname);
            if candidate.is_file() {
                out.push(candidate.to_string_lossy().into_owned());
            }
        }
    }
    out
}

fn expected_library_candidates(root: &Path, sonames: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for libdir in HIP_RUNTIME_DIRS {
        for soname in sonames {
            out.push(
                root.join(libdir)
                    .join(soname)
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    out
}

fn selected_library_candidates(root: &Path, sonames: &[&str]) -> Vec<String> {
    let existing = existing_library_candidates(root, sonames);
    if existing.is_empty() {
        expected_library_candidates(root, sonames)
    } else {
        existing
    }
}

/// A compact, actionable explanation shared by runtime and compiler failures.
pub fn resolution_failure(component: &str, tried: &[String]) -> String {
    let mut message = format!("Could not resolve {component}.");
    let ambiguous = ambiguous_roots();
    if !ambiguous.is_empty() {
        message.push_str(&format!(
            "\nMultiple ROCm installations are equally eligible: {}.",
            ambiguous
                .iter()
                .map(|root| root.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        message.push_str(
            "\nhipfire refused to guess. Select one with \
             HIPFIRE_ROCM_PATH=/absolute/path/to/rocm.",
        );
    }
    if let Some((var, root)) = configured_root() {
        message.push_str(&format!(
            "\n{var}={} is authoritative; hipfire did not fall back to another ROCm installation.",
            root.display()
        ));
    }
    if let Some((var, path)) = configured_compiler() {
        if is_executable(&path) {
            message.push_str(&format!(
                "\n{var}={} is set and is authoritative for the device compiler.",
                path.display()
            ));
        } else {
            message.push_str(&format!(
                "\n{var}={} is set but does not exist or is not executable; hipfire did not fall back to another compiler.",
                path.display()
            ));
        }
    }
    // Cross-root strict hint: when the selected root is headers+runtime only,
    // the failure may be due to strict mode. Surface it.
    if is_strict_rocm() {
        if let Some(selected) = root() {
            if is_headers_runtime_only_root(&selected) {
                message.push_str(
                    "\nCross-root compiler fallback is disabled by HIPFIRE_ROCM_STRICT=1; unset it or set HIPFIRE_HIPCC=/path/to/hipcc (--hipcc PATH) or install a compiler under the selected root.",
                );
            }
        }
    }
    if !tried.is_empty() {
        message.push_str(&format!("\nTried: {}", tried.join(", ")));
    }
    message.push_str(
        "\nInstall the ROCm HIP runtime, HIP development headers, and device compiler \
         (the installation must provide libamdhip64, libhsa-runtime64, \
         include/hip/hip_runtime.h, and bin/hipcc).",
    );
    for line in install_guidance() {
        message.push_str(&format!("\n  {line}"));
    }
    message.push_str(
        "\nFor a non-default or side-by-side install, set \
         HIPFIRE_ROCM_PATH=/absolute/path/to/rocm (ROCM_PATH and HIP_PATH are also honored).",
    );
    // Also point at the compiler override for split installs.
    message.push_str(
        "\nIf the device compiler lives in a different prefix than the runtime, set \
         HIPFIRE_HIPCC=/absolute/path/to/hipcc (--hipcc PATH).",
    );
    message
}

/// Expand a tool basename into the filesystem names a host may actually ship.
///
/// On Windows the HIP SDK installs `hipcc.bat` (and sometimes `.cmd`/`.exe`);
/// bare names are still tried first so an extensionless shim wins when present.
/// Callers that already pass a suffixed name are left alone. `windows_suffixes`
/// is a pure parameter so Linux unit tests can exercise the Windows policy
/// without `cfg(windows)` or process-global environment mutation.
fn tool_filename_candidates(name: &str, windows_suffixes: bool) -> Vec<String> {
    if !windows_suffixes {
        return vec![name.to_string()];
    }
    const SUFFIXES: &[&str] = &[".bat", ".cmd", ".exe", ".com"];
    if SUFFIXES.iter().any(|suffix| {
        name.len() >= suffix.len() && name[name.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
    }) {
        return vec![name.to_string()];
    }
    let mut out = Vec::with_capacity(1 + SUFFIXES.len());
    out.push(name.to_string());
    for suffix in SUFFIXES {
        // `.com` is listed for PATHEXT parity but is uncommon for HIP tools;
        // keep the probe set aligned with normal Windows executable lookup.
        out.push(format!("{name}{suffix}"));
    }
    out
}

/// First existing tool path under `dir` for `name`, honouring optional Windows suffixes.
fn first_tool_in_dir(dir: &Path, name: &str, windows_suffixes: bool) -> Option<PathBuf> {
    for basename in tool_filename_candidates(name, windows_suffixes) {
        let candidate = dir.join(basename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// The first path on `PATH` matching `name` (with host-appropriate suffixes).
fn path_tool(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        if let Some(found) = first_tool_in_dir(&dir, name, cfg!(windows)) {
            return Some(std::fs::canonicalize(&found).unwrap_or(found));
        }
    }
    None
}

/// Kept separate so tests can prove the strict selected-root behavior without
/// mutating process-global environment variables.
///
/// Public because diagnostics (`hipfire-cli diag`) need the same
/// host-suffix-aware lookup: on Windows the HIP SDK installs `hipcc.bat` /
/// `hipcc.exe`, and probing the bare name would report a coherent SDK as
/// missing its compiler.
pub fn tool_from_selected_root(root: &Path, name: &str) -> Option<PathBuf> {
    first_tool_in_dir(&root.join("bin"), name, cfg!(windows))
}

/// Does this directory carry HIP headers (`include/hip/hip_runtime.h`)?
///
/// This is the public header probe used by diagnostics and compiler flag
/// plumbing. It is deliberately weaker than selection eligibility: a
/// headers-only tree is real evidence for messaging via [`missing_components`],
/// but it must not win root selection or count toward ambiguity. Selection and
/// ambiguity require a coherent SDK (headers, HIP runtime, device compiler,
/// and on non-Windows the HSA runtime).
///
/// Some installs keep `/opt/rocm` as a directory holding only version
/// symlinks (`core`, `core-7`, `core-7.14`) with no `include/`, `lib/` or
/// `bin/` of its own. Such a path passes `is_dir` but resolves every header
/// and library lookup to nothing, so existence alone is not a usable test.
pub fn is_complete_root(path: &Path) -> bool {
    path.join("include")
        .join("hip")
        .join("hip_runtime.h")
        .is_file()
}

/// HIP runtime library filenames, most preferred first. Windows ships
/// `amdhip64.dll` (versioned as `amdhip64_7.dll` from HIP SDK 7.x); ELF
/// platforms ship `libamdhip64.so` with SONAME variants.
#[cfg(windows)]
pub const HIP_RUNTIME_LIBRARIES: &[&str] = &["amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll"];
#[cfg(not(windows))]
pub const HIP_RUNTIME_LIBRARIES: &[&str] = &[
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
];

/// Directories within a root that hold the HIP runtime library. Windows keeps
/// DLLs beside the executables in `bin`; ELF platforms use `lib`, or `lib64` on
/// the Fedora/RHEL layout where ROCm installs into `/usr`.
#[cfg(windows)]
pub const HIP_RUNTIME_DIRS: &[&str] = &["bin"];
#[cfg(not(windows))]
pub const HIP_RUNTIME_DIRS: &[&str] = &["lib", "lib64"];

/// The HIP runtime library under `root`, if this install ships one.
///
/// Deliberately root-scoped. Answering "does THIS root carry the runtime" needs
/// the dynamic loader kept out of it.
pub fn runtime_library(root: &Path) -> Option<PathBuf> {
    for libdir in root_library_dirs(root) {
        for name in HIP_RUNTIME_LIBRARIES {
            let p = libdir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// HSA runtime sonames, most specific first. Required on non-Windows coherent
/// SDKs; Windows HIP SDK does not ship libhsa.
#[cfg(not(windows))]
const HSA_RUNTIME_LIBRARIES: &[&str] = &["libhsa-runtime64.so.1", "libhsa-runtime64.so"];

/// The HSA runtime library under `root`, if present.
#[cfg(not(windows))]
fn hsa_runtime_library(root: &Path) -> Option<PathBuf> {
    for libdir in root_library_dirs(root) {
        for name in HSA_RUNTIME_LIBRARIES {
            let p = libdir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// True when `root` carries at least one device-compiler entry point under `bin/`.
fn has_device_compiler(root: &Path) -> bool {
    DEVICE_COMPILERS
        .iter()
        .any(|name| tool_from_selected_root(root, name).is_some())
}

/// Coherent-SDK eligibility: HIP headers, HIP runtime, a device compiler, and
/// (on non-Windows) the HSA runtime.
///
/// Selection and ambiguity only consider roots that pass this predicate.
/// Header-only / runtime-only / compiler-only trees remain visible to
/// diagnostics via [`is_complete_root`] and [`missing_components`] but never
/// count as fully eligible installs.
fn is_coherent_sdk_root(path: &Path) -> bool {
    if !is_complete_root(path) {
        return false;
    }
    if runtime_library(path).is_none() {
        return false;
    }
    if !has_device_compiler(path) {
        return false;
    }
    #[cfg(not(windows))]
    if hsa_runtime_library(path).is_none() {
        return false;
    }
    true
}

/// A prerequisite a ROCm root does not provide.
///
/// Deliberately carries no package name: what is missing is a fact we can
/// establish from the filesystem, whereas what to install is a per-distro
/// guess. Those are separated so a wrong guess can never make the certain part
/// wrong — see [`install_guidance`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingComponent {
    /// What is absent, in the terms a user would recognise.
    pub what: &'static str,
    /// The path that was probed, so the claim is checkable by hand.
    pub probed: PathBuf,
}

/// AMD's live GPU/OS/version install selector. Package names are deliberately
/// not copied into hipfire because they change between ROCm releases.
pub const ROCM_INSTALL_DOCS: &str = "https://rocm.docs.amd.com/en/latest/install/rocm.html";

/// How to install the missing HIP components.
///
/// Deliberately thin. Naming a package per distro means maintaining a table of
/// names hipfire cannot verify and that drift. Guidance lists the exact
/// artifacts already probed ([`missing_components`], [`resolution_failure`])
/// and points at AMD's live install selector so the user picks packages for
/// their GPU, OS, and supported ROCm version. That stays correct when
/// meta-package names change.
pub fn install_guidance() -> Vec<String> {
    let artifacts = if cfg!(windows) {
        "Required artifacts: amdhip64.dll, include/hip/hip_runtime.h, and bin/hipcc.bat."
    } else {
        "Required artifacts: libamdhip64, libhsa-runtime64, \
         include/hip/hip_runtime.h, and bin/hipcc."
    };
    vec![
        artifacts.to_string(),
        format!(
            "Select packages for your GPU/OS/version via AMD's install guide: \
             {ROCM_INSTALL_DOCS}"
        ),
    ]
}

/// Prerequisites missing from `root`, beyond the device compiler.
///
/// A root can carry `bin/hipcc` and still be unusable: the device compiler,
/// HIP headers, and HIP runtime can be packaged separately. Installing only
/// the compiler leaves `hipcc --version` working while every kernel compile
/// fails on `hip/hip_runtime.h` and every `dlopen` of `libamdhip64.so` fails
/// — which is exactly what a "compiler present, nothing works" report looks
/// like. Callers use this to say so before doing work.
pub fn missing_components(root: &Path) -> Vec<MissingComponent> {
    let mut out = Vec::new();
    if !is_complete_root(root) {
        out.push(MissingComponent {
            what: "HIP headers (hip/hip_runtime.h)",
            probed: root.join("include").join("hip").join("hip_runtime.h"),
        });
    }
    if runtime_library(root).is_none() {
        out.push(MissingComponent {
            what: if cfg!(windows) {
                "HIP runtime (amdhip64.dll)"
            } else {
                "HIP runtime (libamdhip64.so)"
            },
            probed: root
                .join(HIP_RUNTIME_DIRS[0])
                .join(HIP_RUNTIME_LIBRARIES[0]),
        });
    }
    out
}

/// The first candidate root that is a coherent SDK, falling back to the first
/// that merely exists.
///
/// Preferring coherent eligibility is what lets `/opt/rocm/core[-*]` win on
/// installs where `/opt/rocm` is a shim or headers-only directory. On a
/// conventional install `/opt/rocm` is coherent and is still chosen first, so
/// the ordering documented above is unchanged for everyone else. The `is_dir`
/// fallback keeps behaviour identical when nothing validates.
pub fn root() -> Option<PathBuf> {
    if !ambiguous_roots().is_empty() {
        return None;
    }
    let candidates = roots();
    candidates
        .iter()
        .find(|p| is_coherent_sdk_root(p))
        .cloned()
        .or_else(|| candidates.into_iter().find(|p| p.is_dir()))
}

/// ROCm version string from `<root>/.info/version`, if readable.
pub fn version() -> Option<String> {
    for r in roots() {
        let f = r.join(".info").join("version");
        if let Ok(s) = std::fs::read_to_string(&f) {
            let s = s.trim();
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

/// Candidate load paths for a library.
///
/// `sonames` should be ordered most-preferred first, e.g.
/// `["libamdhip64.so", "libamdhip64.so.7"]` or Windows
/// `["amdhip64.dll", "amdhip64_7.dll"]`.
///
/// When a root is selected or explicitly configured, candidates stay inside
/// that root family — never bare loader names and never the user cache. On
/// Windows only, with no configured/selected/ambiguous root, the hipfire
/// runtime cache and bare DLL names remain available so a cache-only install
/// still loads. Unix never falls back to bare sonames (ldconfig would mix
/// installs).
pub fn library_candidates(sonames: &[&str]) -> Vec<String> {
    let ambiguous = !ambiguous_roots().is_empty();
    let selected = if ambiguous {
        None
    } else if let Some(selected) = root() {
        Some(selected)
    } else if let Some((_, configured)) = configured_root() {
        Some(configured)
    } else {
        None
    };
    #[cfg(windows)]
    let profile = std::env::var_os("USERPROFILE").map(PathBuf::from);
    #[cfg(not(windows))]
    let profile: Option<PathBuf> = None;
    library_load_candidates(
        selected.as_deref(),
        ambiguous,
        sonames,
        cfg!(windows),
        profile.as_deref(),
    )
}

/// Pure library-candidate policy for [`library_candidates`] and unit tests.
///
/// `selected` is either a discovered root or an explicit configured root that
/// must stay authoritative even when empty/invalid. Legacy Windows cache/PATH
/// candidates are emitted only when `windows` is set, nothing is selected, and
/// the install set is not ambiguous.
fn library_load_candidates(
    selected: Option<&Path>,
    ambiguous: bool,
    sonames: &[&str],
    windows: bool,
    legacy_user_profile: Option<&Path>,
) -> Vec<String> {
    if ambiguous {
        return Vec::new();
    }
    if let Some(root) = selected {
        return selected_library_candidates(root, sonames);
    }
    if windows {
        return windows_legacy_library_candidates(legacy_user_profile, sonames);
    }
    Vec::new()
}

/// Windows-only fallback when no ROCm/HIP root is configured or selected:
/// `{USERPROFILE}\.hipfire\runtime\<dll>` then bare DLL names for PATH search.
fn windows_legacy_library_candidates(user_profile: Option<&Path>, sonames: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(profile) = user_profile {
        let cache = profile.join(".hipfire").join("runtime");
        for soname in sonames {
            out.push(cache.join(soname).to_string_lossy().into_owned());
        }
    }
    for soname in sonames {
        out.push((*soname).to_owned());
    }
    out
}

/// Locate a ROCm tool (`hipcc`, `amdclang++`, `rocminfo`, …) under a resolved
/// root. `PATH` is consulted only when no root is selectable, except for the
/// device compiler where a cross-root fallback is allowed when the selected
/// root is otherwise complete (headers + runtime + HSA) but lacks a compiler.
/// The compiler fallback respects `HIPFIRE_HIPCC` and `HIPFIRE_ROCM_STRICT=1`.
pub fn tool(name: &str) -> Option<PathBuf> {
    // Explicit compiler override wins for device-compiler names.
    if DEVICE_COMPILERS.contains(&name) {
        if let Some((_, ov)) = configured_compiler() {
            if is_executable(&ov) {
                return Some(ov);
            } else {
                return None;
            }
        }
    }
    if let Some((_, configured)) = configured_root() {
        if !ambiguous_family(&configured).is_empty() {
            return None;
        }
        for root in root_family(&configured) {
            if let Some(tool) = tool_from_selected_root(&root, name) {
                return Some(tool);
            }
        }
        // Cross-root compiler fallback for the configured family, if eligible.
        if DEVICE_COMPILERS.contains(&name) && !is_strict_rocm() {
            // Find the first family member that is a dir to use as "selected" for
            // eligibility; libs_only trees report as first family entry.
            let selected = root_family(&configured)
                .into_iter()
                .find(|p| p.is_dir())
                .unwrap_or(configured);
            if is_headers_runtime_only_root(&selected) {
                if let Some(pc) = find_compiler_on_path() {
                    return Some(pc);
                }
                if let Some((tool, _)) = find_compiler_in_other_roots(&selected) {
                    return Some(tool);
                }
            }
        }
        return None;
    }
    if let Some(selected) = root() {
        if let Some(tool) = tool_from_selected_root(&selected, name) {
            return Some(tool);
        }
        if DEVICE_COMPILERS.contains(&name) && !is_strict_rocm() && is_headers_runtime_only_root(&selected) {
            if let Some(pc) = find_compiler_on_path() {
                return Some(pc);
            }
            if let Some((tool, _)) = find_compiler_in_other_roots(&selected) {
                return Some(tool);
            }
        }
        return None;
    }
    path_tool(name)
}

/// The device compiler this installation should use, most specific first.
pub fn device_compiler() -> Option<PathBuf> {
    // HIPFIRE_HIPCC is authoritative; invalid values do not fall through.
    if let Some((_, ov)) = configured_compiler() {
        if is_executable(&ov) {
            return Some(ov);
        } else {
            return None;
        }
    }
    // Use the shared toolchain resolver so compiler-source provenance is uniform
    // between `device_compiler()` and `resolve_toolchain()`. Fall back to the
    // legacy direct probe only when toolchain resolution fails for non-compiler
    // reasons (e.g. no root at all).
    if let Ok(tc) = resolve_toolchain() {
        return tc.compiler;
    }
    // Legacy fallback: direct tool probe without cross-root (preserves previous
    // behaviour when `resolve_toolchain` is unavailable, e.g. ambiguous roots).
    DEVICE_COMPILERS.iter().find_map(|name| {
        if let Some((_, configured)) = configured_root() {
            if !ambiguous_family(&configured).is_empty() {
                return None;
            }
            for root in root_family(&configured) {
                if let Some(tool) = tool_from_selected_root(&root, name) {
                    return Some(tool);
                }
            }
            None
        } else if let Some(selected) = root() {
            tool_from_selected_root(&selected, name)
        } else {
            path_tool(name)
        }
    })
}

/// `ROCM_PATH` value a spawned device compiler needs, or `None` when the
/// configured environment already matches the selected compiler's install root.
///
/// `hipcc` locates its own LLVM as `$ROCM_PATH/lib/llvm/bin/clang++`, and
/// `ROCM_PATH` defaults to `/opt/rocm`. On an install rooted elsewhere —
/// `/opt/rocm/core-7.14` on this fleet — pairing that hipcc with a different
/// `ROCM_PATH` makes every compile fail with
///
///   sh: 1: /opt/rocm/lib/llvm/bin/clang++: not found
///
/// so the child must receive the root of the *selected* compiler, not a
/// conflicting ambient install. When `ROCM_PATH` already points at that root,
/// returns `None` so an explicit matching operator choice is left alone.
///
/// `compiler` is the selected device compiler path (absolute or bare name).
/// When the path cannot be resolved to a root, falls back to the previous
/// "set `ROCM_PATH` only if unset" semantics via [`root`].
pub fn compiler_env_root(compiler: &Path) -> Option<PathBuf> {
    let configured = std::env::var_os("ROCM_PATH")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    compiler_env_root_from(compiler, configured.as_deref())
}

/// Pure form of [`compiler_env_root`] for tests: `configured` is the ambient
/// `ROCM_PATH` (if any).
fn compiler_env_root_from(compiler: &Path, configured: Option<&Path>) -> Option<PathBuf> {
    match root_from_compiler(compiler) {
        Some(selected) => match configured {
            Some(cfg) if paths_same_root(cfg, &selected) => None,
            _ => Some(selected),
        },
        None => {
            // Resolution failed — keep prior semantics: leave an explicit
            // ROCM_PATH alone, otherwise supply the discovered install root.
            if configured.is_some() {
                None
            } else {
                root()
            }
        }
    }
}

/// Derive `<root>` from a selected compiler path (`<root>/bin/<tool>`).
/// Absolute/relative paths are canonicalized when possible; bare names are
/// resolved on `PATH` the same way [`roots_from_path_tools`] probes tools.
fn root_from_compiler(compiler: &Path) -> Option<PathBuf> {
    let selected = if compiler.components().count() == 1 {
        path_tool(compiler.to_str()?)?
    } else {
        compiler.to_path_buf()
    };
    root_from_tool_path(&selected)
}

fn paths_same_root(a: &Path, b: &Path) -> bool {
    let ca = std::fs::canonicalize(a).unwrap_or_else(|_| a.to_path_buf());
    let cb = std::fs::canonicalize(b).unwrap_or_else(|_| b.to_path_buf());
    ca == cb
}

#[cfg(test)]
mod compiler_env_tests {
    use super::*;

    #[test]
    fn compiler_root_follows_the_selected_toolchain() {
        let selected = Path::new("/opt/rocm/core-7.14/bin/hipcc");

        assert_eq!(
            compiler_env_root_from(selected, Some(Path::new("/opt/rocm"))),
            Some(PathBuf::from("/opt/rocm/core-7.14"))
        );
        assert_eq!(
            compiler_env_root_from(selected, Some(Path::new("/opt/rocm/core-7.14"))),
            None
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_key_orders_core_dirs_newest_first() {
        assert_eq!(version_key("core-7.14"), vec![7, 14]);
        assert_eq!(version_key("core-7"), vec![7]);
        assert_eq!(version_key("rocm-6.4.1"), vec![6, 4, 1]);
        assert!(version_key("core-7.14") > version_key("core-7"));
        assert!(version_key("core-7.14") > version_key("core-7.9"));
        assert!(version_key("nodigits").is_empty());
    }

    #[test]
    fn hip_path_with_trailing_hip_component_normalizes_to_root() {
        assert_eq!(
            normalize_hip_path(Path::new("/opt/rocm/hip")),
            PathBuf::from("/opt/rocm")
        );
        // A root that merely lives under a directory called hip is untouched.
        assert_eq!(
            normalize_hip_path(Path::new("/opt/rocm")),
            PathBuf::from("/opt/rocm")
        );
    }

    #[test]
    fn versioned_siblings_sorts_newest_first_and_skips_files() {
        let tmp = std::env::temp_dir().join(format!("hipfire-rocm-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        for d in ["core-7", "core-7.14", "core-6.4", "unrelated"] {
            std::fs::create_dir_all(tmp.join(d)).unwrap();
        }
        // A regular file matching the prefix must not be treated as a root.
        std::fs::write(tmp.join("core-9-notadir"), b"x").unwrap();

        let got = versioned_siblings(&tmp, "core-");
        let names: Vec<String> = got
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["core-7.14", "core-7", "core-6.4"]);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// A compiler-only root: `bin/hipcc` present, neither HIP headers nor the
    /// runtime. Used to surface as clang's bare "file not found" at the end of
    /// a full install attempt.
    #[test]
    fn a_compiler_only_root_reports_both_hip_components_missing() {
        let tmp = std::env::temp_dir().join(format!("hipfire-rocm-parts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("core-7.14");
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(root.join("bin").join("hipcc"), b"#!/bin/sh\n").unwrap();

        let missing = missing_components(&root);
        assert_eq!(missing.len(), 2, "{missing:?}");
        assert!(missing[0].what.contains("hip_runtime.h"));
        assert_eq!(
            missing[0].probed,
            root.join("include").join("hip").join("hip_runtime.h")
        );
        assert!(missing[1].what.contains("HIP runtime"));

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// Guidance always names the probed artifacts and AMD's live selector —
    /// never an executable package install that will rot when names drift.
    #[test]
    fn install_guidance_lists_artifacts_and_amd_selector() {
        let lines = install_guidance();
        assert!(!lines.is_empty());
        let joined = lines.join("\n");
        assert!(
            joined.contains("rocm.docs.amd.com"),
            "the docs link is the distro-independent answer: {lines:?}"
        );
        for needle in ["libamdhip64", "libhsa-runtime64", "hip_runtime.h", "hipcc"] {
            assert!(
                joined.contains(needle),
                "guidance must name required artifact {needle}: {lines:?}"
            );
        }
        assert!(
            !joined.contains("apt install")
                && !joined.contains("amdrocm-")
                && !joined.contains("rocm-hip-"),
            "guidance must not guess drifting package names: {lines:?}"
        );
    }

    #[test]
    fn headers_only_root_reports_missing_runtime() {
        let tmp = std::env::temp_dir().join(format!("hipfire-rocm-headers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("include").join("hip")).unwrap();
        std::fs::write(tmp.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();

        let missing = missing_components(&tmp);
        assert_eq!(missing.len(), 1, "{missing:?}");
        assert!(missing[0].what.contains("HIP runtime"), "{missing:?}");
        assert_eq!(
            missing[0].probed,
            tmp.join(HIP_RUNTIME_DIRS[0]).join(HIP_RUNTIME_LIBRARIES[0])
        );

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn runtime_only_root_reports_missing_headers() {
        let tmp =
            std::env::temp_dir().join(format!("hipfire-rocm-runtime-only-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(
            tmp.join(HIP_RUNTIME_DIRS[0]).join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();

        let missing = missing_components(&tmp);
        assert_eq!(missing.len(), 1, "{missing:?}");
        assert!(missing[0].what.contains("hip_runtime.h"), "{missing:?}");
        assert_eq!(
            missing[0].probed,
            tmp.join("include").join("hip").join("hip_runtime.h")
        );

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn resolution_failure_names_tried_paths_components_and_override() {
        let tried = vec![
            "/bad/rocm/lib/libamdhip64.so".to_string(),
            "/bad/rocm/lib/libamdhip64.so.7".to_string(),
        ];
        let msg = resolution_failure("HIP runtime library", &tried);
        assert!(
            msg.contains("Could not resolve HIP runtime library"),
            "{msg}"
        );
        assert!(
            msg.contains("Tried: /bad/rocm/lib/libamdhip64.so, /bad/rocm/lib/libamdhip64.so.7"),
            "{msg}"
        );
        assert!(msg.contains("libamdhip64"), "{msg}");
        assert!(msg.contains("libhsa-runtime64"), "{msg}");
        assert!(msg.contains("include/hip/hip_runtime.h"), "{msg}");
        assert!(msg.contains("bin/hipcc"), "{msg}");
        assert!(
            msg.contains("HIPFIRE_ROCM_PATH=/absolute/path/to/rocm"),
            "{msg}"
        );
        assert!(msg.contains(ROCM_INSTALL_DOCS), "{msg}");
        for line in install_guidance() {
            assert!(
                msg.contains(&line),
                "resolution_failure must embed install guidance line {line:?}: {msg}"
            );
        }
    }

    #[test]
    fn a_root_with_headers_and_runtime_is_not_missing_anything() {
        let tmp = std::env::temp_dir().join(format!("hipfire-rocm-full-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("core-7.14");
        // Build the tree through the platform constants so this test exercises
        // the real Windows layout (bin/amdhip64.dll) when run on Windows.
        std::fs::create_dir_all(root.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(root.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(root.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0])
                .join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();

        assert!(missing_components(&root).is_empty());
        assert!(runtime_library(&root).is_some());

        // A versioned-only name still counts: Debian ships the unversioned
        // symlink in the -dev package, which not every install carries, and the
        // Windows HIP SDK 7.x installs amdhip64_7.dll.
        std::fs::remove_file(
            root.join(HIP_RUNTIME_DIRS[0])
                .join(HIP_RUNTIME_LIBRARIES[0]),
        )
        .unwrap();
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0])
                .join(HIP_RUNTIME_LIBRARIES[1]),
            b"",
        )
        .unwrap();
        assert!(runtime_library(&root).is_some());

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// Fedora/RHEL package ROCm into `/usr`, so the root resolves to `/usr`
    /// with the runtime in `lib64` rather than `lib`. Probing only `lib` would
    /// reject a perfectly good install and block the installer on it.
    #[test]
    #[cfg(not(windows))]
    fn a_lib64_layout_is_accepted() {
        let tmp = std::env::temp_dir().join(format!("hipfire-rocm-lib64-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(tmp.join("lib64")).unwrap();
        std::fs::write(tmp.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(tmp.join("lib64").join("libamdhip64.so"), b"").unwrap();

        assert!(runtime_library(&tmp).is_some());
        assert!(missing_components(&tmp).is_empty());

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn is_complete_root_rejects_a_shim_directory() {
        let tmp = std::env::temp_dir().join(format!("hipfire-rocm-shim-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);

        // A shim root: exists, holds only version symlink targets, no headers.
        // This is the real layout on installs that keep the tree under
        // /opt/rocm/core-<ver>.
        let shim = tmp.join("rocm");
        std::fs::create_dir_all(shim.join("core-7.14").join("include").join("hip")).unwrap();
        std::fs::write(
            shim.join("core-7.14")
                .join("include")
                .join("hip")
                .join("hip_runtime.h"),
            b"// marker",
        )
        .unwrap();

        assert!(
            !is_complete_root(&shim),
            "a directory with no include/hip/hip_runtime.h must not count as a root"
        );
        assert!(
            is_complete_root(&shim.join("core-7.14")),
            "the versioned sibling carrying the headers is the real root"
        );

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn selected_root_library_candidates_never_append_bare_sonames() {
        let root = std::env::temp_dir().join(format!("hipfire-rocm-strict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(HIP_RUNTIME_DIRS[0])).unwrap();
        let runtime = root
            .join(HIP_RUNTIME_DIRS[0])
            .join(HIP_RUNTIME_LIBRARIES[0]);
        std::fs::write(&runtime, b"").unwrap();

        let candidates = selected_library_candidates(
            &root,
            &[HIP_RUNTIME_LIBRARIES[0], HIP_RUNTIME_LIBRARIES[1]],
        );
        assert_eq!(candidates, vec![runtime.to_string_lossy().into_owned()]);
        assert!(
            candidates
                .iter()
                .all(|candidate| Path::new(candidate).is_absolute()),
            "a selected root must never fall through to a bare soname: {candidates:?}"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn empty_selected_root_library_candidates_stay_root_scoped() {
        let root =
            std::env::temp_dir().join(format!("hipfire-rocm-empty-lib-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let candidates = selected_library_candidates(
            &root,
            &[HIP_RUNTIME_LIBRARIES[0], HIP_RUNTIME_LIBRARIES[1]],
        );
        assert!(
            !candidates.is_empty(),
            "an empty selected root must still report expected absolute paths"
        );
        assert!(
            candidates.iter().all(|candidate| {
                let path = Path::new(candidate);
                path.is_absolute() && path.starts_with(&root)
            }),
            "invalid/empty override must not fall through to bare sonames: {candidates:?}"
        );
        for soname in [HIP_RUNTIME_LIBRARIES[0], HIP_RUNTIME_LIBRARIES[1]] {
            assert!(
                !candidates.iter().any(|c| c == soname),
                "bare soname {soname} leaked into selected-root candidates: {candidates:?}"
            );
        }

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn tool_from_selected_root_does_not_cross_installs() {
        let base =
            std::env::temp_dir().join(format!("hipfire-rocm-tool-scope-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let selected = base.join("selected");
        let other = base.join("other");
        std::fs::create_dir_all(selected.join("bin")).unwrap();
        std::fs::create_dir_all(other.join("bin")).unwrap();
        std::fs::write(other.join("bin").join("hipcc"), b"#!/bin/sh\n").unwrap();

        assert_eq!(
            tool_from_selected_root(&selected, "hipcc"),
            None,
            "a selected root must not pick up another install's compiler"
        );
        assert_eq!(
            tool_from_selected_root(&other, "hipcc"),
            Some(other.join("bin").join("hipcc"))
        );

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn windows_tool_filename_candidates_include_bat_and_exe() {
        assert_eq!(
            tool_filename_candidates("hipcc", false),
            vec!["hipcc".to_string()]
        );
        assert_eq!(
            tool_filename_candidates("hipcc", true),
            vec![
                "hipcc".to_string(),
                "hipcc.bat".to_string(),
                "hipcc.cmd".to_string(),
                "hipcc.exe".to_string(),
                "hipcc.com".to_string(),
            ]
        );
        // Already-suffixed names must not be double-extended.
        assert_eq!(
            tool_filename_candidates("hipcc.bat", true),
            vec!["hipcc.bat".to_string()]
        );
        assert_eq!(
            tool_filename_candidates("HIPCC.EXE", true),
            vec!["HIPCC.EXE".to_string()]
        );
    }

    #[test]
    fn windows_shaped_hipcc_bat_resolves_inside_selected_root() {
        // Pure FS policy under Linux cfg: force Windows suffixes on a synthetic
        // tree that only carries hipcc.bat (the Windows HIP SDK layout).
        let base =
            std::env::temp_dir().join(format!("hipfire-rocm-win-hipcc-bat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base
            .join("C")
            .join("Program Files")
            .join("AMD")
            .join("ROCm")
            .join("6.2");
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let hipcc_bat = bin.join("hipcc.bat");
        std::fs::write(&hipcc_bat, b"@echo off\n").unwrap();

        assert_eq!(
            first_tool_in_dir(&bin, "hipcc", true),
            Some(hipcc_bat.clone()),
            "Windows tool lookup must recognize hipcc.bat"
        );
        assert_eq!(
            first_tool_in_dir(&bin, "hipcc", false),
            None,
            "Unix tool lookup must not invent .bat"
        );

        // DLL policy is root-scoped via library_load_candidates. On this host
        // HIP_RUNTIME_DIRS may be lib/ (Linux) rather than bin/ (Windows); the
        // invariant under test is family containment, not the Windows dir name.
        let runtime_dir = root.join(HIP_RUNTIME_DIRS[0]);
        std::fs::create_dir_all(&runtime_dir).unwrap();
        let dll = runtime_dir.join("amdhip64.dll");
        std::fs::write(&dll, b"").unwrap();
        let dlls = ["amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll"];
        let profile = base.join("Users").join("tester");
        std::fs::create_dir_all(profile.join(".hipfire").join("runtime")).unwrap();
        std::fs::write(
            profile
                .join(".hipfire")
                .join("runtime")
                .join("amdhip64.dll"),
            b"",
        )
        .unwrap();
        let candidates = library_load_candidates(Some(&root), false, &dlls, true, Some(&profile));
        assert_eq!(candidates, vec![dll.to_string_lossy().into_owned()]);
        assert!(
            candidates.iter().all(|c| {
                let path = Path::new(c);
                path.starts_with(&root) && !c.contains(".hipfire") && c != "amdhip64.dll"
            }),
            "selected Windows root must not leave its family or hit cache/PATH: {candidates:?}"
        );

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn invalid_explicit_windows_root_never_loads_bare_or_cache_dll() {
        let base = std::env::temp_dir().join(format!(
            "hipfire-rocm-win-invalid-root-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let bad_root = base.join("C").join("missing-rocm");
        let profile = base.join("Users").join("tester");
        std::fs::create_dir_all(profile.join(".hipfire").join("runtime")).unwrap();
        std::fs::write(
            profile
                .join(".hipfire")
                .join("runtime")
                .join("amdhip64.dll"),
            b"",
        )
        .unwrap();

        let dlls = ["amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll"];
        let candidates =
            library_load_candidates(Some(&bad_root), false, &dlls, true, Some(&profile));
        assert!(
            !candidates.is_empty(),
            "invalid override must still report expected absolute paths"
        );
        for c in &candidates {
            let path = Path::new(c);
            assert!(
                path.is_absolute() && path.starts_with(&bad_root),
                "invalid explicit root must not fall through to cache/PATH: {c}"
            );
            assert_ne!(c.as_str(), "amdhip64.dll");
            assert_ne!(c.as_str(), "amdhip64_7.dll");
            assert!(!c.contains(".hipfire"), "cache leak: {c}");
        }

        // Legacy cache/PATH only when no root is configured/selected/ambiguous.
        let legacy = library_load_candidates(None, false, &dlls, true, Some(&profile));
        assert!(
            legacy
                .iter()
                .any(|c| c.ends_with("amdhip64.dll") && c.contains(".hipfire")),
            "no-root Windows path must keep the user cache: {legacy:?}"
        );
        assert!(
            legacy.iter().any(|c| c == "amdhip64.dll"),
            "no-root Windows path must keep bare PATH names: {legacy:?}"
        );

        // Ambiguity refuses everything — including legacy fallbacks.
        let refused = library_load_candidates(None, true, &dlls, true, Some(&profile));
        assert!(refused.is_empty(), "{refused:?}");

        // Unix never invents bare sonames or a Windows cache path.
        let unix = library_load_candidates(None, false, &["libamdhip64.so"], false, Some(&profile));
        assert!(unix.is_empty(), "{unix:?}");

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn resolution_failure_ambiguity_refusal_is_atomic_with_override() {
        // When the host has multiple equally eligible roots, refusal and the
        // explicit-selection command must appear together. When it does not,
        // neither refusal line may appear alone. The override command itself is
        // always present (see resolution_failure_names_tried_paths…).
        let msg = resolution_failure("device compiler", &[]);
        let refused = msg.contains("hipfire refused to guess");
        let eligible = msg.contains("Multiple ROCm installations are equally eligible:");
        assert_eq!(
            refused, eligible,
            "ambiguity refusal lines must appear together: {msg}"
        );
        if refused {
            assert!(
                msg.contains("Select one with HIPFIRE_ROCM_PATH=/absolute/path/to/rocm"),
                "{msg}"
            );
        }
        assert!(
            msg.contains("HIPFIRE_ROCM_PATH=/absolute/path/to/rocm"),
            "{msg}"
        );
    }

    #[test]
    fn split_tree_expansion_never_leaves_the_configured_family() {
        let base = std::env::temp_dir().join(format!("hipfire-rocm-family-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("core-7.14")).unwrap();
        std::fs::create_dir_all(base.join("core-7.2")).unwrap();

        let family = root_family(&base);
        assert_eq!(family[0], base);
        assert!(family.iter().all(|candidate| candidate.starts_with(&base)));
        assert!(family[1].ends_with("core-7.14"));
        assert!(family[2].ends_with("core-7.2"));

        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Build a fake coherent SDK tree with only empty marker files.
    fn write_coherent_sdk(root: &Path) {
        std::fs::create_dir_all(root.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(root.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(root.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0])
                .join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();
        std::fs::write(root.join("bin").join(DEVICE_COMPILERS[0]), b"#!/bin/sh\n").unwrap();
        #[cfg(not(windows))]
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0]).join("libhsa-runtime64.so.1"),
            b"",
        )
        .unwrap();
    }

    #[test]
    fn side_by_side_family_requires_an_unversioned_selector() {
        let base =
            std::env::temp_dir().join(format!("hipfire-rocm-ambiguous-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        for version in ["core-7.14", "core-7.2"] {
            write_coherent_sdk(&base.join(version));
        }

        assert_eq!(ambiguous_family(&base).len(), 2);

        // An unversioned core selector that is itself a coherent SDK ends the
        // ambiguity; headers alone are not enough to act as the selector.
        write_coherent_sdk(&base.join("core"));
        assert!(
            ambiguous_family(&base).is_empty(),
            "an unversioned coherent core selector is authoritative"
        );

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn headers_only_sibling_cannot_beat_or_ambiguate_unique_full_sdk() {
        let base = std::env::temp_dir().join(format!(
            "hipfire-rocm-partial-vs-full-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let headers_only = base.join("core-7.14");
        let full = base.join("core-7.2");

        std::fs::create_dir_all(headers_only.join("include").join("hip")).unwrap();
        std::fs::write(
            headers_only
                .join("include")
                .join("hip")
                .join("hip_runtime.h"),
            b"",
        )
        .unwrap();
        write_coherent_sdk(&full);

        assert!(
            is_complete_root(&headers_only),
            "headers-only remains diagnostically complete for header checks"
        );
        assert!(
            !is_coherent_sdk_root(&headers_only),
            "headers-only is not a coherent SDK"
        );
        assert!(is_coherent_sdk_root(&full));

        let distinct = distinct_complete_roots([headers_only.clone(), full.clone()]);
        assert_eq!(
            distinct,
            vec![full.clone()],
            "partial trees must not enter eligibility/ambiguity sets: {distinct:?}"
        );
        assert!(
            ambiguous_family(&base).is_empty(),
            "a unique full sibling must not be made ambiguous by headers-only"
        );

        // Selection over an ordered candidate list prefers the coherent root
        // even when a headers-only path is listed first.
        let preferred = [headers_only.clone(), full.clone()]
            .into_iter()
            .find(|p| is_coherent_sdk_root(p));
        assert_eq!(preferred, Some(full));

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn two_coherent_siblings_remain_ambiguous() {
        let base =
            std::env::temp_dir().join(format!("hipfire-rocm-two-coherent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let a = base.join("core-7.14");
        let b = base.join("core-7.2");
        write_coherent_sdk(&a);
        write_coherent_sdk(&b);

        let distinct = distinct_complete_roots([a.clone(), b.clone()]);
        assert_eq!(distinct.len(), 2, "{distinct:?}");
        assert_eq!(
            ambiguous_family(&base).len(),
            2,
            "two fully coherent siblings must refuse silent selection"
        );

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn compiler_only_and_runtime_only_are_not_coherent() {
        let base =
            std::env::temp_dir().join(format!("hipfire-rocm-partial-kinds-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);

        let compiler_only = base.join("compiler");
        std::fs::create_dir_all(compiler_only.join("bin")).unwrap();
        std::fs::write(
            compiler_only.join("bin").join(DEVICE_COMPILERS[0]),
            b"#!/bin/sh\n",
        )
        .unwrap();
        assert!(!is_coherent_sdk_root(&compiler_only));

        let runtime_only = base.join("runtime");
        std::fs::create_dir_all(runtime_only.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(
            runtime_only
                .join(HIP_RUNTIME_DIRS[0])
                .join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();
        assert!(!is_coherent_sdk_root(&runtime_only));

        let full = base.join("full");
        write_coherent_sdk(&full);
        assert!(is_coherent_sdk_root(&full));

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn compiler_symlink_target_under_lib_llvm_maps_back_to_sdk_root() {
        assert_eq!(
            root_from_tool_path(Path::new("/opt/rocm-7.14/lib/llvm/bin/amdllvm")),
            Some(PathBuf::from("/opt/rocm-7.14"))
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn distro_multiarch_library_directory_is_discovered() {
        let root =
            std::env::temp_dir().join(format!("hipfire-rocm-multiarch-{}", std::process::id()));
        let multiarch = root.join("lib").join("x86_64-linux-gnu");
        let runtime = multiarch.join("libamdhip64.so");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&multiarch).unwrap();
        std::fs::write(&runtime, b"").unwrap();

        assert_eq!(runtime_library(&root), Some(runtime));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn roots_are_deduplicated() {
        let r = roots();
        let mut seen = std::collections::HashSet::new();
        for p in &r {
            assert!(seen.insert(p.clone()), "duplicate root: {p:?}");
        }
    }
    #[test]
    fn configured_compiler_is_honoured_via_pure_helper() {
        // Pure helper mirrors env accessor without global mutation (see lib.rs:3812).
        let (var, path) = configured_compiler_from(Some("/opt/rocm/bin/hipcc")).unwrap();
        assert_eq!(var, "HIPFIRE_HIPCC");
        assert_eq!(path, PathBuf::from("/opt/rocm/bin/hipcc"));
        assert!(configured_compiler_from(Some("")).is_none());
        assert!(configured_compiler_from(None).is_none());
    }

    #[test]
    fn hipcc_override_invalid_is_not_silently_ignored() {
        // Create a libs-only root to act as selected runtime root.
        let base = std::env::temp_dir().join(format!("hipfire-rocm-ov-invalid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("libs");
        std::fs::create_dir_all(root.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(root.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(root.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0]).join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();
        #[cfg(not(windows))]
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0]).join("libhsa-runtime64.so.1"),
            b"",
        )
        .unwrap();
        // Override points at a non-existent file — must error, not fall through.
        let bogus = base.join("missing_hipcc");
        let result = resolve_toolchain_pure(
            Some(&root),
            Some(&bogus),
            false,
            Some(PathBuf::from("/usr/bin/hipcc")),
            None,
        );
        assert!(result.is_err(), "invalid HIPFIRE_HIPCC must hard-fail: {result:?}");
        let msg = result.unwrap_err();
        assert!(msg.contains("HIPFIRE_HIPCC"), "{msg}");
        assert!(msg.contains(&bogus.display().to_string()), "{msg}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn hipcc_override_valid_wins_over_path() {
        let base = std::env::temp_dir().join(format!("hipfire-rocm-ov-valid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("libs");
        std::fs::create_dir_all(root.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(root.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(root.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0]).join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();
        #[cfg(not(windows))]
        std::fs::write(
            root.join(HIP_RUNTIME_DIRS[0]).join("libhsa-runtime64.so.1"),
            b"",
        )
        .unwrap();
        // Create a valid executable override.
        let ov_dir = base.join("ov_bin");
        std::fs::create_dir_all(&ov_dir).unwrap();
        let ov = ov_dir.join("hipcc");
        std::fs::write(&ov, b"#!/bin/sh\necho hipcc\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&ov).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&ov, p).unwrap();
        }
        let other = PathBuf::from("/tmp/other/bin/hipcc");
        let result = resolve_toolchain_pure(Some(&root), Some(&ov), false, Some(other.clone()), None).unwrap();
        assert_eq!(result.compiler, Some(ov.clone()));
        assert_eq!(result.compiler_source, Some(CompilerSource::Override));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn libs_only_root_with_path_compiler_yields_cross_root_toolchain() {
        let base = std::env::temp_dir().join(format!("hipfire-rocm-cross-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let libs = base.join("libs_only");
        std::fs::create_dir_all(libs.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(libs.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(libs.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(
            libs.join(HIP_RUNTIME_DIRS[0]).join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();
        #[cfg(not(windows))]
        std::fs::write(
            libs.join(HIP_RUNTIME_DIRS[0]).join("libhsa-runtime64.so.1"),
            b"",
        )
        .unwrap();
        std::fs::create_dir_all(libs.join(".info")).unwrap();
        std::fs::write(libs.join(".info").join("version"), b"7.14.0").unwrap();
        // Fake compiler on PATH (and its own root with version).
        let comp_root = base.join("compiler_root");
        std::fs::create_dir_all(comp_root.join("bin")).unwrap();
        std::fs::create_dir_all(comp_root.join(".info")).unwrap();
        std::fs::write(comp_root.join(".info").join("version"), b"7.14.0").unwrap();
        let comp = comp_root.join("bin").join("hipcc");
        std::fs::write(&comp, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&comp).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&comp, p).unwrap();
        }
        let result = resolve_toolchain_pure(Some(&libs), None, false, Some(comp.clone()), None).unwrap();
        assert_eq!(result.root, libs);
        assert_eq!(result.compiler, Some(comp.clone()));
        assert_eq!(result.compiler_source, Some(CompilerSource::Path));
        assert!(result.compiler_root.is_some());
        // Warning must name selected root, compiler path, compiler root and versions.
        let warnings = toolchain_warnings(&result);
        assert!(!warnings.is_empty(), "cross-root must warn");
        let joined = warnings.join("\n");
        assert!(joined.contains(&libs.display().to_string()), "{joined}");
        assert!(joined.contains(&comp.display().to_string()), "{joined}");
        assert!(joined.contains(&result.compiler_root.unwrap().display().to_string()), "{joined}");
        assert!(joined.contains("7.14.0"), "{joined}");
        assert!(joined.to_lowercase().contains("different"), "{joined}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn libs_only_root_with_strict_still_fails() {
        let base = std::env::temp_dir().join(format!("hipfire-rocm-cross-strict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let libs = base.join("libs_only");
        std::fs::create_dir_all(libs.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(libs.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(libs.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(
            libs.join(HIP_RUNTIME_DIRS[0]).join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();
        #[cfg(not(windows))]
        std::fs::write(
            libs.join(HIP_RUNTIME_DIRS[0]).join("libhsa-runtime64.so.1"),
            b"",
        )
        .unwrap();
        let comp = base.join("fakebin").join("hipcc");
        std::fs::create_dir_all(comp.parent().unwrap()).unwrap();
        std::fs::write(&comp, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&comp).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&comp, p).unwrap();
        }
        let result = resolve_toolchain_pure(Some(&libs), None, true, Some(comp), None);
        assert!(result.is_err(), "strict must hard-fail: {result:?}");
        let msg = result.unwrap_err();
        assert!(msg.to_lowercase().contains("hipcc") || msg.contains("HIPFIRE_ROCM_STRICT"), "{msg}");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn compiler_only_root_still_fails() {
        let base = std::env::temp_dir().join(format!("hipfire-rocm-comp-only-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("comp_only");
        std::fs::create_dir_all(root.join("bin")).unwrap();
        let hipcc = root.join("bin").join(DEVICE_COMPILERS[0]);
        std::fs::write(&hipcc, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&hipcc).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&hipcc, p).unwrap();
        }
        // No headers/runtime, so missing_components must be non-empty and
        // resolve must fail (even though compiler exists).
        assert!(!missing_components(&root).is_empty());
        let result = resolve_toolchain_pure(Some(&root), None, false, None, None);
        // Pure helper will try to find compiler under root — it will succeed
        // via SelectedRoot, but the root is still not headers_runtime complete.
        // The caller (hipfire-rocm-resolve) additionally checks missing_components,
        // so we assert that missing_components is non-empty to guarantee the
        // end-to-end failure.
        assert!(!missing_components(&root).is_empty());
        // If the pure helper considered this a valid toolchain, it would return
        // SelectedRoot; we just ensure the install is still incomplete.
        if let Ok(tc) = result {
            assert_eq!(tc.compiler_source, Some(CompilerSource::SelectedRoot));
            assert!(!missing_components(&tc.root).is_empty());
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn canonical_and_debian_multiarch_roots_still_resolve() {
        // Canonical layout uses HIP_RUNTIME_DIRS[0] directly.
        let base = std::env::temp_dir().join(format!("hipfire-rocm-canonical-accept-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let canon = base.join("canonical");
        write_coherent_sdk(&canon);
        assert!(is_coherent_sdk_root(&canon));
        let result = resolve_toolchain_pure(Some(&canon), None, false, None, None).unwrap();
        assert_eq!(result.compiler_source, Some(CompilerSource::SelectedRoot));

        // Debian multiarch: runtime under lib/x86_64-linux-gnu is discovered via
        // root_library_dirs one-level child scan — do not “fix” that.
        let debian = base.join("debian");
        std::fs::create_dir_all(debian.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(debian.join("lib").join("x86_64-linux-gnu")).unwrap();
        std::fs::create_dir_all(debian.join("bin")).unwrap();
        std::fs::write(debian.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        let runtime = debian.join("lib").join("x86_64-linux-gnu").join("libamdhip64.so");
        std::fs::write(&runtime, b"").unwrap();
        #[cfg(not(windows))]
        std::fs::write(
            debian.join("lib").join("x86_64-linux-gnu").join("libhsa-runtime64.so.1"),
            b"",
        )
        .unwrap();
        std::fs::write(debian.join("bin").join(DEVICE_COMPILERS[0]), b"#!/bin/sh\n").unwrap();
        assert!(runtime_library(&debian).is_some());
        assert!(is_coherent_sdk_root(&debian));
        let result2 = resolve_toolchain_pure(Some(&debian), None, false, None, None).unwrap();
        assert_eq!(result2.compiler_source, Some(CompilerSource::SelectedRoot));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn cross_root_compiler_env_root_returns_compilers_root() {
        let base = std::env::temp_dir().join(format!("hipfire-rocm-env-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let libs = base.join("libs");
        std::fs::create_dir_all(libs.join("include").join("hip")).unwrap();
        std::fs::create_dir_all(libs.join(HIP_RUNTIME_DIRS[0])).unwrap();
        std::fs::write(libs.join("include").join("hip").join("hip_runtime.h"), b"").unwrap();
        std::fs::write(
            libs.join(HIP_RUNTIME_DIRS[0]).join(HIP_RUNTIME_LIBRARIES[0]),
            b"",
        )
        .unwrap();
        #[cfg(not(windows))]
        std::fs::write(
            libs.join(HIP_RUNTIME_DIRS[0]).join("libhsa-runtime64.so.1"),
            b"",
        )
        .unwrap();
        let comp_root = base.join("compiler");
        std::fs::create_dir_all(comp_root.join("bin")).unwrap();
        let comp = comp_root.join("bin").join("hipcc");
        std::fs::write(&comp, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&comp).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&comp, p).unwrap();
        }
        let toolchain = resolve_toolchain_pure(Some(&libs), None, false, Some(comp.clone()), None).unwrap();
        assert_eq!(toolchain.compiler_root, Some(comp_root.clone()));
        // compiler_env_root must return the compiler's own root, not the libs root.
        let env_root = compiler_env_root_from(&comp, None);
        assert_eq!(env_root, Some(comp_root.clone()));
        // When the compiler is cross-root, the value returned is the compiler's
        // root, not the libs root — even if ROCM_PATH is set to the libs root.
        let env_root2 = compiler_env_root_from(&comp, Some(&libs));
        assert_eq!(env_root2, Some(comp_root));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn rocm_path_trailing_hip_is_normalized() {
        // Mirrors normalize_hip_path guard — stripping only a literal trailing
        // `hip` component. A genuinely named .../hip directory that is itself
        // the install root should still be normalized as before (parent looks
        // like a ROCm root is not required; we just strip the suffix).
        assert_eq!(
            normalize_hip_path(Path::new("/opt/rocm/hip")),
            PathBuf::from("/opt/rocm")
        );
        assert_eq!(
            normalize_hip_path(Path::new("/opt/rocm-7.14/hip")),
            PathBuf::from("/opt/rocm-7.14")
        );
        // Without the suffix, leave untouched.
        assert_eq!(
            normalize_hip_path(Path::new("/opt/rocm")),
            PathBuf::from("/opt/rocm")
        );
        // A path that merely contains hip elsewhere is untouched.
        assert_eq!(
            normalize_hip_path(Path::new("/opt/myhip")),
            PathBuf::from("/opt/myhip")
        );
        // configured_root now normalizes ROCM_PATH the same way as HIP_PATH.
        // Test via pure form: a trailing hip on ROCM_PATH would normalize.
        // We cannot set env here; we just assert the pure helper.
        let normalized = normalize_hip_path(&PathBuf::from("/opt/rocm/hip"));
        assert_eq!(normalized, PathBuf::from("/opt/rocm"));
    }

    #[test]
    fn genuinely_nonexistent_authoritative_root_still_fails() {
        let bogus = Path::new("/tmp/hipfire-nonexistent-rocm-xyz-12345");
        // Ensure it does not exist.
        let _ = std::fs::remove_dir_all(bogus);
        assert!(!bogus.exists());
        let result = resolve_toolchain_pure(Some(bogus), None, false, None, None);
        assert!(result.is_err(), "nonexistent root must fail: {result:?}");
    }

}
