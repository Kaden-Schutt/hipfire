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
//!   4. The parent of a resolved device compiler on `PATH` (`hipcc`,
//!      `amdclang++`, …), which is how a `module load`-style environment
//!      identifies itself.
//!   5. `/opt/rocm`.
//!   6. Versioned siblings — `/opt/rocm-*` and `/opt/rocm/core-*` — newest
//!      first, so `core-7.14` beats `core-7`.
//!
//! Nothing here touches the GPU; it is pure path policy and is unit-tested
//! against a synthetic tree.

use std::path::{Path, PathBuf};

/// Device compilers, most specific first. `hipcc` is being wound down upstream
/// in favour of invoking the LLVM driver directly, and on ROCm 7.14 `hipcc` is
/// already a thin wrapper around `amdclang++`, so both are probed.
pub const DEVICE_COMPILERS: &[&str] = &["hipcc", "amdclang++", "amdclang", "clang++"];

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

/// Locate a tool on `PATH` and derive the ROCm root from it (`<root>/bin/tool`).
fn root_from_path_tools() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for tool in DEVICE_COMPILERS {
            let candidate = dir.join(tool);
            if !candidate.exists() {
                continue;
            }
            let resolved = std::fs::canonicalize(&candidate).unwrap_or(candidate);
            // <root>/bin/<tool> -> <root>
            if let Some(root) = resolved.parent().and_then(|b| b.parent()) {
                return Some(root.to_path_buf());
            }
        }
    }
    None
}

/// Ordered candidate ROCm roots. Entries are deduplicated but NOT filtered for
/// existence — callers that need existence should use [`root`].
pub fn roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };

    for var in ["HIPFIRE_ROCM_PATH", "ROCM_PATH"] {
        if let Some(v) = std::env::var_os(var) {
            if !v.is_empty() {
                push(PathBuf::from(v));
            }
        }
    }
    if let Some(v) = std::env::var_os("HIP_PATH") {
        if !v.is_empty() {
            push(normalize_hip_path(Path::new(&v)));
        }
    }
    if let Some(p) = root_from_path_tools() {
        push(p);
    }
    push(PathBuf::from("/opt/rocm"));
    for p in versioned_siblings(Path::new("/opt/rocm"), "core-") {
        push(p);
    }
    for p in versioned_siblings(Path::new("/opt"), "rocm-") {
        push(p);
    }
    out
}

/// Does this directory carry the device-compile prerequisites, i.e. is it a
/// real ROCm root rather than a shim that merely exists?
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

/// The first candidate root that is actually usable, falling back to the first
/// that merely exists.
///
/// Preferring completeness is what lets rule 6 (`/opt/rocm/core-*`) win on
/// installs where rule 5 (`/opt/rocm`) is a shim directory. On a conventional
/// install `/opt/rocm` is complete and is still chosen first, so the ordering
/// documented above is unchanged for everyone else. The `is_dir` fallback
/// keeps behaviour identical when nothing validates.
pub fn root() -> Option<PathBuf> {
    let candidates = roots();
    candidates
        .iter()
        .find(|p| is_complete_root(p))
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

/// Candidate load paths for a library, resolved roots first and the bare
/// sonames last so the dynamic loader still gets its turn (that is what makes
/// a correctly-configured `LD_LIBRARY_PATH` keep working).
///
/// `sonames` should be ordered most-preferred first, e.g.
/// `["libamdhip64.so", "libamdhip64.so.7"]`.
pub fn library_candidates(sonames: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for r in roots() {
        for libdir in ["lib", "lib64"] {
            for soname in sonames {
                let p = r.join(libdir).join(soname);
                if p.exists() {
                    out.push(p.to_string_lossy().into_owned());
                }
            }
        }
    }
    for soname in sonames {
        out.push((*soname).to_string());
    }
    out
}

/// Locate a ROCm tool (`hipcc`, `amdclang++`, `rocminfo`, …) under a resolved
/// root, falling back to bare `PATH` lookup.
pub fn tool(name: &str) -> Option<PathBuf> {
    for r in roots() {
        let p = r.join("bin").join(name);
        if p.exists() {
            return Some(p);
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|p| p.exists())
}

/// The device compiler this installation should use, most specific first.
pub fn device_compiler() -> Option<PathBuf> {
    DEVICE_COMPILERS.iter().find_map(|c| tool(c))
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
/// resolved on `PATH` the same way [`root_from_path_tools`] probes tools.
fn root_from_compiler(compiler: &Path) -> Option<PathBuf> {
    let resolved = if compiler.components().count() == 1 {
        // Bare tool name — walk PATH like root_from_path_tools.
        let name = compiler.as_os_str();
        let path = std::env::var_os("PATH")?;
        let mut found = None;
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(name);
            if candidate.exists() {
                found = Some(std::fs::canonicalize(&candidate).unwrap_or(candidate));
                break;
            }
        }
        found?
    } else {
        std::fs::canonicalize(compiler).unwrap_or_else(|_| compiler.to_path_buf())
    };
    // <root>/bin/<tool> -> <root>
    resolved
        .parent()
        .and_then(|bin| bin.parent())
        .map(|root| root.to_path_buf())
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
    fn library_candidates_always_end_with_bare_sonames() {
        let c = library_candidates(&["libamdhip64.so", "libamdhip64.so.7"]);
        assert!(c.len() >= 2);
        // The loader fallback must be last so an explicit root wins over it.
        assert_eq!(
            &c[c.len() - 2..],
            &["libamdhip64.so".to_string(), "libamdhip64.so.7".to_string()]
        );
    }

    #[test]
    fn roots_are_deduplicated_and_include_opt_rocm() {
        let r = roots();
        let mut seen = std::collections::HashSet::new();
        for p in &r {
            assert!(seen.insert(p.clone()), "duplicate root: {p:?}");
        }
        assert!(r.contains(&PathBuf::from("/opt/rocm")));
    }
}
