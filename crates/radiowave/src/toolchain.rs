// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! ROCm 7.14 toolchain facts for radiowave.
//!
//! Verified on hipcc 7.14.60850 / AMD clang 23.0.0git under `/opt/rocm/core`
//! (ROCM_PATH-sensitive layout). `llc` is **absent** from this install, so
//! scheduler `-mllvm -amdgpu-*` knobs stay passthrough-only — no live
//! enumeration via `llc --help-hidden`.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Snapshot of the hipcc / amdclang toolchain radiowave will invoke.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolchainInfo {
    /// HIP package version string (e.g. `"7.14.60850-0000000"`).
    pub hip_version: String,
    /// AMD clang version string (e.g. `"23.0.0git"`).
    pub clang_version: String,
    /// Detected device offload architectures (host GPU and/or rocminfo).
    pub offload_arches: Vec<String>,
    /// Whether an `llc` binary sits next to hipcc / in the clang install dir.
    ///
    /// Mining proved `llc` absent on ROCm 7.14 core — expect `false`.
    pub llc_available: bool,
    /// Clang resource directory (`…/lib/clang/23`), derived from InstalledDir.
    pub resource_dir: PathBuf,
}

impl ToolchainInfo {
    /// When `llc` is missing, scheduler `-mllvm` knobs cannot be enumerated
    /// from the install. Callers must pass known flags through unchanged.
    pub fn scheduler_mllvm_passthrough_only(&self) -> bool {
        !self.llc_available
    }
}

/// Minimum accepted HIP major.minor for radiowave (ROCm 7.14 floor).
pub const MIN_HIP_MAJOR: u32 = 7;
pub const MIN_HIP_MINOR: u32 = 14;

/// Probe the toolchain rooted at `hipcc`.
///
/// Runs:
/// - `hipcc --version` (HIP + clang banner; needs `ROCM_PATH` / core layout)
/// - sibling `amdclang++ --version` (clang version + InstalledDir)
/// - offload-arch / amdgpu-arch **beside hipcc first**, then core paths,
///   then `rocminfo` name lines
/// - filesystem check for `llc` beside hipcc and under InstalledDir
///
/// # Errors
/// Returns [`Error::ToolFailed`] / [`Error::Io`] when version probes fail,
/// [`Error::UnsupportedHipVersion`] when HIP is below 7.14,
/// or [`Error::InvalidCertification`] when version text / arches cannot be
/// established.
pub fn probe(hipcc: &Path) -> Result<ToolchainInfo> {
    let hipcc_out = run_version(hipcc, "hipcc")?;
    let hip_version = parse_hip_version(&hipcc_out).ok_or_else(|| {
        Error::InvalidCertification(format!(
            "could not parse HIP version from hipcc --version output:\n{hipcc_out}"
        ))
    })?;
    ensure_hip_at_least(&hip_version, MIN_HIP_MAJOR, MIN_HIP_MINOR)?;

    let amdclang = sibling_of(hipcc, "amdclang++");
    let clang_out = if amdclang.is_file() {
        run_version(&amdclang, "amdclang++")?
    } else {
        // hipcc --version already embeds the AMD clang banner.
        hipcc_out.clone()
    };
    let clang_version = parse_clang_version(&clang_out)
        .or_else(|| parse_clang_version(&hipcc_out))
        .ok_or_else(|| {
            // 7.14-only: generic upstream `clang version` banners are rejected.
            // AMD's toolchain always prints `AMD clang version …`.
            Error::NonAmdClang(format!(
                "toolchain did not report an AMD clang version banner \
                 (requires ROCm >= 7.14 / amdclang); got:\n{clang_out}"
            ))
        })?;

    let installed_dir = parse_installed_dir(&clang_out)
        .or_else(|| parse_installed_dir(&hipcc_out))
        .map(PathBuf::from);

    let resource_dir = resolve_resource_dir(installed_dir.as_deref(), &amdclang, hipcc)?;
    let llc_available = llc_present(hipcc, installed_dir.as_deref());
    let offload_arches = probe_offload_arches(hipcc)?;
    if offload_arches.is_empty() {
        return Err(Error::InvalidCertification(
            "could not discover any offload architectures (offload-arch/amdgpu-arch/rocminfo)"
                .to_owned(),
        ));
    }

    Ok(ToolchainInfo {
        hip_version,
        clang_version,
        offload_arches,
        llc_available,
        resource_dir,
    })
}

/// Fail closed when `hip_version` is below `major.minor` (ROCm floor).
pub fn ensure_hip_at_least(hip_version: &str, major: u32, minor: u32) -> Result<()> {
    let (found_major, found_minor) = parse_hip_major_minor(hip_version).ok_or_else(|| {
        Error::InvalidCertification(format!(
            "could not parse structured HIP major.minor from `{hip_version}`"
        ))
    })?;
    if found_major > major || (found_major == major && found_minor >= minor) {
        return Ok(());
    }
    Err(Error::UnsupportedHipVersion {
        found: hip_version.to_owned(),
        required: format!("{major}.{minor}"),
    })
}

/// Extract `(major, minor)` from strings like `7.14.60850-0000000`.
pub fn parse_hip_major_minor(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.split(|c: char| !c.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn run_version(tool: &Path, label: &str) -> Result<String> {
    let output = Command::new(tool)
        .arg("--version")
        .output()
        .map_err(|err| {
            // Surface ROCM_PATH sensitivity: bare hipcc under /opt/rocm/core/bin
            // looks for /opt/rocm/lib/llvm without ROCM_PATH=/opt/rocm/core.
            Error::Io(std::io::Error::new(
                err.kind(),
                format!(
                    "{label} --version failed to spawn ({}): {err}",
                    tool.display()
                ),
            ))
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let combined = if stdout.trim().is_empty() {
        stderr.clone()
    } else if stderr.trim().is_empty() {
        stdout.clone()
    } else {
        format!("{stdout}{stderr}")
    };
    if !output.status.success() {
        return Err(Error::ToolFailed {
            tool: label.to_owned(),
            status: output
                .status
                .code()
                .map_or_else(|| "signal".to_owned(), |c| c.to_string()),
            stdout,
            stderr,
        });
    }
    Ok(combined)
}

/// Parse `HIP version: 7.14.60850-0000000` from hipcc --version text.
pub fn parse_hip_version(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("HIP version:") {
            let v = rest.trim();
            if !v.is_empty() {
                return Some(v.to_owned());
            }
        }
    }
    None
}

/// Parse `AMD clang version 23.0.0git (...)` from amdclang/hipcc --version text.
///
/// **AMD marker required** (7.14-only policy): only lines starting with
/// `AMD clang version` are accepted. Generic upstream `clang version N.M.P`
/// banners return `None` so callers can surface [`Error::NonAmdClang`].
pub fn parse_clang_version(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("AMD clang version ") {
            let ver = rest.split_whitespace().next().unwrap_or("").trim();
            if !ver.is_empty() {
                return Some(ver.to_owned());
            }
        }
    }
    None
}

/// Parse `InstalledDir: /opt/rocm/core-7.14/lib/llvm/bin`.
pub fn parse_installed_dir(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("InstalledDir:") {
            let dir = rest.trim();
            if !dir.is_empty() {
                return Some(dir.to_owned());
            }
        }
    }
    None
}

/// Parse whitespace-separated arch tokens from offload-arch / amdgpu-arch stdout.
pub fn parse_offload_arch_output(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in text.split_whitespace() {
        let t = token.trim();
        if t.starts_with("gfx") {
            out.push(t.to_owned());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Parse device `Name: gfxXXXX` lines from rocminfo (fallback path).
pub fn parse_rocminfo_arches(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        // rocminfo prints "Name:                    gfx1201" under Agent.
        if let Some(rest) = line.strip_prefix("Name:") {
            let name = rest.trim();
            if name.starts_with("gfx") {
                out.push(name.to_owned());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn sibling_of(tool: &Path, name: &str) -> PathBuf {
    tool.parent()
        .map(|p| p.join(name))
        .unwrap_or_else(|| PathBuf::from(name))
}

fn llc_present(hipcc: &Path, installed_dir: Option<&Path>) -> bool {
    let mut candidates = vec![sibling_of(hipcc, "llc")];
    if let Some(dir) = installed_dir {
        candidates.push(dir.join("llc"));
    }
    // Canonical 7.14 llvm bin (ROCM_PATH=/opt/rocm/core layout).
    candidates.push(PathBuf::from("/opt/rocm/core/lib/llvm/bin/llc"));
    candidates.push(PathBuf::from("/opt/rocm/core-7.14/lib/llvm/bin/llc"));
    candidates.iter().any(|p| p.is_file())
}

fn resolve_resource_dir(
    installed_dir: Option<&Path>,
    amdclang: &Path,
    hipcc: &Path,
) -> Result<PathBuf> {
    // Prefer live clang -print-resource-dir when a compiler binary exists.
    for compiler in [amdclang, hipcc] {
        if !compiler.is_file() {
            continue;
        }
        if let Ok(output) = Command::new(compiler).arg("-print-resource-dir").output() {
            if output.status.success() {
                let dir = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                if !dir.is_empty() {
                    return Ok(PathBuf::from(dir));
                }
            }
        }
    }

    if let Some(dir) = installed_dir {
        // InstalledDir = …/lib/llvm/bin → resource = …/lib/llvm/lib/clang/<maj>
        if let Some(resource) = resource_dir_from_installed(dir) {
            return Ok(resource);
        }
    }

    // Last resort stable 7.14 core layout.
    let fallback = PathBuf::from("/opt/rocm/core/lib/llvm/lib/clang/23");
    if fallback.is_dir() {
        return Ok(fallback);
    }
    let fallback714 = PathBuf::from("/opt/rocm/core-7.14/lib/llvm/lib/clang/23");
    if fallback714.is_dir() {
        return Ok(fallback714);
    }

    Err(Error::InvalidCertification(
        "could not resolve clang resource directory (InstalledDir missing; ROCM_PATH layout?)"
            .to_owned(),
    ))
}

fn resource_dir_from_installed(installed_dir: &Path) -> Option<PathBuf> {
    // …/lib/llvm/bin → …/lib/llvm/lib/clang
    let llvm_root = installed_dir.parent()?; // …/lib/llvm
    let clang_root = llvm_root.join("lib").join("clang");
    if !clang_root.is_dir() {
        return None;
    }
    // Prefer numeric version directory (e.g. "23").
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&clang_root)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    versions.pop()
}

fn probe_offload_arches(hipcc: &Path) -> Result<Vec<String>> {
    // Prefer tools from the same install as the probed hipcc to avoid mixing
    // `/opt/rocm/core` with a different hipcc root.
    let mut tools = Vec::new();
    tools.push(sibling_of(hipcc, "offload-arch"));
    tools.push(sibling_of(hipcc, "amdgpu-arch"));
    if let Some(parent) = hipcc.parent() {
        // …/bin/hipcc → try …/lib/llvm/bin companions
        let llvm_bin = parent
            .parent()
            .map(|root| root.join("lib").join("llvm").join("bin"));
        if let Some(llvm_bin) = llvm_bin {
            tools.push(llvm_bin.join("offload-arch"));
            tools.push(llvm_bin.join("amdgpu-arch"));
        }
    }
    // Documented TheRock / core fallbacks only after hipcc-local tools.
    tools.push(PathBuf::from("/opt/rocm/core/bin/offload-arch"));
    tools.push(PathBuf::from("/opt/rocm/core-7.14/bin/offload-arch"));
    tools.push(PathBuf::from("/opt/rocm/core/lib/llvm/bin/offload-arch"));
    tools.push(PathBuf::from("/opt/rocm/core/lib/llvm/bin/amdgpu-arch"));

    for tool in &tools {
        if !tool.is_file() {
            continue;
        }
        if let Ok(output) = Command::new(tool).output() {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                let arches = parse_offload_arch_output(&text);
                if !arches.is_empty() {
                    return Ok(arches);
                }
            }
        }
    }

    // rocminfo fallback when offload-arch is missing — still prefer hipcc root.
    let mut rocminfo_candidates = vec![sibling_of(hipcc, "rocminfo")];
    rocminfo_candidates.push(PathBuf::from("/opt/rocm/core/bin/rocminfo"));
    rocminfo_candidates.push(PathBuf::from("/opt/rocm/core-7.14/bin/rocminfo"));
    rocminfo_candidates.push(PathBuf::from("rocminfo"));

    for tool in &rocminfo_candidates {
        let mut cmd = Command::new(tool);
        if let Ok(output) = cmd.output() {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                let arches = parse_rocminfo_arches(&text);
                if !arches.is_empty() {
                    return Ok(arches);
                }
            }
        }
    }

    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed sample from hipcc --version on ROCm 7.14.60850 (no process spawn).
    const HIPCC_VERSION_SAMPLE: &str = "\
HIP version: 7.14.60850-0000000
AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm/core-7.14/lib/llvm/bin
";

    /// Fixed sample from amdclang++ --version on the same install.
    const AMDCLANG_VERSION_SAMPLE: &str = "\
AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm/core-7.14/lib/llvm/bin
";

    #[test]
    fn parses_hipcc_version_sample() {
        assert_eq!(
            parse_hip_version(HIPCC_VERSION_SAMPLE).as_deref(),
            Some("7.14.60850-0000000")
        );
        assert_eq!(
            parse_clang_version(HIPCC_VERSION_SAMPLE).as_deref(),
            Some("23.0.0git")
        );
        assert_eq!(
            parse_installed_dir(HIPCC_VERSION_SAMPLE).as_deref(),
            Some("/opt/rocm/core-7.14/lib/llvm/bin")
        );
    }

    #[test]
    fn parses_amdclang_version_sample() {
        assert_eq!(
            parse_clang_version(AMDCLANG_VERSION_SAMPLE).as_deref(),
            Some("23.0.0git")
        );
        assert!(parse_hip_version(AMDCLANG_VERSION_SAMPLE).is_none());
        assert_eq!(
            parse_installed_dir(AMDCLANG_VERSION_SAMPLE).as_deref(),
            Some("/opt/rocm/core-7.14/lib/llvm/bin")
        );
    }
    #[test]
    fn parse_clang_version_requires_amd_marker() {
        // Acceptance: AMD banner (as amdclang / hipcc print).
        assert_eq!(
            parse_clang_version(AMDCLANG_VERSION_SAMPLE).as_deref(),
            Some("23.0.0git")
        );
        assert_eq!(
            parse_clang_version(HIPCC_VERSION_SAMPLE).as_deref(),
            Some("23.0.0git")
        );

        // Rejection: generic upstream clang without the AMD marker.
        let upstream = "\
clang version 23.0.0
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/bin
";
        assert!(
            parse_clang_version(upstream).is_none(),
            "generic upstream clang must not parse as AMD clang"
        );
        assert!(parse_clang_version("clang version 18.1.0").is_none());
        assert!(parse_clang_version("not a version banner").is_none());
        assert!(parse_clang_version("").is_none());

        // Named error path used by probe when only upstream clang is present.
        let err = Error::NonAmdClang(
            "toolchain did not report an AMD clang version banner \
             (requires ROCm >= 7.14 / amdclang)"
                .to_owned(),
        );
        let msg = err.to_string();
        assert!(
            msg.contains("AMD clang") && msg.contains("requires ROCm >= 7.14"),
            "{msg}"
        );
    }

    #[test]
    fn parse_hip_version_rejects_noise() {
        assert!(parse_hip_version("not a version banner").is_none());
        assert!(parse_hip_version("HIP version:").is_none());
        assert!(parse_hip_version("").is_none());
    }

    #[test]
    fn hip_major_minor_and_floor_gate() {
        assert_eq!(parse_hip_major_minor("7.14.60850-0000000"), Some((7, 14)));
        assert_eq!(parse_hip_major_minor("7.13.0"), Some((7, 13)));
        assert!(parse_hip_major_minor("not-a-version").is_none());

        assert!(ensure_hip_at_least("7.14.60850-0000000", 7, 14).is_ok());
        assert!(ensure_hip_at_least("8.0.0", 7, 14).is_ok());
        let err = ensure_hip_at_least("7.13.1", 7, 14).unwrap_err();
        match err {
            Error::UnsupportedHipVersion { found, required } => {
                assert!(found.starts_with("7.13"));
                assert_eq!(required, "7.14");
            }
            other => panic!("expected UnsupportedHipVersion, got {other}"),
        }
    }

    #[test]
    fn parse_offload_arch_and_rocminfo_samples() {
        assert_eq!(
            parse_offload_arch_output("gfx1201\n"),
            vec!["gfx1201".to_owned()]
        );
        assert_eq!(
            parse_offload_arch_output("gfx1200 gfx1201 gfx1201"),
            vec!["gfx1200".to_owned(), "gfx1201".to_owned()]
        );

        let rocminfo = "\
*******
  Name:                    gfx1201
  Uuid:                    GPU-XX
  Name:                    AMD Ryzen
";
        assert_eq!(parse_rocminfo_arches(rocminfo), vec!["gfx1201".to_owned()]);
    }

    /// llc-absent policy: scheduler -mllvm knobs remain passthrough-only.
    #[test]
    fn llc_absent_forces_mllvm_passthrough_only() {
        let info = ToolchainInfo {
            hip_version: "7.14.60850-0000000".to_owned(),
            clang_version: "23.0.0git".to_owned(),
            offload_arches: vec!["gfx1201".to_owned()],
            llc_available: false,
            resource_dir: PathBuf::from("/opt/rocm/core-7.14/lib/llvm/lib/clang/23"),
        };
        assert!(info.scheduler_mllvm_passthrough_only());
        // Enumeration is forbidden when llc is missing — no help-hidden scrape.
        assert!(!info.llc_available);

        let with_llc = ToolchainInfo {
            llc_available: true,
            ..info.clone()
        };
        assert!(!with_llc.scheduler_mllvm_passthrough_only());
    }
}
