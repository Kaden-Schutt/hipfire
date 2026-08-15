// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use std::path::Path;
use std::process::ExitCode;

fn fail(component: &str, tried: &[String]) -> ExitCode {
    eprintln!(
        "{}",
        hipfire_config::rocm::resolution_failure(component, tried)
    );
    ExitCode::FAILURE
}

fn fail_msg(msg: &str) -> ExitCode {
    eprintln!("{msg}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    // Prefer the new toolchain resolver so cross-root and HIPFIRE_HIPCC are
    // handled uniformly. Fall back to the legacy root/tool checks when the
    // resolver reports no root at all, so error messages stay actionable.
    let toolchain = match hipfire_config::rocm::resolve_toolchain() {
        Ok(tc) => tc,
        Err(msg) => {
            // If the error already contains a full resolution_failure, print it
            // directly. Otherwise, mimic the old branch-specific failures for
            // compatibility with scripts that parse the message.
            return fail_msg(&msg);
        }
    };

    // Preserve the original “missing components” gate: a compiler-only root
    // that lacks headers must still fail with “complete ROCm HIP development
    // stack”, even though a compiler was found under that root.
    let missing = hipfire_config::rocm::missing_components(&toolchain.root);
    if !missing.is_empty() {
        let tried = missing
            .iter()
            .map(|component| component.probed.display().to_string())
            .collect::<Vec<_>>();
        return fail("a complete ROCm HIP development stack", &tried);
    }

    #[cfg(not(windows))]
    {
        let hsa_candidates = hipfire_config::rocm::library_candidates(&[
            "libhsa-runtime64.so.1",
            "libhsa-runtime64.so",
        ]);
        if !hsa_candidates
            .iter()
            .map(Path::new)
            .any(|candidate| candidate.is_file())
        {
            return fail("the HSA runtime (libhsa-runtime64.so)", &hsa_candidates);
        }
    }

    // The toolchain should always have a compiler when Ok, but be defensive.
    let Some(hipcc) = toolchain.compiler.clone() else {
        return fail(
            "the ROCm HIP compiler (hipcc)",
            &[toolchain.root.join("bin").join("hipcc").display().to_string()],
        );
    };

    let root = std::fs::canonicalize(&toolchain.root).unwrap_or(toolchain.root.clone());
    let hipcc = std::fs::canonicalize(&hipcc).unwrap_or(hipcc);
    let compiler_root = toolchain
        .compiler_root
        .as_ref()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))
        .unwrap_or_else(|| root.clone());
    let runtime_lib = hipfire_config::rocm::runtime_library(&root)
        .map(|p| std::fs::canonicalize(&p).unwrap_or(p))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "not found".to_string());
    let source = toolchain
        .compiler_source
        .as_ref()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let version = hipfire_config::rocm::version_for_root(&root)
        .or_else(hipfire_config::rocm::version)
        .unwrap_or_else(|| "unknown".to_string());

    // Keep historical lines for script compatibility.
    println!("ROCM_ROOT={}", root.display());
    println!("HIPCC={}", hipcc.display());
    // Provenance extensions.
    println!("HIPCC_SOURCE={}", source);
    println!("HIPCC_ROOT={}", compiler_root.display());
    println!("HIP_RUNTIME={}", runtime_lib);
    println!("ROCM_VERSION={}", version);

    // Cross-root warning goes to stderr so stdout stays machine-parseable.
    for line in hipfire_config::rocm::toolchain_warnings(&toolchain) {
        eprintln!("{line}");
    }

    ExitCode::SUCCESS
}
