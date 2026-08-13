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

fn main() -> ExitCode {
    let Some(root) = hipfire_config::rocm::root() else {
        let tried = hipfire_config::rocm::configured_root()
            .map(|(_, root)| {
                [
                    root.join("bin").join("hipcc"),
                    root.join("include").join("hip").join("hip_runtime.h"),
                    root.join("lib").join("libamdhip64.so"),
                    root.join("lib").join("libhsa-runtime64.so.1"),
                ]
                .into_iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        return fail("a complete ROCm installation", &tried);
    };
    let Some(hipcc) = hipfire_config::rocm::tool("hipcc") else {
        return fail(
            "the ROCm HIP compiler (hipcc)",
            &[root.join("bin").join("hipcc").display().to_string()],
        );
    };

    let missing = hipfire_config::rocm::missing_components(&root);
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

    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let hipcc = std::fs::canonicalize(&hipcc).unwrap_or(hipcc);
    println!("ROCM_ROOT={}", root.display());
    println!("HIPCC={}", hipcc.display());
    ExitCode::SUCCESS
}
