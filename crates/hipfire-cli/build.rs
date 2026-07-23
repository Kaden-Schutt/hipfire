// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn clean_env(value: String) -> String {
    value.replace(['\r', '\n'], " ")
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest.join("../..");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let commit = env::var("HIPFIRE_BUILD_COMMIT_OVERRIDE")
        .ok()
        .or_else(|| git(&repo, &["rev-parse", "--verify", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    let reference = env::var("HIPFIRE_BUILD_REF_OVERRIDE")
        .ok()
        .or_else(|| git(&repo, &["describe", "--tags", "--exact-match", "HEAD"]))
        .or_else(|| git(&repo, &["symbolic-ref", "--short", "HEAD"]))
        .unwrap_or_else(|| "detached".into());
    let dirty = git(&repo, &["status", "--porcelain", "--untracked-files=no"])
        .is_some_and(|status| !status.is_empty());
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    let short = commit.get(..12).unwrap_or(&commit);
    let build_version = format!(
        "{version} ({short}; {reference}{})",
        if dirty { "; dirty" } else { "" }
    );

    println!("cargo:rustc-env=HIPFIRE_BUILD_COMMIT={}", clean_env(commit));
    println!("cargo:rustc-env=HIPFIRE_BUILD_REF={}", clean_env(reference));
    println!("cargo:rustc-env=HIPFIRE_BUILD_DIRTY={dirty}");
    println!("cargo:rustc-env=HIPFIRE_BUILD_TARGET={}", clean_env(target));
    println!(
        "cargo:rustc-env=HIPFIRE_BUILD_VERSION={}",
        clean_env(build_version)
    );
    println!("cargo:rerun-if-env-changed=HIPFIRE_BUILD_COMMIT_OVERRIDE");
    println!("cargo:rerun-if-env-changed=HIPFIRE_BUILD_REF_OVERRIDE");
    if let Some(head) = git(&repo, &["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    if let Some(index) = git(&repo, &["rev-parse", "--git-path", "index"]) {
        println!("cargo:rerun-if-changed={index}");
    }
}
