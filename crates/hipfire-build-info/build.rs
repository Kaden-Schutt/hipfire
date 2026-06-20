// SPDX-License-Identifier: Apache-2.0
// hipfire — build-time Git-derived version identity. See LICENSE / NOTICE.

//! Emits `VERGEN_GIT_DESCRIBE` (e.g. `v0.2.1-957-g6536c05a`) as a `rustc-env`
//! so the binaries can report a dev-build identity from `--version`.
//!
//! Uses `vergen-gitcl`, which shells out to the `git` CLI — so the embedded
//! string is byte-identical to the `chaingun-version` GitHub workflow's
//! `git describe` (same tool, same `--tags --dirty --match 'v[0-9]*'` flags).
//! vergen also emits the `cargo:rerun-if-changed` directives for the active
//! ref so the embedded SHA refreshes when new commits land.
//!
//! Fallback: a source checkout without `.git` (release tarball / packaged
//! crate) can't run `git describe`, so we emit the static `CARGO_PKG_VERSION`
//! instead of failing the build. `VERGEN_GIT_DESCRIBE` is therefore always set.

use std::error::Error;

fn main() {
    if let Err(e) = emit_git_describe() {
        // No `.git` (or `git` unavailable): keep the build green with the
        // static crate version. `git describe` parity is a dev-build feature.
        println!("cargo:warning=hipfire-build-info: git describe unavailable ({e}); using CARGO_PKG_VERSION");
        let pkg = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".into());
        println!("cargo:rustc-env=VERGEN_GIT_DESCRIBE=v{pkg}");
    }
}

fn emit_git_describe() -> Result<(), Box<dyn Error>> {
    use vergen_gitcl::{Emitter, Gitcl};
    // `git describe --tags --dirty --match 'v[0-9]*'` (matches the workflow).
    let gitcl = Gitcl::builder()
        .describe(true, true, Some("v[0-9]*"))
        .build();
    Emitter::default()
        .fail_on_error()
        .add_instructions(&gitcl)?
        .emit()?;
    Ok(())
}
