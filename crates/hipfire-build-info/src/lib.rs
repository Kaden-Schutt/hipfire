// SPDX-License-Identifier: Apache-2.0
// hipfire — build-time Git-derived version identity. See LICENSE / NOTICE.

//! Shared `--version` string for hipfire binaries.
//!
//! [`VERSION`] is the Git-derived dev identity (`git describe --tags --dirty
//! --match 'v[0-9]*'`, e.g. `v0.3.0-957-g6536c05a`) embedded at build time by
//! `build.rs` via `vergen-gitcl`. On a checkout without `.git` it falls back to
//! the static `vCARGO_PKG_VERSION`. This matches the `chaingun-version` CI
//! workflow's identity (same `git` invocation), so the binary self-reports the
//! same string the workflow publishes.

/// Git-derived dev-build version (e.g. `v0.3.0-957-g6536c05a`), or
/// `vCARGO_PKG_VERSION` when built without a `.git` (release tarball).
pub const VERSION: &str = env!("VERGEN_GIT_DESCRIBE");
