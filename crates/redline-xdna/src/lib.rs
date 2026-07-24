// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Rust-native amdxdna runtime used by Hipfire's optional XDNA overlay.
//!
//! The production path deliberately has no XRT dependency. GPU allocations
//! remain owned by HIP/ROCr, are exported as dma-bufs, and are imported here
//! as retained amdxdna GEM handles.

mod artifact;
mod runtime;

pub use artifact::{
    ArtifactBundle, ArtifactFile, ArtifactManifest, BindingAccess, BindingLayout,
    FirmwareCompatibility, FirmwareVersion, IoLayout, ProjectionArithmetic, ProjectionShape,
    SUPPORTED_ABI_VERSION, SUPPORTED_MANIFEST_VERSION,
};
pub use runtime::{
    resolve_device_path, Binding, Bo, BoKind, CommandRing, Device, DeviceMetadata, HardwareContext,
    Program, SubmissionTicket, SubmissionTiming,
};

use std::io;
use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, XdnaError>;

#[derive(Debug, thiserror::Error)]
pub enum XdnaError {
    #[error("{operation} failed: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("amdxdna ioctl {operation} failed: {source}")]
    Ioctl {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("invalid amdxdna response: {0}")]
    InvalidResponse(String),
    #[error("artifact manifest {path} is invalid: {message}")]
    ArtifactManifest { path: PathBuf, message: String },
    #[error("artifact {path} checksum mismatch: expected {expected}, got {actual}")]
    ArtifactChecksum {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("artifact compatibility failure: {0}")]
    IncompatibleArtifact(String),
    #[error("submission timed out after {timeout_ms} ms")]
    Timeout { timeout_ms: u32 },
    #[error("submission completed in ERT state {state}, expected 4")]
    TerminalState { state: u32 },
    #[error("command ABI accepts at most five bindings, got {0}")]
    TooManyBindings(usize),
    #[error("artifact ABI requires {expected} bindings, got {actual}")]
    BindingCount { expected: usize, actual: usize },
    #[error("binding {index} ({name}) access is {actual:?}, expected {expected:?}")]
    BindingAccess {
        index: usize,
        name: String,
        expected: BindingAccess,
        actual: BindingAccess,
    },
    #[error("binding {index} ({name}) has {actual} bytes, requires at least {minimum}")]
    BindingTooSmall {
        index: usize,
        name: String,
        minimum: u64,
        actual: usize,
    },
    #[error("binding {index} ({name}) address 0x{address:x} is not {alignment}-byte aligned")]
    BindingAlignment {
        index: usize,
        name: String,
        address: u64,
        alignment: u32,
    },
    #[error("buffer range offset={offset} length={length} exceeds BO length {bo_length}")]
    BufferRange {
        offset: u64,
        length: usize,
        bo_length: usize,
    },
    #[error("unsafe amdxdna device path {path}: {message}")]
    UnsafeDevicePath { path: PathBuf, message: String },
    #[error("amdxdna device discovery failed: {0}")]
    DeviceDiscovery(String),
}

impl XdnaError {
    pub(crate) fn io(operation: &'static str) -> Self {
        Self::Io {
            operation,
            source: io::Error::last_os_error(),
        }
    }

    pub(crate) fn ioctl(operation: &'static str) -> Self {
        Self::Ioctl {
            operation,
            source: io::Error::last_os_error(),
        }
    }
}
