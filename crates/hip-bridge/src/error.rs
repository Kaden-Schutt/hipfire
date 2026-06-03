// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Error types for HIP runtime operations.

use std::ffi::CStr;
use std::fmt;

/// Raw HIP error code.
pub type HipErrorCode = u32;

/// HIP operation result.
pub type HipResult<T> = Result<T, HipError>;

/// `hipErrorInvalidImage` — the device code object handed to `hipModuleLoad`
/// is not valid for this GPU (wrong ISA, or a stale cross-build/cross-toolchain
/// `.hsaco` left in a shared kernel cache). Recoverable by recompiling from source.
pub const HIP_ERROR_INVALID_IMAGE: HipErrorCode = 200;
pub const HIP_ERROR_PEER_ACCESS_UNSUPPORTED: HipErrorCode = 217;
pub const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipErrorCode = 704;
pub const HIP_ERROR_PEER_ACCESS_NOT_ENABLED: HipErrorCode = 705;

#[derive(Debug)]
pub struct HipError {
    pub code: HipErrorCode,
    pub message: String,
}

impl HipError {
    pub fn new(code: HipErrorCode, context: &str) -> Self {
        Self {
            code,
            message: format!("{context} (hipError={code})"),
        }
    }

    pub(crate) fn from_code(
        code: HipErrorCode,
        context: &str,
        get_string: Option<&unsafe extern "C" fn(u32) -> *const i8>,
    ) -> Self {
        let detail = get_string
            .and_then(|f| {
                let ptr = unsafe { f(code) };
                if ptr.is_null() {
                    None
                } else {
                    Some(
                        unsafe { CStr::from_ptr(ptr) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                }
            })
            .unwrap_or_else(|| format!("error code {code}"));
        Self {
            code,
            message: format!("{context}: {detail}"),
        }
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HipError({}): {}", self.code, self.message)
    }
}

impl std::error::Error for HipError {}
