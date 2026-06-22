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

pub const HIP_ERROR_PEER_ACCESS_UNSUPPORTED: HipErrorCode = 217;
pub const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipErrorCode = 704;
pub const HIP_ERROR_PEER_ACCESS_NOT_ENABLED: HipErrorCode = 705;

/// Sentinel code for an "unsupported dispatch route": a requested (op, dtype,
/// arch) combination that hipfire has no kernel path for. Deliberately outside
/// the real HIP error-code range so consumers (eval, serving admission) can
/// classify it as a *capability gap* — not an infra failure or crash — via
/// [`HipError::is_unsupported`]. This is the typed substitute for the old
/// "silently misroute into a kernel that panics" behavior.
pub const HIPFIRE_ERROR_UNSUPPORTED: HipErrorCode = 0xF1F1_0001;

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

    /// Build an "unsupported dispatch route" error — a capability gap, not a
    /// crash. `context` should name the op, dtype, and arch, e.g.
    /// `"weight_gemv: no route for dtype Rq5Protect on gfx1103"`.
    pub fn unsupported(context: &str) -> Self {
        Self {
            code: HIPFIRE_ERROR_UNSUPPORTED,
            message: format!("unsupported dispatch: {context}"),
        }
    }

    /// True iff this is a capability-gap error (see [`HipError::unsupported`]).
    /// Lets a caller distinguish "this config isn't implemented here" from a
    /// genuine runtime failure, so it can skip/mark rather than crash.
    pub fn is_unsupported(&self) -> bool {
        self.code == HIPFIRE_ERROR_UNSUPPORTED
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_is_classified_apart_from_real_errors() {
        let gap = HipError::unsupported("weight_gemv: no route for dtype Rq5Protect on gfx1103");
        assert!(gap.is_unsupported());
        assert_eq!(gap.code, HIPFIRE_ERROR_UNSUPPORTED);
        assert!(gap.message.contains("Rq5Protect"));

        // A genuine HIP runtime failure must NOT be mistaken for a capability gap.
        let real = HipError::new(700, "out of memory");
        assert!(!real.is_unsupported());
    }
}
