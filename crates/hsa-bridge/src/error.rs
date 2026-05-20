// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HSA error handling.

use std::fmt;

pub type HsaStatus = u32;
pub type HsaResult<T> = Result<T, HsaError>;

pub const HSA_STATUS_SUCCESS: u32 = 0;

#[derive(Debug)]
pub struct HsaError {
    pub code: HsaStatus,
    pub message: String,
}

impl HsaError {
    pub fn new(code: HsaStatus, context: &str) -> Self {
        Self {
            code,
            message: format!("{context} (hsa_status=0x{code:x})"),
        }
    }
}

impl fmt::Display for HsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HsaError(0x{:x}): {}", self.code, self.message)
    }
}

impl std::error::Error for HsaError {}

/// Turn a non-success HSA status into a typed error.
#[inline]
pub fn check(code: HsaStatus, context: &str) -> HsaResult<()> {
    if code == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HsaError::new(code, context))
    }
}
