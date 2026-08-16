// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal FFI wrapper around librocsolver.so for FP64 Cholesky / triangular inverse.
//!
//! Mirrors the dlopen pattern in `rocblas.rs`: absence of librocsolver is a
//! recoverable runtime error so the engine/quantizer still builds + runs CPU-only.
//! rocSOLVER shares `rocblas_handle`; this module **does not** create a second
//! handle via `rocblas_create_handle`. Callers must supply an existing
//! `Rocblas` handle (see [`Rocsolver::load`]) and keep the `Rocblas` instance
//! alive while the `Rocsolver` is used. `Rocsolver` never destroys the handle.
//!
//! Bound symbols:
//! - `rocsolver_dpotrf`  — FP64 Cholesky factorization
//! - `rocsolver_dtrtri` — FP64 triangular inverse
//! - `rocsolver_dpotri` — FP64 inverse from Cholesky factor (optional, clean fit)
//!
//! Singularity (`info > 0`) is surfaced as `RocsolverError::NotPositiveDefinite`
//! distinct from transport/status failures, so the GPTQ caller can drive the
//! adaptive-damping retry ladder (`SingularEvenWithMaxDamp`) rather than falling
//! back to plain MQ4 on a missing library.

use crate::rocblas::Rocblas;
use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;

/// rocSOLVER / rocBLAS status codes (from rocblas-types.h).
pub const ROCSOLVER_STATUS_SUCCESS: u32 = 0;

/// rocBLAS fill mode for triangular storage (`rocblas_fill`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocblasFill {
    Upper = 121,
    Lower = 122,
    Full = 123,
}

/// rocBLAS diagonal mode (`rocblas_diagonal`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocblasDiagonal {
    NonUnit = 131,
    Unit = 132,
}

/// Errors from rocSOLVER init / calls.
///
/// `NotPositiveDefinite` corresponds to `info > 0` from the LAPACK convention:
/// the leading minor of order `info` is not positive definite (dpotrf/dpotri) or
/// the triangular matrix has a zero diagonal at `info` (dtrtri). Callers must
/// be able to distinguish this from `LibraryUnavailable` / `Status` to drive
/// damping retries.
#[derive(Debug)]
pub enum RocsolverError {
    /// Library could not be dlopened (soft failure → fall back to CPU).
    LibraryUnavailable { context: String },
    /// Symbol missing from the loaded library.
    SymbolMissing { context: String },
    /// rocSOLVER/rocBLAS returned a non-zero `rocblas_status`.
    Status { status: u32, context: String },
    /// Matrix not positive definite / singular. `info` is 1-indexed LAPACK `info`.
    NotPositiveDefinite { info: i32 },
    /// Invalid argument passed to the wrapper itself.
    InvalidArgument(String),
}

impl std::fmt::Display for RocsolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryUnavailable { context } => write!(f, "rocSOLVER unavailable: {context}"),
            Self::SymbolMissing { context } => write!(f, "rocSOLVER symbol missing: {context}"),
            Self::Status { status, context } => write!(f, "rocSOLVER error {status} in {context}"),
            Self::NotPositiveDefinite { info } => {
                write!(f, "rocSOLVER: matrix not positive definite (info={info})")
            }
            Self::InvalidArgument(s) => write!(f, "rocSOLVER invalid argument: {s}"),
        }
    }
}

impl std::error::Error for RocsolverError {}

pub type RocsolverResult<T> = Result<T, RocsolverError>;

type RocblasHandle = *mut c_void;

type RocsolverDpotrfFn = unsafe extern "C" fn(
    RocblasHandle,
    c_uint, // uplo (rocblas_fill)
    c_int,  // n
    *mut f64,
    c_int, // A, lda
    *mut c_int,
) -> u32;

type RocsolverDpotriFn = unsafe extern "C" fn(
    RocblasHandle,
    c_uint, // uplo
    c_int,  // n
    *mut f64,
    c_int, // A, lda
    *mut c_int,
) -> u32;

type RocsolverDtrtriFn = unsafe extern "C" fn(
    RocblasHandle,
    c_uint, // uplo
    c_uint, // diag
    c_int,  // n
    *mut f64,
    c_int, // A, lda
    *mut c_int,
) -> u32;

/// Loaded rocSOLVER library + resolved function pointers.
///
/// Shares the rocBLAS handle from [`Rocblas`]; this struct holds `Library` to
/// keep the DSO alive but never owns the handle — `Rocblas` remains the sole
/// owner and must outlive `Rocsolver`.
pub struct Rocsolver {
    _lib: Library,
    handle: RocblasHandle,
    fn_dpotrf: RocsolverDpotrfFn,
    fn_dtrtri: RocsolverDtrtriFn,
    fn_dpotri: Option<RocsolverDpotriFn>,
}

impl Rocsolver {
    /// Attempt to dlopen `librocsolver.so` and resolve the FP64 symbols we use.
    ///
    /// Reuses the existing rocBLAS handle from `rocblas` — no second
    /// `rocblas_create_handle` is issued. The caller must keep `rocblas` alive
    /// for the lifetime of the returned `Rocsolver`; this is a raw-pointer tie
    /// (no `Drop` of the handle here).
    ///
    /// On failure (library missing / required symbol missing) returns
    /// `LibraryUnavailable` / `SymbolMissing` so the caller can fall back to the
    /// CPU path. This is the same soft-failure contract as `Rocblas::load`.
    pub fn load(rocblas: &Rocblas) -> RocsolverResult<Self> {
        Self::load_from_handle(rocblas.handle())
    }

    /// As [`Self::load`] but from a raw `rocblas_handle`.
    pub fn load_from_handle(handle: RocblasHandle) -> RocsolverResult<Self> {
        if handle.is_null() {
            return Err(RocsolverError::InvalidArgument("rocBLAS handle is null".into()));
        }
        let candidates = hipfire_config::rocm::library_candidates(&[
            "librocsolver.so",
            "librocsolver.so.1",
            "librocsolver.so.0",
            "librocsolver.so.0.6",
        ]);
        let lib = candidates
            .iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| RocsolverError::LibraryUnavailable {
                context: format!(
                    "dlopen librocsolver.so failed. Tried: {}",
                    candidates.join(", ")
                ),
            })?;

        unsafe {
            let fn_dpotrf: Symbol<RocsolverDpotrfFn> = lib.get(b"rocsolver_dpotrf").map_err(|e| {
                RocsolverError::SymbolMissing {
                    context: format!("resolve rocsolver_dpotrf: {e}"),
                }
            })?;
            let fn_dtrtri: Symbol<RocsolverDtrtriFn> = lib.get(b"rocsolver_dtrtri").map_err(|e| {
                RocsolverError::SymbolMissing {
                    context: format!("resolve rocsolver_dtrtri: {e}"),
                }
            })?;
            // dpotri is a clean fit alongside dpotrf/dtrtri but not strictly
            // required; keep it optional so older librocsolver still provides
            // Cholesky + triangular inverse.
            let fn_dpotri = lib
                .get::<RocsolverDpotriFn>(b"rocsolver_dpotri")
                .ok()
                .map(|s| *s);

            let fn_dpotrf = *fn_dpotrf;
            let fn_dtrtri = *fn_dtrtri;

            Ok(Self {
                _lib: lib,
                handle,
                fn_dpotrf,
                fn_dtrtri,
                fn_dpotri,
            })
        }
    }

    /// Whether `rocsolver_dpotri` was resolved (optional symbol).
    pub fn has_dpotri(&self) -> bool {
        self.fn_dpotri.is_some()
    }

    /// FP64 Cholesky factorization: `A = L*L^T` or `U^T*U`.
    ///
    /// Wraps `rocsolver_dpotrf(handle, uplo, n, A, lda, info)`.
    /// `A` is a device pointer to an `n×n` column-major matrix; on entry the
    /// symmetric matrix, on exit the Cholesky factor in the selected triangle.
    /// `info` is a pointer to `rocblas_int` **on the device** per rocSOLVER
    /// docs; the wrapper checks `rocblas_status` first and then interprets
    /// `*info` as `NotPositiveDefinite(info)` when `info > 0`. `info == 0` is
    /// success.
    ///
    /// # Info handling
    ///
    /// rocSOLVER reports singularity via `*info`, not via `rocblas_status`.
    /// A `rocblas_status == success` with `*info = j > 0` means the leading
    /// minor of order `j` is not positive definite — the GPTQ caller maps this
    /// to "retry with `10×` damping" rather than to "library missing". This
    /// method surfaces that as `Err(RocsolverError::NotPositiveDefinite{info:j})`
    /// distinct from `Err(Status{...})` or `LibraryUnavailable`.
    ///
    /// The `info` value is read with `ptr::read` after the call. On MI300X
    /// `info` must reside in device memory written by the library; callers that
    /// use a device allocation must ensure the memory is accessible from the
    /// host for this read (e.g. via managed/unified allocation or a
    /// host-visible staging copy and stream synchronization). For cargo-check /
    /// CPU fallback this is not exercised.
    ///
    /// # Safety
    ///
    /// `A` and `info` must be valid for the rocSOLVER call and remain alive
    /// until the handle's stream completes. `handle` must be the same handle
    /// supplied at `load()` and must remain valid.
    pub unsafe fn dpotrf(
        &self,
        uplo: RocblasFill,
        n: i32,
        a: *mut f64,
        lda: i32,
        info: *mut c_int,
    ) -> RocsolverResult<()> {
        let st = (self.fn_dpotrf)(self.handle, uplo as c_uint, n, a, lda, info);
        check_status(st, "rocsolver_dpotrf")?;
        check_info(info, "rocsolver_dpotrf")
    }

    /// FP64 triangular inverse.
    ///
    /// Wraps `rocsolver_dtrtri(handle, uplo, diag, n, A, lda, info)`. Inverts
    /// a triangular matrix in place. `info > 0` means `A[info,info]` is zero
    /// (singular) and is surfaced as `NotPositiveDefinite`.
    ///
    /// # Safety
    ///
    /// Same pointer contract as [`Self::dpotrf`]; `info` device integer.
    pub unsafe fn dtrtri(
        &self,
        uplo: RocblasFill,
        diag: RocblasDiagonal,
        n: i32,
        a: *mut f64,
        lda: i32,
        info: *mut c_int,
    ) -> RocsolverResult<()> {
        let st = (self.fn_dtrtri)(
            self.handle,
            uplo as c_uint,
            diag as c_uint,
            n,
            a,
            lda,
            info,
        );
        check_status(st, "rocsolver_dtrtri")?;
        check_info(info, "rocsolver_dtrtri")
    }

    /// FP64 inverse from Cholesky factor (optional).
    ///
    /// Wraps `rocsolver_dpotri(handle, uplo, n, A, lda, info)` when the symbol
    /// is present. Computes `A^{-1}` from the factor produced by `dpotrf`.
    /// Soft-fails with `SymbolMissing` when the library omits it, mirroring the
    /// rocBLAS optional-symbol pattern.
    ///
    /// # Safety
    ///
    /// Same pointer contract as [`Self::dpotrf`].
    pub unsafe fn dpotri(
        &self,
        uplo: RocblasFill,
        n: i32,
        a: *mut f64,
        lda: i32,
        info: *mut c_int,
    ) -> RocsolverResult<()> {
        let Some(dpotri) = self.fn_dpotri else {
            return Err(RocsolverError::SymbolMissing {
                context: "rocsolver_dpotri unavailable (symbol not resolved)".into(),
            });
        };
        let st = dpotri(self.handle, uplo as c_uint, n, a, lda, info);
        check_status(st, "rocsolver_dpotri")?;
        check_info(info, "rocsolver_dpotri")
    }
}

fn check_status(status: u32, context: &str) -> RocsolverResult<()> {
    if status == ROCSOLVER_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(RocsolverError::Status {
            status,
            context: context.into(),
        })
    }
}

fn check_info(info: *mut c_int, _context: &str) -> RocsolverResult<()> {
    if info.is_null() {
        return Ok(());
    }
    // Host-visible read. On GPU the pointer is device memory; callers that
    // use device allocations must ensure visibility (managed memory or stream
    // sync + D2H copy) before this read is meaningful — the wrapper cannot
    // perform a hipMemcpy without a HIP dependency, so it reads as host memory
    // and documents the requirement. For CPU/unit-test paths this suffices and
    // soft-failure is preserved.
    let val = unsafe { std::ptr::read(info) };
    if val == 0 {
        Ok(())
    } else if val > 0 {
        Err(RocsolverError::NotPositiveDefinite { info: val })
    } else {
        Err(RocsolverError::Status {
            status: 0,
            context: format!("rocsolver returned negative info {val}"),
        })
    }
}

// rocSOLVER handle is bound to a GPU context; not shared across threads without sync.
unsafe impl Send for Rocsolver {}
unsafe impl Sync for Rocsolver {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_and_diagonal_values_match_headers() {
        assert_eq!(RocblasFill::Upper as u32, 121);
        assert_eq!(RocblasFill::Lower as u32, 122);
        assert_eq!(RocblasDiagonal::Unit as u32, 132);
        assert_eq!(RocblasDiagonal::NonUnit as u32, 131);
    }

    #[test]
    fn not_positive_definite_is_distinct_from_status() {
        let e = RocsolverError::NotPositiveDefinite { info: 3 };
        assert!(matches!(e, RocsolverError::NotPositiveDefinite { .. }));
        let s = RocsolverError::Status {
            status: 1,
            context: "x".into(),
        };
        assert!(matches!(s, RocsolverError::Status { .. }));
    }

    #[test]
    fn check_info_maps_positive_to_npd() {
        let mut info: c_int = 2;
        let r = check_info(&mut info as *mut c_int, "test");
        assert!(matches!(
            r,
            Err(RocsolverError::NotPositiveDefinite { info: 2 })
        ));
        let mut info2: c_int = 0;
        assert!(check_info(&mut info2 as *mut c_int, "test").is_ok());
    }
}
