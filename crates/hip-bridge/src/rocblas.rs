// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal FFI wrapper around librocblas.so for MI300X MFMA-accelerated GEMMs.
//!
//! Why rocBLAS and not hipBLASLt: rocBLAS has a single entry point
//! (`rocblas_gemm_ex`) that internally dispatches to MFMA-tuned kernels on
//! CDNA3; hipBLASLt's API is more powerful but requires matrix descriptors
//! and algorithm selection handles that add boilerplate without a perf win
//! for the straight-forward M×K · K×N GEMMs we need.
//!
//! Loaded lazily via `libloading`; absence of librocblas is a recoverable
//! runtime error so the engine still builds + runs without it.

use crate::Stream;
use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;

/// Errors from rocBLAS init / calls. Thin wrapper; we surface the rocBLAS
/// status code for debugging.
#[derive(Debug)]
pub struct RocblasError {
    pub status: u32,
    pub context: String,
}

impl std::fmt::Display for RocblasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rocBLAS error {} in {}", self.status, self.context)
    }
}
impl std::error::Error for RocblasError {}

pub type RocblasResult<T> = Result<T, RocblasError>;

/// rocBLAS status codes (from rocblas-types.h).
pub const ROCBLAS_STATUS_SUCCESS: u32 = 0;

/// rocBLAS operation types (from rocblas-types.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum RocblasOperation {
    None = 111,
    Transpose = 112,
    ConjugateTranspose = 113,
}

/// rocBLAS datatypes (from rocblas-types.h). Only the ones we currently use.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocblasDatatype {
    F16 = 150,
    F32 = 151,
    Bf16 = 168,
}

/// rocBLAS algorithm — standard path.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum RocblasGemmAlgo {
    Standard = 160,
    SolutionIndex = 161,
}

type RocblasHandle = *mut c_void;

/// Loaded rocBLAS library + resolved function pointers.
pub struct Rocblas {
    _lib: Library,
    handle: RocblasHandle,

    fn_destroy_handle: unsafe extern "C" fn(RocblasHandle) -> u32,
    fn_set_stream: unsafe extern "C" fn(RocblasHandle, *mut c_void) -> u32,
    fn_gemm_ex: unsafe extern "C" fn(
        RocblasHandle,
        c_uint,
        c_uint, // transA, transB
        c_int,
        c_int,
        c_int,         // m, n, k
        *const c_void, // alpha (pointer to scalar of compute_type)
        *const c_void,
        c_uint,
        c_int, // A, a_type, lda
        *const c_void,
        c_uint,
        c_int,         // B, b_type, ldb
        *const c_void, // beta
        *const c_void,
        c_uint,
        c_int, // C, c_type, ldc
        *mut c_void,
        c_uint,
        c_int,  // D, d_type, ldd
        c_uint, // compute_type
        c_uint, // algo
        i32,
        u32, // solution_index, flags
    ) -> u32,
}

impl Rocblas {
    /// Attempt to dlopen librocblas.so and resolve the subset of symbols we use.
    /// On failure (library missing / symbol missing), returns an error that the
    /// caller can treat as "rocBLAS unavailable, fall back to hand-rolled kernels".
    pub fn load() -> RocblasResult<Self> {
        let candidates = [
            "librocblas.so",
            "librocblas.so.7",
            "librocblas.so.6",
            "librocblas.so.5",
            "/opt/rocm/lib/librocblas.so",
            "/opt/rocm/lib/librocblas.so.7",
            "/opt/rocm/lib/librocblas.so.6",
            "/opt/rocm/lib/librocblas.so.5",
        ];
        let lib = candidates
            .iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| RocblasError {
                status: 0,
                context: "dlopen librocblas.so (tried several names) failed".into(),
            })?;

        unsafe {
            let fn_create_handle: Symbol<unsafe extern "C" fn(*mut RocblasHandle) -> u32> = lib
                .get(b"rocblas_create_handle")
                .map_err(|e| RocblasError {
                    status: 0,
                    context: format!("resolve rocblas_create_handle: {e}"),
                })?;
            let fn_destroy_handle: Symbol<unsafe extern "C" fn(RocblasHandle) -> u32> = lib
                .get(b"rocblas_destroy_handle")
                .map_err(|e| RocblasError {
                    status: 0,
                    context: format!("resolve rocblas_destroy_handle: {e}"),
                })?;
            let fn_set_stream: Symbol<unsafe extern "C" fn(RocblasHandle, *mut c_void) -> u32> =
                lib.get(b"rocblas_set_stream").map_err(|e| RocblasError {
                    status: 0,
                    context: format!("resolve rocblas_set_stream: {e}"),
                })?;
            let fn_gemm_ex: Symbol<
                unsafe extern "C" fn(
                    RocblasHandle,
                    c_uint,
                    c_uint,
                    c_int,
                    c_int,
                    c_int,
                    *const c_void,
                    *const c_void,
                    c_uint,
                    c_int,
                    *const c_void,
                    c_uint,
                    c_int,
                    *const c_void,
                    *const c_void,
                    c_uint,
                    c_int,
                    *mut c_void,
                    c_uint,
                    c_int,
                    c_uint,
                    c_uint,
                    i32,
                    u32,
                ) -> u32,
            > = lib.get(b"rocblas_gemm_ex").map_err(|e| RocblasError {
                status: 0,
                context: format!("resolve rocblas_gemm_ex: {e}"),
            })?;

            let fn_create_handle = *fn_create_handle;
            let fn_destroy_handle = *fn_destroy_handle;
            let fn_set_stream = *fn_set_stream;
            let fn_gemm_ex = *fn_gemm_ex;

            let mut handle: RocblasHandle = std::ptr::null_mut();
            let st = fn_create_handle(&mut handle);
            if st != ROCBLAS_STATUS_SUCCESS {
                return Err(RocblasError {
                    status: st,
                    context: "rocblas_create_handle".into(),
                });
            }

            Ok(Self {
                _lib: lib,
                handle,
                fn_destroy_handle,
                fn_set_stream,
                fn_gemm_ex,
            })
        }
    }

    /// Bind this rocBLAS handle to a HIP stream so calls execute on it.
    pub fn set_stream(&self, stream: &Stream) -> RocblasResult<()> {
        let st = unsafe { (self.fn_set_stream)(self.handle, stream.as_raw()) };
        if st == ROCBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(RocblasError {
                status: st,
                context: "rocblas_set_stream".into(),
            })
        }
    }

    /// Column-major GEMM (rocBLAS convention) wrapping `rocblas_gemm_ex`.
    ///
    /// Computes D = alpha * op(A) * op(B) + beta * C with independent dtype
    /// selection. Pointers must be device pointers. For the prefill GEMM
    /// case we'll typically pass D=C (in-place) and beta=0.
    ///
    /// Note: rocBLAS is column-major. Our engine stores matrices row-major,
    /// so callers flip the operation (A_row · B_row == (B_col^T · A_col^T)^T)
    /// and swap (m, n) / (a, b) / (lda, ldb) / transA, transB when dispatching.
    ///
    /// # Safety
    ///
    /// All matrix pointers and scalar pointers must be valid for the rocBLAS
    /// call, point to GPU memory where rocBLAS expects it, and describe buffers
    /// large enough for the dimensions and leading dimensions passed here.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ex(
        &self,
        trans_a: RocblasOperation,
        trans_b: RocblasOperation,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: RocblasDatatype,
        lda: i32,
        b: *const c_void,
        b_type: RocblasDatatype,
        ldb: i32,
        beta: *const c_void,
        c: *const c_void,
        c_type: RocblasDatatype,
        ldc: i32,
        d: *mut c_void,
        d_type: RocblasDatatype,
        ldd: i32,
        compute_type: RocblasDatatype,
    ) -> RocblasResult<()> {
        let st = (self.fn_gemm_ex)(
            self.handle,
            trans_a as c_uint,
            trans_b as c_uint,
            m,
            n,
            k,
            alpha,
            a,
            a_type as c_uint,
            lda,
            b,
            b_type as c_uint,
            ldb,
            beta,
            c,
            c_type as c_uint,
            ldc,
            d,
            d_type as c_uint,
            ldd,
            compute_type as c_uint,
            RocblasGemmAlgo::Standard as c_uint,
            0,
            0,
        );
        if st == ROCBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(RocblasError {
                status: st,
                context: "rocblas_gemm_ex".into(),
            })
        }
    }
}

impl Drop for Rocblas {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                let _ = (self.fn_destroy_handle)(self.handle);
            }
        }
    }
}

// The handle is bound to a GPU context; we don't share across threads without sync.
unsafe impl Send for Rocblas {}
