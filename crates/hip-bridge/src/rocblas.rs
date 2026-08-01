// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal FFI wrapper around librocblas.so for MI300X MFMA-accelerated GEMMs.
//!
//! rocBLAS provides a compact `rocblas_gemm_ex` entry point that dispatches to
//! MFMA-tuned kernels on CDNA3. hipBLASLt may expose additional exact-shape
//! algorithms, but belongs behind a separate wrapper and measured comparison;
//! this module makes no relative performance claim between the libraries.
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
pub const ROCBLAS_STATUS_INVALID_VALUE: u32 = 11;

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

/// rocBLAS GEMM algorithm selector (from rocblas-types.h).
///
/// These are deliberately zero/one rather than the 160-series datatype
/// constants. ROCm 7.14 declares `rocblas_gemm_algo_standard = 0x0` and
/// `rocblas_gemm_algo_solution_index = 0x1`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocblasGemmAlgo {
    Standard = 0,
    SolutionIndex = 1,
}

type RocblasHandle = *mut c_void;

type RocblasGemmExFn = unsafe extern "C" fn(
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
) -> u32;

/// Optional ROCm beta API. It is resolved independently so an older rocBLAS
/// can still provide the default GEMM path while exact-solution discovery
/// fails closed as unavailable.
type RocblasGemmExGetSolutionsFn = unsafe extern "C" fn(
    RocblasHandle,
    c_uint,
    c_uint, // transA, transB
    c_int,
    c_int,
    c_int,         // m, n, k
    *const c_void, // alpha
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
    u32,    // flags
    *mut i32,
    *mut i32, // list_array, list_size
) -> u32;

/// Loaded rocBLAS library + resolved function pointers.
pub struct Rocblas {
    _lib: Library,
    handle: RocblasHandle,

    fn_destroy_handle: unsafe extern "C" fn(RocblasHandle) -> u32,
    fn_set_stream: unsafe extern "C" fn(RocblasHandle, *mut c_void) -> u32,
    fn_gemm_ex: RocblasGemmExFn,
    fn_gemm_ex_get_solutions: Option<RocblasGemmExGetSolutionsFn>,
}

impl Rocblas {
    /// Attempt to dlopen librocblas.so and resolve the subset of symbols we use.
    /// On failure (library missing / symbol missing), returns an error that the
    /// caller can treat as "rocBLAS unavailable, fall back to hand-rolled kernels".
    pub fn load() -> RocblasResult<Self> {
        // Resolved ROCm roots first, bare sonames last. Replaces a hardcoded
        // /opt/rocm/lib list that missed side-by-side and core-<ver> layouts.
        let candidates = hipfire_config::rocm::library_candidates(&[
            "librocblas.so",
            "librocblas.so.7",
            "librocblas.so.6",
            "librocblas.so.5",
        ]);
        let lib = candidates
            .iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| RocblasError {
                status: 0,
                context: format!(
                    "dlopen librocblas.so failed. Tried: {}",
                    candidates.join(", ")
                ),
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
            let fn_gemm_ex: Symbol<RocblasGemmExFn> =
                lib.get(b"rocblas_gemm_ex").map_err(|e| RocblasError {
                    status: 0,
                    context: format!("resolve rocblas_gemm_ex: {e}"),
                })?;
            // Beta/deprecated in rocBLAS 5.5. Keep this optional: default GEMM
            // remains usable when the installed library omits the symbol, but
            // callers cannot mistake an unavailable discovery API for an empty
            // solution list.
            let fn_gemm_ex_get_solutions = lib
                .get::<RocblasGemmExGetSolutionsFn>(b"rocblas_gemm_ex_get_solutions")
                .ok()
                .map(|symbol| *symbol);

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
                fn_gemm_ex_get_solutions,
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

    /// Whether this installation exports the beta solution-enumeration API.
    ///
    /// Absence does not disable ordinary `gemm_ex`; it only prevents a caller
    /// from performing explicit solution discovery.
    pub fn has_gemm_ex_solution_enumeration(&self) -> bool {
        self.fn_gemm_ex_get_solutions.is_some()
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
        self.call_gemm_ex(
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            a_type,
            lda,
            b,
            b_type,
            ldb,
            beta,
            c,
            c_type,
            ldc,
            d,
            d_type,
            ldd,
            compute_type,
            RocblasGemmAlgo::Standard,
            0,
            0,
        )
    }

    /// Execute `rocblas_gemm_ex` with one explicitly selected solution index.
    ///
    /// This is a channel/discovery surface only: no product route selects an
    /// index automatically. `solution_index` must be positive because rocBLAS
    /// reserves zero for the default solution and deprecates negative indices.
    ///
    /// # Safety
    ///
    /// The matrix/scalar pointer contract is identical to [`Self::gemm_ex`].
    /// The index must have been enumerated for this exact GEMM problem and the
    /// same rocBLAS/device identity.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_ex_with_solution(
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
        solution_index: i32,
    ) -> RocblasResult<()> {
        validate_explicit_solution_index(solution_index)?;
        self.call_gemm_ex(
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            a_type,
            lda,
            b,
            b_type,
            ldb,
            beta,
            c,
            c_type,
            ldc,
            d,
            d_type,
            ldd,
            compute_type,
            RocblasGemmAlgo::SolutionIndex,
            solution_index,
            0,
        )
    }

    /// Enumerate rocBLAS solution indices for an exact `gemm_ex` problem.
    ///
    /// Returns `Ok(None)` when the installed rocBLAS omits the beta
    /// `rocblas_gemm_ex_get_solutions` symbol. `Some([])` means the API was
    /// available but reported no eligible solutions. The returned indices are
    /// library/version/device/shape specific and must not be reused under a
    /// different identity.
    ///
    /// # Safety
    ///
    /// All pointers must obey the same contract as [`Self::gemm_ex`]. rocBLAS
    /// inspects the complete problem, including its pointer and layout fields,
    /// when deciding which solutions are eligible.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enumerate_gemm_ex_solutions(
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
    ) -> RocblasResult<Option<Vec<i32>>> {
        let Some(get_solutions) = self.fn_gemm_ex_get_solutions else {
            return Ok(None);
        };

        let mut list_size = 0_i32;
        let st = get_solutions(
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
            std::ptr::null_mut(),
            &mut list_size,
        );
        check_rocblas_status(st, "rocblas_gemm_ex_get_solutions(count)")?;
        let capacity = checked_solution_count(list_size)?;
        if capacity == 0 {
            return Ok(Some(Vec::new()));
        }

        let mut solutions = vec![0_i32; capacity];
        list_size = capacity as i32;
        let st = get_solutions(
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
            solutions.as_mut_ptr(),
            &mut list_size,
        );
        check_rocblas_status(st, "rocblas_gemm_ex_get_solutions(fill)")?;
        let returned = checked_solution_count(list_size)?;
        if returned > capacity {
            return Err(RocblasError {
                status: ROCBLAS_STATUS_INVALID_VALUE,
                context: format!(
                    "rocblas_gemm_ex_get_solutions returned {returned} entries for capacity {capacity}"
                ),
            });
        }
        solutions.truncate(returned);
        Ok(Some(solutions))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn call_gemm_ex(
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
        algo: RocblasGemmAlgo,
        solution_index: i32,
        flags: u32,
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
            algo as c_uint,
            solution_index,
            flags,
        );
        check_rocblas_status(st, "rocblas_gemm_ex")
    }
}

fn check_rocblas_status(status: u32, context: &str) -> RocblasResult<()> {
    if status == ROCBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(RocblasError {
            status,
            context: context.into(),
        })
    }
}

fn validate_explicit_solution_index(solution_index: i32) -> RocblasResult<()> {
    if solution_index > 0 {
        Ok(())
    } else {
        Err(RocblasError {
            status: ROCBLAS_STATUS_INVALID_VALUE,
            context: format!(
                "explicit rocBLAS solution index must be positive, got {solution_index}"
            ),
        })
    }
}

fn checked_solution_count(count: i32) -> RocblasResult<usize> {
    usize::try_from(count).map_err(|_| RocblasError {
        status: ROCBLAS_STATUS_INVALID_VALUE,
        context: format!("rocBLAS returned a negative solution count: {count}"),
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_algorithm_values_match_rocm_7_14_header() {
        assert_eq!(RocblasGemmAlgo::Standard as u32, 0x0);
        assert_eq!(RocblasGemmAlgo::SolutionIndex as u32, 0x1);
    }

    #[test]
    fn explicit_solution_rejects_default_and_deprecated_indices() {
        assert!(validate_explicit_solution_index(1).is_ok());
        for index in [0, -1] {
            let err = validate_explicit_solution_index(index).unwrap_err();
            assert_eq!(err.status, ROCBLAS_STATUS_INVALID_VALUE);
        }
    }

    #[test]
    fn solution_count_conversion_fails_closed() {
        assert_eq!(checked_solution_count(0).unwrap(), 0);
        assert_eq!(checked_solution_count(7).unwrap(), 7);
        assert_eq!(
            checked_solution_count(-1).unwrap_err().status,
            ROCBLAS_STATUS_INVALID_VALUE
        );
    }
}
