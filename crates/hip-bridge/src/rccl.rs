// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal FFI wrapper around librccl.so for tensor-parallel collectives.
//!
//! Built for the TP=4-on-hiptrx path: per the 2026-05-28 comm baseline
//! (`docs/investigations/2026-05-28-tp-comm-baseline-hiptrx.md`), RCCL
//! `ncclAllReduce` on gfx1201 is 3× faster than a host-driven ring on
//! `boundary_copy` (~110 µs flat from 4-128 KB vs ~340 µs sequential).
//!
//! Loaded lazily via `libloading`; absence of librccl is a recoverable
//! error so the engine still builds + runs without it (caller falls back
//! to the boundary_copy ring path).
//!
//! Scope: just enough to back `Gpus::all_reduce_sum`. The exposed surface
//! is `ncclCommInitAll`, `ncclCommDestroy`, `ncclAllReduce`,
//! `ncclGroupStart`/`ncclGroupEnd`, plus error/version helpers. Point-to-
//! point (`ncclSend`/`ncclRecv`) and other collectives (`ncclBroadcast`,
//! `ncclReduceScatter`) are out for now — add when a caller needs them.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void, CStr};

// ── Status codes (rccl/nccl.h ncclResult_t) ─────────────────────────
pub const NCCL_SUCCESS: u32 = 0;

#[derive(Debug)]
pub struct RcclError {
    pub status: u32,
    pub context: String,
}

impl std::fmt::Display for RcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RCCL error {} in {}", self.status, self.context)
    }
}
impl std::error::Error for RcclError {}

pub type RcclResult<T> = Result<T, RcclError>;

/// `ncclDataType_t` from `rccl/rccl.h`. Subset we currently need; extend on demand.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcclDataType {
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

/// `ncclRedOp_t` from `rccl/rccl.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

// ── Opaque handle ─────────────────────────────────────────────────────
//
// `ncclComm_t` is `struct ncclComm*` — an opaque pointer. We just
// shuttle the raw pointer through FFI; lifetime is managed by
// `RcclComms`'s `Drop`.
type NcclComm = *mut c_void;

/// Loaded RCCL library + resolved function pointers. Holds one
/// communicator per rank in `comms`; the FFI lib stays alive as long as
/// the struct does.
pub struct RcclComms {
    _lib: Library,
    comms: Vec<NcclComm>,

    fn_comm_init_all: unsafe extern "C" fn(*mut NcclComm, c_int, *const c_int) -> u32,
    fn_comm_destroy: unsafe extern "C" fn(NcclComm) -> u32,
    fn_all_reduce: unsafe extern "C" fn(
        *const c_void, // sendbuff
        *mut c_void,   // recvbuff
        usize,         // count
        u32,           // datatype
        u32,           // op
        NcclComm,      // comm
        *mut c_void,   // hipStream_t
    ) -> u32,
    fn_group_start: unsafe extern "C" fn() -> u32,
    fn_group_end: unsafe extern "C" fn() -> u32,
    fn_get_error_string: unsafe extern "C" fn(u32) -> *const c_char,
    fn_get_version: unsafe extern "C" fn(*mut c_int) -> u32,
}

impl RcclComms {
    /// Attempt to dlopen librccl.so and resolve the subset of symbols we
    /// use. On failure (library missing / symbol missing), returns an
    /// error that the caller can treat as "RCCL unavailable, fall back
    /// to boundary_copy ring path".
    ///
    /// Then initializes `n_devices` communicators in one shot via
    /// `ncclCommInitAll`. Each comm[i] binds to `device_ids[i]`.
    pub fn init_all(device_ids: &[i32]) -> RcclResult<Self> {
        let lib = unsafe {
            let candidates = ["librccl.so", "librccl.so.1", "librccl.so.1.0"];
            let mut loaded = None;
            for name in &candidates {
                if let Ok(l) = Library::new(name) {
                    loaded = Some(l);
                    break;
                }
            }
            loaded.ok_or_else(|| RcclError {
                status: 0,
                context: format!(
                    "failed to dlopen librccl.so. Tried: {:?}. Is RCCL installed (apt install rccl, or /opt/rocm/lib/librccl.so.1)?",
                    candidates
                ),
            })?
        };

        macro_rules! load_sym {
            ($name:expr, $ty:ty) => {{
                let sym: Symbol<'_, $ty> = lib.get($name.as_bytes()).map_err(|e| RcclError {
                    status: 0,
                    context: format!("failed to load symbol {}: {}", $name, e),
                })?;
                *sym.into_raw()
            }};
        }

        let (
            fn_comm_init_all,
            fn_comm_destroy,
            fn_all_reduce,
            fn_group_start,
            fn_group_end,
            fn_get_error_string,
            fn_get_version,
        ) = unsafe {
            (
                load_sym!(
                    "ncclCommInitAll",
                    unsafe extern "C" fn(*mut NcclComm, c_int, *const c_int) -> u32
                ),
                load_sym!("ncclCommDestroy", unsafe extern "C" fn(NcclComm) -> u32),
                load_sym!(
                    "ncclAllReduce",
                    unsafe extern "C" fn(
                        *const c_void,
                        *mut c_void,
                        usize,
                        u32,
                        u32,
                        NcclComm,
                        *mut c_void,
                    ) -> u32
                ),
                load_sym!("ncclGroupStart", unsafe extern "C" fn() -> u32),
                load_sym!("ncclGroupEnd", unsafe extern "C" fn() -> u32),
                load_sym!(
                    "ncclGetErrorString",
                    unsafe extern "C" fn(u32) -> *const c_char
                ),
                load_sym!("ncclGetVersion", unsafe extern "C" fn(*mut c_int) -> u32),
            )
        };

        let n = device_ids.len() as c_int;
        let mut comms: Vec<NcclComm> = vec![std::ptr::null_mut(); device_ids.len()];
        let status = unsafe { fn_comm_init_all(comms.as_mut_ptr(), n, device_ids.as_ptr()) };
        if status != NCCL_SUCCESS {
            let msg = unsafe {
                CStr::from_ptr(fn_get_error_string(status))
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(RcclError {
                status,
                context: format!(
                    "ncclCommInitAll(n={}, devices={:?}) failed: {}",
                    device_ids.len(),
                    device_ids,
                    msg,
                ),
            });
        }

        Ok(Self {
            _lib: lib,
            comms,
            fn_comm_init_all,
            fn_comm_destroy,
            fn_all_reduce,
            fn_group_start,
            fn_group_end,
            fn_get_error_string,
            fn_get_version,
        })
    }

    /// Number of communicators (== n_devices passed to `init_all`).
    #[inline]
    pub fn len(&self) -> usize {
        self.comms.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.comms.is_empty()
    }

    /// RCCL runtime version (NCCL_VERSION_CODE format: major*1000 + minor*100 + patch).
    pub fn version(&self) -> RcclResult<i32> {
        let mut v: c_int = 0;
        let status = unsafe { (self.fn_get_version)(&mut v) };
        if status != NCCL_SUCCESS {
            return Err(self.err(status, "ncclGetVersion"));
        }
        Ok(v as i32)
    }

    /// Translate a `ncclResult_t` to a human-readable string via librccl.
    fn err(&self, status: u32, ctx: &str) -> RcclError {
        let msg = unsafe {
            CStr::from_ptr((self.fn_get_error_string)(status))
                .to_string_lossy()
                .into_owned()
        };
        RcclError {
            status,
            context: format!("{ctx}: {msg}"),
        }
    }

    /// Begin a group of collective calls (typical TP usage: wrap N
    /// per-rank `all_reduce` calls in a single group so RCCL can fuse
    /// the kernel launch). Pair every `group_start` with `group_end`.
    pub fn group_start(&self) -> RcclResult<()> {
        let s = unsafe { (self.fn_group_start)() };
        if s != NCCL_SUCCESS {
            return Err(self.err(s, "ncclGroupStart"));
        }
        Ok(())
    }

    pub fn group_end(&self) -> RcclResult<()> {
        let s = unsafe { (self.fn_group_end)() };
        if s != NCCL_SUCCESS {
            return Err(self.err(s, "ncclGroupEnd"));
        }
        Ok(())
    }

    /// Convenience: run `f` inside a `group_start`/`group_end` pair. If
    /// `f` returns Err the group is still closed so RCCL state stays
    /// consistent for the next call.
    pub fn group<F, E>(&self, f: F) -> Result<(), E>
    where
        F: FnOnce() -> Result<(), E>,
        E: From<RcclError>,
    {
        self.group_start()?;
        let r = f();
        let close = self.group_end();
        match (r, close) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), _) => Err(e),
            (Ok(()), Err(e)) => Err(E::from(e)),
        }
    }

    /// AllReduce on a single rank's send/recv buffers, using that rank's
    /// stream. Caller is responsible for: (a) calling `group_start` before
    /// looping over ranks if launching one-process-multi-rank, and
    /// (b) `stream_synchronize` (or stream-wait-event) after `group_end`
    /// to ensure the collective completes before consuming `recvbuff`.
    ///
    /// # Safety
    /// - `sendbuff` and `recvbuff` must be device pointers on the device
    ///   that owns `comm[rank]`.
    /// - `stream` must be a valid `hipStream_t` on that same device.
    /// - `count * sizeof(dtype)` bytes must be valid at both buffers.
    /// - All ranks must call AllReduce with matching `count`, `dtype`, `op`
    ///   between paired `group_start/group_end` calls.
    pub unsafe fn all_reduce(
        &self,
        rank: usize,
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        dtype: RcclDataType,
        op: RcclRedOp,
        stream: *mut c_void,
    ) -> RcclResult<()> {
        let comm = self.comms[rank];
        let status = (self.fn_all_reduce)(
            sendbuff,
            recvbuff,
            count,
            dtype as u32,
            op as u32,
            comm,
            stream,
        );
        if status != NCCL_SUCCESS {
            return Err(self.err(status, &format!("ncclAllReduce rank={rank}")));
        }
        Ok(())
    }

    /// Typed convenience for f32 sum — the load-bearing TP all-reduce
    /// shape (residual stream summation across ranks).
    ///
    /// # Safety
    /// Caller asserts `sendbuff`/`recvbuff` point to `count` valid f32
    /// elements of device memory associated with `rank`, and that `stream`
    /// is a live stream on that device. See `all_reduce` for the rest of the
    /// safety contract.
    pub unsafe fn all_reduce_sum_f32(
        &self,
        rank: usize,
        sendbuff: *const f32,
        recvbuff: *mut f32,
        count: usize,
        stream: *mut c_void,
    ) -> RcclResult<()> {
        self.all_reduce(
            rank,
            sendbuff as *const c_void,
            recvbuff as *mut c_void,
            count,
            RcclDataType::Float32,
            RcclRedOp::Sum,
            stream,
        )
    }

    /// Typed convenience for fp16 sum (residual stream in mixed-precision).
    /// `count` is element count, NOT byte count.
    ///
    /// # Safety
    /// Caller asserts the underlying buffers hold fp16 elements; we don't
    /// have a fp16 Rust type to enforce this at compile time. See
    /// `all_reduce` for the rest of the safety contract.
    pub unsafe fn all_reduce_sum_f16(
        &self,
        rank: usize,
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        stream: *mut c_void,
    ) -> RcclResult<()> {
        self.all_reduce(
            rank,
            sendbuff,
            recvbuff,
            count,
            RcclDataType::Float16,
            RcclRedOp::Sum,
            stream,
        )
    }
}

impl Drop for RcclComms {
    fn drop(&mut self) {
        for &comm in &self.comms {
            if !comm.is_null() {
                let _ = unsafe { (self.fn_comm_destroy)(comm) };
            }
        }
    }
}

// RcclComms holds an opaque `*mut c_void` per rank from RCCL. The
// comms are single-process / multi-device and intended to be driven
// from a single thread (matches hipfire's HIP work invariant). Marking
// `Send` allows the parent `Gpus` to hold it; we deliberately do NOT
// mark `Sync` — calls must be serialized externally.
unsafe impl Send for RcclComms {}
