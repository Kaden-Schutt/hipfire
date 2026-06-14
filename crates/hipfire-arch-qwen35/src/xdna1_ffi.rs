// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FFI bindings to libhipfire_xdna1.so via dlopen.
//!
//! Covers the bf16 SwiGLU kernel needed for the Qwen3.5 dense FFN NPU path.
//! All functions are resolved at runtime — no link-time dependency on the library.
//!
//! Signatures deduced from disassembly of libhipfire_xdna1.so:
//!   `xdna1_bf16_swiglu_create(xclbin_path, instr_path, hidden_size, result_out?) -> *handle`
//!   `xdna1_bf16_swiglu_run_handle(handle, gate_bf16, gate_n, up_bf16, up_n, out_bf16, out_n)`
//!   `xdna1_bf16_swiglu_destroy(handle)`
//!
//! path1 (xclbin_path): compiled AIE tile program — loaded via `xrt::xclbin`.
//! path2 (instr_path):  raw NPU instruction binary — read into the instruction BO.
//!                      Size must be u32-aligned (divisible by 4).

use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::sync::{Mutex, OnceLock};

// ─── Function pointer types ───────────────────────────────────────────────────

type FnCreate =
    unsafe extern "C" fn(*const c_char, *const c_char, usize, *mut c_void) -> *mut c_void;

type FnRunHandle =
    unsafe extern "C" fn(*mut c_void, *const u16, usize, *const u16, usize, *mut u16, usize)
        -> *mut c_void;

type FnDestroy = unsafe extern "C" fn(*mut c_void);

// ─── Library wrapper ──────────────────────────────────────────────────────────

pub struct Xdna1Lib {
    _lib: Library,
    pub fn_swiglu_create: FnCreate,
    pub fn_swiglu_run_handle: FnRunHandle,
    pub fn_swiglu_destroy: FnDestroy,
    // RMSNorm symbols are optional — absent when the .so predates the rmsnorm kernel.
    pub fn_rmsnorm_create: Option<FnCreate>,
    pub fn_rmsnorm_run_handle: Option<FnRunHandle>,
    pub fn_rmsnorm_destroy: Option<FnDestroy>,
    // RoPE Q/K symbols are optional — absent when the .so predates the rope kernel.
    pub fn_rope_q_create: Option<FnCreate>,
    pub fn_rope_q_run_handle: Option<FnRunHandle>,
    pub fn_rope_q_destroy: Option<FnDestroy>,
    pub fn_rope_k_create: Option<FnCreate>,
    pub fn_rope_k_run_handle: Option<FnRunHandle>,
    pub fn_rope_k_destroy: Option<FnDestroy>,
    // Head norm Q/K symbols are optional — absent when the .so predates the headnorm kernel.
    pub fn_headnorm_q_create: Option<FnCreate>,
    pub fn_headnorm_q_run_handle: Option<FnRunHandle>,
    pub fn_headnorm_q_destroy: Option<FnDestroy>,
    pub fn_headnorm_k_create: Option<FnCreate>,
    pub fn_headnorm_k_run_handle: Option<FnRunHandle>,
    pub fn_headnorm_k_destroy: Option<FnDestroy>,
    // Attn output gate symbol is optional — absent when the .so predates this kernel,
    // and unused when config.attn_output_gate is false.
    pub fn_attn_gate_create: Option<FnCreate>,
    pub fn_attn_gate_run_handle: Option<FnRunHandle>,
    pub fn_attn_gate_destroy: Option<FnDestroy>,
}

// Library holds only raw fn-ptrs and a Library — all Send+Sync.
unsafe impl Send for Xdna1Lib {}
unsafe impl Sync for Xdna1Lib {}

impl Xdna1Lib {
    fn load_from(path: &str) -> Result<Self, String> {
        let lib = unsafe { Library::new(path) }
            .map_err(|e| format!("xdna1: failed to dlopen {path}: {e}"))?;
        unsafe {
            let fn_swiglu_create: Symbol<FnCreate> = lib
                .get(b"xdna1_bf16_swiglu_create\0")
                .map_err(|e| format!("xdna1: missing xdna1_bf16_swiglu_create: {e}"))?;
            let fn_swiglu_run_handle: Symbol<FnRunHandle> = lib
                .get(b"xdna1_bf16_swiglu_run_handle\0")
                .map_err(|e| format!("xdna1: missing xdna1_bf16_swiglu_run_handle: {e}"))?;
            let fn_swiglu_destroy: Symbol<FnDestroy> = lib
                .get(b"xdna1_bf16_swiglu_destroy\0")
                .map_err(|e| format!("xdna1: missing xdna1_bf16_swiglu_destroy: {e}"))?;
            // RMSNorm symbols: allowed to be absent (graceful degradation).
            let fn_rmsnorm_create = lib
                .get::<FnCreate>(b"xdna1_bf16_rmsnorm_create\0")
                .ok()
                .map(|s| *s);
            let fn_rmsnorm_run_handle = lib
                .get::<FnRunHandle>(b"xdna1_bf16_rmsnorm_run_handle\0")
                .ok()
                .map(|s| *s);
            let fn_rmsnorm_destroy = lib
                .get::<FnDestroy>(b"xdna1_bf16_rmsnorm_destroy\0")
                .ok()
                .map(|s| *s);
            // RoPE Q/K symbols: allowed to be absent (graceful degradation).
            let fn_rope_q_create = lib
                .get::<FnCreate>(b"xdna1_bf16_rope_q_create\0")
                .ok()
                .map(|s| *s);
            let fn_rope_q_run_handle = lib
                .get::<FnRunHandle>(b"xdna1_bf16_rope_q_run_handle\0")
                .ok()
                .map(|s| *s);
            let fn_rope_q_destroy = lib
                .get::<FnDestroy>(b"xdna1_bf16_rope_q_destroy\0")
                .ok()
                .map(|s| *s);
            let fn_rope_k_create = lib
                .get::<FnCreate>(b"xdna1_bf16_rope_k_create\0")
                .ok()
                .map(|s| *s);
            let fn_rope_k_run_handle = lib
                .get::<FnRunHandle>(b"xdna1_bf16_rope_k_run_handle\0")
                .ok()
                .map(|s| *s);
            let fn_rope_k_destroy = lib
                .get::<FnDestroy>(b"xdna1_bf16_rope_k_destroy\0")
                .ok()
                .map(|s| *s);
            // Head norm Q/K symbols: allowed to be absent (graceful degradation).
            let fn_headnorm_q_create = lib
                .get::<FnCreate>(b"xdna1_bf16_headnorm_q_create\0")
                .ok()
                .map(|s| *s);
            let fn_headnorm_q_run_handle = lib
                .get::<FnRunHandle>(b"xdna1_bf16_headnorm_q_run_handle\0")
                .ok()
                .map(|s| *s);
            let fn_headnorm_q_destroy = lib
                .get::<FnDestroy>(b"xdna1_bf16_headnorm_q_destroy\0")
                .ok()
                .map(|s| *s);
            let fn_headnorm_k_create = lib
                .get::<FnCreate>(b"xdna1_bf16_headnorm_k_create\0")
                .ok()
                .map(|s| *s);
            let fn_headnorm_k_run_handle = lib
                .get::<FnRunHandle>(b"xdna1_bf16_headnorm_k_run_handle\0")
                .ok()
                .map(|s| *s);
            let fn_headnorm_k_destroy = lib
                .get::<FnDestroy>(b"xdna1_bf16_headnorm_k_destroy\0")
                .ok()
                .map(|s| *s);
            // Attn gate: optional, only for configs where attn_output_gate=true.
            let fn_attn_gate_create = lib
                .get::<FnCreate>(b"xdna1_bf16_attn_gate_create\0")
                .ok()
                .map(|s| *s);
            let fn_attn_gate_run_handle = lib
                .get::<FnRunHandle>(b"xdna1_bf16_attn_gate_run_handle\0")
                .ok()
                .map(|s| *s);
            let fn_attn_gate_destroy = lib
                .get::<FnDestroy>(b"xdna1_bf16_attn_gate_destroy\0")
                .ok()
                .map(|s| *s);
            Ok(Xdna1Lib {
                fn_swiglu_create: *fn_swiglu_create,
                fn_swiglu_run_handle: *fn_swiglu_run_handle,
                fn_swiglu_destroy: *fn_swiglu_destroy,
                fn_rmsnorm_create,
                fn_rmsnorm_run_handle,
                fn_rmsnorm_destroy,
                fn_rope_q_create,
                fn_rope_q_run_handle,
                fn_rope_q_destroy,
                fn_rope_k_create,
                fn_rope_k_run_handle,
                fn_rope_k_destroy,
                fn_headnorm_q_create,
                fn_headnorm_q_run_handle,
                fn_headnorm_q_destroy,
                fn_headnorm_k_create,
                fn_headnorm_k_run_handle,
                fn_headnorm_k_destroy,
                fn_attn_gate_create,
                fn_attn_gate_run_handle,
                fn_attn_gate_destroy,
                _lib: lib,
            })
        }
    }
}

// ─── Global library instance ──────────────────────────────────────────────────

static LIB: OnceLock<Result<Xdna1Lib, String>> = OnceLock::new();

pub fn get_lib() -> Option<&'static Xdna1Lib> {
    LIB.get_or_init(|| {
        let path = std::env::var("HIPFIRE_XDNA1_LIB")
            .unwrap_or_else(|_| "target/npu/libhipfire_xdna1.so".to_string());
        Xdna1Lib::load_from(&path)
    })
    .as_ref()
    .ok()
}

// ─── SwiGLU handle cache (per layer_idx) ─────────────────────────────────────

struct RawHandle(*mut c_void);
unsafe impl Send for RawHandle {}

static SWIGLU_HANDLES: OnceLock<Mutex<HashMap<usize, RawHandle>>> = OnceLock::new();

fn handles() -> &'static Mutex<HashMap<usize, RawHandle>> {
    SWIGLU_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Create (or reuse cached) swiglu handle for `layer_idx`.
///
/// Returns `None` if the library isn't loaded, paths aren't configured,
/// or the create call returns null.
pub fn swiglu_handle_for(
    layer_idx: usize,
    hidden_size: usize,
    xclbin_path: &str,
    instr_path: &str,
) -> Option<*mut c_void> {
    let lib = get_lib()?;
    let mut map = handles().lock().unwrap();
    if let Some(h) = map.get(&layer_idx) {
        return Some(h.0);
    }
    let xclbin_c = CString::new(xclbin_path).ok()?;
    let instr_c = CString::new(instr_path).ok()?;
    let handle = unsafe {
        (lib.fn_swiglu_create)(
            xclbin_c.as_ptr(),
            instr_c.as_ptr(),
            hidden_size,
            std::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        eprintln!(
            "[xdna1] swiglu_create returned null for layer={layer_idx} \
             hidden_size={hidden_size} xclbin={xclbin_path} instr={instr_path}"
        );
        return None;
    }
    map.insert(layer_idx, RawHandle(handle));
    Some(handle)
}

/// Call `xdna1_bf16_swiglu_run_handle`.
///
/// `gate` and `up` are BF16 (u16 bits), `out` receives the BF16 result.
/// `len` must equal the `hidden_size` used when creating the handle.
///
/// # Safety
/// Caller guarantees all slices are valid and `len` matches the handle shape.
pub unsafe fn swiglu_run(
    handle: *mut c_void,
    gate: &[u16],
    up: &[u16],
    out: &mut [u16],
) -> bool {
    let lib = match get_lib() {
        Some(l) => l,
        None => return false,
    };
    let len = gate.len();
    debug_assert_eq!(up.len(), len);
    debug_assert_eq!(out.len(), len);
    let ret = (lib.fn_swiglu_run_handle)(
        handle,
        gate.as_ptr(),
        len,
        up.as_ptr(),
        len,
        out.as_mut_ptr(),
        len,
    );
    !ret.is_null()
}

// ─── RMSNorm handle cache (per layer_idx) ────────────────────────────────────

static RMSNORM_HANDLES: OnceLock<Mutex<HashMap<usize, RawHandle>>> = OnceLock::new();

fn rmsnorm_handles() -> &'static Mutex<HashMap<usize, RawHandle>> {
    RMSNORM_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Create (or reuse cached) rmsnorm handle for `layer_idx`.
///
/// Returns `None` if the library isn't loaded, the rmsnorm symbols are absent
/// (older .so), paths aren't configured, or the create call returns null.
pub fn rmsnorm_handle_for(
    layer_idx: usize,
    hidden_size: usize,
    xclbin_path: &str,
    instr_path: &str,
) -> Option<*mut c_void> {
    let lib = get_lib()?;
    let fn_create = lib.fn_rmsnorm_create?;
    let mut map = rmsnorm_handles().lock().unwrap();
    if let Some(h) = map.get(&layer_idx) {
        return Some(h.0);
    }
    let xclbin_c = CString::new(xclbin_path).ok()?;
    let instr_c = CString::new(instr_path).ok()?;
    let handle = unsafe {
        fn_create(
            xclbin_c.as_ptr(),
            instr_c.as_ptr(),
            hidden_size,
            std::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        eprintln!(
            "[xdna1] rmsnorm_create returned null for layer={layer_idx} \
             hidden_size={hidden_size} xclbin={xclbin_path} instr={instr_path}"
        );
        return None;
    }
    map.insert(layer_idx, RawHandle(handle));
    Some(handle)
}

/// Call `xdna1_bf16_rmsnorm_run_handle`.
///
/// `input` and `weight` are BF16 (u16 bits), `out` receives the BF16 result.
/// `len` must equal the `hidden_size` used when creating the handle.
///
/// # Safety
/// Caller guarantees all slices are valid and `len` matches the handle shape.
pub unsafe fn rmsnorm_run(
    handle: *mut c_void,
    input: &[u16],
    weight: &[u16],
    out: &mut [u16],
) -> bool {
    let lib = match get_lib() {
        Some(l) => l,
        None => return false,
    };
    let fn_run = match lib.fn_rmsnorm_run_handle {
        Some(f) => f,
        None => return false,
    };
    let len = input.len();
    debug_assert_eq!(weight.len(), len);
    debug_assert_eq!(out.len(), len);
    let ret = fn_run(
        handle,
        input.as_ptr(),
        len,
        weight.as_ptr(),
        len,
        out.as_mut_ptr(),
        len,
    );
    !ret.is_null()
}

// ─── RoPE Q handle cache ─────────────────────────────────────────────────────

static ROPE_Q_HANDLES: OnceLock<Mutex<HashMap<usize, RawHandle>>> = OnceLock::new();

fn rope_q_handles() -> &'static Mutex<HashMap<usize, RawHandle>> {
    ROPE_Q_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Create (or reuse cached) rope_q handle for `layer_idx`.
///
/// Returns `None` if the library isn't loaded, the rope_q symbols are absent
/// (older .so), paths aren't configured, or the create call returns null.
pub fn rope_q_handle_for(
    layer_idx: usize,
    n_total: usize,
    xclbin_path: &str,
    instr_path: &str,
) -> Option<*mut c_void> {
    let lib = get_lib()?;
    let fn_create = lib.fn_rope_q_create?;
    let mut map = rope_q_handles().lock().unwrap();
    if let Some(h) = map.get(&layer_idx) {
        return Some(h.0);
    }
    let xclbin_c = CString::new(xclbin_path).ok()?;
    let instr_c = CString::new(instr_path).ok()?;
    let handle = unsafe {
        fn_create(
            xclbin_c.as_ptr(),
            instr_c.as_ptr(),
            n_total,
            std::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        eprintln!(
            "[xdna1] rope_q_create returned null for layer={layer_idx} \
             n_total={n_total} xclbin={xclbin_path} instr={instr_path}"
        );
        return None;
    }
    map.insert(layer_idx, RawHandle(handle));
    Some(handle)
}

/// Call `xdna1_bf16_rope_q_run_handle`.
///
/// `input` is BF16 Q tensor (n_heads × head_dim elements).
/// `cs` is the cos/sin buffer ([cos_0..cos_{n_rot/2-1}, sin_0..sin_{n_rot/2-1}]).
/// `out` receives the BF16 rotated Q tensor.
///
/// # Safety
/// Caller guarantees all slices are valid and sizes match the handle shape.
pub unsafe fn rope_q_run(
    handle: *mut c_void,
    input: &[u16],
    cs: &[u16],
    out: &mut [u16],
) -> bool {
    let lib = match get_lib() {
        Some(l) => l,
        None => return false,
    };
    let fn_run = match lib.fn_rope_q_run_handle {
        Some(f) => f,
        None => return false,
    };
    let n = input.len();
    debug_assert_eq!(out.len(), n);
    let ret = fn_run(
        handle,
        input.as_ptr(),
        n,
        cs.as_ptr(),
        cs.len(),
        out.as_mut_ptr(),
        n,
    );
    !ret.is_null()
}

// ─── RoPE K handle cache ─────────────────────────────────────────────────────

static ROPE_K_HANDLES: OnceLock<Mutex<HashMap<usize, RawHandle>>> = OnceLock::new();

fn rope_k_handles() -> &'static Mutex<HashMap<usize, RawHandle>> {
    ROPE_K_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Create (or reuse cached) rope_k handle for `layer_idx`.
pub fn rope_k_handle_for(
    layer_idx: usize,
    n_total: usize,
    xclbin_path: &str,
    instr_path: &str,
) -> Option<*mut c_void> {
    let lib = get_lib()?;
    let fn_create = lib.fn_rope_k_create?;
    let mut map = rope_k_handles().lock().unwrap();
    if let Some(h) = map.get(&layer_idx) {
        return Some(h.0);
    }
    let xclbin_c = CString::new(xclbin_path).ok()?;
    let instr_c = CString::new(instr_path).ok()?;
    let handle = unsafe {
        fn_create(
            xclbin_c.as_ptr(),
            instr_c.as_ptr(),
            n_total,
            std::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        eprintln!(
            "[xdna1] rope_k_create returned null for layer={layer_idx} \
             n_total={n_total} xclbin={xclbin_path} instr={instr_path}"
        );
        return None;
    }
    map.insert(layer_idx, RawHandle(handle));
    Some(handle)
}

/// Call `xdna1_bf16_rope_k_run_handle`.
///
/// `input` is BF16 K tensor (n_kv_heads × head_dim elements).
/// `cs` is the cos/sin buffer ([cos_0..cos_{n_rot/2-1}, sin_0..sin_{n_rot/2-1}]).
/// `out` receives the BF16 rotated K tensor.
///
/// # Safety
/// Caller guarantees all slices are valid and sizes match the handle shape.
pub unsafe fn rope_k_run(
    handle: *mut c_void,
    input: &[u16],
    cs: &[u16],
    out: &mut [u16],
) -> bool {
    let lib = match get_lib() {
        Some(l) => l,
        None => return false,
    };
    let fn_run = match lib.fn_rope_k_run_handle {
        Some(f) => f,
        None => return false,
    };
    let n = input.len();
    debug_assert_eq!(out.len(), n);
    let ret = fn_run(
        handle,
        input.as_ptr(),
        n,
        cs.as_ptr(),
        cs.len(),
        out.as_mut_ptr(),
        n,
    );
    !ret.is_null()
}

// ─── Head norm Q handle cache ─────────────────────────────────────────────────

static HEADNORM_Q_HANDLES: OnceLock<Mutex<HashMap<usize, RawHandle>>> = OnceLock::new();

fn headnorm_q_handles() -> &'static Mutex<HashMap<usize, RawHandle>> {
    HEADNORM_Q_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Create (or reuse cached) headnorm_q handle for `layer_idx`.
///
/// Returns `None` if the library isn't loaded, the headnorm_q symbols are absent
/// (older .so), paths aren't configured, or the create call returns null.
pub fn headnorm_q_handle_for(
    layer_idx: usize,
    n_total: usize,
    xclbin_path: &str,
    instr_path: &str,
) -> Option<*mut c_void> {
    let lib = get_lib()?;
    let fn_create = lib.fn_headnorm_q_create?;
    let mut map = headnorm_q_handles().lock().unwrap();
    if let Some(h) = map.get(&layer_idx) {
        return Some(h.0);
    }
    let xclbin_c = CString::new(xclbin_path).ok()?;
    let instr_c = CString::new(instr_path).ok()?;
    let handle = unsafe {
        fn_create(
            xclbin_c.as_ptr(),
            instr_c.as_ptr(),
            n_total,
            std::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        eprintln!(
            "[xdna1] headnorm_q_create returned null for layer={layer_idx} \
             n_total={n_total} xclbin={xclbin_path} instr={instr_path}"
        );
        return None;
    }
    map.insert(layer_idx, RawHandle(handle));
    Some(handle)
}

/// Call `xdna1_bf16_headnorm_q_run_handle`.
///
/// `input` is BF16 Q tensor (n_heads × head_dim elements).
/// `weight` is the shared per-head norm weight ([head_dim] elements).
/// `out` receives the BF16 normalized Q tensor.
///
/// # Safety
/// Caller guarantees all slices are valid and sizes match the handle shape.
pub unsafe fn headnorm_q_run(
    handle: *mut c_void,
    input: &[u16],
    weight: &[u16],
    out: &mut [u16],
) -> bool {
    let lib = match get_lib() {
        Some(l) => l,
        None => return false,
    };
    let fn_run = match lib.fn_headnorm_q_run_handle {
        Some(f) => f,
        None => return false,
    };
    let n = input.len();
    debug_assert_eq!(out.len(), n);
    let ret = fn_run(
        handle,
        input.as_ptr(),
        n,
        weight.as_ptr(),
        weight.len(),
        out.as_mut_ptr(),
        n,
    );
    !ret.is_null()
}

// ─── Head norm K handle cache ─────────────────────────────────────────────────

static HEADNORM_K_HANDLES: OnceLock<Mutex<HashMap<usize, RawHandle>>> = OnceLock::new();

fn headnorm_k_handles() -> &'static Mutex<HashMap<usize, RawHandle>> {
    HEADNORM_K_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Create (or reuse cached) headnorm_k handle for `layer_idx`.
pub fn headnorm_k_handle_for(
    layer_idx: usize,
    n_total: usize,
    xclbin_path: &str,
    instr_path: &str,
) -> Option<*mut c_void> {
    let lib = get_lib()?;
    let fn_create = lib.fn_headnorm_k_create?;
    let mut map = headnorm_k_handles().lock().unwrap();
    if let Some(h) = map.get(&layer_idx) {
        return Some(h.0);
    }
    let xclbin_c = CString::new(xclbin_path).ok()?;
    let instr_c = CString::new(instr_path).ok()?;
    let handle = unsafe {
        fn_create(
            xclbin_c.as_ptr(),
            instr_c.as_ptr(),
            n_total,
            std::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        eprintln!(
            "[xdna1] headnorm_k_create returned null for layer={layer_idx} \
             n_total={n_total} xclbin={xclbin_path} instr={instr_path}"
        );
        return None;
    }
    map.insert(layer_idx, RawHandle(handle));
    Some(handle)
}

/// Call `xdna1_bf16_headnorm_k_run_handle`.
///
/// `input` is BF16 K tensor (n_kv_heads × head_dim elements).
/// `weight` is the shared per-head norm weight ([head_dim] elements).
/// `out` receives the BF16 normalized K tensor.
///
/// # Safety
/// Caller guarantees all slices are valid and sizes match the handle shape.
pub unsafe fn headnorm_k_run(
    handle: *mut c_void,
    input: &[u16],
    weight: &[u16],
    out: &mut [u16],
) -> bool {
    let lib = match get_lib() {
        Some(l) => l,
        None => return false,
    };
    let fn_run = match lib.fn_headnorm_k_run_handle {
        Some(f) => f,
        None => return false,
    };
    let n = input.len();
    debug_assert_eq!(out.len(), n);
    let ret = fn_run(
        handle,
        input.as_ptr(),
        n,
        weight.as_ptr(),
        weight.len(),
        out.as_mut_ptr(),
        n,
    );
    !ret.is_null()
}

// ─── Attn output gate handle cache ───────────────────────────────────────────

static ATTN_GATE_HANDLES: OnceLock<Mutex<HashMap<usize, RawHandle>>> = OnceLock::new();

fn attn_gate_handles() -> &'static Mutex<HashMap<usize, RawHandle>> {
    ATTN_GATE_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Create (or reuse cached) attn_gate handle for `layer_idx`.
///
/// Returns `None` if the library isn't loaded, the attn_gate symbols are absent
/// (older .so or model without attn_output_gate), or the create call returns null.
pub fn attn_gate_handle_for(
    layer_idx: usize,
    q_dim: usize,
    xclbin_path: &str,
    instr_path: &str,
) -> Option<*mut c_void> {
    let lib = get_lib()?;
    let fn_create = lib.fn_attn_gate_create?;
    let mut map = attn_gate_handles().lock().unwrap();
    if let Some(h) = map.get(&layer_idx) {
        return Some(h.0);
    }
    let xclbin_c = CString::new(xclbin_path).ok()?;
    let instr_c = CString::new(instr_path).ok()?;
    let handle = unsafe {
        fn_create(
            xclbin_c.as_ptr(),
            instr_c.as_ptr(),
            q_dim,
            std::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        eprintln!(
            "[xdna1] attn_gate_create returned null for layer={layer_idx} \
             q_dim={q_dim} xclbin={xclbin_path} instr={instr_path}"
        );
        return None;
    }
    map.insert(layer_idx, RawHandle(handle));
    Some(handle)
}

/// Call `xdna1_bf16_attn_gate_run_handle`.
///
/// `gate` and `x` are BF16 Q-dim tensors (n_heads × head_dim elements each).
/// `out` receives sigmoid(gate) * x in BF16.
///
/// # Safety
/// Caller guarantees all slices are valid and `len` matches the handle shape.
pub unsafe fn attn_gate_run(
    handle: *mut c_void,
    gate: &[u16],
    x: &[u16],
    out: &mut [u16],
) -> bool {
    let lib = match get_lib() {
        Some(l) => l,
        None => return false,
    };
    let fn_run = match lib.fn_attn_gate_run_handle {
        Some(f) => f,
        None => return false,
    };
    let len = gate.len();
    debug_assert_eq!(x.len(), len);
    debug_assert_eq!(out.len(), len);
    let ret = fn_run(
        handle,
        gate.as_ptr(),
        len,
        x.as_ptr(),
        len,
        out.as_mut_ptr(),
        len,
    );
    !ret.is_null()
}
