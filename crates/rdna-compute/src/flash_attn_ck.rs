// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Optional dynamic loader for the FlashAttention CK sidecar experiment.
//!
//! This module only exposes the raw all-FP16 sidecar ABI. It deliberately does
//! not route hipfire attention calls or allocate conversion scratch. Callers
//! must opt in at build time, load an explicit library path, and provide device
//! buffers whose lifetimes cover the asynchronous launch.

use libloading::{Library, Symbol};
use std::error::Error;
use std::ffi::{c_char, c_void};
use std::fmt;
use std::path::{Path, PathBuf};

pub const FLASH_ATTN_CK_ABI_VERSION: u32 = 4;
const FLASH_ATTN_CK_MIN_COMPAT_ABI_VERSION: u32 = 3;
const ERROR_CAPACITY: usize = 512;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttnCkDType {
    F16 = 1,
    Bf16 = 2,
    F32 = 3,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttnCkArch {
    Gfx1100 = 1100,
    Gfx1151 = 1151,
    Gfx1201 = 1201,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttnCkKvFormat {
    DenseF16 = 1,
    DenseBf16 = 2,
    Q8 = 3,
    Asym3Givens = 4,
    Asym3Fwht = 5,
    Lloyd = 6,
    Asym4Givens = 7,
    Asym4Fwht = 8,
}

pub const FLASH_ATTN_CK_CAP_CAUSAL: u32 = 1 << 0;
pub const FLASH_ATTN_CK_CAP_GQA: u32 = 1 << 1;
const FLASH_ATTN_CK_KNOWN_CAP_FLAGS: u32 = FLASH_ATTN_CK_CAP_CAUSAL | FLASH_ATTN_CK_CAP_GQA;

/// One exact-architecture layout cell exported by a sidecar artifact.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttnCkCapability {
    pub abi_version: u32,
    pub struct_size: u32,
    pub arch: i32,
    pub dtype: i32,
    pub k_format: i32,
    pub v_format: i32,
    pub head_dim: i32,
    pub flags: u32,
}

/// Backend-agnostic lookup key produced after native attention layout policy
/// has resolved. Feature flags are requirements: a capability may advertise
/// additional behavior, but it must contain every requested bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttnCkRequest {
    pub arch: FlashAttnCkArch,
    pub dtype: FlashAttnCkDType,
    pub k_format: FlashAttnCkKvFormat,
    pub v_format: FlashAttnCkKvFormat,
    pub head_dim: i32,
    pub required_flags: u32,
}

/// Runtime facts resolved by the native KV policy before an optional CK route
/// is considered. This deliberately contains no pointers: eligibility is a
/// pure, testable decision and launching remains a separate operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttnCkPrefillInput {
    pub request: FlashAttnCkRequest,
    pub batch_size: usize,
    pub nhead_q: usize,
    pub nhead_k: usize,
    pub causal: bool,
    pub contiguous_prefix: bool,
    pub capture_mode: bool,
    pub replay_recording: bool,
    pub has_tree_bias: bool,
    pub window: usize,
    pub block_start: usize,
    pub block_cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttnCkRejectReason {
    Decode,
    SmallQuery,
    UnsupportedFormat,
    UnsupportedHeadDim,
    InvalidGqa,
    NonCausal,
    NonContiguousPrefix,
    GraphCapture,
    ReplayRecording,
    TreeAttention,
    WindowedAttention,
    BlockAttention,
    CapabilityMiss,
}

// The current CK cells are bulk-prefill kernels. DSpark verifies at most 15
// target tokens per step and is faster on the native small-query path.
const FLASH_ATTN_CK_MIN_QUERY_TOKENS: usize = 16;

/// Admit the first production cell only. Later quantized cells should extend
/// this policy explicitly instead of weakening its fail-closed conditions.
pub fn select_q8_d256_prefill(
    runtime: &FlashAttnCk,
    input: FlashAttnCkPrefillInput,
) -> Result<FlashAttnCkRequest, FlashAttnCkRejectReason> {
    select_q8_d256_prefill_capabilities(runtime.capabilities(), input)
}

fn select_q8_d256_prefill_capabilities(
    capabilities: &[FlashAttnCkCapability],
    input: FlashAttnCkPrefillInput,
) -> Result<FlashAttnCkRequest, FlashAttnCkRejectReason> {
    let request = input.request;
    if input.batch_size <= 1 {
        return Err(FlashAttnCkRejectReason::Decode);
    }
    if request.k_format != FlashAttnCkKvFormat::Q8 || request.v_format != FlashAttnCkKvFormat::Q8 {
        return Err(FlashAttnCkRejectReason::UnsupportedFormat);
    }
    if request.head_dim != 256 {
        return Err(FlashAttnCkRejectReason::UnsupportedHeadDim);
    }
    select_packed_prefill_capabilities(capabilities, input)
}

fn select_asym3_givens_prefill_capabilities(
    capabilities: &[FlashAttnCkCapability],
    input: FlashAttnCkPrefillInput,
) -> Result<FlashAttnCkRequest, FlashAttnCkRejectReason> {
    let request = input.request;
    if request.k_format != FlashAttnCkKvFormat::Asym3Givens
        || request.v_format != FlashAttnCkKvFormat::Q8
    {
        return Err(FlashAttnCkRejectReason::UnsupportedFormat);
    }
    if request.head_dim != 256 {
        return Err(FlashAttnCkRejectReason::UnsupportedHeadDim);
    }
    select_packed_prefill_capabilities(capabilities, input)
}

fn select_asym3_fwht_prefill_capabilities(
    capabilities: &[FlashAttnCkCapability],
    input: FlashAttnCkPrefillInput,
) -> Result<FlashAttnCkRequest, FlashAttnCkRejectReason> {
    let request = input.request;
    if request.k_format != FlashAttnCkKvFormat::Asym3Fwht
        || request.v_format != FlashAttnCkKvFormat::Q8
    {
        return Err(FlashAttnCkRejectReason::UnsupportedFormat);
    }
    if request.head_dim != 256 {
        return Err(FlashAttnCkRejectReason::UnsupportedHeadDim);
    }
    select_packed_prefill_capabilities(capabilities, input)
}

fn select_asym4_prefill_capabilities(
    capabilities: &[FlashAttnCkCapability],
    input: FlashAttnCkPrefillInput,
) -> Result<FlashAttnCkRequest, FlashAttnCkRejectReason> {
    let request = input.request;
    if !matches!(
        request.k_format,
        FlashAttnCkKvFormat::Asym4Givens | FlashAttnCkKvFormat::Asym4Fwht
    ) || request.v_format != FlashAttnCkKvFormat::Q8
    {
        return Err(FlashAttnCkRejectReason::UnsupportedFormat);
    }
    if request.head_dim != 256 {
        return Err(FlashAttnCkRejectReason::UnsupportedHeadDim);
    }
    select_packed_prefill_capabilities(capabilities, input)
}

fn select_packed_prefill_capabilities(
    capabilities: &[FlashAttnCkCapability],
    input: FlashAttnCkPrefillInput,
) -> Result<FlashAttnCkRequest, FlashAttnCkRejectReason> {
    let request = input.request;
    if input.batch_size <= 1 {
        return Err(FlashAttnCkRejectReason::Decode);
    }
    if input.batch_size < FLASH_ATTN_CK_MIN_QUERY_TOKENS {
        return Err(FlashAttnCkRejectReason::SmallQuery);
    }
    if input.nhead_k == 0
        || input.nhead_q < input.nhead_k
        || !input.nhead_q.is_multiple_of(input.nhead_k)
    {
        return Err(FlashAttnCkRejectReason::InvalidGqa);
    }
    if !input.causal {
        return Err(FlashAttnCkRejectReason::NonCausal);
    }
    if !input.contiguous_prefix {
        return Err(FlashAttnCkRejectReason::NonContiguousPrefix);
    }
    if input.capture_mode {
        return Err(FlashAttnCkRejectReason::GraphCapture);
    }
    if input.replay_recording {
        return Err(FlashAttnCkRejectReason::ReplayRecording);
    }
    if input.has_tree_bias {
        return Err(FlashAttnCkRejectReason::TreeAttention);
    }
    if input.window != 0 {
        return Err(FlashAttnCkRejectReason::WindowedAttention);
    }
    if input.block_start != 0 || input.block_cols != 0 {
        return Err(FlashAttnCkRejectReason::BlockAttention);
    }
    if !capabilities.iter().any(|cell| cell.supports(request)) {
        return Err(FlashAttnCkRejectReason::CapabilityMiss);
    }
    Ok(request)
}

impl FlashAttnCkCapability {
    pub fn supports(&self, request: FlashAttnCkRequest) -> bool {
        self.arch == request.arch as i32
            && self.dtype == request.dtype as i32
            && self.k_format == request.k_format as i32
            && self.v_format == request.v_format as i32
            && self.head_dim == request.head_dim
            && self.flags & request.required_flags == request.required_flags
    }

    fn is_well_formed(&self) -> bool {
        matches!(
            self.arch,
            value if value == FlashAttnCkArch::Gfx1100 as i32
                || value == FlashAttnCkArch::Gfx1151 as i32
                || value == FlashAttnCkArch::Gfx1201 as i32
        ) && matches!(
            self.dtype,
            value if value == FlashAttnCkDType::F16 as i32
                || value == FlashAttnCkDType::Bf16 as i32
                || value == FlashAttnCkDType::F32 as i32
        ) && matches!(
            self.k_format,
            value if is_known_kv_format(value)
        ) && matches!(
            self.v_format,
            value if is_known_kv_format(value)
        ) && self.head_dim > 0
            && self.flags & !FLASH_ATTN_CK_KNOWN_CAP_FLAGS == 0
    }
}

fn is_known_kv_format(value: i32) -> bool {
    value == FlashAttnCkKvFormat::DenseF16 as i32
        || value == FlashAttnCkKvFormat::DenseBf16 as i32
        || value == FlashAttnCkKvFormat::Q8 as i32
        || value == FlashAttnCkKvFormat::Asym3Givens as i32
        || value == FlashAttnCkKvFormat::Asym3Fwht as i32
        || value == FlashAttnCkKvFormat::Lloyd as i32
        || value == FlashAttnCkKvFormat::Asym4Givens as i32
        || value == FlashAttnCkKvFormat::Asym4Fwht as i32
}

fn is_supported_abi(version: u32) -> bool {
    (FLASH_ATTN_CK_MIN_COMPAT_ABI_VERSION..=FLASH_ATTN_CK_ABI_VERSION).contains(&version)
}

fn capability_is_compatible(version: u32, cell: &FlashAttnCkCapability) -> bool {
    let v3_quant_format = version == 3 && (cell.k_format >= 4 || cell.v_format >= 4);
    cell.abi_version == version
        && cell.struct_size >= std::mem::size_of::<FlashAttnCkCapability>() as u32
        && cell.is_well_formed()
        && !v3_quant_format
}

/// Stable C layout shared with `hipfire_flash_attn_ck.h`.
///
/// Strides are measured in elements, not bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnCkFwdParams {
    pub abi_version: u32,
    pub struct_size: u32,

    pub q: *const c_void,
    pub k: *const c_void,
    pub v: *const c_void,
    pub out: *mut c_void,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub stream: *mut c_void,

    pub dtype: i32,
    pub k_format: i32,
    pub v_format: i32,
    pub batch: i32,
    pub seqlen_q: i32,
    pub seqlen_k: i32,
    pub nhead_q: i32,
    pub nhead_k: i32,
    pub head_dim: i32,
    pub causal: i32,

    pub softmax_scale: f32,

    pub stride_q: i64,
    pub stride_k: i64,
    pub stride_v: i64,
    pub stride_out: i64,
    pub nhead_stride_q: i64,
    pub nhead_stride_k: i64,
    pub nhead_stride_v: i64,
    pub nhead_stride_out: i64,
    pub batch_stride_q: i64,
    pub batch_stride_k: i64,
    pub batch_stride_v: i64,
    pub batch_stride_out: i64,
    pub packed_k_row_stride_bytes: i64,
    pub packed_v_row_stride_bytes: i64,
    pub packed_k_head_stride_bytes: i64,
    pub packed_v_head_stride_bytes: i64,
    pub k_transform0: *const c_void,
    pub k_transform1: *const c_void,
    pub k_transform0_elements: i64,
    pub k_transform1_elements: i64,
}

impl FlashAttnCkFwdParams {
    pub fn new() -> Self {
        Self {
            abi_version: FLASH_ATTN_CK_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            q: std::ptr::null(),
            k: std::ptr::null(),
            v: std::ptr::null(),
            out: std::ptr::null_mut(),
            workspace: std::ptr::null_mut(),
            workspace_bytes: 0,
            stream: std::ptr::null_mut(),
            dtype: FlashAttnCkDType::F16 as i32,
            k_format: FlashAttnCkKvFormat::DenseF16 as i32,
            v_format: FlashAttnCkKvFormat::DenseF16 as i32,
            batch: 0,
            seqlen_q: 0,
            seqlen_k: 0,
            nhead_q: 0,
            nhead_k: 0,
            head_dim: 0,
            causal: 0,
            softmax_scale: 0.0,
            stride_q: 0,
            stride_k: 0,
            stride_v: 0,
            stride_out: 0,
            nhead_stride_q: 0,
            nhead_stride_k: 0,
            nhead_stride_v: 0,
            nhead_stride_out: 0,
            batch_stride_q: 0,
            batch_stride_k: 0,
            batch_stride_v: 0,
            batch_stride_out: 0,
            packed_k_row_stride_bytes: 0,
            packed_v_row_stride_bytes: 0,
            packed_k_head_stride_bytes: 0,
            packed_v_head_stride_bytes: 0,
            k_transform0: std::ptr::null(),
            k_transform1: std::ptr::null(),
            k_transform0_elements: 0,
            k_transform1_elements: 0,
        }
    }
}

impl Default for FlashAttnCkFwdParams {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum FlashAttnCkError {
    Load {
        path: PathBuf,
        source: libloading::Error,
    },
    Symbol {
        name: &'static str,
        source: libloading::Error,
    },
    AbiVersion {
        expected: u32,
        actual: u32,
    },
    Call {
        operation: &'static str,
        status: i32,
        message: String,
    },
}

impl fmt::Display for FlashAttnCkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { path, source } => {
                write!(
                    f,
                    "load FlashAttention CK sidecar {}: {source}",
                    path.display()
                )
            }
            Self::Symbol { name, source } => {
                write!(f, "resolve FlashAttention CK symbol {name}: {source}")
            }
            Self::AbiVersion { expected, actual } => write!(
                f,
                "FlashAttention CK ABI mismatch: expected {expected}, found {actual}"
            ),
            Self::Call {
                operation,
                status,
                message,
            } => write!(
                f,
                "FlashAttention CK {operation} failed with status {status}: {message}"
            ),
        }
    }
}

impl Error for FlashAttnCkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Load { source, .. } | Self::Symbol { source, .. } => Some(source),
            Self::AbiVersion { .. } | Self::Call { .. } => None,
        }
    }
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CapabilitiesFn = unsafe extern "C" fn(*mut FlashAttnCkCapability, usize) -> usize;
type WorkspaceBytesFn = unsafe extern "C" fn(*const FlashAttnCkFwdParams) -> usize;
type FwdFn = unsafe extern "C" fn(*const FlashAttnCkFwdParams, *mut c_char, usize) -> i32;

/// Loaded sidecar and its stable function table.
pub struct FlashAttnCk {
    _library: &'static Library,
    abi_version: u32,
    capabilities: Vec<FlashAttnCkCapability>,
    workspace_bytes: WorkspaceBytesFn,
    fwd_supported: FwdFn,
    fwd: FwdFn,
}

impl FlashAttnCk {
    /// Load one explicit sidecar path. No soname search or implicit fallback is
    /// performed, so enabling the Cargo feature alone cannot change execution.
    ///
    /// The loaded library is intentionally pinned for the rest of the process.
    /// HIP launches are asynchronous, so unloading the code object when this
    /// handle is dropped would be unsafe while any submitted work is pending.
    ///
    /// # Safety
    ///
    /// `path` must identify a trusted native library implementing the declared
    /// ABI. Loading a native library may execute constructors, and the resolved
    /// symbols are trusted to follow their declared function signatures.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self, FlashAttnCkError> {
        let path = path.as_ref();
        let library =
            unsafe { Library::new(path.as_os_str()) }.map_err(|source| FlashAttnCkError::Load {
                path: path.to_path_buf(),
                source,
            })?;

        let (actual_abi, capabilities, workspace_bytes, fwd_supported, fwd) = unsafe {
            let abi_version: Symbol<'_, AbiVersionFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_abi_version",
                "abi_version",
            )?;
            let fwd_supported: Symbol<'_, FwdFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_fwd_supported",
                "fwd_supported",
            )?;
            let capabilities: Symbol<'_, CapabilitiesFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_capabilities",
                "capabilities",
            )?;
            let workspace_bytes: Symbol<'_, WorkspaceBytesFn> = symbol(
                &library,
                b"hipfire_flash_attn_ck_fwd_workspace_bytes",
                "fwd_workspace_bytes",
            )?;
            let fwd: Symbol<'_, FwdFn> = symbol(&library, b"hipfire_flash_attn_ck_fwd", "fwd")?;

            let actual = abi_version();
            if !is_supported_abi(actual) {
                return Err(FlashAttnCkError::AbiVersion {
                    expected: FLASH_ATTN_CK_ABI_VERSION,
                    actual,
                });
            }

            let count = capabilities(std::ptr::null_mut(), 0);
            let mut cells = vec![
                FlashAttnCkCapability {
                    abi_version: actual,
                    struct_size: std::mem::size_of::<FlashAttnCkCapability>() as u32,
                    arch: 0,
                    dtype: 0,
                    k_format: 0,
                    v_format: 0,
                    head_dim: 0,
                    flags: 0,
                };
                count
            ];
            let written = capabilities(cells.as_mut_ptr(), cells.len());
            if written != count {
                return Err(FlashAttnCkError::Call {
                    operation: "capability query",
                    status: -1,
                    message: format!("sidecar reported {count} cells but wrote {written}"),
                });
            }
            if cells.is_empty() {
                return Err(FlashAttnCkError::Call {
                    operation: "capability query",
                    status: -1,
                    message: "sidecar exported no capability cells".to_string(),
                });
            }
            for cell in &cells {
                if !capability_is_compatible(actual, cell) {
                    return Err(FlashAttnCkError::Call {
                        operation: "capability query",
                        status: -1,
                        message: "sidecar returned an incompatible capability cell".to_string(),
                    });
                }
            }

            (actual, cells, *workspace_bytes, *fwd_supported, *fwd)
        };
        let library = Box::leak(Box::new(library));
        Ok(Self {
            _library: library,
            abi_version: actual_abi,
            capabilities,
            workspace_bytes,
            fwd_supported,
            fwd,
        })
    }

    pub fn capabilities(&self) -> &[FlashAttnCkCapability] {
        &self.capabilities
    }

    pub fn supports(&self, request: FlashAttnCkRequest) -> bool {
        request.required_flags & !FLASH_ATTN_CK_KNOWN_CAP_FLAGS == 0
            && self.capabilities.iter().any(|cell| cell.supports(request))
    }

    pub fn workspace_bytes(&self, params: &FlashAttnCkFwdParams) -> usize {
        let params = self.compatible_params(params);
        unsafe { (self.workspace_bytes)(&params) }
    }

    pub fn is_supported(&self, params: &FlashAttnCkFwdParams) -> Result<(), FlashAttnCkError> {
        self.call("support check", self.fwd_supported, params)
    }

    /// Launch the sidecar on the stream stored in `params`.
    ///
    /// # Safety
    ///
    /// All pointers in `params` must name device allocations with the declared
    /// shape and element strides. They must remain valid until the asynchronous
    /// operation on `params.stream` has completed.
    pub unsafe fn forward(&self, params: &FlashAttnCkFwdParams) -> Result<(), FlashAttnCkError> {
        self.call("forward", self.fwd, params)
    }

    fn call(
        &self,
        operation: &'static str,
        function: FwdFn,
        params: &FlashAttnCkFwdParams,
    ) -> Result<(), FlashAttnCkError> {
        let params = self.compatible_params(params);
        let mut error = [0u8; ERROR_CAPACITY];
        let status = unsafe { function(&params, error.as_mut_ptr().cast::<c_char>(), error.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(FlashAttnCkError::Call {
                operation,
                status,
                message: error_message(&error),
            })
        }
    }

    fn compatible_params(&self, params: &FlashAttnCkFwdParams) -> FlashAttnCkFwdParams {
        let mut compatible = *params;
        compatible.abi_version = self.abi_version;
        compatible
    }
}

impl crate::Gpu {
    /// Try the first serving capability cell. `Ok(false)` is an intentional
    /// native fallback, including sidecar support/launch failures.
    #[allow(clippy::too_many_arguments)]
    pub fn try_flash_attn_ck_q8_d256_prefill(
        &mut self,
        q: &crate::GpuTensor,
        k_cache: &crate::GpuTensor,
        v_cache: &crate::GpuTensor,
        output: &crate::GpuTensor,
        seqlen_q: usize,
        seqlen_k: usize,
        nhead_q: usize,
        nhead_k: usize,
        contiguous_prefix: bool,
        has_tree_bias: bool,
        window: usize,
        block_start: usize,
        block_cols: usize,
    ) -> hip_bridge::HipResult<bool> {
        if self.flash_attn_ck.is_none() {
            return Ok(false);
        }
        let Some(arch) = (match self.arch.as_str() {
            "gfx1100" => Some(FlashAttnCkArch::Gfx1100),
            "gfx1151" => Some(FlashAttnCkArch::Gfx1151),
            "gfx1201" => Some(FlashAttnCkArch::Gfx1201),
            _ => None,
        }) else {
            return Ok(false);
        };
        if q.dtype != crate::DType::F32 || output.dtype != crate::DType::F32 {
            self.report_flash_attn_ck_route("dtype_miss");
            return Ok(false);
        }
        // Fail closed when Redline is recording: CK launches bypass the native
        // launch_maybe_blob_bound recorder and would produce a tape missing the
        // attention work. Fall back to the native path alongside the existing
        // graph-capture rejection so capture counters stay consistent.
        if self.replay.is_recording() {
            self.report_flash_attn_ck_route("replay_recording");
            return Ok(false);
        }
        if self.graphs.capture_mode {
            self.report_flash_attn_ck_route("graph_capture");
            return Ok(false);
        }
        let request = FlashAttnCkRequest {
            arch,
            dtype: FlashAttnCkDType::F32,
            k_format: FlashAttnCkKvFormat::Q8,
            v_format: FlashAttnCkKvFormat::Q8,
            head_dim: 256,
            required_flags: FLASH_ATTN_CK_CAP_CAUSAL | FLASH_ATTN_CK_CAP_GQA,
        };
        let input = FlashAttnCkPrefillInput {
            request,
            batch_size: seqlen_q,
            nhead_q,
            nhead_k,
            causal: true,
            contiguous_prefix,
            capture_mode: self.graphs.capture_mode,
            replay_recording: self.replay.is_recording(),
            has_tree_bias,
            window,
            block_start,
            block_cols,
        };
        let decision = select_q8_d256_prefill(self.flash_attn_ck.as_ref().unwrap(), input);
        if let Err(reason) = decision {
            self.report_flash_attn_ck_route(reject_reason_name(reason));
            return Ok(false);
        }

        let mut params = FlashAttnCkFwdParams::new();
        params.q = q.buf.as_ptr();
        params.k = k_cache.buf.as_ptr();
        params.v = v_cache.buf.as_ptr();
        params.out = output.buf.as_ptr();
        params.stream = self
            .active_stream
            .as_ref()
            .map_or(std::ptr::null_mut(), hip_bridge::Stream::as_raw);
        params.dtype = FlashAttnCkDType::F32 as i32;
        params.k_format = FlashAttnCkKvFormat::Q8 as i32;
        params.v_format = FlashAttnCkKvFormat::Q8 as i32;
        params.batch = 1;
        params.seqlen_q = seqlen_q as i32;
        params.seqlen_k = seqlen_k as i32;
        params.nhead_q = nhead_q as i32;
        params.nhead_k = nhead_k as i32;
        params.head_dim = 256;
        params.causal = 1;
        params.softmax_scale = 1.0 / 16.0;
        params.stride_q = (nhead_q * 256) as i64;
        params.stride_k = (nhead_k * 256) as i64;
        params.stride_v = params.stride_k;
        params.stride_out = params.stride_q;
        params.nhead_stride_q = 256;
        params.nhead_stride_k = 256;
        params.nhead_stride_v = 256;
        params.nhead_stride_out = 256;
        params.batch_stride_q = (seqlen_q * nhead_q * 256) as i64;
        params.batch_stride_k = (seqlen_k * nhead_k * 256) as i64;
        params.batch_stride_v = params.batch_stride_k;
        params.batch_stride_out = params.batch_stride_q;
        const PACKED_Q8_D256_HEAD_BYTES: usize = 272;
        params.packed_k_row_stride_bytes = (nhead_k * PACKED_Q8_D256_HEAD_BYTES) as i64;
        params.packed_v_row_stride_bytes = params.packed_k_row_stride_bytes;
        params.packed_k_head_stride_bytes = PACKED_Q8_D256_HEAD_BYTES as i64;
        params.packed_v_head_stride_bytes = PACKED_Q8_D256_HEAD_BYTES as i64;

        let required = self
            .flash_attn_ck
            .as_ref()
            .unwrap()
            .workspace_bytes(&params);
        let Some(workspace) = self.flash_attn_ck_workspace.as_ref() else {
            self.report_flash_attn_ck_route("workspace_unconfigured");
            return Ok(false);
        };
        if workspace.size() < required {
            self.report_flash_attn_ck_route("workspace_too_small");
            return Ok(false);
        }
        params.workspace = workspace.as_ptr();
        params.workspace_bytes = workspace.size();
        let support = self.flash_attn_ck.as_ref().unwrap().is_supported(&params);
        if let Err(error) = support {
            if self.flash_attn_ck_reported_routes.insert("support_error") {
                eprintln!("optional CK attention fallback (support_error): {error}");
            }
            return Ok(false);
        }
        let launch = unsafe { self.flash_attn_ck.as_ref().unwrap().forward(&params) };
        if let Err(error) = launch {
            if self.flash_attn_ck_reported_routes.insert("launch_error") {
                eprintln!("optional CK attention fallback (launch_error): {error}");
            }
            return Ok(false);
        }
        self.report_flash_attn_ck_route("selected_q8_d256");
        Ok(true)
    }

    /// Try the explicit Asym3-Givens-K/Q8-V D256 prefill cell. Unsupported
    /// shapes and sidecar failures preserve the native attention fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn try_flash_attn_ck_asym3_givens_prefill(
        &mut self,
        q: &crate::GpuTensor,
        k_cache: &crate::GpuTensor,
        v_cache: &crate::GpuTensor,
        output: &crate::GpuTensor,
        cos_theta: &crate::GpuTensor,
        sin_theta: &crate::GpuTensor,
        seqlen_q: usize,
        seqlen_k: usize,
        nhead_q: usize,
        nhead_k: usize,
        head_dim: usize,
        contiguous_prefix: bool,
        has_tree_bias: bool,
        window: usize,
        block_start: usize,
        block_cols: usize,
    ) -> hip_bridge::HipResult<bool> {
        self.try_flash_attn_ck_transformed_prefill_impl(
            FlashAttnCkKvFormat::Asym3Givens,
            q,
            k_cache,
            v_cache,
            output,
            cos_theta,
            sin_theta,
            seqlen_q,
            seqlen_k,
            nhead_q,
            nhead_k,
            head_dim,
            contiguous_prefix,
            has_tree_bias,
            window,
            block_start,
            block_cols,
        )
    }

    /// Try the explicit Asym3-FWHT-K/Q8-V D256 prefill cell.
    #[allow(clippy::too_many_arguments)]
    pub fn try_flash_attn_ck_asym3_fwht_prefill(
        &mut self,
        q: &crate::GpuTensor,
        k_cache: &crate::GpuTensor,
        v_cache: &crate::GpuTensor,
        output: &crate::GpuTensor,
        signs1: &crate::GpuTensor,
        signs2: &crate::GpuTensor,
        seqlen_q: usize,
        seqlen_k: usize,
        nhead_q: usize,
        nhead_k: usize,
        head_dim: usize,
        contiguous_prefix: bool,
        has_tree_bias: bool,
        window: usize,
        block_start: usize,
        block_cols: usize,
    ) -> hip_bridge::HipResult<bool> {
        self.try_flash_attn_ck_transformed_prefill_impl(
            FlashAttnCkKvFormat::Asym3Fwht,
            q,
            k_cache,
            v_cache,
            output,
            signs1,
            signs2,
            seqlen_q,
            seqlen_k,
            nhead_q,
            nhead_k,
            head_dim,
            contiguous_prefix,
            has_tree_bias,
            window,
            block_start,
            block_cols,
        )
    }

    /// Try the explicit Asym4-Givens-K/Q8-V D256 prefill cell.
    #[allow(clippy::too_many_arguments)]
    pub fn try_flash_attn_ck_asym4_givens_prefill(
        &mut self,
        q: &crate::GpuTensor,
        k_cache: &crate::GpuTensor,
        v_cache: &crate::GpuTensor,
        output: &crate::GpuTensor,
        cos_theta: &crate::GpuTensor,
        sin_theta: &crate::GpuTensor,
        seqlen_q: usize,
        seqlen_k: usize,
        nhead_q: usize,
        nhead_k: usize,
        head_dim: usize,
        contiguous_prefix: bool,
        has_tree_bias: bool,
        window: usize,
        block_start: usize,
        block_cols: usize,
    ) -> hip_bridge::HipResult<bool> {
        self.try_flash_attn_ck_transformed_prefill_impl(
            FlashAttnCkKvFormat::Asym4Givens,
            q,
            k_cache,
            v_cache,
            output,
            cos_theta,
            sin_theta,
            seqlen_q,
            seqlen_k,
            nhead_q,
            nhead_k,
            head_dim,
            contiguous_prefix,
            has_tree_bias,
            window,
            block_start,
            block_cols,
        )
    }

    /// Try the explicit Asym4-FWHT-K/Q8-V D256 prefill cell.
    #[allow(clippy::too_many_arguments)]
    pub fn try_flash_attn_ck_asym4_fwht_prefill(
        &mut self,
        q: &crate::GpuTensor,
        k_cache: &crate::GpuTensor,
        v_cache: &crate::GpuTensor,
        output: &crate::GpuTensor,
        signs1: &crate::GpuTensor,
        signs2: &crate::GpuTensor,
        seqlen_q: usize,
        seqlen_k: usize,
        nhead_q: usize,
        nhead_k: usize,
        head_dim: usize,
        contiguous_prefix: bool,
        has_tree_bias: bool,
        window: usize,
        block_start: usize,
        block_cols: usize,
    ) -> hip_bridge::HipResult<bool> {
        self.try_flash_attn_ck_transformed_prefill_impl(
            FlashAttnCkKvFormat::Asym4Fwht,
            q,
            k_cache,
            v_cache,
            output,
            signs1,
            signs2,
            seqlen_q,
            seqlen_k,
            nhead_q,
            nhead_k,
            head_dim,
            contiguous_prefix,
            has_tree_bias,
            window,
            block_start,
            block_cols,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_flash_attn_ck_transformed_prefill_impl(
        &mut self,
        k_format: FlashAttnCkKvFormat,
        q: &crate::GpuTensor,
        k_cache: &crate::GpuTensor,
        v_cache: &crate::GpuTensor,
        output: &crate::GpuTensor,
        cos_theta: &crate::GpuTensor,
        sin_theta: &crate::GpuTensor,
        seqlen_q: usize,
        seqlen_k: usize,
        nhead_q: usize,
        nhead_k: usize,
        head_dim: usize,
        contiguous_prefix: bool,
        has_tree_bias: bool,
        window: usize,
        block_start: usize,
        block_cols: usize,
    ) -> hip_bridge::HipResult<bool> {
        if self.flash_attn_ck.is_none() {
            return Ok(false);
        }
        let Some(arch) = (match self.arch.as_str() {
            "gfx1100" => Some(FlashAttnCkArch::Gfx1100),
            "gfx1151" => Some(FlashAttnCkArch::Gfx1151),
            "gfx1201" => Some(FlashAttnCkArch::Gfx1201),
            _ => None,
        }) else {
            return Ok(false);
        };
        let asym4 = matches!(
            k_format,
            FlashAttnCkKvFormat::Asym4Givens | FlashAttnCkKvFormat::Asym4Fwht
        );
        if q.dtype != crate::DType::F32 || output.dtype != crate::DType::F32 {
            self.report_flash_attn_ck_route(if asym4 {
                "asym4_dtype_miss"
            } else {
                "asym3_dtype_miss"
            });
            return Ok(false);
        }
        if self.replay.is_recording() {
            self.report_flash_attn_ck_route("replay_recording");
            return Ok(false);
        }
        if self.graphs.capture_mode {
            self.report_flash_attn_ck_route("graph_capture");
            return Ok(false);
        }
        let request = FlashAttnCkRequest {
            arch,
            dtype: FlashAttnCkDType::F32,
            k_format,
            v_format: FlashAttnCkKvFormat::Q8,
            head_dim: head_dim as i32,
            required_flags: FLASH_ATTN_CK_CAP_CAUSAL | FLASH_ATTN_CK_CAP_GQA,
        };
        let input = FlashAttnCkPrefillInput {
            request,
            batch_size: seqlen_q,
            nhead_q,
            nhead_k,
            causal: true,
            contiguous_prefix,
            capture_mode: self.graphs.capture_mode,
            replay_recording: self.replay.is_recording(),
            has_tree_bias,
            window,
            block_start,
            block_cols,
        };
        let selection = match k_format {
            FlashAttnCkKvFormat::Asym3Givens => select_asym3_givens_prefill_capabilities(
                self.flash_attn_ck.as_ref().unwrap().capabilities(),
                input,
            ),
            FlashAttnCkKvFormat::Asym3Fwht => select_asym3_fwht_prefill_capabilities(
                self.flash_attn_ck.as_ref().unwrap().capabilities(),
                input,
            ),
            FlashAttnCkKvFormat::Asym4Givens | FlashAttnCkKvFormat::Asym4Fwht => {
                select_asym4_prefill_capabilities(
                    self.flash_attn_ck.as_ref().unwrap().capabilities(),
                    input,
                )
            }
            _ => Err(FlashAttnCkRejectReason::UnsupportedFormat),
        };
        if let Err(reason) = selection {
            self.report_flash_attn_ck_route(reject_reason_name(reason));
            return Ok(false);
        }

        let packed_k_head_bytes = match k_format {
            FlashAttnCkKvFormat::Asym3Givens | FlashAttnCkKvFormat::Asym3Fwht => {
                4 + head_dim * 3 / 8
            }
            FlashAttnCkKvFormat::Asym4Givens | FlashAttnCkKvFormat::Asym4Fwht => 4 + head_dim / 2,
            _ => return Ok(false),
        };
        let packed_v_head_bytes = head_dim / 32 * 34;
        let q_elements = seqlen_q
            .checked_mul(nhead_q)
            .and_then(|value| value.checked_mul(head_dim));
        let k_bytes = seqlen_k
            .checked_mul(nhead_k)
            .and_then(|value| value.checked_mul(packed_k_head_bytes));
        let v_bytes = seqlen_k
            .checked_mul(nhead_k)
            .and_then(|value| value.checked_mul(packed_v_head_bytes));
        let transform_elements = match k_format {
            FlashAttnCkKvFormat::Asym3Givens => head_dim / 2,
            FlashAttnCkKvFormat::Asym3Fwht => head_dim,
            FlashAttnCkKvFormat::Asym4Givens | FlashAttnCkKvFormat::Asym4Fwht => 128,
            _ => return Ok(false),
        };
        if q_elements.is_none_or(|required| q.numel() < required || output.numel() < required)
            || k_bytes.is_none_or(|required| k_cache.byte_size() < required)
            || v_bytes.is_none_or(|required| v_cache.byte_size() < required)
            || cos_theta.dtype != crate::DType::F32
            || sin_theta.dtype != crate::DType::F32
            || cos_theta.numel() < transform_elements
            || sin_theta.numel() < transform_elements
        {
            self.report_flash_attn_ck_route(if asym4 {
                "asym4_buffer_contract_miss"
            } else {
                "asym3_buffer_contract_miss"
            });
            return Ok(false);
        }

        let mut params = FlashAttnCkFwdParams::new();
        params.q = q.buf.as_ptr();
        params.k = k_cache.buf.as_ptr();
        params.v = v_cache.buf.as_ptr();
        params.out = output.buf.as_ptr();
        params.stream = self
            .active_stream
            .as_ref()
            .map_or(std::ptr::null_mut(), hip_bridge::Stream::as_raw);
        params.dtype = FlashAttnCkDType::F32 as i32;
        params.k_format = k_format as i32;
        params.v_format = FlashAttnCkKvFormat::Q8 as i32;
        params.batch = 1;
        params.seqlen_q = seqlen_q as i32;
        params.seqlen_k = seqlen_k as i32;
        params.nhead_q = nhead_q as i32;
        params.nhead_k = nhead_k as i32;
        params.head_dim = head_dim as i32;
        params.causal = 1;
        params.softmax_scale = 1.0 / (head_dim as f32).sqrt();
        params.stride_q = (nhead_q * head_dim) as i64;
        params.stride_k = (nhead_k * head_dim) as i64;
        params.stride_v = params.stride_k;
        params.stride_out = params.stride_q;
        params.nhead_stride_q = head_dim as i64;
        params.nhead_stride_k = head_dim as i64;
        params.nhead_stride_v = head_dim as i64;
        params.nhead_stride_out = head_dim as i64;
        params.batch_stride_q = (seqlen_q * nhead_q * head_dim) as i64;
        params.batch_stride_k = (seqlen_k * nhead_k * head_dim) as i64;
        params.batch_stride_v = params.batch_stride_k;
        params.batch_stride_out = params.batch_stride_q;
        params.packed_k_row_stride_bytes = (nhead_k * packed_k_head_bytes) as i64;
        params.packed_v_row_stride_bytes = (nhead_k * packed_v_head_bytes) as i64;
        params.packed_k_head_stride_bytes = packed_k_head_bytes as i64;
        params.packed_v_head_stride_bytes = packed_v_head_bytes as i64;
        params.k_transform0 = cos_theta.buf.as_ptr();
        params.k_transform1 = sin_theta.buf.as_ptr();
        params.k_transform0_elements = cos_theta.numel() as i64;
        params.k_transform1_elements = sin_theta.numel() as i64;

        let required = self
            .flash_attn_ck
            .as_ref()
            .unwrap()
            .workspace_bytes(&params);
        let Some(workspace) = self.flash_attn_ck_workspace.as_ref() else {
            self.report_flash_attn_ck_route("workspace_unconfigured");
            return Ok(false);
        };
        if workspace.size() < required {
            self.report_flash_attn_ck_route("workspace_too_small");
            return Ok(false);
        }
        params.workspace = workspace.as_ptr();
        params.workspace_bytes = workspace.size();
        if let Err(error) = self.flash_attn_ck.as_ref().unwrap().is_supported(&params) {
            let reason = if asym4 {
                "asym4_support_error"
            } else {
                "asym3_support_error"
            };
            if self.flash_attn_ck_reported_routes.insert(reason) {
                eprintln!("optional CK attention fallback ({reason}): {error}");
            }
            return Ok(false);
        }
        if let Err(error) = unsafe { self.flash_attn_ck.as_ref().unwrap().forward(&params) } {
            let reason = if asym4 {
                "asym4_launch_error"
            } else {
                "asym3_launch_error"
            };
            if self.flash_attn_ck_reported_routes.insert(reason) {
                eprintln!("optional CK attention fallback ({reason}): {error}");
            }
            return Ok(false);
        }
        self.report_flash_attn_ck_route(match k_format {
            FlashAttnCkKvFormat::Asym3Givens => "selected_asym3_givens_d256",
            FlashAttnCkKvFormat::Asym3Fwht => "selected_asym3_fwht_d256",
            FlashAttnCkKvFormat::Asym4Givens => "selected_asym4_givens_d256",
            FlashAttnCkKvFormat::Asym4Fwht => "selected_asym4_fwht_d256",
            _ => unreachable!("selector admitted an unsupported transformed format"),
        });
        Ok(true)
    }

    fn report_flash_attn_ck_route(&mut self, reason: &'static str) {
        if self.flash_attn_ck_reported_routes.insert(reason) {
            eprintln!("optional CK attention route: {reason}");
        }
    }
}

fn reject_reason_name(reason: FlashAttnCkRejectReason) -> &'static str {
    match reason {
        FlashAttnCkRejectReason::Decode => "decode",
        FlashAttnCkRejectReason::SmallQuery => "small_query",
        FlashAttnCkRejectReason::UnsupportedFormat => "format_miss",
        FlashAttnCkRejectReason::UnsupportedHeadDim => "head_dim_miss",
        FlashAttnCkRejectReason::InvalidGqa => "gqa_miss",
        FlashAttnCkRejectReason::NonCausal => "non_causal",
        FlashAttnCkRejectReason::NonContiguousPrefix => "non_contiguous_prefix",
        FlashAttnCkRejectReason::GraphCapture => "graph_capture",
        FlashAttnCkRejectReason::ReplayRecording => "replay_recording",
        FlashAttnCkRejectReason::TreeAttention => "tree_attention",
        FlashAttnCkRejectReason::WindowedAttention => "windowed_attention",
        FlashAttnCkRejectReason::BlockAttention => "block_attention",
        FlashAttnCkRejectReason::CapabilityMiss => "capability_miss",
    }
}

unsafe fn symbol<'library, T>(
    library: &'library Library,
    bytes: &[u8],
    name: &'static str,
) -> Result<Symbol<'library, T>, FlashAttnCkError> {
    library
        .get(bytes)
        .map_err(|source| FlashAttnCkError::Symbol { name, source })
}

fn error_message(buffer: &[u8]) -> String {
    let len = buffer
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..len]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn fwd_params_matches_c_abi_layout() {
        assert_eq!(std::mem::size_of::<FlashAttnCkFwdParams>(), 272);
        assert_eq!(std::mem::align_of::<FlashAttnCkFwdParams>(), 8);
        assert_eq!(std::mem::offset_of!(FlashAttnCkFwdParams, q), 8);
        assert_eq!(std::mem::offset_of!(FlashAttnCkFwdParams, workspace), 40);
        assert_eq!(std::mem::offset_of!(FlashAttnCkFwdParams, dtype), 64);
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkFwdParams, softmax_scale),
            104
        );
        assert_eq!(std::mem::offset_of!(FlashAttnCkFwdParams, stride_q), 112);
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkFwdParams, batch_stride_out),
            200
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkFwdParams, packed_k_row_stride_bytes),
            208
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkFwdParams, packed_k_head_stride_bytes),
            224
        );
        assert_eq!(
            std::mem::offset_of!(FlashAttnCkFwdParams, k_transform0),
            240
        );
    }

    #[test]
    fn capability_matches_c_abi_layout() {
        assert_eq!(std::mem::size_of::<FlashAttnCkCapability>(), 32);
        assert_eq!(std::mem::align_of::<FlashAttnCkCapability>(), 4);
        assert_eq!(std::mem::offset_of!(FlashAttnCkCapability, arch), 8);
        assert_eq!(std::mem::offset_of!(FlashAttnCkCapability, flags), 28);
    }

    #[test]
    fn abi_v3_compatibility_is_limited_to_legacy_cells() {
        assert!(!is_supported_abi(2));
        assert!(is_supported_abi(3));
        assert!(is_supported_abi(4));
        assert!(!is_supported_abi(5));

        let mut cell = q8_d256_capability();
        cell.abi_version = 3;
        assert!(capability_is_compatible(3, &cell));
        cell.k_format = FlashAttnCkKvFormat::Asym3Givens as i32;
        assert!(!capability_is_compatible(3, &cell));
    }

    fn dense_capability(arch: FlashAttnCkArch) -> FlashAttnCkCapability {
        FlashAttnCkCapability {
            abi_version: FLASH_ATTN_CK_ABI_VERSION,
            struct_size: std::mem::size_of::<FlashAttnCkCapability>() as u32,
            arch: arch as i32,
            dtype: FlashAttnCkDType::F16 as i32,
            k_format: FlashAttnCkKvFormat::DenseF16 as i32,
            v_format: FlashAttnCkKvFormat::DenseF16 as i32,
            head_dim: 64,
            flags: FLASH_ATTN_CK_CAP_CAUSAL | FLASH_ATTN_CK_CAP_GQA,
        }
    }

    fn q8_d256_capability() -> FlashAttnCkCapability {
        FlashAttnCkCapability {
            dtype: FlashAttnCkDType::F32 as i32,
            k_format: FlashAttnCkKvFormat::Q8 as i32,
            v_format: FlashAttnCkKvFormat::Q8 as i32,
            head_dim: 256,
            ..dense_capability(FlashAttnCkArch::Gfx1100)
        }
    }

    fn asym3_givens_d256_capability() -> FlashAttnCkCapability {
        FlashAttnCkCapability {
            k_format: FlashAttnCkKvFormat::Asym3Givens as i32,
            ..q8_d256_capability()
        }
    }

    fn asym3_givens_d256_capability_gfx1201() -> FlashAttnCkCapability {
        FlashAttnCkCapability {
            arch: FlashAttnCkArch::Gfx1201 as i32,
            ..asym3_givens_d256_capability()
        }
    }

    fn asym3_fwht_d256_capability() -> FlashAttnCkCapability {
        FlashAttnCkCapability {
            k_format: FlashAttnCkKvFormat::Asym3Fwht as i32,
            ..q8_d256_capability()
        }
    }

    fn asym4_d256_capability(format: FlashAttnCkKvFormat) -> FlashAttnCkCapability {
        FlashAttnCkCapability {
            k_format: format as i32,
            ..q8_d256_capability()
        }
    }

    fn eligible_q8_prefill() -> FlashAttnCkPrefillInput {
        FlashAttnCkPrefillInput {
            request: FlashAttnCkRequest {
                arch: FlashAttnCkArch::Gfx1100,
                dtype: FlashAttnCkDType::F32,
                k_format: FlashAttnCkKvFormat::Q8,
                v_format: FlashAttnCkKvFormat::Q8,
                head_dim: 256,
                required_flags: FLASH_ATTN_CK_CAP_CAUSAL | FLASH_ATTN_CK_CAP_GQA,
            },
            batch_size: 128,
            nhead_q: 24,
            nhead_k: 4,
            causal: true,
            contiguous_prefix: true,
            capture_mode: false,
            replay_recording: false,
            has_tree_bias: false,
            window: 0,
            block_start: 0,
            block_cols: 0,
        }
    }

    #[test]
    fn q8_d256_prefill_selector_accepts_exact_cell() {
        let input = eligible_q8_prefill();
        assert_eq!(
            select_q8_d256_prefill_capabilities(&[q8_d256_capability()], input),
            Ok(input.request)
        );
    }

    #[test]
    fn asym3_givens_prefill_selector_is_exact_and_fail_closed() {
        let q8 = eligible_q8_prefill();
        let input = FlashAttnCkPrefillInput {
            request: FlashAttnCkRequest {
                k_format: FlashAttnCkKvFormat::Asym3Givens,
                ..q8.request
            },
            ..q8
        };
        let cell = asym3_givens_d256_capability();
        assert_eq!(
            select_asym3_givens_prefill_capabilities(&[cell], input),
            Ok(input.request)
        );
        let minimum_bulk_query = FlashAttnCkPrefillInput {
            batch_size: FLASH_ATTN_CK_MIN_QUERY_TOKENS,
            ..input
        };
        assert_eq!(
            select_asym3_givens_prefill_capabilities(&[cell], minimum_bulk_query),
            Ok(input.request)
        );
        for (rejected, reason) in [
            (
                FlashAttnCkPrefillInput {
                    batch_size: 1,
                    ..input
                },
                FlashAttnCkRejectReason::Decode,
            ),
            (
                FlashAttnCkPrefillInput {
                    batch_size: FLASH_ATTN_CK_MIN_QUERY_TOKENS - 1,
                    ..input
                },
                FlashAttnCkRejectReason::SmallQuery,
            ),
            (
                FlashAttnCkPrefillInput {
                    capture_mode: true,
                    ..input
                },
                FlashAttnCkRejectReason::GraphCapture,
            ),
            (
                FlashAttnCkPrefillInput {
                    has_tree_bias: true,
                    ..input
                },
                FlashAttnCkRejectReason::TreeAttention,
            ),
        ] {
            assert_eq!(
                select_asym3_givens_prefill_capabilities(&[cell], rejected),
                Err(reason)
            );
        }
        let d512 = FlashAttnCkPrefillInput {
            request: FlashAttnCkRequest {
                head_dim: 512,
                ..input.request
            },
            ..input
        };
        assert_eq!(
            select_asym3_givens_prefill_capabilities(&[cell], d512),
            Err(FlashAttnCkRejectReason::UnsupportedHeadDim)
        );
        let fwht = FlashAttnCkPrefillInput {
            request: FlashAttnCkRequest {
                k_format: FlashAttnCkKvFormat::Asym3Fwht,
                ..input.request
            },
            ..input
        };
        assert_eq!(
            select_asym3_givens_prefill_capabilities(&[cell], fwht),
            Err(FlashAttnCkRejectReason::UnsupportedFormat)
        );
    }

    #[test]
    fn asym3_givens_prefill_selector_accepts_exact_gfx1201_cell() {
        let q8 = eligible_q8_prefill();
        let input = FlashAttnCkPrefillInput {
            request: FlashAttnCkRequest {
                arch: FlashAttnCkArch::Gfx1201,
                k_format: FlashAttnCkKvFormat::Asym3Givens,
                ..q8.request
            },
            ..q8
        };
        assert_eq!(
            select_asym3_givens_prefill_capabilities(
                &[asym3_givens_d256_capability_gfx1201()],
                input,
            ),
            Ok(input.request)
        );
        assert_eq!(
            select_asym3_givens_prefill_capabilities(&[asym3_givens_d256_capability()], input),
            Err(FlashAttnCkRejectReason::CapabilityMiss)
        );
    }

    #[test]
    fn asym3_fwht_prefill_selector_is_exact_and_fail_closed() {
        let q8 = eligible_q8_prefill();
        let input = FlashAttnCkPrefillInput {
            request: FlashAttnCkRequest {
                k_format: FlashAttnCkKvFormat::Asym3Fwht,
                ..q8.request
            },
            ..q8
        };
        let cell = asym3_fwht_d256_capability();
        assert_eq!(
            select_asym3_fwht_prefill_capabilities(&[cell], input),
            Ok(input.request)
        );
        assert_eq!(
            select_asym3_fwht_prefill_capabilities(
                &[cell],
                FlashAttnCkPrefillInput {
                    capture_mode: true,
                    ..input
                }
            ),
            Err(FlashAttnCkRejectReason::GraphCapture)
        );
        assert_eq!(
            select_asym3_fwht_prefill_capabilities(
                &[cell],
                FlashAttnCkPrefillInput {
                    request: FlashAttnCkRequest {
                        head_dim: 512,
                        ..input.request
                    },
                    ..input
                }
            ),
            Err(FlashAttnCkRejectReason::UnsupportedHeadDim)
        );
        assert_eq!(
            select_asym3_fwht_prefill_capabilities(
                &[cell],
                FlashAttnCkPrefillInput {
                    request: FlashAttnCkRequest {
                        v_format: FlashAttnCkKvFormat::Asym3Fwht,
                        ..input.request
                    },
                    ..input
                }
            ),
            Err(FlashAttnCkRejectReason::UnsupportedFormat)
        );
    }

    #[test]
    fn asym4_prefill_selector_is_format_exact_and_fail_closed() {
        let q8 = eligible_q8_prefill();
        for format in [
            FlashAttnCkKvFormat::Asym4Givens,
            FlashAttnCkKvFormat::Asym4Fwht,
        ] {
            let input = FlashAttnCkPrefillInput {
                request: FlashAttnCkRequest {
                    k_format: format,
                    ..q8.request
                },
                ..q8
            };
            let cell = asym4_d256_capability(format);
            assert_eq!(
                select_asym4_prefill_capabilities(&[cell], input),
                Ok(input.request)
            );
            assert_eq!(
                select_asym4_prefill_capabilities(
                    &[cell],
                    FlashAttnCkPrefillInput {
                        capture_mode: true,
                        ..input
                    }
                ),
                Err(FlashAttnCkRejectReason::GraphCapture)
            );
            assert_eq!(
                select_asym4_prefill_capabilities(
                    &[cell],
                    FlashAttnCkPrefillInput {
                        request: FlashAttnCkRequest {
                            head_dim: 128,
                            ..input.request
                        },
                        ..input
                    }
                ),
                Err(FlashAttnCkRejectReason::UnsupportedHeadDim)
            );
        }
    }

    #[test]
    fn q8_prefill_selector_rejects_unsafe_shapes() {
        let cell = q8_d256_capability();
        let cases = [
            (
                FlashAttnCkPrefillInput {
                    batch_size: 1,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::Decode,
            ),
            (
                FlashAttnCkPrefillInput {
                    batch_size: FLASH_ATTN_CK_MIN_QUERY_TOKENS - 1,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::SmallQuery,
            ),
            (
                FlashAttnCkPrefillInput {
                    nhead_q: 23,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::InvalidGqa,
            ),
            (
                FlashAttnCkPrefillInput {
                    contiguous_prefix: false,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::NonContiguousPrefix,
            ),
            (
                FlashAttnCkPrefillInput {
                    capture_mode: true,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::GraphCapture,
            ),
            (
                FlashAttnCkPrefillInput {
                    replay_recording: true,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::ReplayRecording,
            ),
            (
                FlashAttnCkPrefillInput {
                    has_tree_bias: true,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::TreeAttention,
            ),
            (
                FlashAttnCkPrefillInput {
                    window: 4096,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::WindowedAttention,
            ),
            (
                FlashAttnCkPrefillInput {
                    block_cols: 32,
                    ..eligible_q8_prefill()
                },
                FlashAttnCkRejectReason::BlockAttention,
            ),
        ];
        for (input, reason) in cases {
            assert_eq!(
                select_q8_d256_prefill_capabilities(&[cell], input),
                Err(reason)
            );
        }
    }

    #[test]
    fn q8_d256_prefill_selector_rejects_capability_mismatch() {
        assert_eq!(
            select_q8_d256_prefill_capabilities(
                &[dense_capability(FlashAttnCkArch::Gfx1100)],
                eligible_q8_prefill(),
            ),
            Err(FlashAttnCkRejectReason::CapabilityMiss)
        );
    }

    #[test]
    fn capability_matching_is_exact_except_for_flag_subset() {
        let cell = dense_capability(FlashAttnCkArch::Gfx1100);
        let request = FlashAttnCkRequest {
            arch: FlashAttnCkArch::Gfx1100,
            dtype: FlashAttnCkDType::F16,
            k_format: FlashAttnCkKvFormat::DenseF16,
            v_format: FlashAttnCkKvFormat::DenseF16,
            head_dim: 64,
            required_flags: FLASH_ATTN_CK_CAP_CAUSAL,
        };
        assert!(cell.supports(request));
        assert!(!cell.supports(FlashAttnCkRequest {
            arch: FlashAttnCkArch::Gfx1201,
            ..request
        }));
        assert!(!cell.supports(FlashAttnCkRequest {
            k_format: FlashAttnCkKvFormat::Q8,
            v_format: FlashAttnCkKvFormat::Q8,
            ..request
        }));
        assert!(!cell.supports(FlashAttnCkRequest {
            head_dim: 128,
            ..request
        }));
    }

    #[test]
    fn capability_validation_rejects_unknown_contract_values() {
        let valid = dense_capability(FlashAttnCkArch::Gfx1151);
        assert!(valid.is_well_formed());
        assert!(!FlashAttnCkCapability {
            arch: 9999,
            ..valid
        }
        .is_well_formed());
        assert!(!FlashAttnCkCapability {
            k_format: 9999,
            ..valid
        }
        .is_well_formed());
        assert!(!FlashAttnCkCapability {
            head_dim: 0,
            ..valid
        }
        .is_well_formed());
        assert!(!FlashAttnCkCapability {
            flags: 1 << 31,
            ..valid
        }
        .is_well_formed());
    }

    #[test]
    fn defaults_publish_current_abi() {
        let params = FlashAttnCkFwdParams::default();
        assert_eq!(params.abi_version, FLASH_ATTN_CK_ABI_VERSION);
        assert_eq!(params.struct_size as usize, std::mem::size_of_val(&params));
        assert!(params.k_transform0.is_null());
        assert!(params.k_transform1.is_null());
        assert_eq!(params.packed_k_head_stride_bytes, 0);
        assert_eq!(params.packed_v_head_stride_bytes, 0);
    }

    #[test]
    fn missing_sidecar_is_recoverable() {
        let error = unsafe {
            FlashAttnCk::load(OsStr::new(
                "/definitely/missing/libhipfire_flash_attn_ck.so",
            ))
        }
        .err()
        .expect("missing sidecar should return an error");
        assert!(matches!(error, FlashAttnCkError::Load { .. }));
    }

    #[test]
    fn explicit_test_sidecar_loads_and_rejects_invalid_params() {
        let Ok(path) = std::env::var("HIPFIRE_FLASH_ATTN_CK_TEST_LIB") else {
            return;
        };
        let sidecar = unsafe { FlashAttnCk::load(path) }.expect("load explicit test sidecar");
        assert!(sidecar.capabilities().iter().any(|cell| {
            cell.arch == FlashAttnCkArch::Gfx1100 as i32
                && cell.dtype == FlashAttnCkDType::F32 as i32
                && cell.k_format == FlashAttnCkKvFormat::Asym3Givens as i32
                && cell.v_format == FlashAttnCkKvFormat::Q8 as i32
                && cell.head_dim == 256
        }));
        assert!(!sidecar.capabilities().iter().any(|cell| {
            cell.k_format == FlashAttnCkKvFormat::Asym3Givens as i32 && cell.head_dim == 512
        }));
        assert!(sidecar.capabilities().iter().any(|cell| {
            cell.arch == FlashAttnCkArch::Gfx1100 as i32
                && cell.dtype == FlashAttnCkDType::F32 as i32
                && cell.k_format == FlashAttnCkKvFormat::Asym3Fwht as i32
                && cell.v_format == FlashAttnCkKvFormat::Q8 as i32
                && cell.head_dim == 256
        }));
        assert!(!sidecar.capabilities().iter().any(|cell| {
            cell.k_format == FlashAttnCkKvFormat::Asym3Fwht as i32 && cell.head_dim == 512
        }));
        for format in [
            FlashAttnCkKvFormat::Asym4Givens,
            FlashAttnCkKvFormat::Asym4Fwht,
        ] {
            assert!(sidecar.capabilities().iter().any(|cell| {
                cell.arch == FlashAttnCkArch::Gfx1100 as i32
                    && cell.dtype == FlashAttnCkDType::F32 as i32
                    && cell.k_format == format as i32
                    && cell.v_format == FlashAttnCkKvFormat::Q8 as i32
                    && cell.head_dim == 256
            }));
        }
        let error = sidecar
            .is_supported(&FlashAttnCkFwdParams::default())
            .expect_err("zero-shape parameters must be rejected");
        assert!(matches!(
            error,
            FlashAttnCkError::Call {
                operation: "support check",
                status: 1,
                ..
            }
        ));
    }
}
