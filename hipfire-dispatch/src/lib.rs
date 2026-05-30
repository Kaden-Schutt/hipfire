// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::{Arc, Mutex};

// ── 6 family traits (private to dispatch; models never touch these) ──

trait RotationKernel: Send + Sync {
    fn run(&self, cx: &mut Ctx, params: RotationParams) -> Result<(), DispatchError>;
}
trait FusedQkvKernel: Send + Sync {
    fn run(&self, cx: &mut Ctx, params: FusedQkvParams) -> Result<(), DispatchError>;
}
trait AttentionKernel: Send + Sync {
    fn run(&self, cx: &mut Ctx, params: AttnParams) -> Result<(), DispatchError>;
}
trait FusedGateUpKernel: Send + Sync {
    fn run(&self, cx: &mut Ctx, params: FusedGateUpParams) -> Result<(), DispatchError>;
}
trait GemvKernel: Send + Sync {
    fn run(&self, cx: &mut Ctx, params: GemvParams) -> Result<(), DispatchError>;
}
trait GemmKernel: Send + Sync {
    fn run(&self, cx: &mut Ctx, params: GemmParams) -> Result<(), DispatchError>;
}
trait ModelSpecificKernel: Send + Sync {
    fn run(&self, cx: &mut Ctx, params: &dyn std::any::Any) -> Result<(), DispatchError>;
}

// ── Public interface ──

/// One `Dispatch` per model + device. Built once at load time.
pub struct Dispatch {
    rotation: Arc<dyn RotationKernel>,
    fused_qkv: Arc<dyn FusedQkvKernel>,
    attn: Arc<dyn AttentionKernel>,
    fused_gate_up: Arc<dyn FusedGateUpKernel>,
    gemv: Arc<dyn GemvKernel>,
    gemm: Arc<dyn GemmKernel>,
    /// One-off model quirks (DeltaNet state, DeepSeek compressor, …).
    /// Keyed by string tag so models don't import kernel crate types.
    model_specific: Vec<(String, Arc<dyn ModelSpecificKernel>)>,
    /// Graph recorder shared across all dispatch calls for a capture session.
    graph: Mutex<Option<GraphRecorder>>,
}

/// Per-invocation context — temps, position, kv_cache handles.
/// Models own this and pass it through every dispatch call.
pub struct Ctx<'a> {
    pub gpu: &'a Gpu,
    pub pos: u32,
    pub layer_idx: u32,
    pub temps: &'a mut TempPool,
    pub kv_cache: &'a mut KvCache,
    // Graph capture is transparent — Ctx carries the flag, Dispatch checks it.
    pub graph_mode: GraphMode,
}

pub enum GraphMode {
    Direct,                          // launch immediately
    Recording(Option<&'a mut HipGraph>), // record into this graph
}

// ── Param types (owned or borrowed lightweight views) ──

pub struct RotationParams<'a> {
    pub norm_weight: &'a WeightView,
    pub x: &'a TensorView,
    /// Output: plain (post-RMSNorm, pre-rotation).
    pub x_plain: &'a mut TensorView,
    /// Output: FWHT-rotated (post-RMSNorm + post-rotation).
    pub x_rot: &'a mut TensorView,
}

pub enum FusedQkvKind {
    /// Standard: Q, K, V
    Qkv,
    /// DeltaNet: Q, K, V, Z (4 projections)
    Qkvza,
    /// DeepSeek: Q, K, V + compressed KV pair
    QkvCompressed,
}

pub struct FusedQkvParams<'a> {
    pub kind: FusedQkvKind,
    pub weights: &'a [&'a WeightView; 4], // fixed-size, unused entries = None
    pub x: &'a TensorView,
    pub outputs: FusedQkvOutputs<'a>,
}

pub enum FusedQkvOutputs<'a> {
    Qkv(&'a mut TensorView, &'a mut TensorView, &'a mut TensorView),
    Qkvza(
        &'a mut TensorView, &'a mut TensorView,
        &'a mut TensorView, &'a mut TensorView,
    ),
    QkvCompressed(
        &'a mut TensorView, &'a mut TensorView,
        &'a mut TensorView, &'a mut CompressedKvPair,
    ),
}

pub struct AttnParams<'a> {
    pub q: &'a TensorView,
    pub k: &'a TensorView,
    pub v: &'a TensorView,
    pub k_cache: &'a mut KvSlice,
    pub v_cache: &'a mut KvSlice,
    pub mask: Option<&'a TensorView>,
}

pub struct FusedGateUpParams<'a> {
    pub w_gate: &'a WeightView,
    pub w_up: &'a WeightView,
    pub x: &'a TensorView,
    pub output_gate: &'a mut TensorView,
    pub output_up: &'a mut TensorView,
}

pub enum GemvKind {
    Plain,
    SwiGLUResidual {
        gate: &'a TensorView,
        up: &'a TensorView,
        residual: &'a TensorView,
    },
}

pub struct GemvParams<'a> {
    pub kind: GemvKind<'a>,
    pub w: &'a WeightView,
    pub x: &'a TensorView,
    pub y: &'a mut TensorView,
}

pub struct GemmParams<'a> {
    pub w: &'a WeightView,
    pub x: &'a TensorView,
    pub y: &'a mut TensorView,
    pub batch_size: u32,
}

// ── Public methods ──

impl Dispatch {
    pub fn rotation(&self, cx: &mut Ctx, params: RotationParams) -> Result<(), DispatchError> {
        self.dispatch(cx, |inner| inner.rotation.run(cx, params))
    }
    pub fn fused_qkv(&self, cx: &mut Ctx, params: FusedQkvParams) -> Result<(), DispatchError> {
        self.dispatch(cx, |inner| inner.fused_qkv.run(cx, params))
    }
    pub fn attention(&self, cx: &mut Ctx, params: AttnParams) -> Result<(), DispatchError> {
        self.dispatch(cx, |inner| inner.attn.run(cx, params))
    }
    pub fn fused_gate_up(&self, cx: &mut Ctx, params: FusedGateUpParams) -> Result<(), DispatchError> {
        self.dispatch(cx, |inner| inner.fused_gate_up.run(cx, params))
    }
    pub fn gemv(&self, cx: &mut Ctx, params: GemvParams) -> Result<(), DispatchError> {
        self.dispatch(cx, |inner| inner.gemv.run(cx, params))
    }
    pub fn gemm(&self, cx: &mut Ctx, params: GemmParams) -> Result<(), DispatchError> {
        self.dispatch(cx, |inner| inner.gemm.run(cx, params))
    }
}

// ── Error path ──

/// Caught at build time, not runtime.
#[derive(Debug)]
pub enum DispatchError {
    UnsupportedVariant {
        family: &'static str,
        variant: &'static str,
        arch: &'static str,
        quant: &'static str,
    },
    #[allow(dead_code)]
    ModelSpecific { tag: String, detail: String },
}
