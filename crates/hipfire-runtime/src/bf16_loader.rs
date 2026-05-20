//! BF16 HuggingFace safetensors model loader scaffold (Tier 1 foundation).
//!
//! Stretch deliverable in the 2026-05-19 Tier 1 foundation series. When
//! complete, this module loads a BF16 HuggingFace model directly into
//! GPU memory, bypassing the `.hfq` quantized format used by production
//! inference. The Tier 1 calibration path (`collect_imatrix`,
//! `collect_hessian` in `src/bin/`) runs its MFMA-direct forward through
//! the loaded BF16 tensors and feeds activations into the
//! `ActivationCapture` hook on `Gpu`.
//!
//! Status: **scaffold only**. Function signatures, metadata struct, and
//! safetensors-parse plumbing are sketched; the actual `hipMalloc` +
//! `hipMemcpy` of weight bytes is left to the follow-on subagent task.
//!
//! Why a separate loader (not reuse `hfq.rs`):
//! - HFQ files contain quantized weights (HFQ4G256 layout, 4-bit packed
//!   + FP32 scale/zero); calibration needs the FP-precise BF16 source.
//! - HuggingFace safetensors are the canonical BF16 distribution format
//!   used by `llama-imatrix` and `collect_hessian.py`. Loading
//!   safetensors directly removes the GGUF/HFQ-conversion roundtrip
//!   that adds quantization error to the calibration signal.
//! - The captured Hessian / Σx² values produced from a BF16 forward
//!   pass are then used to derive optimal MQ4/MQ3/HFP4 quantization
//!   parameters — feeding the production .hfq files. So this loader
//!   sits at the producer end of the quant pipeline, not the consumer
//!   end.
//!
//! Reference doc for the file format:
//! - safetensors v0.3 spec: https://github.com/huggingface/safetensors
//! - HF model layout: `<dir>/{config.json, tokenizer.json,
//!   model.safetensors[.index.json], *.safetensors}` (single-file or
//!   sharded depending on size).

use hip_bridge::HipResult;
// `DType` will be threaded through once the scaffold gains a real load
// path (Phase 2 — see `load_bf16_model` body TODO). For now, only the
// types referenced in the function signature + struct fields are imported.
use rdna_compute::{Gpu, GpuTensor};
use std::collections::HashMap;
use std::path::Path;

/// A single BF16 weight tensor on the device.
///
/// Mirrors the structure of `rdna_compute::GpuTensor` but tagged with
/// the canonical HuggingFace key so the calibration capture hook can
/// route activations to the right per-tensor accumulator.
#[allow(dead_code)]
pub struct Bf16Tensor {
    /// Canonical HF safetensors key (e.g.
    /// `model.layers.0.self_attn.q_proj.weight`). Mirror of the
    /// `GPTQ_TARGET_SUFFIXES` list in `scripts/collect_hessian.py`
    /// determines which of these get a Hessian collected.
    pub name: String,
    /// Owned device tensor. Always `DType::F16`-tagged for now — the
    /// rdna-compute layer doesn't have a `BF16` enum arm yet. Phase 2
    /// adds `DType::BF16` (a 1-line edit in dispatch.rs::DType once we
    /// have a real BF16 forward path to consume it). For the scaffold,
    /// we mark the tensor with TODO comments so the downstream wiring
    /// knows to re-interpret the bytes.
    ///
    /// TODO(Phase 2): replace with `DType::BF16` once the dtype enum
    /// arm lands. The .size() return is still 2 bytes/element — same
    /// as F16 — so the allocator math is unchanged.
    pub tensor: GpuTensor,
    /// Tensor shape as stored on disk (e.g. `[out_features, in_features]`
    /// for a Linear weight in PyTorch's `[out, in]` row-major convention).
    pub shape: Vec<usize>,
}

/// All BF16 weights for a single model, indexed by HF key.
///
/// The `TrunkBF16` name follows the rustane convention (separate trunk
/// vs. expert / head subtensors). For now, MoE expert tensors live in
/// the same flat map; if memory becomes the bottleneck on large MoEs,
/// Phase 2 can split into `trunk: HashMap<_>, experts: Vec<HashMap<_>>`.
#[allow(dead_code)]
pub struct TrunkBF16 {
    /// All loaded weight tensors keyed by canonical HF safetensors name.
    pub tensors: HashMap<String, Bf16Tensor>,
    /// The model dir we loaded from (for log messages + tokenizer load).
    pub model_dir: std::path::PathBuf,
    /// Model architecture id from `config.json["model_type"]`
    /// (e.g. `qwen3`, `llama`, `mistral`). Drives which arch crate
    /// the calibration forward pass dispatches into.
    pub model_type: String,
    /// Total bytes allocated on device across all tensors (for the
    /// log line at load time).
    pub total_bytes: usize,
}

/// Load a BF16 HuggingFace model directory into GPU memory.
///
/// Scaffold — returns `unimplemented!()`. The real implementation will:
///
/// 1. Parse `<model_dir>/config.json` to get `model_type` + hidden dim
///    + layer count.
/// 2. Parse `<model_dir>/model.safetensors.index.json` (or fallback to
///    single-file `model.safetensors`) for the tensor → shard map.
/// 3. For each tensor key:
///    a. Read the safetensors header to get `dtype`, `shape`,
///       `data_offsets`.
///    b. Verify `dtype == "BF16"` (reject FP16/FP32 inputs — quantizer
///       wants BF16 source-of-truth weights).
///    c. `gpu.alloc_tensor(&shape, DType::F16 /* TODO: BF16 */)` →
///       device buffer.
///    d. `hipMemcpyHtoD` of the raw 2-byte BF16 payload from mmap'd
///       safetensors → device buffer.
/// 4. Build the `TrunkBF16` map and return.
///
/// The scaffold currently allocates no memory and copies nothing; it
/// constructs an empty `TrunkBF16` struct so that downstream Phase 2
/// integration code can compile against the type immediately.
pub fn load_bf16_model(_gpu: &mut Gpu, model_dir: &Path) -> HipResult<TrunkBF16> {
    // TODO(Phase 2):
    //   1. Open <model_dir>/config.json + extract model_type.
    //   2. Walk safetensors index + read per-tensor headers.
    //   3. For each BF16 tensor: alloc on GPU + memcpy bytes.
    //   4. Populate the tensors map.
    //
    // For now, return an empty TrunkBF16 so downstream type-checks pass
    // but a runtime caller sees the scaffold panic.
    unimplemented!(
        "load_bf16_model: scaffold only — Phase 2 wires the actual safetensors \
         parse + device alloc + memcpy. Called with model_dir={}",
        model_dir.display()
    );
}

/// Returns true if a tensor name matches the GPTQ-target whitelist that
/// `collect_hessian` should accumulate a Hessian for. Mirrors
/// `scripts/collect_hessian.py::is_gptq_target` so the Tier 1 binary
/// produces a byte-compatible HFHS-v1 output with the Tier 2 Python
/// path.
///
/// Whitelist (suffixes matched against the last `.`-separated segment):
///
///   - Attention input projections: `q_proj`, `k_proj`, `v_proj`,
///     `qkv_proj`
///   - Attention output: `o_proj`, `out_proj`
///   - MLP: `gate_proj`, `up_proj`, `down_proj`, `gate_up_proj`
///   - Linear-attention (Gated DeltaNet):
///     `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`
///   - MoE router: `gate`
#[allow(dead_code)]
pub fn is_gptq_target(name: &str) -> bool {
    const TARGETS: &[&str] = &[
        "q_proj", "k_proj", "v_proj", "qkv_proj",
        "o_proj", "out_proj",
        "gate_proj", "up_proj", "down_proj", "gate_up_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
        "gate",
    ];
    // Strip a trailing `.weight` (HF safetensors stores Linear weights
    // as `<module>.weight`; the GPTQ targets are checked on the module
    // name, not the parameter name).
    let bare = name.strip_suffix(".weight").unwrap_or(name);
    let last = bare.rsplit('.').next().unwrap_or(bare);
    TARGETS.contains(&last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gptq_target_recognizes_canonical_qwen35_names() {
        assert!(is_gptq_target("model.layers.0.self_attn.q_proj.weight"));
        assert!(is_gptq_target("model.layers.0.self_attn.k_proj.weight"));
        assert!(is_gptq_target("model.layers.0.self_attn.v_proj.weight"));
        assert!(is_gptq_target("model.layers.0.self_attn.o_proj.weight"));
        assert!(is_gptq_target("model.layers.0.mlp.gate_proj.weight"));
        assert!(is_gptq_target("model.layers.0.mlp.up_proj.weight"));
        assert!(is_gptq_target("model.layers.0.mlp.down_proj.weight"));
    }

    #[test]
    fn gptq_target_recognizes_moe_router() {
        // Qwen3.5-A3B MoE router lives at `model.layers.N.mlp.gate.weight`
        assert!(is_gptq_target("model.layers.0.mlp.gate.weight"));
    }

    #[test]
    fn gptq_target_rejects_norms_and_embed() {
        assert!(!is_gptq_target("model.embed_tokens.weight"));
        assert!(!is_gptq_target("model.layers.0.input_layernorm.weight"));
        assert!(!is_gptq_target("model.norm.weight"));
        assert!(!is_gptq_target("lm_head.weight"));
    }

    #[test]
    fn gptq_target_recognizes_deltanet_projections() {
        assert!(is_gptq_target("model.layers.0.linear_attn.in_proj_qkv.weight"));
        assert!(is_gptq_target("model.layers.0.linear_attn.in_proj_z.weight"));
        assert!(is_gptq_target("model.layers.0.linear_attn.out_proj.weight"));
    }
}
