// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    clippy::collapsible_if,
    clippy::doc_lazy_continuation,
    clippy::double_ended_iterator_last,
    clippy::if_same_then_else,
    clippy::identity_op,
    clippy::manual_clamp,
    clippy::manual_checked_ops,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::nonminimal_bool,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_map_or,
    clippy::unnecessary_unwrap
)]

//! hipfire-quantize: Quantize raw FP16/BF16/FP32 model weights to Q4_F16 format.
//!
//! Usage: hipfire-quantize --input <model_dir|.gguf|.hfq> --output <output.hfq> --format <FMT> [--chat-template-file template.jinja]
//!
//! Reads weights from any of three sources and produces a `.hfq` (HipFire
//! Quantized) file with RDNA-native quantized weights:
//!   - a HuggingFace model directory (safetensors) or HF model ID
//!   - a single `.gguf` file
//!   - an existing `.hfq` file (e.g. a bf16 `.hfq`) for requantization;
//!     the `.hfq`-source path supports --format bf16/fp16/q8f16/hfq4/hfq6/mq4/mq6/mq3/qtip3/qtip4.

// The quantization modules now live in the crate library (`lib.rs`) so other
// crates (hipfire-diffusion) can reuse the codecs / LDLQ packers / Hessian I/O.
// Re-export them here so this binary's existing bare (`codecs::…`) and
// `crate::…` / `super::…` (test modules) references keep resolving unchanged.
pub use hipfire_quantize::{
    codecs, fixture, gguf_input, gptq, hessian_io, hfhs_diag, ldlq, qtip, roughquant,
};
#[allow(unused_imports)]
use codecs::*;
// Clip-search toggle now lives in the library too.
pub use hipfire_quantize::{mq_clipsearch_enabled, set_mq_clipsearch};
// KVarN codec + deferred KV-compaction live in the leaf `hipfire-kvquant`
// crate (so the engine read path can share them). Re-export at the crate root so
// the existing `crate::kvarn` references across this bin keep resolving unchanged.
pub use hipfire_kvquant::{kv_compact, kvarn};
// Primitive conversions + FWHT live in the leaf `hipfire-primitives` crate; the
// `fixture`/`roughquant` modules are owned by the library (re-exported above).
pub use hipfire_primitives::conv::{
    bf16_bits_to_f32 as bf16_to_f32, f16_to_f32, f32_slice_to_bf16_bytes, f32_slice_to_f16_bytes,
    f32_to_bf16_bits, f32_to_f16, plain_dtype_to_f32 as to_f32,
};
pub use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
// SpinQuant R1 deploy-time merge (--rotate): fold+rotate readers/writers by FᵀM
// before the Oq4 quantize so the codec FWHT leaves the int4 grid in learned M.
// (fixture/roughquant/codecs/qtip/ldlq are owned by the library, re-exported above.)
mod rotate;

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hasher;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use twox_hash::XxHash64;

// imatrix lookup populated once in main() when --imatrix is supplied; keyed by
// ggml-style tensor name (see safetensors_to_ggml_name), value is the
// per-input-channel `Σ_token act²` vector. Consumed by AWQ pre-scaling to
// derive per-channel `RMS_act` for the smoothing-quant scale.
static IMATRIX: OnceLock<HashMap<String, Vec<f32>>> = OnceLock::new();

// `--ldlq`: full-Hessian error-feedback (GPTQ/OBS) weight quant for OQ4++/OQ8++.
// The calibrated arms read each tensor's full [K,K] Hessian from this index and
// run the matching `ldlq::*_pack` routine instead of plain RTN. Opened from the
// `--hessian` path (HFQM or HFHS-v1, carrying full [K,K], not just the diagonal
// AWQ reads). None unless `--ldlq` + `--hessian` are both given.
enum Oq4LdlqHessian {
    Hfqm(hessian_io::HessianSidecar),
    Hfhs(crate::hfhs_diag::HfhsFull),
}

impl Oq4LdlqHessian {
    fn k_of(&self, name: &str) -> Option<usize> {
        match self {
            Self::Hfqm(sc) => sc.get(name, 0).map(|h| h.k),
            Self::Hfhs(sc) => sc.k_of(name),
        }
    }

    fn get_full(&self, name: &str) -> Option<Vec<f32>> {
        match self {
            Self::Hfqm(sc) => sc
                .get(name, 0)
                .map(|h| h.iter_f64().map(|v| v as f32).collect()),
            Self::Hfhs(sc) => sc.get_full(name),
        }
    }
}

static OQ4_LDLQ_HESSIAN: OnceLock<Oq4LdlqHessian> = OnceLock::new();
static LDLQ_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);
static LDLQ_SUCCESS: AtomicUsize = AtomicUsize::new(0);
static LDLQ_MISSING: AtomicUsize = AtomicUsize::new(0);
static LDLQ_K_MISMATCH: AtomicUsize = AtomicUsize::new(0);
static LDLQ_PACK_FAILED: AtomicUsize = AtomicUsize::new(0);

fn ldlq_hessian_for_tensor(idx: &Oq4LdlqHessian, name: &str, k: usize) -> Option<Vec<f32>> {
    LDLQ_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    let key = name.strip_suffix(".weight").unwrap_or(name);
    let hk = idx.k_of(key).or_else(|| idx.k_of(name));
    let Some(hk) = hk else {
        LDLQ_MISSING.fetch_add(1, Ordering::Relaxed);
        eprintln!("  ldlq: skip {name} (no Hessian entry for {key:?})");
        return None;
    };
    if hk != k {
        LDLQ_K_MISMATCH.fetch_add(1, Ordering::Relaxed);
        eprintln!("  ldlq: skip {name} (Hessian K={hk} != weight K={k})");
        return None;
    }
    idx.get_full(key)
        .or_else(|| idx.get_full(name))
        .or_else(|| {
            LDLQ_MISSING.fetch_add(1, Ordering::Relaxed);
            eprintln!("  ldlq: skip {name} (Hessian entry existed but payload could not be read)");
            None
        })
}

fn ldlq_record_success() {
    LDLQ_SUCCESS.fetch_add(1, Ordering::Relaxed);
}

fn ldlq_record_pack_failed(name: &str) {
    LDLQ_PACK_FAILED.fetch_add(1, Ordering::Relaxed);
    eprintln!("  ldlq: skip {name} (OBS packer failed; falling back to non-LDLQ quantization)");
}

fn ldlq_report_and_validate(strict: bool) -> Result<(), String> {
    let attempts = LDLQ_ATTEMPTS.load(Ordering::Relaxed);
    if attempts == 0 {
        if strict {
            return Err(
                "calibrated plus format requested, but no LDLQ-eligible tensors were attempted"
                    .to_string(),
            );
        }
        return Ok(());
    }
    let success = LDLQ_SUCCESS.load(Ordering::Relaxed);
    let missing = LDLQ_MISSING.load(Ordering::Relaxed);
    let k_mismatch = LDLQ_K_MISMATCH.load(Ordering::Relaxed);
    let pack_failed = LDLQ_PACK_FAILED.load(Ordering::Relaxed);
    eprintln!(
        "  LDLQ tensors:     success={success} attempts={attempts} missing={missing} k_mismatch={k_mismatch} pack_failed={pack_failed}"
    );
    if strict && success == 0 {
        return Err(format!(
            "calibrated plus format requested, but LDLQ applied to zero tensors (attempts={attempts}, missing={missing}, k_mismatch={k_mismatch}, pack_failed={pack_failed})"
        ));
    }
    Ok(())
}

// Phase A Stage A — AWQ (Activation-aware Weight Quantization, Lin et al
// 2023). When AWQ_ALPHA is set (via --awq [<alpha>=0.55]), each linear-layer
// weight gets per-input-channel pre-scaling applied BEFORE the standard
// quantize+rotation path:
//
//   s[j] = (rms_act[j])^α   where rms_act[j] = sqrt(imatrix.in_sum2[j] / n_tok)
//
// Then W'[i,j] = W[i,j] * s[j] is what gets quantized + (for MQ4/MFP4) FWHT-
// rotated + packed into the wire format.
//
// At inference, the runtime must apply x / s element-wise BEFORE the rotation
// kernel — the math `(W·s) · (x/s) = W·x` cancels exactly at infinite
// precision. The quantizer writes the `s` vector as a sidecar 1D F16 tensor
// alongside each weight (name = `<weight_name>.awq_scale`); the runtime
// loader reads it and passes to fused_rmsnorm_rotate_mq (or equivalent for
// HFP4/MFP4).
//
// Why per-channel pre-scaling helps where per-block weighted-LS (L5c)
// failed on rotated formats:
//   - L5c weights individual block-level errors by per-channel importance.
//     For FWHT-rotated weights, rotation flattens per-channel importance
//     within blocks (Var[x_rot[i]] = Σ_j Var[x[j]] = const). The lever
//     has nothing to weight.
//   - AWQ applies its scaling in the UNROTATED basis before the FWHT bake-
//     in. The math composes: rot(W·s) is stored, rot(x/s) is computed at
//     inference. Per-channel importance attribution survives the rotation
//     because s is folded into the activation flow.
//   - Egiazarian et al (2509.23202 §3.2) also caution: at small group sizes
//     (g=16 NVFP4, g=32 MXFP4), "outlier mitigation is provably neutralized".
//     This applies to MFP4G32 but NOT to MQ4G256 — AWQ should work on MQ4.
//
// Default alpha = 0.55 (hipfire F2 sweep winner). --awq alone enables
// AWQ at alpha=0.55; --awq <value> sets explicit alpha. Alpha=0 disables;
// alpha=1 is pure activation-magnitude scaling (no smoothing).
static AWQ_ALPHA: OnceLock<f32> = OnceLock::new();

// --sq-split [<frac>]: outlier-aware SmoothQuant. When set, `compute_awq_scales`
// partitions input channels into the top-`frac` by activation energy (outliers)
// and the remaining bulk, and geo-mean-normalizes EACH group SEPARATELY — so the
// bulk's per-channel migration scale isn't skewed by the outliers' huge energy
// (each group's geo-mean = 1 independently). Default frac = 0.01. Unset = the
// original single-group (uniform) normalization across all K channels.
static SQ_OUTLIER_SPLIT: OnceLock<f32> = OnceLock::new();

// --w8-top <frac>: OQ+ magnitude-tiered. The top-`frac` weights per 256-group
// (by |rotated value|) are stored at full int8 (W8A8); the bulk stays int4
// (W4A8). One iu8 grouped-WMMA kernel, one group scale. Default frac = 0.01.
// Consumed by the `oq+t` (OqPlusTiered) format's codec.
static OQPLUS_W8_FRAC: OnceLock<f32> = OnceLock::new();

// Carries the AWQ sidecar scales out of the HFQ-source per-tensor quantizer
// (`quantize_hfq_source_tensor`'s Oq4 arm) up to the tensor-write loop, which
// emits the `<weight>.awq_scale.weight` sidecar. The HFQ-source path returns a
// fixed 4-tuple with many early returns, so a thread-local avoids re-plumbing
// every arm. Set inside the Oq4 arm, consumed (taken) right after the push.
thread_local! {
    static OQ4_AWQ_SIDECAR: std::cell::RefCell<Option<Vec<f32>>> =
        const { std::cell::RefCell::new(None) };
}

// MQ+ clip-search: when set (by an `mqN+` format), MQ codecs use the MSE-optimal
// clip-searched affine range instead of plain min/max. Off by default, so the
// baseline MQ path (and its golden hashes) is unchanged.
// `MQ_CLIPSEARCH` + `mq_clipsearch_enabled`/`set_mq_clipsearch` now live in the
// library (re-exported above) so the codecs and this binary share one toggle.

// SpinQuant R1 deploy merge (`--rotate`): pre-computed fold+rotate f32 overrides
// keyed by tensor name. When set, `tensor_to_f32_with_optional_fp8_scale` returns
// the rotated tensor instead of the raw source, so every codec branch quantizes
// the R1-rotated weights transparently. See `rotate.rs`.
static ROTATION_OVERRIDE: OnceLock<HashMap<String, Vec<f32>>> = OnceLock::new();

// ─── Safetensors Parser ─────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // First 8 bytes: u64 LE header size
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len]).unwrap();

        // Parse header, filtering out __metadata__ key
        let raw: serde_json::Value = serde_json::from_str(header_json).unwrap();
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = raw {
            for (k, v) in map {
                if k == "__metadata__" {
                    continue;
                }
                let meta: TensorMeta = serde_json::from_value(v).unwrap();
                tensors.insert(k, meta);
            }
        }

        Ok(Self {
            _file: file,
            mmap,
            header_size: 8 + header_len,
            tensors,
        })
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        let meta = self.tensors.get(name)?;
        let start = self.header_size + meta.data_offsets[0];
        let end = self.header_size + meta.data_offsets[1];
        Some((meta, &self.mmap[start..end]))
    }

    /// Advise the kernel to drop page cache for a tensor's data region.
    /// On UMA systems this is critical: 234 GB of mmap'd safetensors
    /// pages compete with hipMalloc for the same physical RAM.
    #[cfg(unix)]
    fn drop_tensor_pages(&self, name: &str) {
        if let Some(meta) = self.tensors.get(name) {
            let start = self.header_size + meta.data_offsets[0];
            let len = meta.data_offsets[1] - meta.data_offsets[0];
            use std::os::unix::io::AsRawFd;
            // POSIX_FADV_DONTNEED = 4
            unsafe {
                extern "C" {
                    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
                }
                posix_fadvise(self._file.as_raw_fd(), start as i64, len as i64, 4);
            }
        }
    }

    #[cfg(not(unix))]
    fn drop_tensor_pages(&self, _name: &str) {}

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

// ─── FP16/BF16 Conversion ───────────────────────────────────────────────────

/// Read `--arch-id <u32>` from `std::env::args` if present. Used by
/// both the GGUF and safetensors entry paths to override the
/// auto-detected `arch_id` stamped into the HFQ header.
///
/// Why an override exists: the auto-detection maps every Qwen2 input
/// to `arch_id=1`, which the daemon dispatches through
/// `hipfire-arch-llama`. That loader doesn't read Q/K/V proj bias,
/// so a Qwen2 model loaded by default would produce wrong outputs.
/// Plain Qwen2 should be `arch_id=7` (hipfire-arch-qwen2) and Qwen2-VL
/// family (dots.ocr) should be `arch_id=8` (hipfire-arch-dots-ocr).
/// See docs/architecture-ids.md and docs/plans/
/// dots-ocr-devlog.md §7 (R1).
fn parse_arch_id_override() -> Option<u32> {
    let args: Vec<String> = std::env::args().collect();
    let pos = args.iter().position(|a| a == "--arch-id")?;
    let raw = args.get(pos + 1).unwrap_or_else(|| {
        eprintln!("error: --arch-id requires a u32 value");
        std::process::exit(1);
    });
    match raw.parse::<u32>() {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("error: --arch-id value '{raw}' is not a valid u32: {e}");
            std::process::exit(1);
        }
    }
}

// ─── FP8 E4M3 + UE8M0-scale dequant (DeepSeek V4 Flash) ─────────────────────
//
// DeepSeek V4 ships its quantized weights as paired safetensors entries:
//   <name>.weight  : I8 raw bytes, each byte one FP8 E4M3 value
//   <name>.scale   : F8_E8M0 raw bytes, each byte one UE8M0 exponent
//
// The block shape on DeepSeek V4-shipped checkpoints is [1, 16] (per-row, 16-col
// groups) — i.e. scale shape `[R, C/16]` for weight shape `[R, C]` — even
// though the `quantization_config.weight_block_size` in `config.json`
// reads `[128, 128]`. We verify the implied block from the actual scale
// shape rather than the config to avoid being misled.
//
// E4M3 format (1 sign + 4 exp + 3 mant, bias=7):
//   - exp=0, mant=0      → ±0
//   - exp=0, mant!=0     → denormal: (-1)^s · 2^-6 · (mant/8)
//   - exp=15, mant=7     → NaN (only one NaN code in E4M3)
//   - otherwise normal:  (-1)^s · 2^(exp-7) · (1 + mant/8)
//
// UE8M0 format (8-bit unsigned exponent only, no sign, no mantissa):
//   scale = 2^(byte - 127)
//
// Returns f32 in row-major order matching `weight_shape`.

fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as f32;
    if exp == 0xf && mant == 7.0 {
        // E4M3's single NaN code — treat as 0 for quant purposes (clean
        // bytes flagged elsewhere; downstream MQ-family quant has no
        // NaN handling and would emit garbage).
        return 0.0;
    }
    if exp == 0 {
        if mant == 0.0 {
            return 0.0;
        }
        return sign * (2.0f32.powi(-6)) * (mant / 8.0);
    }
    sign * (2.0f32.powi(exp - 7)) * (1.0 + mant / 8.0)
}

#[inline]
fn ue8m0_to_scale(byte: u8) -> f32 {
    // 2^(exp - 127). Cheap: shift into f32's exponent field directly.
    // byte=127 → 1.0, byte=0 → 2^-127 (subnormal range — fine, we return 0
    // implicitly through f32 rounding), byte=255 → +inf (won't appear on
    // well-formed checkpoints; if it does we propagate inf and the
    // downstream MQ quant will produce extreme outputs detectable in QA).
    2.0f32.powi(byte as i32 - 127)
}

/// Helper for the main quantize loop: convert one tensor's raw bytes to
/// f32, transparently handling DeepSeek V4's FP8 E4M3 + UE8M0-scale pairs.
///
/// If `meta.dtype == "I8"` and a scale sibling is registered in
/// `fp8_scale_for[weight_name]`, dequant the pair. Otherwise fall back
/// to `to_f32(data, dtype)`.
fn tensor_to_f32_with_optional_fp8_scale(
    name: &str,
    raw_data: &[u8],
    meta: &TensorMeta,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
) -> Vec<f32> {
    // SpinQuant R1 deploy merge: if `--rotate` pre-computed a fold+rotated version
    // of this tensor, return it so the codec quantizes the rotated weight. The
    // override is built (from the raw source) before it is installed, so the
    // pre-pass itself reads through this function and sees the raw tensor.
    if let Some(rot) = ROTATION_OVERRIDE.get().and_then(|m| m.get(name)) {
        return rot.clone();
    }
    // FP8 E4M3 + UE8M0 paired storage (DeepSeek V4). The dtype tag is either
    // `I8` (older safetensors writer) or `F8_E4M3` (newer); both
    // store identical E4M3 bytes, so the dequant math is the same.
    if (meta.dtype == "I8" || meta.dtype == "F8_E4M3") && fp8_scale_for.contains_key(name) {
        let (sfi, sname) = &fp8_scale_for[name];
        let (smeta, sbytes) = st_files[*sfi]
            .tensor_data(sname)
            .unwrap_or_else(|| panic!("FP8 scale tensor missing: {sname}"));
        if smeta.dtype == "F8_E8M0" {
            return dequantize_e4m3_ue8m0_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape);
        } else if smeta.dtype == "F32" {
            // MiniMax-M2: e4m3 + F32 block-[128,128] weight_scale_inv (multiply).
            return dequantize_e4m3_f32scale_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape);
        } else {
            panic!(
                "expected F8_E8M0 or F32 scale for {name}, got {}",
                smeta.dtype
            );
        }
    }
    if meta.dtype == "I8" {
        panic!(
            "tensor {name} has dtype I8 but no .scale sibling registered \
                — unexpected on a non-DeepSeek V4 checkpoint."
        );
    }
    to_f32(raw_data, &meta.dtype)
}

/// Read a learned-R1 `.r1` file (`b"HFR1"` + `h: u32 LE` + `h*h` f32 LE), the
/// output of `hipfire-train`'s `learn_r1_dump`. Validates the magic, header size,
/// and that `h` matches the model's hidden size.
fn read_r1_file(path: &str, expect_h: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read --rotate {path}: {e}"))?;
    if bytes.len() < 8 || &bytes[0..4] != b"HFR1" {
        return Err(format!("--rotate {path}: bad magic (expected HFR1)"));
    }
    let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if h != expect_h {
        return Err(format!(
            "--rotate {path}: rotation h={h} != model hidden_size {expect_h}"
        ));
    }
    let want = 8 + h * h * 4;
    if bytes.len() != want {
        return Err(format!(
            "--rotate {path}: size {} != {want} (HFR1 + h + {h}*{h} f32)",
            bytes.len()
        ));
    }
    let mut m = vec![0.0f32; h * h];
    for (i, v) in m.iter_mut().enumerate() {
        let o = 8 + i * 4;
        *v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    }
    Ok(m)
}

/// SpinQuant R1 deploy pre-pass (`--rotate`): fold each RMSNorm scale into its
/// readers and rotate all residual readers/writers/embedding by `R1 = FᵀM`, so the
/// codec's per-256-group FWHT cancels `Fᵀ` and the int4 grid sees the learned `M`.
/// Returns the name→rotated-f32 override map (installed into `ROTATION_OVERRIDE`
/// so every codec branch quantizes the rotated weights) and the synthesized untied
/// `lm_head` f32 (`final_norm` folded, then rotated) that the caller emits — the
/// runtime auto-prefers an explicit `lm_head.weight` over the tied embedding.
///
/// Dense-llama only (arch_id 0/1); the tensor names follow the HF llama layout.
fn build_rotation_overrides(
    rotate_path: &str,
    config: &serde_json::Value,
    n_layers: usize,
    st_files: &[SafetensorsFile],
    fp8_scale_for: &HashMap<String, (usize, String)>,
) -> Result<(HashMap<String, Vec<f32>>, Option<Vec<f32>>), String> {
    let h = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or("config missing hidden_size")? as usize;
    let vocab = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .ok_or("config missing vocab_size")? as usize;
    if h % 256 != 0 {
        return Err(format!(
            "--rotate needs hidden_size {h} %256==0 (codec FWHT)"
        ));
    }
    let m = read_r1_file(rotate_path, h)?;
    let plan = rotate::R1Plan::from_learned_m(&m, h);
    eprintln!(
        "  --rotate: R1=FᵀM built (h={h}); orthonormality {:.1e}",
        plan.orthonormality_error()
    );

    // f32 fetch through the universal extractor. ROTATION_OVERRIDE is not yet set,
    // so this returns the raw source tensor (dequantized as needed).
    let get_f32 = |name: &str| -> Option<Vec<f32>> {
        for st in st_files {
            if let Some((meta, raw)) = st.tensor_data(name) {
                return Some(tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw,
                    meta,
                    fp8_scale_for,
                    st_files,
                ));
            }
        }
        None
    };

    // RMSNorm scales folded into their readers.
    let mut norm1 = Vec::with_capacity(n_layers);
    let mut norm2 = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        norm1.push(
            get_f32(&format!("model.layers.{l}.input_layernorm.weight"))
                .ok_or_else(|| format!("missing input_layernorm for layer {l}"))?,
        );
        norm2.push(
            get_f32(&format!("model.layers.{l}.post_attention_layernorm.weight"))
                .ok_or_else(|| format!("missing post_attention_layernorm for layer {l}"))?,
        );
    }
    let final_norm = get_f32("model.norm.weight").ok_or("missing model.norm.weight")?;

    let mut ov: HashMap<String, Vec<f32>> = HashMap::new();
    for l in 0..n_layers {
        // Attention readers: fold input_layernorm, then reader-rotate on h.
        for suf in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] {
            let name = format!("model.layers.{l}.{suf}.weight");
            if let Some(mut w) = get_f32(&name) {
                let out = w.len() / h;
                rotate::fold_cols(&mut w, &norm1[l], out, h);
                plan.rotate_reader(&mut w, out);
                ov.insert(name, w);
            }
        }
        // MLP readers: fold post_attention_layernorm, then reader-rotate on h.
        for suf in ["mlp.gate_proj", "mlp.up_proj"] {
            let name = format!("model.layers.{l}.{suf}.weight");
            if let Some(mut w) = get_f32(&name) {
                let out = w.len() / h;
                rotate::fold_cols(&mut w, &norm2[l], out, h);
                plan.rotate_reader(&mut w, out);
                ov.insert(name, w);
            }
        }
        // Writers (o_proj, down_proj): writer-rotate on the output h dim.
        for suf in ["self_attn.o_proj", "mlp.down_proj"] {
            let name = format!("model.layers.{l}.{suf}.weight");
            if let Some(mut w) = get_f32(&name) {
                let cols = w.len() / h;
                plan.rotate_writer(&mut w, cols);
                ov.insert(name, w);
            }
        }
        // Folded norms → identity (ones).
        ov.insert(
            format!("model.layers.{l}.input_layernorm.weight"),
            vec![1.0f32; h],
        );
        ov.insert(
            format!("model.layers.{l}.post_attention_layernorm.weight"),
            vec![1.0f32; h],
        );
    }

    // Head: synthesize the untied lm_head (fold final_norm into the ORIGINAL embed,
    // then reader-rotate) BEFORE rotating the input embedding.
    let mut embed = get_f32("model.embed_tokens.weight").ok_or("missing embed_tokens")?;
    let mut lm_head = embed.clone();
    rotate::fold_cols(&mut lm_head, &final_norm, vocab, h);
    plan.rotate_reader(&mut lm_head, vocab);
    // Input embedding: reader-rotate on h (no fold).
    plan.rotate_reader(&mut embed, vocab);
    ov.insert("model.embed_tokens.weight".to_string(), embed);
    ov.insert("model.norm.weight".to_string(), vec![1.0f32; h]);

    eprintln!(
        "  --rotate: {} tensors folded+rotated; synthesized untied lm_head [{vocab}×{h}]",
        ov.len()
    );
    Ok((ov, Some(lm_head)))
}

/// Convert one E2M1 nibble (4-bit FP: 1 sign + 2 exp + 1 mantissa, bias=1) to f32.
///
/// E2M1 codes (signed magnitude on the 3 low bits, high bit is sign):
///   nibble & 0x7 → magnitude  → value
///   0  → 0          → 0.0
///   1  → denorm 0.5 → 0.5
///   2  → normal 1.0 → 1.0
///   3  → normal 1.5 → 1.5
///   4  → normal 2.0 → 2.0
///   5  → normal 3.0 → 3.0
///   6  → normal 4.0 → 4.0
///   7  → normal 6.0 → 6.0
/// Sign bit: bit 3 (0x8).
///
/// Total range: ±6.0. Per OCP MX spec (FP4 E2M1).
#[inline]
fn e2m1_to_f32(nibble: u8) -> f32 {
    // Lookup table for the 8 magnitude codes; sign is applied after.
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let n = (nibble & 0x0f) as usize;
    let mag = MAG[n & 0x7];
    if (n & 0x8) != 0 {
        -mag
    } else {
        mag
    }
}

/// Dequantize a paired E2M1 weight + UE8M0 scale tensor to f32.
///
/// `storage_shape` is the byte-shape from safetensors: [rows, cols_stored]
/// where `cols_stored = logical_cols / 2` (two E2M1 nibbles per byte; low
/// nibble is the even logical column, high nibble is the odd column).
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size in
/// logical-element units is [rows / scale_rows, logical_cols / scale_cols].
/// Per DeepSeek V4 spec (model.py:132-137): block 32 along logical K → scale_cols
/// = logical_cols / 32.
///
/// Returns row-major f32 of LOGICAL shape, length = rows * cols_stored * 2.
fn dequantize_e2m1_ue8m0_to_f32(
    weight_bytes: &[u8],
    storage_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(
        storage_shape.len(),
        2,
        "expected 2D storage shape, got {:?}",
        storage_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale shape, got {:?}",
        scale_shape
    );
    let (rows, cols_stored) = (storage_shape[0], storage_shape[1]);
    let logical_cols = cols_stored * 2;
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols_stored,
        "FP4 weight byte count mismatch"
    );
    assert_eq!(scale_bytes.len(), sr * sc, "FP4 scale byte count mismatch");
    assert!(
        rows % sr == 0 && logical_cols % sc == 0,
        "FP4 scale shape {:?} doesn't tile logical weight shape [{}, {}]",
        scale_shape,
        rows,
        logical_cols
    );
    let block_rows = rows / sr;
    let block_cols_logical = logical_cols / sc;

    let mut out = vec![0.0f32; rows * logical_cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols_logical {
                    let c = sc_j * block_cols_logical + dj;
                    // c is the LOGICAL column. Byte storing it sits at
                    // (c / 2); low nibble for even c, high nibble for odd.
                    let byte = weight_bytes[r * cols_stored + (c / 2)];
                    let nibble = if (c & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                    out[r * logical_cols + c] = e2m1_to_f32(nibble) * scale;
                }
            }
        }
    }
    (out, vec![rows, logical_cols])
}

/// Dequantize a paired E4M3 weight + UE8M0 scale tensor to f32.
///
/// `weight_shape` is the LOGICAL [rows, cols] of the weight matrix.
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size is
/// [weight_rows / scale_rows, weight_cols / scale_cols].
///
/// Returns row-major f32, length = rows * cols.
fn dequantize_e4m3_ue8m0_to_f32(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Vec<f32> {
    assert_eq!(
        weight_shape.len(),
        2,
        "expected 2D weight, got {:?}",
        weight_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale,  got {:?}",
        scale_shape
    );
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols,
        "weight byte count mismatch"
    );
    assert_eq!(scale_bytes.len(), sr * sc, "scale  byte count mismatch");
    assert!(
        rows % sr == 0 && cols % sc == 0,
        "scale shape {:?} doesn't tile weight shape {:?}",
        scale_shape,
        weight_shape
    );
    let block_rows = rows / sr;
    let block_cols = cols / sc;

    let mut out = vec![0.0f32; rows * cols];
    // Each (sr_i, sc_j) scale governs the block weight[sr_i*block_rows .. (sr_i+1)*block_rows,
    //                                                  sc_j*block_cols .. (sc_j+1)*block_cols].
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    let b = weight_bytes[r * cols + c];
                    out[r * cols + c] = e4m3_to_f32(b) * scale;
                }
            }
        }
    }
    out
}

/// Dequantize FP8 E4M3 weights paired with an F32 block-[128,128]
/// `weight_scale_inv` (MiniMax-M2 / DeepSeek-V3 fp8 block quant). Dequant is
/// MULTIPLY: `out = e4m3_to_f32(b) * scale` (the stored scale ≈ amax/448 per
/// block, verified ~5e-4 on the real checkpoint). Scale tile is [rows/sr, cols/sc]
/// = [128, 128] on MiniMax.
fn dequantize_e4m3_f32scale_to_f32(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Vec<f32> {
    assert_eq!(
        weight_shape.len(),
        2,
        "expected 2D weight, got {:?}",
        weight_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale, got {:?}",
        scale_shape
    );
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols,
        "weight byte count mismatch"
    );
    assert_eq!(
        scale_bytes.len(),
        sr * sc * 4,
        "f32 scale byte count mismatch"
    );
    assert!(
        rows % sr == 0 && cols % sc == 0,
        "scale shape {:?} doesn't tile weight shape {:?}",
        scale_shape,
        weight_shape
    );
    let block_rows = rows / sr;
    let block_cols = cols / sc;
    let mut out = vec![0.0f32; rows * cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let so = (sr_i * sc + sc_j) * 4;
            let scale = f32::from_le_bytes([
                scale_bytes[so],
                scale_bytes[so + 1],
                scale_bytes[so + 2],
                scale_bytes[so + 3],
            ]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    out[r * cols + c] = e4m3_to_f32(weight_bytes[r * cols + c]) * scale;
                }
            }
        }
    }
    out
}

// ─── Q4_F16_G64 Quantization ────────────────────────────────────────────────

/// Quantize F32 weights to HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights (0.531 B/w).
/// 18 VGPRs, 100% occupancy on RDNA1. Beats Q4_K at all matrix sizes.
/// CPU-side FWHT (Walsh-Hadamard Transform) on a 256-element group.
/// Matches the GPU-side fwht_forward_256 in turbo_common: signs1 → butterfly → scale → signs2.
/// Simulated QTIP quantization of a 2D weight (row-major `m × k`), in place,
/// bit-rate-parametric (Phase C: 2-bit primary, 3-bit fallback). Per row, per
/// 256-group: FWHT-rotate (incoherence) → beam-encode trellis → optimal scale →
/// decode → inverse-rotate (FWHT is orthogonal, so the inverse is `cpu_fwht_256`
/// with the sign args swapped). `bits` selects the per-step symbol count; the
/// codebook is bit-rate-independent (state→value map), so the same `cb` serves
/// both 2- and 3-bit. The result is the *effective* weight a fused QTIP kernel
/// would compute, so running it through the normal bf16 forward gives a faithful
/// QTIP PPL without the GPU kernel. The `k%256` tail (if any) is left
/// unquantized. Parallel over rows.
fn qtip_simquant_nbit(
    f32_data: &mut [f32],
    k: usize,
    cb: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    bits: u32,
) {
    use rayon::prelude::*;
    let groups = k / 256;
    if groups == 0 {
        return;
    }
    f32_data.par_chunks_mut(k).for_each(|row| {
        for g in 0..groups {
            let seg = &mut row[g * 256..g * 256 + 256];
            let mut grp = [0.0f32; 256];
            grp.copy_from_slice(seg);
            cpu_fwht_256(&mut grp, signs1, signs2); // rotate
            let scale0 = qtip::group_scale(&grp);
            let sym = qtip::beam_encode_group_bits(&grp, scale0, cb, 128, bits);
            let scale = qtip::optimal_scale_bits(&grp, &sym, cb, bits);
            let mut deq = qtip::decode_group_bits(&sym, scale, cb, bits);
            cpu_fwht_256(&mut deq, signs2, signs1); // inverse rotate (swap signs)
            seg.copy_from_slice(&deq);
        }
    });
}

/// QTIP-trellis the bulk after RoughQuant PCA rotation, protecting the leading
/// `n_prot` columns (PCA-sorted by eigenvalue descending → highest-energy first)
/// at full precision with PER-COLUMN granularity. The protected columns are
// ── BBT-spectral (SpectralLLM) influence scaling ───────────────────────────
// Reference: "Influence-Inspired Spectral Rotations for Extreme Low-Bit LLM
// Quantization" (Pavlov). A math-invariant per-input-channel scale derived from
// per-coordinate spectral (Walsh-Hadamard) activation energy:
//   h = H x,  Inf_l = E[h_l^2]/Σ_k E[h_k^2],  s_l = (d·Inf_l + 1)^alpha.
// E[h_l^2] = (H C H^T)_ll where C = E[x x^T] is the calib Hessian. Applied as
// W' = W·diag(s), x' = x/s (exactly invariant), biasing the quant budget toward
// high-spectral-energy channels. We probe it on the QTIP sim (the W2/W3 regime
// where BBT pays off). This is the input-channel-basis variant of the paper's
// scaling (scale folds into the activation), composed with QTIP's own per-group
// FWHT — documented as a probe, not the paper's full WHT-rotated recipe.

/// In-place plain normalized FWHT (length = power of two), 1/sqrt(n) scaled.
fn fwht_norm_pow2(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Per-coordinate spectral activation energy diag(H C H^T) from a k×k Hessian
/// `c` (row-major), full-dim Hadamard on k padded to the next power of two.
/// Returns the first `k` diagonal entries (the active channels).
fn bbt_spectral_diag(c: &[f32], k: usize) -> Vec<f32> {
    let kp = k.next_power_of_two();
    // Y = C H^T : FWHT each row (zero-padded to kp).
    let mut y = vec![0.0f32; kp * kp];
    let mut row = vec![0.0f32; kp];
    for i in 0..k {
        row[..k].copy_from_slice(&c[i * k..i * k + k]);
        for r in row.iter_mut().take(kp).skip(k) {
            *r = 0.0;
        }
        fwht_norm_pow2(&mut row);
        y[i * kp..i * kp + kp].copy_from_slice(&row);
    }
    // Z = H Y : FWHT each column; keep only diag(Z)[0..k].
    let mut col = vec![0.0f32; kp];
    let mut diag = vec![0.0f32; k];
    for l in 0..k {
        for i in 0..kp {
            col[i] = if i < k { y[i * kp + l] } else { 0.0 };
        }
        fwht_norm_pow2(&mut col);
        diag[l] = col[l]; // (H Y)_{l,l}
    }
    diag
}

/// BBT per-channel scales s_l = (k·Inf_l + 1)^alpha from the spectral diag.
fn bbt_scales(spec_diag: &[f32], k: usize, alpha: f32) -> Vec<f32> {
    let total: f64 = spec_diag
        .iter()
        .map(|&v| v.max(0.0) as f64)
        .sum::<f64>()
        .max(1e-12);
    spec_diag
        .iter()
        .map(|&v| {
            let inf = (v.max(0.0) as f64) / total; // in [0,1], sums to 1
            ((k as f64 * inf + 1.0).powf(alpha as f64) as f32).max(1e-2)
        })
        .collect::<Vec<f32>>()
        .into_iter()
        .take(k)
        .collect()
}

/// Rank-`r` approximation of the m×k matrix `e` (row-major) via a randomized
/// range finder: Ω∼N(0,1) [k×r], Y=EΩ, orthonormalize Y→Q (modified
/// Gram-Schmidt), B=QᵀE, return E_r = QB. One pass (no power iteration) — enough
/// for a quant-error CORRECTION term (we want the dominant directions, not exact
/// singular values). This is the LQER/CALDERA low-rank residual: emit
/// dequant(W4)+E_r so a W4·A8 GEMM + a 2-WMMA UᵥVᵥ correction recovers most of
/// the W4→high-bit gap. Returns zeros for r=0, the full matrix for r≥min(m,k).
/// Randomized rank-`r` range-finder factorization of the error matrix `e`
/// [m×k]: returns `(U, V)` with `U` [m×r] (orthonormal columns) and `V` [r×k]
/// such that `U·V ≈ e`. This is the factored form the runtime low-rank
/// correction consumes directly (`out += U·(V·x)`); [`randomized_lowrank`] is
/// the same computation followed by the `U·V` reconstruction.
fn randomized_lowrank_factors(e: &[f32], m: usize, k: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    if r == 0 {
        return (vec![0.0f32; m * r.max(1)], vec![0.0f32; r * k]);
    }
    let r = r.min(m).min(k);
    // Ω [k×r] — xorshift Gaussian-ish (uniform is fine for a range finder).
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut omega = vec![0.0f32; k * r];
    for v in omega.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0;
    }
    // Y = E·Ω  [m×r] (parallel over rows).
    let mut y = vec![0.0f32; m * r];
    y.par_chunks_mut(r).enumerate().for_each(|(i, yr)| {
        let erow = &e[i * k..i * k + k];
        for (c, yc) in yr.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += erow[j] * omega[j * r + c];
            }
            *yc = acc;
        }
    });
    // Orthonormalize columns of Y → Q (modified Gram-Schmidt; r small).
    for c in 0..r {
        for p in 0..c {
            let mut dot = 0.0f32;
            for i in 0..m {
                dot += y[i * r + c] * y[i * r + p];
            }
            for i in 0..m {
                y[i * r + c] -= dot * y[i * r + p];
            }
        }
        let mut nrm = 0.0f32;
        for i in 0..m {
            nrm += y[i * r + c] * y[i * r + c];
        }
        nrm = nrm.sqrt().max(1e-12);
        for i in 0..m {
            y[i * r + c] /= nrm;
        }
    }
    // B = Qᵀ·E  [r×k] (parallel over the r rows of B).
    let mut b = vec![0.0f32; r * k];
    b.par_chunks_mut(k).enumerate().for_each(|(c, brow)| {
        for (j, bj) in brow.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for l in 0..m {
                acc += y[l * r + c] * e[l * k + j];
            }
            *bj = acc;
        }
    });
    (y, b)
}

fn randomized_lowrank(e: &[f32], m: usize, k: usize, r: usize) -> Vec<f32> {
    use rayon::prelude::*;
    if r == 0 {
        return vec![0.0f32; m * k];
    }
    if r >= m.min(k) {
        return e.to_vec();
    }
    // Same range-finder as the runtime path, then reconstruct E_r = U·V [m×k].
    let (u, v) = randomized_lowrank_factors(e, m, k, r);
    let r = r.min(m).min(k);
    let mut er = vec![0.0f32; m * k];
    er.par_chunks_mut(k).enumerate().for_each(|(i, erow)| {
        for (j, ej) in erow.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for c in 0..r {
                acc += u[i * r + c] * v[c * k + j];
            }
            *ej = acc;
        }
    });
    er
}

/// overwritten exactly after the per-256-group FWHT+trellis pass. Parallel over
/// rows.
///
/// Correctness: in the rotated frame y = W̃·x̃ = Σ_j W̃[:,j]·x̃[j]. After this
/// pass the protected columns carry exact W̃ values and the bulk carries the
/// trellis reconstruction, so the protected subspace contributes zero error and
/// only the bulk is quantized — exactly the RoughQuant split.
fn qtip_simquant_protected(
    f32_data: &mut [f32],
    k: usize,
    n_prot: usize,
    cb: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    bits: u32,
) {
    use rayon::prelude::*;
    let groups = k / 256;
    if groups == 0 {
        return;
    }
    let n_prot = n_prot.min(k);
    f32_data.par_chunks_mut(k).for_each(|row| {
        let saved: Vec<f32> = row[..n_prot].to_vec();
        // Quantize the FULL row, then overwrite the leading n_prot (protected)
        // columns exact — guarantees protect ≤ no-protect. (Earlier zero-before-
        // quant was non-monotonic and could worsen PPL; removed.)
        for g in 0..groups {
            let seg = &mut row[g * 256..g * 256 + 256];
            let mut grp = [0.0f32; 256];
            grp.copy_from_slice(seg);
            cpu_fwht_256(&mut grp, signs1, signs2);
            let scale0 = qtip::group_scale(&grp);
            let sym = qtip::beam_encode_group_bits(&grp, scale0, cb, 128, bits);
            let scale = qtip::optimal_scale_bits(&grp, &sym, cb, bits);
            let mut deq = qtip::decode_group_bits(&sym, scale, cb, bits);
            cpu_fwht_256(&mut deq, signs2, signs1);
            seg.copy_from_slice(&deq);
        }
        row[..n_prot].copy_from_slice(&saved);
    });
}

/// Reorder the columns of a row-major `m×k` matrix: output column `j` takes input
/// column `perm[j]`. Used by roughquant3 to cluster salient input channels into
/// contiguous leading columns before protect+QTIP. Parallel over rows.
fn permute_cols(w: &[f32], m: usize, k: usize, perm: &[usize]) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(perm.len(), k);
    let mut out = vec![0.0f32; m * k];
    out.par_chunks_mut(k)
        .zip(w.par_chunks(k))
        .for_each(|(orow, irow)| {
            for j in 0..k {
                orow[j] = irow[perm[j]];
            }
        });
    out
}

/// Inverse of `permute_cols`: place permuted column `j` back at original column
/// `perm[j]` (`out[:, perm[j]] = wperm[:, j]`). Parallel over rows.
fn unpermute_cols(wperm: &[f32], m: usize, k: usize, perm: &[usize]) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(wperm.len(), m * k);
    debug_assert_eq!(perm.len(), k);
    let mut out = vec![0.0f32; m * k];
    out.par_chunks_mut(k)
        .zip(wperm.par_chunks(k))
        .for_each(|(orow, irow)| {
            for j in 0..k {
                orow[perm[j]] = irow[j];
            }
        });
    out
}

/// Permute ROWS of a row-major `m × k` matrix (gather convention, mirroring
/// `permute_cols`): `out[j, :] = w[perm[j], :]`. Used by the #5 residual
/// permutation to reorder residual-WRITER output rows. Parallel over output rows.
fn permute_rows(w: &[f32], m: usize, k: usize, perm: &[usize]) -> Vec<f32> {
    use rayon::prelude::*;
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(perm.len(), m);
    let mut out = vec![0.0f32; m * k];
    out.par_chunks_mut(k).enumerate().for_each(|(j, orow)| {
        let src = perm[j] * k;
        orow.copy_from_slice(&w[src..src + k]);
    });
    out
}

/// RoughQuant Phase-2d (channel-consistent) QTIP sim: protect arbitrary input
/// COLUMNS (`protected_cols`) and entire output ROWS (`protected_rows`) at full
/// precision, QTIP-trellis the rest. Row-major `m × k`.
///
/// This realizes "energy flows down high-resolution channels" on BOTH sides of
/// the residual stream: a high-energy residual channel is kept exact where it is
/// READ (a reader weight's column) AND where it is WRITTEN (a writer weight's
/// row). Protected rows are left entirely untouched; for other rows the protected
/// columns are overwritten exactly after the per-256-group FWHT+trellis pass.
/// Parallel over rows.
fn qtip_simquant_masked(
    wf: &mut [f32],
    m: usize,
    k: usize,
    protected_cols: &[usize],
    protected_rows: &[bool],
    cb: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    bits: u32,
) {
    use rayon::prelude::*;
    debug_assert_eq!(wf.len(), m * k);
    debug_assert_eq!(protected_rows.len(), m);
    let groups = k / 256;
    if groups == 0 {
        return;
    }
    wf.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
        if protected_rows[r] {
            return; // high-energy residual output channel — keep exact
        }
        let saved: Vec<(usize, f32)> = protected_cols.iter().map(|&c| (c, row[c])).collect();
        // Quantize the FULL group, then overwrite protected positions exact
        // (below) — guarantees protect ≤ no-protect. (Non-monotonic zero-before-
        // quant removed.)
        for g in 0..groups {
            let seg = &mut row[g * 256..g * 256 + 256];
            let mut grp = [0.0f32; 256];
            grp.copy_from_slice(seg);
            cpu_fwht_256(&mut grp, signs1, signs2);
            let scale0 = qtip::group_scale(&grp);
            let sym = qtip::beam_encode_group_bits(&grp, scale0, cb, 128, bits);
            let scale = qtip::optimal_scale_bits(&grp, &sym, cb, bits);
            let mut deq = qtip::decode_group_bits(&sym, scale, cb, bits);
            cpu_fwht_256(&mut deq, signs2, signs1);
            seg.copy_from_slice(&deq);
        }
        for (c, v) in saved {
            row[c] = v;
        }
    });
}

/// Like `qtip_simquant_masked` but the bulk codec is **mq4** (MQ4G256:
/// per-256-group FWHT → asymmetric 4-bit → dequant → inverse FWHT), matching the
/// real production format exactly (`quantize_mq4g256`). Used to measure the
/// marginal value of channel protection ON TOP of mq4 (`mq4+protect` vs `mq4`) —
/// the fair, iso-format comparison: protected columns/rows kept exact, the rest
/// quantized exactly as mq4 would. Parallel over rows.
fn mq4_simquant_masked(
    wf: &mut [f32],
    m: usize,
    k: usize,
    protected_cols: &[usize],
    protected_rows: &[bool],
    signs1: &[f32],
    signs2: &[f32],
    bits: u32,
) {
    use rayon::prelude::*;
    debug_assert_eq!(wf.len(), m * k);
    debug_assert_eq!(protected_rows.len(), m);
    let levels = ((1u32 << bits) - 1) as f32; // 4-bit→15, 5-bit→31, 6-bit→63
    let groups = k / 256;
    if groups == 0 {
        return;
    }
    wf.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
        if protected_rows[r] {
            return;
        }
        let saved: Vec<(usize, f32)> = protected_cols.iter().map(|&c| (c, row[c])).collect();
        // Quantize the FULL group exactly as mq4 does, then overwrite the
        // protected positions with their exact values (below). This guarantees
        // mq4+protect ≤ mq4 — protection only removes error, never adds it.
        // (An earlier "zero protected before FWHT to tighten the bulk range" was
        // NOT monotonic and could WORSEN PPL; removed.)
        for g in 0..groups {
            let seg = &mut row[g * 256..g * 256 + 256];
            let mut grp = [0.0f32; 256];
            grp.copy_from_slice(seg);
            cpu_fwht_256(&mut grp, signs1, signs2);
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &v in grp.iter() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            let range = mx - mn;
            let scale = if range > 0.0 { range / levels } else { 1.0 };
            let inv = if range > 0.0 { 1.0 / scale } else { 0.0 };
            let qmax = levels as u32;
            for v in grp.iter_mut() {
                let q = (((*v - mn) * inv + 0.5) as u32).min(qmax);
                *v = q as f32 * scale + mn;
            }
            cpu_fwht_256(&mut grp, signs2, signs1);
            seg.copy_from_slice(&grp);
        }
        for (c, v) in saved {
            row[c] = v;
        }
    });
}

/// Per-input-column sum-of-squares ‖W[:,c]‖² from a row-major bf16 weight blob
/// (k = input dim). Used for weight-aware saliency metrics (`wnorm`, `product`):
/// the output-error contribution of channel c scales with ‖W[:,c]‖²·E[x_c²], so
/// pure activation energy diag(H) alone under/over-protects depending on weight
/// magnitude (CMPQ: select by quant-error impact, not raw activation).
fn bf16_colnorm2(data: &[u8], k: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; k];
    for (i, c) in data.chunks_exact(2).enumerate() {
        let v = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
        s[i % k] += v * v;
    }
    s
}

/// OBS/GPTQ compensation-aware per-column saliency: `‖W[:,c]‖² / [H⁻¹]_cc`, where
/// `[H⁻¹]_cc` is the diagonal of `(H+λI)⁻¹`. Small `[H⁻¹]_cc` = a "stiff" channel
/// other channels cannot compensate for → high importance. This captures the
/// cross-channel correlation that `diag(H)=H_cc` misses (the hypothesized
/// improvement for the shallow TAIL of the importance ranking). `h_full` is the
/// row-major k×k Hessian; reuses the LDLQ Cholesky. None on breakdown.
fn obs_col_saliency(
    h_full: &[f32],
    colnorm2: &[f32],
    k: usize,
    damp_frac: f64,
) -> Option<Vec<f64>> {
    // damp scaled to the Hessian's diagonal mean, matching the LDLQ convention.
    let mut diag_sum = 0.0f64;
    for i in 0..k {
        diag_sum += h_full[i * k + i] as f64;
    }
    let damp = damp_frac * (diag_sum / k as f64).max(1e-12);
    let l = ldlq::inv_cholesky_lower(h_full, k, damp)?; // L Lᵀ = (H+λI)⁻¹
    let mut sal = vec![0.0f64; k];
    for c in 0..k {
        // [H⁻¹]_cc = (L Lᵀ)_cc = Σ_{j≤c} L[c,j]²  (L lower-triangular)
        let mut hinv_cc = 0.0f64;
        for j in 0..=c {
            let v = l[(c, j)];
            hinv_cc += v * v;
        }
        sal[c] = colnorm2[c] as f64 / (hinv_cc + 1e-12);
    }
    Some(sal)
}

/// RoughQuant Phase-1 sim (no rotation): protect the most-salient input columns
/// of a 2D weight (row-major `m × k`) at full precision, crush the rest to a
/// `bulk_bits` symmetric-uniform grid (per row, per `group` columns), in place.
///
/// `saliency[c]` ranks input column `c` (higher = protect). With a Hessian it is
/// `diag(H) = E[x_c²]` (output-aware: quant noise on a high-energy input channel
/// costs more output error — CMPQ's "quant-error impact"); without one it is the
/// column L2 norm of W. The top `protect_frac · k` columns are left untouched;
/// the bulk is absmax-quantized over the non-protected entries of each group so
/// the protected outliers don't inflate the bulk scale. Running the perturbed
/// weight through the normal bf16 forward gives a faithful RoughQuant PPL with no
/// GPU kernel. Parallel over rows.
fn roughquant_sim_tensor(
    wf: &mut [f32],
    m: usize,
    k: usize,
    saliency: &[f32],
    protect_frac: f64,
    bulk_bits: u32,
    group: usize,
) {
    use rayon::prelude::*;
    debug_assert_eq!(wf.len(), m * k);
    debug_assert_eq!(saliency.len(), k);
    let n_prot = ((protect_frac * k as f64).round() as usize).min(k);
    // Rank columns by saliency desc; mark the top n_prot as protected.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by(|&a, &b| {
        saliency[b]
            .partial_cmp(&saliency[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut protected = vec![false; k];
    for &c in &order[..n_prot] {
        protected[c] = true;
    }
    // Symmetric-uniform levels for `bulk_bits`: q ∈ {-(2^{b-1}-1) .. 2^{b-1}-1}.
    let qmax = (((1u32 << bulk_bits) >> 1) as f32 - 1.0).max(1.0);
    wf.par_chunks_mut(k).for_each(|row| {
        let mut g = 0usize;
        while g < k {
            let end = (g + group).min(k);
            let mut amax = 0.0f32;
            for c in g..end {
                if !protected[c] {
                    amax = amax.max(row[c].abs());
                }
            }
            if amax > 0.0 {
                let scale = amax / qmax;
                let inv = 1.0 / scale;
                for c in g..end {
                    if !protected[c] {
                        let qi = (row[c] * inv).round().clamp(-qmax, qmax);
                        row[c] = qi * scale;
                    }
                }
            }
            g = end;
        }
    });
}

fn roughquant4_is_residual_reader(name: &str) -> bool {
    let name = name.strip_suffix(".weight").unwrap_or(name);
    name.contains(".linear_attn.in_proj_")
        || name.ends_with(".mlp.gate_proj")
        || name.ends_with(".mlp.up_proj")
        || name.ends_with(".self_attn.q_proj")
        || name.ends_with(".self_attn.k_proj")
        || name.ends_with(".self_attn.v_proj")
}

fn roughquant4_is_residual_writer(name: &str) -> bool {
    let name = name.strip_suffix(".weight").unwrap_or(name);
    name.ends_with(".self_attn.o_proj")
        || name.ends_with(".linear_attn.out_proj")
        || name.ends_with(".mlp.down_proj")
}

fn roughquant4_infer_dmodel(tensors: &[HfqTensor]) -> Option<usize> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for t in tensors {
        if matches!(t.quant_type, QuantType::BF16)
            && t.shape.len() == 2
            && (t.shape[1] as usize) % 256 == 0
            && !t.name.contains("embed")
            && !t.name.contains("lm_head")
            && roughquant4_is_residual_reader(&t.name)
        {
            *counts.entry(t.shape[1] as usize).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(dmodel, count)| (count, dmodel))
        .map(|(dmodel, _)| dmodel)
}

#[cfg(test)]
mod awq_tests {
    use super::*;

    /// Verify geometric mean of computed AWQ scales is ~1.0 — the
    /// normalization in compute_awq_scales should center the scale
    /// vector so downstream min-max quantization isn't perturbed.
    #[test]
    fn awq_scales_geomean_is_one() {
        // Realistic-ish imatrix: log-normal-ish per-channel statistics
        let in_sum2: Vec<f32> = (0..256)
            .map(|j| (1.0 + 10.0 * (j as f32 / 256.0)).exp()) // 1.0 → e^11
            .collect();
        for &alpha in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let s = compute_awq_scales(&in_sum2, alpha);
            assert_eq!(s.len(), in_sum2.len());
            // Geometric mean = exp(mean(log(s)))
            let log_mean = s.iter().map(|&v| (v as f64).ln()).sum::<f64>() / s.len() as f64;
            let geo_mean = log_mean.exp();
            assert!(
                (geo_mean - 1.0).abs() < 1e-4,
                "alpha={alpha}: geo_mean={geo_mean} (want 1.0)"
            );
        }
    }

    /// Alpha = 0 should produce all-ones scales (AWQ disabled at layer level).
    #[test]
    fn awq_scales_alpha_zero_is_identity() {
        let in_sum2: Vec<f32> = (1..=128).map(|j| j as f32).collect();
        let s = compute_awq_scales(&in_sum2, 0.0);
        for &v in &s {
            assert!((v - 1.0).abs() < 1e-5, "alpha=0 scale {v} should be 1.0");
        }
    }

    /// Larger imatrix values should produce larger scales for alpha > 0.
    /// Monotonicity check.
    #[test]
    fn awq_scales_monotonic_in_imatrix() {
        let in_sum2 = vec![1.0_f32, 4.0, 16.0, 64.0, 256.0];
        let s = compute_awq_scales(&in_sum2, 0.5);
        for w in s.windows(2) {
            assert!(w[1] > w[0], "scales not monotonic: {} -> {}", w[0], w[1]);
        }
    }

    /// AWQ math identity: `(W · diag(s)) · (x / s) == W · x` at infinite
    /// precision. With fp32 weights + fp32 activations, error should be
    /// at floating-point rounding precision (~1e-5 relative).
    #[test]
    fn awq_math_identity_holds() {
        // Tiny test: 4 output × 8 input matmul
        let m = 4;
        let k = 8;
        // Random-ish weights and activations
        let w: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|j| (j as f32 + 1.0) * 0.5).collect();

        // Reference: y = W * x
        let mut y_ref = vec![0.0_f32; m];
        for i in 0..m {
            for j in 0..k {
                y_ref[i] += w[i * k + j] * x[j];
            }
        }

        // AWQ-scaled: pre-scale W, pre-divide x
        let in_sum2: Vec<f32> = (1..=k).map(|j| j as f32 * 10.0).collect();
        let s = compute_awq_scales(&in_sum2, 0.5);
        let mut w_scaled = w.clone();
        awq_pre_scale_weights(&mut w_scaled, m, k, &s);
        let x_div: Vec<f32> = x.iter().zip(&s).map(|(&xv, &sv)| xv / sv).collect();

        // y' = (W * diag(s)) * (x / s)
        let mut y_awq = vec![0.0_f32; m];
        for i in 0..m {
            for j in 0..k {
                y_awq[i] += w_scaled[i * k + j] * x_div[j];
            }
        }

        // Compare
        for i in 0..m {
            let rel = (y_awq[i] - y_ref[i]).abs() / y_ref[i].abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "row {i}: AWQ y={} ref y={} rel_err={}",
                y_awq[i],
                y_ref[i],
                rel
            );
        }
    }

    /// Edge case: zero imatrix entries should produce finite scales
    /// (clamped via 1e-12 floor in compute_awq_scales).
    #[test]
    fn awq_handles_zero_imatrix() {
        let in_sum2 = vec![0.0_f32, 1.0, 4.0, 0.0];
        let s = compute_awq_scales(&in_sum2, 0.5);
        for &v in &s {
            assert!(
                v.is_finite() && v > 0.0,
                "scale {v} should be finite + positive"
            );
        }
    }
}

#[cfg(test)]
mod hfp4_tests {
    use super::*;

    #[test]
    fn e2m1_round_matches_lattice() {
        // Each lattice value should round to its own code.
        for (i, &val) in E2M1_LUT.iter().enumerate() {
            let nibble = e2m1_round(val);
            // +0 and -0 are both at value 0.0; either nibble is acceptable.
            if val.abs() < 1e-6 {
                assert!(
                    nibble == 0 || nibble == 8,
                    "zero rounds to nibble {}",
                    nibble
                );
            } else {
                assert_eq!(
                    nibble, i as u8,
                    "code {} rounded to nibble {} not {}",
                    i, nibble, i
                );
            }
        }
    }

    #[test]
    fn e2m1_round_midpoint() {
        // Halfway between +1.0 and +1.5 → either is acceptable (tie).
        let n = e2m1_round(1.25);
        assert!(n == 2 || n == 3, "midpoint rounded to {}", n);
        // Halfway between +4.0 and +6.0 (= 5.0) → either is acceptable.
        let n = e2m1_round(5.0);
        assert!(n == 6 || n == 7, "5.0 rounded to {}", n);
    }

    #[test]
    fn round_trip_constant_row() {
        // All-1.0 row: row_scale_a = 1/6, every block_e ≈ 127 + log2(1) = 127, every nibble = 2 (=1.0).
        let row = vec![1.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        for (i, &v) in recovered.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-2, "elem {} recovered to {}", i, v);
        }
    }

    #[test]
    fn round_trip_mixed_magnitudes() {
        // Row with mixed positive/negative E2M1 magnitudes — should round-trip exactly.
        let row: Vec<f32> = (0..64)
            .map(|i| {
                let v = E2M1_LUT[i % 16];
                v * 6.0 // scale up so row_scale_a sees max abs at 6 * 6 = 36, brings code lattice back to [-6, 6]
            })
            .collect();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        // Bound: |recovered - input| ≤ row_scale * 2^(block_e - 127) * 0.5 (half min E2M1 step).
        // With row_scale_a = 36/6 = 6, and block_max_normalized = 6, block_e = 127 → step ≈ 0.5 → tol = 3.0.
        // Actual tolerance should be much tighter for exact lattice values; allow some headroom.
        for (i, (&got, &want)) in recovered.iter().zip(row.iter()).enumerate() {
            let rel_err = (got - want).abs() / want.abs().max(1.0);
            assert!(
                rel_err < 0.1,
                "elem {}: got {} want {} rel_err {}",
                i,
                got,
                want,
                rel_err
            );
        }
    }

    #[test]
    fn round_trip_per_block_error_bound() {
        // Mathematical guarantee: for every element, |recovered - original| must be ≤
        //   row_scale_a * 2^(block_e - 127) * (max_E2M1_step / 2)
        // = effective_block_scale * 1.0  (max E2M1 step is 2.0, half = 1.0)
        //
        // This is the format's correctness contract; if this fails we have a real bug.
        // NRMSE quality on raw weights is a downstream concern (MXFP4 family is documented
        // as needing rotation+smoothing for production accuracy — that's MFP4G32 in v1.5).
        let mut rng_state: u64 = 0xdead_beef_dead_beef;
        let mut next_uniform = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32).max(1e-7)
        };
        // Box-Muller Gaussian std=0.5.
        let row: Vec<f32> = (0..512)
            .flat_map(|_| {
                let u1 = next_uniform();
                let u2 = next_uniform();
                let r = (-2.0 * u1.ln()).sqrt();
                let t = 2.0 * std::f32::consts::PI * u2;
                [r * t.cos() * 0.5, r * t.sin() * 0.5]
            })
            .collect();

        let k = row.len();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, k);

        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));

        // Per-block half-max-step bound. Allow 1% slack for FP16 row-scale rounding.
        for b in 0..(k / 32) {
            let payload_off = 16 + b * 17;
            let block_e = packed[payload_off] as i32;
            let block_scale = ((block_e - 127) as f32).exp2();
            // Max E2M1 step is 2.0 (between 4 and 6); half = 1.0. Round-trip element error must
            // be ≤ effective block scale × 1.0 × (1 + slack). Slack absorbs FP16 row-scale rounding.
            let bound = row_scale_a * block_scale * 1.0 * 1.01 + 1e-5;
            for i in 0..32 {
                let idx = b * 32 + i;
                let err = (recovered[idx] - row[idx]).abs();
                assert!(err <= bound,
                        "block {} elem {} err {} exceeds bound {} (block_e={}, row_scale_a={}, block_scale={})",
                        b, i, err, bound, block_e, row_scale_a, block_scale);
            }
        }
    }

    #[test]
    fn header_layout_matches_spec() {
        // 64 elements = 2 blocks. Row size: 16 + 2*17 = 50 bytes.
        let row = vec![3.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        assert_eq!(packed.len(), 50);
        // Block count == 2.
        let bc = u16::from_le_bytes([packed[4], packed[5]]);
        assert_eq!(bc, 2);
        // Format flags: rotation off, no row_scale_b.
        assert_eq!(packed[6] & 0x0F, 0);
        // First block UE8M0 byte at offset 16.
        // Last block payload ends at 16 + 2*17 = 50 (= total).
        // Sanity: row_scale_a > 0 (FP16 bits non-zero).
        let rs_bits = u16::from_le_bytes([packed[0], packed[1]]);
        assert_ne!(rs_bits, 0);
    }

    #[test]
    fn mfp4_stamps_rotation_flag() {
        // MFP4G32 must stamp format_flags = 0x05 (bit 0 + bits 2-3 = 01) in every row
        // header so loaders/tooling can detect the offline-FWHT variant. Byte length must
        // match HFP4G32 (only the flag byte and the rotated weight content differ).
        let m = 3;
        let k = 256;
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let f32_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let packed = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
        let row_bytes = 16 + 17 * (k / 32);
        assert_eq!(packed.len(), m * row_bytes, "MFP4G32 byte length mismatch");
        for r in 0..m {
            let off = r * row_bytes;
            assert_eq!(
                packed[off + 6],
                0x05,
                "row {} format_flags expected 0x05, got {:#x}",
                r,
                packed[off + 6]
            );
            // block_count must equal k/32.
            let bc = u16::from_le_bytes([packed[off + 4], packed[off + 5]]);
            assert_eq!(bc as usize, k / 32);
        }
    }

    // Orthogonality of the FWHT (`dot(R(W), R(x)) ≈ dot(W, x)`) is the load-bearing
    // correctness property and is empirically validated by `examples/test_gemv_mfp4g32.rs`
    // across K = {512, 1024, 1280, 1536, 1792, 2048} on real GPU hardware (max-abs error
    // ≤ 1.14e-5 vs 5e-3 tolerance — three orders of magnitude under). A CPU-only unit test
    // can't tighten that further without duplicating the GPU's CPU-reference path.
}

/// Map a safetensors parent tensor name to the corresponding llama.cpp
/// imatrix tensor base name. Returns None if the safetensors tensor isn't
/// one of the routed-expert MoE tensors we have imatrix data for.
fn safetensors_to_imatrix_key(parent: &str) -> Option<(String, usize)> {
    // Expected pattern: model.language_model.layers.{N}.mlp.experts.{gate_up_proj|down_proj}
    let suffix_gate = ".mlp.experts.gate_up_proj";
    let suffix_down = ".mlp.experts.down_proj";
    let (prefix, kind) = if let Some(p) = parent.strip_suffix(suffix_gate) {
        (p, "ffn_gate_exps")
    } else if let Some(p) = parent.strip_suffix(suffix_down) {
        (p, "ffn_down_exps")
    } else {
        return None;
    };
    // Extract layer N from "...layers.{N}".
    let layer_marker = ".layers.";
    let layer_idx_start = prefix.rfind(layer_marker)? + layer_marker.len();
    let layer_str = &prefix[layer_idx_start..];
    let n: usize = layer_str.parse().ok()?;
    Some((format!("blk.{}.{}.weight", n, kind), n))
}

/// Pull per-expert column-weights from an imatrix GGUF for a given
/// MoE-expert parent tensor (e.g. `...experts.gate_up_proj`). Returns
/// `Some(per_expert_col_weights)` where the outer Vec has `n_experts`
/// entries, each an inner Vec of length K with `sqrt(in_sum2[j] / counts)`
/// (the per-column importance scale).
///
/// Returns None when the parent doesn't map to a known imatrix key, or
/// the tensor isn't present in the imatrix.
fn imatrix_col_weights_for_parent(
    gguf: &gguf_input::GgufFile,
    parent: &str,
    n_experts: usize,
) -> Option<Vec<Vec<f32>>> {
    let (base_key, _layer) = safetensors_to_imatrix_key(parent)?;
    let in_sum2_name = format!("{}.in_sum2", base_key);
    let counts_name = format!("{}.counts", base_key);
    let in_sum2 = gguf.tensors.iter().find(|t| t.name == in_sum2_name)?;
    let counts = gguf.tensors.iter().find(|t| t.name == counts_name)?;
    // Shape: in_sum2 is [K, n_experts] (GGUF column-major-ish: shape[0]=K is innermost).
    if in_sum2.shape.len() != 2 || counts.shape.len() != 2 {
        return None;
    }
    let k = in_sum2.shape[0];
    let n_exp = in_sum2.shape[1];
    if n_exp != n_experts {
        eprintln!(
            "  imatrix: {} n_experts mismatch ({} vs {})",
            in_sum2_name, n_exp, n_experts
        );
        return None;
    }
    let in_sum2_bytes = gguf.tensor_data(in_sum2);
    let counts_bytes = gguf.tensor_data(counts);
    let in_sum2_flat: Vec<f32> = in_sum2_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let counts_flat: Vec<f32> = counts_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if in_sum2_flat.len() != k * n_exp || counts_flat.len() != n_exp {
        eprintln!("  imatrix: {} length mismatch", in_sum2_name);
        return None;
    }
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let count = counts_flat[e].max(1.0);
        let offset = e * k;
        let mut col_w: Vec<f32> = Vec::with_capacity(k);
        for j in 0..k {
            // in_sum2 stores SUM of x_j² over `count` activations; mean is
            // in_sum2/count. Take sqrt for the per-column importance scale
            // (matches the C-norm used by GPTQ / Hessian-diagonal methods).
            col_w.push((in_sum2_flat[offset + j] / count).sqrt());
        }
        out.push(col_w);
    }
    Some(out)
}

/// Per-layer "importance score" from an imatrix GGUF, used by Phase 5
/// tiered MQ-Lloyd to rank routed-expert layers.
///
/// Importance proxy: **mean activation magnitude per expert** =
/// `sum(in_sum2) / sum(counts)`. The mean (not sum) is the right
/// per-layer comparator because `counts` is approximately constant
/// across layers in a typical imatrix calibration (every layer sees
/// the same total tokens). Per-expert mean activation magnitude varies
/// substantially because different layers operate at different
/// activation scales.
///
/// Returns `None` if the imatrix doesn't have ffn_gate_exps tensors
/// (non-MoE imatrix). Returns a Vec<f64> of length n_layers; layers
/// not present get f64::NAN.
fn imatrix_layer_activation_counts(
    gguf: &gguf_input::GgufFile,
    n_layers: usize,
) -> Option<Vec<f64>> {
    let mut out = vec![f64::NAN; n_layers];
    let mut found_any = false;
    for n in 0..n_layers {
        let in_sum2_name = format!("blk.{}.ffn_gate_exps.weight.in_sum2", n);
        let counts_name = format!("blk.{}.ffn_gate_exps.weight.counts", n);
        let sum2 = gguf.tensors.iter().find(|t| t.name == in_sum2_name);
        let cts = gguf.tensors.iter().find(|t| t.name == counts_name);
        if let (Some(s2), Some(c)) = (sum2, cts) {
            let s2_bytes = gguf.tensor_data(s2);
            let c_bytes = gguf.tensor_data(c);
            let sum2_total: f64 = s2_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
                .sum();
            let counts_total: f64 = c_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
                .sum();
            if counts_total > 0.0 {
                // mean activation magnitude per K-column per expert in this layer
                out[n] = sum2_total / counts_total;
                found_any = true;
            }
        }
    }
    if found_any {
        Some(out)
    } else {
        None
    }
}

/// Imatrix-weighted MQ2-Lloyd quantization. Per-column importance weights
/// from a calibration imatrix shift the Lloyd codebook centroids toward
/// values that minimize the IMPORTANCE-WEIGHTED MSE rather than uniform
/// MSE. Helps preserve precision on high-activation columns.
///
/// Mathematical caveat: the FWHT rotation mixes columns within a block, so
/// per-position weighting in the rotated domain is not exactly equivalent
/// to per-column weighting in the original domain (off-diagonal terms in
/// the rotated Hessian are non-zero). This is a first-order approximation:
/// it tilts centroid choice toward high-importance positions but misses
/// the cross-column coupling that a proper GPTQ-LDLQ solve would capture.
///
/// `col_weights` is shape [K] (per-original-column importance values, e.g.
/// sqrt(E[x²]) from an imatrix). For each 256-weight block at offset b in
/// `f32_data` row-major, the relevant slice is
/// `col_weights[(b % blocks_per_row) * 256 .. + 256]`.
fn quantize_mq2g256_lloyd_weighted(
    f32_data: &[f32],
    col_weights: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let blocks_per_row = col_weights.len() / group_size;
    assert!(blocks_per_row > 0, "col_weights too short");
    let mut output = vec![0u8; n_blocks * block_bytes];

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Per-position weights for this block — from the matching column
            // slice of the importance vector. (See caveat above re: FWHT.)
            let col_off = (b % blocks_per_row) * group_size;
            let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                // 16-iter cap matches the plain Lloyd path; per the
                // lloyd_iteration_headroom probe, this reaches the MSE
                // plateau on heavy-tailed + sparse distributions.
                let max_iter = 16;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    // Weighted centroid update: cb[k] = sum_{i in k} w_i * v_i / sum_{i in k} w_i.
                    // (The assignment step is UNWEIGHTED — w_i is a per-point
                    // scalar that cancels from argmin_k |v_i - cb[k]|²; only
                    // the centroid update changes from uniform Lloyd.)
                    let mut weighted_sums = [0.0f64; 4];
                    let mut weight_totals = [0.0f64; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 {
                            changed += 1;
                        }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        let pw = block_w[i] as f64;
                        weighted_sums[best] += pw * w as f64;
                        weight_totals[best] += pw;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..4 {
                        if weight_totals[k] > 0.0 {
                            cb[k] = (weighted_sums[k] / weight_totals[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending (canonical header).
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 4];
            let mut inv: [u8; 4] = [0; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 {
                indices[i] = inv[indices[i] as usize];
            }

            for k in 0..4 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 {
                    byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

/// Sequential-error-feedback MQ2-Lloyd. Simplified GPTQ-style quant: for
/// each 256-block, fit the Lloyd codebook normally, then quantize columns
/// LEFT-TO-RIGHT with the residual quantization error propagated into
/// the next column's target. Captures the "compensate for past errors"
/// insight of GPTQ-LDLQ without the full Cholesky-of-Hessian solve.
///
/// Mathematical caveat: true LDLQ would use the rotated Hessian
/// `R·diag(c)·R^T` to compute the precise per-column propagation weights.
/// This implementation uses pure forward-propagation (no decay, no off-
/// diagonal Hessian) — a first-order approximation that empirically
/// recovers most of LDLQ's benefit at a fraction of the cost. Per-
/// position imatrix weighting still drives the underlying Lloyd
/// codebook fit.
///
/// Empirical sweep (Qwen3.6-35B-A3B, lloyd-mq2_coherence_harness.py,
/// all-MQ2-GPTQ recipe, greedy decode): damping=0.8 lands at 9 ok /
/// 1 warn / 0 fail on the 10-prompt coherence battery — best in the
/// [0.3, 1.0] sweep. See commit history for full bench numbers.
fn quantize_mq2g256_lloyd_gptq(
    f32_data: &[f32],
    col_weights: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let blocks_per_row = col_weights.len() / group_size;
    assert!(blocks_per_row > 0, "col_weights too short");
    let mut output = vec![0u8; n_blocks * block_bytes];

    // Tunable: forward-propagation damping.
    //
    // 2026-05-19 update — damping default changed to 0.0 (was 0.8) after
    // the gptq_damping_probe synthetic-data sweep showed monotonic MSE
    // regression at every d>0, on every tested distribution including
    // strongly-correlated AR(1) inputs (decay=0.9). The Qwen3.6-35B-A3B
    // sweep below historically picked d=0.8 because the model was
    // quantized with a REAL imatrix file → the imatrix-weighted codebook
    // fit step paid for the noise the sequential pass injects. On models
    // built with unit imatrix (DeepSeek V4 all-MQ2-GPTQ), the codebook fit
    // degenerates to plain Lloyd and the sequential pass contributes ONLY
    // noise — DeepSeek V4 mq2-gptq-all.hfq measured 1.9-3.3x worse PPL than
    // lloyd-mq2 on wikitext2-test as a direct consequence. See
    // project_gptq_lloyd_pretendgptq_finding memory + the probe results.
    //
    //   d=0.3 → PPL 12.24 | 7 ok / 3 warn — fails fibonacci_c (Qwen3.6)
    //   d=0.5 → PPL 12.84 | 6 ok / 4 warn (Qwen3.6)
    //   d=0.8 → PPL 14.66 | 9 ok / 1 warn — passes fibonacci_c (Qwen3.6)
    //   d=1.0 → PPL 18.28 | 9 ok / 1 warn (Qwen3.6)
    //
    // At d=0 the sequential pass is a no-op and the function is byte-
    // identical to quantize_mq2g256_lloyd_weighted (which is the right
    // thing to use directly if you don't need the GPTQ name in the
    // pipeline log). Override via env var.
    let damping_env: f32 = std::env::var("HIPFIRE_GPTQ_DAMPING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    if damping_env > 0.0 {
        let has_real_imatrix = col_weights.iter().any(|&w| (w - 1.0).abs() > 1e-6);
        if !has_real_imatrix {
            eprintln!(
                "warning: HIPFIRE_GPTQ_DAMPING={damping_env} with unit imatrix → \
                 strictly worse than plain Lloyd (see gptq_damping_probe). \
                 Either provide --imatrix or use --format mq4-routed-lloyd-mq2-native."
            );
        }
    }

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            let col_off = (b % blocks_per_row) * group_size;
            let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

            // Step 1: Lloyd codebook fit (imatrix-weighted, same as
            // `quantize_mq2g256_lloyd_weighted`). Used to seed the 4
            // centroids before sequential assignment.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];
            let range = sorted[255] - sorted[0];
            if range > 0.0 {
                // 16-iter cap matches plain Lloyd; see lloyd_iteration_headroom.
                let max_iter = 16;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut weighted_sums = [0.0f64; 4];
                    let mut weight_totals = [0.0f64; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 {
                            changed += 1;
                        }
                        prev_assignments[i] = best as u8;
                        let pw = block_w[i] as f64;
                        weighted_sums[best] += pw * w as f64;
                        weight_totals[best] += pw;
                    }
                    if it > 0 && changed == 0 {
                        break;
                    }
                    for k in 0..4 {
                        if weight_totals[k] > 0.0 {
                            cb[k] = (weighted_sums[k] / weight_totals[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending (canonical header).
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| {
                cb[a]
                    .partial_cmp(&cb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_cb = [0.0f32; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
            }
            let cb_final = sorted_cb;

            // Step 2: Sequential GPTQ-style quantize.
            // Forward-propagate the residual error into each next column's
            // target. The "damping" factor controls how aggressively past
            // errors influence future assignments. Empirically:
            //   factor=1.0 — pure forward propagation (full residual)
            //   factor=0.5 — half-damping; safer against runaway accumulation
            //   factor=0.0 — no propagation (degenerates to standard Lloyd)
            // 0.5 is a conservative starting point.
            let damping = damping_env;
            let mut indices = [0u8; 256];
            let mut residual = 0.0f32;
            for i in 0..256 {
                let target = group[i] + residual;
                let mut best = 0usize;
                let mut best_d = (target - cb_final[0]).abs();
                for k in 1..4 {
                    let d = (target - cb_final[k]).abs();
                    if d < best_d {
                        best_d = d;
                        best = k;
                    }
                }
                indices[i] = best as u8;
                let err = target - cb_final[best];
                residual = err * damping;
            }

            // Pack header + indices.
            for k in 0..4 {
                let bits = f32_to_fp16_bits(cb_final[k]);
                out_chunk[2 * k] = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 {
                    byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

/// Inverse FWHT for MQ-family dequantization (sibling of cpu_fwht_256).
fn cpu_inv_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    cpu_fwht_256(x, signs2, signs1);
}

/// MQ2-Lloyd dequantize for round-trip / re-quant pipelines. Mirrors
/// the kernel's decode: 4-entry fp16 codebook + 2-bit indices per 256-
/// weight group, then inverse FWHT.
fn dequantize_mq2g256_lloyd_to_f32(
    data: &[u8],
    n_weights: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<f32> {
    let group_size = 256;
    let block_bytes = 72;
    let n_blocks = (n_weights + group_size - 1) / group_size;
    assert!(data.len() == n_blocks * block_bytes);
    let mut out = vec![0.0f32; n_weights];
    use rayon::prelude::*;
    out.par_chunks_mut(group_size)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            let mut group = [0.0f32; 256];
            for i in 0..64 {
                let byte_val = blk[8 + i];
                for j in 0..4 {
                    let idx = (byte_val >> (j * 2)) & 0x3;
                    group[4 * i + j] = cb[idx as usize];
                }
            }
            cpu_inv_fwht_256(&mut group, signs1, signs2);
            let actual = out_chunk.len();
            out_chunk.copy_from_slice(&group[..actual]);
        });
    out
}

// ─── HFQ File Format ────────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

// QuantType — the on-disk HFQ `quant_type` byte-contract — now lives in the
// shared `hipfire-quant-format` leaf crate, depended on by BOTH this writer and
// the engine readers (hipfire-runtime + per-arch loaders). It previously lived
// here privately, which forced every reader to re-hardcode the integers and let
// them drift. Re-exported so existing `QuantType::Foo` references and the
// `t.quant_type as u8` byte emission keep working unchanged.
pub use hipfire_quant_format::QuantType;

/// Per-tensor precision level assigned by the K-map pre-pass.
/// Determines whether a tensor gets the base format, a 6-bit promotion,
/// Q8, or F16. See docs/superpowers/specs/2026-05-08-mixed-quant-kmap-design.md.
#[derive(Clone, Copy, Debug, PartialEq)]
enum QuantLevel {
    /// Store as F16 (norms, biases, 1D tensors).
    F16,
    /// Store as Q8_F16 (embeddings, lm_head, MoE routers).
    Q8,
    /// Promote to 6-bit variant of the base format (edge layers, MoE expert FFN).
    Promote6,
    /// Override the default for a specific tensor class (today: lm_head)
    /// to a CLI-specified format. Currently unused on this branch (no emission
    /// site); kept so origin/master's lm_head-format override match arms
    /// compile after the merge. Re-wire to `--lm-head-format` when the
    /// configurable-kmap-pair refactor lands here.
    #[allow(dead_code)]
    Override(GgufFormat),
    /// Use the base format as-is.
    Base,
}

/// Extract layer index from a tensor name.
/// Handles both safetensors (`layers.{N}.`) and GGUF (`blk.{N}.`) patterns.
/// Uses unanchored search to handle any prefix (model.layers, model.language_model.layers, etc.).
fn parse_layer_idx(name: &str) -> Option<usize> {
    // Try safetensors pattern: "layers.{N}."
    if let Some(pos) = name.find("layers.") {
        let after = &name[pos + 7..]; // skip "layers."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    // Try GGUF pattern: "blk.{N}."
    if let Some(pos) = name.find("blk.") {
        let after = &name[pos + 4..]; // skip "blk."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

/// Stride for alternating-mode promotion: edge layers always promoted,
/// plus every Nth middle layer. 3 was chosen empirically — promotes ~40%
/// of middle layers, matching llama.cpp Q4_K_M's budget-allocation pattern.
/// On MoE 3.6-35B-A3B: stride=3 gives PPL 8K=19.96 at 21.8 GB vs full
/// K-map PPL 8K=20.07 at 27.7 GB.
const ALTERNATING_STRIDE: usize = 3;

/// llama.cpp-style alternating promotion: edge layers always promoted,
/// middle layers promoted every `stride` layers.
fn is_positional_promote(idx: usize, n_layers: usize, stride: usize) -> bool {
    if n_layers == 0 || stride == 0 {
        return false;
    }
    if idx < 2 || idx >= n_layers.saturating_sub(2) {
        return true;
    }
    (idx - 2) % stride == 0
}

/// Resolve the quantization level for a tensor based on its name, the model's
/// layer count, whether the model is MoE, and the K-map mode.
///
/// `kmap_mode`: 0 = full (all candidates promoted), 1 = alternating
/// (experts + ffn_down every 3rd middle layer, edge layers always),
/// 2 = typed (ffn_down + attn_v everywhere).
///
/// Note: In the safetensors path, norms/biases are filtered by `should_quantize()`
/// before this function is called. Rules 1-2 exist for the GGUF path and completeness.
#[cfg(test)]
fn kmap_resolve(name: &str, n_layers: usize, is_moe: bool) -> QuantLevel {
    kmap_resolve_mode(name, n_layers, is_moe, 0)
}

fn kmap_resolve_mode(name: &str, n_layers: usize, is_moe: bool, kmap_mode: u8) -> QuantLevel {
    // Rule 1: norms, biases, 1D (GGUF path mainly)
    if name.contains("norm") || name.contains("bias") {
        return QuantLevel::F16;
    }

    // Rule 2: embeddings, lm_head, output projection
    if name.contains("embed_tokens")
        || name.contains("token_embd")
        || name.ends_with("embeddings.weight") // nemotron_h: backbone.embeddings.weight
        || name.contains("lm_head")
        || name.ends_with("output.weight")
    {
        return QuantLevel::Q8;
    }

    // Rule 3: MoE routers
    if is_moe && (name.ends_with("mlp.gate.weight") || name.contains("shared_expert_gate")) {
        return QuantLevel::Q8;
    }

    // Rule 4: MoE expert FFN weights
    if is_moe && name.contains("mlp.experts.") {
        if kmap_mode == 1 {
            // Alternating: promote expert groups only in positional layers
            if let Some(idx) = parse_layer_idx(name) {
                if is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                return QuantLevel::Base;
            }
        }
        return QuantLevel::Promote6;
    }

    // Mode 2 (typed): promote ffn_down and attn_v in all layers.
    if kmap_mode == 2 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        let is_v = name.contains("v_proj") || name.contains("attn_v");
        if is_down || is_v {
            return QuantLevel::Promote6;
        }
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    return QuantLevel::Promote6;
                }
            }
        }
        return QuantLevel::Base;
    }

    // Mode 1 (alternating): ffn_down in edge + every 3rd middle layer.
    // Edge-layer rule mirrors mode 0 below: attn+FFN for MoE (full promotion
    // gives -19.8% PPL on 3.6-35B-A3B), FFN only for dense (attn promotion
    // regresses PPL +3.1% on 27B). Bench: asym4 KV, ctx=8192, wikitext-2-test.
    // See ppl_kmap_20260508.md.
    if kmap_mode == 1 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if is_down && is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                // Edge layers: attn+FFN for MoE, FFN only for dense.
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    if is_moe {
                        return QuantLevel::Promote6;
                    }
                    let is_ffn = name.contains("mlp.") || name.contains("ffn");
                    if is_ffn {
                        return QuantLevel::Promote6;
                    }
                }
            }
        }
        return QuantLevel::Base;
    }

    // Rule 5 (full mode 0): edge layers (first 2 + last 2).
    // Dense models: FFN only — attn promotion regresses PPL (+3.1% on 27B).
    // MoE models: attn+FFN — full promotion gives -19.8% PPL on 3.6-35B-A3B.
    // Bench: asym4 KV, ctx=8192, wikitext-2-test. See ppl_kmap_20260508.md.
    if n_layers > 0 {
        if let Some(idx) = parse_layer_idx(name) {
            if idx < 2 || idx >= n_layers.saturating_sub(2) {
                if is_moe {
                    // MoE: promote all tensors in edge layers (attn + FFN)
                    return QuantLevel::Promote6;
                }
                // Dense: promote FFN only — attn stays at Base
                let is_ffn = name.contains("mlp.") || name.contains("ffn");
                if is_ffn {
                    return QuantLevel::Promote6;
                }
            }
        }
    }

    // Rule 6: everything else
    QuantLevel::Base
}

struct HfqTensor {
    name: String,
    quant_type: QuantType,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
    /// When data is spilled to disk, this holds the byte count.
    /// `data` is empty and the bytes live in the spill file.
    spilled_len: u64,
}

fn tensor_param_count(t: &HfqTensor) -> u64 {
    t.shape
        .iter()
        .fold(1u64, |acc, &dim| acc.saturating_mul(dim as u64))
}

fn config_u64_any(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    fn get_from_scope(scope: &serde_json::Value, keys: &[&str]) -> Option<u64> {
        keys.iter().find_map(|key| scope.get(*key)?.as_u64())
    }

    get_from_scope(config, keys)
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|scope| get_from_scope(scope, keys))
        })
        .or_else(|| {
            config
                .get("moe")
                .and_then(|scope| get_from_scope(scope, keys))
        })
        .or_else(|| {
            config
                .get("ffn_config")
                .and_then(|scope| get_from_scope(scope, keys))
        })
}

fn model_config_from_metadata(metadata: &serde_json::Value) -> &serde_json::Value {
    metadata.get("config").unwrap_or(metadata)
}

fn routed_moe_config(metadata: &serde_json::Value) -> Option<(u64, u64)> {
    let config = model_config_from_metadata(metadata);
    let num_experts = config_u64_any(
        config,
        &[
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "n_experts",
        ],
    )?;
    let top_k = config_u64_any(
        config,
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "n_experts_per_tok",
            "moe_top_k",
            "top_k",
            "num_selected_experts",
        ],
    )?;
    if num_experts == 0 || top_k == 0 {
        None
    } else {
        Some((num_experts, top_k))
    }
}

fn is_routed_expert_tensor_name(name: &str) -> bool {
    if name.contains(".shared_expert") || name.contains(".shared_experts.") {
        return false;
    }
    name.contains(".mlp.experts.")
        || name.contains(".ffn.experts.")
        || name.contains(".block_sparse_moe.experts.")
        || name.contains(".feed_forward.experts.")
        || name.contains(".mixer.experts.")
}

fn parameter_counts_metadata(
    metadata: &serde_json::Value,
    tensors: &[HfqTensor],
    total_params: u64,
    quantized_params: u64,
    skipped_params: u64,
) -> serde_json::Value {
    let mut routed_expert_params = 0u64;
    for t in tensors {
        if is_routed_expert_tensor_name(&t.name) {
            routed_expert_params = routed_expert_params.saturating_add(tensor_param_count(t));
        }
    }

    let (active_params, effective_params, moe) = if routed_expert_params > 0 {
        if let Some((num_experts, top_k)) = routed_moe_config(metadata) {
            let numerator = routed_expert_params.saturating_mul(top_k);
            let routed_active = numerator / num_experts;
            let active = total_params
                .saturating_sub(routed_expert_params)
                .saturating_add(routed_active);
            (
                active,
                active,
                Some(serde_json::json!({
                    "num_experts": num_experts,
                    "num_experts_per_tok": top_k,
                    "routed_expert_params": routed_expert_params,
                    "routed_expert_active_params": routed_active,
                    "active_rule": "dense_and_shared_full_plus_routed_top_k_over_num_experts",
                    "routed_active_fraction": {
                        "numerator": numerator,
                        "denominator": num_experts,
                    },
                })),
            )
        } else {
            (
                total_params,
                total_params,
                Some(serde_json::json!({
                    "routed_expert_params": routed_expert_params,
                    "active_rule": "unknown_top_k_or_num_experts",
                })),
            )
        }
    } else {
        (total_params, total_params, None)
    };

    let source_total_params = total_params.saturating_add(skipped_params);
    let mut counts = serde_json::json!({
        "schema": "hipfire.parameter_counts.v1",
        "total_params": total_params,
        "source_total_params": source_total_params,
        "active_params": active_params,
        "effective_params": effective_params,
        "quantized_params": quantized_params,
        "skipped_params": skipped_params,
    });
    if let Some(moe) = moe {
        if let serde_json::Value::Object(ref mut map) = counts {
            map.insert("moe".to_string(), moe);
        }
    }
    counts
}

fn insert_parameter_counts_metadata(
    metadata: &mut serde_json::Value,
    tensors: &[HfqTensor],
    total_params: u64,
    quantized_params: u64,
    skipped_params: u64,
) {
    let counts = parameter_counts_metadata(
        metadata,
        tensors,
        total_params,
        quantized_params,
        skipped_params,
    );
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("parameter_counts".to_string(), counts);
    }
}

fn insert_quant_format_metadata(metadata: &mut serde_json::Value, format: &str) {
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("quant_format".to_string(), serde_json::json!(format));
    }
}

struct HfqInputTensor {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_offset: usize,
    data_size: usize,
}

struct HfqInputFile {
    _file: File,
    mmap: Mmap,
    arch_id: u32,
    metadata_json: String,
    tensors: Vec<HfqInputTensor>,
}

impl HfqInputFile {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 32 || &mmap[0..4] != HFQ_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not an HFQM container",
            ));
        }
        let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

        if metadata_offset >= data_offset || data_offset > mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid HFQM offsets metadata={metadata_offset} data={data_offset}"),
            ));
        }

        let meta_bytes = &mmap[metadata_offset..data_offset];
        let mut brace_depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        let mut json_end = None;
        for (i, &b) in meta_bytes.iter().enumerate() {
            if escape {
                escape = false;
                continue;
            }
            if b == b'\\' && in_string {
                escape = true;
                continue;
            }
            if b == b'"' {
                in_string = !in_string;
                continue;
            }
            if !in_string {
                if b == b'{' {
                    brace_depth += 1;
                } else if b == b'}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        json_end = Some(i + 1);
                        break;
                    }
                }
            }
        }
        let json_end = json_end.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HFQM metadata JSON did not end",
            )
        })?;
        let metadata_json = String::from_utf8(meta_bytes[..json_end].to_vec()).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HFQM metadata is not UTF-8: {e}"),
            )
        })?;

        let mut pos = metadata_offset + json_end;
        if pos + 4 > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HFQM index missing tensor count",
            ));
        }
        let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        if idx_n != n_tensors {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HFQM index count {idx_n} != header count {n_tensors}"),
            ));
        }
        pos += 4;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut cumulative_offset = data_offset;
        for _ in 0..n_tensors {
            if pos + 2 > data_offset {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "HFQM index truncated at name length",
                ));
            }
            let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            if pos + name_len + 2 > data_offset {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "HFQM index truncated at name/shape header",
                ));
            }
            let name = String::from_utf8(mmap[pos..pos + name_len].to_vec()).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("HFQM tensor name is not UTF-8: {e}"),
                )
            })?;
            pos += name_len;
            let quant_type = mmap[pos];
            pos += 1;
            let n_dims = mmap[pos] as usize;
            pos += 1;
            if pos + n_dims * 4 + 12 > data_offset {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "HFQM index truncated at shape/data size",
                ));
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            if cumulative_offset + data_size > mmap.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "HFQM tensor {name} range {}..{} exceeds file size {}",
                        cumulative_offset,
                        cumulative_offset + data_size,
                        mmap.len()
                    ),
                ));
            }
            tensors.push(HfqInputTensor {
                name,
                quant_type,
                shape,
                group_size,
                data_offset: cumulative_offset,
                data_size,
            });
            cumulative_offset += data_size;
        }

        Ok(Self {
            _file: file,
            mmap,
            arch_id,
            metadata_json,
            tensors,
        })
    }

    fn tensor_data(&self, t: &HfqInputTensor) -> &[u8] {
        &self.mmap[t.data_offset..t.data_offset + t.data_size]
    }
}

// ─── XXH64 provenance hashing ───────────────────────────────────────────────

struct Xxh64 {
    inner: XxHash64,
}

impl Xxh64 {
    fn new(seed: u64) -> Self {
        Self {
            inner: XxHash64::with_seed(seed),
        }
    }

    fn update(&mut self, input: &[u8]) {
        self.inner.write(input);
    }

    fn digest(&self) -> u64 {
        self.inner.finish()
    }
}

#[cfg(test)]
fn xxh64_hex(bytes: &[u8]) -> String {
    let mut h = Xxh64::new(0);
    h.update(bytes);
    format!("{:016x}", h.digest())
}

fn xxh64_update_u8(h: &mut Xxh64, v: u8) {
    h.update(&[v]);
}

fn xxh64_update_u32(h: &mut Xxh64, v: u32) {
    h.update(&v.to_le_bytes());
}

fn xxh64_update_u64(h: &mut Xxh64, v: u64) {
    h.update(&v.to_le_bytes());
}

fn hfq_quantization_hash_metadata(
    tensors: &[HfqTensor],
    spill: Option<&TensorSpill>,
) -> std::io::Result<serde_json::Value> {
    let mut h = Xxh64::new(0);
    let mut payload_bytes = 0u64;
    h.update(b"hipfire-hfq-quantized-tensor-payload-v1");

    let mut spill_reader = if let Some(spill) = spill {
        Some(std::io::BufReader::new(File::open(&spill.path)?))
    } else {
        None
    };
    let mut buf = vec![0u8; 4 * 1024 * 1024];

    for t in tensors {
        let name_bytes = t.name.as_bytes();
        xxh64_update_u64(&mut h, name_bytes.len() as u64);
        h.update(name_bytes);
        xxh64_update_u8(&mut h, t.quant_type as u8);
        xxh64_update_u64(&mut h, t.shape.len() as u64);
        for &dim in &t.shape {
            xxh64_update_u32(&mut h, dim);
        }
        xxh64_update_u32(&mut h, t.group_size);
        let data_len = if t.spilled_len > 0 {
            t.spilled_len
        } else {
            t.data.len() as u64
        };
        xxh64_update_u64(&mut h, data_len);
        payload_bytes += data_len;

        if t.spilled_len > 0 {
            let reader = spill_reader
                .as_mut()
                .expect("spilled tensor requires spill reader");
            let mut remaining = t.spilled_len as usize;
            while remaining > 0 {
                let chunk = remaining.min(buf.len());
                use std::io::Read;
                reader.read_exact(&mut buf[..chunk])?;
                h.update(&buf[..chunk]);
                remaining -= chunk;
            }
        } else {
            h.update(&t.data);
        }
    }

    Ok(serde_json::json!({
        "algorithm": "xxh64",
        "seed": 0,
        "scope": "hfq_tensor_index_and_payload_v1",
        "value": format!("{:016x}", h.digest()),
        "tensor_count": tensors.len(),
        "payload_bytes": payload_bytes,
        "producer": {
            "package": "hipfire-quantize",
            "hipfire_version": env!("CARGO_PKG_VERSION"),
            "git_commit": git_commit(),
            "git_branch": git_branch(),
            "git_describe": git_describe(),
            "git_dirty": git_dirty(),
        },
    }))
}

fn metadata_with_quantization_hash(
    mut metadata: serde_json::Value,
    tensors: &[HfqTensor],
    spill: Option<&TensorSpill>,
) -> std::io::Result<String> {
    let hash = hfq_quantization_hash_metadata(tensors, spill)?;
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("quantization_hash".to_string(), hash);
    }
    serde_json::to_string(&metadata)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn git_commit() -> Option<String> {
    command_stdout("git", &["rev-parse", "HEAD"])
}

fn git_branch() -> Option<String> {
    command_stdout("git", &["rev-parse", "--abbrev-ref", "HEAD"])
}

fn git_describe() -> Option<String> {
    command_stdout("git", &["describe", "--always", "--dirty", "--tags"])
}

fn git_dirty() -> Option<bool> {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(!out.stdout.is_empty())
}

/// Streaming tensor spill file. When the quantizer accumulates more than
/// `SPILL_THRESHOLD` bytes of tensor data in memory, it flushes completed
/// tensors to this file. At write_hfq time, spilled data is copied from
/// the spill file instead of from memory, keeping peak RSS bounded.
struct TensorSpill {
    file: std::io::BufWriter<File>,
    path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    fn new(dir: &Path) -> std::io::Result<Self> {
        // PID-unique so concurrent quantize runs in the same output dir don't
        // share a spill path (a sibling run's Drop would otherwise delete this
        // run's spill file → write_hfq NotFound panic).
        let path = dir.join(format!(".hipfire_quant_spill.{}.tmp", std::process::id()));
        let file = std::io::BufWriter::with_capacity(4 * 1024 * 1024, File::create(&path)?);
        Ok(Self {
            file,
            path,
            offset: 0,
        })
    }

    /// Write tensor data to the spill file. Returns the byte count written.
    fn spill(&mut self, data: &[u8]) -> std::io::Result<u64> {
        use std::io::Write;
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(data.len() as u64)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }

    fn cleanup(self) {
        // Explicit cleanup — Drop impl handles the actual removal.
        drop(self);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        // Ensure the temp file is removed even on panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Spill tensors whose data is in memory to the spill file, freeing RAM.
/// Called after each layer's expert batch to keep peak RSS bounded.
fn maybe_spill(tensors: &mut [HfqTensor], spill: &mut TensorSpill, threshold: usize) {
    let in_mem: usize = tensors
        .iter()
        .filter(|t| t.spilled_len == 0)
        .map(|t| t.data.len())
        .sum();
    if in_mem < threshold {
        return;
    }
    for t in tensors.iter_mut() {
        if t.spilled_len == 0 && !t.data.is_empty() {
            let len = spill.spill(&t.data).unwrap_or(0);
            t.spilled_len = len;
            t.data = Vec::new(); // free the memory
        }
    }
    let _ = spill.flush();
}

fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    spill: Option<&mut TensorSpill>,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    let metadata_bytes = metadata_json.as_bytes();

    // Calculate offsets
    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    // Tensor index follows metadata
    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    // Write tensor count
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        // name length + name
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        // quant type
        index_bytes.push(t.quant_type as u8);
        // n_dims + shape
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        // group size
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        // data size (offset computed at read time from cumulative sizes)
        let data_len = if t.spilled_len > 0 {
            t.spilled_len
        } else {
            t.data.len() as u64
        };
        index_bytes.extend_from_slice(&data_len.to_le_bytes());
    }

    // Data starts after index, aligned to 4096
    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    // Write header (32 bytes)
    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    // Write metadata
    f.write_all(metadata_bytes)?;

    // Write tensor index
    f.write_all(&index_bytes)?;

    // Pad to data alignment
    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    // Write tensor data — from spill file or from memory
    if let Some(spill) = spill {
        let _ = spill.flush();
        let mut spill_reader = std::io::BufReader::new(File::open(&spill.path)?);
        let mut buf = vec![0u8; 4 * 1024 * 1024]; // 4 MB copy buffer
        for t in tensors {
            if t.spilled_len > 0 {
                // Copy from spill file
                let mut remaining = t.spilled_len as usize;
                while remaining > 0 {
                    let chunk = remaining.min(buf.len());
                    use std::io::Read;
                    spill_reader.read_exact(&mut buf[..chunk])?;
                    f.write_all(&buf[..chunk])?;
                    remaining -= chunk;
                }
            } else {
                f.write_all(&t.data)?;
            }
        }
    } else {
        for t in tensors {
            f.write_all(&t.data)?;
        }
    }

    Ok(())
}

// ─── Model Discovery ────────────────────────────────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

/// Determine which tensors to quantize (weight matrices) vs keep as F16 (norms, embeddings)
fn should_quantize(name: &str) -> bool {
    // Vision encoder weights stay FP16 (only ~500M params, run once per image).
    // Qwen3.5-VL uses `model.visual.*` / `visual.*`; dots.ocr uses
    // `vision_tower.*`. Both arches keep vision F16 during bring-up so the
    // per-stage diff against the HF reference activations
    // (`benchmarks/references/<image>_activations/`) doesn't have to absorb
    // both forward-pass implementation noise AND quant noise — clean
    // attribution. See memory `feedback_dots_ocr_vision_f16_during_bringup`.
    if name.starts_with("model.visual.")
        || name.starts_with("visual.")
        || name.starts_with("vision_tower.")
        // Gemma3 multimodal projector (mm_input_projection_weight) — kept F32 so
        // the SigLIP/projector loader's source-precision reader consumes it.
        || name.starts_with("multi_modal_projector.")
    {
        return false;
    }
    if name.contains("norm") || name.contains("bias") {
        return false;
    }
    // Depthwise causal short-conv filters (Mamba-2 / nemotron_h `*.conv1d.weight`,
    // LFM2 / GDN convs) are tiny per-channel recurrence filters of shape
    // [channels, 1, K] — NOT a 2D linear weight. Keep them F16: quantizing a
    // recurrence filter corrupts the SSM/conv state, and the 3D shape isn't a
    // valid mq4 [out, in] matrix anyway. (`conv1d.bias` is already caught above.)
    if name.contains("conv1d") {
        return false;
    }
    // ZAYA1 CCA conv_qk filters (`self_attn.qkv_proj.conv_qk_{depthwise,grouped}.weight`,
    // shapes [conv_ch, 1, K] / [conv_ch, in_per_group, K]) are short causal convs run
    // by a custom f32 kernel, not a 2D linear — keep them F16.
    if name.contains("conv_qk") {
        return false;
    }
    // Quantize everything including embeddings (Q8 embedding saves ~2.3GB for 8B models)
    name.contains("weight")
}

/// antirez ds4 reference keeps three classes at F16 because Q8 measurably
/// regresses PPL on DeepSeek V4: (1) attn compressor wkv + wgate, (2) indexer wq_b +
/// weights_proj, (3) indexer compressor wkv + wgate. All small (≤32 MiB
/// combined across 43 layers).
///
/// Router gate.weight (.ffn.gate.weight) is NOT kept at F16: antirez
/// actually ships it as MQ4G256, and the known-good DeepSeek V4 quant
/// matches. Falling back to the format's default (Q8F16 in deepseek4-q8-mtp)
/// is fine — the router is dispatched via `gemv_auto`.
///
/// `attn.indexer.compressor.*` is a substring of `attn.compressor.*` only
/// in the literal-prefix sense, so order doesn't matter — the substring
/// `.compressor.wkv.weight` matches both `.attn.compressor.wkv.weight` and
/// `.attn.indexer.compressor.wkv.weight` deliberately.
fn is_deepseek4_keep_f16(name: &str) -> bool {
    name.ends_with(".compressor.wkv.weight")
        || name.ends_with(".compressor.wgate.weight")
        || name.ends_with(".indexer.wq_b.weight")
        || name.ends_with(".indexer.weights_proj.weight")
}

/// For mixed quant: should this tensor be Q8 (fast) or Q4 (compressed)?
/// Q8: attention weights, embeddings, lm_head (need occupancy)
/// Q4: FFN weights (bulk of model, benefits from compression)
fn is_q8_tensor(name: &str) -> bool {
    name.contains("self_attn") || name.contains("attn_q") || name.contains("attn_k")
        || name.contains("attn_v") || name.contains("attn_output")
        || name.contains("q_proj") || name.contains("k_proj")
        || name.contains("v_proj") || name.contains("o_proj")
        || name.contains("embed") || name.contains("lm_head")
        // Qwen3.5 DeltaNet attention
        || name.contains("linear_attn")
        // Qwen3.5-MoE: the router (`mlp.gate.weight`, hidden_size × num_experts)
        // is small but precision-sensitive — flat-routing on a quantized router
        // shifts which experts a token sees. Same for the per-layer scalar
        // `mlp.shared_expert_gate.weight` that scales the shared expert. Keep
        // both at Q8 even in Q4-bulk modes.
        || name.ends_with("mlp.gate.weight")
        || name.ends_with("mlp.shared_expert_gate.weight")
        // Nemotron-H Nano-30B A3B: routed MoE router lives under the flat
        // block's mixer namespace.
        || name.ends_with(".mixer.gate.weight")
}

/// Qwen3.5 DeltaNet conv1d weight: `{prefix}.linear_attn.conv1d.weight`,
/// shape [conv_channels, 1, 4]. Small (~32K elem) and runs every token —
/// Q8 is the safe default; lossy 4-bit FWHT formats (mq4/mq3) measurably
/// hurt the gated-delta path.
fn is_conv1d_tensor(name: &str) -> bool {
    name.ends_with("conv1d.weight")
}

/// Nemotron-H projections that should stay Q8 in MQ-family artifacts.
///
/// Local Nano-4B evidence (native-Mamba Python reference + Hipfire f32/Q8/MQ4
/// comparison) shows uncalibrated MQ4 flips the close first-token boundary from
/// `<|im_end|>` to newline when projection-back weights are lossy. Nano-30B
/// bring-up also marked `mixer.in_proj.weight` as ingress-sensitive: it
/// generates the SSM gate, x, B/C, and dt streams, and the Q8 candidate moved
/// the fixed-scale boundary slightly closer to BF16. Keep both classes Q8 for
/// base MQ-family artifacts until an imatrix/AWQ/Lloyd policy is validated for
/// this arch.
fn is_nemotron_h_mq4_q8_protected(name: &str) -> bool {
    name.starts_with("backbone.layers.")
        && (name.ends_with(".mixer.in_proj.weight") || is_nemotron_h_residual_writer(name))
}

/// Nemotron-H projections that write back into the residual stream.
fn is_nemotron_h_residual_writer(name: &str) -> bool {
    name.starts_with("backbone.layers.")
        && (name.ends_with(".mixer.out_proj.weight")
            || name.ends_with(".mixer.down_proj.weight")
            || name.ends_with(".mixer.o_proj.weight"))
}

// ─── Main ────────────────────────────────────────────────────────────────────

/// Resolve a model input to a local directory path.
/// Accepts: local path, HuggingFace model ID (org/name), or HF cache path.
/// If the input looks like a HF model ID and isn't a local path, tries to find it
/// in the HF cache or downloads it via huggingface-cli.
fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);

    // If it's already a valid local directory with config.json, use it directly
    if path.join("config.json").exists() {
        return input.to_string();
    }

    // Check if it looks like a HuggingFace model ID (contains exactly one /)
    if input.contains('/') && !input.contains(std::path::MAIN_SEPARATOR)
        || (cfg!(unix) && input.matches('/').count() == 1)
    {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];

            // Check HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*/
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_dir = format!("{home}/.cache/huggingface/hub/models--{org}--{name}");
            let snapshots_dir = Path::new(&cache_dir).join("snapshots");

            if snapshots_dir.exists() {
                // Find the first snapshot directory
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let snap_path = entry.path();
                        if snap_path.is_dir() && snap_path.join("config.json").exists() {
                            eprintln!("Resolved {input} -> {}", snap_path.display());
                            return snap_path.to_string_lossy().to_string();
                        }
                    }
                }
            }

            // Not in cache — try to download
            eprintln!("Model {input} not found locally. Downloading via huggingface-cli...");
            let status = std::process::Command::new("huggingface-cli")
                .args(["download", input])
                .status();

            match status {
                Ok(s) if s.success() => {
                    // Retry cache lookup after download
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snap_path = entry.path();
                            if snap_path.is_dir() && snap_path.join("config.json").exists() {
                                eprintln!("Downloaded {input} -> {}", snap_path.display());
                                return snap_path.to_string_lossy().to_string();
                            }
                        }
                    }
                }
                Ok(s) => eprintln!("huggingface-cli download failed with status {s}"),
                Err(e) => eprintln!(
                    "Failed to run huggingface-cli: {e}. Install with: pip install huggingface_hub"
                ),
            }
        }
    }

    // Fall through: return as-is, will fail at config.json read with a helpful error
    input.to_string()
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn required_format_arg<'a>(args: &'a [String]) -> Result<&'a str, &'static str> {
    arg_value(args, "--format").ok_or(
        "--format <FMT> is required; hipfire-quantize does not choose a default quant format",
    )
}

fn normalize_format_flag(flag: &str) -> String {
    flag.trim().to_ascii_lowercase()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OqCalibrationRecipe {
    Plain,
    Awq,
    AwqLdlq,
}

fn oq4_calibration_recipe(format: &str) -> OqCalibrationRecipe {
    match format {
        "oq4+" => OqCalibrationRecipe::Awq,
        // Legacy OP plus spellings predate the positional OQ+ / OQ++ taxonomy.
        // Keep parsing them as the older LDLQ recipe, but emit canonical OQ names
        // in docs and artifacts.
        "oq4++" | "op4+" | "op4-4+" | "op4-8+" => OqCalibrationRecipe::AwqLdlq,
        _ => OqCalibrationRecipe::Plain,
    }
}

fn oq8_calibration_recipe(format: &str) -> OqCalibrationRecipe {
    match format {
        "oq8+" | "oq8-plus" => OqCalibrationRecipe::Awq,
        "oq8++" | "op8+" | "op8-16+" | "op8-plus" => OqCalibrationRecipe::AwqLdlq,
        _ => OqCalibrationRecipe::Plain,
    }
}

fn read_chat_template_file(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to read --chat-template-file {} as UTF-8: {e}",
            path.display()
        );
        std::process::exit(1);
    })
}

fn tokenizer_config_with_chat_template(
    tokenizer_config: Option<serde_json::Value>,
    chat_template: String,
) -> serde_json::Value {
    match tokenizer_config {
        Some(serde_json::Value::Object(mut map)) => {
            map.insert(
                "chat_template".to_string(),
                serde_json::Value::String(chat_template),
            );
            serde_json::Value::Object(map)
        }
        _ => serde_json::json!({ "chat_template": chat_template }),
    }
}

// ─── GGUF input pipeline ────────────────────────────────────────────────────

/// True if the path points to a `.gguf` file on disk.
fn is_gguf_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf")
}

fn is_hfq_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("hfq")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HfqInputFormat {
    F16,
    Bf16,
    Q8F16,
    Hfq4,
    Hfq6,
    Mq4,
    Mq6,
    Mq3,
    Qtip3,
    Qtip4,
    Oq3,
    Oq6,
    Oq4,
    OqPlus,
    OqPlusTiered,
    OqPlusCompact,
    Oq8,
    Oq8Plus,
}

impl HfqInputFormat {
    fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "fp16" | "f16" | "float16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "q8f16" | "q8" => Some(Self::Q8F16),
            "hfq4" | "hfq4g256" | "hf4" => Some(Self::Hfq4),
            "hfq6" | "hfq6g256" | "hf6" => Some(Self::Hfq6),
            "mq4" | "mq4g256" | "magnum" => Some(Self::Mq4),
            "mq6" | "mq6g256" => Some(Self::Mq6),
            "mq3" | "mq3g256" => Some(Self::Mq3),
            "qtip3" => Some(Self::Qtip3),
            "qtip4" => Some(Self::Qtip4),
            "op3" | "op3g256" | "oq3" | "oq3g256" => Some(Self::Oq3),
            "op6" | "op6g256" | "oq6" | "oq6g256" => Some(Self::Oq6),
            "op4" | "op4-4" | "op4g256" | "op4+" | "op4-4+" | "op4-8+" | "oq4" | "oq4+"
            | "oq4++" | "oq4g256" => Some(Self::Oq4),
            "opplus" | "op4-plus" => Some(Self::OqPlus),
            "op4+t" | "opplus-tiered" | "op4-tiered" => Some(Self::OqPlusTiered),
            "op4+c" | "opplus-compact" | "op4-compact" => Some(Self::OqPlusCompact),
            "op8" | "op8-16" | "op8g256" | "oq8" | "oq8g256" => Some(Self::Oq8),
            "op8+" | "op8-16+" | "op8-plus" | "oq8+" | "oq8++" | "oq8-plus" => Some(Self::Oq8Plus),
            _ => None,
        }
    }
}

fn hfq_source_dtype(qt: u8) -> Option<&'static str> {
    match qt {
        1 => Some("F16"),
        2 => Some("F32"),
        16 => Some("BF16"),
        _ => None,
    }
}

fn hfq_source_to_f32(name: &str, qt: u8, raw: &[u8]) -> Result<Vec<f32>, String> {
    match qt {
        1 => Ok(to_f32(raw, "F16")),
        2 => Ok(to_f32(raw, "F32")),
        16 => Ok(to_f32(raw, "BF16")),
        other => Err(format!(
            "HFQ input tensor '{name}' has quant_type={other}; only source-precision HFQ tensors are supported as quantizer input today (F16=1, F32=2, BF16=16)"
        )),
    }
}

fn quantize_hfq_source_tensor(
    name: &str,
    raw: &[u8],
    src_qt: u8,
    shape: &[u32],
    format: HfqInputFormat,
) -> Result<(Vec<u8>, QuantType, u32, &'static str), String> {
    let src_dtype = hfq_source_dtype(src_qt).ok_or_else(|| {
        format!(
            "HFQ input tensor '{name}' has quant_type={src_qt}; only source-precision HFQ tensors are supported as quantizer input today (F16=1, F32=2, BF16=16)"
        )
    })?;
    let f32_data = hfq_source_to_f32(name, src_qt, raw)?;

    if format == HfqInputFormat::Bf16 {
        let (data, qt, label) = source_precision_tensor_bytes(raw, src_dtype, &f32_data);
        return Ok((data, qt, 0, label));
    }
    if format == HfqInputFormat::F16 {
        let data = match src_dtype {
            "F16" => raw.to_vec(),
            _ => f32_slice_to_f16_bytes(&f32_data),
        };
        return Ok((data, QuantType::F16, 0, "F16"));
    }
    if matches!(format, HfqInputFormat::Qtip3 | HfqInputFormat::Qtip4) {
        // Stage for the shared real-QTIP post-pass (`pack_qtip_real_tensors`),
        // mirroring the HF/GGUF dispatch: gather / elementwise tensors the
        // trellis can't serve are finalized now; 2D weights are staged BF16 and
        // the post-pass packs k%256==0 → Qtip3G256 (and embed/lm_head → Q8F16).
        if !should_quantize(name) {
            let (data, qt, label) = source_precision_tensor_bytes(raw, src_dtype, &f32_data);
            return Ok((data, qt, 0, label));
        }
        let is_moe_router =
            name.ends_with("mlp.gate.weight") || name.ends_with("mlp.shared_expert_gate.weight");
        if is_moe_router || is_conv1d_tensor(name) || shape.len() != 2 {
            return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
        }
        return Ok((
            f32_slice_to_bf16_bytes(&f32_data),
            QuantType::BF16,
            0,
            "BF16",
        ));
    }
    if !should_quantize(name) {
        // Preserve source precision for non-quantizable tensors (norms, decay
        // scalars, etc.) — BF16 source should not be silently downcast to F16.
        let (data, qt, label) = source_precision_tensor_bytes(raw, src_dtype, &f32_data);
        return Ok((data, qt, 0, label));
    }
    // `embed_tokens` (llama/qwen/…) and `backbone.embeddings.weight` (nemotron_h)
    // are both embedding tables — keep them Q8 (row-lookup-able; Q4 is too lossy).
    let is_embed = name.contains("embed_tokens") || name.ends_with("embeddings.weight");
    let is_moe_router =
        name.ends_with("mlp.gate.weight") || name.ends_with("mlp.shared_expert_gate.weight");
    if is_embed || is_moe_router || is_conv1d_tensor(name) || shape.len() != 2 {
        return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
    }

    let k = shape[1] as usize;
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let out = match format {
        HfqInputFormat::Q8F16 => (quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"),
        HfqInputFormat::Hfq4 => {
            if k % 256 == 0 {
                (
                    quantize_hfq4g256(&f32_data),
                    QuantType::HFQ4G256,
                    256,
                    "HFQ4G256",
                )
            } else if k % 128 == 0 {
                (
                    quantize_hfq4g128(&f32_data),
                    QuantType::HFQ4G128,
                    128,
                    "HFQ4G128",
                )
            } else {
                // k divides neither 256 nor 128 (e.g. nemotron_h hidden=3136 =
                // 64·49) → no valid HFQ4 grouping; fall back to Q8 (group 32).
                (quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16")
            }
        }
        HfqInputFormat::Hfq6 => {
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            (
                quantize_hfq6g256(&f32_data),
                QuantType::HFQ6G256,
                256,
                "HFQ6G256",
            )
        }
        HfqInputFormat::Mq4 => {
            if is_nemotron_h_mq4_q8_protected(name) {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            if k % 256 != 0 {
                // MQ4G256 needs k%256==0. Sub-256 columns fall back to HFQ4G128
                // (k%128==0) or, when k divides neither (nemotron_h hidden=3136),
                // to Q8 (group 32) — HFQ4G128 on k%128!=0 emits garbage.
                return Ok(if k % 128 == 0 {
                    (
                        quantize_hfq4g128(&f32_data),
                        QuantType::HFQ4G128,
                        128,
                        "HFQ4G128",
                    )
                } else {
                    (quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16")
                });
            }
            (
                quantize_mq4g256(&f32_data, &signs1, &signs2),
                QuantType::MQ4G256,
                256,
                "MQ4G256",
            )
        }
        HfqInputFormat::Mq6 => {
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            (
                quantize_mq6g256(&f32_data, &signs1, &signs2),
                QuantType::MQ6G256,
                256,
                "MQ6G256",
            )
        }
        HfqInputFormat::Mq3 => {
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            (
                quantize_mq3g256(&f32_data, &signs1, &signs2),
                QuantType::MQ3G256,
                256,
                "MQ3G256",
            )
        }
        HfqInputFormat::Oq6 => {
            // Opus Quant W6 — symmetric signed-int6, FWHT-256, 6-bit packed (Oq6G256,
            // 194 B/group, the near-lossless mid-tier between Oq4 and Oq8). Requires
            // 256-aligned K; ragged dims fall back to Q8. Family completion; LDLQ + the
            // iu8 W6A8 loader/kernel are pending.
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            (
                quantize_oq6g256(&f32_data, &signs1, &signs2),
                QuantType::Oq6G256,
                256,
                "OQ6G256",
            )
        }
        HfqInputFormat::Oq3 => {
            // Opus Quant W3A4 — symmetric signed-int3, FWHT-256, bit-plane packed
            // (Oq3G256, 98 B/group). Requires 256-aligned K; ragged dims fall back to
            // Q8. Shares the Oq4 calibration machinery: `--ldlq` (full-Hessian OBS
            // error-feedback, `oq3++`) takes precedence, else `--awq` SmoothQuant
            // (`oq3+`), else plain RTN+FWHT (`oq3`). At 3 bits the ±3 grid is coarse,
            // so error feedback matters MORE than at int4. 3-bit is meant to ride the
            // SpinQuant learned rotation — see the W3A4 memory note.
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            let m_dim = shape[0] as usize;
            let ldlq_q = OQ4_LDLQ_HESSIAN.get().and_then(|idx| {
                let mut h = ldlq_hessian_for_tensor(idx, name, k)?;
                // Optional AWQ composition (same as oq4): fold W·diag(s) and rebase
                // H' = diag(1/s) H diag(1/s) so OBS minimizes the SMOOTHED output error.
                let awq_scales = awq_scales_for(name);
                let wbuf: std::borrow::Cow<[f32]> = if let Some(s) = &awq_scales {
                    let mut scaled = f32_data.clone();
                    awq_pre_scale_weights(&mut scaled, m_dim, k, s);
                    for i in 0..k {
                        let si = s[i] as f64;
                        for j in 0..k {
                            h[i * k + j] = (h[i * k + j] as f64 / (si * s[j] as f64)) as f32;
                        }
                    }
                    std::borrow::Cow::Owned(scaled)
                } else {
                    std::borrow::Cow::Borrowed(&f32_data[..])
                };
                let diag_sum: f64 = (0..k).map(|i| h[i * k + i] as f64).sum();
                let damp = 0.01 * (diag_sum / k as f64).max(1e-12);
                let out = ldlq::oq3_ldlq_pack(&wbuf, m_dim, k, &h, &signs1, &signs2, damp);
                if out.is_some() {
                    ldlq_record_success();
                    if let Some(s) = awq_scales {
                        OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(s));
                        eprintln!("  ldlq+awq: {name} [{m_dim}x{k}] OBS int3 + smooth");
                    } else {
                        eprintln!("  ldlq: {name} [{m_dim}x{k}] OBS error-feedback int3");
                    }
                } else {
                    ldlq_record_pack_failed(name);
                }
                out
            });
            let q = if let Some(q) = ldlq_q {
                q
            } else if let Some(scales) = awq_scales_for(name) {
                let mut scaled = f32_data.clone();
                awq_pre_scale_weights(&mut scaled, m_dim, k, &scales);
                OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(scales));
                quantize_oq3g256(&scaled, &signs1, &signs2)
            } else {
                quantize_oq3g256(&f32_data, &signs1, &signs2)
            };
            (q, QuantType::Oq3G256, 256, "OQ3G256")
        }
        HfqInputFormat::Oq4 | HfqInputFormat::OqPlus => {
            // Opus Quant W4A4 (Oq4) / OQ+ Opus Plus W4A8 (OqPlus). IDENTICAL weight
            // quantization — symmetric signed-int4, FWHT-256, clip-search, plus the
            // shared LDLQ/AWQ calibration below — producing the same packed bytes.
            // The two formats differ ONLY in the runtime contract: Oq4 → qt=34
            // (int4 activations, iu4 path); OQ+ → qt=33 (loader nibble-expands to
            // int8, int8 activations, iu8 W8A8 path). See QuantType::OqPlusG256.
            // Requires 256-aligned K (FWHT-256); ragged dims fall
            // back to Q8. Loader is qwen35 qt=34; forward int4-quantizes activations.
            // SmoothQuant/AWQ: when --awq + an imatrix (e.g. via --hessian) are
            // present and the tensor is awq_eligible, fold W·s offline (in the
            // UNROTATED basis, before the codec's FWHT) and stash s for the
            // `<weight>.awq_scale.weight` sidecar — the runtime divides x/s before
            // its FWHT+int4-quant, completing (W·s)·(x/s)=W·x. This is the dominant
            // W4A4 quality lever (migrates activation outliers into the weight).
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            let m_dim = shape[0] as usize;
            // `--ldlq`: full-Hessian error-feedback weight quant takes precedence
            // over AWQ. Uses the SAME packed oq4 layout (so the loader/runtime are
            // unchanged), but compensates each column's int4 error against the
            // off-diagonal Hessian instead of RTN. No AWQ sidecar in this path.
            let ldlq_q = OQ4_LDLQ_HESSIAN.get().and_then(|idx| {
                let mut h = ldlq_hessian_for_tensor(idx, name, k)?;
                // Optional AWQ composition: when --awq is also active, smooth the
                // activation outliers into the weights (W·diag(s)) AND rebase the
                // Hessian into the smoothed input space H' = diag(1/s) H diag(1/s),
                // so the OBS feedback minimizes the SMOOTHED output error. Runtime
                // divides x/s via the awq_scale sidecar → (W·s)·(x/s) = W·x.
                let awq_scales = awq_scales_for(name);
                let wbuf: std::borrow::Cow<[f32]> = if let Some(s) = &awq_scales {
                    let mut scaled = f32_data.clone();
                    awq_pre_scale_weights(&mut scaled, m_dim, k, s);
                    for i in 0..k {
                        let si = s[i] as f64;
                        for j in 0..k {
                            h[i * k + j] = (h[i * k + j] as f64 / (si * s[j] as f64)) as f32;
                        }
                    }
                    std::borrow::Cow::Owned(scaled)
                } else {
                    std::borrow::Cow::Borrowed(&f32_data[..])
                };
                let diag_sum: f64 = (0..k).map(|i| h[i * k + i] as f64).sum();
                let damp = 0.01 * (diag_sum / k as f64).max(1e-12);
                let out = ldlq::oq4_ldlq_pack(&wbuf, m_dim, k, &h, &signs1, &signs2, damp);
                if let Some(_) = &out {
                    ldlq_record_success();
                    if let Some(s) = awq_scales {
                        OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(s));
                        eprintln!("  ldlq+awq: {name} [{m_dim}x{k}] OBS int4 + smooth");
                    } else {
                        eprintln!("  ldlq: {name} [{m_dim}x{k}] OBS error-feedback int4");
                    }
                } else {
                    ldlq_record_pack_failed(name);
                }
                out
            });
            let q = if let Some(q) = ldlq_q {
                q
            } else if let Some(scales) = awq_scales_for(name) {
                let mut scaled = f32_data.clone();
                awq_pre_scale_weights(&mut scaled, m_dim, k, &scales);
                OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(scales));
                quantize_oq4g256(&scaled, &signs1, &signs2)
            } else {
                quantize_oq4g256(&f32_data, &signs1, &signs2)
            };
            // Same packed int4 bytes; the format tag selects W4A4 vs W4A8 dispatch.
            match format {
                HfqInputFormat::OqPlus => (q, QuantType::OqPlusG256, 256, "OQPLUS"),
                _ => (q, QuantType::Oq4G256, 256, "OQ4G256"),
            }
        }
        HfqInputFormat::Oq8 | HfqInputFormat::Oq8Plus => {
            // Opus Quant W8A8. Plain OQ8 uses RTN; OQ8+ adds AWQ smoothing and
            // OQ8++ adds full-Hessian LDLQ into the same Oq8G256 wire
            // format/runtime tag. Requires 256-aligned K (FWHT-256); ragged dims
            // fall back to Q8. Loader is qwen35 qt=35; forward int8-quantizes
            // activations and runs the iu8 GEMM.
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            let m_dim = shape[0] as usize;
            let ldlq_q = if format == HfqInputFormat::Oq8Plus {
                OQ4_LDLQ_HESSIAN.get().and_then(|idx| {
                    let mut h = ldlq_hessian_for_tensor(idx, name, k)?;
                    let awq_scales = awq_scales_for(name);
                    let wbuf: std::borrow::Cow<[f32]> = if let Some(s) = &awq_scales {
                        let mut scaled = f32_data.clone();
                        awq_pre_scale_weights(&mut scaled, m_dim, k, s);
                        for i in 0..k {
                            let si = s[i] as f64;
                            for j in 0..k {
                                h[i * k + j] = (h[i * k + j] as f64 / (si * s[j] as f64)) as f32;
                            }
                        }
                        std::borrow::Cow::Owned(scaled)
                    } else {
                        std::borrow::Cow::Borrowed(&f32_data[..])
                    };
                    let diag_sum: f64 = (0..k).map(|i| h[i * k + i] as f64).sum();
                    let damp = 0.01 * (diag_sum / k as f64).max(1e-12);
                    let out = ldlq::oq8_ldlq_pack(&wbuf, m_dim, k, &h, &signs1, &signs2, damp);
                    if out.is_some() {
                        ldlq_record_success();
                        if let Some(s) = awq_scales {
                            OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(s));
                            eprintln!("  ldlq+awq: {name} [{m_dim}x{k}] OBS int8 + smooth");
                        } else {
                            eprintln!("  ldlq: {name} [{m_dim}x{k}] OBS error-feedback int8");
                        }
                    } else {
                        ldlq_record_pack_failed(name);
                    }
                    out
                })
            } else {
                None
            };
            let q = if let Some(q) = ldlq_q {
                q
            } else if let Some(scales) = awq_scales_for(name) {
                let mut scaled = f32_data.clone();
                awq_pre_scale_weights(&mut scaled, m_dim, k, &scales);
                OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(scales));
                quantize_oq8g256(&scaled, &signs1, &signs2)
            } else {
                quantize_oq8g256(&f32_data, &signs1, &signs2)
            };
            (q, QuantType::Oq8G256, 256, "OQ8G256")
        }
        HfqInputFormat::OqPlusTiered | HfqInputFormat::OqPlusCompact => {
            // OQ+ magnitude-tiered W4A8: bulk int4, top-`w8_frac` weights/group at
            // int8 — ONE iu8 grouped-WMMA kernel. `OqPlusTiered` stores the int8
            // Oq8 layout (258 B/group, qt=35 loader); `OqPlusCompact` stores the
            // compact ~4 b/w layout (130 + 2·N_out B/group, qt=36 loader). Same
            // tiered VALUES either way. Composes AWQ + LDLQ like the Oq4 arm.
            let compact = matches!(format, HfqInputFormat::OqPlusCompact);
            if k % 256 != 0 {
                return Ok((quantize_q8f16(&f32_data), QuantType::Q8F16, 32, "Q8_F16"));
            }
            let m_dim = shape[0] as usize;
            let w8_frac = OQPLUS_W8_FRAC.get().copied().unwrap_or(0.01);
            // `--ldlq`: tiered GPTQ/OBS error-feedback (Hessian) → tiered int8
            // layout. Composes AWQ exactly like the Oq4 LDLQ arm (W·s offline +
            // rebase H' = diag(1/s) H diag(1/s) + x/s sidecar).
            let ldlq_q = OQ4_LDLQ_HESSIAN.get().and_then(|idx| {
                let mut h = ldlq_hessian_for_tensor(idx, name, k)?;
                let awq_scales = awq_scales_for(name);
                let wbuf: std::borrow::Cow<[f32]> = if let Some(s) = &awq_scales {
                    let mut scaled = f32_data.clone();
                    awq_pre_scale_weights(&mut scaled, m_dim, k, s);
                    for i in 0..k {
                        let si = s[i] as f64;
                        for j in 0..k {
                            h[i * k + j] = (h[i * k + j] as f64 / (si * s[j] as f64)) as f32;
                        }
                    }
                    std::borrow::Cow::Owned(scaled)
                } else {
                    std::borrow::Cow::Borrowed(&f32_data[..])
                };
                let diag_sum: f64 = (0..k).map(|i| h[i * k + i] as f64).sum();
                let damp = 0.01 * (diag_sum / k as f64).max(1e-12);
                let out = if compact {
                    ldlq::oqplus_compact_ldlq_pack(
                        &wbuf, m_dim, k, &h, &signs1, &signs2, damp, w8_frac,
                    )
                } else {
                    ldlq::oqplus_tiered_ldlq_pack(
                        &wbuf, m_dim, k, &h, &signs1, &signs2, damp, w8_frac,
                    )
                };
                if out.is_some() {
                    ldlq_record_success();
                    if let Some(s) = awq_scales {
                        OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(s));
                        eprintln!("  ldlq+awq: {name} [{m_dim}x{k}] tiered OBS int4/int8 + smooth");
                    } else {
                        eprintln!("  ldlq: {name} [{m_dim}x{k}] tiered OBS int4/int8");
                    }
                } else {
                    ldlq_record_pack_failed(name);
                }
                out
            });
            let q = if let Some(q) = ldlq_q {
                q
            } else {
                let scaled = awq_scales_for(name).map(|scales| {
                    let mut scaled = f32_data.clone();
                    awq_pre_scale_weights(&mut scaled, m_dim, k, &scales);
                    OQ4_AWQ_SIDECAR.with(|c| *c.borrow_mut() = Some(scales));
                    scaled
                });
                let w: &[f32] = scaled.as_deref().unwrap_or(&f32_data);
                if compact {
                    quantize_oqplus_compact(w, &signs1, &signs2, w8_frac)
                } else {
                    quantize_oqplus_tiered(w, &signs1, &signs2, w8_frac)
                }
            };
            // OQ+C → compact qt=36 layout; OQ+T → int8 Oq8 qt=35 layout.
            if compact {
                (q, QuantType::OqPlusCompact, 256, "OQ+C")
            } else {
                (q, QuantType::Oq8G256, 256, "OQ+T")
            }
        }
        HfqInputFormat::F16
        | HfqInputFormat::Bf16
        | HfqInputFormat::Qtip3
        | HfqInputFormat::Qtip4 => unreachable!(),
    };
    Ok(out)
}

/// QTIP beam-search width for the real trellis encoder. Default 128 (near-
/// Viterbi); override with `--beam N` / `HIPFIRE_QTIP_BEAM=N` to trade a little
/// quality for a large encode speedup on big models (the beam search is the
/// offline bottleneck). `--beam` sets this env in `main`, so both the
/// `.hfq`-requantize and direct-source paths read it here.
fn qtip_beam_width() -> usize {
    std::env::var("HIPFIRE_QTIP_BEAM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(128)
}

/// Pack the real QTIP format (`bits`-bit trellis) over a set of staged BF16
/// tensors, in place.
///
/// Shared by the HF/GGUF dispatch and the `.hfq`-source requantization path so
/// `--format qtip3` / `--format qtip4` produce byte-identical artifacts
/// regardless of input source. Expects 2D weight tensors staged as
/// `QuantType::BF16`:
///   - tied embed / lm_head / output → Q8F16 (gather-friendly, the trellis
///     format can't be random-accessed by an embedding lookup),
///   - every other 2D BF16 tensor with `k % 256 == 0` → Qtip3G256 (3-bit, 100
///     B/group) or Qtip4G256 (4-bit, 132 B/group) rotated-frame trellis symbols
///     + scale.
/// All other tensors (norms, 1D scalars, ragged dims) are left untouched.
/// `bits` is 3 or 4; the trellis codebook/state is bit-agnostic, only the
/// per-symbol width and the on-disk packing change. `beam` is the beam-search
/// width — 128 is near-Viterbi; smaller trades a little quality for a large
/// encode speedup (the beam search is the offline bottleneck on big models).
fn pack_qtip_real_tensors(
    tensors: &mut Vec<HfqTensor>,
    qtip_cb: &[f32],
    qtip_s1: &[f32],
    qtip_s2: &[f32],
    lowrank_r: usize,
    bits: u32,
    beam: usize,
) {
    assert!(bits == 3 || bits == 4, "QTIP real pack supports 3 or 4 bits");
    let block_bytes = if bits == 4 {
        qtip::QTIP4_BLOCK_BYTES
    } else {
        qtip::QTIP3_BLOCK_BYTES
    };
    let tag = if bits == 4 {
        QuantType::Qtip4G256
    } else {
        QuantType::Qtip3G256
    };
    use rayon::prelude::*;
    // The tied embed/lm_head ([vocab × dim]) is gather-accessed (embedding
    // lookup), which the trellis format can't random-access — and it is the
    // single largest tensor, read every decode token via lm_head. Leaving it
    // bf16 erased the transformer-weight savings (measured: qtip3 40.9 tok/s
    // vs mq4 57.4 with bf16 lm_head). Quantize it Q8F16 (gather-friendly,
    // 1 B/w), matching what the mq4 path does for tied embed/lm_head.
    let mut n_q8 = 0usize;
    for t in tensors.iter_mut() {
        if !(matches!(t.quant_type, QuantType::BF16)
            && t.shape.len() == 2
            && (t.name.contains("embed")
                || t.name.contains("lm_head")
                || t.name.ends_with("output.weight")))
        {
            continue;
        }
        let wf: Vec<f32> = t
            .data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        t.data = quantize_q8f16(&wf);
        t.quant_type = QuantType::Q8F16;
        t.group_size = 32;
        n_q8 += 1;
    }
    if n_q8 > 0 {
        eprintln!("  qtip{bits} (real): embed/lm_head → Q8F16 ({n_q8} tensors, gather-friendly)");
    }
    let (mut n_packed, mut max_err) = (0usize, 0.0f32);
    // Optional LQER low-rank correction: when `lowrank_r > 0`, each packed tensor
    // also emits a rank-r factorization (U,V) of its ROTATED-frame quant error
    // E_rot = W_rot − dequant(Q(W_rot)) as `<base>.lr_u/.lr_v` sidecars. The
    // runtime adds U·(V·x_rot) after the trellis GEMV; since x_rot = FWHT(x), this
    // recovers W_rot·x_rot = W·x. Sidecars are staged here and appended after the
    // packing loop (can't push while iterating).
    let want_lr = lowrank_r > 0;
    let mut new_sidecars: Vec<HfqTensor> = Vec::new();
    let mut n_lr = 0usize;
    for t in tensors.iter_mut() {
        if !(matches!(t.quant_type, QuantType::BF16)
            && t.shape.len() == 2
            && (t.shape[1] as usize) % 256 == 0
            && !t.name.contains("embed")
            && !t.name.contains("lm_head"))
        {
            continue;
        }
        let m = t.shape[0] as usize;
        let k = t.shape[1] as usize;
        let groups = k / 256;
        let stem = t
            .name
            .strip_suffix(".weight")
            .unwrap_or(&t.name)
            .to_string();
        let wf: Vec<f32> = t
            .data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        // Per row → packed records; collect per-row max-error (and, when low-rank
        // is enabled, the per-row rotated-frame error E_rot) in parallel.
        let row_out: Vec<(Vec<u8>, f32, Vec<f32>)> = wf
            .par_chunks(k)
            .map(|row| {
                let mut packed = Vec::with_capacity(groups * block_bytes);
                let mut rerr = 0.0f32;
                let mut erow = if want_lr { vec![0.0f32; k] } else { Vec::new() };
                for g in 0..groups {
                    let mut grp = [0.0f32; 256];
                    grp.copy_from_slice(&row[g * 256..g * 256 + 256]);
                    cpu_fwht_256(&mut grp, qtip_s1, qtip_s2); // rotate
                    let scale0 = qtip::group_scale(&grp);
                    let sym = qtip::beam_encode_group_bits(&grp, scale0, qtip_cb, beam, bits);
                    let scale = qtip::optimal_scale_bits(&grp, &sym, qtip_cb, bits);
                    // Self-check: decode in the rotated frame vs the encode target.
                    let deq = qtip::decode_group_bits(&sym, scale, qtip_cb, bits);
                    for (idx, (a, b)) in grp.iter().zip(&deq).enumerate() {
                        let d = a - b;
                        rerr = rerr.max(d.abs());
                        if want_lr {
                            erow[g * 256 + idx] = d;
                        }
                    }
                    let group_bytes = if bits == 4 {
                        qtip::pack_qtip4_group(&sym, scale)
                    } else {
                        qtip::pack_qtip3_group(&sym, scale)
                    };
                    packed.extend_from_slice(&group_bytes);
                }
                (packed, rerr, erow)
            })
            .collect();
        let mut data = Vec::with_capacity(row_out.len() * groups * block_bytes);
        let mut e_rot = if want_lr {
            Vec::with_capacity(m * k)
        } else {
            Vec::new()
        };
        for (packed, rerr, erow) in &row_out {
            data.extend_from_slice(packed);
            max_err = max_err.max(*rerr);
            if want_lr {
                e_rot.extend_from_slice(erow);
            }
        }
        t.data = data;
        t.quant_type = tag;
        t.group_size = 256;
        n_packed += 1;

        if want_lr {
            let (u, v) = randomized_lowrank_factors(&e_rot, m, k, lowrank_r);
            let rr = lowrank_r.min(m).min(k);
            let to_le = |f: &[f32]| -> Vec<u8> { f.iter().flat_map(|x| x.to_le_bytes()).collect() };
            new_sidecars.push(HfqTensor {
                name: format!("{stem}.lr_u.weight"),
                quant_type: QuantType::F32,
                shape: vec![m as u32, rr as u32],
                group_size: 0,
                data: to_le(&u),
                spilled_len: 0,
            });
            new_sidecars.push(HfqTensor {
                name: format!("{stem}.lr_v.weight"),
                quant_type: QuantType::F32,
                shape: vec![rr as u32, k as u32],
                group_size: 0,
                data: to_le(&v),
                spilled_len: 0,
            });
            n_lr += 1;
        }
    }
    if want_lr {
        tensors.extend(new_sidecars);
    }
    eprintln!(
        "  qtip{bits} (real): packed {n_packed} tensors as {tag:?} ({block_bytes} B/group, \
         {:.3} B/w); rotated-frame decode max-abs-err {max_err:.5}{}",
        block_bytes as f32 / 256.0,
        if want_lr {
            format!("; emitted rank-{lowrank_r} lr_u/lr_v for {n_lr} tensors")
        } else {
            String::new()
        }
    );
}

fn run_hfq_source_pipeline(
    input: &Path,
    output: &Path,
    format: HfqInputFormat,
    format_label: &str,
) -> Result<(), String> {
    let hfq = HfqInputFile::open(input).map_err(|e| format!("open HFQ input: {e}"))?;
    eprintln!(
        "HFQ input: arch_id={} tensors={} format={format:?}",
        hfq.arch_id,
        hfq.tensors.len()
    );

    let mut metadata: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).unwrap_or_else(|_| serde_json::json!({}));
    if let serde_json::Value::Object(ref mut map) = metadata {
        map.insert("source".to_string(), serde_json::json!("hfq"));
        map.insert(
            "source_hfq".to_string(),
            serde_json::json!({
                "path": input.display().to_string(),
                "input_arch_id": hfq.arch_id,
                "accepted_quant_types": ["F16", "F32", "BF16"],
            }),
        );
    }

    let mut hfq_tensors = Vec::with_capacity(hfq.tensors.len());
    let mut total_params = 0u64;
    let mut quantized_params = 0u64;
    for t in &hfq.tensors {
        let raw = hfq.tensor_data(t);
        let n_elements = t.shape.iter().map(|&d| d as u64).product::<u64>();
        total_params += n_elements;
        let (data, qt, group_size, label) =
            quantize_hfq_source_tensor(&t.name, raw, t.quant_type, &t.shape, format)?;
        if qt as u8 != t.quant_type || group_size != t.group_size {
            quantized_params += n_elements;
        }
        eprintln!(
            "  {label:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB)",
            t.name,
            t.shape,
            n_elements,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0
        );
        hfq_tensors.push(HfqTensor {
            name: t.name.clone(),
            quant_type: qt,
            shape: t.shape.clone(),
            group_size,
            data,
            spilled_len: 0,
        });
        // Emit the AWQ sidecar if the Oq4 arm produced SmoothQuant scales for
        // this tensor (`<weight>.awq_scale.weight`, 1D F16, length K).
        if let Some(scales) = OQ4_AWQ_SIDECAR.with(|c| c.borrow_mut().take()) {
            let sidecar_name = match t.name.strip_suffix(".weight") {
                Some(stem) => format!("{stem}.awq_scale.weight"),
                None => format!("{}.awq_scale.weight", t.name),
            };
            let bytes = awq_scales_to_f16_bytes(&scales);
            eprintln!(
                "    AWQ:    {sidecar_name} [{}] (1D F16, {} B)",
                scales.len(),
                bytes.len()
            );
            hfq_tensors.push(HfqTensor {
                name: sidecar_name,
                quant_type: QuantType::F16,
                shape: vec![scales.len() as u32],
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
        }
    }

    // Real QTIP-3 is a post-pass over the BF16-staged 2D weights, shared with
    // the HF/GGUF dispatch so a bf16 `.hfq` requantized to qtip3 is byte-
    // identical to quantizing the original safetensors with `--format qtip3`.
    if matches!(format, HfqInputFormat::Qtip3 | HfqInputFormat::Qtip4) {
        let bits = if format == HfqInputFormat::Qtip4 { 4 } else { 3 };
        let qtip_cb = qtip::build_codebook();
        let qtip_s1 = gen_fwht_signs(42, 256);
        let qtip_s2 = gen_fwht_signs(1042, 256);
        pack_qtip_real_tensors(
            &mut hfq_tensors,
            &qtip_cb,
            &qtip_s1,
            &qtip_s2,
            std::env::var("HIPFIRE_LOWRANK_R")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            bits,
            qtip_beam_width(),
        );
        // Recount rewritten params: the post-pass changes quant types after the
        // per-tensor loop above bumped the counter for the BF16 staging.
        quantized_params = hfq_tensors
            .iter()
            .filter(|t| {
                matches!(
                    t.quant_type,
                    QuantType::Qtip3G256 | QuantType::Qtip4G256 | QuantType::Q8F16
                ) && t.shape.len() == 2
            })
            .map(|t| t.shape.iter().map(|&d| d as u64).product::<u64>())
            .sum();
    }

    let total_bytes: usize = hfq_tensors.iter().map(|t| t.data.len()).sum();
    eprintln!("\n=== HFQ Input Quantization Summary ===");
    eprintln!("  Total params:     {total_params}");
    eprintln!(
        "  Rewritten params: {quantized_params} ({:.1}%)",
        if total_params > 0 {
            100.0 * quantized_params as f64 / total_params as f64
        } else {
            0.0
        }
    );
    eprintln!("  Output size:      {:.1} MB", total_bytes as f64 / 1e6);
    ldlq_report_and_validate(false)?;
    eprintln!("\nWriting: {}", output.display());
    insert_parameter_counts_metadata(
        &mut metadata,
        &hfq_tensors,
        total_params,
        quantized_params,
        0,
    );
    insert_quant_format_metadata(&mut metadata, format_label);
    let metadata_json =
        metadata_with_quantization_hash(metadata, &hfq_tensors, None).map_err(|e| e.to_string())?;
    write_hfq(output, hfq.arch_id, &metadata_json, &hfq_tensors, None)
        .map_err(|e| format!("write HFQ output: {e}"))?;
    let file_size = std::fs::metadata(output)
        .map_err(|e| format!("stat HFQ output: {e}"))?
        .len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
    Ok(())
}

/// Translate llama.cpp GGUF tensor names to the HuggingFace safetensors
/// names that `hipfire_runtime::hfq::load_weights_hfq` expects. The mapping is
/// the canonical llama.cpp ↔ HF convention.
///
/// Returns None for tensors that don't have a known safetensors equivalent
/// (we then keep them under their GGUF name; the future loader can decide
/// what to do, or they're skipped).
fn gguf_to_safetensors_name(gguf_name: &str) -> Option<String> {
    // Top-level tensors.
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        _ => {}
    }
    // Per-layer: blk.{N}.<slot>.weight  →  model.layers.{N}.<slot>.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        // rest = "{N}.<slot>.weight"
        let dot = rest.find('.')?;
        let layer_idx = &rest[..dot];
        let slot_full = &rest[dot + 1..]; // "<slot>.weight"
                                          // Drop the trailing ".weight" so we can rewrite slots like "attn_q"→"self_attn.q_proj".
        let slot = slot_full.strip_suffix(".weight")?;
        let translated = match slot {
            "attn_norm" => "input_layernorm".to_string(),
            "ffn_norm" => "post_attention_layernorm".to_string(),
            "attn_q" => "self_attn.q_proj".to_string(),
            "attn_k" => "self_attn.k_proj".to_string(),
            "attn_v" => "self_attn.v_proj".to_string(),
            "attn_output" => "self_attn.o_proj".to_string(),
            "attn_q_norm" => "self_attn.q_norm".to_string(),
            "attn_k_norm" => "self_attn.k_norm".to_string(),
            "ffn_gate" => "mlp.gate_proj".to_string(),
            "ffn_up" => "mlp.up_proj".to_string(),
            "ffn_down" => "mlp.down_proj".to_string(),
            other => return Some(format!("model.layers.{layer_idx}.{other}.weight")),
        };
        return Some(format!("model.layers.{layer_idx}.{translated}.weight"));
    }
    None
}

/// True if the GGUF tensor's name is a 1D norm / RMSNorm scaling vector.
/// These stay F16 in the .hfq (no benefit from quantization, precision-sensitive).
fn gguf_is_norm_tensor(name: &str) -> bool {
    name.contains("_norm") || name.contains("norm.weight")
}

/// Translate a hipfire safetensors-style tensor name to the ggml-style name
/// used by llama.cpp's imatrix output (and the rest of llama.cpp's tooling).
///
/// Verified by shape-alignment on Qwen3.5-0.8B imatrix vs safetensors load log
/// (2026-05-11):
///   - K dims match for every covered tensor class (mlp.* , self_attn.* ,
///     linear_attn.in_proj_qkv/z/a/b, linear_attn.out_proj).
///   - Layer-pattern: FullAttention layers (3, 7, 11, ...) carry standard
///     `attn_q/k/v/output`; LinearAttention layers carry `attn_qkv`/
///     `attn_gate`/`ssm_alpha`/`ssm_beta`/`ssm_out` — the SSM-naming
///     convention llama.cpp uses for Mamba-style sub-blocks.
///
/// Returns `None` for tensors that don't have an imatrix counterpart
/// (norms / biases / 1D scalars / lookup-only tables). Those fall back to
/// non-imatrix-weighted quantization in the call site.
fn safetensors_to_ggml_name(name: &str) -> Option<String> {
    // Drop the architecture-specific "language_model." prefix (Qwen3.5
    // structure has model.language_model.layers.{N}.* — the linear-attn
    // crate uses this nested layout, llama.cpp flattens to blk.{N}.*).
    let normalized = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    // Top-level (currently no imatrix coverage; default is --process-output OFF).
    match normalized {
        "embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        "norm.weight" => return Some("output_norm.weight".to_string()),
        _ => {}
    }

    // Per-layer: "layers.{N}.<slot>.weight"
    let rest = normalized.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer_idx = &rest[..dot];
    let slot_full = &rest[dot + 1..];
    let slot = slot_full.strip_suffix(".weight")?;

    let translated = match slot {
        // MLP — present on every layer.
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        // FullAttention layer tensors (standard names).
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        // LinearAttention layer tensors (Mamba-2 / hybrid-arch SSM naming).
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.in_proj_a" => "ssm_alpha",
        "linear_attn.in_proj_b" => "ssm_beta",
        "linear_attn.out_proj" => "ssm_out",
        // Unmapped: conv1d.weight (special-cased to HFQ4G128 at quantize
        // time; small, not multiplied by activation in the standard sense),
        // norm.weight, A_log, dt_bias (1D or scalars, no imatrix entry).
        _ => return None,
    };

    Some(format!("blk.{layer_idx}.{translated}.weight"))
}

/// Load an llama.cpp-compatible imatrix GGUF file and build a lookup
/// keyed by ggml-style tensor name. The GGUF stores per-linear-layer
/// pairs:
///   {name}.in_sum2     F32[k, n_mat]   sum of squared activations per channel
///   {name}.counts      F32[1, n_mat]   token count contributing per matrix
///
/// For non-MoE models n_mat=1; the [k] vector goes into the map directly.
/// For MoE we'd need per-expert handling — out of scope for Step 5a
/// (Qwen3.5 dense + Qwen3.6 dense are the first cohort targets; A3B MoE
/// is deferred to a future iteration that handles n_mat > 1).
///
/// Returns `HashMap<ggml_name, Vec<f32>>` with the .in_sum2 values keyed by
/// the BASE tensor name (the ".in_sum2" suffix stripped).
fn load_imatrix(path: &Path) -> HashMap<String, Vec<f32>> {
    use gguf_input::GgmlType;
    let gguf = gguf_input::GgufFile::open(path).unwrap_or_else(|e| {
        eprintln!("error: failed to open imatrix file {}: {e}", path.display());
        std::process::exit(1);
    });

    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut total_entries = 0usize;
    let mut skipped_moe = 0usize;
    for t in &gguf.tensors {
        let name = match t.name.strip_suffix(".in_sum2") {
            Some(n) => n.to_string(),
            None => continue, // ignore .counts and any other entries
        };
        if t.dtype != GgmlType::F32 {
            eprintln!(
                "warning: imatrix entry {} has non-F32 dtype {:?}; skipping",
                t.name, t.dtype
            );
            continue;
        }
        // Shape is [k] (1D) for non-MoE; [k, n_mat] for MoE. Skip multi-mat
        // tensors with a warning — Step 5a doesn't handle them yet.
        let n_mat = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
        if n_mat != 1 {
            skipped_moe += 1;
            continue;
        }
        let k = t.shape[0];

        // Read the F32 values from the tensor data segment.
        let data = gguf.tensor_data(t);
        let mut values = Vec::with_capacity(k);
        for i in 0..k {
            let off = i * 4;
            let v = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            values.push(v);
        }
        map.insert(name, values);
        total_entries += 1;
    }

    eprintln!(
        "imatrix: loaded {} entries from {} ({} MoE multi-matrix entries skipped — Step 5a is dense-only)",
        total_entries,
        path.display(),
        skipped_moe,
    );
    if total_entries == 0 {
        if skipped_moe > 0 {
            // MoE-only imatrix (e.g. MiniMax routed experts): no 1D dense
            // entries for the legacy dense-AWQ table, but the file IS valid.
            // The MiniMax AWQ-on-experts path reads the raw imatrix GGUF
            // (imatrix_gguf) directly, so an empty dense table is harmless —
            // dense tensors just fall back to non-imatrix quantization.
            eprintln!(
                "imatrix: 0 dense entries, {skipped_moe} MoE multi-matrix entries — \
                 dense table empty (MoE-only imatrix; expert AWQ uses the raw GGUF)"
            );
        } else {
            eprintln!("error: imatrix file contains no usable .in_sum2 entries");
            std::process::exit(1);
        }
    }
    map
}

/// Look up imatrix per-channel weights for a given safetensors tensor name.
/// Returns `None` (caller falls back to non-imatrix-weighted quantization) if:
///   - --imatrix wasn't passed (IMATRIX not initialized), OR
///   - the tensor name doesn't have a ggml-mapping (norms, small 1D, etc.), OR
///   - the imatrix file doesn't carry this tensor (rare; usually means the
///     tensor wasn't exercised by the calibration corpus).
fn imatrix_weights_for(safetensors_name: &str) -> Option<&'static [f32]> {
    let im = IMATRIX.get()?;
    // `load_imatrix` keys the map by the imatrix FILE's tensor names (`.in_sum2`
    // stripped). hipfire's `collect_imatrix` emits *safetensors* names
    // (`model.language_model.layers.N.linear_attn.in_proj_qkv.weight`), so try the
    // direct safetensors name FIRST — this was the AWQ no-op: the map is
    // safetensors-keyed but we only tried the GGML-converted name, which always
    // missed (and 27B-3.6 hybrid linear_attn names don't round-trip anyway).
    // Fall back to the GGML name for llama.cpp-style (blk.*) imatrices.
    if let Some(v) = im.get(safetensors_name) {
        return Some(v.as_slice());
    }
    let ggml_name = safetensors_to_ggml_name(safetensors_name)?;
    im.get(&ggml_name).map(|v| v.as_slice())
}

/// Compute AWQ per-channel scales `s[j]` for one linear-layer weight tensor.
///
/// Inputs:
///   - `in_sum2`: imatrix data — Σ_token act²[j] per input channel, length K.
///     Source: hipfire's `imatrix_collect` (llama.cpp `--imatrix` output).
///   - `alpha`: AWQ tuning parameter ∈ [0, 1]. Paper-original default = 0.5.
///
/// Output:
///   - `Vec<f32>` of length K, with geometric mean normalized to ≈ 1.0.
///
/// Formula (AWQ-paper-original simplified for hipfire's data shape):
///   1. RMS_act[j] = sqrt(in_sum2[j] / N_tok). The N_tok term is a global
///      constant for the tensor and gets absorbed by the geo-mean normalization
///      below, so we can omit it from the per-channel computation.
///      Equivalent: use sqrt(in_sum2[j]) directly.
///   2. s_raw[j] = (RMS_act[j])^alpha
///   3. Normalize: s[j] = s_raw[j] / exp(mean_j log(s_raw[j]))
///      This keeps the post-AWQ-scaled weight tensor's overall magnitude
///      in the same range as the input — important for the downstream MQ4
///      min-max scale fitter not to suddenly compress/expand its dynamic
///      range based on alpha.
///
/// Edge cases:
///   - Zero in_sum2[j] (channel never exercised by calibration): clamp to
///     a tiny floor (1e-12) before sqrt to avoid log(0). Practically rare;
///     would mean a channel is unused in the calibration corpus.
///   - alpha == 0 → all s[j] = 1.0 (AWQ disabled at this layer). Caller
///     can short-circuit before invoking this function.
///
/// Cost: O(K). For 9B Qwen3.5 ~32 calls × ~4096 elements = ~131K ops total
/// across the whole quantize. Negligible.
/// Parse the layer index N from a MiniMax expert tensor name
/// `…layers.N.block_sparse_moe.experts.E.wX.weight`.
fn minimax_layer_index(name: &str) -> Option<usize> {
    let after = name.split(".layers.").nth(1)?;
    after.split('.').next()?.parse::<usize>().ok()
}

/// True if layer `l` falls in the comma-separated range list held in env `var`
/// (e.g. "12-45,50,55-60"; inclusive ranges or bare singles). Unset/empty →
/// false. Drives per-layer mixed-precision expert promotion for MiniMax.
fn minimax_layer_in_env_set(var: &str, l: usize) -> bool {
    let spec = match std::env::var(var) {
        Ok(v) => v,
        Err(_) => return false,
    };
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((a, b)) = tok.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                if l >= a.min(b) && l <= a.max(b) {
                    return true;
                }
            }
        } else if let Ok(n) = tok.parse::<usize>() {
            if l == n {
                return true;
            }
        }
    }
    false
}

/// Shared-per-layer AWQ scales for MiniMax routed experts from an imatrix GGUF.
/// Aggregates per-expert activation energy (in_sum2) across ALL experts of
/// layer `n` into one shared per-input-channel scale: gate(w1)/up(w3) share the
/// MoE-input channels (s_gate_up, len hidden); down(w2) uses the intermediate
/// channels (s_down, len inter). The forward applies these via experts[0], so
/// one scale per layer is exactly what the runtime consumes. None if absent.
fn minimax_layer_awq_scales(
    gguf: &gguf_input::GgufFile,
    n: usize,
    alpha: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let agg = |kind: &str| -> Option<Vec<f32>> {
        let nm = format!("blk.{n}.ffn_{kind}_exps.weight.in_sum2");
        let t = gguf.tensors.iter().find(|t| t.name == nm)?;
        if t.shape.len() != 2 {
            return None;
        }
        let k = t.shape[0];
        let n_exp = t.shape[1];
        let flat: Vec<f32> = gguf
            .tensor_data(t)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if flat.len() != k * n_exp {
            return None;
        }
        let mut a = vec![0.0f32; k];
        for e in 0..n_exp {
            let off = e * k;
            for j in 0..k {
                a[j] += flat[off + j];
            }
        }
        Some(a)
    };
    let g = agg("gate")?;
    let gu: Vec<f32> = match agg("up") {
        Some(u) if u.len() == g.len() => g.iter().zip(&u).map(|(a, b)| a + b).collect(),
        _ => g.clone(),
    };
    let d = agg("down")?;
    Some((
        compute_awq_scales(&gu, alpha),
        compute_awq_scales(&d, alpha),
    ))
}

/// Shared-per-layer AWQ scales for LFM2 routed experts from native HFQM
/// `<name>.imatrix` vectors. LFM2 captures per selected expert/source tensor
/// (`...experts.{e}.w1|w2|w3`); the runtime consumes one gate/up scale and one
/// down scale per layer via expert 0, so aggregate the per-expert activation
/// energy before computing scales.
fn lfm2_layer_awq_scales_from_imatrix(n: usize, alpha: f32) -> Option<(Vec<f32>, Vec<f32>)> {
    let im = IMATRIX.get()?;
    let prefix = format!("model.layers.{n}.feed_forward.experts.");
    let mut gate_up: Option<Vec<f32>> = None;
    let mut down: Option<Vec<f32>> = None;
    let mut n_gate_up = 0usize;
    let mut n_down = 0usize;

    for (name, values) in im {
        if name.ends_with(".weight") || !name.starts_with(&prefix) {
            continue;
        }
        let target = if name.ends_with(".w1") || name.ends_with(".w3") {
            &mut gate_up
        } else if name.ends_with(".w2") {
            &mut down
        } else {
            continue;
        };
        match target {
            Some(acc) if acc.len() == values.len() => {
                for (a, b) in acc.iter_mut().zip(values) {
                    *a += *b;
                }
            }
            Some(_) => {
                eprintln!(
                    "  lfm2 AWQ L{n}: skipped imatrix {name} with mismatched K={}",
                    values.len()
                );
                continue;
            }
            slot @ None => {
                *slot = Some(values.clone());
            }
        }
        if name.ends_with(".w2") {
            n_down += 1;
        } else {
            n_gate_up += 1;
        }
    }

    let gate_up = gate_up?;
    let down = down?;
    eprintln!("  lfm2 AWQ L{n}: aggregated {n_gate_up} gate/up + {n_down} down imatrix vectors");
    Some((
        compute_awq_scales(&gate_up, alpha),
        compute_awq_scales(&down, alpha),
    ))
}

fn compute_awq_scales(in_sum2: &[f32], alpha: f32) -> Vec<f32> {
    let k = in_sum2.len();
    debug_assert!(k > 0, "empty imatrix vector");

    // Step 1+2: RMS_act^alpha, with the constant N_tok factor absorbed into
    // the geo-mean normalization. The sqrt and (·)^alpha combine into
    // (·)^(alpha/2) on the raw in_sum2 values.
    //
    // Implementation choice: compute log(s_raw) directly so we can do the
    // geo-mean normalization in log space (numerically more stable for
    // wide dynamic-range imatrix values).
    let half_alpha = (alpha as f64) * 0.5;
    let mut log_s_raw = Vec::with_capacity(k);
    let mut energy = Vec::with_capacity(k); // clamped in_sum2, for outlier ranking
    for &v in in_sum2 {
        // Floor dead channels to 1e-12 (NaN also maps here: f64::max returns the
        // non-NaN arg) AND cap non-finite / pathologically-large values to a
        // finite ceiling. An inf in_sum2 — f32 overflow during imatrix
        // collection, which the 27B tier1 imatrix actually contains — would
        // otherwise make this tensor's `mean_log = inf`, and then `l - mean_log`
        // = inf - inf = NaN for the inf channel. That NaN survives the output
        // clamp below (f32::clamp propagates NaN), poisoning the F16 sidecar and
        // NaN'ing the whole forward (37747 such values measured pre-fix).
        // Capping the input keeps mean_log finite; the output clamp then bounds
        // the final scale. 1e30 is well inside f64 range (ln ≈ 69).
        let v_clamped = (v as f64).max(1e-12).min(1e30);
        log_s_raw.push(half_alpha * v_clamped.ln()); // log(v^(alpha/2)) = (alpha/2)·log(v)
        energy.push(v_clamped);
    }

    // Per-channel log-space normalization offset. Default: ONE geo-mean over all
    // K channels (subtracting it makes geo_mean(s) = 1 exactly). With --sq-split
    // (SQ_OUTLIER_SPLIT): the top-`frac` channels by activation energy (outliers)
    // and the remaining bulk are geo-mean-normalized SEPARATELY — each group gets
    // its OWN mean_log, so the bulk's migration isn't skewed by the outliers' huge
    // energy. The split changes only the s VALUES, not the per-channel cancellation
    // (W·s)·(x/s)=W·x; each group's geo-mean stays 1, so overall weight magnitude
    // is preserved for the downstream int4 scale fitter.
    let offset: Vec<f64> = match SQ_OUTLIER_SPLIT.get().copied() {
        Some(frac) if frac > 0.0 && frac < 1.0 && k >= 2 => {
            let n_out = (((frac as f64) * k as f64).round() as usize).clamp(1, k - 1);
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_unstable_by(|&a, &b| {
                energy[b]
                    .partial_cmp(&energy[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut is_outlier = vec![false; k];
            for &i in &order[..n_out] {
                is_outlier[i] = true;
            }
            let (mut sum_o, mut sum_r) = (0.0f64, 0.0f64);
            for j in 0..k {
                if is_outlier[j] {
                    sum_o += log_s_raw[j];
                } else {
                    sum_r += log_s_raw[j];
                }
            }
            let mean_o = sum_o / n_out as f64;
            let mean_r = sum_r / (k - n_out) as f64;
            (0..k)
                .map(|j| if is_outlier[j] { mean_o } else { mean_r })
                .collect()
        }
        _ => {
            let mean_log = log_s_raw.iter().sum::<f64>() / k as f64;
            vec![mean_log; k]
        }
    };

    // Subtract the (per-group) mean in log space, then exp back.
    //
    // Step 4 (CRITICAL — f16 safety): clamp to an f16-representable,
    // non-exploding range. The geo-mean is 1.0 by construction, so the bulk
    // of channels sit near 1; only pathological outliers reach the rails —
    // dead channels floored to 1e-12, or hot channels with huge activation
    // sums. Without this, exp() overflows to f32 inf and/or the F16 sidecar
    // under/overflows, and the inference-time `x / awq_scale` divide produces
    // inf → NaN. (Verified via dump_awq_scales on the 27B tier1 imatrix:
    // 49293 scales underflowed to 0.0 and 37747 stored as inf/NaN pre-clamp,
    // which NaN'd the whole forward — KLD 0.0 / PPL NaN on gfx11.)
    //
    // The SAME clamped vector is used for both the weight pre-scale (W*s) and
    // the emitted sidecar (x/s at inference), so the cancellation stays exact;
    // clamping only limits how aggressively pathological channels redistribute
    // quant difficulty. Real AWQ scales live in ~[0.2, 5]; [1e-2, 1e2] keeps
    // all genuine signal while removing the representability blow-ups.
    const AWQ_SCALE_MIN: f32 = 1e-2;
    const AWQ_SCALE_MAX: f32 = 1e2;
    log_s_raw
        .into_iter()
        .zip(offset)
        .map(|(l, m)| ((l - m).exp() as f32).clamp(AWQ_SCALE_MIN, AWQ_SCALE_MAX))
        .collect()
}

/// SmoothQuant/AWQ per-input-channel scales for `name`, or `None` when the
/// active recipe does not apply AWQ to this tensor. This is the single guard
/// every weight path shares — extracted so dense projections AND routed experts
/// (`…experts.{e}.gate_up_proj.weight`) flow through ONE arch-agnostic, name-keyed
/// path instead of per-format / per-arch copies.
///
/// AWQ is on iff a `+` recipe set [`AWQ_ALPHA`] (alpha > 0), the tensor is
/// [`awq_eligible`], and a per-input-channel imatrix (`<name>.imatrix`, from
/// `--hessian`/`--imatrix`) is present. Does NOT touch the weights — the caller
/// folds `W·diag(s)` ([`awq_pre_scale_weights`]) and emits the `<name>.awq_scale`
/// sidecar; the LDLQ paths additionally rebase the Hessian by `1/(s_i·s_j)`.
fn awq_scales_for(name: &str) -> Option<Vec<f32>> {
    let alpha = AWQ_ALPHA.get().copied()?;
    if alpha <= 0.0 || !awq_eligible(name) {
        return None;
    }
    let im = imatrix_weights_for(name)?;
    Some(compute_awq_scales(im, alpha))
}

/// Apply AWQ pre-scaling to a row-major [m, k] weight tensor in place:
/// `W'[i,j] = W[i,j] * s[j]` for every (i, j).
///
/// AWQ scales are per-INPUT-channel (length K). The same s[j] vector
/// broadcasts across every output row i.
///
/// Done in-place to avoid allocating a second [m, k] buffer. The caller
/// owns the W slice and is responsible for ensuring this pre-scaling
/// happens BEFORE any subsequent transformation (e.g. FWHT rotation).
fn awq_pre_scale_weights(weights: &mut [f32], m: usize, k: usize, scales: &[f32]) {
    debug_assert_eq!(weights.len(), m * k, "weight buffer size mismatch");
    debug_assert_eq!(scales.len(), k, "AWQ scale vector must have length K");
    for r in 0..m {
        let row = &mut weights[r * k..(r + 1) * k];
        for j in 0..k {
            row[j] *= scales[j];
        }
    }
}

/// Helper: convert a `Vec<f32>` AWQ-scale vector into the F16 byte
/// payload that `HfqTensor` consumes for sidecar emission.
fn awq_scales_to_f16_bytes(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|&s| f32_to_f16(s).to_le_bytes())
        .collect()
}

/// AWQ pre-scaling is mathematically valid only for weights whose runtime
/// path applies the inverse divide-by-scale. As of F2 (2026-05-14), this
/// covers both the input-side projections (fed via the AWQ-aware variants
/// of `fused_rmsnorm_rotate_mq` from F1) AND the output-side projections
/// (`o_proj` / `out_proj` / `down_proj` / `w_down`, fed via the AWQ-aware
/// variants `rotate_x_mq_awq` and `fused_silu_mul_mq_rotate_awq` from F2).
///
/// Runtime path mapping for AWQ inverse divide-by-scale:
/// - `fused_rmsnorm_mq_rotate_awq`: post-RMSNorm input projections
///   (q/k/v/qkv, gate/up, in_proj_*, router, gate_up_proj)
/// - `rotate_x_mq_awq`: post-attention input to o_proj / out_proj
/// - `fused_silu_mul_mq_rotate_awq`: post-SwiGLU input to down_proj
///
/// Pre-F2 history: until 2026-05-14, output-side projections (o_proj /
/// out_proj / down_proj / w_down) were NOT on this whitelist because
/// their runtime path lacked AWQ-aware kernels. Pre-scaling them without
/// a runtime compensating divide produces `(W·s) · x ≠ W · x` — measured
/// 0.8B Qwen3.5 KLD blowup 0.6721 → 13.4893; see `awq_fix_claude.md`.
/// F2 added those kernels (`rotate_x_mq_awq` / `fused_silu_mul_mq_rotate_awq`)
/// plus `_for` helper routing in hipfire-runtime/llama.rs, so the whitelist
/// is now safe to expand.
///
/// Whitelist (vs blacklist) is still the safe default: a new tensor name
/// in a future arch fails closed (no AWQ) until someone confirms its
/// runtime path is AWQ-aware.
fn awq_eligible(name: &str) -> bool {
    // F1-vs-F2 A/B gate. When `HIPFIRE_AWQ_F1_ONLY=1` is set, the F2
    // additions below (o_proj / wo / out_proj / down_proj / w_down)
    // are excluded — produces an F1-equivalent quant for comparison
    // bench against the same binary's F2 quant. Default (env unset):
    // the full F2 whitelist applies.
    let f1_only = std::env::var("HIPFIRE_AWQ_F1_ONLY").ok().as_deref() == Some("1");
    let f1_match =
    // Full-attention input projections (HF naming + fused variants).
    name.ends_with("q_proj.weight")
        || name.ends_with("k_proj.weight")
        || name.ends_with("v_proj.weight")
        || name.ends_with("qkv_proj.weight")
        || name.ends_with("wqkv.weight")
        // MLP input projections (HF + hipfire-internal naming).
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with(".feed_forward.w1.weight")
        || name.ends_with(".feed_forward.w3.weight")
        || name.ends_with("w_gate.weight")
        || name.ends_with("w_up.weight")
        // MoE fused expert gate+up projection (Qwen3-MoE convention —
        // experts.gate_up_proj is [num_experts, 2*intermediate, hidden]
        // with rows split between gate and up halves). Same input-side
        // semantics as gate_proj/up_proj: post-RMSNorm hidden state
        // routed via the MoE dispatch.
        || name.ends_with("gate_up_proj.weight")
        // Linear-attention input projections (Qwen3.5 Gated-DeltaNet).
        // Suffix varies (in_proj_qkv / _z / _a / _b); the substring is
        // anchored enough that no non-linear-attn tensor name should match.
        || name.contains(".in_proj_")
        // LFM2 LIV conv input projection is named conv.in_proj.
        || name.ends_with(".conv.in_proj.weight")
        // MoE router (HF naming for Qwen3-MoE / DeepSeek family — single
        // linear projecting post-RMSNorm hidden state to num_experts
        // logits). The quantizer's q8_router rule (set when is_moe)
        // promotes this to Q8 before reaching the MQ4G256 branch, so
        // this match is effectively dead code today. Kept for intent:
        // if Q8 auto-promotion is ever disabled, this preserves
        // correctness. `router.weight` would be a non-HF naming an
        // arch might choose; kept for safety.
        || name.ends_with("mlp.gate.weight")
        // MiniMax-M2 MoE router (block_sparse_moe.gate.weight). Same intent
        // as mlp.gate.weight: q8_router (set for is_minimax via is_moe_like)
        // keeps the router at Q8 so HFQ4 noise can't flip top-k selection.
        || name.ends_with("block_sparse_moe.gate.weight")
        || name.ends_with("router.weight");
    if f1_only {
        return f1_match;
    }
    let f2_match =
        // ── F2 (2026-05-14): output-side projections ────────────────────
        // These now have AWQ-aware runtime kernels (rotate_x_mq_awq for
        // o_proj/out_proj/wo; fused_silu_mul_mq_rotate_awq for down_proj/w_down).
        // Runtime dispatch routes through _for helpers in llama.rs based on
        // WeightTensor.awq_scale.
        //
        // FullAttention output projection (HF + hipfire-internal naming).
        name.ends_with("o_proj.weight")
        || name.ends_with("wo.weight")
        // LinearAttention output projection (Qwen3.5 Gated-DeltaNet).
        || name.ends_with("out_proj.weight")
        // MLP down projection (HF + hipfire-internal naming).
        || name.ends_with("down_proj.weight")
        || name.ends_with(".feed_forward.w2.weight")
        || name.ends_with("w_down.weight");
    f1_match || f2_match
}

/// True if the tensor is the token embedding. We Q8 these (matches the
/// safetensors path's `is_embed` rule — Q4 is too lossy for embedding tables).
fn gguf_is_embed_tensor(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Build the `config` JSON object that `hipfire_runtime::hfq::config_from_hfq`
/// reads. Mirrors the field names HuggingFace uses in `config.json` for
/// LlamaForCausalLM / Qwen3ForCausalLM, populated from the GGUF
/// `<arch>.*` metadata keys.
fn config_json_from_gguf(gguf: &gguf_input::GgufFile, arch_str: &str) -> serde_json::Value {
    // GGUF prefixes its model hyperparameters with the architecture name —
    // e.g. for `general.architecture=llama` the keys live under `llama.*`.
    let prefix = arch_str;

    let read_u = |k: &str| -> Option<u64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::U8(x) => Some(*x as u64),
            gguf_input::MetaValue::I8(x) => Some(*x as u64),
            gguf_input::MetaValue::U16(x) => Some(*x as u64),
            gguf_input::MetaValue::I16(x) => Some(*x as u64),
            gguf_input::MetaValue::U32(x) => Some(*x as u64),
            gguf_input::MetaValue::I32(x) => Some(*x as u64),
            gguf_input::MetaValue::U64(x) => Some(*x),
            gguf_input::MetaValue::I64(x) => Some(*x as u64),
            _ => None,
        })
    };
    let read_f = |k: &str| -> Option<f64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::F32(x) => Some(*x as f64),
            gguf_input::MetaValue::F64(x) => Some(*x),
            _ => None,
        })
    };

    let dim = read_u(&format!("{prefix}.embedding_length"));
    let n_layers = read_u(&format!("{prefix}.block_count"));
    let n_heads = read_u(&format!("{prefix}.attention.head_count"));
    let n_kv_heads = read_u(&format!("{prefix}.attention.head_count_kv")).or(n_heads);
    let hidden_dim = read_u(&format!("{prefix}.feed_forward_length"));
    // vocab_size: prefer metadata, fall back to token_embd shape[1].
    let vocab_size = read_u(&format!("{prefix}.vocab_size")).or_else(|| {
        gguf.tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.get(1).map(|&s| s as u64))
    });
    let max_seq_len = read_u(&format!("{prefix}.context_length"));
    let rope_theta = read_f(&format!("{prefix}.rope.freq_base"));
    let rms_eps = read_f(&format!("{prefix}.attention.layer_norm_rms_epsilon"));
    let head_dim = read_u(&format!("{prefix}.attention.key_length")).or_else(|| {
        // Fall back: head_dim = dim / n_heads.
        dim.zip(n_heads).map(|(d, h)| if h > 0 { d / h } else { d })
    });
    let bos = read_u("tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos = read_u("tokenizer.ggml.eos_token_id").unwrap_or(2);

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "model_type".to_string(),
        serde_json::Value::from(arch_str.to_string()),
    );
    if let Some(v) = dim {
        cfg.insert("hidden_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_layers {
        cfg.insert("num_hidden_layers".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_heads {
        cfg.insert(
            "num_attention_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = n_kv_heads {
        cfg.insert(
            "num_key_value_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = hidden_dim {
        cfg.insert("intermediate_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = vocab_size {
        cfg.insert("vocab_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = max_seq_len {
        cfg.insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = rope_theta {
        cfg.insert("rope_theta".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = rms_eps {
        cfg.insert("rms_norm_eps".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = head_dim {
        cfg.insert("head_dim".to_string(), serde_json::Value::from(v));
    }
    cfg.insert("bos_token_id".to_string(), serde_json::Value::from(bos));
    cfg.insert("eos_token_id".to_string(), serde_json::Value::from(eos));
    serde_json::Value::Object(cfg)
}

/// Translate the GGUF metadata HashMap into a JSON object that ends up in
/// the `.hfq` header's metadata blob. A future engine-side `from_hfq` for
/// Llama-style models can read these fields the same way the existing
/// `from_gguf` reads them today.
fn gguf_meta_to_json(meta: &HashMap<String, gguf_input::MetaValue>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in meta {
        let json_v = mv_to_json(v);
        map.insert(k.clone(), json_v);
    }
    serde_json::Value::Object(map)
}

fn mv_to_json(v: &gguf_input::MetaValue) -> serde_json::Value {
    use gguf_input::MetaValue as MV;
    match v {
        MV::U8(x) => serde_json::Value::from(*x),
        MV::I8(x) => serde_json::Value::from(*x),
        MV::U16(x) => serde_json::Value::from(*x),
        MV::I16(x) => serde_json::Value::from(*x),
        MV::U32(x) => serde_json::Value::from(*x),
        MV::I32(x) => serde_json::Value::from(*x),
        MV::F32(x) => serde_json::Value::from(*x),
        MV::Bool(x) => serde_json::Value::from(*x),
        MV::String(s) => serde_json::Value::from(s.clone()),
        MV::U64(x) => serde_json::Value::from(*x),
        MV::I64(x) => serde_json::Value::from(*x),
        MV::F64(x) => serde_json::Value::from(*x),
        // Tokenizer arrays (tokens, scores, merges, ...) can be huge —
        // serialize them as JSON arrays so the engine side can re-parse.
        MV::Array(arr) => serde_json::Value::Array(arr.iter().map(mv_to_json).collect()),
    }
}

/// 2D-weight quantization target chosen at the per-tensor level. The choice
/// per format flag:
///
/// | --format | 2D weights      | embedding | comment                          |
/// |----------|-----------------|-----------|----------------------------------|
/// | fp16     | F16             | F16       | reference / maximum-size smoke   |
/// | bf16     | BF16/F16        | BF16/F16  | source-precision smoke container |
/// | hfq4     | HFQ4G256        | Q8F16     | dense plain format — no FWHT     |
/// | hfq6     | HFQ6G256        | Q8F16     | dense + higher quality           |
/// | mq4      | MQ4G256         | Q8F16     | Qwen3.5+ (DeltaNet) — FWHT-rot   |
/// | mq6      | MQ6G256         | Q8F16     | Qwen3.5+ (DeltaNet) + higher q   |
/// | mq3      | MQ3G256         | Q8F16     | Sub-4-bit FWHT (3.25 bpw)        |
/// | mq2      | MQ2G256         | Q8F16     | Sub-4-bit FWHT (2.25 bpw)        |
///
/// **MQ4/MQ6 for non-Qwen3.5 dense produces correct output on the Llama path
/// (the rotation cancels via `gemv_mq4g256_with_rotate`) but adds per-layer
/// `rotate_x_mq` overhead with no quality benefit — those rotations were
/// calibrated for Qwen3.5+ training.** Use `--format hfq4` for the plain dense
/// path; pass `--format mq4` only when the source is a Qwen3.5+ family model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GgufFormat {
    F16,
    Bf16,
    Hfq4,
    Hfq6,
    Mq4,
    Mq6,
    Mq3,
    Mq2,
    Mq2Lloyd,
    Mq3Lloyd,
    Mq4Lloyd,
    Hfp4, // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale)
    Mfp4, // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement)
}

impl GgufFormat {
    fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "fp16" | "f16" | "float16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "hfq4" | "hfq4g256" | "hf4" => Some(Self::Hfq4),
            "hfq6" | "hfq6g256" | "hf6" => Some(Self::Hfq6),
            "mq4" | "mq4g256" | "magnum" => Some(Self::Mq4),
            "mq6" | "mq6g256" => Some(Self::Mq6),
            "mq3" | "mq3g256" => Some(Self::Mq3),
            "mq2" | "mq2g256" => Some(Self::Mq2),
            "lloyd-mq2" => Some(Self::Mq2Lloyd),
            "lloyd-mq3" => Some(Self::Mq3Lloyd),
            "lloyd-mq4" => Some(Self::Mq4Lloyd),
            "hfp4" | "hfp4g32" | "hf4p" | "fp4" => Some(Self::Hfp4),
            "mfp4" | "mfp4g32" | "mf4p" => Some(Self::Mfp4),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Hfq4 => "HFQ4G256",
            Self::Hfq6 => "HFQ6G256",
            Self::Mq4 => "MQ4G256",
            Self::Mq6 => "MQ6G256",
            Self::Mq3 => "MQ3G256",
            Self::Mq2 => "MQ2G256",
            Self::Mq2Lloyd => "MQ2G256Lloyd",
            Self::Mq3Lloyd => "MQ3G256Lloyd",
            Self::Mq4Lloyd => "MQ4G256Lloyd",
            Self::Hfp4 => "HFP4G32",
            Self::Mfp4 => "MFP4G32",
        }
    }
}

fn source_precision_tensor_bytes(
    raw_data: &[u8],
    dtype: &str,
    f32_data: &[f32],
) -> (Vec<u8>, QuantType, &'static str) {
    match dtype {
        "BF16" => (raw_data.to_vec(), QuantType::BF16, "BF16"),
        "F16" => (raw_data.to_vec(), QuantType::F16, "F16"),
        "F32" => (f32_slice_to_f16_bytes(f32_data), QuantType::F16, "F16"),
        other => panic!("unsupported dtype for source-precision HFQ: {other}"),
    }
}

/// Convert a GGUF file to a hipfire `.hfq`. Per-format quantization target
/// applies to 2D weight matrices; the embedding table is always Q8F16
/// (Q4-grade is too lossy for embeddings) and 1D norms stay F16. Tensor
/// names are translated GGUF → safetensors style so the engine's existing
/// `load_weights_hfq` can consume the output.
fn run_gguf_pipeline(
    input: &Path,
    output: &Path,
    format: GgufFormat,
    format_label: &str,
    no_kmap: bool,
    kmap_dense: bool,
    kmap_mode: u8,
) -> std::io::Result<()> {
    eprintln!("=== GGUF → {} conversion ===", format.label());
    eprintln!("Input:  {}", input.display());
    eprintln!("Output: {}", output.display());

    let gguf = gguf_input::GgufFile::open(input)?;
    eprintln!("GGUF version: {}", gguf.version);
    eprintln!("Tensors: {}", gguf.tensors.len());

    let arch_str = gguf
        .meta_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    let auto_arch_id: u32 = match arch_str.as_str() {
        "llama" => 0,
        "qwen3" | "qwen2" => 1,
        "qwen3moe" => 6,
        other => {
            eprintln!("warning: unknown GGUF architecture '{other}', tagging as llama-compatible");
            0
        }
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id: u32 = parse_arch_id_override().unwrap_or(auto_arch_id);
    if arch_id != auto_arch_id {
        eprintln!("Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})");
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }

    // Metadata JSON: must populate `config.*` so engine's `config_from_hfq`
    // can reconstruct LlamaConfig at load time. Also keep the raw GGUF
    // metadata tree under `gguf_meta` for any consumer that wants original
    // values (chat template, vocab, scores, merges, etc.).
    let config_json = config_json_from_gguf(&gguf, &arch_str);
    let mut metadata = serde_json::json!({
        "architecture": arch_str,
        "source": "gguf",
        "quant_format": format_label,
        "config": config_json,
        "gguf_meta": gguf_meta_to_json(&gguf.metadata),
    });

    // FWHT signs — only used by MQ/MFP formats. Same seed pair as the
    // safetensors path so the engine's runtime FWHT inverse stays identical.
    let needs_signs = matches!(
        format,
        GgufFormat::Mq4
            | GgufFormat::Mq6
            | GgufFormat::Mq3
            | GgufFormat::Mq2
            | GgufFormat::Mq2Lloyd
            | GgufFormat::Mq3Lloyd
            | GgufFormat::Mq4Lloyd
            | GgufFormat::Mfp4
    );
    let signs1 = if needs_signs {
        gen_fwht_signs(42, 256)
    } else {
        Vec::new()
    };
    let signs2 = if needs_signs {
        gen_fwht_signs(1042, 256)
    } else {
        Vec::new()
    };

    // K-map setup for GGUF path
    let is_moe = arch_id == 6;
    let n_layers: usize = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build K-map using translated (safetensors-style) names where available,
    // falling back to raw GGUF names for untranslated tensors.
    //
    // K-map is gated to MoE models only. On dense models the author's own
    // bench shows a mixed picture (PPL +1.5% to +2.5% at 2K context on 4B
    // and 27B; PPL -4.8% on 27B at 8K context — crossover at ~3K). The
    // ship-default is the conservative shape per maintainer directive
    // (2026-05-08): never silently change dense quantization. Users who
    // want K-map on dense pass `--kmap-dense` (see flag parsing below).
    let kmap: HashMap<String, QuantLevel> =
        if format == GgufFormat::F16 || no_kmap || (!is_moe && !kmap_dense) {
            HashMap::new()
        } else {
            let mut map = HashMap::new();
            let mut counts = [0u32; 4];
            for info in &gguf.tensors {
                let out_name =
                    gguf_to_safetensors_name(&info.name).unwrap_or_else(|| info.name.clone());
                let level = kmap_resolve_mode(&out_name, n_layers, is_moe, kmap_mode);
                match level {
                    QuantLevel::F16 => counts[0] += 1,
                    QuantLevel::Q8 => counts[1] += 1,
                    QuantLevel::Promote6 => counts[2] += 1,
                    QuantLevel::Override(_) => counts[3] += 1,
                    QuantLevel::Base => counts[3] += 1,
                }
                map.insert(out_name, level);
            }
            if !map.is_empty() {
                let mode_label = match kmap_mode {
                    0 => "full",
                    1 => "alternating",
                    2 => "typed",
                    _ => "?",
                };
                eprintln!(
                    "K-map plan ({} base, {n_layers} layers{}, mode={mode_label}):",
                    format.label(),
                    if is_moe { ", MoE" } else { "" }
                );
                eprintln!("  F16:       {:>4} tensors", counts[0]);
                eprintln!("  Q8:        {:>4} tensors", counts[1]);
                eprintln!("  Promote6:  {:>4} tensors", counts[2]);
                eprintln!("  Base:      {:>4} tensors", counts[3]);
            }
            map
        };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(gguf.tensors.len());
    let mut total_params: u64 = 0;
    let mut quant_params: u64 = 0;
    let mut total_bytes_in: u64 = 0;
    let mut total_bytes_out: u64 = 0;

    for info in &gguf.tensors {
        let raw = gguf.tensor_data(info);
        let n_elements = info.numel();
        total_params += n_elements as u64;
        total_bytes_in += raw.len() as u64;

        let shape: Vec<u32> = info.shape.iter().map(|&s| s as u32).collect();

        // Tensor classification (uses the original GGUF name).
        let is_norm = gguf_is_norm_tensor(&info.name);
        let is_embed = gguf_is_embed_tensor(&info.name);
        let is_2d = info.shape.len() == 2;
        let k_dim = if is_2d { info.shape[0] } else { n_elements };

        // Translate to the safetensors-style name `hipfire_runtime::hfq::load_weights_hfq`
        // expects. If we don't have a translation, keep the original name —
        // the future loader can ignore unknown tensors.
        let out_name = gguf_to_safetensors_name(&info.name).unwrap_or_else(|| info.name.clone());

        let kmap_level = kmap.get(&out_name).copied().unwrap_or(QuantLevel::Base);

        let (data, quant_type, group_size, label) = if format == GgufFormat::F16
            || is_norm
            || !is_2d
        {
            // Full-F16 artifacts, norms, and 1D tensors use raw half precision.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
            (f16_bytes, QuantType::F16, 0u32, "F16")
        } else if kmap_level == QuantLevel::Q8 || is_embed {
            // K-map Q8 or embedding
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0 {
            // K-map promote to 6-bit
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Mq4
                | GgufFormat::Mq3
                | GgufFormat::Mq2
                | GgufFormat::Mq2Lloyd
                | GgufFormat::Mq3Lloyd
                | GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq4 | GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Hfp4 => {
                    // No HFP6 variant in v1. Promote6 for HFP4 stays at HFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    // No MFP6 variant. Promote6 for MFP4 stays at MFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mq4Lloyd => {
                    // Promote6 -> MQ6, consistent with default_promote_target
                    // (Mq4Lloyd -> Mq6) and the Lloyd siblings above.
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::F16 | GgufFormat::Bf16 => {
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                }
            }
        } else if let (QuantLevel::Override(override_fmt), true) = (kmap_level, k_dim % 256 == 0) {
            // K-map says override (lm_head when --lm-head-format set).
            // GGUF pipeline has no AWQ wiring (AWQ is safetensors-only today),
            // so this is a plain quantize on the carried target format.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match override_fmt {
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::F16 | GgufFormat::Bf16 => {
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                }
            }
        } else if k_dim % 256 == 0 {
            // 256-aligned 2D weight — quantize per the chosen format (Base level).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Mq4Lloyd => {
                    let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::F16 | GgufFormat::Bf16 => {
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                }
            }
        } else {
            // K not divisible by 256 — fall back to HFQ4-G128 (no rotation).
            // This branch fires for the rare ragged dim; ignores --format
            // (no G128 variant of mq4/mq6 exists).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_hfq4g128(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
        };

        total_bytes_out += data.len() as u64;
        eprintln!(
            "  {label:>9}: {} → {} {:?} ({} src={:?}, {:.1} KB → {:.1} KB)",
            info.name,
            out_name,
            info.shape,
            n_elements,
            info.dtype,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0,
        );

        hfq_tensors.push(HfqTensor {
            name: out_name,
            quant_type,
            shape,
            group_size,
            data,
            spilled_len: 0,
        });
    }

    eprintln!("\n=== GGUF → MQ4 Summary ===");
    eprintln!("  Tensors:        {}", hfq_tensors.len());
    eprintln!("  Total params:   {total_params}");
    eprintln!(
        "  Quant'd params: {quant_params} ({:.1}%)",
        100.0 * quant_params as f64 / total_params as f64
    );
    eprintln!("  Input size:     {:.1} MB", total_bytes_in as f64 / 1e6);
    eprintln!(
        "  Output size:    {:.1} MB ({:.1}% of input)",
        total_bytes_out as f64 / 1e6,
        100.0 * total_bytes_out as f64 / total_bytes_in as f64,
    );

    insert_parameter_counts_metadata(&mut metadata, &hfq_tensors, total_params, quant_params, 0);
    let metadata_json = metadata_with_quantization_hash(metadata, &hfq_tensors, None)?;
    write_hfq(output, arch_id, &metadata_json, &hfq_tensors, None)?;
    eprintln!("\nWrote: {}", output.display());
    Ok(())
}

/// Full `--help` / `-h` text. Printed to stdout; exits 0 from `main`.
fn print_help() {
    print!(
        r#"hipfire-quantize — quantize model weights to a HipFire .hfq artifact

USAGE:
    hipfire-quantize --input <model_dir|.gguf|.hfq> --output <out.hfq> --format <FMT> [OPTIONS]

INPUT SOURCES (--input):
    <model_dir>   HuggingFace model directory (safetensors) or HF model ID (e.g. Qwen/Qwen3.5-4B)
    <file.gguf>   a single GGUF file
    <file.hfq>    an existing .hfq (e.g. a bf16 .hfq) for requantization.
                  The .hfq-source path supports --format
                  bf16 / fp16 / q8f16 / hfq4 / hfq6 / mq4 / mq6 / mq3 / qtip3 /
                  qtip4 / oq4 (opus; legacy op4 aliases) / oq4+ / oq8 (opus8) / oq8+.
                  Other formats (roughquant, lloyd-*, mfp4, …) require a HF/GGUF source.

REQUIRED:
    --input <PATH>     source model (see INPUT SOURCES above)
    --output <PATH>    destination .hfq file
    --format <FMT>     quant format (see FORMAT)

FORMAT (--format <FMT>):
    Full precision     bf16 (bfloat16) · fp16 (f16/float16) · f32 (oracle/passthrough)
    Production quant   q8f16 (q8) · mq4 (magnum) · mq6 · mq3 · hfq4 · hfq6 · mfp4 (hfp4g32) · q8hfq
    Opus Quant         oq4 / op4 / op4-4 (alias: opus) — 4-bit-resident Opus Quant.
                       oq4+ is oq4 plus activation-aware AWQ/SmoothQuant
                       calibration; it requires --imatrix or --hessian.
                       oq4++ adds full-Hessian LDLQ feedback and requires
                       --hessian. Legacy op4+ spellings remain aliases for oq4++.
                       oq8 / op8 / op8-16 (alias: opus8) — 8-bit Opus Quant.
                       oq8+ is oq8 plus activation-aware AWQ/SmoothQuant
                       calibration; oq8++ adds full-Hessian LDLQ feedback.
                       Both keep the Oq8G256 W8A8 runtime format. Legacy op8+
                       spellings remain aliases for oq8++.
    Legacy Opus-A8     opplus / op4-plus — older W4A8 experimental tag
                       (qt=33), distinct from oq4+.
    MoE / routed       mq4-mq6exp · mq4-routed-lloyd-mq-tiered (needs --imatrix) · antirez-mq · …
    Research (gated)   mq2 · lloyd-mq2 · lloyd-mq3 · lloyd-mq4 · qtip3 · qtip3-sim · qtip4 ·
                       roughquant (rq) · roughquant{{2,3,4}}-sim · permute5 (rq5)
                       Sub-4-bit and Lloyd formats refuse to run without the matching
                       --allow-* flag / HIPFIRE_ALLOW_* env (see "Research opt-in" below).

OPTIONS:
    --imatrix <PATH>           llama.cpp imatrix GGUF; enables importance-weighted Lloyd and AWQ
    --awq                      activation-aware weight pre-scaling (default alpha=0.55,
                               nemotron_h=0.1); requires --imatrix or --hessian
    --awq-alpha <F>            enable AWQ at an explicit alpha (overrides the per-arch default)
    --ldlq                     full-Hessian (GPTQ/OBS) error-feedback for oq4++/oq8++
                               weights instead of RTN; requires --hessian. Composes
                               with --awq (smooths activations + rebases the Hessian)
    --chat-template-file <P>   override the embedded chat template (Jinja file)
    --threads <N>              rayon worker threads (default 80% of cores; env HIPFIRE_QUANT_THREADS)
    --beam <N>                 QTIP trellis beam-search width (default 128, near-Viterbi); lower =
                               much faster encode on big models, slight quality loss (env HIPFIRE_QTIP_BEAM)
    --arch-id <U32>            override the auto-detected arch id stamped in the .hfq header

  MoE / K-map:
    --kmap-dense               enable K-map promotion on dense models (default: MoE-only)
    --kmap-mode <full|alt|typed>  K-map candidate set (also 0/1/2); default alt
    --no-kmap, --uniform       disable K-map promotion entirely
    --q8-router                quantize the MoE router/gate to Q8
    --no-q8-conv1d             keep DeltaNet conv1d at the --format quant (default: forced Q8)
    --tier-ratio <F>           MQ tier split for tiered routed formats (default 0.30; env HIPFIRE_TIER_RATIO)

  Tensor selection:
    --include-prefix <P>       ingest ONLY tensors whose name starts with <P> (build sidecars, e.g. mtp.)
    --include-vision           include vision-tower tensors (default: skipped)
    --vision-quant <FMT>       format for vision tensors when --include-vision is set

  Research opt-in (formats refuse unless the gate is set):
    --allow-mq2          / HIPFIRE_ALLOW_MQ2=1
    --allow-mq3-lloyd    / HIPFIRE_ALLOW_MQ3_LLOYD=1
    --allow-mq2-lloyd    / HIPFIRE_ALLOW_MQ2_LLOYD=1
    --allow-mq4-lloyd    / HIPFIRE_ALLOW_MQ4_LLOYD=1

  Fixtures:
    --emit-fixture <ARCH>      write a tiny random-init HF model (for gating) and exit
    --out, --output <PATH>     fixture destination (default ./tiny-fixture)
    --seed <U64>               RNG seed for --emit-fixture (default 0x00C0FFEE)

ENVIRONMENT:
    HIPFIRE_QUANT_THREADS      worker-thread override (see --threads)
    HIPFIRE_TIER_RATIO         tiered routed MQ split (see --tier-ratio)
    HIPFIRE_GPTQ_DAMPING       GPTQ/LDLQ diagonal ridge damping (default 0.01)
    HIPFIRE_QTIP_HESSIAN       path to a .calib.hfq (HFQM) package → enables QTIP-LDLQ
    HIPFIRE_ALLOW_MQ2 / _MQ3_LLOYD / _MQ2_LLOYD / _MQ4_LLOYD   research-format gates

    -h, --help                 print this help and exit
"#
    );
}

/// GuidedQuant validation (`--ldlq-probe`). LDLQ-quantize ONE weight `<key>`
/// (the safetensors `<key>.weight`) with a guided vs a plain **training**
/// Hessian, then score reconstruction on held-out guided/plain **eval**
/// Hessians. The crossover — guided lower on the Fisher-eval, plain lower on the
/// plain-eval — shows the Fisher weighting generalizes on its own objective.
///
/// Flags: --model <dir> --tensor <key> --hg-train <f> --hp-train <f>
///        --hg-eval <f> --hp-eval <f> [--bits N]
fn ldlq_recon_probe(args: &[String]) -> Result<(), String> {
    use rayon::prelude::*;
    let val = |k: &str| arg_value(args, k);
    let model = val("--model").ok_or("need --model <dir>")?;
    let key = val("--tensor").ok_or("need --tensor <key>")?.to_string();
    let hg_train = val("--hg-train").ok_or("need --hg-train")?;
    let hp_train = val("--hp-train").ok_or("need --hp-train")?;
    let hg_eval = val("--hg-eval").ok_or("need --hg-eval")?;
    let hp_eval = val("--hp-eval").ok_or("need --hp-eval")?;
    let bits: u32 = val("--bits").and_then(|s| s.parse().ok()).unwrap_or(3);

    // Weight W [m,k] f32 from safetensors.
    let wname = format!("{key}.weight");
    let st_paths = find_safetensors(Path::new(model));
    let (w, m, k) = {
        let mut found = None;
        for p in &st_paths {
            let sf = SafetensorsFile::open(p).map_err(|e| e.to_string())?;
            if let Some((meta, bytes)) = sf.tensor_data(&wname) {
                if meta.shape.len() != 2 {
                    return Err(format!("{wname} is not 2D"));
                }
                found = Some((to_f32(bytes, &meta.dtype), meta.shape[0], meta.shape[1]));
                break;
            }
        }
        found.ok_or(format!("tensor {wname} not found under {model}"))?
    };
    eprintln!("W {key} [{m}×{k}], {bits}-bit LDLQ");

    // Materialize a [k,k] Hessian from a .calib.hfq by key.
    let load_h = |path: &str| -> Result<Vec<f32>, String> {
        let sc = hessian_io::HessianSidecar::open(Path::new(path)).map_err(|e| e.to_string())?;
        let h = sc
            .get(&key, 0)
            .ok_or_else(|| format!("no hessian for {key} in {path}"))?;
        if h.k != k {
            return Err(format!("hessian k={} != weight k={k} ({path})", h.k));
        }
        let mut out = vec![0.0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                out[i * k + j] = h.at(i, j) as f32;
            }
        }
        Ok(out)
    };
    let hgt = load_h(hg_train)?;
    let hpt = load_h(hp_train)?;

    // Diagnostic: how much did the Fisher weighting actually perturb the Hessian?
    // Both train Hessians are over the SAME tokens, so diag(Hg)/diag(Hp) per input
    // channel c = (Σ_n w_n x_n[c]²)/(Σ_n x_n[c]²) is the effective per-channel
    // weight. A tight ratio (CV≈0) means w is near-uniform ⇒ g=1 is a no-op, not
    // a "hurt". `‖Hg-Hp‖/‖Hp‖` is the overall relative perturbation.
    {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for idx in 0..k * k {
            let d = (hgt[idx] - hpt[idx]) as f64;
            num += d * d;
            den += (hpt[idx] as f64).powi(2);
        }
        let (mut rmin, mut rmax, mut rsum, mut rsum2, mut nr) =
            (f64::MAX, 0.0f64, 0.0f64, 0.0f64, 0usize);
        for c in 0..k {
            let dp = hpt[c * k + c] as f64;
            if dp > 0.0 {
                let r = hgt[c * k + c] as f64 / dp;
                rmin = rmin.min(r);
                rmax = rmax.max(r);
                rsum += r;
                rsum2 += r * r;
                nr += 1;
            }
        }
        let rmean = rsum / nr.max(1) as f64;
        let rstd = (rsum2 / nr.max(1) as f64 - rmean * rmean).max(0.0).sqrt();
        eprintln!(
            "diag: ‖Hg-Hp‖/‖Hp‖ = {:.4}  |  diag(Hg)/diag(Hp): mean {:.3} std {:.3} CV {:.3} range [{:.3}, {:.3}]",
            (num / den).sqrt(),
            rmean,
            rstd,
            rstd / rmean.max(1e-9),
            rmin,
            rmax
        );
    }

    // LDLQ-quantize with each training Hessian (damp = 1% of mean diagonal).
    let cb = qtip::build_codebook();
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let damp = |h: &[f32]| -> f64 {
        let s: f64 = (0..k).map(|i| h[i * k + i] as f64).sum();
        0.01 * (s / k as f64).max(1e-12)
    };
    let wq_g = ldlq::qtip_ldlq_dequant_bits(&w, m, k, &hgt, &s1, &s2, 128, damp(&hgt), bits, &cb)
        .ok_or("LDLQ(guided) failed")?;
    let wq_p = ldlq::qtip_ldlq_dequant_bits(&w, m, k, &hpt, &s1, &s2, 128, damp(&hpt), bits, &cb)
        .ok_or("LDLQ(plain) failed")?;
    drop((hgt, hpt));
    eprintln!("LDLQ done (guided + plain); scoring held-out …");

    // Held-out error under an eval Hessian: E = Σ_j (w_j−ŵ_j)ᵀ H (w_j−ŵ_j),
    // plus the weight energy En = Σ_j w_jᵀ H w_j for a relative figure.
    let err = |wq: &[f32], h: &[f32]| -> (f64, f64) {
        (0..m)
            .into_par_iter()
            .map(|j| {
                let wr = &w[j * k..(j + 1) * k];
                let qr = &wq[j * k..(j + 1) * k];
                let (mut e, mut en) = (0.0f64, 0.0f64);
                for a in 0..k {
                    let hrow = &h[a * k..(a + 1) * k];
                    let (mut hd, mut hw) = (0.0f64, 0.0f64);
                    for b in 0..k {
                        let hab = hrow[b] as f64;
                        hd += hab * (wr[b] - qr[b]) as f64;
                        hw += hab * wr[b] as f64;
                    }
                    e += (wr[a] - qr[a]) as f64 * hd;
                    en += wr[a] as f64 * hw;
                }
                (e, en)
            })
            .reduce(|| (0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1))
    };

    let hge = load_h(hg_eval)?;
    let (eg_f, en_f) = err(&wq_g, &hge);
    let (ep_f, _) = err(&wq_p, &hge);
    drop(hge);
    let hpe = load_h(hp_eval)?;
    let (eg_p, en_p) = err(&wq_g, &hpe);
    let (ep_p, _) = err(&wq_p, &hpe);
    drop(hpe);

    println!("=== GuidedQuant LDLQ recon crossover — {key} ({bits}-bit) ===");
    println!(
        "held-out FISHER eval:  guided rel {:.5}  plain rel {:.5}  -> {}",
        eg_f / en_f,
        ep_f / en_f,
        if eg_f < ep_f {
            "GUIDED wins ✓"
        } else {
            "plain"
        }
    );
    println!(
        "held-out PLAIN  eval:  guided rel {:.5}  plain rel {:.5}  -> {}",
        eg_p / en_p,
        ep_p / en_p,
        if ep_p < eg_p {
            "PLAIN wins ✓"
        } else {
            "guided"
        }
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    // `--emit-fixture <arch>`: write a tiny random-init HF model (safetensors +
    // config.json) for gating, then exit. Flows through the normal `--input`
    // quantize path afterward (separate invocation). See src/fixture.rs.
    if let Some(arch) = arg_value(&args, "--emit-fixture") {
        let out = arg_value(&args, "--out")
            .or_else(|| arg_value(&args, "--output"))
            .unwrap_or("./tiny-fixture");
        let seed = arg_value(&args, "--seed")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0x00C0_FFEE);
        match fixture::emit_fixture(arch, Path::new(out), seed) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("emit-fixture: {e}");
                std::process::exit(2);
            }
        }
    }

    // `--ldlq-probe`: GuidedQuant validation. LDLQ-quantize ONE weight with a
    // guided vs a plain calibration Hessian, then score the reconstruction on
    // held-out guided/plain eval Hessians — the crossover (guided wins the
    // Fisher-eval, plain wins the plain-eval) shows the Fisher weighting does
    // what it should. No full quant, no model load. See crates/hipfire-train
    // examples/calib_guided.rs for producing the .calib.hfq inputs.
    if args.iter().any(|a| a == "--ldlq-probe") {
        match ldlq_recon_probe(&args) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("ldlq-probe: {e}");
                std::process::exit(2);
            }
        }
    }

    // Bound rayon's pool to 80% of cores (default cap; override with --threads N
    // or HIPFIRE_QUANT_THREADS env). Quantization is CPU-bound and saturates
    // memory bandwidth, so leaving headroom for the rest of the system avoids
    // making the whole box unresponsive during a multi-hour quantize run.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let default_threads = ((cores * 8) / 10).max(1);
    let threads = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse::<usize>().ok()))
        .or_else(|| {
            std::env::var("HIPFIRE_QUANT_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(default_threads);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
    eprintln!("Rayon: {threads} worker threads ({cores} cores available, default 80% = {default_threads})");

    let input_dir = arg_value(&args, "--input").unwrap_or_else(|| {
        eprintln!("Usage: hipfire-quantize --input <model_dir|.gguf|.hfq> --output <output.hfq> --format <FMT>");
        std::process::exit(1);
    });

    let output_path = arg_value(&args, "--output")
        .unwrap_or_else(|| { eprintln!("Usage: hipfire-quantize --input <model_dir|.gguf|.hfq> --output <output.hfq> --format <FMT>"); std::process::exit(1); });

    let chat_template_override =
        arg_value(&args, "--chat-template-file").map(|p| read_chat_template_file(Path::new(p)));

    let format_arg = required_format_arg(&args).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        eprintln!(
            "       choose an explicit format, e.g. --format bf16, --format q8f16, --format mq4, or --format oq4+"
        );
        std::process::exit(2);
    });
    let mut format_storage = normalize_format_flag(format_arg);
    // OQ4+/OQ4++ are not separate storage tags: they use OQ4 bytes plus
    // calibration sidecars/packing. Normalize to OQ4, then enforce the recipe
    // after argument parsing has loaded calibration sidecars.
    let oq4_recipe = oq4_calibration_recipe(format_storage.as_str());
    let oq4_plus_recipe = oq4_recipe != OqCalibrationRecipe::Plain;
    if oq4_plus_recipe {
        format_storage = "oq4".to_string();
    }
    // OQ8+/OQ8++ are calibrated OQ8 (same Oq8G256 runtime format). Keep a
    // distinct dispatch token so the OQ8 branch can run OQ8-specific packing.
    let oq8_recipe = oq8_calibration_recipe(format_storage.as_str());
    let oq8_plus_recipe = oq8_recipe != OqCalibrationRecipe::Plain;
    if oq8_plus_recipe {
        format_storage = "oq8+".to_string();
    }
    let oq_plus_recipe = oq4_plus_recipe || oq8_plus_recipe;
    let oq_ldlq_recipe =
        oq4_recipe == OqCalibrationRecipe::AwqLdlq || oq8_recipe == OqCalibrationRecipe::AwqLdlq;
    // Legacy Opus-A8 is a distinct W4A8 FORMAT, not the generic `+`
    // clip/AWQ modifier and not calibrated OQ4+.
    let is_legacy_opus_plus = matches!(format_storage.as_str(), "opplus" | "op4-plus");
    // `mqN+` modifier: clip-search + AWQ on top of the base MQ format. Strip the
    // trailing `+` so downstream format matching sees the base (e.g. "mq4"), and
    // enable clip-search globally. AWQ is auto-enabled below.
    // MQ+ keeps the generic `+` suffix; calibrated OQ4+ was normalized above.
    let mq_plus = format_storage.ends_with('+') && !oq_plus_recipe && !is_legacy_opus_plus;
    if mq_plus {
        format_storage.pop();
        set_mq_clipsearch(true);
        eprintln!("mq+ modifier: clip-search enabled; AWQ auto-on (needs --hessian/--imatrix)");
    }
    let format = format_storage.as_str();
    eprintln!("Format: {format}");

    // Optional imatrix (llama.cpp GGUF format with .in_sum2 / .counts per-tensor).
    // When provided, MQ2-Lloyd quantization uses per-column importance weights
    // to bias centroid placement. See `quantize_mq2g256_lloyd_weighted`.
    let imatrix_path: Option<&str> = args
        .iter()
        .position(|a| a == "--imatrix")
        .map(|i| args[i + 1].as_str());
    let imatrix_gguf: Option<gguf_input::GgufFile> = imatrix_path.map(|p| {
        eprintln!("Loading imatrix: {p}");
        gguf_input::GgufFile::open(Path::new(p)).unwrap_or_else(|e| {
            eprintln!("imatrix open failed: {e}");
            std::process::exit(2);
        })
    });
    if let Some(ref gg) = imatrix_gguf {
        let n_in_sum2 = gg
            .tensors
            .iter()
            .filter(|t| t.name.ends_with(".in_sum2"))
            .count();
        let n_counts = gg
            .tensors
            .iter()
            .filter(|t| t.name.ends_with(".counts"))
            .count();
        eprintln!(
            "  imatrix: {} in_sum2 + {} counts tensors",
            n_in_sum2, n_counts
        );
    }
    // bf16 = source-precision container: BF16 tensors stay raw BF16 (qt=16);
    //        F16 tensors stay F16; F32 tensors fall back to F16.
    // fp16 = all eligible weights stored as raw F16 (qt=1)
    // q8f16 = all weights Q8 (interleaved blocks)
    // q4f16 = all weights Q4_F16_G64
    // q8-mixed = Q8 attn + Q4_K FFN (best tok/s for VRAM-constrained)
    // q8-fast = Q8 attn + Q4-as-Q8 FFN (all Q8 occupancy, most VRAM)
    // q8hfq = all weights Q8_HFQ (split-metadata, 128B-aligned rows)
    let use_fp16 = format == "fp16" || format == "f16" || format == "float16";
    // qtip2-sim: emit a bf16 .hfq whose 2D weights carry *simulated* QTIP-2
    // error (FWHT-rotated bitshift-trellis, beam=128), for a kernel-free PPL
    // verdict via the normal bf16 forward. Embeddings/lm_head kept bf16.
    // qtip3-sim: the 3-bit fallback (Phase C). 2-bit QTIP is unusable on the
    // 0.8B dense model (PPL 53.6 with LDLQ vs MQ4 14.0); 3-bit is still a
    // bandwidth win vs MQ4 and the documented fallback. Same trellis, bits=3.
    // `qtip<N>-sim` for N ∈ {2,3,4,6,8}: bit-parametric trellis sim (the codec
    // is bit-general — `qtip::*_bits`/`qtip_ldlq_dequant_bits` take `bits`), for
    // a bandwidth-vs-quality sweep across QTIP widths. Quality only (emits bf16);
    // a real packed qtip<N> kernel is separate. LDLQ on/off = HIPFIRE_QTIP_HESSIAN
    // present/absent.
    let qtip_sim_bits: Option<u32> = format
        .strip_prefix("qtip")
        .and_then(|s| s.strip_suffix("-sim"))
        .and_then(|n| n.parse::<u32>().ok())
        .filter(|b| matches!(b, 2 | 3 | 4 | 6 | 8));
    let use_qtip_sim = qtip_sim_bits.is_some();
    // roughquant-sim (Phase 1, no rotation): emit a bf16 .hfq where each 2D
    // weight's most-salient input columns (ranked by diag(H)) are kept exact
    // and the rest are crushed to a low-bit uniform grid, baked back into bf16
    // for a kernel-free PPL verdict. Swept via env:
    //   HIPFIRE_RQ_PROTECT_FRAC (default 0.015) — fraction of columns protected
    //   HIPFIRE_RQ_BULK_BITS    (default 2)      — bulk uniform bit-width
    //   HIPFIRE_RQ_GROUP        (default 256)    — bulk quant group size (cols)
    // The Hessian sidecar comes from HIPFIRE_QTIP_HESSIAN (shared with qtip-sim);
    // tensors without one fall back to a column-L2-norm saliency proxy.
    let use_roughquant_sim = format == "roughquant-sim" || format == "rq-sim";
    // roughquant2-sim (Phase 2): PCA-rotate each 2D weight into the eigenbasis of
    // its activation Hessian C=XᵀX, protect the top `protect_frac` highest-energy
    // columns at full precision, quantize the bulk with the QTIP trellis (which
    // supplies the within-tier Hadamard + low-bit format), then inverse-rotate
    // back. Kernel-free PPL verdict via the normal bf16 forward. Swept via env:
    //   HIPFIRE_RQ2_PROTECT_FRAC (default 0.015) — top columns kept exact
    //   HIPFIRE_RQ2_BULK_BITS    (default 3)     — QTIP trellis bits for the bulk
    //   HIPFIRE_RQ2_DAMP         (default 0.01)  — Hessian diagonal ridge fraction
    // Needs HIPFIRE_QTIP_HESSIAN; tensors without a Hessian are left bf16.
    let use_roughquant2_sim = format == "roughquant2-sim" || format == "rq2-sim";
    // roughquant3-sim (Phase 2c): Phase 2 with the dense PCA rotation replaced by
    // a PERMUTATION — reorder each weight's input columns by diag(H) saliency so
    // the salient channels become contiguous leading columns, protect them, QTIP
    // the bulk (QTIP's per-256 Hadamard is free, like mq4), un-permute back.
    // Permutations fold for FREE (reindex, no runtime matmul), unlike the dense
    // rotation that died in de-risk B. Tests whether reordering (no channel
    // mixing) keeps enough of the win to be deployable. Env:
    //   HIPFIRE_RQ3_PROTECT_FRAC (default 0.03), HIPFIRE_RQ3_BULK_BITS (default 3).
    let use_roughquant3_sim = format == "roughquant3-sim" || format == "rq3-sim";
    // roughquant4-sim (Phase 2d, the "think in channels" variant): channel-
    // consistent mixed precision on the residual stream — NO rotation, NO
    // permutation. Rank residual channels by aggregated activation energy once,
    // keep the top set high-res in (a) the COLUMNS of every residual reader
    // (inferred d_model input) AND (b) the ROWS of every residual writer
    // (inferred d_model output), so a high-energy channel is exact where it is
    // written and where it is read. Non-residual inputs (o_proj/down_proj's
    // internal activations) get per-weight diag(H) column protection. QTIP the
    // bulk. Folds for free (it's a per-channel bit map, no runtime transform). Env:
    //   HIPFIRE_RQ4_PROTECT_FRAC (default 0.03), HIPFIRE_RQ4_BULK_BITS (default 3),
    //   HIPFIRE_RQ4_Q8_EMBED.
    let use_roughquant4_sim = format == "roughquant4-sim" || format == "rq4-sim";
    // roughquant (REAL, shippable): the de-risked sim verdict in a real packed
    // format. Bulk = real MQ4G256 (existing kernel); protected residual channels
    // (diag(H)-selected, role-based: reader cols + writer rows) get an exact bf16
    // CORRECTION SIDECAR storing R = W − dequant(mq4(W)) over the protected set,
    // applied at GEMV time as y += R_S·x_S. No rotation/fold; absent sidecar =
    // plain mq4 (backward-compatible). diag is the production selector (ablation
    // oracle: Spearman 0.90). Env: HIPFIRE_RQ4_PROTECT_FRAC (default 0.03).
    let use_roughquant_real = format == "roughquant" || format == "rq";
    // permute5 (rq5): apply the #5 residual-stream permutation OFFLINE — cluster the
    // diag(H)-selected protected residual set S into a contiguous front block and
    // propagate the permutation across embed cols + every reader input-col + every
    // writer output-row + all dim-wide RMSNorm γ + lm_head. Bijective (output
    // unchanged) — the foundation for a gather-free RoughQuant correction. Weights
    // stay bf16 (verification: permuted vs original KLD ≈ 0). See
    // docs/roughquant/permutation-bijectivity.md.
    let use_roughquant5 = format == "permute5" || format == "rq5";
    // Real packed QTIP-3 (vs the bf16 sim): emit QuantType::Qtip3G256 records
    // (rotated-frame symbols), decoded by the gemv_qtip3g256 kernel. The sim
    // path is for kernel-free PPL; this is the shippable bandwidth format.
    let use_qtip3_real = format == "qtip3";
    let use_qtip4_real = format == "qtip4";
    let qtip_bits: u32 = if let Some(b) = qtip_sim_bits {
        b
    } else if use_qtip3_real {
        3
    } else if use_qtip4_real {
        4
    } else {
        2
    };
    // Both sim and real QTIP first stage every 2D weight as BF16, then the
    // post-pass either bakes sim error into bf16 (sim) or packs Qtip3G256 (real).
    let use_bf16 = format == "bf16"
        || format == "bfloat16"
        || use_qtip_sim
        || use_qtip3_real
        || use_qtip4_real
        || use_roughquant_sim
        || use_roughquant2_sim
        || use_roughquant3_sim
        || use_roughquant4_sim
        || use_roughquant_real
        || use_roughquant5;
    let use_source_precision = use_fp16 || use_bf16;
    let (qtip_cb, qtip_s1, qtip_s2) = if use_qtip_sim
        || use_qtip3_real
        || use_qtip4_real
        || use_roughquant2_sim
        || use_roughquant3_sim
        || use_roughquant4_sim
        || use_roughquant_real
        || use_roughquant5
    {
        if use_qtip3_real || use_qtip4_real {
            let bpg = if use_qtip4_real { 132 } else { 100 };
            eprintln!(
                "qtip{qtip_bits} (real): packing 2D weights as Qtip{qtip_bits}G256 \
                 (FWHT-rotated bitshift trellis beam=128, {bpg} B/group)"
            );
        } else {
            eprintln!(
                "qtip{qtip_bits}-sim: simulated QTIP-{qtip_bits} on 2D weights \
                 (FWHT + bitshift trellis beam=128) → bf16"
            );
        }
        // HIPFIRE_QTIP_CODEBOOK=3inst selects the QTIP 3INST computed codebook
        // (closer-to-Gaussian than 1MAD) for the SIM paths only — the real qtip
        // kernel computes 1MAD on-device, so its codebook must stay 1MAD.
        let cb = if !(use_qtip3_real || use_qtip4_real)
            && std::env::var("HIPFIRE_QTIP_CODEBOOK").as_deref() == Ok("3inst")
        {
            eprintln!("qtip: using 3INST computed codebook (sim)");
            qtip::build_codebook_3inst()
        } else {
            qtip::build_codebook()
        };
        (cb, gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256))
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    // Optional Hessian → QTIP-LDLQ (output-aware). HIPFIRE_QTIP_HESSIAN points
    // at a `.calib.hfq` (HFQM) package from the native collector; tensors with a
    // `<name>.hessian` use LDLQ, the rest fall back to plain (MSE) simulated QTIP.
    let qtip_hessian: Option<hessian_io::HessianSidecar> = if use_qtip_sim
        || use_roughquant_sim
        || use_roughquant2_sim
        || use_roughquant3_sim
        || use_roughquant4_sim
        || use_roughquant_real
        || use_roughquant5
    {
        std::env::var("HIPFIRE_QTIP_HESSIAN").ok().and_then(|p| {
            match hessian_io::HessianSidecar::open(std::path::Path::new(&p)) {
                Ok(s) => {
                    eprintln!(
                        "qtip{qtip_bits}-sim: LDLQ ENABLED — Hessian sidecar {p} ({} tensors)",
                        s.n_tensors()
                    );
                    Some(s)
                }
                Err(e) => {
                    eprintln!(
                        "qtip{qtip_bits}-sim: WARN cannot open Hessian {p}: {e:?} — plain QTIP"
                    );
                    None
                }
            }
        })
    } else {
        None
    };
    let use_q8 = format == "q8f16" || format == "q8";
    // F1 native-bf16 oracle: full-precision passthrough. Every tensor stored
    // as QuantType::F32 (qt=2) -- weights, norms, embeddings.
    let use_f32_passthrough = format == "f32" || format == "f32-passthrough" || format == "oracle";
    let use_mixed = format == "q8-mixed" || format == "mixed";
    let use_fast = format == "q8-fast" || format == "fast";
    let use_q8hfq = format == "q8hfq";
    let use_q4k_all = format == "q4k";
    let use_q4k_q8embed = format == "q4k-q8embed";
    let use_mq8g256 = format == "mq8" || format == "mq8g256";
    let use_oq4 = format == "op4"
        || format == "op4-4"
        || format == "op4g256"
        || format == "oq4"
        || format == "oq4g256"
        || format == "opus";
    let use_oq8 = format == "op8"
        || format == "op8-16"
        || format == "op8g256"
        || format == "oq8"
        || format == "oq8g256"
        || format == "opus8";
    let use_oq8_plus = format == "oq8+"
        || format == "oq8++"
        || format == "oq8-plus"
        || format == "op8+"
        || format == "op8-16+"
        || format == "op8-plus";
    let lfm2_oq_format = HfqInputFormat::from_flag(format).filter(|fmt| {
        matches!(
            fmt,
            HfqInputFormat::Oq4
                | HfqInputFormat::OqPlus
                | HfqInputFormat::OqPlusTiered
                | HfqInputFormat::OqPlusCompact
                | HfqInputFormat::Oq8
                | HfqInputFormat::Oq8Plus
        )
    });
    // DeepSeek V4 recipe (2026-05-20): routed experts → MQ2-Lloyd, every other
    // 2D weight → Q8F16, with norms/biases/HC matrices falling through
    // to the F16 fallback path via `should_quantize() == false`.
    // No K-map, no imatrix promotions, no source-dtype distinctions in
    // the quant branch — uniform Q8F16 for everything that's a real
    // matmul weight. Designed to re-quant DeepSeek-V4-Flash including
    // the MTP head at maximum precision for the dense path.
    let use_deepseek4_source_precision = format == "deepseek4-q8-mtp"
        || format == "deepseek4-q8"
        || format == "deepseek4-source-precision"
        || format == "deepseek4-source"
        || format == "deepseek4-mtp-precise";
    // deepseek4-mtp-precise: addon-only build (use with --include-prefix mtp.) that
    // keeps every mtp.0.* DENSE weight at F16 instead of Q8F16. Doubles the
    // addon size (~2 GB → ~3 GB) but eliminates Q8 quant noise on the MTP
    // attn projections, e_proj, h_proj, and shared experts. MTP is small
    // enough that the precision matters disproportionately — V3 paper's
    // 60-80% acceptance benchmark assumes weights at training precision,
    // not 8-bit. Routed experts stay MQ2-Lloyd (no precision-upgrade option
    // available without a new MoE GEMV kernel).
    let use_mtp_precise = format == "deepseek4-mtp-precise";
    let use_mq4g256 = format == "mq4" || format == "mq4g256" || format == "magnum";
    let use_hfq4g256 = format == "hfq4g256" || format == "hfq4" || format == "hf4";
    let use_hfq3g256 = format == "hfq3g256";
    let use_hfq3g128 = format == "hfq3g128" || format == "hfq3" || format == "hf3"; // default HF3 = G128
    let use_hfq2g256 = format == "hfq2g256";
    let use_hfq2g128 = format == "hfq2g128" || format == "hfq2" || format == "hf2";
    let use_hfq_mixed = format == "hfq-mixed"; // Q8 attn + HFQ4 FFN
    let use_mq6g256 = format == "mq6" || format == "mq6g256";
    // Mixed: MQ4 for attention/shared-expert + MQ6 for routed experts only.
    // Saves ~15 GB vs full MQ6 on 122B-A10B (75 GB vs 90 GB), fits in 125 GB UMA.
    let use_mq4_mq6exp = format == "mq4-mq6exp" || format == "mq4-mq6experts";
    // Round-trip quality probe: route routed-MoE experts through MQ2-Lloyd
    // quantize → dequantize → re-quantize as HFQ4. The .hfq ships as plain
    // MQ4 (HFQ4G256), no runtime changes. Measures whether 2-bit noise on
    // routed experts survives the MoE sparse-usage rescue, before sinking
    // a week into new MoE-2bit GEMV kernels.
    let use_mq4_routed_lloyd_mq2_exp =
        format == "mq4-routed-lloyd-mq2-exp" || format == "mq4-routed-lloyd-mq2-experts";
    if use_mq4_routed_lloyd_mq2_exp {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq2-exp is a quality probe — routed MoE\n\
             experts go through MQ2-Lloyd round-trip (quantize → dequantize)\n\
             before being re-quantized as MQ4. Output is shipped as plain\n\
             MQ4 (no runtime changes needed). Measures whether MoE sparse\n\
             usage rescues MQ2-Lloyd at the experts before investing in new\n\
             MoE-2bit GEMV kernels."
        );
    }
    // Native Phase-2 form: routed MoE experts ship as native MQ2G256Lloyd
    // (qt=19). Requires runtime support — the qwen35 MoE forward path must
    // dispatch the new gemv_mq2g256_lloyd_moe_*_indexed* kernels (or fall
    // through to weight_gemv's MQ2G256Lloyd arm for the slow per-expert
    // path).
    let use_mq4_routed_lloyd_mq2_native = format == "mq4-routed-lloyd-mq2-native"
        || format == "mq4-routed-lloyd-mq2-exp-native"
        || format == "mq4-routed-lloyd-mq2";
    // kmap-respecting variant: like mq4-routed-lloyd-mq2-native, but routed-expert
    // tensors that the kmap flags as Promote6 stay at MQ6 (instead of being
    // demoted to MQ2-Lloyd). Reduces precision-loss on the ~30% of layers
    // that the alternating K-map identifies as important. Larger file
    // (extra MQ6 layers) but expected to recover quality on attractor-prone
    // prompts that mq4-routed-lloyd-mq2-native truncated early.
    let use_mq4_routed_lloyd_mq2_kmap = format == "mq4-routed-lloyd-mq2-kmap"
        || format == "mq4-routed-lloyd-mq2-respectkmap"
        || format == "mq4-routed-lloyd-mq2-kmap-promote";
    // Imatrix-weighted variant: like mq4-routed-lloyd-mq2-kmap, but the Lloyd
    // codebook for each non-promoted expert is fit with per-column
    // importance weights from a llama.cpp imatrix file (--imatrix flag).
    // The kmap-promoted ~30 % of expert layers still stay at MQ6.
    let use_mq4_routed_lloyd_mq2_imatrix = format == "mq4-routed-lloyd-mq2-imatrix"
        || format == "mq4-routed-lloyd-mq2-kmap-imatrix"
        || format == "mq4-routed-lloyd-mq2-imatrix-kmap";
    // MQ3-Lloyd-on-routed-experts: 3 bpw alternative when 2 bpw isn't enough.
    // Kmap-respecting: promoted experts → MQ6, rest → MQ3-Lloyd (qt=20).
    // No imatrix variant for MQ3 in this commit — MQ3-Lloyd is empirically
    // production-grade on Qwen3.5-MoE A3B, so uniform Lloyd is the baseline.
    let use_mq4_routed_lloyd_mq3_kmap = format == "mq4-routed-lloyd-mq3-kmap"
        || format == "mq4-routed-lloyd-mq3"
        || format == "mq4-routed-lloyd-mq3-exp";
    let allow_mq3_lloyd_for_mixed = args.iter().any(|a| a == "--allow-mq3-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ3_LLOYD").ok().as_deref() == Some("1");
    if use_mq4_routed_lloyd_mq3_kmap && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq3-kmap requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (same gate as bare --format lloyd-mq3)."
        );
        std::process::exit(2);
    }
    if use_mq4_routed_lloyd_mq3_kmap {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq3-kmap ships routed experts as MQ3G256Lloyd\n\
             (qt=20, 112 B / 256 weights, 3.5 bpw). Promoted experts stay at MQ6.\n\
             3 bpw fallback when 2 bpw can't avoid attractors on code-gen."
        );
    }
    // Phase 5: importance-aware MQ2/MQ3 layer tiering. Requires --imatrix.
    // Per-layer aggregate counts rank layers by routing activity; the top
    // `tier_ratio` fraction of NON-PROMOTED layers gets MQ3-Lloyd (3.5 bpw)
    // for higher precision on hot layers, the bottom fraction gets
    // MQ2-Lloyd (2.25 bpw) for size. K-map-promoted layers stay at MQ6.
    //
    // Granularity is PER LAYER (not per expert within a layer) because the
    // MoE-indexed kernels require uniform dtype across experts within a
    // tensor — the kernel reads expert_ptrs and assumes a fixed byte
    // stride per group (72 B for MQ2 vs 112 B for MQ3).
    let use_mq4_routed_lloyd_mq_tiered = format == "mq4-routed-lloyd-mq-tiered"
        || format == "mq4-routed-lloyd-mq-tiered-imatrix"
        || format == "routed-lloyd-mq-tiered";
    // Phase 6: antirez-style asymmetric-tensor recipe. Routed-expert
    // gate_up_proj → MQ2-Lloyd (imatrix-weighted), routed-expert
    // down_proj → MQ3-Lloyd (no imatrix, fixed-precision protection of
    // the residual-write direction). K-map promoted layers still get
    // MQ6 on both tensors.
    //
    // Rationale: antirez (V4 Flash) uses IQ2_XXS on up/gate and Q2_K
    // on down. The empirical claim is that `down` is the more sensitive
    // direction because it writes back into the residual stream — gate/up
    // errors get partially absorbed by silu. Mirror that asymmetry in
    // MQ-family: 2-bit on gate_up, 3-bit on down.
    let use_mq4_routed_lloyd_mq_antirez = format == "mq4-routed-lloyd-mq-antirez"
        || format == "mq4-routed-lloyd-mq-asym"
        || format == "antirez-mq";
    // Lever 2: same recipe as antirez but with sequential-GPTQ Lloyd
    // on the gate_up_proj path instead of plain imatrix-weighted Lloyd.
    // Aims to reduce attractor risk at 2 bpw — if successful, opens path
    // to ALL-MQ2 routed experts (no down=MQ3 compensation needed) and
    // a further size reduction.
    let use_mq4_routed_lloyd_mq_antirez_gptq = format == "mq4-routed-lloyd-mq-antirez-gptq"
        || format == "mq4-routed-lloyd-mq-asym-gptq"
        || format == "antirez-mq-gptq";
    if use_mq4_routed_lloyd_mq_antirez_gptq && imatrix_path.is_none() {
        eprintln!("error: --format mq4-routed-lloyd-mq-antirez-gptq requires --imatrix <PATH>");
        std::process::exit(2);
    }
    if use_mq4_routed_lloyd_mq_antirez_gptq && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq-antirez-gptq requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
        );
        std::process::exit(2);
    }
    if use_mq4_routed_lloyd_mq_antirez_gptq {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq-antirez-gptq — same routed-expert split\n\
             as antirez (gate_up=MQ2-Lloyd, down=MQ3-Lloyd), but gate_up uses\n\
             SEQUENTIAL-error-feedback Lloyd (simplified GPTQ-LDLQ) for\n\
             reduced attractor risk at 2 bpw."
        );
    }
    // All-MQ2-GPTQ: route BOTH gate_up AND down through MQ2-Lloyd-GPTQ.
    // Tests whether sequential error feedback closes the attractor gap
    // enough to drop the down=MQ3 compensation antirez uses, saving
    // ~30 % more on routed-expert size.
    let use_mq4_routed_lloyd_mq2_gptq_all = format == "mq4-routed-lloyd-mq2-gptq-all"
        || format == "mq4-routed-lloyd-mq2-gptq"
        || format == "all-mq2-gptq";
    if use_mq4_routed_lloyd_mq2_gptq_all
        && imatrix_path.is_none()
        && std::env::var("HIPFIRE_ALLOW_UNIT_IMATRIX").ok().as_deref() != Some("1")
    {
        eprintln!("error: --format mq4-routed-lloyd-mq2-gptq-all requires --imatrix <PATH>");
        eprintln!(
            "       (DeepSeek V4: set HIPFIRE_ALLOW_UNIT_IMATRIX=1 to use unit column weights —"
        );
        eprintln!(
            "        captures GPTQ sequential error-feedback win without imatrix calibration.)"
        );
        std::process::exit(2);
    }
    if use_mq4_routed_lloyd_mq2_gptq_all {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq2-gptq-all — ALL routed experts (both\n\
             gate_up AND down) at MQ2-Lloyd with sequential-GPTQ codebook\n\
             assignment. Tests the size-reduction hypothesis from Lever 2."
        );
    }
    if use_mq4_routed_lloyd_mq_antirez {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-routed-lloyd-mq-antirez requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-routed-lloyd-mq-antirez requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-routed-lloyd-mq-antirez ships routed experts as\n\
             gate_up_proj → MQ2-Lloyd (imatrix-weighted, qt=19), down_proj\n\
             → MQ3-Lloyd (qt=20). K-map-promoted layers stay at MQ6 on both.\n\
             Mirrors antirez/ds4 V4 Flash recipe (IQ2_XXS gate/up, Q2_K down).\n\
             Estimated DeepSeek V4 size: 70% × MQ2 + 20% × MQ3 + 10% × MQ4 ≈ 96 GB."
        );
    }
    let tier_ratio: f64 = args
        .iter()
        .position(|a| a == "--tier-ratio")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .or_else(|| {
            std::env::var("HIPFIRE_TIER_RATIO")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(0.30);
    if use_mq4_routed_lloyd_mq_tiered {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-routed-lloyd-mq-tiered requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-routed-lloyd-mq-tiered requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (uses MQ3-Lloyd on the hot layers)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-routed-lloyd-mq-tiered uses imatrix .counts to rank\n\
             routed-expert layers by aggregate activation. Top {:.0}% of\n\
             non-promoted layers go to MQ3-Lloyd (3.5 bpw); the rest go to\n\
             MQ2-Lloyd (2.25 bpw). K-map-promoted layers stay at MQ6.",
            tier_ratio * 100.0
        );
    }
    if use_mq4_routed_lloyd_mq2_imatrix {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-routed-lloyd-mq2-imatrix requires --imatrix <PATH>");
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-routed-lloyd-mq2-imatrix uses per-column importance\n\
             weights from the supplied calibration imatrix. Promoted experts\n\
             still stay at MQ6 (kmap-respect). Falls back to uniform Lloyd\n\
             for any expert whose imatrix tensor is missing."
        );
    }
    if use_mq4_routed_lloyd_mq2_kmap {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq2-kmap respects K-map promotion —\n\
             experts flagged Promote6 (~30 % of layers) stay at MQ6G256;\n\
             remaining ~70 % get MQ2G256Lloyd (qt=19). File size is larger\n\
             than mq4-routed-lloyd-mq2-native but quality on attractor-prone prompts\n\
             should be markedly better."
        );
    }
    if use_mq4_routed_lloyd_mq2_native {
        eprintln!(
            "note: --format mq4-routed-lloyd-mq2-native ships routed MoE experts as\n\
             native MQ2G256Lloyd (qt=19, 72 B/group). Runtime must support\n\
             the MQ2-Lloyd MoE dispatch (weight_gemv arm exists; indexed\n\
             fast path requires forward-path arms in hipfire-arch-qwen35)."
        );
    }
    if use_mq4_mq6exp {
        eprintln!(
            "warning: --format mq4-mq6exp is deprecated. Use --format mq4 instead — \
             K-map promotes expert FFNs (and edge layers) to MQ6 automatically. \
             Proceeding as --format mq4."
        );
    }
    let use_mq3g256 = format == "mq3" || format == "mq3g256";
    let use_mq2g256 = format == "mq2" || format == "mq2g256";
    let use_lloyd_mq2g256 = format == "lloyd-mq2";
    let use_lloyd_mq3g256 = format == "lloyd-mq3";
    let use_lloyd_mq4g256 = format == "lloyd-mq4";
    let use_hfq6 = format == "hfq6" || format == "hfq6g256" || format == "hf6";
    // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale). Spec at docs/quant-formats/hfp4.md.
    let use_hfp4 = format == "hfp4" || format == "hfp4g32" || format == "hf4p" || format == "fp4";
    // MFP4G32 — HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same per-row layout
    // as HFP4G32 with format_flags bit 0 + bits 2-3 = 01 stamping the rotation kind.
    let use_mfp4 = format == "mfp4" || format == "mfp4g32" || format == "mf4p";
    let q8_router_flag = args.iter().any(|a| a == "--q8-router");
    // Conv1d (DeltaNet) defaults to Q8 regardless of --format — the tensor is
    // small (~32K elem) but runs every token and lossy 4-bit FWHT formats
    // measurably hurt the gated-delta path. Override with --no-q8-conv1d to
    // keep conv1d at the same quant as the rest of the model.
    let q8_conv1d_default = !args.iter().any(|a| a == "--no-q8-conv1d");
    let no_kmap = args.iter().any(|a| a == "--no-kmap" || a == "--uniform");

    // ── imatrix loader (consumed by AWQ pre-scaling) ──
    // --imatrix <path>: load an llama-imatrix-produced GGUF (per `examples/
    // imatrix_collect.rs`). Populates the IMATRIX OnceLock with per-channel
    // `Σ_token act²` values keyed by ggml-style tensor name. Quantizer behavior
    // with no `--imatrix` is byte-equivalent to baseline.
    //
    // For Qwen3.5 hybrid layers, the mapper covers: ffn_{gate,up,down},
    // self_attn.{q,k,v,o}_proj (full-attention layers), and
    // linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}
    // (linear-attention layers via SSM-naming). Norms / biases / 1D scalars /
    // conv1d / lookup tables have no imatrix entry.
    let imatrix_path = args
        .iter()
        .position(|a| a == "--imatrix")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    if let Some(path) = &imatrix_path {
        if !path.exists() {
            eprintln!("error: --imatrix path not found: {}", path.display());
            std::process::exit(1);
        }
        let table = load_imatrix(path);
        IMATRIX
            .set(table)
            .expect("IMATRIX set twice — should not happen");
        eprintln!("imatrix loaded from {}", path.display());
    }

    // --hessian <path>: reuse an existing HFQM `.calib.hfq` or legacy HFHS-v1
    // *.hessian.bin as the AWQ activation statistic. For HFQM, explicit
    // `<name>.imatrix` vectors are imported first-class; full-Hessian tensors
    // also contribute their diagonal because H[j,j] = Σ_token x[j]² is AWQ's
    // in_sum2[j]. Keyed by the weight tensor name (AWQ looks up
    // `<...>.weight`). Ignored if --imatrix already populated IMATRIX.
    if IMATRIX.get().is_none() {
        if let Some(hpath) = args
            .iter()
            .position(|a| a == "--hessian")
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
        {
            if !hpath.exists() {
                eprintln!("error: --hessian path not found: {}", hpath.display());
                std::process::exit(1);
            }
            match hessian_io::HessianSidecar::open(&hpath) {
                Ok(sc) => {
                    let n_hessian = sc.n_tensors();
                    let n_imatrix = sc.n_imatrix_tensors();
                    let mut table: HashMap<String, Vec<f32>> =
                        HashMap::with_capacity((n_hessian + n_imatrix) * 2);
                    for href in sc.tensors() {
                        let diag: Vec<f32> = (0..href.k).map(|i| href.at(i, i) as f32).collect();
                        table.insert(format!("{}.weight", href.name), diag.clone());
                        table.insert(href.name.to_string(), diag);
                    }
                    for iref in sc.imatrices() {
                        let imatrix: Vec<f32> = iref.iter_f32().collect();
                        table.insert(format!("{}.weight", iref.name), imatrix.clone());
                        table.insert(iref.name.to_string(), imatrix);
                    }
                    let n = table.len();
                    IMATRIX
                        .set(table)
                        .expect("IMATRIX set twice — should not happen");
                    eprintln!(
                        "imatrix derived from HFQM calibration ({hpath:?}): {n} keys \
                         ({n_hessian} Hessian diagonals, {n_imatrix} imatrix vectors)"
                    );
                }
                Err(_) => match hfhs_diag::read_diagonals(&hpath) {
                    Ok(diags) => {
                        let mut table: HashMap<String, Vec<f32>> =
                            HashMap::with_capacity(diags.len() * 2);
                        for (name, diag) in diags {
                            table.insert(format!("{name}.weight"), diag.clone());
                            table.insert(name, diag); // bare-name fallback
                        }
                        let n = table.len();
                        IMATRIX
                            .set(table)
                            .expect("IMATRIX set twice — should not happen");
                        eprintln!("imatrix derived from hessian diagonals ({hpath:?}): {n} keys");
                    }
                    Err(e) => {
                        eprintln!("error: --hessian read failed ({}): {e}", hpath.display());
                        std::process::exit(1);
                    }
                },
            }
        }
    }

    // --ldlq / OQ++: full-Hessian error-feedback weight quant for calibrated plus
    // formats. Loads the full [K,K] payloads from the same --hessian file the
    // AWQ diagonal came from. Requires --hessian.
    let ldlq_requested = oq_ldlq_recipe || args.iter().any(|a| a == "--ldlq");
    if ldlq_requested {
        match args
            .iter()
            .position(|a| a == "--hessian")
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
        {
            Some(hpath) if hpath.exists() => {
                let full = match hessian_io::HessianSidecar::open(&hpath) {
                    Ok(sc) => {
                        eprintln!("ldlq: HFQM full Hessian index opened ({hpath:?})");
                        Oq4LdlqHessian::Hfqm(sc)
                    }
                    Err(_) => match hfhs_diag::HfhsFull::open(&hpath) {
                        Ok(sc) => {
                            eprintln!("ldlq: HFHS full Hessian index opened ({hpath:?})");
                            Oq4LdlqHessian::Hfhs(sc)
                        }
                        Err(e) => {
                            eprintln!("error: --ldlq could not open full Hessian ({hpath:?}): {e}");
                            std::process::exit(1);
                        }
                    },
                };
                OQ4_LDLQ_HESSIAN
                    .set(full)
                    .ok()
                    .expect("OQ4_LDLQ_HESSIAN set twice");
            }
            _ => {
                eprintln!(
                    "error: --format oq4++/oq8++ or --ldlq requires --hessian <HFHS .hessian.bin>"
                );
                std::process::exit(1);
            }
        }
    }

    // ── Phase A Stage A: AWQ (Activation-aware Weight Quantization) ──
    // --awq           → enable AWQ at default alpha=0.55
    // --awq-alpha <f> → enable AWQ at explicit alpha (overrides default)
    // Requires --imatrix (we derive RMS_act from imatrix's in_sum2 values).
    // Per-channel scaling: W' = W · diag(s) at quantize time, sidecar
    // 1D F16 tensor <weight>.awq_scale stored alongside the parent weight.
    // Runtime path divides activations by s before the rotation kernel —
    // separate change, not in this patch. Implementation reference:
    // docs/plans/awq_hipfire.md.
    //
    // Stage A targets MQ4G256 specifically (large g=256 → AWQ's outlier-
    // mitigation works; per Egiazarian et al 2509.23202 §3.2, small-group
    // formats (g=16/32 NVFP4/MXFP4) "provably neutralize traditional
    // outlier mitigation techniques" — MR-GPTQ is the right lever there,
    // tracked as Stage C). HFP4/MFP4 are explicitly NOT awq-pre-scaled
    // in this patch.
    let awq_enabled = mq_plus
        || oq_plus_recipe
        || args.iter().any(|a| a == "--awq")
        || args.iter().any(|a| a == "--awq-alpha");
    // Explicit `--awq-alpha <f>` always wins. When absent, the default is
    // resolved *after* arch detection (see the per-arch `AWQ_ALPHA.set` below),
    // because nemotron_h's Mamba activations need a much lower alpha than the
    // transformer default. Nothing between here and that point reads AWQ_ALPHA.
    let awq_alpha_explicit = args
        .iter()
        .position(|a| a == "--awq-alpha")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f32>().ok());
    if awq_enabled && IMATRIX.get().is_none() {
        eprintln!(
            "error: activation-aware quantization requires --imatrix or --hessian \
             (we derive RMS_act per channel from imatrix/Hessian diagonal values)"
        );
        std::process::exit(1);
    }
    // --sq-split [<frac>]: outlier-aware SmoothQuant (separate geo-mean
    // normalization for the top-frac activation-energy channels vs the bulk).
    // Default frac = 0.01. Requires AWQ to be active (it tunes the AWQ scale).
    if let Some(i) = args.iter().position(|a| a == "--sq-split") {
        let frac = args
            .get(i + 1)
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|f| *f > 0.0 && *f < 1.0)
            .unwrap_or(0.01);
        if !awq_enabled {
            eprintln!(
                "warning: --sq-split has no effect without --awq (it normalizes the AWQ scale)"
            );
        }
        SQ_OUTLIER_SPLIT
            .set(frac)
            .expect("SQ_OUTLIER_SPLIT set twice — should not happen");
        eprintln!(
            "SmoothQuant outlier-split: ENABLED (outlier_frac={frac}, top-{:.2}% channels by energy normalized separately from the bulk)",
            frac * 100.0
        );
    }
    // --w8-top <frac>: OQ+ magnitude-tiered top-frac weights kept at W8A8.
    if let Some(i) = args.iter().position(|a| a == "--w8-top") {
        let frac = args
            .get(i + 1)
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|f| *f > 0.0 && *f < 1.0)
            .unwrap_or(0.01);
        OQPLUS_W8_FRAC
            .set(frac)
            .expect("OQPLUS_W8_FRAC set twice — should not happen");
        eprintln!(
            "OQ+ magnitude-tiering: ENABLED (top-{:.2}% weights/group kept at W8A8, bulk W4A8, single iu8 kernel)",
            frac * 100.0
        );
    }

    // `--beam N` sets the QTIP trellis beam-search width (default 128, near-
    // Viterbi). Smaller = much faster offline encode on big models, slightly
    // lower quality. Bridged through HIPFIRE_QTIP_BEAM so both the .hfq-requant
    // and direct-source pack paths (`qtip_beam_width`) read one value.
    if let Some(i) = args.iter().position(|a| a == "--beam") {
        if let Some(b) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()).filter(|&b| b > 0) {
            std::env::set_var("HIPFIRE_QTIP_BEAM", b.to_string());
            eprintln!("QTIP trellis beam width: {b} (--beam)");
        } else {
            eprintln!("--beam requires a positive integer; ignoring");
        }
    }
    // K-map gate: applies to MoE models by default. Dense models opt in
    // via --kmap-dense (the K-map dense PPL effect is mixed: regression at
    // short context, win at long context — see benchmarks/results/
    // ppl_kmap_20260508.md). Maintainer directive 2026-05-08: "intends to
    // help ONLY (never on dense)" by default.
    let kmap_dense = args.iter().any(|a| a == "--kmap-dense");
    // K-map mode: 0=full (all candidates promoted), 1=alternating (edge + every 3rd),
    // 2=typed (ffn_down+attn_v everywhere). Default: alternating — same PPL as full
    // at 17% less model size on MoE (22.9 vs 27.7 GB, PPL 8K: 19.96 vs 20.07).
    let kmap_mode: u8 = args
        .iter()
        .position(|a| a == "--kmap-mode")
        .and_then(|i| args.get(i + 1))
        .map(|v| match v.as_str() {
            "full" | "0" => 0,
            "alternating" | "alt" | "1" => 1,
            "typed" | "2" => 2,
            _ => {
                eprintln!("warning: unknown --kmap-mode '{v}', using alternating");
                1
            }
        })
        .unwrap_or(1);

    // ── Sub-4-bit guards (2026-04-30 sweep) ─────────────────────────────
    // MQ2 with the current uniform 4-level codebook collapses at every
    // model size validated locally (0.8B / 4B / 9B Qwen 3.5 → multilingual
    // mojibake on all 4 coherence-gate prompts). Refuse by default until
    // Path D Lloyd-Max non-uniform codebooks land (PRD §5.2).
    let allow_mq2 = args.iter().any(|a| a == "--allow-mq2")
        || std::env::var("HIPFIRE_ALLOW_MQ2").ok().as_deref() == Some("1");
    if use_mq2g256 && !allow_mq2 {
        eprintln!(
            "error: --format mq2 is reserved — empirical quality verdict is collapse on every model\n\
             size validated locally (0.8B / 4B / 9B Qwen 3.5 → mojibake / symbol soup on all 4\n\
             coherence-gate prompts). The current uniform 4-level codebook is fundamentally too\n\
             lossy; Path D Lloyd-Max non-uniform codebooks (per-block squared-error-minimising)\n\
             are the planned remediation per PRD §5.2.\n\
             \n\
             To opt in for research / ablation purposes anyway, pass --allow-mq2 or set\n\
             HIPFIRE_ALLOW_MQ2=1. Don't ship MQ2 artifacts to users until the codebook\n\
             improvement lands."
        );
        std::process::exit(1);
    }
    // MQ2-Lloyd: rescues uniform MQ2 by 41–55× (per benchmarks/results/
    // lloyd_max_findings_20260501.md) but still text-collapse — 9B ppl=2,163
    // vs 9B MQ4 ppl=10. Research-only: same opt-in gate so users don't
    // accidentally ship a 2-bpw model that won't produce coherent output.
    let allow_mq3_lloyd = args.iter().any(|a| a == "--allow-mq3-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ3_LLOYD").ok().as_deref() == Some("1");
    if use_lloyd_mq3g256 && !allow_mq3_lloyd {
        eprintln!(
            "note: --format lloyd-mq3 is research — Lloyd-Max 8-entry codebook +\n\
             3-bit indices (112 B/group, +7.7% over uniform MQ3). Hypothesis is\n\
             non-uniform codebook lifts sub-9B MQ3 out of collapse (#114) and\n\
             tightens 9B MQ3's 4× ppl gap vs MQ4. Ppl evidence pending — DO NOT\n\
             ship MQ3-Lloyd artifacts to users until quality is validated against\n\
             baseline MQ3/MQ4 ppl.\n\
             \n\
             To proceed, pass --allow-mq3-lloyd or set HIPFIRE_ALLOW_MQ3_LLOYD=1."
        );
        std::process::exit(1);
    }
    let allow_mq2_lloyd = args.iter().any(|a| a == "--allow-mq2-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ2_LLOYD").ok().as_deref() == Some("1");
    if (use_lloyd_mq2g256
        || use_mq4_routed_lloyd_mq2_exp
        || use_mq4_routed_lloyd_mq2_native
        || use_mq4_routed_lloyd_mq2_kmap
        || use_mq4_routed_lloyd_mq2_imatrix
        || use_mq4_routed_lloyd_mq3_kmap
        || use_mq4_routed_lloyd_mq2_kmap
        || use_mq4_routed_lloyd_mq_tiered
        || use_mq4_routed_lloyd_mq_antirez
        || use_mq4_routed_lloyd_mq_antirez_gptq
        || use_mq4_routed_lloyd_mq2_gptq_all
        || use_deepseek4_source_precision)
        && !allow_mq2_lloyd
    {
        eprintln!(
            "error: --format lloyd-mq2 is research-only — Lloyd-Max codebook lifts\n\
             uniform MQ2 by 41–55× ppl but absolute quality is still collapse\n\
             (9B Qwen 3.5 wikitext2-test ppl=2,163 vs MQ4=10, MQ3=42; 0.8B ppl=19,651).\n\
             2 bpw is fundamentally too aggressive for usable text; the format\n\
             is plumbed for follow-on Lloyd-Max MQ3 (qt=20) experiments only.\n\
             \n\
             To opt in for research anyway, pass --allow-mq2-lloyd or set\n\
             HIPFIRE_ALLOW_MQ2_LLOYD=1. Don't ship MQ2-Lloyd artifacts to users."
        );
        std::process::exit(1);
    }
    // MQ4-Lloyd: extension of MQ3-Lloyd to K=16 centroids. Conjectured to
    // narrow the MQ4 → MQ6 ppl gap at +17.6% bandwidth over uniform MQ4
    // (160 vs 136 B/group). Per
    // benchmarks/results/devlog_20260506_lloyd_mq4_extension.md the
    // 9B projection is ppl 8.0–9.3 (vs uniform MQ4 ppl 10.34, MQ6 ppl 9.36).
    // Quality not yet validated — same opt-in gate as MQ3-Lloyd until ppl
    // numbers land.
    let allow_mq4_lloyd = args.iter().any(|a| a == "--allow-mq4-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ4_LLOYD").ok().as_deref() == Some("1");
    if use_lloyd_mq4g256 && !allow_mq4_lloyd {
        eprintln!(
            "note: --format lloyd-mq4 is research — Lloyd-Max 16-entry codebook +\n\
             4-bit indices (160 B/group, +17.6% over uniform MQ4). Hypothesis is\n\
             non-uniform codebook narrows the MQ4 → MQ6 ppl gap at lower bandwidth\n\
             than uniform MQ6. Ppl evidence pending — DO NOT ship MQ4-Lloyd\n\
             artifacts to users until quality is validated against baseline\n\
             MQ4/MQ6 ppl on the target model.\n\
             \n\
             To proceed, pass --allow-mq4-lloyd or set HIPFIRE_ALLOW_MQ4_LLOYD=1."
        );
        std::process::exit(1);
    }
    // MQ3 quality threshold ≈ 9B from the same sweep — 27B + 9B fluent,
    // 4B partial-collapse (intent recognised, language drifts), 0.8B
    // gibberish. Print a soft advisory so users running --format mq3
    // against small models don't think the engine is broken.
    if use_mq3g256 {
        eprintln!(
            "note: MQ3 empirical quality threshold ≈ 9B params. 27B / 9B Qwen 3.5 produce\n\
             fluent output across the coherence-gate battery; 4B partially collapses\n\
             (intent recognised, language mixes / loops); 0.8B is incoherent. For models\n\
             below ~9B, prefer --format mq4 (same kernel family, ~30% larger but\n\
             reliably coherent).\n"
        );
    }

    // GGUF input branch: if --input is a `.gguf` file, run the GGUF
    // pipeline and exit. Tensor names are translated GGUF → safetensors
    // style. The 2D quantization target follows --format:
    //   hfq4 | hfq6 | mq4 | mq6
    // Per CLAUDE.md guidance: dense (non-DeltaNet) models should use
    // hfq4/hfq6. mq4/mq6 are calibrated for Qwen3.5+ — using them on a
    // Llama-style model produces correct output (the FWHT cancels in
    // `gemv_mq4g256_with_rotate`) but adds runtime rotation overhead
    // with no quality benefit.
    {
        let raw_input = Path::new(input_dir);
        if is_hfq_input(raw_input) {
            let hfq_format = HfqInputFormat::from_flag(format).unwrap_or_else(|| {
                eprintln!(
                    "HFQ input: --format '{format}' not recognized. \
                     Supported: bf16, fp16, q8f16, hfq4, hfq6, mq4, mq6, mq3, qtip3."
                );
                std::process::exit(2);
            });
            let out = Path::new(output_path);
            if let Err(e) = run_hfq_source_pipeline(raw_input, out, hfq_format, format) {
                eprintln!("HFQ input pipeline failed: {e}");
                std::process::exit(2);
            }
            return;
        }
        if is_gguf_input(raw_input) {
            let gguf_format = GgufFormat::from_flag(format).unwrap_or_else(|| {
                eprintln!(
                    "GGUF input: --format '{format}' not recognized. \
                     Supported: bf16, fp16, hfq4, hfq6, mq4, mq6, mq3, mq2, lloyd-mq*, hfp4, mfp4."
                );
                std::process::exit(2);
            });
            let out = Path::new(output_path);
            if let Err(e) = run_gguf_pipeline(
                raw_input,
                out,
                gguf_format,
                format,
                no_kmap,
                kmap_dense,
                kmap_mode,
            ) {
                eprintln!("GGUF pipeline failed: {e}");
                std::process::exit(2);
            }
            return;
        }
    }

    // Resolve input: local path or HuggingFace model ID (e.g. "Qwen/Qwen3-8B")
    let input_dir = resolve_model_path(input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(output_path);

    // Read model config
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|_| panic!("Cannot read {}. If using a HuggingFace model ID, ensure it's downloaded: huggingface-cli download {}", config_path.display(), input_dir.display()));
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let is_mamba2_config = config
        .get("ssm_cfg")
        .and_then(|v| v.get("layer"))
        .and_then(|v| v.as_str())
        .map(|s| s.eq_ignore_ascii_case("mamba2"))
        .unwrap_or(false)
        || config
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|archs| {
                archs
                    .iter()
                    .filter_map(|v| v.as_str())
                    .any(|arch| arch.to_ascii_lowercase().contains("mamba2"))
            })
            .unwrap_or(false);
    let arch_str = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");
    let auto_arch_id = if is_mamba2_config {
        15
    } else {
        match arch_str {
            "llama" => 0u32,
            "qwen3" | "qwen2" => 1,
            "qwen3_5" | "qwen3_5_text" => 5,
            // Qwen3.5 MoE (Qwen3.5-35B-A3B and friends): hybrid LA+FA attention identical
            // to qwen3_5 dense, but every layer's FFN is MoE with stacked-3D expert
            // tensors (mlp.experts.gate_up_proj/down_proj are [num_experts, ...]).
            "qwen3_5_moe" | "qwen3_5_moe_text" => 6,
            // dots.ocr (Qwen2-VL family layout-extraction VLM): plain Qwen2-1.5B
            // text decoder + 42-block DotsVisionTransformer with 2-D RoPE,
            // SwiGLU, RMSNorm. Crate: hipfire-arch-dots-ocr. See docs/plans/
            // dots-ocr-prd.md.
            "dots_ocr" => 8,
            // DeepSeek V4 Flash: 256 routed + 1 shared experts, Hyper-Connections,
            // compressed-KV indexer, FP8 E4M3 + UE8M0 block-scale storage. See
            // crates/hipfire-arch-deepseek4. Phase 1 ingest only — no forward
            // path yet; tensor names ship in DeepSeek V4's native shape (split w1/w2/w3,
            // per-expert) and are translated when the forward bring-up lands.
            "deepseek_v4" => 9,
            // MiniMax-M2 (Mixtral-style MoE): GQA + per-layer QK-norm + partial
            // rotate_half RoPE; 256 routed experts top-8 sigmoid+e_score_bias, no
            // shared expert; FP8 E4M3 + F32 weight_scale_inv block-128 storage;
            // split per-expert w1/w3/w2 (like deepseek_v4). Crate hipfire-arch-minimax.
            "minimax_m2" => 10,
            // LFM2.5 (LiquidAI): hybrid short-conv + GQA-attn layers, SwiGLU FFN.
            //   "lfm2_moe" = A1B (dense MLP head layers + top-4 MoE); per-expert
            //               pre-split w1/w2/w3 → MQ4G256, everything else → Q8.
            //   "lfm2"     = dense (Lfm2ForCausalLM, e.g. 350M/1.2B) — no experts,
            //               every layer dense SwiGLU; the ingest Q8s all tensors.
            // Crate hipfire-arch-lfm2moe (arch_id 11); loader handles both via
            // num_dense_layers == num_hidden_layers for the dense variant.
            "lfm2_moe" | "lfm2" => 11,
            // Gemma3 (text). `gemma3_text` = Gemma3ForCausalLM (clean
            // model.layers.* names, e.g. medgemma-27b-text-it); `gemma3` =
            // Gemma3ForConditionalGeneration (multimodal wrapper — text fields
            // under text_config, SigLIP vision deferred to arch_id 13). Dense
            // decoder with the Gemma quirks: (1+w) zero-centered RMSNorm (baked
            // below), 4 norms/layer, per-head QK-norm, head_dim independent of
            // dim/n_heads, custom attn scale query_pre_attn_scalar^-0.5, dual-theta
            // sliding-window interleave, GeGLU gelu-tanh. Crate hipfire-arch-gemma3.
            // See docs/plans/2026-06-19-gemma3-bringup.md.
            "gemma3_text" | "gemma3" => 12,
            // nemotron_h (NVIDIA Nemotron-3): Mamba-2 + GQA-attn + ReLU²-MLP hybrid
            // (Nano-4B dense; Nano-30B adds MoE). Crate hipfire-arch-nemotron
            // (arch_id 14). Quantizes the linear projections; keeps conv1d/A_log/D/
            // dt_bias/norms F16 (see should_quantize).
            "nemotron_h" => 14,
            // state-spaces Mamba-2: pure Mamba-2 mixer stack. Uses the same
            // Mamba block machinery as nemotron_h but remains its own served arch.
            "mamba2" => 15,
            // Zyphra ZAYA1 (CCA attention + EDA/MoD-routed MoE). Crate
            // hipfire-arch-zaya (arch_id 16). Native checkpoint stores experts as
            // stacked 3D `mlp.experts.{gate_up,down}_proj`, like Qwen3.5-MoE, so it
            // rides the same is_moe 3D-split path below.
            "zaya" => 16,
            other => {
                eprintln!("Warning: unknown architecture '{other}', treating as llama");
                0
            }
        }
    };
    // Gemma3 multimodal (Gemma3ForConditionalGeneration) carries a `vision_config`
    // → arch_id 13 (gemma3-vl: SigLIP tower + projector + the gemma3 text
    // decoder). Pure-text `gemma3`/`gemma3_text` stay 12. Text tensors are
    // `language_model.*`-prefixed; vision/projector stay F32 (see should_quantize).
    let auto_arch_id = if auto_arch_id == 12 && config.get("vision_config").is_some() {
        13
    } else {
        auto_arch_id
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id = parse_arch_id_override().unwrap_or(auto_arch_id);
    if arch_id != auto_arch_id {
        eprintln!("Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})");
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }
    // arch_id 6 = Qwen3.5-MoE, 16 = ZAYA1: both store routed experts as stacked 3D
    // `mlp.experts.{gate_up,down}_proj` tensors that the ingest path must split
    // per-expert (see the 3D split gated on `is_moe`).
    let is_moe = arch_id == 6 || arch_id == 16;
    // DeepSeek V4 (arch_id=9 post-2026-05-26 upstream merge that promoted
    // Qwen2-dense to 7 and dots.ocr to 8) is also MoE but ships per-expert
    // separate 2D tensors (`layers.L.ffn.experts.E.{w1,w2,w3}.weight`)
    // instead of Qwen3.5's stacked 3D `mlp.experts.gate_up_proj`. Phase 1
    // ingest handles DeepSeek V4's per-expert tensors individually through
    // the standard 2D quant path; the routing fan-out into top-k experts
    // happens at forward time, not quant time.
    let is_deepseek4 = arch_id == 9;
    // MiniMax-M2 (arch_id=10): MoE like DeepSeek V4, ships per-expert pre-split
    // 2D tensors (`...block_sparse_moe.experts.E.{w1,w2,w3}.weight`). Quantized
    // as HFQ4G256 (the only 4-bit format with a complete indexed-MoE GEMV
    // kernel family). Raw HF tensor names are written verbatim (no rename);
    // the hipfire loader looks them up.
    let is_minimax = arch_id == 10;
    // LFM2.5-MoE (arch_id 11): per-expert pre-split 2D experts (like minimax),
    // bf16 source. Conv-block + dense-MLP + router + expert_bias get dedicated
    // ingest branches; routed experts → MQ4G256, everything else → Q8.
    let is_lfm2moe = arch_id == 11;
    // Nemotron-H (arch_id 14) is dense for Nano-4B and MoE for Nano-30B. The
    // router-protection rule is harmless for dense 4B because it has no
    // `.mixer.gate.weight` tensors, and necessary for 30B because router noise
    // can flip top-k expert selection.
    let is_nemotron_h = arch_id == 14;
    // Resolve the AWQ alpha now that the arch is known. The hipfire F2 sweep
    // winner (0.55) suits transformer attention/MLP activations, but
    // nemotron_h's Mamba-2 mixer projections have a far heavier-tailed
    // activation distribution: a 2026-06-28 KLD sweep on Nano-4B oq4+ (vs a Q8
    // ref) found 0.55 → garbage (KLD 5.89), with a clean minimum at 0.1
    // (KLD 0.0105, beating plain oq4's 0.015). Neighbours: 0.05→0.0120,
    // 0.15→0.0108, 0.2→0.0175. So default nemotron_h to 0.1. An explicit
    // `--awq-alpha` always overrides.
    if awq_enabled {
        let default_alpha = if is_nemotron_h { 0.1f32 } else { 0.55f32 };
        let awq_alpha = awq_alpha_explicit.unwrap_or(default_alpha);
        if !(0.0..=1.0).contains(&awq_alpha) {
            eprintln!(
                "warning: --awq-alpha {awq_alpha} outside typical [0, 1] range; using anyway"
            );
        }
        AWQ_ALPHA
            .set(awq_alpha)
            .expect("AWQ_ALPHA set twice — should not happen");
        let note = if awq_alpha_explicit.is_none() && is_nemotron_h {
            " [nemotron_h default; transformer default is 0.55]"
        } else {
            ""
        };
        eprintln!("AWQ pre-scaling: ENABLED (alpha={awq_alpha}{note}, formula: s[j]=(RMS_act[j])^alpha, geo-mean normalized to 1)");
    }
    let is_moe_like = is_moe || is_deepseek4 || is_minimax || is_lfm2moe || is_nemotron_h;
    // Q8 router: always on for MoE-class models.
    let q8_router = is_moe_like || q8_router_flag;
    if is_moe {
        eprintln!("  MoE detected — will split 3D expert tensors per-expert before quantization.");
    }
    if is_deepseek4 {
        eprintln!("  DeepSeek V4 detected — per-expert tensors ship pre-split; quantizing each as 2D weight.");
    }
    if is_minimax {
        eprintln!("  MiniMax-M2 detected — per-expert tensors ship pre-split; quantizing each as HFQ4G256 2D weight.");
    }
    if is_lfm2moe {
        if use_source_precision {
            eprintln!("  LFM2.5 detected — source-precision {format} passthrough.");
        } else if let Some(fmt) = lfm2_oq_format {
            eprintln!(
                "  LFM2.5 detected — dense conv/attn/FFN linears → {fmt:?}, routed experts → MQ4G256, expert_bias → F32, router/embed/norm/conv-filter → Q8."
            );
        } else {
            eprintln!("  LFM2.5 detected — experts → MQ4G256, expert_bias → F32, dense projections follow explicit --format when supported, remaining tensors → Q8.");
        }
    }
    if arch_id == 15 {
        eprintln!("  Mamba-2 detected — pure Mamba mixer stack; recurrence/norm tensors stay plain precision.");
    }

    // Extract layer count for K-map edge-layer promotion.
    // Qwen3.5+ nests config under "text_config"; try both paths.
    let n_layers: usize = config
        .get("num_hidden_layers")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("num_hidden_layers"))
        })
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if n_layers == 0 {
        eprintln!(
            "  warning: num_hidden_layers not found in config.json — edge-layer promotion disabled"
        );
    }

    // Read tokenizer if present
    let tokenizer_json = input_dir.join("tokenizer.json");
    let tokenizer_str = if tokenizer_json.exists() {
        std::fs::read_to_string(&tokenizer_json).ok()
    } else {
        None
    };

    // Read tokenizer_config.json (has chat_template)
    let tokenizer_config_path = input_dir.join("tokenizer_config.json");
    let tokenizer_config: Option<serde_json::Value> = if tokenizer_config_path.exists() {
        std::fs::read_to_string(&tokenizer_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };
    let mut tokenizer_config = match chat_template_override {
        Some(template) => Some(tokenizer_config_with_chat_template(
            tokenizer_config,
            template,
        )),
        None => tokenizer_config,
    };

    // Fallback: many recent models (MiniMax-M2, newer Qwen/Gemma) ship the Jinja
    // chat template as a separate `chat_template.jinja` rather than inside
    // tokenizer_config.json. The daemon reads `tokenizer_config.chat_template`
    // from the embedded HFQ metadata (see resolve_chat_template / hfq.chat_template),
    // so fold the sidecar in when tokenizer_config has no usable template —
    // otherwise serve runs raw with no chat formatting.
    {
        let has_tpl = tokenizer_config
            .as_ref()
            .and_then(|v| v.get("chat_template"))
            .and_then(|t| t.as_str())
            .map(|t| !t.trim().is_empty())
            .unwrap_or(false);
        let jinja_path = input_dir.join("chat_template.jinja");
        if !has_tpl && jinja_path.exists() {
            if let Ok(tpl) = std::fs::read_to_string(&jinja_path) {
                let obj = tokenizer_config.get_or_insert_with(|| serde_json::json!({}));
                if let Some(map) = obj.as_object_mut() {
                    map.insert("chat_template".into(), serde_json::Value::String(tpl));
                    eprintln!("  chat_template: folded chat_template.jinja into tokenizer_config metadata");
                }
            }
        }
    }

    // Read generation_config.json. HF stores some sampler-side defaults
    // here (eos_token_id, pad_token_id, bos_token_id, do_sample, etc.)
    // separately from config.json. For most checkpoints these duplicate
    // config.json fields, but dots.ocr's config.json carries no
    // eos_token_id at all — the [151643, 151673] array lives only in
    // generation_config.json. Packing it here lets the arch-side parser
    // (e.g. `hipfire-arch-qwen2::Qwen2Config::from_hfq`) fall back to
    // generation_config when config.eos_token_id is absent. Resolves
    // R5 in docs/plans/dots-ocr-devlog.md §7.
    let generation_config_path = input_dir.join("generation_config.json");
    let generation_config: Option<serde_json::Value> = if generation_config_path.exists() {
        std::fs::read_to_string(&generation_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Build metadata for .hfq. The quantization_hash is added after tensor
    // production so it covers the final quantized payload bytes.
    let mut metadata = serde_json::json!({
        "architecture": arch_str,
        "quant_format": format,
        "config": config,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
        "tokenizer_config": tokenizer_config,
        "generation_config": generation_config,
    });

    // Gemma3 RMSNorm uses the (1 + weight) zero-centered convention. We bake the
    // +1 into every norm tensor (post-pass below, before write) so the standard
    // rmsnorm kernel — which applies plain `w` — is numerically correct at
    // runtime, with no per-layer special-casing in the gemma3 forward. Record
    // the offset for provenance and to make a re-quantize double-bake detectable.
    if arch_id == 12 {
        if let serde_json::Value::Object(ref mut m) = metadata {
            m.insert("gemma_norm_offset".to_string(), serde_json::json!(1.0_f32));
        }
    }

    // Load all safetensors files
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .map(|p| {
            eprintln!("Loading: {}", p.display());
            SafetensorsFile::open(p).unwrap()
        })
        .collect();

    // Collect all tensor names.
    //
    // DeepSeek V4 note: tensors come in `<name>.weight` (I8 = E4M3) + `<name>.scale`
    // (F8_E8M0) pairs. We index the `.scale` siblings into a side map
    // keyed by the weight tensor's full name and skip them in the main
    // iteration. When we encounter the `.weight` half we look up the
    // sibling and call `dequantize_e4m3_ue8m0_to_f32` to recover f32
    // before the existing MQ-family pipeline runs.
    let mut all_tensors: Vec<(&str, usize)> = Vec::new();
    let mut fp8_scale_for: HashMap<String, (usize, String)> = HashMap::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            // MiniMax-M2 FP8: `<w>.weight` (e4m3) + `<w>.weight_scale_inv` (F32
            // block-[128,128] scale). Strip the longer suffix FIRST.
            if let Some(stem) = name.strip_suffix(".weight_scale_inv") {
                let w_name = format!("{stem}.weight");
                fp8_scale_for.insert(w_name, (fi, name.to_string()));
                continue;
            }
            if let Some(stem) = name.strip_suffix(".scale") {
                // Sibling weight name (drop `.scale`, add `.weight`).
                let w_name = format!("{stem}.weight");
                fp8_scale_for.insert(w_name, (fi, name.to_string()));
                continue;
            }
            all_tensors.push((name, fi));
        }
    }
    all_tensors.sort_by_key(|(name, _)| name.to_string());
    eprintln!(
        "Found {} tensors ({} FP8 scale siblings indexed)",
        all_tensors.len(),
        fp8_scale_for.len()
    );

    // ── SpinQuant R1 deploy pre-pass (`--rotate <M.r1>`) ─────────────────────
    // Fold RMSNorm scales into readers and rotate residual readers/writers by
    // FᵀM so the codec FWHT leaves the int4 grid in the learned M. Dense-llama
    // only. Installs the rotated overrides globally (consumed transparently by
    // every codec branch) and holds the synthesized untied lm_head for emission
    // after the main tensor loop.
    let mut rotate_lm_head: Option<Vec<f32>> = None;
    if let Some(rotate_path) = arg_value(&args, "--rotate") {
        if arch_id != 0 && arch_id != 1 {
            eprintln!("error: --rotate is dense-llama only (arch_id 0/1); got arch_id {arch_id}");
            std::process::exit(2);
        }
        match build_rotation_overrides(rotate_path, &config, n_layers, &st_files, &fp8_scale_for) {
            Ok((overrides, lm_head)) => {
                ROTATION_OVERRIDE
                    .set(overrides)
                    .expect("ROTATION_OVERRIDE set once");
                rotate_lm_head = lm_head;
            }
            Err(e) => {
                eprintln!("error: --rotate: {e}");
                std::process::exit(2);
            }
        }
    }

    // ── K-map pre-pass ──────────────────────────────────────────────────────
    // Build per-tensor quant level map. Gated to MoE models by default
    // (maintainer directive 2026-05-08): K-map's dense PPL effect is mixed
    // (+1.5% to +2.5% at 2K, -4.8% at 8K — crossover at ~3K context). To
    // avoid silently changing dense quantization output, dense models opt
    // out by default and require `--kmap-dense` to enable. MoE models keep
    // the K-map default-on path because the routed-expert promotion is
    // the headline win and the empirical regression there is tighter
    // (+1.7% PPL at 2K, gated below the dense regression threshold).
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4]; // F16, Q8, Promote6, Base
        for (name, _fi) in &all_tensors {
            let level = kmap_resolve_mode(name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Override(_) => counts[3] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(name.to_string(), level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode {
                0 => "full",
                1 => "alternating",
                2 => "typed",
                _ => "?",
            };
            eprintln!(
                "K-map plan ({format} base, {n_layers} layers{}, mode={mode_label}):",
                if is_moe { ", MoE" } else { "" }
            );
            eprintln!("  F16:       {:>4} tensors (norms, biases)", counts[0]);
            eprintln!(
                "  Q8:        {:>4} tensors (embed, lm_head, routers)",
                counts[1]
            );
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors (remaining)", counts[3]);
        }
        map
    };

    // Phase 5: per-layer tier set — which routed-expert layers go MQ3-Lloyd
    // vs MQ2-Lloyd. Only populated for `--format mq4-routed-lloyd-mq-tiered`.
    // Computed once from imatrix .counts; kmap-promoted layers are excluded
    // (they always go MQ6).
    let mq3_tier_layers: std::collections::HashSet<usize> = if use_mq4_routed_lloyd_mq_tiered {
        if let Some(ref gguf) = imatrix_gguf {
            if let Some(layer_counts) = imatrix_layer_activation_counts(gguf, n_layers) {
                // Indexes of layers NOT promoted by K-map. We need a name
                // representative of each layer's expert tensor to query
                // kmap; use the canonical safetensors name format.
                let candidates: Vec<usize> = (0..n_layers)
                    .filter(|&l| {
                        let probe_name =
                            format!("model.language_model.layers.{}.mlp.experts.gate_up_proj", l);
                        kmap.get(&probe_name) != Some(&QuantLevel::Promote6)
                    })
                    .collect();
                let mut ranked: Vec<(usize, f64)> = candidates
                    .iter()
                    .filter(|&&l| layer_counts[l].is_finite())
                    .map(|&l| (l, layer_counts[l]))
                    .collect();
                // Sort by count DESC (hot layers first).
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let n_mq3 = ((ranked.len() as f64) * tier_ratio).round() as usize;
                let n_mq3 = n_mq3.min(ranked.len());
                let set: std::collections::HashSet<usize> =
                    ranked.iter().take(n_mq3).map(|&(l, _)| l).collect();
                eprintln!(
                    "Tiered MQ-Lloyd: {} candidate non-promoted layers; \
                     {} (top {:.0}%) → MQ3-Lloyd, {} → MQ2-Lloyd",
                    ranked.len(),
                    set.len(),
                    tier_ratio * 100.0,
                    ranked.len().saturating_sub(set.len())
                );
                if set.len() <= 16 {
                    eprintln!(
                        "  MQ3-Lloyd layers (by count): {:?}",
                        ranked
                            .iter()
                            .take(n_mq3)
                            .map(|&(l, c)| (l, c as u64))
                            .collect::<Vec<_>>()
                    );
                }
                set
            } else {
                eprintln!("warning: imatrix has no ffn_gate_exps counts — tiering disabled");
                std::collections::HashSet::new()
            }
        } else {
            std::collections::HashSet::new()
        }
    } else {
        std::collections::HashSet::new()
    };

    // Quantize
    let mut hfq_tensors = Vec::new();
    let mut total_params = 0u64;
    let mut quantized_params = 0u64;
    // Spill file for large models — keeps peak RSS bounded by flushing
    // completed tensor data to disk when accumulated memory exceeds 32 GB.
    let spill_dir = output_path.parent().unwrap_or(Path::new("."));
    let mut spill = TensorSpill::new(spill_dir).ok();
    let mut total_quant_error = 0.0f64;
    let mut max_quant_error = 0.0f32;
    let mut _n_quant_groups = 0u64;

    // arch_id 13 (gemma3-vl) is multimodal — the SigLIP vision tower is REQUIRED,
    // not optional, so auto-include it (no --include-vision needed). Other arches
    // keep the opt-in default (vision skipped unless the flag is passed).
    let include_vision = std::env::args().any(|a| a == "--include-vision") || arch_id == 13;
    let vision_quant = std::env::args()
        .position(|a| a == "--vision-quant")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_default();
    // --include-prefix <prefix>: when set, ONLY tensors whose name starts
    // with this prefix are ingested; everything else is silently skipped.
    // Used to produce side-car HFQs (e.g. `--include-prefix mtp.` builds an
    // MTP-only addon that pairs with an existing base HFQ via the loader's
    // `.mtp.hfq` sidecar discovery; some legacy loaders also accept
    // `.mtp-addon.hfq`). When unset (default), all tensors pass
    // this gate and the usual mtp/vision skip rules below apply.
    let include_prefix = std::env::args()
        .position(|a| a == "--include-prefix")
        .and_then(|i| std::env::args().nth(i + 1));
    if let Some(ref p) = include_prefix {
        eprintln!(
            "  [filter] --include-prefix {p:?} — only tensors with this prefix will be ingested"
        );
    }
    let mut skipped_params = 0u64;
    // MiniMax AWQ: shared-per-layer expert scales, cached + sidecars emitted once.
    let mut mm_awq_cache: std::collections::HashMap<usize, Option<(Vec<f32>, Vec<f32>)>> =
        std::collections::HashMap::new();
    let mut mm_awq_emitted: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // LFM2 AWQ: native HFQM imatrix vectors aggregate into the same shared
    // per-layer gate/up + down sidecars the LFM2 runtime consumes.
    let mut lfm2_awq_cache: std::collections::HashMap<usize, Option<(Vec<f32>, Vec<f32>)>> =
        std::collections::HashMap::new();
    let mut lfm2_awq_emitted: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (name, file_idx) in &all_tensors {
        // --include-prefix filter (highest priority — runs before mtp/vision skips).
        if let Some(ref p) = include_prefix {
            if !name.starts_with(p) {
                let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
                let n: usize = meta.shape.iter().product();
                skipped_params += n as u64;
                continue;
            }
        }
        // Skip MTP head; optionally include vision encoder for VL inference.
        // Qwen3.5-VL names vision tensors `model.visual.*` / `visual.*`;
        // dots.ocr names them `vision_tower.*`. Both fall through to the
        // F16 fallback path (see should_quantize: vision_tower.* is
        // skipped from quantization) when --include-vision is set.
        let is_vision = name.starts_with("model.visual.")
            || name.starts_with("visual.")
            || name.starts_with("vision_tower.");
        if is_vision && !include_vision {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }
        // MTP (Multi-Token Prediction) head: pre-Phase-5 quants skipped these
        // because no forward path consumed them. deepseek4-q8-mtp is the first format
        // that ingests the MTP layer; v3 spec-decode requires it. For other
        // formats we still skip to avoid bloating the HFQ with unused tensors.
        if name.starts_with("mtp.") && !use_deepseek4_source_precision {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }

        let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        // ── F1 native-bf16 oracle passthrough ──────────────────────────────
        // Store EVERY tensor as F32 (qt=2): no quantization, bf16/f16->f32
        // widened losslessly. This bypasses every per-format branch below so
        // the produced .hfq is a full-precision reference the qwen35 loader
        // reads via its qt=2 arm and the engine forwards through the existing
        // F32 GEMV / attention_f32 path.
        if use_f32_passthrough {
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
            quantized_params += n_elements as u64;
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [F32 oracle passthrough]",
                "F32",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                bytes.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F32,
                shape,
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut sp) = spill {
                maybe_spill(&mut hfq_tensors, sp, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }

        // ── LFM2.5 ingest (arch_id 11) ─────────────────────────────────────────
        // Routed experts (A1B only) → MQ4G256; expert_bias → F32; everything else
        // (conv in/out_proj, conv depthwise filter, attn q/k/v/out_proj + qk-norm,
        // dense w1/w2/w3, router gate, operator/ffn/embedding norms, tied embed/
        // lm_head) → Q8 (qt=3 Q8F16), except explicit OQ formats route the dense
        // projection/FFN linears through the OQ branch below. Dense lfm2
        // (350M/1.2B) has no experts, so its large linears can use OQ while
        // embed/norm/router/conv-filter stay Q8/F32. load_f32 dequantizes Q8
        // norms / conv-filter back to F32 on load.
        if is_lfm2moe && !use_source_precision {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            if name.contains(".feed_forward.experts.")
                && (name.ends_with(".w1.weight")
                    || name.ends_with(".w2.weight")
                    || name.ends_with(".w3.weight"))
                && meta.shape.len() == 2
                && meta.shape[1] % 256 == 0
            {
                let mut f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let k = meta.shape[1];
                let m = meta.shape[0];
                if awq_enabled {
                    if let Some(layer_n) = minimax_layer_index(name) {
                        let alpha = AWQ_ALPHA.get().copied().unwrap_or(0.55);
                        let entry = lfm2_awq_cache
                            .entry(layer_n)
                            .or_insert_with(|| lfm2_layer_awq_scales_from_imatrix(layer_n, alpha));
                        if let Some((s_gu, s_dn)) = entry.as_ref() {
                            let scale = if name.ends_with(".w2.weight") {
                                s_dn
                            } else {
                                s_gu
                            };
                            if scale.len() == k {
                                awq_pre_scale_weights(&mut f32_data, m, k, scale);
                            } else {
                                eprintln!(
                                    "  lfm2 AWQ L{layer_n}: scale len {} != k {} ({name}); skipped",
                                    scale.len(),
                                    k
                                );
                            }
                            if lfm2_awq_emitted.insert(layer_n) {
                                hfq_tensors.push(HfqTensor {
                                    name: format!(
                                        "model.layers.{layer_n}.feed_forward.awq_scale_gate_up.weight"
                                    ),
                                    quant_type: QuantType::F16,
                                    shape: vec![s_gu.len() as u32],
                                    group_size: 0,
                                    data: awq_scales_to_f16_bytes(s_gu),
                                    spilled_len: 0,
                                });
                                hfq_tensors.push(HfqTensor {
                                    name: format!(
                                        "model.layers.{layer_n}.feed_forward.awq_scale_down.weight"
                                    ),
                                    quant_type: QuantType::F16,
                                    shape: vec![s_dn.len() as u32],
                                    group_size: 0,
                                    data: awq_scales_to_f16_bytes(s_dn),
                                    spilled_len: 0,
                                });
                                eprintln!(
                                    "  AWQ-LFM: emitted gate_up + down scales for L{layer_n}"
                                );
                            }
                        }
                    }
                }
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                eprintln!(
                    "  {:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                    "MQ4-LFM",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::MQ4G256,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            if name.ends_with(".feed_forward.expert_bias") {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let mut bytes = Vec::with_capacity(f32_data.len() * 4);
                for v in &f32_data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                eprintln!(
                    "  {:>8}: {} {:?} (expert_bias F32)",
                    "F32-LFM", name, meta.shape
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::F32,
                    shape,
                    group_size: 1,
                    data: bytes,
                    spilled_len: 0,
                });
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            // OQ route for LFM2 dense linears. This covers both dense LFM2 and
            // the non-expert linears in LFM2-MoE. Routed experts stay MQ4/MQ6
            // until the indexed MoE OQ kernels exist; router/embed/norm/conv
            // filter remain Q8/F32 for their existing runtime paths.
            let is_lfm2_dense_linear = meta.shape.len() == 2
                && meta.shape[1] % 256 == 0
                && (name.ends_with(".conv.in_proj.weight")
                    || name.ends_with(".conv.out_proj.weight")
                    || name.ends_with(".self_attn.q_proj.weight")
                    || name.ends_with(".self_attn.k_proj.weight")
                    || name.ends_with(".self_attn.v_proj.weight")
                    || name.ends_with(".self_attn.out_proj.weight")
                    || name.ends_with(".feed_forward.w1.weight")
                    || name.ends_with(".feed_forward.w2.weight")
                    || name.ends_with(".feed_forward.w3.weight"));
            if let Some(oq_format) = lfm2_oq_format.filter(|_| is_lfm2_dense_linear) {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let (q, qt, gs, label) =
                    quantize_hfq_source_tensor(name, &f32_bytes, 2, &shape, oq_format)
                        .unwrap_or_else(|e| panic!("lfm2 oq quantize {name}: {e}"));
                eprintln!(
                    "  {label:>8}: {} {:?} ({:.1} KB -> {:.1} KB) [LFM2 dense OQ]",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: qt,
                    shape: shape.clone(),
                    group_size: gs,
                    data: q,
                    spilled_len: 0,
                });
                if let Some(scales) = OQ4_AWQ_SIDECAR.with(|c| c.borrow_mut().take()) {
                    let sidecar_name = match name.strip_suffix(".weight") {
                        Some(stem) => format!("{stem}.awq_scale.weight"),
                        None => format!("{name}.awq_scale.weight"),
                    };
                    let bytes = awq_scales_to_f16_bytes(&scales);
                    eprintln!(
                        "    AWQ:    {sidecar_name} [{}] (1D F16, {} B)",
                        scales.len(),
                        bytes.len()
                    );
                    hfq_tensors.push(HfqTensor {
                        name: sidecar_name,
                        quant_type: QuantType::F16,
                        shape: vec![scales.len() as u32],
                        group_size: 0,
                        data: bytes,
                        spilled_len: 0,
                    });
                }
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // Dense mq4 (--format mq4): route the big 2D proj/FFN weight matrices
            // (conv in/out_proj, attn q/k/v/out_proj, dense w1/w2/w3) → MQ4G256.
            // The loader's weight_gemv / weight_gemv_residual auto-FWHT-rotate
            // MQ4G256, so no forward change is needed. Keep the tied embed/lm_head
            // (model.embed_tokens.weight), the router gate, norms, and the depthwise
            // conv filter at Q8/F32 (small + precision-sensitive). Non-MQ4 LFM2
            // quant formats keep the Q8 bring-up recipe for those tensors.
            if use_mq4g256
                && meta.shape.len() == 2
                && meta.shape[1] % 256 == 0
                && !name.ends_with("embed_tokens.weight")
                && (name.ends_with("_proj.weight")
                    || name.ends_with(".w1.weight")
                    || name.ends_with(".w2.weight")
                    || name.ends_with(".w3.weight"))
            {
                let f32_data = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                eprintln!(
                    "  {:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                    "MQ4-LFM",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::MQ4G256,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }

            // All remaining LFM2 tensors → Q8 (qt=3). quantize_q8f16 handles any
            // 1D/2D/3D shape elementwise (conv.conv.weight is [hidden,1,K]).
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let q = quantize_q8f16(&f32_data);
            eprintln!("  {:>8}: {} {:?} (Q8)", "Q8-LFM", name, meta.shape);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            quantized_params += n_elements as u64;
            st_files[*file_idx].drop_tensor_pages(name);
            continue;
        }

        // DeepSeek V4's `tid2eid` hash-routing tables: source I64 in safetensors,
        // shape [vocab=129280, k=6]. The values are token-id × expert-id
        // pairs that all fit in i32 (vocab < 2^31, n_experts < 2^31), so
        // we downcast I64 → U32 (4 bytes/element) before write — antirez
        // does the same and the DeepSeek V4 loader at arch.rs reads them as U32
        // (`bytes.chunks_exact(4)`). Without these in the HFQ, the loader
        // sees an empty `tid2eid_host` and `ffn_hash_routed` falls back
        // to shared-only on the first `num_hash_layers` (3) layers —
        // measured 2× wikitext2 PPL regression on deepseek4-q8-mtp (21.85
        // vs 11.42 antirez) before this fix landed.
        //
        // QuantType=22 is "reserved-but-unused" in our enum (HFP4G16
        // ablation slot, never built); we use it for tid2eid storage to
        // stay byte-compatible with antirezQ8.hfq which also writes 22.
        // The loader is name-gated (looks for "tid2eid" substring), so
        // qt value doesn't actually steer dispatch — only matters for
        // cross-tooling identification.
        if meta.dtype == "I64" {
            if name.ends_with("tid2eid") {
                if n_elements * 8 != raw_data.len() {
                    panic!(
                        "tid2eid '{name}': expected {} bytes (8 × {}), got {}",
                        n_elements * 8,
                        n_elements,
                        raw_data.len()
                    );
                }
                let mut u32_bytes: Vec<u8> = Vec::with_capacity(n_elements * 4);
                for i in 0..n_elements {
                    let off = i * 8;
                    let v = i64::from_le_bytes(raw_data[off..off + 8].try_into().unwrap());
                    let v_u32 = v as u32; // downcast — values fit
                    u32_bytes.extend_from_slice(&v_u32.to_le_bytes());
                }
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {:>8}: {} {:?} (I64 → U32, {} elements, {:.1} KB)",
                    "TID2EID",
                    name,
                    meta.shape,
                    n_elements,
                    u32_bytes.len() as f64 / 1024.0
                );
                quantized_params += n_elements as u64;
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::TidI32,
                    shape,
                    group_size: 0,
                    data: u32_bytes,
                    spilled_len: 0,
                });
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            // Other I64 (none expected in DeepSeek V4): skip with explicit warning.
            eprintln!(
                "  [skip-I64] {} {:?} ({} elements) — unexpected I64 tensor, not ingested",
                name, meta.shape, n_elements
            );
            skipped_params += n_elements as u64;
            continue;
        }

        // ── MoE 3D-stacked expert tensor split ─────────────────────────────────
        // Qwen3.5-MoE stores routed experts as 3D tensors:
        //   model.language_model.layers.{N}.mlp.experts.gate_up_proj
        //     shape: [num_experts, 2 * moe_intermediate, hidden_size]
        //   model.language_model.layers.{N}.mlp.experts.down_proj
        //     shape: [num_experts, hidden_size, moe_intermediate]
        // Note: no `.weight` suffix on these, so should_quantize() returns false
        // and the standard path would store them as F16 — defeating the purpose.
        // We split into per-expert 2D MQ4G256 quantized tensors named
        //   model.language_model.layers.{N}.mlp.experts.{X}.{base}.weight
        // so the engine loader can fish them out by expert index.
        // ── DeepSeek V4 per-expert tensor path ─────────────────────────────────────
        // DeepSeek V4 ships per-expert 2D tensors at `layers.L.ffn.experts.E.{w1,w2,w3}.weight`.
        // (Not 3D-stacked like Qwen3.5 MoE.) Route them through the MQ-family
        // quant path directly. No imatrix yet for DeepSeek V4 — pass unit column
        // weights so the underlying Lloyd codebook fit is uniform; the
        // GPTQ sequential error-feedback assignment still applies and is
        // worth +1-2 % coherence (project_gptq_lloyd-mq2_win.md).
        if is_deepseek4
            && name.contains(".ffn.experts.")
            && name.ends_with(".weight")
            && meta.shape.len() == 2
        {
            // DeepSeek V4 routed experts are FP4 (E2M1) per upstream `inference/
            // model.py:132-137` and config `expert_dtype:"fp4"`. Safetensors
            // shape is [out, in/2] with each byte packing two nibbles; the
            // paired scale tensor is `<name>.scale` UE8M0 with block size 32
            // along logical K.
            //
            // The outer condition `name.contains(".ffn.experts.")` already
            // excludes shared_experts (which use the non-routed `.shared_
            // experts.` infix). So everything reaching here is a routed
            // expert → unconditionally FP4 unpack. Logical K dim doubles.
            let name_owned = name.to_string();
            let (f32_data, logical_shape) = if (meta.dtype == "I8" || meta.dtype == "F8_E4M3")
                && fp8_scale_for.contains_key(&name_owned)
            {
                let (sfi, sname) = &fp8_scale_for[&name_owned];
                let (smeta, sbytes) = st_files[*sfi]
                    .tensor_data(sname)
                    .unwrap_or_else(|| panic!("FP scale tensor missing: {sname}"));
                dequantize_e2m1_ue8m0_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape)
            } else {
                let vals = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                (vals, meta.shape.clone())
            };
            let k = logical_shape[1];
            if k % 256 == 0
                && (use_mq4_routed_lloyd_mq2_gptq_all
                    || use_mq4_routed_lloyd_mq_antirez_gptq
                    || use_mq4_routed_lloyd_mq2_native
                    || use_mq4_routed_lloyd_mq2_imatrix
                    || use_mq4_routed_lloyd_mq_antirez
                    || use_deepseek4_source_precision)
            {
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let unit_col_weights: Vec<f32> = vec![1.0; k];
                let q = if use_mq4_routed_lloyd_mq2_gptq_all || use_mq4_routed_lloyd_mq_antirez_gptq
                {
                    quantize_mq2g256_lloyd_gptq(&f32_data, &unit_col_weights, &signs1, &signs2)
                } else {
                    quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2)
                };
                let shape: Vec<u32> = logical_shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {:>8}: {} storage{:?} → logical{:?} ({:.1} KB → {:.1} KB)",
                    "MQ2L-DeepSeek V4",
                    name,
                    meta.shape,
                    logical_shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::MQ2G256Lloyd,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (logical_shape[0] * logical_shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // Fall through to standard path for non-MQ2 formats.
        }

        // ── MiniMax-M2 OQ dense path ───────────────────────────────────────────
        // When an OQ format is requested, route the dense attention projections
        // (q/k/v/o_proj) and lm_head through the requested OQ4 format (OQ4++ runs
        // LDLQ+AWQ from the calib `.hfq` Hessians), and PROMOTE the precision-
        // sensitive MoE router to OQ8++ rather than dropping it to Q8. Oq4G256 and
        // Oq8G256 are one Opus kernel family (iu8 grouped-WMMA), so the router
        // keeps 8-bit precision (its top-k is fragile) while staying on the same
        // runtime path — the design rule for OQ mixed precision (cf. Qwen3.5).
        // Routed experts are handled by the per-expert branch below.
        if is_minimax
            && lfm2_oq_format.is_some()
            && meta.shape.len() == 2
            && meta.shape[1] % 256 == 0
            && (name.ends_with(".self_attn.q_proj.weight")
                || name.ends_with(".self_attn.k_proj.weight")
                || name.ends_with(".self_attn.v_proj.weight")
                || name.ends_with(".self_attn.o_proj.weight")
                || name.ends_with("lm_head.weight")
                || name.ends_with(".block_sparse_moe.gate.weight"))
        {
            // Promote the router (top-k routing) to OQ8++; everything else dense
            // takes the requested OQ4 width.
            let oq_format = if name.ends_with(".block_sparse_moe.gate.weight") {
                HfqInputFormat::Oq8Plus
            } else {
                lfm2_oq_format.unwrap()
            };
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let (q, qt, gs, label) =
                quantize_hfq_source_tensor(name, &f32_bytes, 2, &shape, oq_format)
                    .unwrap_or_else(|e| panic!("minimax oq quantize {name}: {e}"));
            eprintln!(
                "  {label:>8}: {} {:?} ({:.1} KB -> {:.1} KB) [MiniMax dense OQ]",
                name,
                meta.shape,
                raw_data.len() as f64 / 1024.0,
                q.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape: shape.clone(),
                group_size: gs,
                data: q,
                spilled_len: 0,
            });
            if let Some(scales) = OQ4_AWQ_SIDECAR.with(|c| c.borrow_mut().take()) {
                let sidecar_name = match name.strip_suffix(".weight") {
                    Some(stem) => format!("{stem}.awq_scale.weight"),
                    None => format!("{name}.awq_scale.weight"),
                };
                let bytes = awq_scales_to_f16_bytes(&scales);
                eprintln!(
                    "    AWQ:    {sidecar_name} [{}] (1D F16, {} B)",
                    scales.len(),
                    bytes.len()
                );
                hfq_tensors.push(HfqTensor {
                    name: sidecar_name,
                    quant_type: QuantType::F16,
                    shape: vec![scales.len() as u32],
                    group_size: 0,
                    data: bytes,
                    spilled_len: 0,
                });
            }
            quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }

        // ── MiniMax-M2 router: keep Q8 ─────────────────────────────────────────
        // The MoE router (`block_sparse_moe.gate.weight`) is precision-sensitive
        // (4-bit noise flips top-k on borderline tokens) but must NOT be F16:
        // weight_gemv's F16 arm dispatches gemm_f16_batched_lmhead, which is a
        // WMMA lm-head kernel that produces garbage for the router's tiny m
        // (=n_exp). Q8 (gemv_q8_0) is well-behaved at any m and ~0.4% noise.
        // (Non-OQ formats only — the OQ path above promotes the router to OQ8++.)
        if is_minimax && name.ends_with("block_sparse_moe.gate.weight") {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let q = quantize_q8f16(&f32_data);
            eprintln!("  {:>8}: {} {:?} (router Q8)", "Q8-MM", name, meta.shape);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            continue;
        }

        // ── MiniMax-M2 per-expert pre-split path ───────────────────────────────
        // Experts ship as 2D `...block_sparse_moe.experts.E.{w1,w2,w3}.weight`
        // (F32 in the tiny oracle; FP8 e4m3 + F32 weight_scale_inv in the 229B
        // ckpt — handled transparently by tensor_to_f32_with_optional_fp8_scale).
        // Quantize each as MQ4G256 (FWHT-pre-rotated 4-bit): byte-compatible with
        // the gemv_hfq4g256_moe_* indexed kernels — passing FWHT-rotated input to
        // those kernels is mathematically equivalent to gemv_mq4g256 (the exact
        // path qwen35's MoE uses). This IS the user-facing "mq4" format. Names
        // are written verbatim; the loader fuses w1||w3 into the gate_up blob.
        if is_minimax
            && name.contains(".block_sparse_moe.experts.")
            && name.ends_with(".weight")
            && meta.shape.len() == 2
        {
            let mut f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let k = meta.shape[1];
            let m = meta.shape[0];
            if use_fp16 || use_bf16 {
                // qtip2-sim: simulate QTIP-2 on quantizable 2D weights (skip
                // embeddings/lm_head and any k not divisible by 256), then emit
                // bf16 from the perturbed f32 so the normal forward yields a
                // faithful QTIP-2 PPL.
                let (q, qt, label) = if use_bf16 {
                    source_precision_tensor_bytes(raw_data, &meta.dtype, &f32_data)
                } else {
                    (f32_slice_to_f16_bytes(&f32_data), QuantType::F16, "F16")
                };
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {:>8}: {} {:?} ({:.1} KB -> {:.1} KB)",
                    label,
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: qt,
                    shape,
                    group_size: 0,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            if k % 256 == 0 {
                // AWQ shared-per-layer pre-scaling of the routed experts (--awq +
                // --imatrix). w1/w3 use s_gate_up (MoE-input channels), w2 uses
                // s_down (intermediate channels). Math W·s @ x/s = W·x is exact;
                // the forward divides the activation by experts[0]'s scale.
                if awq_enabled {
                    if let (Some(layer_n), Some(gg)) =
                        (minimax_layer_index(name), imatrix_gguf.as_ref())
                    {
                        let alpha = AWQ_ALPHA.get().copied().unwrap_or(0.55);
                        let entry = mm_awq_cache
                            .entry(layer_n)
                            .or_insert_with(|| minimax_layer_awq_scales(gg, layer_n, alpha));
                        if let Some((s_gu, _s_dn)) = entry.as_ref() {
                            // gate/up-AWQ ONLY: down-AWQ (shared s_down) is harmful for MoE
                            // (per-expert down-input saliency differs), AND the loader divides
                            // only the gate_up input — so pre-scaling w2 would leave an
                            // uncancelled scale. Leave w2 unscaled; emit only the gate_up sidecar.
                            if !name.ends_with(".w2.weight") {
                                if s_gu.len() == k {
                                    awq_pre_scale_weights(&mut f32_data, m, k, s_gu);
                                } else {
                                    eprintln!("  minimax AWQ L{layer_n}: s_gu len {} != k {} ({name}); skipped", s_gu.len(), k);
                                }
                            }
                            if mm_awq_emitted.insert(layer_n) {
                                let p = name.split(".block_sparse_moe.").next().unwrap();
                                hfq_tensors.push(HfqTensor {
                                    name: format!("{p}.block_sparse_moe.awq_scale_gate_up.weight"),
                                    quant_type: QuantType::F16,
                                    shape: vec![s_gu.len() as u32],
                                    group_size: 0,
                                    data: awq_scales_to_f16_bytes(s_gu),
                                    spilled_len: 0,
                                });
                                eprintln!("  AWQ-MM: emitted gate_up scale for L{layer_n}");
                            }
                        }
                    }
                }
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                // Expert format by --format: mq2-lloyd (MQ2G256Lloyd, hipx sub-4-bit
                // target — has deepseek4 indexed-MoE kernels), mq6 (oracle check /
                // HIPFIRE_MINIMAX_EXPERT_MQ6), else mq4 (MQ4G256, validated baseline).
                let mm_mq6 =
                    use_mq6g256 || std::env::var_os("HIPFIRE_MINIMAX_EXPERT_MQ6").is_some();
                let mm_mq2l =
                    use_lloyd_mq2g256 || std::env::var_os("HIPFIRE_MINIMAX_EXPERT_MQ2L").is_some();
                let mm_mq3l =
                    use_lloyd_mq3g256 || std::env::var_os("HIPFIRE_MINIMAX_EXPERT_MQ3L").is_some();
                // Per-layer mixed-precision promotion. HIPFIRE_MINIMAX_PROMOTE_MQ4 /
                // _MQ6 hold comma-separated layer ranges ("12-45,50") whose experts are
                // forced UP to MQ4 / MQ6 regardless of the base --format. The forward
                // dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model
                // carries an MQ2-Lloyd base with MQ4 on the quant-sensitive middle layers.
                let mm_layer = minimax_layer_index(name);
                let promote_mq6 = mm_layer.map_or(false, |l| {
                    minimax_layer_in_env_set("HIPFIRE_MINIMAX_PROMOTE_MQ6", l)
                });
                let promote_mq4 = mm_layer.map_or(false, |l| {
                    minimax_layer_in_env_set("HIPFIRE_MINIMAX_PROMOTE_MQ4", l)
                });
                // Per-projection promotion: the down proj (w2) sees ~24x the
                // activation magnitude of gate/up (the SwiGLU intermediate), so its
                // 2-bit error dominates the block output. HIPFIRE_MINIMAX_DOWN_FORMAT=
                // {mq6,mq4,mq3-lloyd} promotes ONLY w2, keeping w1/w3 at the base.
                // The forward dispatches down on its own dtype, so they can differ.
                let down_fmt = if name.ends_with(".w2.weight") {
                    std::env::var("HIPFIRE_MINIMAX_DOWN_FORMAT").ok()
                } else {
                    None
                };
                // OQ format requested (oq4/oq4++): emit ALL experts as OQ4G256 so
                // the per-layer fused gate_up/down blobs stay uniform-stride for the
                // indexed gemv_oq4g256_moe_* kernels. Takes precedence over the MQ
                // env knobs (which target MQ-base models). Per-expert LDLQ does not
                // apply (no per-expert Hessian); AWQ pre-scaling above still holds.
                //
                // `--w8-top <frac>` promotes the top-frac (e.g. 0.01 = 1%) magnitude
                // weights per group to int8 via OqPlusCompact (int4 bulk + sparse
                // int8 outliers; loader expands to Oq8G256 → gemv_oq8g256_moe_*). The
                // "important expert weights → OQ8++" tier from
                // docs/plans/2026-06-23-oqplus-activation-quality.md.
                let mm_oq4 = lfm2_oq_format.is_some();
                let mm_w8_frac = OQPLUS_W8_FRAC.get().copied();
                let (q, qt, label) = if mm_oq4 {
                    if let Some(frac) = mm_w8_frac {
                        (
                            quantize_oqplus_compact(&f32_data, &signs1, &signs2, frac),
                            QuantType::OqPlusCompact,
                            "OQ+C-MM",
                        )
                    } else {
                        (
                            quantize_oq4g256(&f32_data, &signs1, &signs2),
                            QuantType::Oq4G256,
                            "OQ4-MM",
                        )
                    }
                } else if let Some(df) = down_fmt.as_deref() {
                    match df {
                        "mq6" => (
                            quantize_mq6g256(&f32_data, &signs1, &signs2),
                            QuantType::MQ6G256,
                            "MQ6-DN",
                        ),
                        "mq4" => (
                            quantize_mq4g256(&f32_data, &signs1, &signs2),
                            QuantType::MQ4G256,
                            "MQ4-DN",
                        ),
                        "lloyd-mq3" => (
                            quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ3G256Lloyd,
                            "MQ3L-DN",
                        ),
                        _ => (
                            quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ2G256Lloyd,
                            "MQ2L-DN",
                        ),
                    }
                } else if promote_mq6 {
                    (
                        quantize_mq6g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ6G256,
                        "MQ6-PROMO",
                    )
                } else if promote_mq4 {
                    (
                        quantize_mq4g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ4G256,
                        "MQ4-PROMO",
                    )
                } else if mm_mq3l {
                    (
                        quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2),
                        QuantType::MQ3G256Lloyd,
                        "MQ3L-MM",
                    )
                } else if mm_mq2l {
                    (
                        quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2),
                        QuantType::MQ2G256Lloyd,
                        "MQ2L-MM",
                    )
                } else if mm_mq6 {
                    (
                        quantize_mq6g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ6G256,
                        "MQ6-MM",
                    )
                } else {
                    (
                        quantize_mq4g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ4G256,
                        "MQ4-MM",
                    )
                };
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {label:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: qt,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // k not %256 → fall through to standard path (real MiniMax inter=1536,
            // hidden=3072 are both %256, so this only guards degenerate tinies).
        }

        if is_moe
            && name.contains("mlp.experts.")
            && (name.ends_with("gate_up_proj") || name.ends_with("down_proj"))
            && meta.shape.len() == 3
        {
            let n_experts = meta.shape[0];
            let inner_n: usize = meta.shape[1..].iter().product();
            let elem_size = match meta.dtype.as_str() {
                "F32" => 4,
                "F16" | "BF16" => 2,
                other => panic!("unsupported expert tensor dtype: {other}"),
            };
            let inner_bytes = inner_n * elem_size;
            let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&s| s as u32).collect();
            let base_name = if name.ends_with("gate_up_proj") {
                "gate_up_proj"
            } else {
                "down_proj"
            };
            // Strip the trailing base; what remains is the parent path with `experts.` already on the end
            let parent = &name[..name.len() - base_name.len()];

            // Inner quantization for experts — respects --format flag.
            // MQ6 reduces quantization error that compounds across 48 MoE
            // layers × 9 expert contributions per layer at the cost of ~50%
            // more VRAM per expert. MQ4 is the default for VRAM efficiency.
            let signs1 = gen_fwht_signs(42, 256);
            let signs2 = gen_fwht_signs(1042, 256);
            let inner_k = inner_shape[1] as usize;
            let supports_g256 = inner_k % 256 == 0;
            // K-map: check the parent tensor name directly. The parent
            // (e.g. "...mlp.experts.gate_up_proj") contains "mlp.experts."
            // so kmap_resolve rule 4 matches it. The kmap HashMap was built
            // from all_tensors which has these parent names as keys.
            let kmap_promote = kmap.get(*name) == Some(&QuantLevel::Promote6);
            // Phase 5 tiering decision needs the layer index for this parent.
            // Computed once here and reused by both expert_lloyd_mq2_native
            // and expert_lloyd_mq3_native below.
            let parent_layer: Option<usize> = {
                let marker = ".layers.";
                parent.rfind(marker).and_then(|i| {
                    let rest = &parent[i + marker.len()..];
                    rest.split('.').next().and_then(|s| s.parse().ok())
                })
            };
            let tiered_layer_is_mq3 = use_mq4_routed_lloyd_mq_tiered
                && !kmap_promote
                && parent_layer
                    .map(|l| mq3_tier_layers.contains(&l))
                    .unwrap_or(false);
            let tiered_layer_is_mq2 = use_mq4_routed_lloyd_mq_tiered
                && !kmap_promote
                && parent_layer
                    .map(|l| !mq3_tier_layers.contains(&l))
                    .unwrap_or(false);
            // Antirez-style: gate_up → MQ2, down → MQ3 (kmap-respecting).
            // Selects based on `base_name` ("gate_up_proj" vs "down_proj").
            let is_gate_up = base_name == "gate_up_proj";
            let antirez_mq3 = (use_mq4_routed_lloyd_mq_antirez
                || use_mq4_routed_lloyd_mq_antirez_gptq)
                && !kmap_promote
                && !is_gate_up;
            let antirez_mq2 = (use_mq4_routed_lloyd_mq_antirez
                || use_mq4_routed_lloyd_mq_antirez_gptq)
                && !kmap_promote
                && is_gate_up;
            // Lever 2: GPTQ-style sequential Lloyd specifically for the
            // gate_up MQ2 path. Sets a flag the inner quant dispatch will
            // honor (separate from the imatrix-only path).
            let use_gptq_for_gate_up = use_mq4_routed_lloyd_mq_antirez_gptq && antirez_mq2;
            // For the kmap-respecting MQ2-Lloyd variants, kmap_promote experts
            // get MQ6 instead of MQ2-Lloyd. Falls through to expert_mq6 below.
            let expert_mq6 = (use_mq6g256
                || use_mq4_mq6exp
                || (kmap_promote && use_mq4g256)
                || (kmap_promote && use_mq4_routed_lloyd_mq2_kmap)
                || (kmap_promote && use_mq4_routed_lloyd_mq2_imatrix)
                || (kmap_promote && use_mq4_routed_lloyd_mq2_gptq_all)
                || (kmap_promote && use_mq4_routed_lloyd_mq3_kmap))
                && supports_g256;
            let expert_hfq6 = (use_hfq6 || (kmap_promote && use_hfq4g256)) && supports_g256;
            let expert_hfq4 = use_hfq4g256 && !kmap_promote && supports_g256;
            // mq4-routed-lloyd-mq2-exp round-trip probe: ALWAYS hits routed experts
            // (overrides any kmap promotion). The intent is to inject MQ2
            // noise specifically on the routed-expert tensors, so even
            // K-map "Promote6" experts get the MQ2-Lloyd round-trip here.
            let expert_lloyd_mq2_roundtrip = use_mq4_routed_lloyd_mq2_exp && supports_g256;
            // Native MQ2-Lloyd: ship qt=19 bytes directly, no round-trip.
            // Requires runtime support for DType::MQ2G256Lloyd on experts.
            // For -native (no kmap respect): always MQ2-Lloyd on every expert.
            // For -kmap / -imatrix (kmap respect): only non-promoted experts
            // go MQ2-Lloyd; promoted ones hit `expert_mq6` above.
            // All-MQ2-GPTQ test: ALL routed experts at MQ2-Lloyd, both
            // gate_up and down. Respects kmap_promote (promoted layers
            // still get MQ6). Uses sequential-GPTQ Lloyd everywhere via
            // the `use_gptq_for_all_mq2` flag below.
            let all_mq2_gptq = use_mq4_routed_lloyd_mq2_gptq_all && !kmap_promote;
            let expert_lloyd_mq2_native = (use_mq4_routed_lloyd_mq2_native
                || (use_mq4_routed_lloyd_mq2_kmap && !kmap_promote)
                || (use_mq4_routed_lloyd_mq2_imatrix && !kmap_promote)
                || tiered_layer_is_mq2
                || antirez_mq2
                || all_mq2_gptq)
                && supports_g256;
            // GPTQ assignment fires for both gate_up and down when in
            // all-MQ2-GPTQ mode (not just gate_up like the antirez split).
            let use_gptq_for_gate_up =
                use_gptq_for_gate_up || (all_mq2_gptq && imatrix_path.is_some());
            // MQ3-Lloyd asymmetric: non-promoted experts → qt=20 (3.5 bpw).
            // Promoted ones hit `expert_mq6` above (note: kmap_promote already
            // includes use_mq4_routed_lloyd_mq3_kmap via the expert_mq6 expression).
            //
            // Phase 5 tiered variant: also MQ3-Lloyd on hot non-promoted
            // layers (the ones in `mq3_tier_layers`, decided above by imatrix
            // .counts ranking).
            let expert_lloyd_mq3_native = ((use_mq4_routed_lloyd_mq3_kmap && !kmap_promote)
                || tiered_layer_is_mq3
                || antirez_mq3)
                && supports_g256;
            // Per-expert column-weights from the imatrix file, used only by
            // the imatrix variant. Built once per parent (cheap), then sliced
            // per expert inside the rayon loop. Falls back to None when the
            // imatrix tensor for this parent isn't found (e.g. a non-expert
            // tensor we accidentally route here, or a layer that wasn't in
            // the calibration set).
            let imatrix_lookup_name = format!("{}{}", parent, base_name);
            let imatrix_per_expert: Option<Vec<Vec<f32>>> = if (use_mq4_routed_lloyd_mq2_imatrix
                || use_mq4_routed_lloyd_mq_antirez
                || use_mq4_routed_lloyd_mq_antirez_gptq
                || use_mq4_routed_lloyd_mq2_gptq_all)
                && imatrix_gguf.is_some()
                && expert_lloyd_mq2_native
            {
                imatrix_col_weights_for_parent(
                    imatrix_gguf.as_ref().unwrap(),
                    &imatrix_lookup_name,
                    n_experts,
                )
            } else {
                None
            };
            if use_mq4_routed_lloyd_mq2_imatrix
                && expert_lloyd_mq2_native
                && imatrix_per_expert.is_none()
            {
                eprintln!(
                    "  imatrix: no entry for {} → falling back to uniform Lloyd",
                    imatrix_lookup_name
                );
            }

            // Parallelize across the 256 expert slices via rayon. Each slice
            // dequant→FWHT→quant→pack is a CPU-bound, self-contained job.
            // The outer Rayon pool size is set in main() before this runs.
            use rayon::prelude::*;
            let dtype = meta.dtype.clone();
            let parent_owned = parent.to_string();
            let inner_shape_clone = inner_shape.clone();
            let base_owned = base_name.to_string();
            // `HIPFIRE_NO_EXPERT_AWQ=1` suppresses per-expert AWQ smoothing (the
            // experts fall back to plain MQ4/MQ8) — an A/B knob for measuring the
            // expert-AWQ quality delta; does not affect dense tensors.
            let no_expert_awq = std::env::var("HIPFIRE_NO_EXPERT_AWQ").ok().as_deref() == Some("1");
            let nested: Vec<Vec<HfqTensor>> = (0..n_experts)
                .into_par_iter()
                .map(|x| {
                    let slice_off = x * inner_bytes;
                    let slice = &raw_data[slice_off..slice_off + inner_bytes];
                    let mut f32_slice = to_f32(slice, &dtype);
                    let expert_name = format!("{parent_owned}{x}.{base_owned}.weight");
                    // mq4+ / oq-calibrated experts: SmoothQuant/AWQ from the per-expert
                    // imatrix via the shared name-keyed path. Gated on AWQ_ALPHA (only the
                    // `+` recipes set it) so base mq4/oq4 experts are byte-identical to
                    // before; applied only in the plain MQ4/MQ8 arms below (the Lloyd/MQ6
                    // arms calibrate their own way). Skipped on a length mismatch.
                    let expert_awq: Option<Vec<f32>> = if no_expert_awq {
                        None
                    } else {
                        awq_scales_for(&expert_name).filter(|s| s.len() == inner_k)
                    };
                    let m_expert = inner_shape_clone[0] as usize;
                    let (quantized, qt, gs) = if use_bf16 && dtype == "BF16" {
                        (slice.to_vec(), QuantType::BF16, 0u32)
                    } else if use_fp16 || use_bf16 {
                        (f32_slice_to_f16_bytes(&f32_slice), QuantType::F16, 0u32)
                    } else if expert_lloyd_mq3_native {
                        let q = quantize_mq3g256_lloyd(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ3G256Lloyd, 256u32)
                    } else if expert_lloyd_mq2_native {
                        // Native MQ2-Lloyd: ship qt=19 bytes (72 B / 256 weights).
                        // Selection order:
                        //   1. GPTQ-Lloyd (sequential error feedback) — Lever 2
                        //      path, requires imatrix.
                        //   2. Imatrix-weighted Lloyd — standard Phase 3b path.
                        //   3. Uniform Lloyd — fallback when no imatrix available.
                        let q = match imatrix_per_expert.as_ref() {
                            Some(table)
                                if x < table.len()
                                    && !table[x].is_empty()
                                    && use_gptq_for_gate_up =>
                            {
                                quantize_mq2g256_lloyd_gptq(&f32_slice, &table[x], &signs1, &signs2)
                            }
                            Some(table) if x < table.len() && !table[x].is_empty() => {
                                quantize_mq2g256_lloyd_weighted(
                                    &f32_slice, &table[x], &signs1, &signs2,
                                )
                            }
                            _ => quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                        };
                        (q, QuantType::MQ2G256Lloyd, 256u32)
                    } else if expert_lloyd_mq2_roundtrip {
                        // MQ2-Lloyd → F32 → HFQ4 round-trip. The MQ2 step injects
                        // the 2-bit Lloyd-codebook noise; the HFQ4 step re-packs
                        // for runtime. Final on-disk format is HFQ4G256, no
                        // engine changes required.
                        let mq2_bytes = quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2);
                        let dequant = dequantize_mq2g256_lloyd_to_f32(
                            &mq2_bytes,
                            f32_slice.len(),
                            &signs1,
                            &signs2,
                        );
                        let q = quantize_hfq4g256(&dequant);
                        (q, QuantType::HFQ4G256, 256u32)
                    } else if expert_mq6 {
                        let q = quantize_mq6g256(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ6G256, 256u32)
                    } else if expert_hfq6 {
                        let q = quantize_hfq6g256(&f32_slice);
                        (q, QuantType::HFQ6G256, 256u32)
                    } else if expert_hfq4 {
                        let q = quantize_hfq4g256(&f32_slice);
                        (q, QuantType::HFQ4G256, 256u32)
                    } else if use_mq8g256 && supports_g256 {
                        if let Some(s) = &expert_awq {
                            awq_pre_scale_weights(&mut f32_slice, m_expert, inner_k, s);
                        }
                        let q = quantize_mq8g256(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ8G256, 256u32)
                    } else if supports_g256 {
                        if let Some(s) = &expert_awq {
                            awq_pre_scale_weights(&mut f32_slice, m_expert, inner_k, s);
                        }
                        let q = quantize_mq4g256(&f32_slice, &signs1, &signs2);
                        (q, QuantType::MQ4G256, 256u32)
                    } else {
                        let q = quantize_hfq4g128(&f32_slice);
                        (q, QuantType::HFQ4G128, 128u32)
                    };
                    let mut produced = vec![HfqTensor {
                        name: expert_name.clone(),
                        quant_type: qt,
                        shape: inner_shape_clone.clone(),
                        group_size: gs,
                        data: quantized,
                        spilled_len: 0,
                    }];
                    // Emit the per-expert AWQ sidecar only when scales were actually
                    // applied (the plain MQ4/MQ8 arms). The loader attaches it because
                    // MQ4G256/MQ8G256 `supports_awq_sidecar()`; the gemv divides x/s.
                    if let Some(s) = expert_awq {
                        if matches!(qt, QuantType::MQ4G256 | QuantType::MQ8G256) {
                            let sidecar_name = expert_name
                                .strip_suffix(".weight")
                                .map(|st| format!("{st}.awq_scale.weight"))
                                .unwrap_or_else(|| format!("{expert_name}.awq_scale.weight"));
                            produced.push(HfqTensor {
                                name: sidecar_name,
                                quant_type: QuantType::F16,
                                shape: vec![s.len() as u32],
                                group_size: 0,
                                data: awq_scales_to_f16_bytes(&s),
                                spilled_len: 0,
                            });
                        }
                    }
                    produced
                })
                .collect::<Vec<Vec<HfqTensor>>>();
            let mut new_tensors: Vec<HfqTensor> = nested.into_iter().flatten().collect();
            quantized_params += inner_n as u64 * n_experts as u64;
            // Single eprintln to summarize the whole expert sweep.
            let label = if use_bf16 && meta.dtype == "BF16" {
                "BF16"
            } else if use_fp16 || use_bf16 {
                "F16"
            } else if expert_lloyd_mq3_native {
                "MQ3G256L"
            } else if expert_lloyd_mq2_native {
                if imatrix_per_expert.is_some() {
                    "MQ2L+imatrix"
                } else {
                    "MQ2G256L"
                }
            } else if expert_lloyd_mq2_roundtrip {
                "MQ2L→HFQ4"
            } else if expert_mq6 {
                "MQ6G256"
            } else if expert_hfq6 {
                "HFQ6G256"
            } else if expert_hfq4 {
                "HFQ4G256"
            } else if use_mq8g256 && supports_g256 {
                "MQ8G256"
            } else if supports_g256 {
                "MQ4G256"
            } else {
                "HFQ4G128"
            };
            let bytes_per = new_tensors.first().map(|t| t.data.len()).unwrap_or(0);
            eprintln!("  {label:>8}: {parent_owned}{{0..{n_experts}}}.{base_owned}.weight {:?} (×{n_experts} experts || {:.1} KB/expert, parallel)",
                inner_shape, bytes_per as f64 / 1024.0);
            hfq_tensors.append(&mut new_tensors);
            // Drop source pages and spill quantized data after each expert batch.
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024); // 2 GB threshold
            }
            continue;
        }

        // ── deepseek4-q8-mtp short-circuit ───────────────────────────────────────
        // Routed experts (.ffn.experts.*) were claimed by the MQ2-Lloyd
        // branch above. Here we handle everything else:
        //
        //   - antirez-precision-sensitive (compressor / indexer /
        //     router gate.weight): keep as F16 on disk. The compressor
        //     class alone regresses PPL +40-81% if dropped to MQ4
        //     (memory: project_deepseek4_compressor_must_stay_f16); F16 → Q8
        //     on these classes is a smaller hit but still unnecessary.
        //   - All other weights: uniform Q8F16.
        //   - Norms / biases / HC matrices: should_quantize() returns
        //     false → fall through to F16 fallback at the bottom.
        // deepseek4-mtp-precise: all mtp.0.* dense weights (anything that goes
        // through gemv_auto in mtp_forward — wq_a/b, wkv, wo_a/b, e_proj,
        // h_proj, shared experts, gate.weight) stay F16 to eliminate Q8
        // quant noise on the MTP block. Routed experts (".ffn.experts.")
        // are excluded — they MUST stay MQ2-Lloyd because the MoE GEMV
        // kernel (`deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed`) only
        // handles that format.
        let keep_f16_mtp = use_mtp_precise
            && name.starts_with("mtp.")
            && !name.contains(".ffn.experts.")
            && should_quantize(name);
        if (use_deepseek4_source_precision && is_deepseek4_keep_f16(name) || keep_f16_mtp)
            && n_elements >= 32
        {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let src_dtype = meta.dtype.as_str();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB) [src={src_dtype}, keep-F16]",
                "F16",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                f16_bytes.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_bytes,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }
        if use_deepseek4_source_precision && should_quantize(name) && n_elements >= 32 {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let src_dtype = meta.dtype.as_str();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;
            let q = quantize_q8f16(&f32_data);
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB) [src={src_dtype}]",
                "Q8_F16",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                q.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }

        if should_quantize(name) && n_elements >= 32 {
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

            // Q8HFQ path: split-metadata per-row layout (needs M and K)
            // Exclude embeddings — they use a lookup kernel, not GEMV
            if use_q8hfq && meta.shape.len() == 2 && !name.contains("embed_tokens") {
                let m = meta.shape[0];
                let k = meta.shape[1];
                let (quantized, row_stride) = quantize_q8hfq(&f32_data, m, k);

                // Compute quantization error for Q8HFQ
                let n_groups = k / 32;
                let scales_bytes = n_groups * 2;
                for row in 0..m {
                    let row_off = row * row_stride;
                    for g in 0..n_groups {
                        let scale = f16_to_f32(u16::from_le_bytes([
                            quantized[row_off + g * 2],
                            quantized[row_off + g * 2 + 1],
                        ]));
                        for i in 0..32 {
                            let qval = quantized[row_off + scales_bytes + g * 32 + i] as i8;
                            let dequant = scale * qval as f32;
                            let orig_idx = row * k + g * 32 + i;
                            let err = (dequant - f32_data[orig_idx]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                        _n_quant_groups += 1;
                    }
                }

                eprintln!(
                    "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB, stride={})",
                    "Q8_HFQ",
                    name,
                    meta.shape,
                    n_elements,
                    raw_data.len() as f64 / 1024.0,
                    quantized.len() as f64 / 1024.0,
                    row_stride
                );

                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::Q8HFQ,
                    shape,
                    group_size: 32,
                    data: quantized,
                    spilled_len: 0,
                });
            } else {
                // ── K-map override ──────────────────────────────────────────────
                let kmap_level = kmap.get(&**name).copied().unwrap_or(QuantLevel::Base);

                // AWQ sidecar scales for this tensor — populated only inside the
                // MQ4G256 arm when --awq is enabled and an imatrix entry exists
                // for this tensor's ggml-translated name. After the main tensor
                // push, we emit an `<name>.awq_scale` 1D F16 sidecar tensor so
                // the runtime can apply `x / s` before the rotation kernel at
                // inference time.
                let mut awq_sidecar_scales: Option<Vec<f32>> = None;
                let is_embed = name.contains("embed_tokens");

                let (quantized, qt, gs, label) = if use_bf16 {
                    let (data, qt, label) =
                        source_precision_tensor_bytes(raw_data, &meta.dtype, &f32_data);
                    (data, qt, 0u32, label)
                } else if use_fp16 {
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                } else if q8_conv1d_default && is_conv1d_tensor(name) {
                    // DeltaNet conv1d defaults to Q8 (see --no-q8-conv1d to disable).
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if (use_oq4 || use_oq8 || use_oq8_plus) && is_embed {
                    // Embedding lookup has its own loader path. It supports Q8
                    // directly, while OQ4/OQ8 are GEMV/GEMM weight formats.
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if kmap_level == QuantLevel::Q8 {
                    // K-map says Q8 (embed, lm_head, router)
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if kmap_level == QuantLevel::F16 {
                    // K-map says F16 (should not normally reach here — should_quantize filters first)
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                } else if kmap_level == QuantLevel::Promote6 {
                    // K-map says promote to 6-bit
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if (use_mq4g256
                        || use_mq4_mq6exp
                        || use_mq4_routed_lloyd_mq2_exp
                        || use_mq4_routed_lloyd_mq2_native
                        || use_mq4_routed_lloyd_mq2_kmap
                        || use_mq4_routed_lloyd_mq2_imatrix
                        || use_mq4_routed_lloyd_mq3_kmap
                        || use_mq4_routed_lloyd_mq_tiered
                        || use_mq4_routed_lloyd_mq_antirez
                        || use_mq4_routed_lloyd_mq_antirez_gptq
                        || use_mq4_routed_lloyd_mq2_gptq_all
                        || use_mq3g256
                        || use_mq2g256
                        || use_lloyd_mq2g256
                        || use_lloyd_mq3g256)
                        && k_dim % 256 == 0
                    {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                    } else if (use_hfq4g256
                        || use_hfq3g256
                        || use_hfq3g128
                        || use_hfq2g256
                        || use_hfq2g128)
                        && k_dim % 256 == 0
                    {
                        let q = quantize_hfq6g256(&f32_data);
                        (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                    } else if use_mq6g256 && k_dim % 256 == 0 {
                        // Already 6-bit MQ — no-op promotion
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                    } else if use_hfq6 && k_dim % 256 == 0 {
                        // Already 6-bit HFQ — no-op promotion
                        let q = quantize_hfq6g256(&f32_data);
                        (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                    } else {
                        // Non-256-aligned fallback: Q8
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                } else if let QuantLevel::Override(override_fmt) = kmap_level {
                    // K-map says override (today: lm_head when --lm-head-format set).
                    // Dispatch on the carried format. For MQ4 with AWQ enabled,
                    // apply AWQ pre-scaling + emit a sidecar so the runtime
                    // (once the CUDA-branch AWQ-aware lm_head dispatch lands)
                    // sees scaled bytes and inverse-divides correctly. For any
                    // other format, plain quantize (the AWQ wiring outside MQ4
                    // is a follow-up).
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        match override_fmt {
                            GgufFormat::F16 | GgufFormat::Bf16 => {
                                let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                                (f16_bytes, QuantType::F16, 0u32, "F16")
                            }
                            GgufFormat::Mq4 => {
                                // Inline AWQ + MQ4 dance (mirrors the Base MQ4 arm).
                                let q = if let (Some(alpha), Some(im_weights)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if awq_eligible(name) {
                                        let scales = compute_awq_scales(im_weights, alpha);
                                        awq_sidecar_scales = Some(scales.clone());
                                        let m_dim = meta.shape[0];
                                        let mut scaled = f32_data.clone();
                                        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                        quantize_mq4g256(&scaled, &signs1, &signs2)
                                    } else {
                                        quantize_mq4g256(&f32_data, &signs1, &signs2)
                                    }
                                } else {
                                    quantize_mq4g256(&f32_data, &signs1, &signs2)
                                };
                                (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                            }
                            GgufFormat::Mq6 => {
                                let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                            }
                            GgufFormat::Mq3 => {
                                // MQ3 + AWQ on lm_head: runtime supports the sidecar via
                                // DType::supports_awq_sidecar(MQ3G256)=true (per the
                                // fix/lm-head-awq-runtime branch). Wire the same AWQ
                                // inline-quantize dance as the MQ4 arm.
                                let q = if let (Some(alpha), Some(im_weights)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if awq_eligible(name) {
                                        let scales = compute_awq_scales(im_weights, alpha);
                                        awq_sidecar_scales = Some(scales.clone());
                                        let m_dim = meta.shape[0];
                                        let mut scaled = f32_data.clone();
                                        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                        quantize_mq3g256(&scaled, &signs1, &signs2)
                                    } else {
                                        quantize_mq3g256(&f32_data, &signs1, &signs2)
                                    }
                                } else {
                                    quantize_mq3g256(&f32_data, &signs1, &signs2)
                                };
                                (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                            }
                            GgufFormat::Hfq4 => {
                                let q = quantize_hfq4g256(&f32_data);
                                (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                            }
                            GgufFormat::Hfq6 => {
                                let q = quantize_hfq6g256(&f32_data);
                                (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                            }
                            // Other Override targets: not yet wired with AWQ;
                            // emit plain quantization. Used in Phase 0 sweeps
                            // for non-AWQ lm_head experiments.
                            GgufFormat::Mq2 => {
                                let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                            }
                            GgufFormat::Mq2Lloyd => {
                                let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                            }
                            GgufFormat::Mq3Lloyd => {
                                let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                            }
                            GgufFormat::Mq4Lloyd => {
                                let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                                (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                            }
                            GgufFormat::Mfp4 => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                                (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                            }
                            GgufFormat::Hfp4 => {
                                let m = if meta.shape.len() == 2 {
                                    meta.shape[0]
                                } else {
                                    1
                                };
                                let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                                (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                            }
                        }
                    } else {
                        // Non-256-aligned override target: Q8 fallback.
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                } else {
                    // QuantLevel::Base — existing format-specific logic below

                    // Choose quant format per tensor
                    let this_q8 = if use_q4k_all {
                        false // everything Q4_K
                    } else if use_q4k_q8embed {
                        name.contains("embed") || name.contains("lm_head") // only embed/output Q8
                    } else if use_mixed || use_fast {
                        is_q8_tensor(name)
                    } else {
                        use_q8 || use_q8hfq // 1D Q8HFQ tensors fall back to Q8F16
                    };
                    let this_q4as8 = use_fast && !this_q8; // FFN tensors in q8-fast mode
                    let this_q4k = use_q4k_all || use_q4k_q8embed || use_mixed;

                    // Embeddings stored as Q8 in HFQ4 mode — Q4 is too lossy for
                    // large-dim models (9B: dim=4096, values ~0.016, Q4 step ~0.007)
                    // `embed_tokens` (llama/qwen/…) and `backbone.embeddings.weight` (nemotron_h)
                    // are both embedding tables — keep them Q8 (row-lookup-able; Q4 is too lossy).
                    let is_embed =
                        name.contains("embed_tokens") || name.ends_with("embeddings.weight");
                    let use_mq4_family = use_mq4g256
                        || use_mq4_mq6exp
                        || use_mq4_routed_lloyd_mq2_exp
                        || use_mq4_routed_lloyd_mq2_native
                        || use_mq4_routed_lloyd_mq2_kmap
                        || use_mq4_routed_lloyd_mq2_imatrix
                        || use_mq4_routed_lloyd_mq3_kmap
                        || use_mq4_routed_lloyd_mq_tiered
                        || use_mq4_routed_lloyd_mq_antirez
                        || use_mq4_routed_lloyd_mq_antirez_gptq
                        || use_mq4_routed_lloyd_mq2_gptq_all;

                    if use_hfq_mixed {
                        // hfq-mixed: Q8 for attention, HFQ4 for FFN (fits 9B in 8GB VRAM)
                        let is_ffn = name.contains("mlp.") || name.contains("ffn");
                        if !is_ffn {
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        } else {
                            let k_dim = if meta.shape.len() == 2 {
                                meta.shape[1]
                            } else {
                                n_elements
                            };
                            if k_dim % 256 == 0 {
                                let q = quantize_hfq4g256(&f32_data);
                                (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                            } else {
                                let q = quantize_hfq4g128(&f32_data);
                                (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                            }
                        }
                    } else if use_hfq6 {
                        // HFQ6-G256: all weights 6-bit, embeddings Q8
                        if is_embed {
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        } else {
                            let q = quantize_hfq6g256(&f32_data);
                            (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                        }
                    } else if (use_hfq2g256 || use_hfq2g128) && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_hfq2g128 {
                        let q = quantize_hfq2g128(&f32_data);
                        (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                    } else if use_hfq2g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq2g256(&f32_data);
                            (q, QuantType::HFQ2G256, 256u32, "HFQ2G256")
                        } else {
                            // Fallback to HFQ4 for non-256-aligned
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if q8_router && is_q8_tensor(name) {
                        // Q8 router for MoE: keep mlp.gate.weight and
                        // shared_expert_gate.weight at Q8 regardless of --format.
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq8g256 && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq8g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let q = quantize_mq8g256(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ8G256, 256u32, "MQ8G256")
                        } else {
                            // Fallback to Q8 for non-256-aligned
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        }
                    } else if use_mq4_family && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq4_family && is_nemotron_h_mq4_q8_protected(name) {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq4_family {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // Phase A Stage A — AWQ pre-scaling, when --awq is enabled
                            // AND we have imatrix data for this tensor AND the tensor
                            // is on the AWQ whitelist (see `awq_eligible`). Mutates a
                            // local copy of the weights so the original f32_data
                            // returned by to_f32() is left intact for downstream
                            // consumers (we don't currently have any here, but this
                            // is hygienic).
                            //
                            // The `awq_eligible(name)` guard is critical: pre-scaling
                            // weights whose runtime path lacks the inverse divide
                            // produces `(W·s)·x ≠ W·x` and catastrophically corrupts
                            // logits (KLD 0.67 → 13.5 measured on 0.8B Qwen3.5 before
                            // this guard landed). See `docs/plans/awq_fix_claude.md`.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    debug_assert_eq!(
                                        im_weights.len(),
                                        k_dim,
                                        "imatrix length ({}) != K dim ({}) for {}",
                                        im_weights.len(),
                                        k_dim,
                                        name
                                    );
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    // Stash for sidecar emission after the main tensor push.
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    // Copy weights so we don't mutate to_f32's buffer
                                    // (might be shared/borrowed depending on dtype path).
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq4g256(&scaled, &signs1, &signs2)
                                } else {
                                    // Runtime path for this weight has no AWQ inverse
                                    // (rotate_x_mq for o_proj/out_proj/wo, or
                                    // fused_silu_mul_rotate_mq for down_proj/w_down).
                                    // Skip AWQ for this tensor — emit plain MQ4 and
                                    // no sidecar.
                                    quantize_mq4g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq4g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                        } else if k_dim % 128 == 0 {
                            // Non-256 but 128-aligned → HFQ4-G128.
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        } else {
                            // k divides neither 256 nor 128 (nemotron_h hidden=3136
                            // = 64·49) → HFQ4-G128 would emit garbage (the kernel
                            // assumes k%128==0). Fall back to Q8 (group 32).
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        }
                    } else if use_hfp4 && is_embed {
                        // HFP4 embeddings stay Q8F16 (matches MQ4 / HFQ4 pattern — embedding lookup is
                        // accuracy-sensitive, FP4 codes too lossy for vocab-sized tables).
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_hfp4 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 32 == 0 && meta.shape.len() == 2 {
                            let m = meta.shape[0];
                            let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                            (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                        } else {
                            // Fallback to HFQ4-G128 for non-32-aligned ragged dims (rare).
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mfp4 && is_embed {
                        // MFP4 embeddings stay Q8F16 (same rationale as HFP4 / MQ4).
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mfp4 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 && meta.shape.len() == 2 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                        } else {
                            // Fallback to HFQ4-G128 for non-256-aligned ragged dims (rotation
                            // requires 256-element segments). Matches MQ4's ragged fallback.
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_mq6g256 && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_mq6g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                        } else {
                            // Fallback to HFQ6-G256 for non-256-aligned (no rotation)
                            let q = quantize_hfq6g256(&f32_data);
                            (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                        }
                    } else if (use_mq3g256
                        || use_mq2g256
                        || use_lloyd_mq2g256
                        || use_lloyd_mq3g256
                        || use_lloyd_mq4g256)
                        && is_embed
                    {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_lloyd_mq4g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                        } else {
                            // Fallback to HFQ4-G128 for non-256-aligned (no rotation).
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_lloyd_mq3g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ × MQ3-Lloyd composition (MQ3G256Lloyd is forward-path-ready +
                            // now in supports_awq_sidecar). Pre-scale by imatrix, then Lloyd-fit.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq3g256_lloyd(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                        } else {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_lloyd_mq2g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ × MQ2-Lloyd (MQ2G256Lloyd is in supports_awq_sidecar): pre-scale
                            // by imatrix first, then Lloyd-fit (K=4, or K=3-ternary under the flag).
                            let awq_scaled: Option<Vec<f32>> =
                                if let (Some(alpha), Some(im_weights)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if awq_eligible(name) {
                                        let scales = compute_awq_scales(im_weights, alpha);
                                        awq_sidecar_scales = Some(scales.clone());
                                        let m_dim = meta.shape[0];
                                        let mut scaled = f32_data.clone();
                                        awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                        Some(scaled)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };
                            let data: &[f32] = awq_scaled.as_deref().unwrap_or(&f32_data);
                            // HIPFIRE_LLOYD_K3=1 → ternary "MQ1.58" (3-level codebook, reuses kernel).
                            let q =
                                if std::env::var("HIPFIRE_LLOYD_K3").ok().as_deref() == Some("1") {
                                    quantize_mq2g256_lloyd_k3(data, &signs1, &signs2)
                                } else {
                                    quantize_mq2g256_lloyd(data, &signs1, &signs2)
                                };
                            (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                        } else {
                            // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                            let q = quantize_hfq2g128(&f32_data);
                            (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                        }
                    } else if use_mq3g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ pre-scaling for MQ3 base body (mirrors the MQ4 base arm).
                            // MQ3G256 is on DType::supports_awq_sidecar, so the runtime applies
                            // the inverse divide via rotate_x_mq. Without this, `--format mq3
                            // --awq` was a silent no-op on body tensors (md5(mq3-awq)==md5(mq3)).
                            // awq_eligible gates to tensors whose runtime path has the inverse.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq3g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq3g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq3g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                        } else {
                            // Fallback to HFQ3-G128 for non-256-aligned (no rotation)
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_mq2g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            // AWQ × plain MQ2 (MQ2G256 now in supports_awq_sidecar). Pre-scale by
                            // imatrix, then quantize. (Plain MQ2 collapses uncalibrated; AWQ is the
                            // test of whether activation-aware scaling rescues uniform 2-bit.)
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq2g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq2g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq2g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                        } else {
                            // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                            let q = quantize_hfq2g128(&f32_data);
                            (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                        }
                    } else if (use_hfq3g256 || use_hfq3g128) && is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else if use_hfq3g128 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 128 == 0 {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        } else {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_hfq3g256 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq3g256(&f32_data);
                            (q, QuantType::HFQ3G256, 256u32, "HFQ3G256")
                        } else {
                            let q = quantize_hfq3g128(&f32_data);
                            (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                        }
                    } else if use_hfq4g256 && is_embed {
                        // HFQ4 embeddings: half the size of Q8, same 18-VGPR lookup kernel
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq4g256(&f32_data);
                            (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                        } else {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_hfq4g256 {
                        // Auto-select G128 vs G256 based on K dimension
                        // G256 preferred: better coalescing, fewer scale/zero overheads
                        // G128 only as fallback when K isn't divisible by 256
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq4g256(&f32_data);
                            (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                        } else if k_dim % 128 == 0 {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        } else {
                            // Pad to 128-element boundary
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    } else if use_oq4 {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if meta.shape.len() == 2 && k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m_dim = meta.shape[0];
                            let ldlq_q = OQ4_LDLQ_HESSIAN.get().and_then(|idx| {
                                let mut h = ldlq_hessian_for_tensor(idx, name, k_dim)?;
                                let awq_scales = if let (Some(alpha), Some(im)) =
                                    (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                {
                                    if alpha > 0.0 && awq_eligible(name) {
                                        Some(compute_awq_scales(im, alpha))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };
                                let wbuf: std::borrow::Cow<[f32]> = if let Some(s) = &awq_scales {
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, s);
                                    for i in 0..k_dim {
                                        let si = s[i] as f64;
                                        for j in 0..k_dim {
                                            h[i * k_dim + j] =
                                                (h[i * k_dim + j] as f64 / (si * s[j] as f64))
                                                    as f32;
                                        }
                                    }
                                    std::borrow::Cow::Owned(scaled)
                                } else {
                                    std::borrow::Cow::Borrowed(&f32_data[..])
                                };
                                let diag_sum: f64 =
                                    (0..k_dim).map(|i| h[i * k_dim + i] as f64).sum();
                                let damp = 0.01 * (diag_sum / k_dim as f64).max(1e-12);
                                let out = ldlq::oq4_ldlq_pack(
                                    &wbuf, m_dim, k_dim, &h, &signs1, &signs2, damp,
                                );
                                if out.is_some() {
                                    ldlq_record_success();
                                    if let Some(s) = awq_scales {
                                        awq_sidecar_scales = Some(s);
                                        eprintln!(
                                            "  ldlq+awq: {name} [{m_dim}x{k_dim}] OBS int4 + smooth"
                                        );
                                    } else {
                                        eprintln!(
                                            "  ldlq: {name} [{m_dim}x{k_dim}] OBS error-feedback int4"
                                        );
                                    }
                                } else {
                                    ldlq_record_pack_failed(name);
                                }
                                out
                            });
                            let q = if let Some(q) = ldlq_q {
                                q
                            } else if let (Some(alpha), Some(im)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if alpha > 0.0 && awq_eligible(name) {
                                    let scales = compute_awq_scales(im, alpha);
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    awq_sidecar_scales = Some(scales);
                                    quantize_oq4g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_oq4g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_oq4g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::Oq4G256, 256u32, "OQ4G256")
                        } else {
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        }
                    } else if use_oq8 || use_oq8_plus {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if meta.shape.len() == 2 && k_dim % 256 == 0 {
                            let signs1 = gen_fwht_signs(42, 256);
                            let signs2 = gen_fwht_signs(1042, 256);
                            let m_dim = meta.shape[0];
                            let ldlq_q = if use_oq8_plus {
                                OQ4_LDLQ_HESSIAN.get().and_then(|idx| {
                                    let mut h = ldlq_hessian_for_tensor(idx, name, k_dim)?;
                                    let awq_scales = if let (Some(alpha), Some(im)) =
                                        (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                                    {
                                        if alpha > 0.0 && awq_eligible(name) {
                                            Some(compute_awq_scales(im, alpha))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };
                                    let wbuf: std::borrow::Cow<[f32]> =
                                        if let Some(s) = &awq_scales {
                                            let mut scaled = f32_data.clone();
                                            awq_pre_scale_weights(&mut scaled, m_dim, k_dim, s);
                                            for i in 0..k_dim {
                                                let si = s[i] as f64;
                                                for j in 0..k_dim {
                                                    h[i * k_dim + j] =
                                                        (h[i * k_dim + j] as f64
                                                            / (si * s[j] as f64))
                                                            as f32;
                                                }
                                            }
                                            std::borrow::Cow::Owned(scaled)
                                        } else {
                                            std::borrow::Cow::Borrowed(&f32_data[..])
                                        };
                                    let diag_sum: f64 =
                                        (0..k_dim).map(|i| h[i * k_dim + i] as f64).sum();
                                    let damp = 0.01 * (diag_sum / k_dim as f64).max(1e-12);
                                    let out = ldlq::oq8_ldlq_pack(
                                        &wbuf, m_dim, k_dim, &h, &signs1, &signs2, damp,
                                    );
                                    if out.is_some() {
                                        ldlq_record_success();
                                        if let Some(s) = awq_scales {
                                            awq_sidecar_scales = Some(s);
                                            eprintln!(
                                                "  ldlq+awq: {name} [{m_dim}x{k_dim}] OBS int8 + smooth"
                                            );
                                        } else {
                                            eprintln!(
                                                "  ldlq: {name} [{m_dim}x{k_dim}] OBS error-feedback int8"
                                            );
                                        }
                                    } else {
                                        ldlq_record_pack_failed(name);
                                    }
                                    out
                                })
                            } else {
                                None
                            };
                            let q = if let Some(q) = ldlq_q {
                                q
                            } else if let (Some(alpha), Some(im)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if alpha > 0.0 && awq_eligible(name) {
                                    let scales = compute_awq_scales(im, alpha);
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    awq_sidecar_scales = Some(scales);
                                    quantize_oq8g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_oq8g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_oq8g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::Oq8G256, 256u32, "OQ8G256")
                        } else {
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        }
                    } else if this_q8 {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_FP16")
                    } else if this_q4as8 {
                        let q = quantize_q4_as_q8(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q4asQ8")
                    } else if this_q4k {
                        let q = quantize_q4k(&f32_data);
                        (q, QuantType::Q4K, 256u32, "Q4_K")
                    } else {
                        let q = quantize_q4f16_g64(&f32_data);
                        (q, QuantType::Q4F16G64, 64u32, "Q4_F16")
                    }
                }; // end K-map outer if-else

                // Compute quantization error (skip for Q8 embeddings — always negligible)
                let block_size = gs as usize;
                let is_hfq4 = label == "HFQ4G256" || label == "HFQ4G128";
                // Only compute detailed error for HFQ4 tensors — Q8/HFQ6 error is negligible
                let skip_error = !is_hfq4;
                let n_blocks = if !skip_error {
                    (n_elements + block_size - 1) / block_size
                } else {
                    0
                };
                for b in 0..n_blocks {
                    let start = b * block_size;
                    let end = (start + block_size).min(n_elements);
                    if is_hfq4 {
                        // Both G128 (72B) and G256 (136B): [f32 scale][f32 zero][nibbles]
                        let block_bytes = if block_size == 256 { 136 } else { 72 };
                        let off = b * block_bytes;
                        let scale = f32::from_le_bytes([
                            quantized[off],
                            quantized[off + 1],
                            quantized[off + 2],
                            quantized[off + 3],
                        ]);
                        let zero = f32::from_le_bytes([
                            quantized[off + 4],
                            quantized[off + 5],
                            quantized[off + 6],
                            quantized[off + 7],
                        ]);
                        for i in 0..(end - start) {
                            let byte_idx = i / 2;
                            let nibble = if i % 2 == 0 {
                                quantized[off + 8 + byte_idx] & 0xF
                            } else {
                                quantized[off + 8 + byte_idx] >> 4
                            };
                            let dequant = scale * nibble as f32 + zero;
                            let err = (dequant - f32_data[start + i]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                    } else if label == "Q8_FP16" || label == "Q4asQ8" || label == "Q8_F16" {
                        // NB: string match because this_q8/this_q4as8 are scoped inside Base block.
                        let off = b * 34;
                        let scale =
                            f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                        for i in 0..(end - start) {
                            let qval = quantized[off + 2 + i] as i8;
                            let dequant = scale * qval as f32;
                            let err = (dequant - f32_data[start + i]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                    } else {
                        let off = b * 36;
                        let scale =
                            f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                        let min_val = f16_to_f32(u16::from_le_bytes([
                            quantized[off + 2],
                            quantized[off + 3],
                        ]));
                        for i in 0..(end - start) {
                            let byte_idx = if i < 32 { i } else { i - 32 };
                            let nibble = if i < 32 {
                                quantized[off + 4 + byte_idx] & 0xF
                            } else {
                                quantized[off + 4 + byte_idx] >> 4
                            };
                            let dequant = nibble as f32 * scale + min_val;
                            let err = (dequant - f32_data[start + i]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                    }
                    _n_quant_groups += 1;
                }

                eprintln!(
                    "  {label:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB)",
                    name,
                    meta.shape,
                    n_elements,
                    raw_data.len() as f64 / 1024.0,
                    quantized.len() as f64 / 1024.0
                );

                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: qt,
                    shape: shape.clone(),
                    group_size: gs,
                    data: quantized,
                    spilled_len: 0,
                });
                // Phase A Stage A — emit AWQ scale sidecar tensor immediately
                // after the parent weight. Naming convention:
                // `<weight_name>.awq_scale` (strip the trailing `.weight` and
                // append `.awq_scale.weight` so the runtime loader recognizes
                // it as a 1D F16 tensor of length K). 1D shape [K]; runtime
                // pairs it with the parent weight at model open.
                if let Some(scales) = awq_sidecar_scales.take() {
                    let sidecar_name = match name.strip_suffix(".weight") {
                        Some(stem) => format!("{stem}.awq_scale.weight"),
                        None => format!("{name}.awq_scale.weight"),
                    };
                    let bytes = awq_scales_to_f16_bytes(&scales);
                    eprintln!(
                        "    AWQ:    {} [{}] (1D F16, {} B)",
                        sidecar_name,
                        scales.len(),
                        bytes.len()
                    );
                    hfq_tensors.push(HfqTensor {
                        name: sidecar_name,
                        quant_type: QuantType::F16,
                        shape: vec![scales.len() as u32],
                        group_size: 0,
                        data: bytes,
                        spilled_len: 0,
                    });
                }
            } // end else (non-Q8HFQ path)
        } else if is_vision && vision_quant == "hfq4" && n_elements >= 32 {
            // Quantize vision weights to HFQ4G256 (for speed-critical VL workloads)
            let f32_data = to_f32(raw_data, &meta.dtype);
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let k_dim = if shape.len() == 2 {
                shape[1] as usize
            } else {
                n_elements
            };
            let (quantized, gs) = if k_dim % 256 == 0 {
                (quantize_hfq4g256(&f32_data), 256u32)
            } else {
                (quantize_hfq4g128(&f32_data), 128u32)
            };
            let qt = if gs == 256 {
                QuantType::HFQ4G256
            } else {
                QuantType::HFQ4G128
            };
            let label = if gs == 256 { "HFQ4G256" } else { "HFQ4G128" };
            eprintln!(
                "  {label:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [vision]",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                quantized.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            });
        } else if (use_bf16 || (is_vision && vision_quant == "bf16")) && meta.dtype == "BF16" {
            // Store original BF16 bytes losslessly in source-precision containers.
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let scope = if is_vision && vision_quant == "bf16" {
                "vision/source"
            } else {
                "source"
            };
            eprintln!(
                "  BF16:       {} {:?} ({} elements, {:.1} KB) [{scope}, lossless]",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::BF16,
                shape,
                group_size: 0,
                data: raw_data.to_vec(),
                spilled_len: 0,
            });
        } else if use_bf16 || (is_vision && vision_quant == "bf16") {
            // Non-BF16 source (F16/F32) — store as F16
            let data = if meta.dtype == "F16" {
                raw_data.to_vec()
            } else {
                let f32_vals = to_f32(raw_data, &meta.dtype);
                f32_vals
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect()
            };
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let scope = if is_vision && vision_quant == "bf16" {
                "vision/source fallback"
            } else {
                "source fallback"
            };
            eprintln!(
                "  F16:        {} {:?} ({:.1} KB) [{scope}]",
                name,
                meta.shape,
                data.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data,
                spilled_len: 0,
            });
        } else if let Some(rot) = ROTATION_OVERRIDE.get().and_then(|m| m.get(*name)) {
            // --rotate: a non-quantized tensor with a fold+rotate override — most
            // importantly an RMSNorm folded to ones. The raw source bytes are the
            // PRE-fold gamma, which the folded readers already absorbed; storing raw
            // here (as the sibling arm does) would re-apply gamma at runtime (gamma²
            // → garbage). Encode the OVERRIDE at the source container width instead.
            let (data, qt, label): (Vec<u8>, QuantType, &str) = match meta.dtype.as_str() {
                "F32" => (
                    rot.iter().flat_map(|&v| v.to_le_bytes()).collect(),
                    QuantType::F32,
                    "F32",
                ),
                "F16" => (f32_slice_to_f16_bytes(rot), QuantType::F16, "F16"),
                _ => (f32_slice_to_bf16_bytes(rot), QuantType::BF16, "BF16"),
            };
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!(
                "  {label:<10} {} {:?} ({} elements) [--rotate override]",
                name, meta.shape, n_elements
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: 0,
                data,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            continue;
        } else {
            // Preserve source precision for non-quantized tensors (norms,
            // biases, DeltaNet scalars, etc.). BF16 and F16 have the same byte
            // width, so keeping BF16 in normal MQ/HFQ artifacts remains
            // portable: older arches can downgrade to F16 at load time.
            let f32_data = if meta.dtype == "F32" {
                to_f32(raw_data, "F32")
            } else {
                Vec::new()
            };
            let (data, qt, label) = source_precision_tensor_bytes(raw_data, &meta.dtype, &f32_data);
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!(
                "  {label:<10} {} {:?} ({} elements, {:.1} KB)",
                name,
                meta.shape,
                n_elements,
                data.len() as f64 / 1024.0
            );

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: 0,
                data,
                spilled_len: 0,
            });
        }
        // Release source file page cache after each tensor to prevent
        // mmap'd pages from starving GPU allocations on UMA systems.
        st_files[*file_idx].drop_tensor_pages(name);
    }

    // Emit the synthesized untied lm_head (`--rotate`). Stored F16 [vocab, h]: the
    // output head is not on the int4 GEMM hot path, and F16 avoids replicating a
    // per-format weight-codec branch. The runtime prefers an explicit lm_head.weight
    // over the tied embedding (see hfq.rs loader), so this deploys the rotated head.
    if let Some(lm) = rotate_lm_head.take() {
        let h = config
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .expect("hidden_size (validated in --rotate pre-pass)") as usize;
        let vocab = lm.len() / h;
        let bytes = f32_slice_to_f16_bytes(&lm);
        eprintln!(
            "  {:>8}: lm_head.weight [{vocab}, {h}] (untied R1 head, F16, {:.1} KB) [--rotate]",
            "F16-R1",
            bytes.len() as f64 / 1024.0
        );
        total_params += lm.len() as u64;
        quantized_params += lm.len() as u64;
        hfq_tensors.push(HfqTensor {
            name: "lm_head.weight".to_string(),
            quant_type: QuantType::F16,
            shape: vec![vocab as u32, h as u32],
            group_size: 0,
            data: bytes,
            spilled_len: 0,
        });
    }

    // Summary
    let total_bytes: usize = hfq_tensors
        .iter()
        .map(|t| {
            if t.spilled_len > 0 {
                t.spilled_len as usize
            } else {
                t.data.len()
            }
        })
        .sum();
    let mean_quant_error = if quantized_params > 0 {
        total_quant_error / quantized_params as f64
    } else {
        0.0
    };

    eprintln!("\n=== Quantization Summary ===");
    if skipped_params > 0 {
        eprintln!(
            "  Skipped params:   {skipped_params} (mtp/visual — use --include-vision for VL)"
        );
    }
    eprintln!("  Total params:     {total_params}");
    eprintln!(
        "  Quantized params: {quantized_params} ({:.1}%)",
        100.0 * quantized_params as f64 / total_params as f64
    );
    eprintln!("  Mean quant error: {mean_quant_error:.8}");
    eprintln!("  Max quant error:  {max_quant_error:.8}");
    eprintln!("  Output size:      {:.1} MB", total_bytes as f64 / 1e6);
    if let Err(e) = ldlq_report_and_validate(oq_ldlq_recipe) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    // qtip2-sim post-pass: simulate QTIP-2 on every eligible 2D BF16 weight,
    // branch-independent (operates on the finalized tensor list, so it catches
    // dense/MoE/attn weights regardless of which producer branch built them).
    // Skips embeddings/lm_head and any k not divisible by 256.
    if use_qtip_sim {
        let (mut n_ldlq, mut n_plain) = (0usize, 0usize);
        // BBT-spectral influence scaling (SpectralLLM): HIPFIRE_QTIP_BBT_ALPHA=α
        // enables per-channel scaling by spectral activation energy (from the
        // calib Hessian) before the trellis, math-invariant. α=0.5 is the paper
        // default; unset = no BBT.
        let bbt_alpha: Option<f32> = std::env::var("HIPFIRE_QTIP_BBT_ALPHA")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&a: &f32| a > 0.0);
        // HIPFIRE_LOWRANK_R=r adds a rank-r correction of the quant error back
        // into the emitted weight (LQER/CALDERA low-rank residual probe). This
        // sims the quality of W4 + a 2-WMMA UᵥVᵥ correction; rank=0 = off.
        let lowrank_r: usize = std::env::var("HIPFIRE_LOWRANK_R")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let (mut n_bbt, mut n_lowrank) = (0usize, 0usize);
        for t in hfq_tensors.iter_mut() {
            if !(matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && (t.shape[1] as usize) % 256 == 0
                && !t.name.contains("embed")
                && !t.name.contains("lm_head"))
            {
                continue;
            }
            let m = t.shape[0] as usize;
            let k = t.shape[1] as usize;
            let mut wf: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            // The true (pre-quant) weight, kept for the low-rank error residual.
            let wf_orig: Option<Vec<f32>> = (lowrank_r > 0).then(|| wf.clone());

            // Prefer LDLQ (output-aware) when this tensor has a Hessian; else
            // plain QTIP. The block-trellis OBS encode is now bit-parametric,
            // so 3-bit gets the same Hessian-aware feedback as 2-bit.
            let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);

            // BBT-spectral: derive per-channel scales from the Hessian's spectral
            // diagonal, scale weight columns (W' = W·diag(s)); the trellis sees the
            // scaled weight, and we divide the dequantized result back by s so the
            // emitted bf16 reproduces W·x with an s-shaped quant-error budget.
            let bbt_s: Option<Vec<f32>> = bbt_alpha.and_then(|a| {
                qtip_hessian.as_ref().and_then(|sc| {
                    let href = sc.get(key, 0)?;
                    if href.k != k {
                        return None;
                    }
                    let mut c = vec![0.0f32; k * k];
                    for i in 0..k {
                        for j in 0..k {
                            c[i * k + j] = href.at(i, j) as f32;
                        }
                    }
                    let spec = bbt_spectral_diag(&c, k);
                    Some(bbt_scales(&spec, k, a))
                })
            });
            if let Some(s) = &bbt_s {
                for i in 0..m {
                    for l in 0..k {
                        wf[i * k + l] *= s[l];
                    }
                }
                n_bbt += 1;
            }

            let ldlq_out = qtip_hessian.as_ref().and_then(|sc| {
                let href = sc.get(key, 0)?;
                if href.k != k {
                    return None;
                }
                // Materialize k×k Hessian (f32) + 1% diagonal-mean ridge. Under
                // BBT the input is x'=x/s, so Cov(x')_{ij}=C_{ij}/(s_i s_j).
                let mut h = vec![0.0f32; k * k];
                let mut diag_sum = 0.0f64;
                for i in 0..k {
                    for j in 0..k {
                        let mut v = href.at(i, j) as f32;
                        if let Some(s) = &bbt_s {
                            v /= s[i] * s[j];
                        }
                        h[i * k + j] = v;
                    }
                    diag_sum += h[i * k + i] as f64;
                }
                let damp = 0.01 * (diag_sum / k as f64).max(1e-12);
                ldlq::qtip_ldlq_dequant_bits(
                    &wf, m, k, &h, &qtip_s1, &qtip_s2, 128, damp, qtip_bits, &qtip_cb,
                )
            });
            let mut deq = match ldlq_out {
                Some(d) => {
                    n_ldlq += 1;
                    d
                }
                None => {
                    qtip_simquant_nbit(&mut wf, k, &qtip_cb, &qtip_s1, &qtip_s2, qtip_bits);
                    n_plain += 1;
                    wf
                }
            };
            // Undo the BBT column scaling: emit ≈ dequant(Q(W·s))·diag(1/s) ≈ W.
            if let Some(s) = &bbt_s {
                for i in 0..m {
                    for l in 0..k {
                        deq[i * k + l] /= s[l];
                    }
                }
            }
            // Low-rank residual: E = W − dequant(W4); add E_r (rank-r) back. Sims
            // emit(W4) + UᵥVᵥ — the 2-WMMA weight correction.
            if let Some(orig) = &wf_orig {
                let e: Vec<f32> = orig.iter().zip(&deq).map(|(a, b)| a - b).collect();
                let er = randomized_lowrank(&e, m, k, lowrank_r);
                for (d, c) in deq.iter_mut().zip(&er) {
                    *d += *c;
                }
                n_lowrank += 1;
            }
            t.data = f32_slice_to_bf16_bytes(&deq);
        }
        eprintln!(
            "  qtip{qtip_bits}-sim: LDLQ on {n_ldlq}, plain {n_plain}, BBT-scaled {n_bbt}, lowrank-r{lowrank_r} on {n_lowrank}"
        );
    }

    // roughquant-sim post-pass (Phase 1, no rotation): for every eligible 2D
    // BF16 weight, protect the top `protect_frac` highest-diag(H) input columns
    // at full precision and crush the rest to a `bulk_bits` uniform grid, baked
    // back into bf16. Saliency = diag(H) when the tensor has a Hessian, else the
    // column L2 norm of W. Mirrors the qtip-sim post-pass above.
    if use_roughquant_sim {
        let protect_frac: f64 = std::env::var("HIPFIRE_RQ_PROTECT_FRAC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.015);
        let bulk_bits: u32 = std::env::var("HIPFIRE_RQ_BULK_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let group: usize = std::env::var("HIPFIRE_RQ_GROUP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        eprintln!(
            "  roughquant-sim: protect_frac={protect_frac} bulk_bits={bulk_bits} group={group}"
        );
        let (mut n_hess, mut n_proxy) = (0usize, 0usize);
        for t in hfq_tensors.iter_mut() {
            if !(matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && (t.shape[1] as usize) % group == 0
                && !t.name.contains("embed")
                && !t.name.contains("lm_head"))
            {
                continue;
            }
            let m = t.shape[0] as usize;
            let k = t.shape[1] as usize;
            let mut wf: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            // Saliency per input column c: diag(H)=E[x_c²] if Hessian present,
            // else column L2 norm of W.
            let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
            let saliency: Vec<f32> = match qtip_hessian
                .as_ref()
                .and_then(|sc| sc.get(key, 0))
                .filter(|h| h.k == k)
            {
                Some(href) => {
                    n_hess += 1;
                    (0..k).map(|i| href.at(i, i) as f32).collect()
                }
                None => {
                    n_proxy += 1;
                    let mut s = vec![0.0f32; k];
                    for r in 0..m {
                        let row = &wf[r * k..r * k + k];
                        for c in 0..k {
                            s[c] += row[c] * row[c];
                        }
                    }
                    for v in s.iter_mut() {
                        *v = v.sqrt();
                    }
                    s
                }
            };
            roughquant_sim_tensor(&mut wf, m, k, &saliency, protect_frac, bulk_bits, group);
            t.data = f32_slice_to_bf16_bytes(&wf);
        }
        eprintln!("  roughquant-sim: diag(H)-saliency on {n_hess} tensors, L2-proxy on {n_proxy}");
    }

    // roughquant2-sim post-pass (Phase 2): PCA-rotate each eligible 2D BF16
    // weight into its activation-Hessian eigenbasis, protect the top
    // `protect_frac` highest-energy columns at full precision, QTIP-trellis the
    // bulk, inverse-rotate back, bake into bf16. Tensors lacking a Hessian (or
    // whose eigensolve fails) are left as the staged bf16.
    if use_roughquant2_sim {
        let protect_frac: f64 = std::env::var("HIPFIRE_RQ2_PROTECT_FRAC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.015);
        let bulk_bits: u32 = std::env::var("HIPFIRE_RQ2_BULK_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let damp: f64 = std::env::var("HIPFIRE_RQ2_DAMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.01);
        eprintln!(
            "  roughquant2-sim: PCA rotation, protect_frac={protect_frac} bulk_bits={bulk_bits} damp={damp}"
        );
        // De-risk B: single shared, foldable residual-stream rotation. With
        // HIPFIRE_RQ2_SHARE_RESID=1, every k==1024 weight (the d_model residual
        // readers: in_proj_*, gate/up, q/k/v) uses ONE global rotation aggregated
        // from their summed Hessians — the foldable ResQ-U_A design. Weights with
        // k!=1024 (o_proj/out_proj=2048, down_proj=3584) read internal activations
        // and keep their own per-weight rotation (the runtime-rotation tier).
        // Tests whether forcing the foldable shared rotation preserves the win.
        let share_resid = std::env::var("HIPFIRE_RQ2_SHARE_RESID").ok().as_deref() == Some("1");
        let r_global: Option<Vec<f32>> = if share_resid {
            let kk = 1024usize;
            let mut csum = vec![0.0f32; kk * kk];
            let mut n_agg = 0usize;
            if let Some(sc) = qtip_hessian.as_ref() {
                for t in hfq_tensors.iter() {
                    if matches!(t.quant_type, QuantType::BF16)
                        && t.shape.len() == 2
                        && t.shape[1] as usize == kk
                        && !t.name.contains("embed")
                        && !t.name.contains("lm_head")
                    {
                        let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
                        if let Some(href) = sc.get(key, 0).filter(|h| h.k == kk) {
                            for i in 0..kk {
                                for j in 0..kk {
                                    csum[i * kk + j] += href.at(i, j) as f32;
                                }
                            }
                            n_agg += 1;
                        }
                    }
                }
            }
            let rg = roughquant::pca_basis(&csum, kk, damp).map(|(p, _)| p);
            eprintln!(
                "  roughquant2-sim: SHARE_RESID — global k=1024 rotation from {n_agg} tensors ({})",
                if rg.is_some() { "ok" } else { "FAILED" }
            );
            rg
        } else {
            None
        };
        let (mut n_rot, mut n_skip) = (0usize, 0usize);
        for t in hfq_tensors.iter_mut() {
            if !(matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && (t.shape[1] as usize) % 256 == 0
                && !t.name.contains("embed")
                && !t.name.contains("lm_head"))
            {
                continue;
            }
            let m = t.shape[0] as usize;
            let k = t.shape[1] as usize;
            let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
            // Rotation basis: the shared global residual rotation for k==1024 when
            // SHARE_RESID is on, else the per-weight PCA basis from this tensor's
            // own Hessian (skip the tensor if it has none / the eigensolve fails).
            let p: Vec<f32> = if share_resid && k == 1024 {
                match r_global.as_ref() {
                    Some(rg) => rg.clone(),
                    None => {
                        n_skip += 1;
                        continue;
                    }
                }
            } else {
                let cmat: Option<Vec<f32>> = qtip_hessian
                    .as_ref()
                    .and_then(|sc| sc.get(key, 0))
                    .filter(|h| h.k == k)
                    .map(|href| {
                        let mut c = vec![0.0f32; k * k];
                        for i in 0..k {
                            for j in 0..k {
                                c[i * k + j] = href.at(i, j) as f32;
                            }
                        }
                        c
                    });
                let Some(c) = cmat else {
                    n_skip += 1;
                    continue;
                };
                match roughquant::pca_basis(&c, k, damp) {
                    Some((p, _ev)) => p,
                    None => {
                        n_skip += 1;
                        continue;
                    }
                }
            };
            let mut wf: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            // Rotate into PCA frame, protect top cols (rounded to 256), QTIP the
            // bulk, rotate back.
            let mut wt = roughquant::rotate_w(&wf, &p, m, k, false);
            let n_prot = ((protect_frac * k as f64).round() as usize).min(k);
            qtip_simquant_protected(&mut wt, k, n_prot, &qtip_cb, &qtip_s1, &qtip_s2, bulk_bits);
            wf = roughquant::rotate_w(&wt, &p, m, k, true);
            t.data = f32_slice_to_bf16_bytes(&wf);
            n_rot += 1;
        }
        eprintln!("  roughquant2-sim: PCA-rotated+quantized {n_rot} tensors, left-bf16 {n_skip}");
        // De-risk A: optional iso-bit embed/lm_head. The win-vs-mq4 comparison is
        // confounded because this sim leaves embed/lm_head at bf16 while mq4 uses
        // Q8 (~20% of params on a tied-embedding 0.8B). With HIPFIRE_RQ2_Q8_EMBED=1,
        // simulate Q8 (8-bit per-256-group uniform, no protection) on those tensors
        // so the comparison is honest.
        if std::env::var("HIPFIRE_RQ2_Q8_EMBED").ok().as_deref() == Some("1") {
            let mut n_e = 0usize;
            for t in hfq_tensors.iter_mut() {
                if matches!(t.quant_type, QuantType::BF16)
                    && t.shape.len() == 2
                    && (t.shape[1] as usize) % 256 == 0
                    && (t.name.contains("embed") || t.name.contains("lm_head"))
                {
                    let m = t.shape[0] as usize;
                    let k = t.shape[1] as usize;
                    let mut wf: Vec<f32> = t
                        .data
                        .chunks_exact(2)
                        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    // 8-bit per-group uniform, protect nothing → Q8G256-equivalent.
                    roughquant_sim_tensor(&mut wf, m, k, &vec![0.0; k], 0.0, 8, 256);
                    t.data = f32_slice_to_bf16_bytes(&wf);
                    n_e += 1;
                }
            }
            eprintln!(
                "  roughquant2-sim: HIPFIRE_RQ2_Q8_EMBED — Q8-sim on {n_e} embed/lm_head tensors"
            );
        }
    }

    // roughquant3-sim post-pass (Phase 2c): permutation instead of dense rotation.
    // Reorder each weight's input columns by diag(H) saliency so the salient
    // channels are contiguous leading columns, protect them, QTIP the bulk,
    // un-permute back. A permutation folds for free (reindex), so this is the
    // foldable analog of Phase 2 — minus the channel-mixing decorrelation.
    if use_roughquant3_sim {
        let protect_frac: f64 = std::env::var("HIPFIRE_RQ3_PROTECT_FRAC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.03);
        let bulk_bits: u32 = std::env::var("HIPFIRE_RQ3_BULK_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        eprintln!(
            "  roughquant3-sim: permutation+protection, protect_frac={protect_frac} bulk_bits={bulk_bits}"
        );
        let (mut n_hess, mut n_proxy) = (0usize, 0usize);
        for t in hfq_tensors.iter_mut() {
            if !(matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && (t.shape[1] as usize) % 256 == 0
                && !t.name.contains("embed")
                && !t.name.contains("lm_head"))
            {
                continue;
            }
            let m = t.shape[0] as usize;
            let k = t.shape[1] as usize;
            let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
            let mut wf: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            // Saliency per input column: diag(H) if available, else column L2.
            let saliency: Vec<f32> = match qtip_hessian
                .as_ref()
                .and_then(|sc| sc.get(key, 0))
                .filter(|h| h.k == k)
            {
                Some(href) => {
                    n_hess += 1;
                    (0..k).map(|i| href.at(i, i) as f32).collect()
                }
                None => {
                    n_proxy += 1;
                    let mut s = vec![0.0f32; k];
                    for r in 0..m {
                        let row = &wf[r * k..r * k + k];
                        for c in 0..k {
                            s[c] += row[c] * row[c];
                        }
                    }
                    s
                }
            };
            // Permutation = saliency descending → salient channels lead.
            let mut perm: Vec<usize> = (0..k).collect();
            perm.sort_unstable_by(|&a, &b| {
                saliency[b]
                    .partial_cmp(&saliency[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut wp = permute_cols(&wf, m, k, &perm);
            let n_prot = ((protect_frac * k as f64).round() as usize).min(k);
            qtip_simquant_protected(&mut wp, k, n_prot, &qtip_cb, &qtip_s1, &qtip_s2, bulk_bits);
            wf = unpermute_cols(&wp, m, k, &perm);
            t.data = f32_slice_to_bf16_bytes(&wf);
        }
        eprintln!("  roughquant3-sim: diag(H)-saliency on {n_hess} tensors, L2-proxy on {n_proxy}");
        // Iso-bit embed for an honest mq4 comparison (same as roughquant2 de-risk A).
        if std::env::var("HIPFIRE_RQ3_Q8_EMBED").ok().as_deref() == Some("1") {
            let mut n_e = 0usize;
            for t in hfq_tensors.iter_mut() {
                if matches!(t.quant_type, QuantType::BF16)
                    && t.shape.len() == 2
                    && (t.shape[1] as usize) % 256 == 0
                    && (t.name.contains("embed") || t.name.contains("lm_head"))
                {
                    let m = t.shape[0] as usize;
                    let k = t.shape[1] as usize;
                    let mut wf: Vec<f32> = t
                        .data
                        .chunks_exact(2)
                        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    roughquant_sim_tensor(&mut wf, m, k, &vec![0.0; k], 0.0, 8, 256);
                    t.data = f32_slice_to_bf16_bytes(&wf);
                    n_e += 1;
                }
            }
            eprintln!(
                "  roughquant3-sim: HIPFIRE_RQ3_Q8_EMBED — Q8-sim on {n_e} embed/lm_head tensors"
            );
        }
    }

    // roughquant4-sim post-pass (Phase 2d): channel-consistent residual-stream
    // mixed precision. Rank residual channels by aggregated energy once; keep the
    // top set exact in true residual-reader COLUMNS and residual-writer ROWS.
    // Non-residual inputs use per-weight diag(H) column protection.
    if use_roughquant4_sim {
        let protect_frac: f64 = std::env::var("HIPFIRE_RQ4_PROTECT_FRAC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.03);
        let bulk_bits: u32 = std::env::var("HIPFIRE_RQ4_BULK_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        // Bulk codec: "mq4" → real mq4 format (fair mq4+protect-vs-mq4 test, set
        // protect_frac=0 for the plain-mq4 baseline); else QTIP-{bulk_bits}.
        let bulk_kind = std::env::var("HIPFIRE_RQ4_BULK").ok().unwrap_or_default();
        let bulk_mq4 = bulk_kind == "mq4";
        let bulk_void = bulk_kind == "void";
        // OBS saliency Hessian ridge (fraction of diag mean), for SALIENCY=obs.
        let damp: f64 = std::env::var("HIPFIRE_RQ4_OBS_DAMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.01);
        // Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0
        // + this gives a fair FWHT uniform-N-bit anchor on the same machinery.
        let mq_bits: u32 = std::env::var("HIPFIRE_RQ4_MQ_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        // Protect at 8-bit (per-channel Q8) instead of bf16 — honest bit-cost.
        let protect_q8 = std::env::var("HIPFIRE_RQ4_PROTECT_Q8").ok().as_deref() == Some("1");
        // Saliency metric for which channels to protect (user steer: don't rely
        // on diag(H) alone). diag = E[x²] (activation energy); wnorm = ‖W[:,c]‖²
        // (weight energy); product = ‖W[:,c]‖²·E[x²] (output-error contribution).
        let saliency_metric = std::env::var("HIPFIRE_RQ4_SALIENCY")
            .ok()
            .unwrap_or_else(|| "diag".into());
        let dmodel = roughquant4_infer_dmodel(&hfq_tensors).unwrap_or_else(|| {
            eprintln!(
                "  roughquant4-sim: WARNING could not infer residual d_model; \
                 falling back to 1024"
            );
            1024
        });
        // Aggregate per-residual-channel saliency from true residual readers.
        let mut resid_energy = vec![0.0f64; dmodel];
        for t in hfq_tensors.iter() {
            if matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && t.shape[1] as usize == dmodel
                && !t.name.contains("embed")
                && !t.name.contains("lm_head")
                && roughquant4_is_residual_reader(&t.name)
            {
                let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
                if saliency_metric == "obs" {
                    // OBS: needs the FULL Hessian + ‖W[:,c]‖²; compensation-aware.
                    let hfull: Option<Vec<f32>> = qtip_hessian
                        .as_ref()
                        .and_then(|sc| sc.get(key, 0))
                        .filter(|h| h.k == dmodel)
                        .map(|h| {
                            let mut c = vec![0.0f32; dmodel * dmodel];
                            for i in 0..dmodel {
                                for j in 0..dmodel {
                                    c[i * dmodel + j] = h.at(i, j) as f32;
                                }
                            }
                            c
                        });
                    if let Some(hf) = hfull {
                        let cn2 = bf16_colnorm2(&t.data, dmodel);
                        if let Some(sal) = obs_col_saliency(&hf, &cn2, dmodel, damp) {
                            for i in 0..dmodel {
                                resid_energy[i] += sal[i];
                            }
                        }
                    }
                    continue;
                }
                let diag: Option<Vec<f64>> = qtip_hessian
                    .as_ref()
                    .and_then(|sc| sc.get(key, 0))
                    .filter(|h| h.k == dmodel)
                    .map(|h| (0..dmodel).map(|i| h.at(i, i)).collect());
                let cn2: Option<Vec<f32>> = if saliency_metric != "diag" {
                    Some(bf16_colnorm2(&t.data, dmodel))
                } else {
                    None
                };
                for i in 0..dmodel {
                    let d = diag.as_ref().map(|d| d[i]).unwrap_or(0.0);
                    let w = cn2.as_ref().map(|c| c[i] as f64).unwrap_or(0.0);
                    resid_energy[i] += match saliency_metric.as_str() {
                        "wnorm" => w,
                        "product" => w * d,
                        _ => d,
                    };
                }
            }
        }
        // CONTROL (HIPFIRE_RQ4_SALIENCY=random): replace importance with a seeded
        // random ranking. If random ties our metric, it means OUR selector
        // (energy/product) is no better than chance at finding the important
        // channels — not that importance is worthless. Reproducible via the seed.
        if saliency_metric == "random" {
            let seed: u64 = std::env::var("HIPFIRE_RQ4_RANDOM_SEED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1234567);
            for (i, e) in resid_energy.iter_mut().enumerate() {
                let mut z = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                *e = (z as f64) / (u64::MAX as f64);
            }
        }
        // HIPFIRE_RQ4_DUMP_RANK=1: print the residual-channel saliency ranking
        // (descending) as `RANK<tab>channel<tab>energy` and exit. Cheap (no quant);
        // used to pick ablation-oracle targets sampled across the diag spectrum.
        if std::env::var("HIPFIRE_RQ4_DUMP_RANK").ok().as_deref() == Some("1") {
            let mut idx: Vec<usize> = (0..dmodel).collect();
            idx.sort_unstable_by(|&a, &b| {
                resid_energy[b]
                    .partial_cmp(&resid_energy[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (rank, &c) in idx.iter().enumerate() {
                println!("RANK\t{rank}\t{c}\t{:.10e}", resid_energy[c]);
            }
            std::process::exit(0);
        }
        // HIPFIRE_RQ4_INVERT=1: protect the LOWEST-saliency channels instead of the
        // highest. Used by the TAIL-RANKING control: with bulk=void + protect_frac
        // high (e.g. 0.95), the non-protected (voided) set is the BOTTOM (1-frac)
        // by the chosen metric; INVERT flips it so the voided set is the TOP. If
        // void-bottom (our metric) hurts LESS than void-random and void-top hurts
        // MOST, the metric ranks the tail correctly (not just the outliers).
        let invert_select = std::env::var("HIPFIRE_RQ4_INVERT").ok().as_deref() == Some("1");
        // Top residual channels (shared across readers' cols and writers' rows).
        let n_prot_resid = ((protect_frac * dmodel as f64).round() as usize).min(dmodel);
        let protected_resid: Vec<usize> = {
            let mut idx: Vec<usize> = (0..dmodel).collect();
            idx.sort_unstable_by(|&a, &b| {
                let o = resid_energy[b]
                    .partial_cmp(&resid_energy[a])
                    .unwrap_or(std::cmp::Ordering::Equal);
                if invert_select {
                    o.reverse()
                } else {
                    o
                }
            });
            idx.truncate(n_prot_resid);
            idx
        };
        // ABLATION ORACLE (HIPFIRE_RQ4_VOID_ONLY=c1,c2,...): void EXACTLY the listed
        // residual channels and protect ALL others exact bf16. With bulk=void this
        // isolates the marginal KLD damage of ablating those specific channels —
        // the gold-standard per-channel importance signal to validate diag against.
        // Overrides protect_frac/saliency selection for the residual set.
        let (protected_resid, n_prot_resid) =
            if let Ok(spec) = std::env::var("HIPFIRE_RQ4_VOID_ONLY") {
                let void_set: std::collections::HashSet<usize> = spec
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .filter(|&c| c < dmodel)
                    .collect();
                let keep: Vec<usize> = (0..dmodel).filter(|c| !void_set.contains(c)).collect();
                eprintln!(
                    "  roughquant4-sim: ABLATION ORACLE — voiding {} residual channels {:?}, \
                 protecting the other {}",
                    void_set.len(),
                    {
                        let mut v: Vec<usize> = void_set.iter().copied().collect();
                        v.sort_unstable();
                        v
                    },
                    keep.len()
                );
                let n = keep.len();
                (keep, n)
            } else {
                (protected_resid, n_prot_resid)
            };
        eprintln!(
            "  roughquant4-sim: channel-consistent, protect_frac={protect_frac} bulk={}; \
             {n_prot_resid}/{dmodel} residual channels protected (read cols + write rows)",
            if bulk_void {
                "void(prune)".to_string()
            } else if bulk_mq4 {
                "mq4".to_string()
            } else {
                format!("qtip{bulk_bits}")
            }
        );
        let (mut n_w, mut n_r) = (0usize, 0usize);
        for t in hfq_tensors.iter_mut() {
            if !(matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && (t.shape[1] as usize) % 256 == 0
                && !t.name.contains("embed")
                && !t.name.contains("lm_head"))
            {
                continue;
            }
            let m = t.shape[0] as usize;
            let k = t.shape[1] as usize;
            let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
            let mut wf: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            // Read side: true residual readers share the global residual channel
            // set; internal-output projections protect their own input channels.
            let protected_cols: Vec<usize> = if roughquant4_is_residual_reader(&t.name)
                && k == dmodel
            {
                protected_resid.clone()
            } else {
                // Per-weight saliency for non-residual inputs, by the chosen metric.
                let diag: Option<Vec<f64>> = qtip_hessian
                    .as_ref()
                    .and_then(|sc| sc.get(key, 0))
                    .filter(|h| h.k == k)
                    .map(|h| (0..k).map(|i| h.at(i, i)).collect());
                if diag.is_none() && saliency_metric == "diag" {
                    Vec::new()
                } else {
                    let cn2: Option<Vec<f32>> =
                        if saliency_metric != "diag" && saliency_metric != "random" {
                            let mut s = vec![0.0f32; k];
                            for r in 0..m {
                                let row = &wf[r * k..r * k + k];
                                for c in 0..k {
                                    s[c] += row[c] * row[c];
                                }
                            }
                            Some(s)
                        } else {
                            None
                        };
                    let rng_seed: u64 = std::env::var("HIPFIRE_RQ4_RANDOM_SEED")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1234567);
                    let sal: Vec<f64> = (0..k)
                        .map(|c| {
                            let d = diag.as_ref().map(|d| d[c]).unwrap_or(1.0);
                            let w = cn2.as_ref().map(|x| x[c] as f64).unwrap_or(1.0);
                            match saliency_metric.as_str() {
                                "wnorm" => w,
                                "product" => w * d,
                                "random" => {
                                    // seeded per (tensor-key-hash, column) so it's
                                    // reproducible but independent per tensor.
                                    let kh = key.bytes().fold(rng_seed, |a, b| {
                                        (a ^ b as u64).wrapping_mul(0x100000001B3)
                                    });
                                    let mut z = kh
                                        .wrapping_add((c as u64).wrapping_mul(0x9E3779B97F4A7C15));
                                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                                    z ^= z >> 31;
                                    (z as f64) / (u64::MAX as f64)
                                }
                                _ => d,
                            }
                        })
                        .collect();
                    let mut idx: Vec<usize> = (0..k).collect();
                    idx.sort_unstable_by(|&a, &b| {
                        let o = sal[b]
                            .partial_cmp(&sal[a])
                            .unwrap_or(std::cmp::Ordering::Equal);
                        if invert_select {
                            o.reverse()
                        } else {
                            o
                        }
                    });
                    idx.truncate(((protect_frac * k as f64).round() as usize).min(k));
                    idx
                }
            };
            // Write side: residual writers (m==dmodel) keep high-energy output
            // rows exact.
            let mut protected_rows = vec![false; m];
            if roughquant4_is_residual_writer(&t.name) && m == dmodel {
                for &c in &protected_resid {
                    protected_rows[c] = true;
                }
                n_w += 1;
            } else {
                n_r += 1;
            }
            if bulk_void {
                // VOID: zero the non-protected (low-energy) entries entirely —
                // structured prune. An entry (r,c) is kept iff its row OR column
                // is protected (high-energy residual write row / read col);
                // everything else is set to 0. Tests true redundancy: how much
                // of the low-energy tail is genuinely unneeded.
                let mut col_keep = vec![false; k];
                for &c in &protected_cols {
                    col_keep[c] = true;
                }
                for r in 0..m {
                    if protected_rows[r] {
                        continue;
                    }
                    let row = &mut wf[r * k..r * k + k];
                    for c in 0..k {
                        if !col_keep[c] {
                            row[c] = 0.0;
                        }
                    }
                }
            } else if bulk_mq4 {
                mq4_simquant_masked(
                    &mut wf,
                    m,
                    k,
                    &protected_cols,
                    &protected_rows,
                    &qtip_s1,
                    &qtip_s2,
                    mq_bits,
                );
            } else {
                qtip_simquant_masked(
                    &mut wf,
                    m,
                    k,
                    &protected_cols,
                    &protected_rows,
                    &qtip_cb,
                    &qtip_s1,
                    &qtip_s2,
                    bulk_bits,
                );
            }
            // HIPFIRE_RQ4_PROTECT_Q8: protect at 8-bit precision (per-column /
            // per-row symmetric Q8) instead of bf16. bf16 already only has ~8-bit
            // mantissa, so this is ~same KLD at HALF the protected bit-cost — the
            // honest bit-accounting for a deployable protected format.
            if protect_q8 {
                for &c in &protected_cols {
                    let mut amax = 0.0f32;
                    for r in 0..m {
                        amax = amax.max(wf[r * k + c].abs());
                    }
                    if amax > 0.0 {
                        let s = amax / 127.0;
                        let inv = 1.0 / s;
                        for r in 0..m {
                            let q = (wf[r * k + c] * inv).round().clamp(-127.0, 127.0);
                            wf[r * k + c] = q * s;
                        }
                    }
                }
                for r in 0..m {
                    if !protected_rows[r] {
                        continue;
                    }
                    let row = &mut wf[r * k..r * k + k];
                    let amax = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                    if amax > 0.0 {
                        let s = amax / 127.0;
                        let inv = 1.0 / s;
                        for v in row.iter_mut() {
                            let q = (*v * inv).round().clamp(-127.0, 127.0);
                            *v = q * s;
                        }
                    }
                }
            }
            t.data = f32_slice_to_bf16_bytes(&wf);
        }
        eprintln!("  roughquant4-sim: {n_w} residual writers (row-protected), {n_r} other tensors");
        if std::env::var("HIPFIRE_RQ4_Q8_EMBED").ok().as_deref() == Some("1") {
            let mut n_e = 0usize;
            for t in hfq_tensors.iter_mut() {
                if matches!(t.quant_type, QuantType::BF16)
                    && t.shape.len() == 2
                    && (t.shape[1] as usize) % 256 == 0
                    && (t.name.contains("embed") || t.name.contains("lm_head"))
                {
                    let m = t.shape[0] as usize;
                    let k = t.shape[1] as usize;
                    let mut wf: Vec<f32> = t
                        .data
                        .chunks_exact(2)
                        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    roughquant_sim_tensor(&mut wf, m, k, &vec![0.0; k], 0.0, 8, 256);
                    t.data = f32_slice_to_bf16_bytes(&wf);
                    n_e += 1;
                }
            }
            eprintln!(
                "  roughquant4-sim: HIPFIRE_RQ4_Q8_EMBED — Q8-sim on {n_e} embed/lm_head tensors"
            );
        }
    }

    // roughquant (REAL) post-pass: bulk = real MQ4G256 packed bytes (existing
    // kernel) + an exact bf16 CORRECTION SIDECAR over the diag(H)-selected shared
    // residual channel set. For residual READERS the protected COLUMNS get a
    // [m × |S|] correction; for residual WRITERS the protected ROWS get a [|S| × k]
    // correction. The correction stores R = W − dequant(mq4(W)) over the protected
    // entries, so y = mq4_gemv(W) + R_S·x_S yields the EXACT bf16 contribution for
    // those channels (dequant_mq4g256 is bit-identical to the kernel). Channel
    // indices live in metadata["roughquant_sidecar"]; values in `<name>.rqcorr`
    // BF16 tensors. Absent sidecar ⇒ plain mq4 (backward-compatible loader).
    if use_roughquant_real {
        let protect_frac: f64 = std::env::var("HIPFIRE_RQ4_PROTECT_FRAC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.03);
        let dmodel = roughquant4_infer_dmodel(&hfq_tensors).unwrap_or(1024);
        // Aggregate per-residual-channel diag(H) energy from true residual readers.
        let mut resid_energy = vec![0.0f64; dmodel];
        for t in hfq_tensors.iter() {
            if matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && t.shape[1] as usize == dmodel
                && !t.name.contains("embed")
                && !t.name.contains("lm_head")
                && roughquant4_is_residual_reader(&t.name)
            {
                let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
                if let Some(diag) = qtip_hessian
                    .as_ref()
                    .and_then(|sc| sc.get(key, 0))
                    .filter(|h| h.k == dmodel)
                {
                    for i in 0..dmodel {
                        resid_energy[i] += diag.at(i, i);
                    }
                }
            }
        }
        let n_prot = ((protect_frac * dmodel as f64).round() as usize).min(dmodel);
        let protected_resid: Vec<usize> = {
            let mut idx: Vec<usize> = (0..dmodel).collect();
            idx.sort_unstable_by(|&a, &b| {
                resid_energy[b]
                    .partial_cmp(&resid_energy[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(n_prot);
            idx.sort_unstable();
            idx
        };
        eprintln!(
            "  roughquant (real): d_model={dmodel}, protect_frac={protect_frac} \
             → {n_prot} shared residual channels (reader cols + writer rows)"
        );
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let mut sidecars: Vec<HfqTensor> = Vec::new();
        let mut sidecar_meta = serde_json::Map::new();
        let (mut n_r, mut n_w, mut max_resid) = (0usize, 0usize, 0.0f32);
        let mut rq_recon_err = 0.0f32;
        for t in hfq_tensors.iter_mut() {
            if !(matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && (t.shape[1] as usize) % 256 == 0
                && !t.name.contains("embed")
                && !t.name.contains("lm_head"))
            {
                continue;
            }
            let m = t.shape[0] as usize;
            let k = t.shape[1] as usize;
            let is_reader = roughquant4_is_residual_reader(&t.name) && k == dmodel;
            let is_writer = roughquant4_is_residual_writer(&t.name) && m == dmodel;
            // f32 weights, real mq4 pack, kernel-faithful dequant.
            let wf: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let packed = quantize_mq4g256(&wf, &signs1, &signs2);
            // Emit the bulk regardless; only attach a sidecar for residual roles.
            if is_reader || is_writer {
                let recon = dequant_mq4g256(&packed, m * k, &signs1, &signs2);
                let (corr_f32, shape, role): (Vec<f32>, Vec<u32>, &str) = if is_reader {
                    // protected COLUMNS: [m × |S|], corr[r,j] = (W − recon)[r, S[j]]
                    let s = &protected_resid;
                    let mut corr = vec![0.0f32; m * s.len()];
                    for r in 0..m {
                        for (j, &c) in s.iter().enumerate() {
                            corr[r * s.len() + j] = wf[r * k + c] - recon[r * k + c];
                        }
                    }
                    (corr, vec![m as u32, s.len() as u32], "reader")
                } else {
                    // protected ROWS: [|S| × k], corr[j,c] = (W − recon)[S[j], c]
                    let s = &protected_resid;
                    let mut corr = vec![0.0f32; s.len() * k];
                    for (j, &row) in s.iter().enumerate() {
                        for c in 0..k {
                            corr[j * k + c] = wf[row * k + c] - recon[row * k + c];
                        }
                    }
                    (corr, vec![s.len() as u32, k as u32], "writer")
                };
                for &v in corr_f32.iter() {
                    max_resid = max_resid.max(v.abs());
                }
                // Self-check: recon(bulk)[protected] + bf16(R) must reconstruct the
                // protected entries to bf16 precision (proves the sidecar carries the
                // right correction; error is only the bf16 rounding of R, ~1e-4).
                {
                    let corr_bf16: Vec<f32> = f32_slice_to_bf16_bytes(&corr_f32)
                        .chunks_exact(2)
                        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    let s = &protected_resid;
                    let mut e = 0.0f32;
                    if is_reader {
                        for r in 0..m {
                            for (j, &c) in s.iter().enumerate() {
                                let got = recon[r * k + c] + corr_bf16[r * s.len() + j];
                                e = e.max((got - wf[r * k + c]).abs());
                            }
                        }
                    } else {
                        for (j, &row) in s.iter().enumerate() {
                            for c in 0..k {
                                let got = recon[row * k + c] + corr_bf16[j * k + c];
                                e = e.max((got - wf[row * k + c]).abs());
                            }
                        }
                    }
                    rq_recon_err = rq_recon_err.max(e);
                }
                let corr_name = format!("{}.rqcorr", t.name);
                sidecars.push(HfqTensor {
                    name: corr_name.clone(),
                    quant_type: QuantType::BF16,
                    shape,
                    group_size: 0,
                    data: f32_slice_to_bf16_bytes(&corr_f32),
                    spilled_len: 0,
                });
                sidecar_meta.insert(
                    t.name.clone(),
                    serde_json::json!({
                        "role": role,
                        "channels": protected_resid,
                        "corr": corr_name,
                    }),
                );
                if is_reader {
                    n_r += 1;
                } else {
                    n_w += 1;
                }
            }
            t.data = packed;
            t.quant_type = QuantType::MQ4G256;
            t.group_size = 256;
        }
        hfq_tensors.extend(sidecars);
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert(
                "roughquant_sidecar".to_string(),
                serde_json::json!({
                    "version": 1,
                    "saliency": "diag",
                    "protect_frac": protect_frac,
                    "d_model": dmodel,
                    "n_channels": n_prot,
                    "tensors": serde_json::Value::Object(sidecar_meta),
                }),
            );
        }
        eprintln!(
            "  roughquant (real): {n_r} reader + {n_w} writer correction sidecars \
             (bf16, max|R|={max_resid:.4}); bulk=MQ4G256; \
             protected-channel recon max-err={rq_recon_err:.2e} (bf16 rounding of R)"
        );
    }

    // permute5 (rq5) post-pass: apply the #5 residual-stream permutation OFFLINE.
    // Cluster the diag(H)-selected protected residual set S into a contiguous front
    // block and propagate the same permutation P across every residual-touching
    // tensor (embed cols, reader input-cols, writer output-rows, dim-wide RMSNorm γ,
    // lm_head cols). Bijective ⇒ model output unchanged. Verify: permuted-vs-original
    // KLD ≈ 0 on the working forward path.
    if use_roughquant5 {
        let protect_frac: f64 = std::env::var("HIPFIRE_RQ4_PROTECT_FRAC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.05);
        let dmodel = roughquant4_infer_dmodel(&hfq_tensors).unwrap_or(1024);
        // diag(H) residual-channel energy from true residual readers.
        let mut resid_energy = vec![0.0f64; dmodel];
        for t in hfq_tensors.iter() {
            if matches!(t.quant_type, QuantType::BF16)
                && t.shape.len() == 2
                && t.shape[1] as usize == dmodel
                && !t.name.contains("embed")
                && !t.name.contains("lm_head")
                && roughquant4_is_residual_reader(&t.name)
            {
                let key = t.name.strip_suffix(".weight").unwrap_or(&t.name);
                if let Some(diag) = qtip_hessian
                    .as_ref()
                    .and_then(|sc| sc.get(key, 0))
                    .filter(|h| h.k == dmodel)
                {
                    for i in 0..dmodel {
                        resid_energy[i] += diag.at(i, i);
                    }
                }
            }
        }
        let n_prot = ((protect_frac * dmodel as f64).round() as usize).min(dmodel);
        // Protected (top-energy) channels, then the rest — both ascending so the
        // permutation is deterministic. P[new] = old channel mapped to slot `new`.
        let mut order: Vec<usize> = (0..dmodel).collect();
        order.sort_unstable_by(|&a, &b| {
            resid_energy[b]
                .partial_cmp(&resid_energy[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut protected: Vec<usize> = order[..n_prot].to_vec();
        let mut rest: Vec<usize> = order[n_prot..].to_vec();
        protected.sort_unstable();
        rest.sort_unstable();
        let perm: Vec<usize> = protected.iter().chain(rest.iter()).copied().collect();
        debug_assert_eq!(perm.len(), dmodel);

        let (mut n_e, mut n_r, mut n_w, mut n_n) = (0usize, 0usize, 0usize, 0usize);
        for t in hfq_tensors.iter_mut() {
            if !matches!(t.quant_type, QuantType::BF16) {
                continue;
            }
            let mut wf: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let is_embed_or_head = t.name.contains("embed")
                || t.name.contains("lm_head")
                || t.name.ends_with("output.weight");
            if t.shape.len() == 2 {
                let m = t.shape[0] as usize;
                let k = t.shape[1] as usize;
                if k == dmodel && (is_embed_or_head || roughquant4_is_residual_reader(&t.name)) {
                    // input/residual columns (readers + embed/lm_head)
                    wf = permute_cols(&wf, m, k, &perm);
                    if is_embed_or_head {
                        n_e += 1;
                    } else {
                        n_r += 1;
                    }
                } else if m == dmodel && roughquant4_is_residual_writer(&t.name) {
                    // output/residual rows (writers)
                    wf = permute_rows(&wf, m, k, &perm);
                    n_w += 1;
                } else {
                    continue;
                }
            } else if t.shape.len() == 1 && t.shape[0] as usize == dmodel && t.name.contains("norm")
            {
                // dim-wide RMSNorm γ (input/post-attn/final) — elementwise on residual
                let g = wf.clone();
                for (j, &old) in perm.iter().enumerate() {
                    wf[j] = g[old];
                }
                n_n += 1;
            } else {
                continue;
            }
            t.data = f32_slice_to_bf16_bytes(&wf);
        }
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert(
                "roughquant_permutation".to_string(),
                serde_json::json!({
                    "version": 1,
                    "saliency": "diag",
                    "d_model": dmodel,
                    "n_protected": n_prot,
                    "protected_contiguous_front": true,
                    "perm": perm,
                }),
            );
        }
        eprintln!(
            "  permute5 (#5): d_model={dmodel}, S={n_prot} clustered to front [0..{n_prot}); \
             permuted {n_e} embed/head + {n_r} readers + {n_w} writers + {n_n} norms"
        );
    }

    // ── Gemma3 (1+w) RMSNorm baking (arch_id 12) ───────────────────────
    // Gemma stores norm weights `w` and applies `(1 + w)`. Add 1.0 to every
    // norm-weight tensor here so the standard rmsnorm kernel (plain `w`) is
    // numerically correct at runtime — no per-layer special-casing in the
    // gemma3 forward. Norms ship at source precision (F32/F16/BF16); convert,
    // offset, convert back to the same dtype. The `gemma_norm_offset=1.0`
    // metadata marker records that this happened.
    if arch_id == 12 || arch_id == 13 {
        let mut n_baked = 0usize;
        for t in hfq_tensors.iter_mut() {
            // Bake the gemma (1+w) RMSNorms (text norms + the projector's
            // mm_soft_emb_norm), but NOT the SigLIP vision tower's standard
            // LayerNorms — `vision_tower.*.post_layernorm.weight` also ends in
            // `norm.weight` and must be left untouched (arch_id 13).
            if !t.name.ends_with("norm.weight") || t.name.starts_with("vision_tower.") {
                continue;
            }
            if t.spilled_len > 0 {
                // A spilled norm can't be offset in place; shipping it unbaked
                // would silently corrupt the (1+w) convention. Refuse loudly.
                eprintln!(
                    "gemma3: FATAL norm tensor {} was spilled to disk before the +1 bake; \
                     re-run with a smaller --format or more RAM so norms stay resident",
                    t.name
                );
                std::process::exit(2);
            }
            let dtype = match t.quant_type {
                QuantType::F32 => "F32",
                QuantType::F16 => "F16",
                QuantType::BF16 => "BF16",
                other => {
                    eprintln!(
                        "gemma3: FATAL norm tensor {} has quant_type {} (expected \
                         F32/F16/BF16); cannot bake the (1+w) offset",
                        t.name, other as u8
                    );
                    std::process::exit(2);
                }
            };
            let mut vals = to_f32(&t.data, dtype);
            for v in vals.iter_mut() {
                *v += 1.0;
            }
            t.data = match t.quant_type {
                QuantType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
                QuantType::F16 => f32_slice_to_f16_bytes(&vals),
                QuantType::BF16 => f32_slice_to_bf16_bytes(&vals),
                _ => unreachable!(),
            };
            n_baked += 1;
        }
        eprintln!(
            "gemma3: baked +1.0 into {n_baked} RMSNorm weight tensors (zero-centered (1+w) convention)"
        );
    }

    // qtip3 (real) post-pass: pack every eligible 2D BF16 weight into
    // QuantType::Qtip3G256 records (rotated-frame 3-bit trellis symbols + scale,
    // 100 B/group), decoded at runtime by gemv_qtip3g256. Unlike the sim, the
    // weights are stored in the FWHT-rotated frame (NO inverse rotation); the
    // runtime FWHT-rotates x. A self-check re-decodes and reports max abs error
    // vs the effective sim weight so the producer is verified offline.
    if use_qtip3_real || use_qtip4_real {
        pack_qtip_real_tensors(
            &mut hfq_tensors,
            &qtip_cb,
            &qtip_s1,
            &qtip_s2,
            std::env::var("HIPFIRE_LOWRANK_R")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            qtip_bits,
            qtip_beam_width(),
        );
    }

    insert_parameter_counts_metadata(
        &mut metadata,
        &hfq_tensors,
        total_params,
        quantized_params,
        skipped_params,
    );

    // Write .hfq file
    eprintln!("\nWriting: {}", output_path.display());
    // Final spill before writing
    if let Some(ref mut s) = spill {
        maybe_spill(&mut hfq_tensors, s, 0); // spill everything remaining
    }
    let metadata_json =
        metadata_with_quantization_hash(metadata, &hfq_tensors, spill.as_ref()).unwrap();
    write_hfq(
        output_path,
        arch_id,
        &metadata_json,
        &hfq_tensors,
        spill.as_mut(),
    )
    .unwrap();
    if let Some(s) = spill {
        s.cleanup();
    }

    let file_size = std::fs::metadata(output_path).unwrap().len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
}

#[cfg(test)]
mod xxh64_provenance_tests {
    use super::*;

    #[test]
    fn xxh64_matches_known_vectors() {
        assert_eq!(xxh64_hex(b""), "ef46db3751d8e999");
        assert_eq!(xxh64_hex(b"hello"), "26c7827d889f6da3");
    }

    #[test]
    fn quantization_hash_is_inserted_into_metadata() {
        let tensors = vec![HfqTensor {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            quant_type: QuantType::MQ4G256,
            shape: vec![2, 4],
            group_size: 256,
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            spilled_len: 0,
        }];
        let metadata = serde_json::json!({ "architecture": "qwen3" });
        let metadata_json =
            metadata_with_quantization_hash(metadata, &tensors, None).expect("metadata");
        let parsed: serde_json::Value = serde_json::from_str(&metadata_json).unwrap();
        let hash = &parsed["quantization_hash"];
        assert_eq!(hash["algorithm"], "xxh64");
        assert_eq!(hash["scope"], "hfq_tensor_index_and_payload_v1");
        assert_eq!(hash["tensor_count"], 1);
        assert_eq!(hash["payload_bytes"], 8);
        assert!(hash["producer"]["hipfire_version"].is_string());
        assert!(hash["producer"].get("git_commit").is_some());
        assert!(hash["producer"].get("git_branch").is_some());
        assert!(hash["producer"].get("git_describe").is_some());
        assert!(hash["producer"].get("git_dirty").is_some());
    }

    #[test]
    fn quant_format_is_inserted_into_metadata() {
        let mut metadata = serde_json::json!({ "architecture": "qwen3" });
        insert_quant_format_metadata(&mut metadata, "mq4");

        assert_eq!(metadata["quant_format"], "mq4");
    }

    #[test]
    fn parameter_counts_metadata_records_dense_counts() {
        let tensors = vec![
            HfqTensor {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                quant_type: QuantType::MQ4G256,
                shape: vec![2, 4],
                group_size: 256,
                data: vec![],
                spilled_len: 0,
            },
            HfqTensor {
                name: "model.layers.0.input_layernorm.weight".to_string(),
                quant_type: QuantType::F16,
                shape: vec![4],
                group_size: 0,
                data: vec![],
                spilled_len: 0,
            },
        ];
        let metadata = serde_json::json!({
            "architecture": "qwen3_5",
            "config": { "hidden_size": 4 },
        });
        let counts = parameter_counts_metadata(&metadata, &tensors, 12, 8, 3);

        assert_eq!(counts["schema"], "hipfire.parameter_counts.v1");
        assert_eq!(counts["total_params"], 12);
        assert_eq!(counts["source_total_params"], 15);
        assert_eq!(counts["active_params"], 12);
        assert_eq!(counts["effective_params"], 12);
        assert_eq!(counts["quantized_params"], 8);
        assert_eq!(counts["skipped_params"], 3);
        assert!(counts.get("moe").is_none());
    }

    #[test]
    fn parameter_counts_metadata_scales_routed_moe_by_top_k() {
        let mut tensors = Vec::new();
        tensors.push(HfqTensor {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            quant_type: QuantType::MQ4G256,
            shape: vec![10],
            group_size: 256,
            data: vec![],
            spilled_len: 0,
        });
        for expert in 0..4 {
            tensors.push(HfqTensor {
                name: format!(
                    "model.language_model.layers.0.mlp.experts.{expert}.gate_up_proj.weight"
                ),
                quant_type: QuantType::MQ4G256,
                shape: vec![3, 5],
                group_size: 256,
                data: vec![],
                spilled_len: 0,
            });
            tensors.push(HfqTensor {
                name: format!(
                    "model.language_model.layers.0.mlp.experts.{expert}.down_proj.weight"
                ),
                quant_type: QuantType::MQ4G256,
                shape: vec![3, 5],
                group_size: 256,
                data: vec![],
                spilled_len: 0,
            });
        }
        let metadata = serde_json::json!({
            "architecture": "qwen3_5_moe",
            "config": {
                "text_config": {
                    "num_experts": 4,
                    "num_experts_per_tok": 2
                }
            },
        });
        let counts = parameter_counts_metadata(&metadata, &tensors, 130, 120, 0);

        assert_eq!(counts["total_params"], 130);
        assert_eq!(counts["source_total_params"], 130);
        assert_eq!(counts["moe"]["routed_expert_params"], 120);
        assert_eq!(counts["moe"]["routed_expert_active_params"], 60);
        assert_eq!(counts["active_params"], 70);
        assert_eq!(counts["effective_params"], 70);
        assert_eq!(counts["moe"]["num_experts"], 4);
        assert_eq!(counts["moe"]["num_experts_per_tok"], 2);
    }
}

#[cfg(test)]
mod gptq_damping_probe {
    //! Offline GPTQ-Lloyd damping sweep. Runs the GPTQ-Lloyd quant pipeline
    //! against synthetic DeepSeek V4-realistic weight distributions across a damping
    //! range, compares per-block reconstruction MSE to plain Lloyd. Catches
    //! a known failure mode where forward-error-propagation on FWHT-rotated
    //! (largely-decorrelated) weights INJECTS noise rather than removing it
    //! at moderate-to-high damping values — what the DeepSeek V4 MQ2-GPTQ-all run
    //! is suspected to be hitting.
    //!
    //! Run with:
    //!   cargo test -p hipfire-quantize gptq_damping_probe -- --nocapture
    use super::*;

    /// Deterministic Box-Muller-from-LCG Gaussian sampler — no external dep.
    /// Returns N samples with zero mean and unit variance.
    fn gaussian_samples(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut step = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as u64 & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut u1 = step() as f64;
            if u1 < 1e-12 {
                u1 = 1e-12;
            }
            let u2 = step() as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out.push((r * theta.cos()) as f32);
            if out.len() < n {
                out.push((r * theta.sin()) as f32);
            }
        }
        out
    }

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = *x as f64 - *y as f64;
            acc += d * d;
        }
        acc / a.len() as f64
    }

    fn run_one_distribution(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        // Unit column weights — what DeepSeek V4's mq2-gptq-all build passes.
        let unit: Vec<f32> = vec![1.0; n];

        eprintln!("\n=== {label} (n={n}) ===");

        let lloyd_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let lloyd_recon = dequantize_mq2g256_lloyd_to_f32(&lloyd_bytes, n, &signs1, &signs2);
        let lloyd_mse = mse(weights, &lloyd_recon);
        eprintln!("  Lloyd                  MSE = {:.6e}", lloyd_mse);

        for damping in [0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            // Inject env override since the quantizer reads it at fn entry.
            std::env::set_var("HIPFIRE_GPTQ_DAMPING", format!("{damping}"));
            let gptq_bytes = quantize_mq2g256_lloyd_gptq(weights, &unit, &signs1, &signs2);
            let gptq_recon = dequantize_mq2g256_lloyd_to_f32(&gptq_bytes, n, &signs1, &signs2);
            let gptq_mse = mse(weights, &gptq_recon);
            let delta = ((gptq_mse - lloyd_mse) / lloyd_mse) * 100.0;
            eprintln!(
                "  GPTQ d={damping:>4.1}             MSE = {:.6e}  ({:+.2}% vs Lloyd)",
                gptq_mse, delta
            );
        }
        std::env::remove_var("HIPFIRE_GPTQ_DAMPING");
    }

    /// Variant of plain Lloyd with tunable iteration count. Used to test
    /// whether the production 8-iter cap is leaving headroom.
    fn quantize_mq2g256_lloyd_niter(
        f32_data: &[f32],
        signs1: &[f32],
        signs2: &[f32],
        max_iter: usize,
    ) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        let mut sums = [0.0f64; 4];
                        let mut counts = [0u32; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev_assignments[i] != best as u8 {
                                changed += 1;
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                            sums[best] += w as f64;
                            counts[best] += 1;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if counts[k] > 0 {
                                cb[k] = (sums[k] / counts[k] as f64) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    fn run_lloyd_iter_sweep(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (n={n}) — Lloyd iteration sweep ===");
        let mut prev = f64::NAN;
        for niter in [1usize, 2, 4, 8, 16, 32, 64] {
            let bytes = quantize_mq2g256_lloyd_niter(weights, &signs1, &signs2, niter);
            let recon = dequantize_mq2g256_lloyd_to_f32(&bytes, n, &signs1, &signs2);
            let m = mse(weights, &recon);
            let delta = if prev.is_finite() {
                format!("  ({:+.3}% vs niter=prev)", ((m - prev) / prev) * 100.0)
            } else {
                String::new()
            };
            eprintln!("  Lloyd niter={niter:>3}        MSE = {m:.6e}{delta}");
            prev = m;
        }
    }

    /// Huber-Lloyd: same Lloyd loop but the centroid update is the
    /// weighted-mean of points with |w - cb| ≤ k_huber * sigma, where
    /// sigma is the within-cluster standard deviation. Points with
    /// larger residuals get clipped (treated as `cb ± k_huber * sigma`)
    /// so they don't drag centroids toward outlier values. With FWHT-
    /// rotated weights the long tails are dampened but not eliminated;
    /// this tests whether residual heavy-tailedness is hurting MSE.
    fn quantize_mq2g256_huber_lloyd(
        f32_data: &[f32],
        signs1: &[f32],
        signs2: &[f32],
        k_huber: f32,
        max_iter: usize,
    ) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        // Assignment pass — same as plain Lloyd.
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                        }
                        // Within-cluster sigma estimate (one pass).
                        let mut sums = [0.0f64; 4];
                        let mut sqs = [0.0f64; 4];
                        let mut cnts = [0u32; 4];
                        for i in 0..256 {
                            let k = indices[i] as usize;
                            let d = (group[i] - cb[k]) as f64;
                            sums[k] += group[i] as f64;
                            sqs[k] += d * d;
                            cnts[k] += 1;
                        }
                        let mut sigma = [0.0f64; 4];
                        for k in 0..4 {
                            if cnts[k] > 0 {
                                sigma[k] = (sqs[k] / cnts[k] as f64).sqrt();
                            }
                        }
                        // Huber-clipped update.
                        let mut wsums = [0.0f64; 4];
                        let mut wcnts = [0.0f64; 4];
                        for i in 0..256 {
                            let k = indices[i] as usize;
                            let lim = (k_huber as f64) * sigma[k].max(1e-9);
                            let resid = (group[i] - cb[k]) as f64;
                            let clipped = resid.max(-lim).min(lim);
                            let effective_w = cb[k] as f64 + clipped;
                            wsums[k] += effective_w;
                            wcnts[k] += 1.0;
                        }
                        let mut changed = 0u32;
                        for k in 0..4 {
                            if wcnts[k] > 0.0 {
                                let new_cb = (wsums[k] / wcnts[k]) as f32;
                                if new_cb != cb[k] {
                                    changed += 1;
                                }
                                cb[k] = new_cb;
                            }
                        }
                        // Suppress unused warnings on sums.
                        let _ = sums;
                        if it > 0 && changed == 0 {
                            break;
                        }
                    }
                    // Final argmin pass to lock indices to the final centroids.
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d {
                                best_d = d;
                                best = k;
                            }
                        }
                        indices[i] = best as u8;
                    }
                }
                // Sort centroids, remap, pack.
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    fn run_huber_sweep(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (n={n}) — Huber-Lloyd sweep (16 iter) ===");
        // Reference: plain Lloyd at 16 iter.
        let ref_bytes = quantize_mq2g256_lloyd_niter(weights, &signs1, &signs2, 16);
        let ref_recon = dequantize_mq2g256_lloyd_to_f32(&ref_bytes, n, &signs1, &signs2);
        let ref_mse = mse(weights, &ref_recon);
        eprintln!("  Lloyd (niter=16)          MSE = {ref_mse:.6e}");
        for k_huber in [1.0_f32, 1.5, 2.0, 2.5, 3.0, 10.0] {
            let bytes = quantize_mq2g256_huber_lloyd(weights, &signs1, &signs2, k_huber, 16);
            let recon = dequantize_mq2g256_lloyd_to_f32(&bytes, n, &signs1, &signs2);
            let m = mse(weights, &recon);
            let delta = ((m - ref_mse) / ref_mse) * 100.0;
            eprintln!(
                "  Huber k={k_huber:>4.1} (niter=16)   MSE = {m:.6e}  ({delta:+.2}% vs Lloyd16)"
            );
        }
    }

    /// GPTQ sequential pass on already-FWHT'd weights, no inner FWHT.
    /// Used to A/B test the FWHT-position hypothesis: production GPTQ
    /// FWHTs then propagates → noise injection. Pre-FWHT GPTQ
    /// (correlated input) should help when input weights have
    /// channel correlation.
    fn quantize_mq2g256_lloyd_gptq_no_fwht(
        f32_data: &[f32],
        damping: f32,
        max_iter: usize,
    ) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                // NO FWHT here — operate on raw correlated weights.
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                if range > 0.0 {
                    let mut prev = [0u8; 256];
                    for it in 0..max_iter {
                        let mut sums = [0.0f64; 4];
                        let mut counts = [0u32; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev[i] != best as u8 {
                                changed += 1;
                            }
                            prev[i] = best as u8;
                            sums[best] += w as f64;
                            counts[best] += 1;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if counts[k] > 0 {
                                cb[k] = (sums[k] / counts[k] as f64) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                }
                let cb_final = sorted_cb;
                // Sequential GPTQ with no inner FWHT.
                let mut indices = [0u8; 256];
                let mut residual = 0.0f32;
                for i in 0..256 {
                    let target = group[i] + residual;
                    let mut best = 0usize;
                    let mut best_d = (target - cb_final[0]).abs();
                    for k in 1..4 {
                        let d = (target - cb_final[k]).abs();
                        if d < best_d {
                            best_d = d;
                            best = k;
                        }
                    }
                    indices[i] = best as u8;
                    let err = target - cb_final[best];
                    residual = err * damping;
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(cb_final[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    /// Dequant the no-FWHT variant: indices + codebook, no inv-FWHT step.
    fn dequant_no_fwht(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let global_i = b * 256 + 4 * i + j;
                    if global_i < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[global_i] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    fn correlated_weights(n: usize, seed: u64, decay: f32) -> Vec<f32> {
        // AR(1) process: x_t = decay * x_{t-1} + sqrt(1 - decay^2) * z_t.
        // Produces channel-correlated weights (decay > 0).
        let gauss = gaussian_samples(n, seed);
        let mut out = Vec::with_capacity(n);
        let mut prev = 0.0f32;
        let noise_scale = (1.0f32 - decay * decay).sqrt();
        for &g in &gauss {
            let v = decay * prev + noise_scale * g;
            out.push(v);
            prev = v;
        }
        out
    }

    /// Dequant for MQ3-Lloyd (qt=20): 16 B fp16 codebook (8 entries) +
    /// 96 B 3-bit packed indices = 112 B / 256 weights.
    fn dequantize_mq3g256_lloyd_to_f32(
        data: &[u8],
        n_weights: usize,
        signs1: &[f32],
        signs2: &[f32],
    ) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 112;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        assert!(data.len() >= n_blocks * block_bytes);
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let mut cb = [0.0f32; 8];
            for k in 0..8 {
                cb[k] = f16_to_f32(u16::from_le_bytes([blk[2 * k], blk[2 * k + 1]]));
            }
            let mut group = [0.0f32; 256];
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = blk[bo];
                let b1 = blk[bo + 1];
                let b2 = blk[bo + 2];
                let mut q = [0u8; 8];
                q[0] = b0 & 7;
                q[1] = (b0 >> 3) & 7;
                q[2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2);
                q[3] = (b1 >> 1) & 7;
                q[4] = (b1 >> 4) & 7;
                q[5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1);
                q[6] = (b2 >> 2) & 7;
                q[7] = (b2 >> 5) & 7;
                for j in 0..8 {
                    group[chunk * 8 + j] = cb[q[j] as usize];
                }
            }
            cpu_inv_fwht_256(&mut group, signs1, signs2);
            let actual = (n_weights - b * 256).min(256);
            for j in 0..actual {
                out[b * 256 + j] = group[j];
            }
        }
        out
    }

    /// Quantifies the MSE cost of antirez's MQ3 → MQ2 down-projection
    /// downgrade. Procedure: take a synthetic DeepSeek V4-realistic weight
    /// distribution, quantize via MQ3-Lloyd (treat its dequant as the
    /// best-fit-available reference), then RE-quantize that dequant via
    /// MQ2-Lloyd. MSE delta = "what antirez loses by dropping MQ3 down".
    ///
    /// Result feeds the question: is the antirez precision tax (2/3 × MQ2
    /// + 1/3 × MQ3 ≈ 2.7 bpw vs 2.25 bpw all-MQ2, ~13 GB on a 256-expert
    /// 43-layer DeepSeek V4) buying meaningful per-tensor MSE reduction, or is
    /// the antirez win at high ctx mostly from Q8 attention?
    fn antirez_downgrade_cost(label: &str, weights: &[f32]) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        let mq3_bytes = quantize_mq3g256_lloyd(weights, &signs1, &signs2);
        let mq3_recon = dequantize_mq3g256_lloyd_to_f32(&mq3_bytes, n, &signs1, &signs2);
        let mq2_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let mq2_recon = dequantize_mq2g256_lloyd_to_f32(&mq2_bytes, n, &signs1, &signs2);
        // Direct MSE against the synthetic input (ground truth):
        let mq3_mse = mse(weights, &mq3_recon);
        let mq2_mse = mse(weights, &mq2_recon);
        let downgrade_pct = ((mq2_mse - mq3_mse) / mq3_mse) * 100.0;
        eprintln!("  {label} (n={n})");
        eprintln!("    MQ3-Lloyd (3.5 bpw) MSE = {mq3_mse:.6e}");
        eprintln!("    MQ2-Lloyd (2.25 bpw) MSE = {mq2_mse:.6e}");
        eprintln!("    MQ3→MQ2 downgrade cost: {downgrade_pct:+.1}% MSE");
    }

    #[test]
    fn antirez_mq3_to_mq2_downgrade_cost() {
        // Tests on the same DeepSeek V4-realistic distributions as the GPTQ probe.
        eprintln!("\n=== Antirez MQ3-down → MQ2-down downgrade cost ===");
        antirez_downgrade_cost("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        antirez_downgrade_cost("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        antirez_downgrade_cost("Sparse + outliers 16x256", &sw);
    }

    #[test]
    fn gptq_on_correlated_pre_fwht() {
        // The whole point of GPTQ is to exploit channel correlation.
        // Test it on correlated (decay=0.7), modestly-correlated (0.4),
        // and uncorrelated (0.0) inputs WITHOUT the inner FWHT step.
        //
        // If d>0 wins on correlated inputs but loses on uncorrelated,
        // that confirms: the production code's mistake is FWHT-then-GPTQ.
        // Fix path: drop the FWHT before the sequential pass (move it
        // into dequant or change the runtime kernel to apply it on
        // dequant'd values).
        eprintln!("\n=== GPTQ on correlated weights (no inner FWHT) ===");
        for (label, decay) in [
            ("decay=0.0 (uncorrelated)", 0.0f32),
            ("decay=0.4 (moderately correlated)", 0.4),
            ("decay=0.7 (strongly correlated)", 0.7),
            ("decay=0.9 (very correlated)", 0.9),
        ] {
            let n = 16 * 256;
            let w = correlated_weights(n, 0xc011a7ed, decay);
            // Reference: plain Lloyd via no-FWHT path with d=0.
            let ref_bytes = quantize_mq2g256_lloyd_gptq_no_fwht(&w, 0.0, 16);
            let ref_recon = dequant_no_fwht(&ref_bytes, n);
            let ref_mse = mse(&w, &ref_recon);
            eprintln!("\n  {label} (n={n})");
            eprintln!("    Lloyd                  MSE = {ref_mse:.6e}");
            for damping in [0.05f32, 0.1, 0.2, 0.3, 0.5, 0.8] {
                let b = quantize_mq2g256_lloyd_gptq_no_fwht(&w, damping, 16);
                let r = dequant_no_fwht(&b, n);
                let m = mse(&w, &r);
                let delta = ((m - ref_mse) / ref_mse) * 100.0;
                eprintln!(
                    "    GPTQ d={damping:>4.2} (no-fwht)   MSE = {m:.6e}  ({delta:+.2}% vs Lloyd)"
                );
            }
        }
    }

    #[test]
    fn huber_lloyd_headroom() {
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_huber_sweep("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_huber_sweep("Sparse + outliers 16x256", &sw);
        run_huber_sweep("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
    }

    /// Test "weight-norm proxy imatrix": a calibration-free approximation
    /// using column 2-norm of the weight matrix itself as the per-channel
    /// importance signal. Real AWQ uses sum_t |a_tj|^2; we substitute
    /// sum_i |w_ij|^2. Both produce a [K]-shaped vector that's used to
    /// weight the Lloyd codebook fit.
    ///
    /// If this gives meaningful MSE improvement over uniform Lloyd on
    /// heavy-tailed distributions, it's a viable calibration-free path
    /// to better DeepSeek V4 quants. Bench-falsified if it doesn't beat uniform
    /// by a clear margin.
    fn weight_norm_proxy_imatrix(weights: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut col_norms = vec![0.0f32; k];
        for r in 0..m {
            for j in 0..k {
                let w = weights[r * k + j];
                col_norms[j] += w * w;
            }
        }
        for v in col_norms.iter_mut() {
            *v = v.sqrt();
        }
        // Normalize so geometric mean is 1.0 (matches AWQ convention).
        let mut sum_log = 0.0f64;
        for &v in &col_norms {
            sum_log += (v.max(1e-12) as f64).ln();
        }
        let mean_log = sum_log / k as f64;
        for v in col_norms.iter_mut() {
            *v = ((*v as f64).ln() - mean_log).exp() as f32;
        }
        col_norms
    }

    fn run_weight_norm_proxy_sweep(label: &str, weights: &[f32], m: usize, k: usize) {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let n = weights.len();
        eprintln!("\n=== {label} (m={m}, k={k}, n={n}) ===");
        // Uniform Lloyd baseline.
        let ref_bytes = quantize_mq2g256_lloyd(weights, &signs1, &signs2);
        let ref_recon = dequantize_mq2g256_lloyd_to_f32(&ref_bytes, n, &signs1, &signs2);
        let ref_mse = mse(weights, &ref_recon);
        eprintln!("  Uniform Lloyd                MSE = {ref_mse:.6e}");
        // Weight-norm proxy imatrix.
        let col_imatrix = weight_norm_proxy_imatrix(weights, m, k);
        let proxy_bytes = quantize_mq2g256_lloyd_weighted(weights, &col_imatrix, &signs1, &signs2);
        let proxy_recon = dequantize_mq2g256_lloyd_to_f32(&proxy_bytes, n, &signs1, &signs2);
        let proxy_mse = mse(weights, &proxy_recon);
        let delta = ((proxy_mse - ref_mse) / ref_mse) * 100.0;
        eprintln!(
            "  Weight-norm-proxy Lloyd      MSE = {proxy_mse:.6e}  ({delta:+.2}% vs uniform)"
        );
    }

    /// Quantize via Lloyd WITHOUT the FWHT step — Lloyd applied directly
    /// to the natural (pre-rotation) weight distribution. Same 4-codepoint
    /// codebook + 2-bit indices.
    fn quantize_mq2g256_lloyd_no_fwht(f32_data: &[f32]) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = f32_data.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&f32_data[start..end]);
                // NO FWHT — Lloyd directly on natural distribution.
                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let max_iter = 16;
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        let mut sums = [0.0f64; 4];
                        let mut counts = [0u32; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev_assignments[i] != best as u8 {
                                changed += 1;
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                            sums[best] += w as f64;
                            counts[best] += 1;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if counts[k] > 0 {
                                cb[k] = (sums[k] / counts[k] as f64) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    fn dequant_mq2_no_fwht(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let global_i = b * 256 + 4 * i + j;
                    if global_i < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[global_i] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    /// Quantize W (natural basis) with imatrix-weighted Lloyd, no FWHT.
    /// Returns (codebook, indices) — both in natural basis.
    fn lloyd_imatrix_no_fwht(weights: &[f32], col_weights: &[f32]) -> Vec<u8> {
        use rayon::prelude::*;
        let group_size = 256;
        let block_bytes = 72;
        let n = weights.len();
        let n_blocks = (n + group_size - 1) / group_size;
        let mut output = vec![0u8; n_blocks * block_bytes];
        let blocks_per_row = col_weights.len() / group_size;
        output
            .par_chunks_mut(block_bytes)
            .enumerate()
            .for_each(|(b, out_chunk)| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let actual_len = end - start;
                let mut group = [0.0f32; 256];
                group[..actual_len].copy_from_slice(&weights[start..end]);
                // Use natural distribution; NO FWHT.
                let col_off = (b % blocks_per_row) * group_size;
                let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

                let mut sorted: [f32; 256] = group;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let percentile = |frac: f32| -> f32 {
                    let idx = ((frac * 255.0).round() as usize).min(255);
                    sorted[idx]
                };
                let mut cb: [f32; 4] = [
                    percentile(0.125),
                    percentile(0.375),
                    percentile(0.625),
                    percentile(0.875),
                ];
                let range = sorted[255] - sorted[0];
                let mut indices = [0u8; 256];
                if range > 0.0 {
                    let max_iter = 16;
                    let mut prev_assignments = [0u8; 256];
                    for it in 0..max_iter {
                        let mut wsums = [0.0f64; 4];
                        let mut wtotals = [0.0f64; 4];
                        let mut changed = 0u32;
                        for i in 0..256 {
                            let w = group[i];
                            let mut best = 0usize;
                            let mut best_d = (w - cb[0]).abs();
                            for k in 1..4 {
                                let d = (w - cb[k]).abs();
                                if d < best_d {
                                    best_d = d;
                                    best = k;
                                }
                            }
                            if it == 0 || prev_assignments[i] != best as u8 {
                                changed += 1;
                            }
                            prev_assignments[i] = best as u8;
                            indices[i] = best as u8;
                            let pw = block_w[i] as f64;
                            wsums[best] += pw * w as f64;
                            wtotals[best] += pw;
                        }
                        if it > 0 && changed == 0 {
                            break;
                        }
                        for k in 0..4 {
                            if wtotals[k] > 0.0 {
                                cb[k] = (wsums[k] / wtotals[k]) as f32;
                            }
                        }
                    }
                }
                let mut order: [usize; 4] = [0, 1, 2, 3];
                order.sort_by(|&a, &b| {
                    cb[a]
                        .partial_cmp(&cb[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut sorted_cb = [0.0f32; 4];
                let mut inv: [u8; 4] = [0; 4];
                for new_idx in 0..4 {
                    sorted_cb[new_idx] = cb[order[new_idx]];
                    inv[order[new_idx]] = new_idx as u8;
                }
                for i in 0..256 {
                    indices[i] = inv[indices[i] as usize];
                }
                for k in 0..4 {
                    let bits = f32_to_fp16_bits(sorted_cb[k]);
                    out_chunk[2 * k] = (bits & 0xFF) as u8;
                    out_chunk[2 * k + 1] = (bits >> 8) as u8;
                }
                for i in 0..64 {
                    let mut byte_val = 0u8;
                    for j in 0..4 {
                        byte_val |= (indices[4 * i + j] & 0x3) << (j * 2);
                    }
                    out_chunk[8 + i] = byte_val;
                }
            });
        output
    }

    fn dequant_no_fwht_natural(data: &[u8], n_weights: usize) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 72;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let cb: [f32; 4] = [
                f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
                f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
            ];
            for i in 0..64 {
                let bv = blk[8 + i];
                for j in 0..4 {
                    let gi = b * 256 + 4 * i + j;
                    if gi < n_weights {
                        let idx = (bv >> (j * 2)) & 0x3;
                        out[gi] = cb[idx as usize];
                    }
                }
            }
        }
        out
    }

    fn gemv_f32(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for r in 0..m {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += w[r * k + j] as f64 * x[j] as f64;
            }
            y[r] = acc as f32;
        }
        y
    }

    #[test]
    fn prefwht_imatrix_lloyd_value() {
        // Activation-weighted A/B test of post-FWHT vs pre-FWHT imatrix-Lloyd.
        // Generate W [m=256, k=4096] with HETEROGENEOUS column variances —
        // some columns have stddev=3, others stddev=0.1. Imatrix captures the
        // ground-truth importance. Run a gemv with this W against a random
        // unit-Gaussian X, then compare gemv-error for the two quant methods.
        //
        // If pre-FWHT-imatrix-Lloyd reduces gemv error meaningfully on
        // activations vs post-FWHT, that's the green light for the
        // pre-FWHT-Lloyd refactor (Action 5 in playbook).
        let m = 256;
        let k = 4096;
        let n = m * k;

        // Build heterogeneous-column W: column j has scale = log-uniform in
        // [0.1, 3.0] — gives 30x spread, mimics real LLM channel importance.
        let mut w = gaussian_samples(n, 0xc011c011);
        let mut col_scales = vec![0.0f32; k];
        let mut state: u64 = 0xc0ffeeed;
        for j in 0..k {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64;
            // log-uniform in [0.1, 3.0]
            col_scales[j] = (0.1_f64.ln() + u * (3.0_f64.ln() - 0.1_f64.ln())).exp() as f32;
        }
        for r in 0..m {
            for j in 0..k {
                w[r * k + j] *= col_scales[j];
            }
        }
        // Imatrix: per-column 2-norm of W (mimics what a real activation
        // imatrix produces — bigger for important channels). Geomean-normalize.
        let mut imatrix = vec![0.0f32; k];
        for j in 0..k {
            let mut sum2 = 0.0f64;
            for r in 0..m {
                sum2 += (w[r * k + j] as f64).powi(2);
            }
            imatrix[j] = sum2.sqrt() as f32;
        }
        let mut sum_log = 0.0f64;
        for &v in &imatrix {
            sum_log += (v.max(1e-12) as f64).ln();
        }
        let mean_log = sum_log / k as f64;
        for v in imatrix.iter_mut() {
            *v = ((*v as f64).ln() - mean_log).exp() as f32;
        }

        // Random unit-Gaussian X for activations.
        let x = gaussian_samples(k, 0xacd1ac);
        let y_ref = gemv_f32(&w, &x, m, k);

        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        // METHOD A: post-FWHT imatrix-Lloyd (production).
        let bytes_a = quantize_mq2g256_lloyd_weighted(&w, &imatrix, &signs1, &signs2);
        let recon_a = dequantize_mq2g256_lloyd_to_f32(&bytes_a, n, &signs1, &signs2);
        let y_a = gemv_f32(&recon_a, &x, m, k);
        let err_a: f64 = y_ref
            .iter()
            .zip(y_a.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        // METHOD B: pre-FWHT imatrix-Lloyd (proposed refactor).
        let bytes_b = lloyd_imatrix_no_fwht(&w, &imatrix);
        let recon_b = dequant_no_fwht_natural(&bytes_b, n);
        let y_b = gemv_f32(&recon_b, &x, m, k);
        let err_b: f64 = y_ref
            .iter()
            .zip(y_b.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        // METHOD C: post-FWHT uniform Lloyd (current production w/o imatrix).
        let bytes_c = quantize_mq2g256_lloyd(&w, &signs1, &signs2);
        let recon_c = dequantize_mq2g256_lloyd_to_f32(&bytes_c, n, &signs1, &signs2);
        let y_c = gemv_f32(&recon_c, &x, m, k);
        let err_c: f64 = y_ref
            .iter()
            .zip(y_c.iter())
            .map(|(r, q)| (*r as f64 - *q as f64).powi(2))
            .sum::<f64>()
            / m as f64;

        eprintln!("\n=== Pre-FWHT vs post-FWHT imatrix-Lloyd (activation-weighted) ===");
        eprintln!("  W shape [{m}, {k}], heterogeneous column variances (0.1-3.0x)");
        eprintln!("  Method A: post-FWHT imatrix-Lloyd (current prod)   gemv MSE = {err_a:.6e}");
        eprintln!("  Method B: pre-FWHT  imatrix-Lloyd (proposed)       gemv MSE = {err_b:.6e}");
        eprintln!("  Method C: post-FWHT uniform Lloyd (no imatrix)     gemv MSE = {err_c:.6e}");
        eprintln!();
        let ab = ((err_b - err_a) / err_a) * 100.0;
        let ac = ((err_a - err_c) / err_c) * 100.0;
        let bc = ((err_b - err_c) / err_c) * 100.0;
        eprintln!("  Δ A→B (pre-FWHT win):              {ab:+.2}%");
        eprintln!("  Δ C→A (current imatrix vs uniform):{ac:+.2}%");
        eprintln!("  Δ C→B (pre-FWHT vs uniform):       {bc:+.2}%");
    }

    #[test]
    fn fwht_value_audit() {
        // Hypothesis: FWHT-rotation makes Lloyd more accurate because the
        // rotation decorrelates weights toward a Gaussian distribution, and
        // Lloyd's 4 codepoints are MSE-optimal for Gaussian.
        //
        // Test: quantize the SAME synthetic distribution two ways:
        //   A) Lloyd with FWHT (production path)
        //   B) Lloyd without FWHT (natural distribution)
        // Compute MSE for each. If FWHT wins consistently, the rotation is
        // earning its complexity. If they're close, dropping FWHT unblocks
        // proper imatrix integration (per
        // project_lloyd_imatrix_fwht_channel_mixing).
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        let cases: &[(&str, Box<dyn Fn() -> Vec<f32>>)] = &[
            (
                "Gaussian 16x256",
                Box::new(|| gaussian_samples(16 * 256, 0xc001cafe)),
            ),
            (
                "Heavy-tailed 16x256",
                Box::new(|| {
                    let mut htw = gaussian_samples(16 * 256, 0xfeed);
                    let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
                    for (i, t) in tail.iter().enumerate() {
                        htw[i * 20] = t * 3.0;
                    }
                    htw
                }),
            ),
            (
                "Sparse + outliers 16x256",
                Box::new(|| {
                    let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
                    for v in sw.iter_mut() {
                        *v *= 0.1;
                    }
                    for i in 0..(16 * 256 / 20) {
                        sw[i * 20] *= 30.0;
                    }
                    sw
                }),
            ),
            (
                "Bimodal (50% near -1, 50% near +1)",
                Box::new(|| {
                    let mut bw = gaussian_samples(16 * 256, 0xb1ba1);
                    for (i, v) in bw.iter_mut().enumerate() {
                        *v = 0.3 * *v + if i % 2 == 0 { -1.0 } else { 1.0 };
                    }
                    bw
                }),
            ),
        ];

        eprintln!("\n=== FWHT value audit ===");
        eprintln!(
            "{:35} {:>14} {:>14} {:>10}",
            "distribution", "fwht MSE", "no-fwht MSE", "fwht win %"
        );
        for (label, gen) in cases {
            let w = gen();
            let n = w.len();
            let fwht_bytes = quantize_mq2g256_lloyd(&w, &signs1, &signs2);
            let fwht_recon = dequantize_mq2g256_lloyd_to_f32(&fwht_bytes, n, &signs1, &signs2);
            let fwht_mse = mse(&w, &fwht_recon);
            let nofwht_bytes = quantize_mq2g256_lloyd_no_fwht(&w);
            let nofwht_recon = dequant_mq2_no_fwht(&nofwht_bytes, n);
            let nofwht_mse = mse(&w, &nofwht_recon);
            let win_pct = ((nofwht_mse - fwht_mse) / nofwht_mse) * 100.0;
            eprintln!(
                "{:35} {:14.6e} {:14.6e} {:+9.2}%",
                label, fwht_mse, nofwht_mse, win_pct
            );
        }
    }

    #[test]
    fn weight_norm_proxy_imatrix_sweep() {
        // Generate synthetic [m, k] matrices that mimic DeepSeek V4's expert
        // shapes (m=2048, k=4096 for gate; m=4096, k=2048 for down).
        // Use heavy-tailed and sparse-outlier variants to stress the
        // proxy.
        let m = 2048;
        let k = 4096;
        let n = m * k;
        eprintln!("\n=== Weight-norm proxy imatrix sweep ===");
        run_weight_norm_proxy_sweep(
            "Gaussian [2048, 4096]",
            &gaussian_samples(n, 0xc001cafe),
            m,
            k,
        );
        // Heavy-tailed: 5% of weights drawn from N(0, 3).
        let mut htw = gaussian_samples(n, 0xfeed);
        let tail_count = n / 20;
        let tail = gaussian_samples(tail_count, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_weight_norm_proxy_sweep("Heavy-tailed [2048, 4096]", &htw, m, k);
        // Per-column variance heterogeneity: make column j scale with j/k.
        let mut col_het = gaussian_samples(n, 0xc011c011);
        for r in 0..m {
            for j in 0..k {
                let scale = 0.1 + 1.9 * (j as f32 / k as f32);
                col_het[r * k + j] *= scale;
            }
        }
        run_weight_norm_proxy_sweep("Per-column var heterogeneity", &col_het, m, k);
    }

    #[test]
    fn lloyd_iteration_headroom() {
        // The production 8-iter cap may or may not converge on heavy-tailed
        // distributions. Sweep niter ∈ {1, 2, 4, 8, 16, 32, 64} to find the
        // convergence floor — if 32 or 64 iter gives meaningfully lower
        // MSE than 8, that's free headroom (offline quant cost only).
        run_lloyd_iter_sweep("Gaussian 16x256", &gaussian_samples(16 * 256, 0xc001cafe));
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            htw[i * 20] = t * 3.0;
        }
        run_lloyd_iter_sweep("Heavy-tailed 16x256", &htw);
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_lloyd_iter_sweep("Sparse + outliers 16x256", &sw);
    }

    #[test]
    fn sweep_deepseek4_like_distributions() {
        // 1) Pure Gaussian — baseline.
        run_one_distribution("N(0,1), 256 weights", &gaussian_samples(256, 0xc001cafe));

        // 2) Pure Gaussian, larger sample — averages across multiple blocks.
        run_one_distribution(
            "N(0,1), 16x256 weights",
            &gaussian_samples(16 * 256, 0xc001cafe),
        );

        // 3) Heavy-tailed mixture — 5% from N(0, 3), rest N(0, 1).
        //    Mimics DeepSeek V4's expert distributions with occasional outliers.
        let mut htw = gaussian_samples(16 * 256, 0xfeed);
        let tail = gaussian_samples((16 * 256) / 20, 0xbeef);
        for (i, t) in tail.iter().enumerate() {
            // Sprinkle the tail in every 20th slot.
            htw[i * 20] = t * 3.0;
        }
        run_one_distribution("Heavy-tailed, 16x256 weights", &htw);

        // 4) Sparse weights — most near zero, a few large. Sometimes
        //    happens in attention-related projections.
        let mut sw = gaussian_samples(16 * 256, 0x5_a55e);
        for v in sw.iter_mut() {
            *v *= 0.1;
        }
        // Inject 5% large values.
        for i in 0..(16 * 256 / 20) {
            sw[i * 20] *= 30.0;
        }
        run_one_distribution("Sparse (10% scale, 5% × 30 outliers)", &sw);
    }
}

/// Real-DeepSeek V4 per-block diagnostic. Reads an HFQ file directly via memmap2
/// (bypasses the hipfire-runtime hfq reader which currently has a broken
/// arch dep — keeps this probe self-contained inside hipfire-quantize).
/// For each MQ2-Lloyd (qt=19) and MQ3-Lloyd (qt=20) tensor, samples up to
/// MAX_SAMPLE_BLOCKS blocks and computes per-block stats:
///   - codebook range (max_cb - min_cb)
///   - codepoint spacing variance (how uneven the codebook is)
///   - index entropy (uniform = 2 bits for MQ2, log2(8)=3 for MQ3)
/// Then ranks tensors by mean per-block range to identify which tensors
/// have the highest dynamic range (= hardest to compress at given bpw).
///
/// Run with: cargo test --release -p hipfire-quantize --
///           --ignored hfq_block_range_diag -- --nocapture
///
/// Reads path from HIPFIRE_QUANT_DIAG_PATH env var (default points at
/// a local DeepSeek V4 HFQ snapshot).
#[cfg(test)]
mod hfq_block_diag {
    use super::*;
    use memmap2::Mmap;
    use std::fs::File;
    use std::path::Path;

    struct TensorInfo {
        name: String,
        quant_type: u8,
        data_offset: usize,
        data_size: usize,
    }

    fn parse_hfq_metadata(path: &Path) -> std::io::Result<String> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        assert_eq!(&mmap[0..4], b"HFQM");
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut json_end = 0usize;
        for (i, &b) in mmap[metadata_offset..data_offset].iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                if b == b'\\' {
                    esc = true;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            if b == b'"' {
                in_str = true;
                continue;
            }
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
        Ok(String::from_utf8_lossy(&mmap[metadata_offset..metadata_offset + json_end]).to_string())
    }

    #[test]
    #[ignore]
    fn hfq_dump_metadata() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash-lloyd-mq2.hfq".to_string());
        let path = Path::new(&path_str);
        let json = parse_hfq_metadata(path).expect("parse");
        // Print just keys at top level + any "source" / "path" / "input" hints.
        eprintln!("=== Metadata from {path:?} (top 2000 chars) ===");
        let truncated: String = json.chars().take(2000).collect();
        eprintln!("{}", truncated);
        if json.len() > 2000 {
            eprintln!("... ({} chars total)", json.len());
        }
    }

    fn parse_hfq(path: &Path) -> std::io::Result<(Mmap, Vec<TensorInfo>)> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        assert_eq!(&mmap[0..4], b"HFQM", "Not HFQ");
        let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        // Find JSON end by brace-matching.
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut json_end = 0usize;
        for (i, &b) in mmap[metadata_offset..data_offset].iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                if b == b'\\' {
                    esc = true;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            if b == b'"' {
                in_str = true;
                continue;
            }
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
        let mut pos = metadata_offset + json_end;
        let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        assert_eq!(idx_n, n_tensors);
        pos += 4;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut cum = data_offset;
        for _ in 0..n_tensors {
            let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).into_owned();
            pos += name_len;
            let quant_type = mmap[pos];
            pos += 1;
            let n_dims = mmap[pos] as usize;
            pos += 1;
            for _ in 0..n_dims {
                pos += 4;
            }
            // Skip group_size u32.
            pos += 4;
            let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            tensors.push(TensorInfo {
                name,
                quant_type,
                data_offset: cum,
                data_size,
            });
            cum += data_size;
        }
        Ok((mmap, tensors))
    }

    fn classify(name: &str) -> &'static str {
        if name.contains("ffn.experts.") && name.ends_with("w1.weight") {
            return "routed.w1 (gate)";
        }
        if name.contains("ffn.experts.") && name.ends_with("w2.weight") {
            return "routed.w2 (down)";
        }
        if name.contains("ffn.experts.") && name.ends_with("w3.weight") {
            return "routed.w3 (up)";
        }
        if name.contains("shared_experts.w1") {
            return "shared.w1";
        }
        if name.contains("shared_experts.w2") {
            return "shared.w2";
        }
        if name.contains("shared_experts.w3") {
            return "shared.w3";
        }
        if name.ends_with("attn.wq_a.weight") || name.ends_with("attn.wq_b.weight") {
            return "attn.q";
        }
        if name.ends_with("attn.wkv.weight") {
            return "attn.kv";
        }
        if name.ends_with("attn.wo_a.weight") || name.ends_with("attn.wo_b.weight") {
            return "attn.wo";
        }
        if name.contains("compressor.wkv") || name.contains("compressor.wgate") {
            return "compressor";
        }
        if name.contains("indexer.") {
            return "indexer";
        }
        "other"
    }

    /// Stats per block at MQ2 (4 codepoints, 8 B codebook + 64 B indices = 72 B/group).
    fn block_stats_mq2(data: &[u8]) -> Option<(f32, f32, f32)> {
        if data.len() < 8 {
            return None;
        }
        let mut cb = [0.0f32; 4];
        for k in 0..4 {
            cb[k] = f16_to_f32(u16::from_le_bytes([data[2 * k], data[2 * k + 1]]));
        }
        let lo = cb.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let mean = cb.iter().sum::<f32>() / 4.0;
        let spacing_var = cb.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / 4.0;
        // Index histogram.
        let mut hist = [0u32; 4];
        for i in 0..64 {
            let b = data[8 + i];
            for j in 0..4 {
                hist[((b >> (j * 2)) & 0x3) as usize] += 1;
            }
        }
        let total: u32 = hist.iter().sum();
        let mut h_bits = 0.0f32;
        for &c in &hist {
            if c > 0 {
                let p = c as f32 / total as f32;
                h_bits -= p * p.log2();
            }
        }
        Some((range, spacing_var, h_bits))
    }

    /// Stats per block at MQ3 (8 codepoints, 16 B codebook + 96 B indices = 112 B/group).
    fn block_stats_mq3(data: &[u8]) -> Option<(f32, f32, f32)> {
        if data.len() < 16 {
            return None;
        }
        let mut cb = [0.0f32; 8];
        for k in 0..8 {
            cb[k] = f16_to_f32(u16::from_le_bytes([data[2 * k], data[2 * k + 1]]));
        }
        let lo = cb.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let mean = cb.iter().sum::<f32>() / 8.0;
        let spacing_var = cb.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / 8.0;
        // Reconstruct indices.
        let mut hist = [0u32; 8];
        for chunk in 0..32 {
            let bo = 16 + chunk * 3;
            let b0 = data[bo];
            let b1 = data[bo + 1];
            let b2 = data[bo + 2];
            let q = [
                b0 & 7,
                (b0 >> 3) & 7,
                ((b0 >> 6) & 3) | ((b1 & 1) << 2),
                (b1 >> 1) & 7,
                (b1 >> 4) & 7,
                ((b1 >> 7) & 1) | ((b2 & 3) << 1),
                (b2 >> 2) & 7,
                (b2 >> 5) & 7,
            ];
            for v in q {
                hist[v as usize] += 1;
            }
        }
        let total: u32 = hist.iter().sum();
        let mut h_bits = 0.0f32;
        for &c in &hist {
            if c > 0 {
                let p = c as f32 / total as f32;
                h_bits -= p * p.log2();
            }
        }
        Some((range, spacing_var, h_bits))
    }

    fn cpu_inv_fwht_local(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
        super::cpu_inv_fwht_256(x, signs1, signs2);
    }

    fn dequant_mq3_lloyd(
        data: &[u8],
        n_weights: usize,
        signs1: &[f32],
        signs2: &[f32],
    ) -> Vec<f32> {
        let group_size = 256;
        let block_bytes = 112;
        let n_blocks = (n_weights + group_size - 1) / group_size;
        let mut out = vec![0.0f32; n_weights];
        for b in 0..n_blocks {
            let blk = &data[b * block_bytes..(b + 1) * block_bytes];
            let mut cb = [0.0f32; 8];
            for k in 0..8 {
                cb[k] = f16_to_f32(u16::from_le_bytes([blk[2 * k], blk[2 * k + 1]]));
            }
            let mut group = [0.0f32; 256];
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = blk[bo];
                let b1 = blk[bo + 1];
                let b2 = blk[bo + 2];
                let q = [
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) & 3) | ((b1 & 1) << 2),
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) & 1) | ((b2 & 3) << 1),
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ];
                for j in 0..8 {
                    group[chunk * 8 + j] = cb[q[j] as usize];
                }
            }
            cpu_inv_fwht_local(&mut group, signs1, signs2);
            let actual = (n_weights - b * 256).min(256);
            for j in 0..actual {
                out[b * 256 + j] = group[j];
            }
        }
        out
    }

    fn qt_name(qt: u8) -> &'static str {
        match qt {
            1 => "F16",
            2 => "F32",
            3 => "Q8F16",
            5 => "Q8HFQ",
            6 => "HFQ4G256",
            7 => "HFQ4G128",
            13 => "MQ4G256",
            14 => "MQ8G256",
            15 => "MQ6G256",
            17 => "MQ3G256",
            18 => "MQ2G256",
            19 => "MQ2G256Lloyd",
            20 => "MQ3G256Lloyd",
            21 => "HFP4G32",
            24 => "MFP4G32",
            _ => "?",
        }
    }

    /// Sample a real DeepSeek V4 MQ2-Lloyd tensor, dequant a few blocks, and
    /// report the distribution shape. Compares against the synthetic
    /// distributions used in fwht_value_audit + GPTQ probes to see which
    /// our DeepSeek V4 weights actually resemble.
    #[test]
    #[ignore]
    fn hfq_dist_sample() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash-lloyd-mq2.hfq".to_string());
        let path = Path::new(&path_str);
        let (mmap, tensors) = parse_hfq(path).expect("parse hfq");

        // Take 8 different routed-expert tensors (w1, w2, w3 from a few
        // layers/experts) and one attention tensor + one shared tensor.
        let sample_names = [
            "layers.5.ffn.experts.0.w1.weight",   // gate (mid layer)
            "layers.5.ffn.experts.0.w2.weight",   // down
            "layers.5.ffn.experts.0.w3.weight",   // up
            "layers.20.ffn.experts.50.w1.weight", // gate (later layer)
            "layers.20.ffn.experts.50.w2.weight",
            "layers.40.ffn.experts.100.w2.weight", // down (deep layer)
            "layers.5.ffn.shared_experts.w2.weight", // shared down
            "layers.5.attn.wo_b.weight",           // attention output
        ];
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        eprintln!("\n=== Real DeepSeek V4 weight distribution stats (4096 weights per tensor) ===");
        eprintln!(
            "{:55} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "tensor", "qt", "mean", "stddev", "p99/sd", "kurtosis"
        );
        for sname in sample_names {
            let t_idx = tensors.iter().position(|t| t.name == sname);
            let t = match t_idx {
                Some(i) => &tensors[i],
                None => continue,
            };
            // Sample first 16 blocks = 4096 weights. Skip unsupported qts.
            let block_bytes = match t.quant_type {
                19 => 72,
                20 => 112,
                3 => 34,
                _ => {
                    eprintln!("  {:55} {:>2} (skip qt)", sname, t.quant_type);
                    continue;
                }
            };
            let n_blocks = (t.data_size / block_bytes).min(16);
            if n_blocks == 0 {
                continue;
            }
            let n_w = n_blocks * 256;
            let recon: Vec<f32> = if t.quant_type == 19 {
                let raw = &mmap[t.data_offset..t.data_offset + n_blocks * 72];
                super::dequantize_mq2g256_lloyd_to_f32(raw, n_w, &signs1, &signs2)
            } else if t.quant_type == 20 {
                let raw = &mmap[t.data_offset..t.data_offset + n_blocks * 112];
                dequant_mq3_lloyd(raw, n_w, &signs1, &signs2)
            } else {
                eprintln!(
                    "  {:55} {:>2} (unsupported qt for dequant, skipping)",
                    sname, t.quant_type
                );
                continue;
            };
            // Compute stats.
            let n = recon.len() as f64;
            let mean = recon.iter().map(|&x| x as f64).sum::<f64>() / n;
            let var = recon
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let stddev = var.sqrt();
            // Kurtosis (Pearson) — measures heavy-tailedness; Gaussian = 3.
            let mu4 = recon
                .iter()
                .map(|&x| (x as f64 - mean).powi(4))
                .sum::<f64>()
                / n;
            let kurt = mu4 / var.powi(2);
            // p99/sd — ratio of 99th percentile abs value to sd.
            let mut absvals: Vec<f64> = recon.iter().map(|&x| (x as f64 - mean).abs()).collect();
            absvals
                .sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99 = absvals[(absvals.len() * 99 / 100).min(absvals.len() - 1)];
            let p99_over_sd = p99 / stddev;
            eprintln!(
                "{:55} {:>2} {:>10.4e} {:>10.4e} {:>10.3} {:>10.3}",
                sname, t.quant_type, mean, stddev, p99_over_sd, kurt
            );
        }
        // Reference values from synthetic distributions:
        eprintln!("\nReference (synthetic):");
        eprintln!("  Gaussian:            p99/sd ≈ 2.33    kurtosis ≈ 3.0");
        eprintln!("  Heavy-tailed (5% × 3): p99/sd ≈ 2.5-3   kurtosis ≈ 3-6");
        eprintln!("  Sparse outliers:     p99/sd ≈ 10+     kurtosis ≈ 30+");
        eprintln!("  Bimodal:             p99/sd ≈ 1.5-2   kurtosis < 3 (platykurtic)");
    }

    #[test]
    #[ignore]
    fn hfq_inventory() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash-lloyd-mq2.hfq".to_string());
        let path = Path::new(&path_str);
        eprintln!("opening {path:?}");
        let (_mmap, tensors) = parse_hfq(path).expect("parse hfq");
        eprintln!("{} tensors", tensors.len());
        // Bucket by (family, qt).
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<(&'static str, u8), (u64, u64)> = BTreeMap::new();
        let mut total_bytes: u64 = 0;
        for t in &tensors {
            let fam = classify(&t.name);
            let e = counts.entry((fam, t.quant_type)).or_default();
            e.0 += 1;
            e.1 += t.data_size as u64;
            total_bytes += t.data_size as u64;
        }
        eprintln!(
            "{:30} {:>14} {:>8} {:>14}",
            "family", "qt", "count", "bytes"
        );
        for ((fam, qt), (cnt, bytes)) in &counts {
            eprintln!(
                "{:30} {:>2} {:12} {:>8} {:>14}",
                fam,
                qt,
                qt_name(*qt),
                cnt,
                bytes
            );
        }
        eprintln!(
            "\ntotal data bytes: {} ({:.2} GiB)",
            total_bytes,
            total_bytes as f64 / (1024.0_f64.powi(3))
        );
    }

    #[test]
    #[ignore]
    fn hfq_block_range_diag() {
        let path_str = std::env::var("HIPFIRE_QUANT_DIAG_PATH")
            .unwrap_or_else(|_| "/data/hipfire-models/deepseek-v4-flash-lloyd-mq2.hfq".to_string());
        let path = Path::new(&path_str);
        eprintln!("opening {path:?}");
        let (mmap, tensors) = parse_hfq(path).expect("parse hfq");
        eprintln!("{} tensors, file mapped", tensors.len());

        // Bucket by (family, qt) → list of (mean_range, mean_var, mean_entropy, n_blocks).
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<(&'static str, u8), Vec<(f32, f32, f32, usize)>> =
            BTreeMap::new();

        // Sample at most this many blocks per tensor; routed-expert tensors are
        // huge (~1 MB each in the layer's batched blob form, 256 experts × 43
        // layers = ~30k tensors). Cap CPU time.
        const MAX_BLOCKS_PER_TENSOR: usize = 64;

        for t in &tensors {
            if t.quant_type != 19 && t.quant_type != 20 {
                continue;
            }
            let block_bytes = if t.quant_type == 19 { 72 } else { 112 };
            let raw = &mmap[t.data_offset..t.data_offset + t.data_size];
            let n_blocks = t.data_size / block_bytes;
            if n_blocks == 0 {
                continue;
            }
            let stride = (n_blocks / MAX_BLOCKS_PER_TENSOR.min(n_blocks)).max(1);
            let mut sum_range = 0.0f64;
            let mut sum_var = 0.0f64;
            let mut sum_h = 0.0f64;
            let mut n_sampled = 0usize;
            let mut bi = 0;
            while bi < n_blocks {
                let blk = &raw[bi * block_bytes..(bi + 1) * block_bytes];
                let stats = if t.quant_type == 19 {
                    block_stats_mq2(blk)
                } else {
                    block_stats_mq3(blk)
                };
                if let Some((r, v, h)) = stats {
                    sum_range += r as f64;
                    sum_var += v as f64;
                    sum_h += h as f64;
                    n_sampled += 1;
                }
                bi += stride;
            }
            if n_sampled == 0 {
                continue;
            }
            let fam = classify(&t.name);
            buckets.entry((fam, t.quant_type)).or_default().push((
                (sum_range / n_sampled as f64) as f32,
                (sum_var / n_sampled as f64) as f32,
                (sum_h / n_sampled as f64) as f32,
                n_sampled,
            ));
        }

        eprintln!("\n=== Per-family block stats (sampled {MAX_BLOCKS_PER_TENSOR}/tensor) ===");
        eprintln!(
            "{:30} {:3} {:>6} {:>10} {:>10} {:>10}",
            "family", "qt", "tensors", "mean_range", "mean_var", "mean_entropy"
        );
        for ((fam, qt), entries) in &buckets {
            let n_tensors = entries.len();
            let mean_range =
                entries.iter().map(|(r, _, _, _)| *r as f64).sum::<f64>() / n_tensors as f64;
            let mean_var =
                entries.iter().map(|(_, v, _, _)| *v as f64).sum::<f64>() / n_tensors as f64;
            let mean_h =
                entries.iter().map(|(_, _, h, _)| *h as f64).sum::<f64>() / n_tensors as f64;
            eprintln!(
                "{:30} {:3} {:>6} {:>10.4} {:>10.4} {:>10.4}",
                fam, qt, n_tensors, mean_range, mean_var, mean_h
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn randomized_lowrank_recovers_lowrank() {
        // E = u1 v1ᵀ + u2 v2ᵀ (exactly rank 2). rank-2 approx ≈ exact; rank-0 = 0.
        let (m, k) = (24usize, 40usize);
        let u1: Vec<f32> = (0..m).map(|i| (i as f32 * 0.1).sin()).collect();
        let v1: Vec<f32> = (0..k).map(|j| (j as f32 * 0.2).cos()).collect();
        let u2: Vec<f32> = (0..m).map(|i| (i as f32 * 0.05 + 1.0).cos()).collect();
        let v2: Vec<f32> = (0..k).map(|j| (j as f32 * 0.07).sin()).collect();
        let mut e = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                e[i * k + j] = u1[i] * v1[j] + u2[i] * v2[j];
            }
        }
        let fro = |a: &[f32]| a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let e_norm = fro(&e);
        let resid = |r: usize| {
            let er = randomized_lowrank(&e, m, k, r);
            let d: Vec<f32> = e.iter().zip(&er).map(|(a, b)| a - b).collect();
            fro(&d) / e_norm
        };
        assert!((resid(0) - 1.0).abs() < 1e-6, "rank0 residual not full");
        assert!(
            resid(2) < 1e-3,
            "rank2 didn't recover rank-2 matrix: {}",
            resid(2)
        );
        assert!(resid(1) < resid(0), "rank1 should beat rank0");
    }

    #[test]
    fn bbt_spectral_diag_identity_and_rank1() {
        // H orthonormal ⇒ H·I·Hᵀ = I ⇒ diag = 1 for every coordinate.
        let k = 8;
        let mut c = vec![0.0f32; k * k];
        for i in 0..k {
            c[i * k + i] = 1.0;
        }
        let d = bbt_spectral_diag(&c, k);
        assert!(
            d.iter().all(|&v| (v - 1.0).abs() < 1e-4),
            "C=I diag {d:?} not all 1"
        );
        // Rank-1 C = v vᵀ ⇒ diag(H C Hᵀ)_l = (H v)_l². Total energy ‖v‖² preserved.
        let v: Vec<f32> = (0..k).map(|i| i as f32 + 1.0).collect();
        for i in 0..k {
            for j in 0..k {
                c[i * k + j] = v[i] * v[j];
            }
        }
        let d = bbt_spectral_diag(&c, k);
        let trace: f32 = d.iter().sum();
        let vnorm2: f32 = v.iter().map(|x| x * x).sum();
        assert!(
            (trace - vnorm2).abs() / vnorm2 < 1e-3,
            "spectral energy {trace} != ‖v‖² {vnorm2}"
        );
        // Scales positive; uniform influence (C=I) ⇒ all s ≈ (1+1)^α = √2 at α=0.5.
        let s = bbt_scales(&vec![1.0f32; k], k, 0.5);
        assert!(
            s.iter().all(|&x| (x - 2.0f32.sqrt()).abs() < 1e-3),
            "uniform scales {s:?} not √2"
        );
    }

    #[test]
    fn chat_template_override_replaces_existing_template() {
        let original = Some(json!({
            "chat_template": "old",
            "eos_token": "<|im_end|>"
        }));
        let updated = tokenizer_config_with_chat_template(original, "new-template".to_string());
        assert_eq!(updated["chat_template"], "new-template");
        assert_eq!(updated["eos_token"], "<|im_end|>");
    }

    #[test]
    fn chat_template_override_creates_minimal_config_when_missing() {
        let updated = tokenizer_config_with_chat_template(None, "{{ messages }}".to_string());
        assert_eq!(updated, json!({ "chat_template": "{{ messages }}" }));
    }

    #[test]
    fn chat_template_override_replaces_non_object_config_with_minimal_object() {
        let updated = tokenizer_config_with_chat_template(
            Some(json!("unexpected")),
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );
        assert_eq!(
            updated,
            json!({ "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}" })
        );
    }

    #[test]
    fn gguf_format_accepts_fp16_aliases() {
        for alias in ["fp16", "f16", "float16"] {
            assert_eq!(GgufFormat::from_flag(alias), Some(GgufFormat::F16));
        }
        assert_eq!(GgufFormat::F16.label(), "F16");
        for alias in ["bf16", "bfloat16"] {
            assert_eq!(GgufFormat::from_flag(alias), Some(GgufFormat::Bf16));
        }
        assert_eq!(GgufFormat::Bf16.label(), "BF16");
    }

    #[test]
    fn format_flags_are_canonicalized_before_dispatch() {
        assert_eq!(normalize_format_flag(" BF16 "), "bf16");
        assert_eq!(normalize_format_flag("Mq4G256"), "mq4g256");
        assert_eq!(normalize_format_flag("op4+"), "op4+");
        assert_eq!(normalize_format_flag("OQ4+"), "oq4+");
    }

    #[test]
    fn oq_plus_recipe_keeps_awq_and_ldlq_levels_distinct() {
        assert_eq!(oq4_calibration_recipe("oq4"), OqCalibrationRecipe::Plain);
        assert_eq!(oq4_calibration_recipe("oq4+"), OqCalibrationRecipe::Awq);
        assert_eq!(
            oq4_calibration_recipe("oq4++"),
            OqCalibrationRecipe::AwqLdlq
        );
        assert_eq!(oq4_calibration_recipe("op4+"), OqCalibrationRecipe::AwqLdlq);
        assert_eq!(oq8_calibration_recipe("oq8"), OqCalibrationRecipe::Plain);
        assert_eq!(oq8_calibration_recipe("oq8+"), OqCalibrationRecipe::Awq);
        assert_eq!(
            oq8_calibration_recipe("oq8++"),
            OqCalibrationRecipe::AwqLdlq
        );
        assert_eq!(oq8_calibration_recipe("op8+"), OqCalibrationRecipe::AwqLdlq);
    }

    #[test]
    fn required_format_arg_refuses_missing_format() {
        let args = vec![
            "hipfire-quantize".to_string(),
            "--input".to_string(),
            "model".to_string(),
            "--output".to_string(),
            "model.hfq".to_string(),
        ];
        assert!(required_format_arg(&args).is_err());
    }

    #[test]
    fn hfq_input_format_accepts_explicit_formats() {
        assert_eq!(
            HfqInputFormat::from_flag("q8f16"),
            Some(HfqInputFormat::Q8F16)
        );
        assert_eq!(HfqInputFormat::from_flag("q8"), Some(HfqInputFormat::Q8F16));
        assert_eq!(HfqInputFormat::from_flag("mq4"), Some(HfqInputFormat::Mq4));
        assert_eq!(HfqInputFormat::from_flag("op4"), Some(HfqInputFormat::Oq4));
        assert_eq!(
            HfqInputFormat::from_flag("op4-4"),
            Some(HfqInputFormat::Oq4)
        );
        assert_eq!(HfqInputFormat::from_flag("op4+"), Some(HfqInputFormat::Oq4));
        assert_eq!(
            HfqInputFormat::from_flag("op4-4+"),
            Some(HfqInputFormat::Oq4)
        );
        assert_eq!(
            HfqInputFormat::from_flag("op4-8+"),
            Some(HfqInputFormat::Oq4)
        );
        assert_eq!(
            HfqInputFormat::from_flag("op4+t"),
            Some(HfqInputFormat::OqPlusTiered)
        );
        assert_eq!(
            HfqInputFormat::from_flag("op4+c"),
            Some(HfqInputFormat::OqPlusCompact)
        );
        assert_eq!(HfqInputFormat::from_flag("op8"), Some(HfqInputFormat::Oq8));
        assert_eq!(
            HfqInputFormat::from_flag("op8-16"),
            Some(HfqInputFormat::Oq8)
        );
        assert_eq!(HfqInputFormat::from_flag("oq4"), Some(HfqInputFormat::Oq4));
        assert_eq!(HfqInputFormat::from_flag("oq4+"), Some(HfqInputFormat::Oq4));
        assert_eq!(
            HfqInputFormat::from_flag("oq4++"),
            Some(HfqInputFormat::Oq4)
        );
        assert_eq!(HfqInputFormat::from_flag("oq8"), Some(HfqInputFormat::Oq8));
        assert_eq!(
            HfqInputFormat::from_flag("oq8+"),
            Some(HfqInputFormat::Oq8Plus)
        );
        assert_eq!(
            HfqInputFormat::from_flag("oq8++"),
            Some(HfqInputFormat::Oq8Plus)
        );
        assert_eq!(
            HfqInputFormat::from_flag("op8+"),
            Some(HfqInputFormat::Oq8Plus)
        );
        assert_eq!(
            HfqInputFormat::from_flag("op8-16+"),
            Some(HfqInputFormat::Oq8Plus)
        );
        assert_eq!(
            HfqInputFormat::from_flag("bf16"),
            Some(HfqInputFormat::Bf16)
        );
    }

    #[test]
    fn hfq_input_rejects_already_quantized_source_tensors() {
        let result = quantize_hfq_source_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &[0; 136],
            QuantType::MQ4G256 as u8,
            &[1, 256],
            HfqInputFormat::Mq4,
        );
        let err = match result {
            Ok(_) => panic!("already-quantized HFQ tensor was accepted"),
            Err(err) => err,
        };
        assert!(err.contains("only source-precision HFQ tensors are supported"));
    }

    #[test]
    fn hfq_input_nemotron_mq4_promotes_sensitive_mamba_paths_to_q8() {
        let raw = f32_slice_to_f16_bytes(&vec![0.125; 2 * 256]);

        for name in [
            "backbone.layers.0.mixer.in_proj.weight",
            "backbone.layers.0.mixer.out_proj.weight",
        ] {
            let (_, qt, group, label) = quantize_hfq_source_tensor(
                name,
                &raw,
                QuantType::F16 as u8,
                &[2, 256],
                HfqInputFormat::Mq4,
            )
            .unwrap();
            assert_eq!(qt as u8, QuantType::Q8F16 as u8, "{name}");
            assert_eq!(group, 32, "{name}");
            assert_eq!(label, "Q8_F16", "{name}");
        }
    }

    #[test]
    fn awq_eligible_includes_lfm2_dense_and_conv_linears() {
        for name in [
            "model.layers.0.conv.in_proj.weight",
            "model.layers.0.conv.out_proj.weight",
            "model.layers.2.feed_forward.w1.weight",
            "model.layers.2.feed_forward.w3.weight",
            "model.layers.2.feed_forward.w2.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.self_attn.out_proj.weight",
        ] {
            assert!(awq_eligible(name), "{name}");
        }
    }

    fn roughquant4_test_tensor(name: &str, shape: &[u32]) -> HfqTensor {
        HfqTensor {
            name: name.to_string(),
            quant_type: QuantType::BF16,
            shape: shape.to_vec(),
            group_size: 0,
            data: Vec::new(),
            spilled_len: 0,
        }
    }

    #[test]
    fn roughquant4_classifies_residual_readers_and_writers_by_role() {
        for name in [
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.3.self_attn.k_proj.weight",
            "model.language_model.layers.3.self_attn.v_proj.weight",
        ] {
            assert!(roughquant4_is_residual_reader(name), "{name}");
            assert!(!roughquant4_is_residual_writer(name), "{name}");
        }

        for name in [
            "model.language_model.layers.3.self_attn.o_proj.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
        ] {
            assert!(roughquant4_is_residual_writer(name), "{name}");
            assert!(!roughquant4_is_residual_reader(name), "{name}");
        }
    }

    #[test]
    fn roughquant4_infers_dmodel_from_residual_readers_not_shape_literals() {
        let tensors = vec![
            // A writer with a 1024-wide internal input must not force d_model=1024.
            roughquant4_test_tensor("model.layers.0.self_attn.o_proj.weight", &[4096, 1024]),
            roughquant4_test_tensor("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]),
            roughquant4_test_tensor("model.layers.0.mlp.gate_proj.weight", &[11008, 4096]),
            roughquant4_test_tensor("model.layers.0.mlp.up_proj.weight", &[11008, 4096]),
        ];

        assert_eq!(roughquant4_infer_dmodel(&tensors), Some(4096));
    }

    #[test]
    fn source_precision_preserves_bf16_bytes() {
        let raw = vec![0x34, 0x12, 0x78, 0x56];
        let f32_data = [1.0, 2.0];
        let (data, quant_type, label) = source_precision_tensor_bytes(&raw, "BF16", &f32_data);
        assert_eq!(data, raw);
        assert_eq!(quant_type as u8, QuantType::BF16 as u8);
        assert_eq!(label, "BF16");
    }

    #[test]
    fn source_precision_converts_f32_to_f16_for_same_width_fallback() {
        let raw = vec![0; 8];
        let f32_data = [1.0, 2.0];
        let (data, quant_type, label) = source_precision_tensor_bytes(&raw, "F32", &f32_data);
        assert_eq!(data, vec![0x00, 0x3c, 0x00, 0x40]);
        assert_eq!(quant_type as u8, QuantType::F16 as u8);
        assert_eq!(label, "F16");
    }

    #[test]
    fn source_precision_preserves_f16_bytes() {
        let raw = vec![0x00, 0x3c, 0x00, 0x40];
        let f32_data = [1.0, 2.0];
        let (data, quant_type, label) = source_precision_tensor_bytes(&raw, "F16", &f32_data);
        assert_eq!(data, raw);
        assert_eq!(quant_type as u8, QuantType::F16 as u8);
        assert_eq!(label, "F16");
    }

    #[test]
    fn e2m1_lookup_matches_ocp_spec() {
        // OCP MX FP4 (E2M1) spec values for the 8 magnitude codes.
        // Sign bit (0x8) flips sign of the magnitude.
        let expected: &[(u8, f32)] = &[
            (0x0, 0.0),
            (0x1, 0.5),
            (0x2, 1.0),
            (0x3, 1.5),
            (0x4, 2.0),
            (0x5, 3.0),
            (0x6, 4.0),
            (0x7, 6.0),
            (0x8, -0.0),
            (0x9, -0.5),
            (0xA, -1.0),
            (0xB, -1.5),
            (0xC, -2.0),
            (0xD, -3.0),
            (0xE, -4.0),
            (0xF, -6.0),
        ];
        for &(nib, want) in expected {
            assert_eq!(
                e2m1_to_f32(nib),
                want,
                "e2m1_to_f32(0x{:x}) = {} want {}",
                nib,
                e2m1_to_f32(nib),
                want
            );
        }
    }

    #[test]
    fn e2m1_dequant_unpacks_nibbles_and_doubles_logical_cols() {
        // Storage: 1 row × 1 col-byte. Byte = 0x42 → low nibble 0x2 (=1.0),
        // high nibble 0x4 (=2.0). Scale: 1 row × 1 col, UE8M0=127 (=2^0=1.0).
        // → logical row should be [1.0, 2.0] (length 2).
        let (vals, shape) = dequantize_e2m1_ue8m0_to_f32(&[0x42], &[1, 1], &[127], &[1, 1]);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn e2m1_dequant_applies_ue8m0_scale() {
        // Byte = 0x12 → low=2 (=1.0), high=1 (=0.5). Scale byte 128 → 2^1=2.0.
        // → logical [2.0, 1.0].
        let (vals, _) = dequantize_e2m1_ue8m0_to_f32(&[0x12], &[1, 1], &[128], &[1, 1]);
        assert_eq!(vals, vec![2.0, 1.0]);
    }

    #[test]
    fn parse_layer_idx_safetensors_dense() {
        assert_eq!(
            parse_layer_idx("model.layers.0.self_attn.q_proj.weight"),
            Some(0)
        );
        assert_eq!(
            parse_layer_idx("model.layers.63.mlp.gate_proj.weight"),
            Some(63)
        );
    }

    #[test]
    fn parse_layer_idx_safetensors_moe() {
        assert_eq!(
            parse_layer_idx("model.language_model.layers.5.mlp.experts.0.gate_up_proj.weight"),
            Some(5)
        );
    }

    #[test]
    fn parse_layer_idx_gguf() {
        assert_eq!(parse_layer_idx("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_idx("blk.31.ffn_gate.weight"), Some(31));
    }

    #[test]
    fn parse_layer_idx_no_match() {
        assert_eq!(parse_layer_idx("token_embd.weight"), None);
        assert_eq!(parse_layer_idx("output.weight"), None);
    }

    #[test]
    fn kmap_norms_are_f16() {
        assert_eq!(
            kmap_resolve("model.layers.0.input_layernorm.weight", 64, false),
            QuantLevel::F16
        );
        assert_eq!(
            kmap_resolve("model.layers.30.post_attention_layernorm.weight", 64, false),
            QuantLevel::F16
        );
    }

    #[test]
    fn kmap_embeds_are_q8() {
        assert_eq!(
            kmap_resolve("model.embed_tokens.weight", 64, false),
            QuantLevel::Q8
        );
        assert_eq!(kmap_resolve("lm_head.weight", 64, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve("output.weight", 64, false), QuantLevel::Q8);
    }

    #[test]
    fn nemotron_h_keep_list() {
        // Quantize the linear projections (the bulk).
        for n in [
            "backbone.layers.0.mixer.in_proj.weight",
            "backbone.layers.0.mixer.out_proj.weight",
            "backbone.layers.1.mixer.up_proj.weight",
            "backbone.layers.1.mixer.down_proj.weight",
            "backbone.layers.12.mixer.q_proj.weight",
            "backbone.layers.12.mixer.k_proj.weight",
            "backbone.layers.12.mixer.v_proj.weight",
            "backbone.layers.12.mixer.o_proj.weight",
            "backbone.layers.1.mixer.gate.weight",
            "backbone.layers.1.mixer.shared_experts.up_proj.weight",
            "backbone.layers.1.mixer.shared_experts.down_proj.weight",
            "backbone.layers.1.mixer.experts.0.up_proj.weight",
            "backbone.layers.1.mixer.experts.0.down_proj.weight",
            "backbone.embeddings.weight",
            "lm_head.weight",
        ] {
            assert!(should_quantize(n), "{n} should quantize");
        }
        // Keep F16: recurrence + norm tensors (quantizing these corrupts the SSM).
        for n in [
            "backbone.layers.0.mixer.conv1d.weight", // depthwise filter [conv_dim,1,K]
            "backbone.layers.0.mixer.conv1d.bias",
            "backbone.layers.0.mixer.A_log",
            "backbone.layers.0.mixer.D",
            "backbone.layers.0.mixer.dt_bias",
            "backbone.layers.0.mixer.norm.weight", // RMSNormGated
            "backbone.layers.0.norm.weight",       // pre-block RMSNorm
            "backbone.norm_f.weight",              // final norm
            "backbone.layers.1.mixer.gate.e_score_correction_bias",
        ] {
            assert!(!should_quantize(n), "{n} should stay F16");
        }
        // Embeddings + lm_head → Q8 (not base mq4).
        assert_eq!(
            kmap_resolve("backbone.embeddings.weight", 42, false),
            QuantLevel::Q8
        );
        assert_eq!(kmap_resolve("lm_head.weight", 42, false), QuantLevel::Q8);
        assert!(
            is_q8_tensor("backbone.layers.1.mixer.gate.weight"),
            "Nemotron MoE router should be Q8-protected"
        );

        for n in [
            "backbone.layers.0.mixer.in_proj.weight",
            "backbone.layers.0.mixer.out_proj.weight",
            "backbone.layers.1.mixer.down_proj.weight",
            "backbone.layers.12.mixer.o_proj.weight",
        ] {
            assert!(is_nemotron_h_mq4_q8_protected(n), "{n} should be protected");
        }
        assert!(!is_nemotron_h_mq4_q8_protected(
            "backbone.layers.0.mixer.up_proj.weight"
        ));
    }

    #[test]
    fn kmap_moe_router_q8() {
        assert_eq!(
            kmap_resolve("model.language_model.layers.5.mlp.gate.weight", 64, true),
            QuantLevel::Q8
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.5.mlp.shared_expert_gate.weight",
                64,
                true
            ),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_router_not_promoted_on_dense() {
        // On a dense model, mlp.gate.weight is not a router — falls to edge/base
        assert_ne!(
            kmap_resolve("model.layers.30.mlp.gate.weight", 64, false),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_expert_ffn_promote6() {
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.30.mlp.experts.5.gate_up_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.30.mlp.experts.5.down_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_edge_layers_dense_ffn_only() {
        // Dense: FFN in edge layers — promoted
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.down_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.62.mlp.up_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.63.mlp.down_proj.weight", 64, false),
            QuantLevel::Promote6
        );
        // Dense: attn in edge layers — NOT promoted
        assert_eq!(
            kmap_resolve("model.layers.0.self_attn.q_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.63.self_attn.v_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.0.linear_attn.in_proj_qkv.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_edge_layers_moe_attn_and_ffn() {
        // MoE: both attn and FFN in edge layers — promoted
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve(
                "model.language_model.layers.63.self_attn.v_proj.weight",
                64,
                true
            ),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_middle_layers_base() {
        assert_eq!(
            kmap_resolve("model.layers.2.self_attn.q_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.30.mlp.gate_proj.weight", 64, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.61.mlp.down_proj.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_edge_layers_small_model_24_layers() {
        // 24 layers: edge = 0,1 and 22,23
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.2.mlp.gate_proj.weight", 24, false),
            QuantLevel::Base
        );
        assert_eq!(
            kmap_resolve("model.layers.22.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.23.mlp.gate_proj.weight", 24, false),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_n_layers_zero_disables_edge() {
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 0, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_edge_layers_tiny_model_3_layers() {
        // 3 layers: first-2 = {0,1}, last-2 = {1,2}. All layers promoted.
        assert_eq!(
            kmap_resolve("model.layers.0.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.1.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.layers.2.mlp.gate_proj.weight", 3, false),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_expert_not_promoted_on_dense() {
        // "mlp.experts." in name but is_moe=false — should NOT trigger rule 4
        assert_eq!(
            kmap_resolve(
                "model.layers.30.mlp.experts.5.gate_up_proj.weight",
                64,
                false
            ),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_gguf_names() {
        // GGUF edge-layer FFN (dense) — promoted
        assert_eq!(
            kmap_resolve("blk.0.ffn_gate.weight", 64, false),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("blk.63.ffn_gate.weight", 64, false),
            QuantLevel::Promote6
        );
        // GGUF edge-layer attn (dense) — NOT promoted
        assert_eq!(
            kmap_resolve("blk.0.attn_q.weight", 64, false),
            QuantLevel::Base
        );
        // GGUF edge-layer attn (MoE) — promoted
        assert_eq!(
            kmap_resolve("blk.0.attn_q.weight", 64, true),
            QuantLevel::Promote6
        );
        // GGUF middle-layer — base
        assert_eq!(
            kmap_resolve("blk.30.ffn_gate.weight", 64, false),
            QuantLevel::Base
        );
    }

    // ── Alternating mode tests ───────────────────────────────────────────

    #[test]
    fn positional_promote_edges() {
        assert!(is_positional_promote(0, 40, 3));
        assert!(is_positional_promote(1, 40, 3));
        assert!(is_positional_promote(38, 40, 3));
        assert!(is_positional_promote(39, 40, 3));
    }

    #[test]
    fn positional_promote_stride3() {
        // Middle layers: every 3rd starting from idx 2
        assert!(is_positional_promote(2, 40, 3)); // edge
        assert!(!is_positional_promote(3, 40, 3));
        assert!(!is_positional_promote(4, 40, 3));
        assert!(is_positional_promote(5, 40, 3));
        assert!(!is_positional_promote(6, 40, 3));
        assert!(!is_positional_promote(7, 40, 3));
        assert!(is_positional_promote(8, 40, 3));
    }

    #[test]
    fn kmap_alternating_moe_experts() {
        // MoE experts: promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.0.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Promote6 // edge layer
        );
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.5.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Promote6 // stride hit (5-2=3, 3%3==0)
        );
        assert_eq!(
            kmap_resolve_mode(
                "model.language_model.layers.3.mlp.experts.5.gate_up_proj.weight",
                40,
                true,
                1
            ),
            QuantLevel::Base // not on stride
        );
    }

    #[test]
    fn kmap_alternating_ffn_down() {
        // ffn_down promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // edge
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // stride
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.3.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Base // not on stride
        );
        // gate_proj NOT promoted in middle layers
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.gate_proj.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_n_layers_zero() {
        // With n_layers=0, alternating mode should return Base for everything
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 0, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_gguf_names() {
        // GGUF ffn_down in edge layer
        assert_eq!(
            kmap_resolve_mode("blk.0.ffn_down.weight", 40, false, 1),
            QuantLevel::Promote6
        );
        // GGUF ffn_down in middle non-stride layer
        assert_eq!(
            kmap_resolve_mode("blk.3.ffn_down.weight", 40, false, 1),
            QuantLevel::Base
        );
        // GGUF ffn_gate stays base in middle
        assert_eq!(
            kmap_resolve_mode("blk.5.ffn_gate.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_typed_promotes_down_and_v() {
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.down_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.15.self_attn.v_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        // gate_proj stays base
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.gate_proj.weight", 40, false, 2),
            QuantLevel::Base
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec golden-hash characterization battery.
//
// Pins the byte-exact output of every pure quantize/dequant codec on a fixed
// deterministic input, so the planned main.rs decomposition is provably
// behavior-preserving: move a codec into a module → its golden hash must not
// change. Reuses the in-tree xxh64. Harvest goldens with:
//   cargo test -p hipfire-quantize --bin hipfire-quantize \
//       codec_golden::harvest -- --ignored --nocapture
// then paste the table into `GOLDENS` and the locked test enforces it.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod codec_golden {
    use super::*;
    // oq3/oq6 live in the `codecs` library module; the parent's private
    // `use codecs::*` glob isn't re-exported through `use super::*`, so import
    // the ones this battery characterizes explicitly.
    use crate::codecs::{
        dequant_oq3g256, dequant_oq6g256, quantize_oq3g256, quantize_oq6g256,
    };

    /// Deterministic f32 stream with a few outliers (LCG; no rng dep).
    fn det_input(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
                let base = (s as f32 / 2_147_483_648.0) - 0.5;
                if i % 137 == 0 {
                    base * 12.0
                } else {
                    base
                } // sparse outliers
            })
            .collect()
    }

    /// (name, hash) for every characterized codec on the fixed input.
    fn codec_hashes() -> Vec<(&'static str, String)> {
        let x = det_input(1024, 7); // 4 groups of 256
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let (m, k) = (4usize, 256usize);
        let mut out: Vec<(&'static str, String)> = Vec::new();
        let mut h = |name: &'static str, bytes: &[u8]| out.push((name, xxh64_hex(bytes)));

        h("q4f16_g64", &quantize_q4f16_g64(&x));
        h("q4k", &quantize_q4k(&x));
        h("q4_as_q8", &quantize_q4_as_q8(&x));
        h("q8f16", &quantize_q8f16(&x));
        h("q8hfq", &quantize_q8hfq(&x, m, k).0);
        h("mq4g256", &quantize_mq4g256(&x, &s1, &s2));
        h(
            "mq4g256_clipsearch",
            &quantize_mq4g256_clipsearch(&x, &s1, &s2),
        );
        h(
            "mq6g256_clipsearch",
            &quantize_mq6g256_clipsearch(&x, &s1, &s2),
        );
        h(
            "mq3g256_clipsearch",
            &quantize_mq3g256_clipsearch(&x, &s1, &s2),
        );
        h(
            "mq2g256_clipsearch",
            &quantize_mq2g256_clipsearch(&x, &s1, &s2),
        );
        h(
            "mq8g256_clipsearch",
            &quantize_mq8g256_clipsearch(&x, &s1, &s2),
        );
        h("oq4g256", &quantize_oq4g256(&x, &s1, &s2));
        h("oq3g256", &quantize_oq3g256(&x, &s1, &s2));
        h("oq6g256", &quantize_oq6g256(&x, &s1, &s2));
        h("mq6g256", &quantize_mq6g256(&x, &s1, &s2));
        h("mq8g256", &quantize_mq8g256(&x, &s1, &s2));
        h("mq3g256", &quantize_mq3g256(&x, &s1, &s2));
        h("mq2g256", &quantize_mq2g256(&x, &s1, &s2));
        h("mq3g256_lloyd", &quantize_mq3g256_lloyd(&x, &s1, &s2));
        h("mq4g256_lloyd", &quantize_mq4g256_lloyd(&x, &s1, &s2));
        h("mq2g256_lloyd", &quantize_mq2g256_lloyd(&x, &s1, &s2));
        h("mq2g256_lloyd_k3", &quantize_mq2g256_lloyd_k3(&x, &s1, &s2));
        h("hfq4g256", &quantize_hfq4g256(&x));
        h("hfq4g128", &quantize_hfq4g128(&x));
        h("hfq6g256", &quantize_hfq6g256(&x));
        h("hfq3g256", &quantize_hfq3g256(&x));
        h("hfq3g128", &quantize_hfq3g128(&x));
        h("hfq2g256", &quantize_hfq2g256(&x));
        h("hfq2g128", &quantize_hfq2g128(&x));
        h("hfp4g32_2d", &quantize_hfp4g32_2d(&x, m, k));
        out
    }

    #[test]
    #[ignore = "harvest: run with --ignored --nocapture to print the GOLDENS table"]
    fn harvest() {
        println!("\n// ── paste into GOLDENS ──");
        for (name, hash) in codec_hashes() {
            println!("    (\"{name}\", \"{hash}\"),");
        }
        println!("// ── end ──\n");
    }

    /// Baked golden hashes (harvested 2026-06-20, pre-decomposition). Any change
    /// means a codec's byte output changed — intentional only with a deliberate
    /// re-harvest, NOT during a refactor.
    const GOLDENS: &[(&str, &str)] = &[
        ("q4f16_g64", "d73539f183a5fc5c"),
        ("q4k", "a0e0ef608325a2b3"),
        ("q4_as_q8", "40cc44f3ad42b5c1"),
        ("q8f16", "43be8c0f93de9cb3"),
        ("q8hfq", "29ca0c52ad9b58dc"),
        ("mq4g256", "6e9d532bbe5d38eb"),
        ("mq4g256_clipsearch", "7978e3644f11ed99"),
        ("mq6g256_clipsearch", "f906f337b9bd4df7"),
        ("mq3g256_clipsearch", "a57eada9ebb78586"),
        ("mq2g256_clipsearch", "a95cdd8e7672e915"),
        ("mq8g256_clipsearch", "8987f0aa7fdfb487"),
        ("oq4g256", "fceec61d1cb735b3"),
        ("oq3g256", "2e40f019fba65028"),
        ("oq6g256", "694673ac9a654c7d"),
        ("mq6g256", "c43cbf518aae87fe"),
        ("mq8g256", "8987f0aa7fdfb487"),
        ("mq3g256", "0c2f928a4236cf57"),
        ("mq2g256", "59868cdc5c1365e5"),
        ("mq3g256_lloyd", "74f67e18a2d18664"),
        ("mq4g256_lloyd", "cb3d8d86986d7c96"),
        ("mq2g256_lloyd", "03c6764c0758ee16"),
        ("mq2g256_lloyd_k3", "00373d88942eb232"),
        ("hfq4g256", "5992821562cf0292"),
        ("hfq4g128", "faa7aa15210a17c0"),
        ("hfq6g256", "3c03921c63d1e695"),
        ("hfq3g256", "9e7c9e1a051a0af3"),
        ("hfq3g128", "78bcc10ab672861d"),
        ("hfq2g256", "5b4f21ad97442c8d"),
        ("hfq2g128", "2c1e8211ff7f5e03"),
        ("hfp4g32_2d", "ef22d36907fb8454"),
    ];

    /// Locked characterization: every codec's byte output must match its golden.
    /// The safety net for the main.rs decomposition — moving a codec into a
    /// module must not change its output.
    #[test]
    fn codec_outputs_are_byte_stable() {
        let actual = codec_hashes();
        let want: std::collections::HashMap<&str, &str> = GOLDENS.iter().copied().collect();
        assert_eq!(
            actual.len(),
            GOLDENS.len(),
            "codec count drifted from goldens"
        );
        let mut drifted = Vec::new();
        for (name, hash) in &actual {
            match want.get(name) {
                Some(g) if *g == hash.as_str() => {}
                Some(g) => drifted.push(format!("  {name}: golden {g} != actual {hash}")),
                None => drifted.push(format!("  {name}: missing from GOLDENS")),
            }
        }
        assert!(
            drifted.is_empty(),
            "codec byte output drifted:\n{}",
            drifted.join("\n")
        );
    }

    /// MQ4+ clip-search must not increase (and on outlier data, must reduce)
    /// reconstruction error vs plain MQ4 — same byte layout, better-fitted scale.
    #[test]
    fn mq4_clipsearch_beats_or_matches_plain() {
        // Outlier-heavy input (where clip-search helps most).
        let x = det_input(1024, 11);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let plain = dequant_mq4g256(&quantize_mq4g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let clip = dequant_mq4g256(
            &quantize_mq4g256_clipsearch(&x, &s1, &s2),
            x.len(),
            &s1,
            &s2,
        );
        let mse = |rec: &[f32]| -> f64 {
            x.iter()
                .zip(rec)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / x.len() as f64
        };
        let (mp, mc) = (mse(&plain), mse(&clip));
        assert!(
            mc <= mp * 1.0001,
            "clip-search MSE {mc:.3e} worse than plain {mp:.3e}"
        );
        eprintln!(
            "mq4 plain MSE={mp:.4e}  clipsearch MSE={mc:.4e}  ({:.1}% lower)",
            100.0 * (mp - mc) / mp
        );
    }

    /// Opus OQ4 (symmetric signed-int4) must round-trip with quality comparable
    /// to affine MQ4 on FWHT-rotated weights (E6: affine vs symmetric is a wash).
    #[test]
    fn oq4_roundtrip_comparable_to_mq4() {
        let x = det_input(1024, 5);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let mq4 = dequant_mq4g256(&quantize_mq4g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let oq4 = dequant_oq4g256(&quantize_oq4g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let sqnr = |rec: &[f32]| -> f64 {
            let (mut sig, mut noise) = (0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(rec) {
                sig += (a as f64).powi(2);
                noise += ((a - b) as f64).powi(2);
            }
            10.0 * (sig / noise.max(1e-30)).log10()
        };
        let (m, o) = (sqnr(&mq4), sqnr(&oq4));
        eprintln!("mq4 SQNR={m:.2} dB  oq4 SQNR={o:.2} dB");
        assert!(o > 8.0, "oq4 SQNR {o:.2} dB too low (broken codec?)");
        assert!(o > m - 3.0, "oq4 {o:.2} dB >3 dB worse than mq4 {m:.2} dB");
    }

    /// Opus OQ3 (symmetric signed-int3, bit-plane) must round-trip coherently on
    /// FWHT-rotated weights: well below Oq4 (one fewer bit ≈ −6 dB) but far above
    /// noise, proving the bit-plane pack + 3-bit two's-complement sext are correct.
    #[test]
    fn oq3_roundtrip_below_oq4_above_noise() {
        let x = det_input(1024, 5);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let oq4 = dequant_oq4g256(&quantize_oq4g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let oq3 = dequant_oq3g256(&quantize_oq3g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let sqnr = |rec: &[f32]| -> f64 {
            let (mut sig, mut noise) = (0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(rec) {
                sig += (a as f64).powi(2);
                noise += ((a - b) as f64).powi(2);
            }
            10.0 * (sig / noise.max(1e-30)).log10()
        };
        let (o3, o4) = (sqnr(&oq3), sqnr(&oq4));
        eprintln!(
            "oq3 SQNR={o3:.2} dB  oq4 SQNR={o4:.2} dB  ({:.1} dB)",
            o3 - o4
        );
        assert!(
            o3 > 4.0,
            "oq3 SQNR {o3:.2} dB too low (broken bit-plane/sext?)"
        );
        assert!(
            o3 < o4,
            "oq3 {o3:.2} dB should be below oq4 {o4:.2} dB (one fewer bit)"
        );
    }

    /// Oq3 on-disk block is exactly 98 B per 256-group (3.0625 b/w): [f16][8×3 u32].
    #[test]
    fn oq3_block_size_is_98_bytes() {
        let x = det_input(256, 7);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        assert_eq!(quantize_oq3g256(&x, &s1, &s2).len(), 98);
    }

    /// Opus OQ6 (symmetric signed-int6) is the near-lossless mid-tier: well above Oq4
    /// (2 extra bits ≈ +12 dB) and below Oq8 — proves the 6-bit pack + sext6 are right.
    #[test]
    fn oq6_roundtrip_between_oq4_and_oq8() {
        let x = det_input(1024, 5);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let oq4 = dequant_oq4g256(&quantize_oq4g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let oq6 = dequant_oq6g256(&quantize_oq6g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let oq8 = dequant_oq8g256(&quantize_oq8g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let sqnr = |rec: &[f32]| -> f64 {
            let (mut sig, mut noise) = (0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(rec) {
                sig += (a as f64).powi(2);
                noise += ((a - b) as f64).powi(2);
            }
            10.0 * (sig / noise.max(1e-30)).log10()
        };
        let (o4, o6, o8) = (sqnr(&oq4), sqnr(&oq6), sqnr(&oq8));
        eprintln!("oq4 SQNR={o4:.2}  oq6={o6:.2}  oq8={o8:.2} dB");
        assert!(o6 > o4 + 6.0, "oq6 {o6:.2} dB not >6 dB over oq4 {o4:.2}");
        assert!(o6 < o8, "oq6 {o6:.2} dB should be below oq8 {o8:.2}");
    }

    /// W8A8 weight codec (Oq8G256) is near-lossless and far better than the W4A4
    /// (int4) codec — 4 extra bits ≈ +24 dB SQNR. Rung 1 of the W8A8 test ladder.
    #[test]
    fn oq8_roundtrip_is_near_lossless() {
        let x = det_input(1024, 5);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let oq4 = dequant_oq4g256(&quantize_oq4g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let oq8 = dequant_oq8g256(&quantize_oq8g256(&x, &s1, &s2), x.len(), &s1, &s2);
        let sqnr = |rec: &[f32]| -> f64 {
            let (mut sig, mut noise) = (0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(rec) {
                sig += (a as f64).powi(2);
                noise += ((a - b) as f64).powi(2);
            }
            10.0 * (sig / noise.max(1e-30)).log10()
        };
        let (o4, o8) = (sqnr(&oq4), sqnr(&oq8));
        eprintln!(
            "oq4 SQNR={o4:.2} dB  oq8 SQNR={o8:.2} dB  (+{:.1} dB)",
            o8 - o4
        );
        assert!(
            o8 > 35.0,
            "oq8 SQNR {o8:.2} dB not near-lossless (broken codec?)"
        );
        assert!(
            o8 > o4 + 15.0,
            "oq8 {o8:.2} dB should be >=15 dB better than oq4 {o4:.2} dB"
        );
    }

    /// FWHT must be exactly invertible (forward then inverse = identity).
    #[test]
    fn fwht_256_roundtrip_is_identity() {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let orig = det_input(256, 3);
        let mut buf = [0.0f32; 256];
        buf.copy_from_slice(&orig);
        cpu_fwht_256(&mut buf, &s1, &s2);
        cpu_inv_fwht_256(&mut buf, &s1, &s2);
        let max = orig
            .iter()
            .zip(buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "FWHT not invertible: max abs err {max}");
    }
}
