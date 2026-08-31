// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Maple-Preview weights + decode state.
//!
//! HFQ files carry RAW HF tensor names; the loader looks each up by exact name
//! (no rename). Mirrors the Cohere2-MoE loader (shared `WeightTensor`,
//! `KvCache`, indexed-MoE GEMV kernels) but reflects Maple's structure:
//!   * Standard pre-norm block: `input_layernorm` for attention,
//!     `post_attention_layernorm` for the MoE branch (NOT cohere2's parallel
//!     block with one shared norm).
//!   * **QK-norm**: per-head RMSNorm gammas of width `head_dim` on q and k.
//!   * **Every** layer is MoE — 256 experts, no dense prefix, no shared expert,
//!     no routing bias.
//!   * **Untied** lm_head (`lm_head.weight` is its own tensor), and the
//!     embedding is `model.word_embeddings.weight` — NOT `embed_tokens`.
//!
//! Expert weights ship pre-split (gate_proj/up_proj/down_proj); the loader
//! byte-fuses gate_proj‖up_proj into the per-expert `gate_up` blob the indexed
//! GEMV kernels expect. For Maple this fuse is always same-dtype (everything
//! ternary is qt=51), but the mismatch check is kept: a hand-assembled or
//! partially-requantized checkpoint would otherwise mis-read the up half.

use crate::config::MapleConfig;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Maple's embedding tensor. Named `word_embeddings`, not `embed_tokens` —
/// looking up the conventional name fails at load with "tensor not found".
pub const EMBED_TENSOR_NAME: &str = "model.word_embeddings.weight";
/// Untied output head.
pub const LM_HEAD_TENSOR_NAME: &str = "lm_head.weight";
/// Final RMSNorm gamma.
pub const FINAL_NORM_TENSOR_NAME: &str = "model.norm.weight";

/// Which projection of an expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProj {
    Gate,
    Up,
    Down,
}

impl ExpertProj {
    fn suffix(self) -> &'static str {
        match self {
            ExpertProj::Gate => "gate_proj",
            ExpertProj::Up => "up_proj",
            ExpertProj::Down => "down_proj",
        }
    }
}

/// Name of one expert projection tensor.
///
/// 18,432 of these (256 experts × 3 projections × 24 layers). A naming slip
/// fails at load, not at build, so the shape is pinned by test.
pub fn expert_tensor_name(layer: usize, expert: usize, proj: ExpertProj) -> String {
    format!(
        "model.layers.{layer}.mlp.experts.{expert}.{}.weight",
        proj.suffix()
    )
}

/// Name of a layer's ROUTER. Note this is `mlp.gate.weight` — distinct from an
/// expert's `mlp.experts.N.gate_proj.weight`. Matching on "gate" alone
/// conflates them.
pub fn router_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.weight")
}

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("maple: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Load a 1D/raw F16/BF16/F32/Q8 vector → F32 GpuTensor.
///
/// Used for RMSNorm gammas (per-layer, QK-norm, and final). Maple's converter
/// carries every norm as **BF16**, so the BF16 arm is the hot one here — an
/// F16-only loader would reject the model outright.
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data = widen_to_f32(qt, &data)
        .ok_or_else(|| format!("maple: expected F16/BF16/F32/Q8 for {name}, got qt={qt}"))?;
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("maple: upload {name}: {e:?}"))
}

/// Minimal Q8_0 dequant (32-elem blocks: little-endian f16 scale + 32 int8).
fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("maple: load_wt {name}: {e}"))
}

/// quant_type → DType. **qt=51 (`MQ2G256LloydU`) is the whole point of this
/// arch**: it is the unrotated MQ2-Lloyd sibling that carries Maple's native
/// ternary weights losslessly, and the dispatcher must NOT rotate x for it.
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        1 => DType::F16,
        2 => DType::F32,
        16 => DType::BF16,
        3 => DType::Q8_0,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        19 => DType::MQ2G256Lloyd,
        // qt=30 arrives only from `--head-quant mq4`. Unlike qt=51 this one IS
        // FWHT-rotated (`dtype_rotation_plan(MQ4G256Lloyd) == FwhtG256`), so
        // `weight_gemv` rotates x with `ensure_mq_signs` (seeds 42/1042) and
        // dispatches the *Prerotated* kernel. That is the same pair of seeds
        // `pack_maple_head` quantized against; if the two ever diverge the
        // result is not an error but silently wrong logits.
        30 => DType::MQ4G256Lloyd,
        // qt=44 arrives only from `--head-quant mq4v2`. Same FWHT-rotated
        // contract as qt=30 above — it resolves to GemvMq4G256V2Prerotated, so
        // weight_gemv rotates x with the same ensure_mq_signs seeds (42/1042)
        // that pack_maple_head quantized against. It differs from qt=30 only in
        // the 8 header bytes: a separate fp16 scale/zero per 128-weight half
        // rather than one pair per 256, at 4.25 bpw instead of 5.0.
        44 => DType::MQ4G256V2,
        51 => DType::MQ2G256LloydU,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

/// Widen raw F16/BF16/F32/Q8 bytes to F32, or `None` for anything else.
///
/// Split out of `load_f32` so the router can be materialised TWICE from one
/// read (once as its native `WeightTensor` for decode, once as the F16 mirror
/// batched prefill needs) without re-reading the tensor.
fn widen_to_f32(qt: u8, data: &[u8]) -> Option<Vec<f32>> {
    match qt {
        1 => Some(
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        ),
        2 => Some(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        3 => Some(dequant_q8_0(data)),
        16 => Some(
            data.chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        ),
        _ => None,
    }
}

/// F16 mirror of the router, `[n_exp × hidden]`, for the batched prefill GEMM.
///
/// **The router is the ONE Maple weight that is not qt51.** The published
/// checkpoint carries every attention projection and every expert as
/// MQ2G256LloydU (qt=51) but `model.layers.N.mlp.gate.weight` as **BF16**
/// (qt=16) — the reference even declares `router_dtype: fp32`. So
/// `batch::dense_qt51_gemm` CANNOT drive the router: that entry decodes
/// MQ2-Lloyd 72-byte groups, and pointed at BF16 bytes it would read them as
/// 2-bit codes and return plausible-looking garbage with no error and no
/// out-of-bounds fault (the BF16 blob is 8× the size the decoder reads).
///
/// `gemm_f16_x_f16_wmma` instead takes an F16 weight against the same F16
/// activation the rest of the batched MoE already consumes, and writes the
/// `[b × n_exp]` row-major logits the batched top-k expects.
///
/// BF16→F16 is bit-exact across F16's NORMAL range (BF16 carries 8 significand
/// bits, F16 carries 11). Measured on the published checkpoint, ~0.2% of each
/// router's 524,288 weights fall below that (|w| < 6.1e-5, against a max |w| of
/// ~0.44) and lose precision or flush to zero; those terms are >3 orders of
/// magnitude below the dominant weights, so they sit far under the F32→F16
/// ACTIVATION narrowing the gate_up GEMM already pays. See the two
/// `bf16_to_f16_narrowing_*` tests.
///
/// Returns `None` for a router dtype with no host widening (e.g. a future
/// requantized MQ router). That is not a load failure — decode still runs
/// through `weight_gemv` — it just disables batched prefill via
/// `forward::forward_batch_supported`.
fn upload_router_f16(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    n: usize,
) -> Result<Option<GpuTensor>, String> {
    let Some(vals) = widen_to_f32(qt, data) else {
        return Ok(None);
    };
    if vals.len() != n {
        return Err(format!(
            "maple: router has {} elements, expected {n}",
            vals.len()
        ));
    }
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
        .collect();
    let t = gpu
        .alloc_tensor(&[n], DType::F16)
        .map_err(|e| format!("maple: alloc router_f16: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&t.buf, &bytes)
        .map_err(|e| format!("maple: htod router_f16: {e:?}"))?;
    Ok(Some(t))
}

// ──────────────────────────── Weights ────────────────────────────

/// One MoE expert: fused gate(gate_proj)‖up(up_proj) and down(down_proj).
pub struct MapleExpert {
    pub gate_up: WeightTensor, // [2*moe_inter, hidden]
    pub down: WeightTensor,    // [hidden, moe_inter]
}

/// 256-expert MoE FFN (softmax top-8 + renorm, no bias, no shared expert).
pub struct MapleMoeFfn {
    pub router: WeightTensor, // mlp.gate.weight [n_exp, hidden]
    /// F16 copy of `router`, `[n_exp × hidden]`, for the batched prefill
    /// router GEMM. `None` when the router dtype has no host widening — see
    /// `upload_router_f16` for why the router cannot go through the qt51 path.
    pub router_f16: Option<GpuTensor>,
    pub experts: Vec<MapleExpert>, // per-expert buffers (owned here)
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
}

/// One-entry pointer tables for the dense attention projections, built ONCE at
/// load. Rebuilding them per call would add four allocations per layer per
/// token-chunk.
pub struct AttnPtrTables {
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
}

pub struct MapleLayerWeights {
    pub input_norm: GpuTensor,     // input_layernorm.weight [hidden]
    pub post_attn_norm: GpuTensor, // post_attention_layernorm.weight [hidden]
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    /// Per-head QK-norm gammas, width `head_dim` (NOT hidden). Applied to q/k
    /// BEFORE RoPE.
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// Single-expert device-pointer tables for `wq`/`wk`/`wv`/`wo`, feeding
    /// `batch::dense_qt51_gemm` for batched prefill.
    pub attn_ptr_tables: AttnPtrTables,
    pub moe: MapleMoeFfn,
}

pub struct MapleWeights {
    pub embed: GpuTensor,      // model.word_embeddings.weight (raw bytes)
    pub embed_dtype: DType,    // dtype of `embed` (drives the lookup path)
    pub final_norm: GpuTensor, // model.norm.weight (RMSNorm gamma)
    pub lm_head: WeightTensor, // UNTIED — lm_head.weight
    pub layers: Vec<MapleLayerWeights>,
}

impl MapleWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &MapleConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let head_dim = cfg.head_dim;

        // Embedding. There is no BF16 embedding-lookup kernel, and Maple's
        // converter carries `word_embeddings` as BF16, so widen to F32 on the
        // host at load and hand the F32 path a buffer it can actually read.
        // Costs ~620 MB extra over the BF16 bytes (151936 × 2048); the
        // alternative is a new HIP kernel for one lookup per token.
        let (eqt, embed_bytes) = read_tensor(hfq, EMBED_TENSOR_NAME)?;
        let (embed, embed_dtype) = match eqt {
            16 => {
                let widened: Vec<f32> = embed_bytes
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let t = gpu
                    .upload_f32(&widened, &[widened.len()])
                    .map_err(|e| format!("maple: upload embed (bf16→f32): {e:?}"))?;
                (t, DType::F32)
            }
            2 | 3 => {
                let t = gpu
                    .upload_raw(&embed_bytes, &[embed_bytes.len()])
                    .map_err(|e| format!("maple: upload embed: {e:?}"))?;
                (t, if eqt == 2 { DType::F32 } else { DType::Q8_0 })
            }
            other => {
                return Err(format!(
                    "maple: embed quant_type {other} has no lookup path (expected BF16, F32 or Q8)"
                ))
            }
        };
        // Untied: a separate lm_head tensor, not a second view of the embedding.
        let lm_head = load_wt(hfq, gpu, LM_HEAD_TENSOR_NAME, cfg.vocab_size, hidden)?;
        let final_norm = load_f32(hfq, gpu, FINAL_NORM_TENSOR_NAME, &[hidden])?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_norm = load_f32(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let post_attn_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let wq = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                hidden,
            )?;
            let wk = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wv = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wo = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
            )?;
            // QK-norm gammas are head_dim wide, not hidden.
            let q_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_norm.weight"),
                &[head_dim],
            )?;
            let k_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_norm.weight"),
                &[head_dim],
            )?;

            let attn_ptr_tables = AttnPtrTables {
                wq: crate::batch::upload_single_expert_ptr_table(gpu, &wq)?,
                wk: crate::batch::upload_single_expert_ptr_table(gpu, &wk)?,
                wv: crate::batch::upload_single_expert_ptr_table(gpu, &wv)?,
                wo: crate::batch::upload_single_expert_ptr_table(gpu, &wo)?,
            };

            // Read the router ONCE and materialise both views: the native
            // `WeightTensor` decode drives through `weight_gemv`, and the F16
            // mirror batched prefill needs (the router is BF16, not qt51).
            let router_name = router_tensor_name(l);
            let (router_qt, router_bytes) = read_tensor(hfq, &router_name)?;
            let router = wt_from_raw(gpu, router_qt, &router_bytes, n_exp, hidden)
                .map_err(|e| format!("maple: load_wt {router_name}: {e}"))?;
            let router_f16 = upload_router_f16(gpu, router_qt, &router_bytes, n_exp * hidden)?;
            let mut experts = Vec::with_capacity(n_exp);
            for e in 0..n_exp {
                let (qt_g, g) = read_tensor(hfq, &expert_tensor_name(l, e, ExpertProj::Gate))?;
                let (qt_u, u) = read_tensor(hfq, &expert_tensor_name(l, e, ExpertProj::Up))?;
                // gate_up is byte-fused and tagged with ONE dtype; a mixed pair
                // would mis-read the up half as qt_g. Refuse at load rather than
                // serve silently-wrong inference.
                if qt_g != qt_u {
                    return Err(format!(
                        "maple L{l}E{e}: gate/up dtype mismatch ({qt_g} vs {qt_u}) — cannot byte-fuse gate_up"
                    ));
                }
                let mut gate_up_bytes = g;
                gate_up_bytes.extend_from_slice(&u);
                let gate_up = wt_from_raw(gpu, qt_g, &gate_up_bytes, 2 * moe_inter, hidden)
                    .map_err(|e2| format!("maple: fuse gate_up L{l}E{e}: {e2}"))?;
                let (qt_d, d) = read_tensor(hfq, &expert_tensor_name(l, e, ExpertProj::Down))?;
                let down = wt_from_raw(gpu, qt_d, &d, hidden, moe_inter)
                    .map_err(|e2| format!("maple: down L{l}E{e}: {e2}"))?;
                experts.push(MapleExpert { gate_up, down });
            }
            // Device pointer tables for the indexed-MoE GEMV kernels.
            let gu_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let dn_bytes: Vec<u8> = experts
                .iter()
                .flat_map(|e| (e.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                .collect();
            let expert_gate_up_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("maple: alloc gu_ptrs: {e:?}"))?;
            let expert_down_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("maple: alloc dn_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                .map_err(|e| format!("maple: htod gu_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                .map_err(|e| format!("maple: htod dn_ptrs: {e:?}"))?;

            layers.push(MapleLayerWeights {
                input_norm,
                post_attn_norm,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                attn_ptr_tables,
                moe: MapleMoeFfn {
                    router,
                    router_f16,
                    experts,
                    expert_gate_up_ptrs,
                    expert_down_ptrs,
                },
            });
        }

        Ok(MapleWeights {
            embed,
            embed_dtype,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleWeights {
            embed,
            embed_dtype: _,
            final_norm,
            lm_head,
            layers,
        } = self;
        let _ = gpu.free_tensor(embed);
        let _ = gpu.free_tensor(final_norm);
        lm_head.free_all(gpu);
        for layer in layers {
            layer.free_gpu(gpu);
        }
    }
}

impl MapleExpert {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleExpert { gate_up, down } = self;
        gate_up.free_all(gpu);
        down.free_all(gpu);
    }
}

impl MapleMoeFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleMoeFfn {
            router,
            router_f16,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
        } = self;
        router.free_all(gpu);
        if let Some(t) = router_f16 {
            let _ = gpu.free_tensor(t);
        }
        for e in experts {
            e.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(expert_gate_up_ptrs);
        let _ = gpu.free_tensor(expert_down_ptrs);
    }
}

impl MapleLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleLayerWeights {
            input_norm,
            post_attn_norm,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            attn_ptr_tables,
            moe,
        } = self;
        for t in [input_norm, post_attn_norm, q_norm, k_norm] {
            let _ = gpu.free_tensor(t);
        }
        wq.free_all(gpu);
        wk.free_all(gpu);
        wv.free_all(gpu);
        wo.free_all(gpu);
        let AttnPtrTables {
            wq: pq,
            wk: pk,
            wv: pv,
            wo: po,
        } = attn_ptr_tables;
        for t in [pq, pk, pv, po] {
            let _ = gpu.free_tensor(t);
        }
        moe.free_gpu(gpu);
    }
}

// ──────────────────────────── State ────────────────────────────

/// Batched-prefill flash sub-batch size at full context — the trailing factor
/// of the `flash_partials` allocation (see the Cohere2 note; same kernel).
const FLASH_PREFILL_SUBBATCH: usize = 64;

/// Default KV window. Maple advertises `max_position_embeddings` 131072, but KV
/// is allocated up front; 32k is the same generous-but-not-maximal default the
/// other arches use, and the daemon honours a larger explicit `max_seq`.
const DEFAULT_MAX_SEQ: usize = 32_768;

/// Per-decode GPU scratch + KV cache (one slot per layer — every Maple layer is
/// attention). Buffers are eager-allocated.
pub struct MapleState {
    pub kv: KvCache,
    pub pos_buf: hip_bridge::DeviceBuffer,
    pub max_seq: usize,
    /// How far the KV cache is populated: one past the highest position any
    /// `decode_step` or `forward_batch` has written. `KvCache` carries no
    /// cursor of its own, so this is the record. Maintained by BOTH forward
    /// entry points and zeroed by `reset`.
    pub n_tokens: usize,

    pub h: GpuTensor,      // [hidden] residual stream
    pub normed: GpuTensor, // [hidden] pre-branch RMSNorm output

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]

    // moe scratch
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor,  // [k_top] i32-in-F32
    pub topk_weights: GpuTensor,  // [k_top]
    pub gate_batch: GpuTensor,    // [k_top*moe_inter]
    pub up_batch: GpuTensor,      // [k_top*moe_inter]
    pub act_batch: GpuTensor,     // [k_top*moe_inter] clamped SwiGLU output
    /// `[k_top*hidden]` — per-rank down-projection outputs, UNWEIGHTED.
    /// `DownMode::Expanded` (the decode default) writes one cell per
    /// `(krank, row)` here and `moe_down_combine_k8_batched` folds the k_top
    /// slices into `h` with `topk_weights` applied. Unused by `DownMode::Atomic`.
    pub down_expanded: GpuTensor,

    // head
    pub final_norm_buf: GpuTensor,
    pub logits: GpuTensor,
    pub flash_partials: GpuTensor,

    // ── batched prefill scratch (sized for MAPLE_PREFILL_MAX_B) ──
    //
    // These are PER-CHUNK scratch, not model-constant data, which is why the
    // shared tile-id / slot-index tables live here and not on `MapleWeights`.
    pub max_b: usize,
    pub b_h: GpuTensor,      // [max_b × hidden] batched residual stream
    pub b_normed: GpuTensor, // [max_b × hidden] batched RMSNorm output
    /// F16 mirror of `b_normed`, converted explicitly every layer. MUST be a
    /// distinct allocation from `b_attn_out_f16`: the grouped GEMM's F32 path
    /// caches its conversion on the source pointer, and passing F16 straight
    /// through (as we do) keeps both activations out of that shared scratch
    /// entirely.
    pub b_normed_f16: GpuTensor, // [max_b × hidden] F16
    pub b_q: GpuTensor,      // [max_b × q_dim]
    pub b_k: GpuTensor,      // [max_b × kv_dim]
    pub b_v: GpuTensor,      // [max_b × kv_dim]
    pub b_attn_out: GpuTensor, // [max_b × q_dim]
    pub b_attn_out_f16: GpuTensor, // [max_b × q_dim] F16 — distinct ptr
    /// o_proj destination: the grouped GEMM writes `[dense_m_total(b) × hidden]`,
    /// so the BLOCK_M padding tail is allocated but never read back.
    pub b_proj_out: GpuTensor, // [dense_m_total(max_b) × hidden]
    pub b_positions: GpuTensor, // [max_b] i32-in-f32
    pub b_tile_ids: GpuTensor, // [dense_m_total(max_b)/16] i32, all expert 0
    pub b_slot_index: GpuTensor, // [dense_m_total(max_b)] i32, rewritten per call

    // ── grouped-MoE prefill scratch ──
    //
    // Worst-case padded slot count over the whole `max_b` range. Every LIVE
    // expert can waste up to BLOCK_M-1 pad slots, so this is strictly larger
    // than `max_b * k_top`; sizing on the raw slot count instead would let the
    // scatter write past the end of `b_sorted_slot` on a spread-out routing.
    pub moe_m_total_max: usize,
    pub b_router_logits: GpuTensor,  // [max_b × n_exp]
    pub b_router_probs: GpuTensor,   // [max_b × n_exp] softmax output (separate: no aliasing)
    pub b_topk_indices: GpuTensor,   // [max_b × k_top] i32-in-F32
    pub b_topk_weights: GpuTensor,   // [max_b × k_top]
    pub b_expert_counts: GpuTensor,  // [n_exp] i32-in-F32, padded counts
    pub b_expert_offsets: GpuTensor, // [n_exp + 1] i32-in-F32, exclusive scan
    pub b_sorted_slot: GpuTensor,    // [moe_m_total_max] i32-in-F32, -1 = padding
    pub b_expert_tiles: GpuTensor,   // [moe_m_total_max / 16] i32-in-F32, -1 = dead tile
    pub b_inverse_perm: GpuTensor,   // [max_b × k_top] i32-in-F32
    pub b_y_gate_up: GpuTensor,      // [moe_m_total_max × 2*moe_inter]
    pub b_y_down: GpuTensor,         // [moe_m_total_max × hidden]
    pub b_act: GpuTensor,            // [max_b × k_top × moe_inter] clamped SwiGLU
    /// F16 mirror of `b_act`, converted explicitly every layer for the same
    /// reason `b_normed_f16` exists: the grouped GEMM's F32 arm caches its
    /// conversion on the SOURCE POINTER, and `b_act` is one buffer refilled
    /// with new contents 24 times per prefill. Handing it the F32 tensor
    /// would run the down GEMM of every layer on layer 0's activations —
    /// silently. This buffer is a distinct allocation for that reason.
    pub b_act_f16: GpuTensor, // [max_b × k_top × moe_inter] F16
}

impl MapleState {
    pub fn new(gpu: &mut Gpu, cfg: &MapleConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(DEFAULT_MAX_SEQ);
        Self::new_with_max_seq(gpu, cfg, max_seq, KvMode::Q8)
    }

    /// `kv_mode` must already be resolved through `MAPLE_POLICY` — this is the
    /// allocation site, not the policy site. Only `Q8` and `Bf16` are
    /// serviceable; anything else is rejected rather than silently downgraded,
    /// because the other tiers have no sliding-window attention kernel and
    /// Maple's 3:1 sliding layers would then attend the full context.
    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &MapleConfig,
        max_seq: usize,
        kv_mode: KvMode,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k = cfg.num_experts_per_tok;

        // The FWHT sign LUT is still required: the shared MoE helpers reference
        // it even though MQ2G256LloydU itself never rotates.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("maple: ensure_mq_signs: {e:?}"))?;

        let kv = match kv_mode {
            KvMode::Q8 => KvCache::new_gpu_q8(
                gpu,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                max_seq,
            ),
            KvMode::Bf16 => KvCache::new_gpu_bf16(
                gpu,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                max_seq,
            ),
            // Unreachable through the carrier: MAPLE_POLICY accepts only
            // {Q8, Bf16} and `resolve` falls back to the site default for
            // everything else. Reject loudly rather than serve a tier whose
            // windowed kernels do not exist.
            other => {
                return Err(format!(
                    "maple: KV mode {other:?} has no sliding-window attention kernel; \
                     arch 15 supports q8 and bf16 only"
                ))
            }
        }
        .map_err(|e| format!("maple: kv cache: {e:?}"))?;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("maple: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("maple: alloc {label}: {e:?}"))
        };
        let alloc_f16 = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F16)
                .map_err(|e| format!("maple: alloc {label}: {e:?}"))
        };
        // i32 payload in an F32-typed tensor — the same 4-byte-per-element
        // convention `topk_indices` and `b_positions` already use.
        let alloc_i32 = |g: &mut Gpu, v: &[i32], label: &str| -> Result<GpuTensor, String> {
            let t = g
                .alloc_tensor(&[v.len()], DType::F32)
                .map_err(|e| format!("maple: alloc {label}: {e:?}"))?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
            g.hip
                .memcpy_htod(&t.buf, &bytes)
                .map_err(|e| format!("maple: htod {label}: {e:?}"))?;
            Ok(t)
        };
        let max_b = crate::batch::MAPLE_PREFILL_MAX_B;
        let dm = crate::batch::dense_m_total(max_b);
        let moe_m_total_max = crate::batch::moe_grouped_m_total_bound(max_b * k, n_exp);

        Ok(MapleState {
            kv,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            normed: alloc(gpu, hidden, "normed")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            router_logits: alloc(gpu, n_exp, "router_logits")?,
            topk_indices: alloc(gpu, k, "topk_indices")?,
            topk_weights: alloc(gpu, k, "topk_weights")?,
            gate_batch: alloc(gpu, k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, k * moe_inter, "up_batch")?,
            act_batch: alloc(gpu, k * moe_inter, "act_batch")?,
            down_expanded: alloc(gpu, k * hidden, "down_expanded")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            flash_partials: alloc(
                gpu,
                cfg.num_attention_heads
                    * max_seq.div_ceil(128)
                    * (2 + cfg.head_dim)
                    * FLASH_PREFILL_SUBBATCH,
                "flash_partials",
            )?,
            max_b,
            b_h: alloc(gpu, max_b * hidden, "b_h")?,
            b_normed: alloc(gpu, max_b * hidden, "b_normed")?,
            b_normed_f16: alloc_f16(gpu, max_b * hidden, "b_normed_f16")?,
            b_q: alloc(gpu, max_b * q_dim, "b_q")?,
            b_k: alloc(gpu, max_b * kv_dim, "b_k")?,
            b_v: alloc(gpu, max_b * kv_dim, "b_v")?,
            b_attn_out: alloc(gpu, max_b * q_dim, "b_attn_out")?,
            b_attn_out_f16: alloc_f16(gpu, max_b * q_dim, "b_attn_out_f16")?,
            b_proj_out: alloc(gpu, dm * hidden, "b_proj_out")?,
            b_positions: alloc(gpu, max_b, "b_positions")?,
            b_tile_ids: alloc_i32(gpu, &crate::batch::dense_tile_ids_host(max_b), "b_tile_ids")?,
            b_slot_index: alloc_i32(
                gpu,
                &crate::batch::dense_slot_index_host(max_b),
                "b_slot_index",
            )?,
            moe_m_total_max,
            b_router_logits: alloc(gpu, max_b * n_exp, "b_router_logits")?,
            b_router_probs: alloc(gpu, max_b * n_exp, "b_router_probs")?,
            b_topk_indices: alloc(gpu, max_b * k, "b_topk_indices")?,
            b_topk_weights: alloc(gpu, max_b * k, "b_topk_weights")?,
            b_expert_counts: alloc(gpu, n_exp, "b_expert_counts")?,
            b_expert_offsets: alloc(gpu, n_exp + 1, "b_expert_offsets")?,
            b_sorted_slot: alloc(gpu, moe_m_total_max, "b_sorted_slot")?,
            b_expert_tiles: alloc(
                gpu,
                moe_m_total_max / crate::batch::MOE_GROUPED_BLOCK_M,
                "b_expert_tiles",
            )?,
            b_inverse_perm: alloc(gpu, max_b * k, "b_inverse_perm")?,
            b_y_gate_up: alloc(gpu, moe_m_total_max * 2 * moe_inter, "b_y_gate_up")?,
            b_y_down: alloc(gpu, moe_m_total_max * hidden, "b_y_down")?,
            b_act: alloc(gpu, max_b * k * moe_inter, "b_act")?,
            b_act_f16: alloc_f16(gpu, max_b * k * moe_inter, "b_act_f16")?,
        })
    }

    /// Reset for a fresh conversation: rewind the KV cursor AND zero the KV
    /// buffers. Maple is pure attention with no recurrent state, so the rewind
    /// alone is sufficient for correctness; zeroing makes the reset holistic.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        self.kv
            .clear_gpu(gpu)
            .map_err(|e| format!("maple reset: clear kv: {e:?}"))?;
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MapleState {
            kv,
            pos_buf,
            max_seq: _,
            n_tokens: _,
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            act_batch,
            down_expanded,
            final_norm_buf,
            logits,
            flash_partials,
            max_b: _,
            b_h,
            b_normed,
            b_normed_f16,
            b_q,
            b_k,
            b_v,
            b_attn_out,
            b_attn_out_f16,
            b_proj_out,
            b_positions,
            b_tile_ids,
            b_slot_index,
            moe_m_total_max: _,
            b_router_logits,
            b_router_probs,
            b_topk_indices,
            b_topk_weights,
            b_expert_counts,
            b_expert_offsets,
            b_sorted_slot,
            b_expert_tiles,
            b_inverse_perm,
            b_y_gate_up,
            b_y_down,
            b_act,
            b_act_f16,
        } = self;
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            act_batch,
            down_expanded,
            final_norm_buf,
            logits,
            flash_partials,
            b_h,
            b_normed,
            b_normed_f16,
            b_q,
            b_k,
            b_v,
            b_attn_out,
            b_attn_out_f16,
            b_proj_out,
            b_positions,
            b_tile_ids,
            b_slot_index,
            b_router_logits,
            b_router_probs,
            b_topk_indices,
            b_topk_weights,
            b_expert_counts,
            b_expert_offsets,
            b_sorted_slot,
            b_expert_tiles,
            b_inverse_perm,
            b_y_gate_up,
            b_y_down,
            b_act,
            b_act_f16,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_tensor_names_match_the_published_checkpoint() {
        // 18,432 expert tensors (256 experts x 3 x 24 layers). Getting the
        // naming wrong fails at load, not at build.
        assert_eq!(
            expert_tensor_name(7, 255, ExpertProj::Down),
            "model.layers.7.mlp.experts.255.down_proj.weight"
        );
        assert_eq!(
            expert_tensor_name(0, 0, ExpertProj::Gate),
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        );
        assert_eq!(
            expert_tensor_name(3, 12, ExpertProj::Up),
            "model.layers.3.mlp.experts.12.up_proj.weight"
        );
    }

    #[test]
    fn router_is_mlp_gate_not_an_expert_gate_proj() {
        // `mlp.gate.weight` (router) vs `mlp.experts.N.gate_proj.weight`
        // (expert). Conflating them loads a [256, 2048] router as an expert.
        assert_eq!(router_tensor_name(3), "model.layers.3.mlp.gate.weight");
        assert_ne!(
            router_tensor_name(3),
            expert_tensor_name(3, 0, ExpertProj::Gate)
        );
    }

    #[test]
    fn embedding_is_word_embeddings_not_embed_tokens() {
        assert_eq!(EMBED_TENSOR_NAME, "model.word_embeddings.weight");
        assert_ne!(EMBED_TENSOR_NAME, "model.embed_tokens.weight");
        // Untied: the head is its own tensor, not a second view of the embedding.
        assert_eq!(LM_HEAD_TENSOR_NAME, "lm_head.weight");
        assert_ne!(LM_HEAD_TENSOR_NAME, EMBED_TENSOR_NAME);
    }

    #[test]
    fn the_router_is_not_ternary_so_it_cannot_take_the_qt51_path() {
        // The published checkpoint carries every attention projection and all
        // 18,432 expert tensors as qt=51, but `mlp.gate.weight` as BF16
        // (qt=16). Driving the router through `batch::dense_qt51_gemm` would
        // decode BF16 bytes as MQ2-Lloyd 72-byte groups: in-bounds (the BF16
        // blob is 8x larger than the decoder reads) and therefore SILENT.
        // `widen_to_f32` is the gate — it accepts the float/Q8 dtypes the F16
        // router mirror can be built from and refuses the MQ ones, which is
        // what makes `forward_batch_supported` false for such a checkpoint.
        assert!(widen_to_f32(16, &[0x80, 0x3F]).is_some(), "BF16 router");
        assert!(widen_to_f32(1, &[0x00, 0x3C]).is_some(), "F16 router");
        assert!(widen_to_f32(2, &[0, 0, 0x80, 0x3F]).is_some(), "F32 router");
        assert!(
            widen_to_f32(51, &[0u8; 72]).is_none(),
            "qt51 has no host widening"
        );
        assert!(
            widen_to_f32(19, &[0u8; 72]).is_none(),
            "qt19 has no host widening"
        );
    }

    /// Smallest F16 NORMAL. Below this, F16 goes subnormal and stops carrying
    /// BF16's 8 significand bits; below ~6e-8 it flushes to zero.
    const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;

    #[test]
    fn bf16_to_f16_narrowing_is_exact_over_the_f16_normal_range() {
        // BF16 carries 8 significand bits, F16 carries 11, so every BF16 value
        // in F16's NORMAL range round-trips bit-exactly — the F16 router mirror
        // costs no weight precision there. Exhaustive over the BF16 domain.
        let mut checked = 0usize;
        for bits in 0u16..=0xFFFF {
            let v = bf16_to_f32(bits);
            if !v.is_finite() || v.abs() > 65504.0 || (v != 0.0 && v.abs() < F16_MIN_NORMAL) {
                continue;
            }
            assert_eq!(
                f16_to_f32(f32_to_f16(v)),
                v,
                "bf16 0x{bits:04X} ({v}) did not survive the f16 narrowing"
            );
            checked += 1;
        }
        // 2 signs × 30 in-range exponents (2^-14 .. 2^15) × 128 BF16 mantissas
        // + the zeros. Pinned so a bad filter cannot make the sweep vacuous.
        assert_eq!(checked, 7682, "sweep covered the wrong value set");
    }

    #[test]
    fn the_narrowing_is_lossy_below_the_f16_normal_range() {
        // State the limit rather than overclaim. Measured on the published
        // checkpoint (2026-08-22): per router, |w| spans ~4e-9 to ~0.44 and
        // ~0.2% of the 524,288 weights fall below F16_MIN_NORMAL; the very
        // smallest flush to zero here. Those terms are >3 orders of magnitude
        // below the dominant weights, so their contribution sits far under the
        // F32→F16 ACTIVATION error the batched MoE already accepts — but the
        // narrowing is not unconditionally exact, and this pins that.
        assert_eq!(f32_to_f16(1e-8), 0, "flushes to zero");
        let tiny = bf16_to_f32(0x3800); // 6.1035e-5 — exactly F16_MIN_NORMAL
        assert_eq!(f16_to_f32(f32_to_f16(tiny)), tiny, "the boundary is exact");
    }

    #[test]
    fn bf16_widening_matches_the_reference_bit_pattern() {
        // The converter carries norms/router/embeddings as BF16, so this is the
        // hot path in load_f32. BF16 is the top 16 bits of the f32.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x4049), f32::from_bits(0x40490000));
    }
}
