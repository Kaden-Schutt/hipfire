// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Escha-W2 routed-expert loading for arch-6 (Task 10).
//!
//! An Escha-W2 `.hfq` stores each MoE projection as ONE tensor per layer
//! covering all `num_experts` experts:
//!
//! | tensor | qt | shape | meaning |
//! |---|---|---|---|
//! | `…experts.gate_up_proj.escha_code` | 42 (`ESCHA2T16`) | `[E, ic/16, oc/16, 16*2]` i16 | K=2 trellis code |
//! | `…experts.gate_up_proj.escha_rin_eff` | 2 (F32) | `[E, ic]` | folded input scales |
//! | `…experts.gate_up_proj.escha_rout_eff` | 2 (F32) | `[E, oc]` | folded output scales (carries the prune mask) |
//! | `…experts.down_proj.escha_code` | 43 (`ESCHA3T16`) | `[E, ic/16, oc/16, 16*3]` i16 | K=3 trellis code |
//! | `…experts.down_proj.escha_rin_eff` / `…rout_eff` | 2 (F32) | `[E, ic]` / `[E, oc]` | |
//!
//! ## Orientation — the one thing that must not be got wrong
//!
//! Escha's code tile grid is **IN-MAJOR** (`[in/16, out/16]`) and
//! `Gpu::escha_decode_tiles` writes bare fp16 **row-major `[in_features,
//! out_features]`**. hipfire's expert slots are **OUT-MAJOR** —
//! `experts[X].gate_up` is `[2*moe_intermediate, hidden]` (see
//! `weights.rs:69`) and every hipfire GEMV walks K contiguously along a row.
//! So a transpose happens, and it happens exactly once, folded into the
//! quantise pass (`Gpu::escha_bare_to_q8_0`). A wrong orientation still
//! yields a full-rank, plausible weight matrix, so it is gated by the G4
//! block gate, never by "the output looks sane".
//!
//! hipfire's `gate_up` slot is already FUSED (gate ‖ up), matching escha's
//! single fused `gate_up_proj`, so there is no concat step.
//!
//! ## Why the weights land as `Q8_0`
//!
//! The decoded weight is bare fp16. Storing it as fp16 for the whole model is
//! 60 GiB of experts; `Q8_0` is 32 GiB. The re-quantisation is NOT free — it
//! is the dominant error term in the G4 block gate, measured there against a
//! weight-exact F32 control arm. [`EschaWeightStore::F32`] exists to make that
//! measurement possible; it is a diagnostic, not a way to run the 35B model.

use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

use super::weights::ExpertWeights;

/// How the decoded fp16 expert weight is stored in the expert slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EschaWeightStore {
    /// Production: transpose + Q8_0 re-quantise (1.0625 B/weight).
    Q8_0,
    /// Diagnostic control arm: transpose only, F32 store (4 B/weight), so a
    /// caller can separate "the H128 wiring is wrong" from "Q8_0 costs this
    /// much". Do not use for a whole model.
    F32,
}

/// One layer's Escha-W2 transform tables plus the per-layer decode scratch the
/// batched routed executor needs.
///
/// The `[E, ·]` tables stay resident in full — they are 5.5 MB/layer at A3B
/// shapes (2+1+0.5+2), i.e. 220 MB for the whole model, and keeping them whole
/// is precisely what lets one H128 launch serve all `top_k` experts: slot `s`
/// indexes row `ids[s]`, no gather.
///
/// The scratch is per-layer rather than model-global purely so ownership is
/// simple (it is freed with the layer). At `k=8` / A3B shapes it is ~272 KB
/// per layer, 11 MB for the model.
pub struct EschaMoeTables {
    pub gate_up_rin: GpuTensor,
    pub gate_up_rout: GpuTensor,
    pub down_rin: GpuTensor,
    pub down_rout: GpuTensor,
    pub ids: GpuTensor,
    pub weights: GpuTensor,
    pub xh_gu: GpuTensor,
    pub mid_gu: GpuTensor,
    pub y_gu: GpuTensor,
    pub h: GpuTensor,
    pub xh_dn: GpuTensor,
    pub mid_dn: GpuTensor,
    pub y_dn: GpuTensor,
    pub hidden: usize,
    pub mi: usize,
    pub k: usize,
}

impl EschaMoeTables {
    /// Borrow as the dispatch-crate view. Logic-free adapter.
    pub fn refs(&self) -> hipfire_dispatch::pipeline::escha::EschaRoutedRefs<'_> {
        hipfire_dispatch::pipeline::escha::EschaRoutedRefs {
            gate_up_rin: &self.gate_up_rin,
            gate_up_rout: &self.gate_up_rout,
            down_rin: &self.down_rin,
            down_rout: &self.down_rout,
            ids: &self.ids,
            weights: &self.weights,
            xh_gu: &self.xh_gu,
            mid_gu: &self.mid_gu,
            y_gu: &self.y_gu,
            h: &self.h,
            xh_dn: &self.xh_dn,
            mid_dn: &self.mid_dn,
            y_dn: &self.y_dn,
        }
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.gate_up_rin,
            self.gate_up_rout,
            self.down_rin,
            self.down_rout,
            self.ids,
            self.weights,
            self.xh_gu,
            self.mid_gu,
            self.y_gu,
            self.h,
            self.xh_dn,
            self.mid_dn,
            self.y_dn,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Kill switch for the escha INDEXED (GPU-resident top-K) routed route.
///
/// `HIPFIRE_ESCHA_INDEXED=0` withholds `MoeDtypes::routed_escha_transforms`,
/// which drops `routed_indexable_escha_q8`, which drops `use_gpu_topk`, which
/// sends the layer back down the CPU-top-K route and its host-routed escha
/// executor. Everything stays consistent on the way — including
/// `check_moe_decode_supported`, which sees a non-indexed escha layer and
/// admits it — so this is a genuine A/B of the two routes in ONE build, not a
/// half-disabled state.
///
/// It exists because the two routes are BIT-IDENTICAL (gated by
/// `examples/escha_moe_block_gate.rs`) and differ only in cost, so the
/// performance claim for the indexed route is checkable at any time without a
/// rebuild or a revert. It is also the escape hatch if the indexed route ever
/// needs to be taken out of service in the field.
///
/// Default ON. Read once.
pub fn escha_indexed_route_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_ESCHA_INDEXED").as_deref() != Ok("0")
    })
}

/// `.hfq` tensor name for one of the six escha MoE leaves of a layer, BEFORE
/// candidate expansion. `p` is the bare layer prefix `load.rs` uses
/// (`layers.N`); the caller's `resolve` is what turns that into the
/// checkpoint's actual `model.language_model.layers.N…` name, exactly as
/// every other tensor in this loader is resolved.
pub fn escha_leaf(p: &str, proj: &str, leaf: &str) -> String {
    format!("{p}.mlp.experts.{proj}_proj.escha_{leaf}")
}

/// Candidate-expanding lookup. Mirrors `hfq::load_weight_tensor`'s contract so
/// an escha layer resolves through the same name aliasing as everything else
/// in the checkpoint (`layers.0.…` -> `model.language_model.layers.0.…`).
pub type NameResolver = fn(&str) -> Vec<String>;

/// `tensor_data_vec`, NOT `tensor_data`: on a unified-memory APU the qwen35
/// loader drops the mmap in `prepare()` (mapped pages cannot be evicted while
/// the mapping exists, and they starve `hipMalloc`), after which
/// `tensor_data` returns `None` for every tensor while `find_tensor_info`
/// keeps working. Reading through the mmap here therefore fails only on the
/// full-model path and not in a single-layer probe — exactly the shape of bug
/// that ships. `tensor_data_vec` takes the pread + `FADV_DONTNEED` route the
/// rest of the loader uses.
fn find<'a>(
    hfq: &'a HfqFile,
    name: &str,
    resolve: NameResolver,
) -> Option<(&'a hipfire_runtime::hfq::HfqTensorInfo, Vec<u8>)> {
    resolve(name)
        .into_iter()
        .find_map(|c| hfq.tensor_data_vec(&c))
}

/// True iff this layer's routed experts are Escha-W2 coded. Keyed on the
/// `gate_up` code tensor's presence AND its quant type, so a checkpoint that
/// happened to carry a same-named tensor of another format is rejected by the
/// loader rather than mis-decoded.
pub fn layer_is_escha(hfq: &HfqFile, p: &str, resolve: NameResolver) -> bool {
    resolve(&escha_leaf(p, "gate_up", "code"))
        .into_iter()
        .find_map(|c| hfq.find_tensor_info(&c))
        .is_some_and(|i| i.quant_type == 42 || i.quant_type == 43)
}

fn read_f32_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    want: usize,
    resolve: NameResolver,
) -> HipResult<GpuTensor> {
    let (info, data) = find(hfq, name, resolve)
        .ok_or_else(|| HipError::new(0, &format!("escha: tensor not found: {name}")))?;
    if info.quant_type != 2 {
        return Err(HipError::new(
            0,
            &format!(
                "escha: {name} has quant_type {} (expected 2 = F32)",
                info.quant_type
            ),
        ));
    }
    if data.len() != want * 4 {
        return Err(HipError::new(
            0,
            &format!(
                "escha: {name} is {} bytes, expected {} ({want} f32)",
                data.len(),
                want * 4
            ),
        ));
    }
    gpu.upload_raw(&data, &[want])
}

/// K (trellis order) implied by the on-disk quant type.
fn k_from_quant_type(qt: u8, name: &str) -> HipResult<u32> {
    match qt {
        42 => Ok(2),
        43 => Ok(3),
        other => Err(HipError::new(
            0,
            &format!("escha: {name} has quant_type {other}, expected 42 (K=2) or 43 (K=3)"),
        )),
    }
}

/// Decode one layer's escha experts into hipfire's expert slots, and build the
/// layer's transform tables.
///
/// `expert_ids` selects which experts to materialise, in slot order — the
/// caller's REAP/EP mapping, or simply `0..n_exp`. Passing a short list is how
/// the G4 gate keeps a single-layer probe cheap.
#[allow(clippy::too_many_arguments)]
pub fn load_escha_moe_experts(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    expert_ids: &[usize],
    n_exp: usize,
    hidden: usize,
    mi: usize,
    k: usize,
    store: EschaWeightStore,
    resolve: NameResolver,
) -> HipResult<(Vec<ExpertWeights>, EschaMoeTables)> {
    // gate_up: [ic = hidden, oc = 2*mi]; down: [ic = mi, oc = hidden].
    let gu = (hidden, 2 * mi);
    let dn = (mi, hidden);

    let tables = EschaMoeTables {
        gate_up_rin: read_f32_tensor(
            hfq,
            gpu,
            &escha_leaf(p, "gate_up", "rin_eff"),
            n_exp * gu.0,
            resolve,
        )?,
        gate_up_rout: read_f32_tensor(
            hfq,
            gpu,
            &escha_leaf(p, "gate_up", "rout_eff"),
            n_exp * gu.1,
            resolve,
        )?,
        down_rin: read_f32_tensor(
            hfq,
            gpu,
            &escha_leaf(p, "down", "rin_eff"),
            n_exp * dn.0,
            resolve,
        )?,
        down_rout: read_f32_tensor(
            hfq,
            gpu,
            &escha_leaf(p, "down", "rout_eff"),
            n_exp * dn.1,
            resolve,
        )?,
        // DELIBERATE DTYPE REINTERPRETATION: `ids` holds `k` 32-bit signed
        // INTEGERS — the H128 batched kernels bind it as `const int*`. It is
        // declared `DType::F32` only because `rdna_compute::DType` has no
        // integer variant; F32 is the 4-byte-per-element stand-in, and the
        // allocation size is therefore correct. This mirrors
        // `qwen35::forward`'s `topk_indices`, which does the same thing for
        // the same reason.
        //
        // Consequence: `gpu.download_f32(ids)` returns GARBAGE (int bit
        // patterns reinterpreted as floats), and so would any f32 kernel
        // pointed at it. Read it back with a raw byte download and
        // `i32::from_le_bytes`. Fixing this properly means adding an integer
        // DType to rdna-compute, which is out of scope here.
        ids: gpu.alloc_tensor(&[k], DType::F32)?,
        // `weights` genuinely IS f32 (the f16-rounded combine scores).
        weights: gpu.alloc_tensor(&[k], DType::F32)?,
        xh_gu: gpu.alloc_tensor(&[k * gu.0], DType::F32)?,
        mid_gu: gpu.alloc_tensor(&[k * gu.1], DType::F32)?,
        y_gu: gpu.alloc_tensor(&[k * gu.1], DType::F32)?,
        h: gpu.alloc_tensor(&[k * mi], DType::F32)?,
        xh_dn: gpu.alloc_tensor(&[k * dn.0], DType::F32)?,
        mid_dn: gpu.alloc_tensor(&[k * dn.1], DType::F32)?,
        y_dn: gpu.alloc_tensor(&[k * dn.1], DType::F32)?,
        hidden,
        mi,
        k,
    };

    let mut gate_ups = decode_projection(
        hfq, gpu, p, "gate_up", expert_ids, n_exp, gu, store, resolve,
    )?;
    let mut downs = decode_projection(hfq, gpu, p, "down", expert_ids, n_exp, dn, store, resolve)?;

    let experts = gate_ups
        .drain(..)
        .zip(downs.drain(..))
        .map(|(gate_up, down)| ExpertWeights { gate_up, down })
        .collect();
    Ok((experts, tables))
}

/// Decode every requested expert of ONE projection.
///
/// Staging is reused across experts: one device code buffer, one device bare
/// buffer. At A3B gate_up shapes that is 512 KB + 4 MB held for the whole
/// layer instead of 256 allocations, and the decode never round-trips through
/// the host (`escha_decode_tiles` is the device-resident entry; the `_host`
/// wrapper exists only for the G2 parity gate).
#[allow(clippy::too_many_arguments)]
fn decode_projection(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    proj: &str,
    expert_ids: &[usize],
    n_exp: usize,
    shape: (usize, usize),
    store: EschaWeightStore,
    resolve: NameResolver,
) -> HipResult<Vec<WeightTensor>> {
    let (ic, oc) = shape;
    let name = escha_leaf(p, proj, "code");
    let (info, data) = find(hfq, &name, resolve)
        .ok_or_else(|| HipError::new(0, &format!("escha: tensor not found: {name}")))?;
    let k = k_from_quant_type(info.quant_type, &name)?;

    let words_per_expert = (ic / 16) * (oc / 16) * 16 * k as usize;
    let bytes_per_expert = words_per_expert * 2;
    if data.len() != n_exp * bytes_per_expert {
        return Err(HipError::new(
            0,
            &format!(
                "escha: {name} is {} bytes, expected {} for {n_exp} experts of {ic}x{oc} K={k}",
                data.len(),
                n_exp * bytes_per_expert
            ),
        ));
    }

    // `escha_decode_tiles` validates `code.numel()` in SHORTS, so the staging
    // tensor's logical length must be the i16 count (F16 gives the right
    // 2-bytes-per-element sizing; the payload is trellis code, not floats).
    let code_stage = gpu.alloc_tensor(&[words_per_expert], DType::F16)?;
    let bare = gpu.alloc_tensor(&[ic * oc], DType::F16)?;

    let mut out = Vec::with_capacity(expert_ids.len());
    for &x in expert_ids {
        if x >= n_exp {
            let _ = gpu.free_tensor(code_stage);
            let _ = gpu.free_tensor(bare);
            return Err(HipError::new(
                0,
                &format!("escha: expert id {x} out of range for {n_exp} experts ({name})"),
            ));
        }
        let src = &data[x * bytes_per_expert..(x + 1) * bytes_per_expert];
        gpu.hip.memcpy_htod(&code_stage.buf, src)?;
        gpu.escha_decode_tiles(&code_stage, &bare, ic as u32, oc as u32, k)?;

        // The transpose to hipfire's OUT-major slot lives here, folded into
        // the store. See the module docs.
        let wt = match store {
            EschaWeightStore::Q8_0 => {
                let nbytes = oc * (ic / 32) * 34;
                let buf = gpu.alloc_tensor(&[nbytes], DType::Q8_0)?;
                gpu.escha_bare_to_q8_0(&bare, &buf, ic, oc)?;
                WeightTensor {
                    buf,
                    gpu_dtype: DType::Q8_0,
                    m: oc,
                    k: ic,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                }
            }
            EschaWeightStore::F32 => {
                let buf = gpu.alloc_tensor(&[oc, ic], DType::F32)?;
                gpu.escha_bare_to_f32(&bare, &buf, ic, oc)?;
                WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m: oc,
                    k: ic,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                }
            }
        };
        out.push(wt);
    }
    let _ = gpu.free_tensor(code_stage);
    let _ = gpu.free_tensor(bare);
    Ok(out)
}
