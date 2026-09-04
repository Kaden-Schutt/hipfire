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
//! So on the three DECODING stores a transpose happens, and it happens exactly
//! once, folded into the store pass (`Gpu::escha_bare_to_q8_0` and friends). A
//! wrong orientation still yields a full-rank, plausible weight matrix, so it
//! is gated by the G4 block gate, never by "the output looks sane".
//!
//! [`EschaWeightStore::Native`] keeps escha's in-major grid instead, because
//! it stores the code and never materialises a matrix at all; the fused GEMV
//! addresses the in-major tile grid directly. Same hazard, same gate.
//!
//! hipfire's `gate_up` slot is already FUSED (gate ‖ up), matching escha's
//! single fused `gate_up_proj`, so there is no concat step.
//!
//! ## Why production stores the CODE (Phase 2)
//!
//! Phase 1 decoded to `Q8_0` at load: 1.0625 B/weight, 34.2 GB of routed
//! experts, 37.55 GB resident, and 1.07 GB of routed-expert traffic on every
//! decode token. That put a hard 69 tok/s roofline on a box measured at
//! 209 GB/s — under the 71.8 tok/s the comparable `qwen3.6:35b-a3b-mq4r` SKU
//! already reaches, so Q8_0 could not have won at any efficiency.
//!
//! [`EschaWeightStore::Native`] stores the trellis code verbatim (2.00 bpw for
//! the K=2 gate_up, 3.00 for the K=3 down) and decodes it inside the routed
//! GEMV. It is both smaller AND weight-exact — the `Q8_0` re-quantisation that
//! dominates the G4 block gate's error simply does not happen. The remaining
//! decoding stores are measurement arms: [`EschaWeightStore::Q8_0`] is Phase 1,
//! kept because every published Phase 1 number was measured on it,
//! [`EschaWeightStore::F16`] is the G5 KLD reference, and
//! [`EschaWeightStore::F32`] is a small-layer weight-exact control.

use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

use super::weights::ExpertWeights;
use super::weights::PackedExpertOwners;

/// How the decoded fp16 expert weight is stored in the expert slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EschaWeightStore {
    /// PRODUCTION (Phase 2): the trellis code, verbatim, not decoded at all.
    /// 0.25 B/weight at K=2 and 0.375 at K=3 — a quarter of Q8_0 — and the
    /// routed GEMV decodes it in-register
    /// (`Gpu::escha_gemv_native_moe_k8_indexed_batched`).
    ///
    /// This is the arm that makes the port fast, and it is fast for exactly
    /// one reason: the expanded copy stops existing, so it stops crossing the
    /// bus. At A3B shapes the routed half of a decode token moves 0.294 GB
    /// instead of 1.07 GB, and the whole model 2.23 GB/token instead of 3.01
    /// — a 94 tok/s roofline instead of 69 on this box's measured 209 GB/s.
    /// Q8_0's 69 was BELOW the 71.8 the comparable `qwen3.6:35b-a3b-mq4r` SKU
    /// already achieves, so no amount of efficiency could have closed it.
    ///
    /// It is also weight-EXACT: the fused GEMV consumes the same fp16 values
    /// `escha_decode_tiles` would have produced, so unlike [`Self::Q8_0`] it
    /// carries no re-quantisation error at all. Gated bit-for-bit against the
    /// F16 store and against `escha_ref` by
    /// `rdna-compute/examples/test_escha_native_gemv_gpu_vs_cpu.rs`.
    ///
    /// Requires the indexed (GPU-top-K) route: there is no per-expert
    /// native GEMV, so `HIPFIRE_ESCHA_INDEXED=0` with this store fails loudly
    /// in `GemvFamily::run_auto` (`RotationPlan::EschaH128` has no plain GEMV)
    /// rather than running unrotated. The A/B lever for this store is
    /// `HIPFIRE_ESCHA_EXPERT_STORE=q8_0`, which restores Phase 1 entirely.
    Native,
    /// Phase 1 production, now the A/B arm: transpose + Q8_0 re-quantise
    /// (1.0625 B/weight). Kept because it is what every published Phase 1
    /// number was measured on — including the G4 block gate's Q8_0 arm and the
    /// G5 KLD headline — and because it is the only routed store that also
    /// works through the per-expert host route.
    Q8_0,
    /// Diagnostic control arm: transpose only, F32 store (4 B/weight), so a
    /// caller can separate "the H128 wiring is wrong" from "Q8_0 costs this
    /// much". Do not use for a whole model.
    F32,
    /// Weight-exact arm that DOES fit a whole model: transpose only, F16
    /// store (2 B/weight). The decode already produced fp16, so this holds
    /// bit-identically the same values as [`EschaWeightStore::F32`] in half
    /// the bytes. It is the G5 KLD reference arm.
    ///
    /// It costs **2x production's expert bytes**, and that is now the whole
    /// difference. It did not used to be: while every per-expert buffer was
    /// its own allocation, the HIP allocator's 2 MiB granule rounded Q8_0's
    /// 2.125 MiB gate_up / 1.0625 MiB down up to exactly the 4 MiB / 2 MiB
    /// F16 needed outright, so both arms sat at 60 GiB of experts (measured
    /// 67.9 GB of GTT for the whole Q8_0 model on gfx1151, against a 34.2 GB
    /// logical expert size) and F16 was free. Since the projections are packed
    /// one buffer per (layer, projection) — see [`PackedExpertOwners`] — the
    /// granule is charged 80 times instead of 20,480 and Q8_0 is measured at
    /// 37.6 GB. F16 would be ~32 GB more. It remains the G5 KLD reference arm
    /// and still fits; it is no longer a free upgrade.
    ///
    /// Like F32 this loses the indexed GPU-top-K fast path (admission is
    /// `routed_gate_up == Q8_0 && routed_down == Q8_0`, see
    /// hipfire-dispatch `families/moe.rs`) and runs host-routed instead. That
    /// is slower and numerically identical.
    F16,
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
/// Leaf name for a DENSE escha linear: `{p}.{proj}.escha_{leaf}`.
///
/// Distinct from `escha_leaf`, which is MoE-shaped
/// (`{p}.mlp.experts.{proj}_proj.escha_{leaf}`). Here `proj` is the full path
/// below the layer — `linear_attn.in_proj_qkv`, `self_attn.q_proj`,
/// `mlp.gate_proj` — because the dense export codes every projection in place
/// rather than gathering experts under one name.
pub fn escha_dense_leaf(p: &str, proj: &str, leaf: &str) -> String {
    format!("{p}.{proj}.escha_{leaf}")
}

/// Is this individual dense projection escha-coded?
///
/// Keyed on the CODE tensor's quant type, exactly as `layer_is_escha` is: a
/// projection is escha if and only if its code is qt=42/43. Presence of the
/// name alone is not enough — `escha_config` and friends are optional leaves
/// (§1.4 of the design doc), and the required trio is code/rin/rout.
///
/// Per-projection rather than per-layer because the dense 27B mixes K within
/// a layer: `mlp.gate_proj` is K=2 while `mlp.up_proj` is K=3, so nothing at
/// layer granularity can describe it.
pub fn dense_proj_is_escha(hfq: &HfqFile, p: &str, proj: &str, resolve: NameResolver) -> bool {
    resolve(&escha_dense_leaf(p, proj, "code"))
        .into_iter()
        .find_map(|c| hfq.find_tensor_info(&c))
        .is_some_and(|i| i.quant_type == 42 || i.quant_type == 43)
}

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
///
/// ## Ownership
///
/// The returned [`ExpertWeights`] are **non-owning views** into the returned
/// [`PackedExpertOwners`] pair — one device buffer per projection covering
/// every requested expert. The caller must keep the owners alive for as long
/// as the views are used and free the owners (not the views) exactly once. In
/// the model loader that is `MoeFfnWeights::packed_expert_owners`, whose
/// existing free path (`free_moe_ffn`) already frees per-expert metadata only
/// and returns the two blobs; a direct caller such as the G4 gate must do the
/// same. `Gpu::free_tensor` refuses a borrowed view, so a caller that gets
/// this wrong gets an error rather than a double free — but it also leaks the
/// blob, so it is not a substitute for freeing the owners.
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
) -> HipResult<(Vec<ExpertWeights>, EschaMoeTables, PackedExpertOwners)> {
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

    let (mut gate_ups, gate_up_owner) = decode_projection(
        hfq, gpu, p, "gate_up", expert_ids, n_exp, gu, store, resolve, escha_leaf)?;
    let (mut downs, down_owner) =
        match decode_projection(hfq, gpu, p, "down", expert_ids, n_exp, dn, store, resolve, escha_leaf) {
            Ok(ok) => ok,
            Err(error) => {
                // The gate_up blob is already on the device and its per-expert
                // views are about to be dropped without ever reaching a
                // caller, so nothing else can free it. Return it here or the
                // whole projection (544 MiB at A3B shapes) leaks on every
                // failed layer load.
                let _ = gpu.free_tensor(gate_up_owner);
                return Err(error);
            }
        };

    let experts = gate_ups
        .drain(..)
        .zip(downs.drain(..))
        .map(|(gate_up, down)| ExpertWeights { gate_up, down })
        .collect();
    Ok((
        experts,
        tables,
        PackedExpertOwners {
            gate_up: gate_up_owner,
            down: down_owner,
        },
    ))
}

/// Bytes and elements one expert slot of this projection occupies, for a given
/// store. `(elems_per_slot, dtype)` — `sub_offset` counts in `dtype.size()`
/// units, and `DType::Q8_0::size()` is 1, so the Q8_0 arm's "elements" are
/// bytes. Pure, so the packing arithmetic is checkable without a GPU.
fn slot_extent(store: EschaWeightStore, ic: usize, oc: usize, trellis_k: usize) -> (usize, DType) {
    match store {
        // Native holds the code stream itself: `(ic/16) * (oc/16)` tiles of
        // `16 * trellis_k` int16. `Escha{2,3}T16::size()` is 1 (a byte dtype,
        // like Q8_0), so these "elements" are BYTES.
        EschaWeightStore::Native => (
            (ic / 16) * (oc / 16) * 16 * trellis_k * 2,
            escha_dtype(trellis_k),
        ),
        // Q8_0 rows are `ic/32` blocks of 34 B (32 int8 + one f16 scale).
        EschaWeightStore::Q8_0 => (oc * (ic / 32) * 34, DType::Q8_0),
        EschaWeightStore::F32 => (ic * oc, DType::F32),
        EschaWeightStore::F16 => (ic * oc, DType::F16),
    }
}

/// The `DType` that names a trellis order. This is the dtype the routed expert
/// slot CARRIES under [`EschaWeightStore::Native`], and it is what
/// `MoeResolution::routed_indexable_escha_native` and the batched-prefill
/// admission arm key on — so the layer's route is decided by the same fact the
/// GEMV's bit geometry is decided by, not by two independently-maintained
/// flags.
fn escha_dtype(trellis_k: usize) -> DType {
    if trellis_k == 2 {
        DType::Escha2T16
    } else {
        DType::Escha3T16
    }
}

/// Decode every requested expert of ONE projection into ONE device buffer.
///
/// Staging is reused across experts: one device code buffer, one device bare
/// buffer. At A3B gate_up shapes that is 512 KB + 4 MB held for the whole
/// layer instead of 256 allocations, and the decode never round-trips through
/// the host (`escha_decode_tiles` is the device-resident entry; the `_host`
/// wrapper exists only for the G2 parity gate).
///
/// The returned `WeightTensor`s are non-owning `sub_offset` views into the
/// returned owner buffer — see [`load_escha_moe_experts`] for why, and
/// [`PackedExpertOwners`] for how much it is worth. Each slot's byte offset is
/// `slot * slot_extent(...)`; at A3B shapes that stride is a multiple of 1024,
/// so every view is at least as aligned as an independent allocation would be
/// and no kernel's vector loads are disturbed. The values written are
/// byte-identical to the per-allocation version: `escha_bare_to_*` takes a
/// base pointer and a size, and both are unchanged.
#[allow(clippy::too_many_arguments)]
/// How a projection's leaves are named. `escha_leaf` for the MoE export,
/// `escha_dense_leaf` for the dense one — the two namespaces are disjoint and
/// nothing else about the decode differs, so the namer is a parameter rather
/// than a second copy of this function.
pub type LeafNamer = fn(&str, &str, &str) -> String;

/// One escha-coded DENSE linear, loaded and ready for the forward pass.
///
/// The dense export (Qwen3.8-27B) codes every projection in place rather than
/// gathering experts, so there is no expert table and no routing — just a
/// weight, the two rotation vectors, and the additive bias the end-to-end
/// fine-tune leaves behind.
pub struct EschaDenseLinear {
    /// Decoded weight when `store` is Q8_0/F16, or the verbatim trellis code
    /// when it is Native.
    pub w: WeightTensor,
    /// `escha_rin_eff`, `[ic]` f32 — pre-multiplied into x before the input
    /// H128.
    pub rin: GpuTensor,
    /// `escha_rout_eff`, `[oc]` f32 — applied after the output H128.
    pub rout: GpuTensor,
    /// `bias`, `[oc]`. Present on the 27B, absent on the 35B. Base
    /// Qwen3.8-27B has `attention_bias: false` and no MLP bias, so this is
    /// purely Escha's additive output correction and is applied AFTER the
    /// output transform, per `ref.py::dense_linear`. Applying it before the
    /// H128 would be silently wrong rather than a crash.
    pub bias: Option<GpuTensor>,
    /// Buffer owning `w`'s bytes; freed with the layer.
    pub owner: GpuTensor,
    /// One-element `[0]` slot table, so the BATCHED H128 kernels can serve a
    /// dense linear as the degenerate single-slot case. See
    /// `escha_dense_linear_forward` for why the batched form and not the
    /// single one.
    pub ids0: GpuTensor,
    /// One-element expert-pointer table holding `w.buf`'s device address.
    ///
    /// Present only for `EschaWeightStore::Native`, where `w` IS the trellis
    /// code and there is no decoded weight for a normal GEMV to read. Every
    /// escha GEMV kernel is expert-INDEXED, so rather than write a second
    /// near-identical kernel, a dense linear is served as the degenerate
    /// one-expert case: `expert_ptrs = [&code]`, `ids = [0]`, `slots = 1`.
    /// That reuses the kernel G2 already gates bit-exact against the oracle
    /// instead of forking the trellis inner loop.
    pub ptr0: Option<GpuTensor>,
}

/// Per-projection escha runtime data for ONE dense linear, held alongside the
/// `WeightTensor` rather than replacing it.
///
/// The weight itself stays a `WeightTensor` (dtype `Escha2T16`/`Escha3T16`,
/// buffer = verbatim trellis code) so every existing `layer.wqkv.gpu_dtype`
/// check keeps working. What a trellis weight needs BEYOND that — the two
/// rotation vectors and the one-element pointer table the indexed GEMV wants —
/// lives here.
pub struct EschaProj {
    pub rin: GpuTensor,
    pub rout: GpuTensor,
    pub ptr0: GpuTensor,
}

impl EschaProj {
    /// Run this projection: H128 in -> trellis GEMV -> H128 out.
    ///
    /// Bias is NOT applied here — it is added by the existing per-op bias
    /// path, so there is exactly one place that knows bias ordering.
    ///
    /// `slots` is the token count: 1 for decode, n for batched prefill. The
    /// indexed GEMV serves a dense linear as `slots` copies of expert 0, so
    /// `ids` must be a slots-long run of zeros.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        gpu: &mut Gpu,
        w: &WeightTensor,
        ids: &GpuTensor,
        x: &GpuTensor,
        xh: &GpuTensor,
        mid: &GpuTensor,
        y: &GpuTensor,
        slots: usize,
        // `(expert_offsets, sorted_slot_index)` for the grouped GEMM, or
        // `None` to force the per-slot GEMV. Decode passes `None`.
        grouped: Option<(&GpuTensor, &GpuTensor)>,
    ) -> HipResult<()> {
        let (ic, oc) = (w.k, w.m);
        let tk = match w.gpu_dtype {
            DType::Escha2T16 => 2u32,
            DType::Escha3T16 => 3u32,
            other => {
                return Err(HipError::new(
                    0,
                    &format!("EschaProj::forward: dtype {other:?} is not a trellis code"),
                ))
            }
        };
        let xg = if slots == 1 {
            rdna_compute::EschaXGroup::Broadcast
        } else {
            rdna_compute::EschaXGroup::PerSlot
        };
        gpu.escha_h128_batched("escha_h128_in_batched", x, &self.rin, ids, xh, ic, slots, xg)?;
        match grouped {
            // BATCHED: one group holding every slot. The indexed GEMV re-reads
            // the weight ONCE PER SLOT — correct for MoE, where each slot is a
            // different expert, and 512x the weight traffic for a dense linear
            // where every token shares one weight. The grouped WMMA GEMM reads
            // it once per (layer, batch) instead, which is the same fix that
            // took the 35B's expert path from 4.525 to 2.657 ms/token.
            Some((offsets, iota)) if slots > 1 => {
                gpu.escha_gemm_native_moe_grouped_wmma(
                    &self.ptr0, offsets, iota, xh, mid, oc, ic, slots, 1, tk,
                )?;
            }
            _ => {
                gpu.escha_gemv_native_moe_k8_indexed_batched(
                    &self.ptr0, ids, xh, mid, oc, ic, slots, tk,
                )?;
            }
        }
        gpu.escha_h128_batched(
            "escha_h128_out_batched",
            mid,
            &self.rout,
            ids,
            y,
            oc,
            slots,
            rdna_compute::EschaXGroup::PerSlot,
        )?;
        Ok(())
    }
}

/// Load the escha runtime data for a projection whose weight is a trellis
/// code. `None` when the weight is any other dtype — that is the signal a
/// layer is not escha and should take its ordinary path.
pub fn load_escha_proj(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    proj: &str,
    w: &WeightTensor,
    resolve: NameResolver,
) -> HipResult<Option<EschaProj>> {
    if !matches!(w.gpu_dtype, DType::Escha2T16 | DType::Escha3T16) {
        return Ok(None);
    }
    let (ic, oc) = (w.k, w.m);
    let rin = read_f32_tensor(hfq, gpu, &escha_dense_leaf(p, proj, "rin_eff"), ic, resolve)?;
    let rout = read_f32_tensor(hfq, gpu, &escha_dense_leaf(p, proj, "rout_eff"), oc, resolve)?;
    let addr = w.buf.buf.as_ptr() as u64;
    let ptr0 = gpu.upload_raw(&addr.to_le_bytes(), &[1])?;
    Ok(Some(EschaProj { rin, rout, ptr0 }))
}

/// Load one dense escha linear: `{p}.{proj}` with `[ic, oc]`.
///
/// `proj` is the full path below the layer (`linear_attn.in_proj_qkv`,
/// `mlp.gate_proj`, `self_attn.q_proj`). Reuses `decode_projection` at
/// `n_exp = 1` — a dense linear is exactly the degenerate case of one expert,
/// and duplicating that decode would mean two places to keep bit-exact
/// against the oracle.
#[allow(clippy::too_many_arguments)]
pub fn load_escha_dense_linear(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    proj: &str,
    ic: usize,
    oc: usize,
    store: EschaWeightStore,
    resolve: NameResolver,
) -> HipResult<EschaDenseLinear> {
    let (mut ws, owner) = decode_projection(
        hfq,
        gpu,
        p,
        proj,
        &[0],
        1,
        (ic, oc),
        store,
        resolve,
        escha_dense_leaf,
    )?;
    if ws.len() != 1 {
        return Err(HipError::new(
            0,
            &format!("escha: {p}.{proj} decoded {} slots, expected 1", ws.len()),
        ));
    }
    let w = ws.remove(0);
    let rin = read_f32_tensor(hfq, gpu, &escha_dense_leaf(p, proj, "rin_eff"), ic, resolve)?;
    let rout = read_f32_tensor(hfq, gpu, &escha_dense_leaf(p, proj, "rout_eff"), oc, resolve)?;

    // Bias is OPTIONAL by the leaf contract (§1.4): an export without the
    // end-to-end stage ships none and must still load. So absence is not an
    // error — but a bias of the wrong length is, because it would broadcast
    // or truncate into plausible-looking output.
    let bias_name = format!("{p}.{proj}.bias");
    let bias = match find(hfq, &bias_name, resolve) {
        None => None,
        Some((info, data)) => {
            let elems = match info.quant_type {
                1 => data.len() / 2, // F16
                2 => data.len() / 4, // F32
                other => {
                    return Err(HipError::new(
                        0,
                        &format!("escha: {bias_name} has quant_type {other} (expected F16 or F32)"),
                    ))
                }
            };
            if elems != oc {
                return Err(HipError::new(
                    0,
                    &format!("escha: {bias_name} has {elems} elements, expected oc = {oc}"),
                ));
            }
            Some(read_bias_f32(gpu, info.quant_type, &data, oc)?)
        }
    };
    // `ids` holds a 32-bit signed INTEGER, declared F32 only because
    // rdna_compute::DType has no integer variant — the same deliberate
    // reinterpretation `EschaMoeTables::ids` documents. A dense linear is
    // slot 0 of a one-entry table, so the bytes are four zeros.
    let ids0 = gpu.upload_f32(&[f32::from_bits(0)], &[1])?;
    // Native store keeps the code verbatim, so the GEMV needs its address in
    // a device-side table. Decoded stores (Q8_0/F16/F32) go through the
    // ordinary GEMV and need none.
    let ptr0 = if matches!(store, EschaWeightStore::Native) {
        let addr = w.buf.buf.as_ptr() as u64;
        Some(gpu.upload_raw(&addr.to_le_bytes(), &[1])?)
    } else {
        None
    };
    Ok(EschaDenseLinear {
        w,
        rin,
        rout,
        bias,
        owner,
        ids0,
        ptr0,
    })
}

/// Run one escha-coded dense linear.
///
/// The whole point of the format lives in these three steps, in this order:
///
/// ```text
///   xh  = f16( H128(x * rin) * RS )      escha_h128_in
///   mid = xh @ W                         plain GEMV on the decoded weight
///   y   = H128(mid) * RS * rout + bias   escha_h128_out, then the bias
/// ```
///
/// Skipping either H128 does NOT crash. It produces a full-rank, finite,
/// entirely plausible activation that is simply wrong — which is why this is
/// one function rather than three calls open-coded at ten call sites.
///
/// The bias goes on AFTER the output transform, per `ref.py::dense_linear`.
/// Folding it in before the H128 would be rotated along with the signal and
/// is the single easiest way to get this silently wrong.
///
/// `xh` and `mid` are caller-owned scratch so a layer can reuse one pair
/// across its projections instead of allocating per call.
pub fn escha_dense_linear_forward(
    gpu: &mut Gpu,
    lin: &EschaDenseLinear,
    x: &GpuTensor,
    xh: &GpuTensor,
    mid: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    // BATCHED H128, even for a single vector. The two variants differ in
    // OUTPUT TYPE, not just shape: `escha_h128_in` writes `__half*` (it is the
    // G3 parity form, matching `ref.py`'s f16 return) while
    // `escha_h128_in_batched` writes `float*`. Using the single form with an
    // f32 scratch buffer silently produced a zero activation — the f16 pairs
    // reinterpret as denormal-scale f32 — which then flowed through the GEMV
    // as zeros and left only the bias in the output. Caught by
    // `test_escha_dense_linear_gpu_vs_cpu` before any of this was wired.
    let ic = lin.w.k;
    let oc = lin.w.m;
    gpu.escha_h128_batched(
        "escha_h128_in_batched",
        x,
        &lin.rin,
        &lin.ids0,
        xh,
        ic,
        1,
        rdna_compute::EschaXGroup::Broadcast,
    )?;
    match lin.ptr0.as_ref() {
        // NATIVE: the trellis code decoded inside the GEMV, served as the
        // degenerate one-expert case of the indexed kernel.
        Some(ptr0) => {
            let tk = match lin.w.gpu_dtype {
                DType::Escha2T16 => 2u32,
                DType::Escha3T16 => 3u32,
                other => {
                    return Err(HipError::new(
                        0,
                        &format!("escha dense native: unexpected dtype {other:?}"),
                    ))
                }
            };
            gpu.escha_gemv_native_moe_k8_indexed_batched(
                ptr0, &lin.ids0, xh, mid, oc, ic, 1, tk,
            )?;
        }
        // Plain `weight_gemv`, NOT `weight_gemv_prerotated`. The decoded
        // stores are Q8_0/F16, neither of which wants an FWHT rotation, and
        // `xh` is already H128-rotated — the prerotated path would rotate a
        // rotated activation.
        None => hipfire_runtime::llama::weight_gemv(gpu, &lin.w, xh, mid)?,
    }
    gpu.escha_h128_batched(
        "escha_h128_out_batched",
        mid,
        &lin.rout,
        &lin.ids0,
        y,
        oc,
        1,
        rdna_compute::EschaXGroup::Broadcast,
    )?;
    if let Some(b) = lin.bias.as_ref() {
        gpu.add_inplace_f32(y, b)?;
    }
    Ok(())
}

/// Upload a bias as f32 regardless of whether it was stored F16 or F32.
fn read_bias_f32(gpu: &mut Gpu, qt: u8, data: &[u8], oc: usize) -> HipResult<GpuTensor> {
    let mut v = Vec::with_capacity(oc);
    match qt {
        1 => {
            for c in data.chunks_exact(2) {
                v.push(hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        _ => {
            for c in data.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
    }
    v.truncate(oc);
    gpu.upload_f32(&v, &[oc])
}


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
    namer: LeafNamer,
) -> HipResult<(Vec<WeightTensor>, GpuTensor)> {
    let (ic, oc) = shape;
    let name = namer(p, proj, "code");
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

    // Reject out-of-range ids BEFORE allocating anything, so the error path
    // has nothing to unwind. (Previously this check sat inside the decode loop
    // and had to free the staging buffers by hand.)
    if let Some(&bad) = expert_ids.iter().find(|&&x| x >= n_exp) {
        return Err(HipError::new(
            0,
            &format!("escha: expert id {bad} out of range for {n_exp} experts ({name})"),
        ));
    }

    // ONE buffer for the whole projection. See `PackedExpertOwners`: the 2 MiB
    // allocation granule is charged once here instead of once per expert.
    let (slot_elems, slot_dtype) = slot_extent(store, ic, oc, k as usize);
    let total_elems = slot_elems
        .checked_mul(expert_ids.len())
        .ok_or_else(|| HipError::new(0, &format!("escha: {name} packed size overflow")))?;
    let owner = gpu.alloc_tensor(&[total_elems], slot_dtype)?;

    // ── Native: there is nothing to decode ────────────────────────────────
    // The slot IS the code. No `escha_decode_tiles`, no transpose, no
    // requantise, no staging buffers — the bytes go from the file to their
    // final resting place and the GEMV decodes them per token. This is also
    // why an escha model loads faster on this store than on Q8_0: the decode
    // that used to run 20 480 times at load does not run at all.
    //
    // The code keeps escha's own IN-major `[ic/16, oc/16, 16*k]` tile grid.
    // The out-major transpose the other three stores fold in is not skipped
    // here so much as absorbed: the fused GEMV addresses tiles in the in-major
    // grid directly (see `escha_moe_gemv_native.hip`).
    if store == EschaWeightStore::Native {
        debug_assert_eq!(slot_elems, bytes_per_expert);
        let mut out = Vec::with_capacity(expert_ids.len());
        for (slot, &x) in expert_ids.iter().enumerate() {
            let src = &data[x * bytes_per_expert..(x + 1) * bytes_per_expert];
            let buf = owner.sub_offset(slot * slot_elems, slot_elems);
            if let Err(error) = gpu.hip.memcpy_htod(&buf.buf, src) {
                let _ = gpu.free_tensor(owner);
                return Err(error);
            }
            out.push(WeightTensor {
                buf,
                gpu_dtype: slot_dtype,
                // `m` / `k` stay the LOGICAL matrix shape, exactly as on every
                // other store, because that is what the executor passes to the
                // GEMV. The dtype is what says the bytes are trellis code.
                m: oc,
                k: ic,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            });
        }
        return Ok((out, owner));
    }

    // `escha_decode_tiles` validates `code.numel()` in SHORTS, so the staging
    // tensor's logical length must be the i16 count (F16 gives the right
    // 2-bytes-per-element sizing; the payload is trellis code, not floats).
    let code_stage = match gpu.alloc_tensor(&[words_per_expert], DType::F16) {
        Ok(t) => t,
        Err(error) => {
            let _ = gpu.free_tensor(owner);
            return Err(error);
        }
    };
    let bare = match gpu.alloc_tensor(&[ic * oc], DType::F16) {
        Ok(t) => t,
        Err(error) => {
            let _ = gpu.free_tensor(code_stage);
            let _ = gpu.free_tensor(owner);
            return Err(error);
        }
    };

    let mut out = Vec::with_capacity(expert_ids.len());
    let mut decode = |gpu: &mut Gpu| -> HipResult<()> {
        for (slot, &x) in expert_ids.iter().enumerate() {
            let src = &data[x * bytes_per_expert..(x + 1) * bytes_per_expert];
            gpu.hip.memcpy_htod(&code_stage.buf, src)?;
            gpu.escha_decode_tiles(&code_stage, &bare, ic as u32, oc as u32, k)?;

            // Non-owning window onto this expert's slice of the layer blob.
            // The device pointer this yields is what lands in
            // `expert_{gate_up,down}_ptrs`, so the indexed GEMV addresses the
            // expert exactly as it did when each slot was its own allocation.
            let buf = owner.sub_offset(slot * slot_elems, slot_elems);

            // The transpose to hipfire's OUT-major slot lives here, folded
            // into the store. See the module docs.
            match store {
                EschaWeightStore::Q8_0 => gpu.escha_bare_to_q8_0(&bare, &buf, ic, oc)?,
                EschaWeightStore::F32 => gpu.escha_bare_to_f32(&bare, &buf, ic, oc)?,
                EschaWeightStore::F16 => gpu.escha_bare_to_f16(&bare, &buf, ic, oc)?,
                // Returned above, before any staging buffer was allocated.
                EschaWeightStore::Native => unreachable!("native store returns before decoding"),
            }
            out.push(WeightTensor {
                buf,
                gpu_dtype: slot_dtype,
                m: oc,
                k: ic,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            });
        }
        Ok(())
    };
    let result = decode(gpu);
    let _ = gpu.free_tensor(code_stage);
    let _ = gpu.free_tensor(bare);
    if let Err(error) = result {
        let _ = gpu.free_tensor(owner);
        return Err(error);
    }
    Ok((out, owner))
}

#[cfg(test)]
mod tests {
    use super::escha_dense_leaf;
    use super::escha_leaf;
    use super::slot_extent;
    use super::EschaWeightStore;
    use rdna_compute::DType;

    /// The dense and MoE leaf namers are NOT interchangeable, and using the
    /// wrong one yields a name that simply is not in the file — which reads
    /// as "this projection is not escha" and silently takes the plain-weight
    /// path. Pin both against names copied out of the real checkpoints.
    #[test]
    fn dense_and_moe_leaf_names_do_not_collide() {
        let p = "model.language_model.layers.0";
        // 27B dense export, verbatim from model.safetensors.index.json.
        assert_eq!(
            escha_dense_leaf(p, "linear_attn.in_proj_qkv", "code"),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.escha_code"
        );
        assert_eq!(
            escha_dense_leaf(p, "mlp.gate_proj", "rin"),
            "model.language_model.layers.0.mlp.gate_proj.escha_rin"
        );
        assert_eq!(
            escha_dense_leaf(p, "self_attn.q_proj", "rout"),
            "model.language_model.layers.0.self_attn.q_proj.escha_rout"
        );
        // 35B MoE export gathers experts under one name instead.
        assert_eq!(
            escha_leaf(p, "gate_up", "code"),
            "model.language_model.layers.0.mlp.experts.gate_up_proj.escha_code"
        );
        assert_ne!(
            escha_dense_leaf(p, "mlp.gate_proj", "code"),
            escha_leaf(p, "gate", "code")
        );
    }

    /// The packing arithmetic, at the real A3B shapes, against the sizes the
    /// allocator-granularity diagnosis is built on. A slot stride that is not
    /// a multiple of the Q8_0 block (34 B) or that disagrees with
    /// `escha_bare_to_q8_0`'s own `oc * (ic/32) * 34` would put every expert
    /// after slot 0 at a wrong offset — plausible, finite, wrong weights.
    #[test]
    fn q8_0_slot_extent_matches_the_a3b_projection_sizes() {
        // gate_up: ic = hidden = 2048, oc = 2*mi = 1024.
        let (gu, gu_dtype) = slot_extent(EschaWeightStore::Q8_0, 2048, 1024, 2);
        assert_eq!(gu_dtype, DType::Q8_0);
        assert_eq!(gu, 2_228_224, "gate_up slot is 2.125 MiB");
        // down: ic = mi = 512, oc = hidden = 2048.
        let (dn, _) = slot_extent(EschaWeightStore::Q8_0, 512, 2048, 3);
        assert_eq!(dn, 1_114_112, "down slot is 1.0625 MiB");
        // 256 experts x 40 layers x both projections = the 34.2 GB of real
        // weight bytes the 67.9 GB of granules was hiding.
        assert_eq!((gu + dn) * 256 * 40, 34_225_520_640);
    }

    /// The Native (Phase 2) store's slot arithmetic, at the real A3B shapes,
    /// against the code sizes on disk. These are the numbers the whole Phase-2
    /// memory claim rests on, so they are asserted rather than asserted-about:
    /// a wrong stride would place every expert after slot 0 at a wrong offset
    /// and — because trellis code decodes to *something* from any bit pattern
    /// — produce finite, plausible, wrong weights rather than a fault.
    #[test]
    fn native_slot_extent_matches_the_a3b_code_sizes() {
        // gate_up: ic = hidden = 2048, oc = 2*mi = 1024, K=2.
        let (gu, gu_dtype) = slot_extent(EschaWeightStore::Native, 2048, 1024, 2);
        assert_eq!(gu_dtype, DType::Escha2T16);
        assert_eq!(gu_dtype.size(), 1, "escha code offsets are byte offsets");
        assert_eq!(gu, 524_288, "gate_up code is 512 KiB (2.00 bpw)");
        // down: ic = mi = 512, oc = hidden = 2048, K=3.
        let (dn, dn_dtype) = slot_extent(EschaWeightStore::Native, 512, 2048, 3);
        assert_eq!(dn_dtype, DType::Escha3T16);
        assert_eq!(dn, 393_216, "down code is 384 KiB (3.00 bpw)");

        // Exactly 2.00 / 3.00 bits per weight — the format's own figures, so
        // this also catches a tile-count or word-count slip.
        assert_eq!(gu * 8, 2048 * 1024 * 2);
        assert_eq!(dn * 8, 512 * 2048 * 3);

        // 256 experts x 40 layers x both projections. The Q8_0 store's 34.2 GB
        // of the same weights is 3.73x this.
        assert_eq!((gu + dn) * 256 * 40, 9_395_240_960);
        let (q8_gu, _) = slot_extent(EschaWeightStore::Q8_0, 2048, 1024, 2);
        let (q8_dn, _) = slot_extent(EschaWeightStore::Q8_0, 512, 2048, 3);
        assert!((q8_gu + q8_dn) > 3 * (gu + dn));
    }

    /// `sub_offset` counts in `dtype.size()` units. Q8_0 is a byte dtype, so
    /// the Q8_0 stride is a byte stride while F16/F32 strides are element
    /// counts. Getting that wrong scales every offset by 2 or 4.
    #[test]
    fn slot_extent_is_in_dtype_units_not_bytes() {
        let (f32_elems, f32_dtype) = slot_extent(EschaWeightStore::F32, 2048, 1024, 2);
        assert_eq!(f32_dtype, DType::F32);
        assert_eq!(f32_elems, 2048 * 1024);
        assert_eq!(f32_elems * DType::F32.size(), 8 * 1024 * 1024);

        let (f16_elems, f16_dtype) = slot_extent(EschaWeightStore::F16, 2048, 1024, 2);
        assert_eq!(f16_dtype, DType::F16);
        assert_eq!(f16_elems, 2048 * 1024);
        assert_eq!(f16_elems * DType::F16.size(), 4 * 1024 * 1024);

        let (q8_elems, q8_dtype) = slot_extent(EschaWeightStore::Q8_0, 2048, 1024, 2);
        assert_eq!(q8_dtype.size(), 1, "Q8_0 offsets are byte offsets");
        assert_eq!(q8_elems * q8_dtype.size(), 2_228_224);
    }

    /// Every A3B slot stride is a multiple of 1024 B, so no expert view is
    /// less aligned than the 2 MiB-granule allocation it replaces and the
    /// kernels' vector loads are undisturbed.
    #[test]
    fn a3b_slot_strides_are_widely_aligned() {
        for (ic, oc) in [(2048usize, 1024usize), (512, 2048)] {
            let (elems, dtype) = slot_extent(EschaWeightStore::Q8_0, ic, oc, 2);
            assert_eq!(elems * dtype.size() % 1024, 0, "{ic}x{oc} stride alignment");
        }
    }
}
