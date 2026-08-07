// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for DeepSeek V4 Flash (`arch_id = 9`).
//!
//! DeepSeek V4 diverges from the Qwen3.5 / LLaMA paths in several places —
//! Hyper-Connections, compressed-KV indexer, tail-only RoPE,
//! Q/O-LoRA, raw SWA cache, FP4 experts — but the bring-up triple
//! (`config_from_hfq` / `load_weights` / `new_state`) follows the
//! same Architecture-trait shape as the other arch crates.
//!
//! At scaffold stage (this commit) `load_weights` and forward are
//! stubbed; only `config_from_hfq` and `new_state` are wired through
//! so the workspace builds and the metadata parser is exercised by
//! the tests.

use crate::backend::Mq2rBackend;
use crate::deepseek4::{
    DeepseekV4Config, DeepseekV4DenseWeights, DeepseekV4HarmonicReplicaWeights,
    DeepseekV4HeterogeneousWeights, DeepseekV4LayerWeights, DeepseekV4RoutedWeights,
    DeepseekV4State, DeepseekV4Weights, DsparkConfig, DsparkWeights,
};
use crate::harmonic::{HarmonicExpertResidencyPlan, HarmonicOwner, HARMONIC_LAYER_COUNT};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DeepseekV4HeterogeneousProjection {
    pub dense_record_count: usize,
    pub dense_allocation_count: usize,
    pub dense_bytes: usize,
    pub f16_expansion_bytes: usize,
    pub routed_record_count: usize,
    pub routed_allocation_count: usize,
    pub routed_bytes: usize,
    pub pointer_table_bytes: usize,
    pub host_only_record_count: usize,
}

/// CPU-only proof that a harmonic gfx1100 expert-residency plan names exact
/// MQ2-Lloyd payloads from the frozen artifact and fits its declared budget.
/// This receipt performs no allocation and does not enable hybrid execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV4HarmonicResidencyProjection {
    pub plan_identity: u64,
    pub slot_count: usize,
    pub exact_bytes: u64,
    pub budget_bytes: u64,
    pub headroom_bytes: u64,
    pub layer_slot_counts: [u16; HARMONIC_LAYER_COUNT as usize],
}

/// Deterministic G2 failure points used to certify transactional cleanup.
/// This is a typed test seam, never an environment-controlled product mode.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepseekV4HeterogeneousFault {
    AfterDenseWeights,
    AfterRoutedLayer(usize),
    AfterOwnershipAudit,
    AfterState,
    AfterScratch,
}

/// Load-scoped owner for partially populated DS4 weights. `GpuTensor` is an
/// explicit resource handle rather than a `Drop` type, so every early `?`
/// during the 34k-record upload must walk the tensors already installed.
///
/// Raw device pointers avoid holding an exclusive Rust borrow for the entire
/// upload. They are valid for this guard's strictly nested function lifetime;
/// the heterogeneous constructor additionally proves that the two pointers
/// refer to distinct exact devices before this guard is armed.
struct DeepseekV4WeightStaging {
    dense: Option<DeepseekV4Weights>,
    routed: Option<DeepseekV4RoutedWeights>,
    dense_gpu: *mut Gpu,
    routed_gpu: Option<*mut Gpu>,
}

impl DeepseekV4WeightStaging {
    fn new(
        dense: DeepseekV4Weights,
        routed: Option<DeepseekV4RoutedWeights>,
        dense_gpu: &mut Gpu,
        routed_gpu: Option<&mut Gpu>,
    ) -> Self {
        debug_assert_eq!(routed.is_some(), routed_gpu.is_some());
        Self {
            dense: Some(dense),
            routed,
            dense_gpu,
            routed_gpu: routed_gpu.map(|gpu| gpu as *mut Gpu),
        }
    }

    fn into_single(mut self) -> DeepseekV4Weights {
        assert!(
            self.routed.is_none(),
            "single-owner DS4 retained routed split"
        );
        self.dense.take().expect("DS4 staging disarmed twice")
    }

    fn into_heterogeneous(mut self) -> Result<DeepseekV4HeterogeneousWeights, String> {
        DeepseekV4DenseWeights::validate_loaded(
            self.dense.as_ref().expect("DS4 staging disarmed twice"),
        )?;
        let dense = self.dense.take().expect("DS4 staging disarmed twice");
        let routed = self
            .routed
            .take()
            .ok_or_else(|| "deepseek4: heterogeneous staging lost routed owner".to_string())?;
        Ok(DeepseekV4HeterogeneousWeights {
            dense: DeepseekV4DenseWeights::from_loaded(dense),
            routed,
        })
    }

    fn into_dense(mut self) -> Result<DeepseekV4DenseWeights, String> {
        if self.routed.is_some() {
            return Err("deepseek4: dense-only staging retained a routed owner".to_owned());
        }
        let dense = self.dense.take().expect("DS4 staging disarmed twice");
        DeepseekV4DenseWeights::validate_loaded(&dense)?;
        Ok(DeepseekV4DenseWeights::from_loaded(dense))
    }

    fn routed_layer_mut(&mut self, index: usize) -> Option<&mut DeepseekV4LayerWeights> {
        self.routed.as_mut()?.layer_mut(index)
    }
}

impl std::ops::Deref for DeepseekV4WeightStaging {
    type Target = DeepseekV4Weights;

    fn deref(&self) -> &Self::Target {
        self.dense.as_ref().expect("DS4 staging disarmed")
    }
}

impl std::ops::DerefMut for DeepseekV4WeightStaging {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dense.as_mut().expect("DS4 staging disarmed")
    }
}

impl Drop for DeepseekV4WeightStaging {
    fn drop(&mut self) {
        let dense = self.dense.take();
        let routed = self.routed.take();
        if dense.is_none() && routed.is_none() {
            return;
        }
        eprintln!("deepseek4: weight load failed; reclaiming partially uploaded tensors");
        // SAFETY: both pointers originate from live `&mut Gpu` parameters of
        // the enclosing load call. This guard is created after exact-device
        // admission and is dropped before those parameters leave scope.
        unsafe {
            let dense_gpu = &mut *self.dense_gpu;
            if let (Some(routed), Some(routed_gpu)) = (routed, self.routed_gpu) {
                let errors = routed.free_gpu_now(&mut *routed_gpu);
                if !errors.is_empty() {
                    eprintln!(
                        "deepseek4: routed failure cleanup reported {} error(s): {}",
                        errors.len(),
                        errors.join("; ")
                    );
                }
            }
            if let Some(dense) = dense {
                dense.free_gpu(dense_gpu);
            }
        }
    }
}

/// Owner-correct cleanup for an independently loaded gfx1151 expert worker.
/// The worker never constructs a dense weight object or another `Gpu`.
struct DeepseekV4RoutedWeightStaging {
    routed: Option<DeepseekV4RoutedWeights>,
    gpu: *mut Gpu,
}

impl DeepseekV4RoutedWeightStaging {
    fn new(routed: DeepseekV4RoutedWeights, gpu: &mut Gpu) -> Self {
        Self {
            routed: Some(routed),
            gpu,
        }
    }

    fn routed_mut(&mut self) -> &mut DeepseekV4RoutedWeights {
        self.routed.as_mut().expect("DS4 routed staging disarmed")
    }

    fn publish(mut self) -> DeepseekV4RoutedWeights {
        self.routed
            .take()
            .expect("DS4 routed staging disarmed twice")
    }
}

impl Drop for DeepseekV4RoutedWeightStaging {
    fn drop(&mut self) {
        let Some(routed) = self.routed.take() else {
            return;
        };
        eprintln!("deepseek4: routed-worker load failed; reclaiming expert tensors");
        // SAFETY: `gpu` is the live, uniquely borrowed gfx1151 owner passed to
        // the enclosing loader. The staging guard cannot outlive that call.
        let errors = unsafe { routed.free_gpu_now(&mut *self.gpu) };
        if !errors.is_empty() {
            eprintln!(
                "deepseek4: routed-worker cleanup reported {} error(s): {}",
                errors.len(),
                errors.join("; ")
            );
        }
    }
}
use hipfire_reap::hook::ReapArchHook;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::hfq_parallel::{read_hfq_jobs_ordered, HfqReadJob};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::{bf16_bytes_to_f16, bf16_to_f32};
use rdna_compute::{DType, Gpu};

/// Preserve the HFQ wire dtype when uploading dense DeepSeek projections.
///
/// `Raw` remains the compatibility fallback for the historical MQ4 container,
/// but formats with distinct decode kernels must never collapse into it:
/// doing so makes dispatch interpret their bytes as MQ4G256.
fn dense_hfq_dtype(quant_type: u8) -> Option<DType> {
    match quant_type {
        1 => Some(DType::F16),
        3 => Some(DType::Q8_0),
        13 => Some(DType::MQ4G256),
        24 => Some(DType::MFP4G32),
        33 => Some(DType::MFP4G32P),
        34 => Some(DType::MFP4G32E8),
        35 => Some(DType::MFP4G32E8SOA),
        _ => None,
    }
}

/// Type marker for DeepSeek V4 Flash. `arch_id = 9` — next free slot
/// after `8 = Qwen2-VL (dots.ocr)` reserved in `docs/architecture-ids.md`.
/// The marker is zero-sized; trait dispatch uses the type, not a value.
pub struct DeepseekV4;

impl DeepseekV4 {
    /// Phase 1.5 walk: verify every expected DeepSeek V4 tensor is present in
    /// the HFQ index. No GPU upload. Returns a populated `Weights` with
    /// `_scaffold: ()` per layer; the real `WeightTensor` handles get
    /// filled in as Phases 2-5 wire the kernels.
    ///
    /// Catches missing-tensor / naming-mismatch problems before forward
    /// triggers them. Per-layer tensor inventory derived from the DeepSeek V4
    /// safetensors index (see Phase 1 commit 8ccfa42).
    /// Upload one global HFQ tensor verbatim (raw bytes) to GPU.
    /// Used for embed/quantized-weights where the on-disk quant format
    /// matches the format the kernels expect to consume.
    fn upload_global_raw(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        // pread + fadvise(DONTNEED) keeps page-cache footprint bounded
        // under unified memory (Strix Halo etc.). mmap-based `tensor_data`
        // would hold the read pages until the kernel reclaims them, which
        // can't keep up with the ~80 GB of subsequent routed-expert
        // hipMallocs on the 88 GB deepseek4-q8-mtp build — OOM at layer 42.
        let (info, bytes) = hfq
            .tensor_data_pread(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        gpu.upload_raw(&bytes, &shape)
            .map_err(|e| format!("deepseek4: upload '{name}' failed: {e:?}"))
    }

    /// Upload a weight whose HFQ format is one of:
    ///   - F16 (quant_type=1): keep native F16 bytes and route through the
    ///     F16 decode/prefill kernels with plain (non-FWHT) input.
    ///   - Q8F16 (quant_type=3): upload raw bytes, set GpuTensor.dtype =
    ///     Q8_0. Forward routes to `gemv_q8_0` with plain input.
    ///   - MQ4/MFP4-family formats: preserve their concrete dtype so dispatch
    ///     selects the matching prerotated decoder. Unknown historical wire
    ///     types retain the old `Raw` compatibility fallback.
    ///
    /// Distinct from `upload_global_raw` because the HC kernels
    /// (hc_compute_control, hc_apply_alpha) expect their weights as
    /// `__half*` — those tensors must use `upload_global_raw`, NOT this
    /// helper, so the GPU pointer is a raw F16 byte buffer.
    fn upload_quant_or_f16(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        // pread-based read (see upload_global_raw note); avoids the
        // mmap-backed page-cache pressure that OOMs on UMA with the
        // 88 GB deepseek4-q8-mtp build.
        let (info, bytes) = hfq
            .tensor_data_pread(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        if info.quant_type == 1 {
            // F16 source: KEEP F16 on device. Forward routes F16 weights
            // through `gemm_f16_x_f16_wmma` in the batched path and a
            // thin convert+WMMA wrapper in the single-decode path — both
            // ~10–25× faster than the old F32-decoded scalar GEMM.
            let n: usize = shape.iter().product();
            if bytes.len() != n * 2 {
                return Err(format!(
                    "deepseek4: '{name}' marked F16 but byte size {} != 2 × {n}",
                    bytes.len()
                ));
            }
            let mut t = gpu
                .upload_raw(&bytes, &shape)
                .map_err(|e| format!("deepseek4: upload f16-native '{name}' failed: {e:?}"))?;
            t.dtype = DType::F16;
            return Ok(t);
        }
        let mut t = gpu
            .upload_raw(&bytes, &shape)
            .map_err(|e| format!("deepseek4: upload '{name}' failed: {e:?}"))?;
        if let Some(dtype) = dense_hfq_dtype(info.quant_type) {
            t.dtype = dtype;
        }
        Ok(t)
    }

    /// Upload an F16-on-disk HFQ tensor as F16 bytes on GPU (no
    /// conversion). Marks `dtype = F16`. Used for the WMMA GEMM path
    /// that consumes F16 weights directly. Errors if the source isn't
    /// F16 (quant_type != 1).
    fn upload_quant_as_f16_native(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = hfq
            .tensor_data_pread(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        if info.quant_type != 1 {
            return Err(format!(
                "deepseek4: '{name}' not F16 (quant_type={}); cannot upload as F16 native",
                info.quant_type
            ));
        }
        let n: usize = shape.iter().product();
        if bytes.len() != n * 2 {
            return Err(format!(
                "deepseek4: '{name}' marked F16 but byte size {} != 2 × {n}",
                bytes.len()
            ));
        }
        let mut t = gpu
            .upload_raw(&bytes, &shape)
            .map_err(|e| format!("deepseek4: upload f16-native '{name}' failed: {e:?}"))?;
        t.dtype = rdna_compute::DType::F16;
        Ok(t)
    }

    /// Upload routed-expert blobs for one "layer-shaped" block (a normal
    /// transformer layer or the MTP layer). Mirrors the original
    /// inline logic but is parameterized on `prefix` so the same code
    /// runs for `layers.{L}` and `mtp.0`. Writes `expert_w2_blob/_ptrs/
    /// _stride` and `expert_gate_up_blob/_ptrs/_stride` on the layer.
    ///
    /// `shard = Some((cfg, rank))` enables **EP shard-aware loading**: every
    /// expert is `pread` from the file (for stride validation) but ONLY the
    /// rank-owned experts are uploaded into a compact packed blob, so an
    /// 81 GB model fits across N×32 GB cards. The per-expert pointer table
    /// then maps owned `e` → its compact-blob slot; non-owned `e` → a shared
    /// ZEROED gate_up dummy (SwiGLU(0,0)=0 ⇒ 0 routed contribution, even for
    /// the MQ2/MQ3-Lloyd codebook path: an all-zero buffer dequantizes to 0).
    /// The non-owned w2 (down) ptr reuses the compact base — its rotate input
    /// is 0 regardless, so the down weights read don't matter. `shard = None`
    /// uploads all experts (single-GPU, byte-identical to the original).
    fn upload_layer_routed_experts_parallel(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        prefix: &str,
        n_exp: usize,
        layer: &mut DeepseekV4LayerWeights,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
        keep: Option<&[u32]>,
    ) -> Result<(), String> {
        if keep.is_some() && shard.is_some() {
            return Err("deepseek4: REAP keep-map + EP sharding are mutually exclusive".into());
        }
        if let Some(k) = keep {
            if k.len() != n_exp {
                return Err(format!(
                    "deepseek4: {prefix} keep slice len {} != n_exp {n_exp}",
                    k.len()
                ));
            }
        }
        let src = |slot: usize| -> usize { keep.map(|k| k[slot] as usize).unwrap_or(slot) };
        let owns = |e: usize| {
            shard
                .map(|(s, rank)| s.owns_expert(rank, e))
                .unwrap_or(true)
        };
        let mut local_of_global = vec![usize::MAX; n_exp];
        let mut n_owned = 0usize;
        for e in 0..n_exp {
            if owns(e) {
                local_of_global[e] = n_owned;
                n_owned += 1;
            }
        }
        if n_owned == 0 {
            return Err(format!("deepseek4: {prefix} shard rank owns no experts"));
        }

        let w2_name0 = format!("{prefix}.ffn.experts.{}.w2.weight", src(0));
        let w2_info0 = hfq
            .find_tensor_info(&w2_name0)
            .ok_or_else(|| format!("deepseek4: missing {w2_name0}"))?;
        let w2_stride = w2_info0.data_size;
        let w2_shape: Vec<usize> = w2_info0.shape.iter().map(|&s| s as usize).collect();
        let mut w2_names = Vec::with_capacity(n_owned);
        for e in 0..n_exp {
            if !owns(e) {
                continue;
            }
            let name = format!("{prefix}.ffn.experts.{}.w2.weight", src(e));
            let info = hfq
                .find_tensor_info(&name)
                .ok_or_else(|| format!("deepseek4: missing {name}"))?;
            if info.data_size != w2_stride {
                return Err(format!(
                    "deepseek4: {name} size {} != stride {w2_stride}",
                    info.data_size
                ));
            }
            w2_names.push(name);
        }

        let w1_name0 = format!("{prefix}.ffn.experts.{}.w1.weight", src(0));
        let w3_name0 = format!("{prefix}.ffn.experts.{}.w3.weight", src(0));
        let w1_stride = hfq
            .find_tensor_info(&w1_name0)
            .ok_or_else(|| format!("deepseek4: missing {w1_name0}"))?
            .data_size;
        let w3_stride = hfq
            .find_tensor_info(&w3_name0)
            .ok_or_else(|| format!("deepseek4: missing {w3_name0}"))?
            .data_size;
        if w1_stride != w3_stride {
            return Err(format!(
                "deepseek4: {prefix} w1/w3 stride mismatch: w1={w1_stride} w3={w3_stride}"
            ));
        }
        let combined_stride = w1_stride + w3_stride;
        let mut gate_up_names = Vec::with_capacity(n_owned * 2);
        for e in 0..n_exp {
            if !owns(e) {
                continue;
            }
            for projection in ["w1", "w3"] {
                let name = format!("{prefix}.ffn.experts.{}.{projection}.weight", src(e));
                let info = hfq
                    .find_tensor_info(&name)
                    .ok_or_else(|| format!("deepseek4: missing {name}"))?;
                if info.data_size != w1_stride {
                    return Err(format!(
                        "deepseek4: {name} size {} != stride {w1_stride}",
                        info.data_size
                    ));
                }
                gate_up_names.push(name);
            }
        }

        let jobs = [
            HfqReadJob::packed(hfq, format!("{prefix}.w2"), &w2_names)
                .map_err(|e| format!("deepseek4: plan {prefix}.w2: {e}"))?,
            HfqReadJob::packed(hfq, format!("{prefix}.gate_up"), &gate_up_names)
                .map_err(|e| format!("deepseek4: plan {prefix}.gate_up: {e}"))?,
        ];
        let mut buffers = read_hfq_jobs_ordered(hfq, &jobs)
            .map_err(|e| format!("deepseek4: parallel expert read {prefix}: {e}"))?
            .into_iter();
        let w2_blob = buffers.next().expect("two planned expert jobs").data;
        let gate_up_blob = buffers.next().expect("two planned expert jobs").data;
        debug_assert_eq!(w2_blob.len(), w2_stride * n_owned);
        debug_assert_eq!(gate_up_blob.len(), combined_stride * n_owned);

        // Preserve the historical allocation order exactly: w2 owner, w2
        // pointer table, gate_up owner, optional dummy, gate_up pointer table.
        let mut w2_blob_shape = vec![n_owned];
        w2_blob_shape.extend_from_slice(&w2_shape);
        let w2_tensor = gpu
            .upload_raw(&w2_blob, &w2_blob_shape)
            .map_err(|e| format!("deepseek4: upload blob {prefix}.w2: {e:?}"))?;
        let w2_base = w2_tensor.buf.as_ptr() as u64;
        let w2_ptrs: Vec<u64> = (0..n_exp)
            .map(|e| {
                if owns(e) {
                    w2_base + (local_of_global[e] * w2_stride) as u64
                } else {
                    w2_base
                }
            })
            .collect();
        let w2_ptr_bytes: Vec<u8> = w2_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let w2_ptr_tensor = gpu
            .alloc_tensor(&[2 * n_exp], DType::F32)
            .map_err(|e| format!("deepseek4: alloc ptr table {prefix}.w2: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&w2_ptr_tensor.buf, &w2_ptr_bytes)
            .map_err(|e| format!("deepseek4: copy ptr table {prefix}.w2: {e:?}"))?;
        layer.expert_w2_blob = Some(w2_tensor);
        layer.expert_w2_ptrs = Some(w2_ptr_tensor);
        layer.expert_w2_stride = w2_stride;

        let gate_up_tensor = gpu
            .upload_raw(&gate_up_blob, &[n_owned, combined_stride])
            .map_err(|e| format!("deepseek4: upload gate_up {prefix}: {e:?}"))?;
        let gate_up_base = gate_up_tensor.buf.as_ptr() as u64;
        let dummy_gate_up = if shard.is_some() && n_owned < n_exp {
            Some(
                gpu.zeros(&[combined_stride / 4], DType::F32)
                    .map_err(|e| format!("deepseek4: {prefix} zero gate_up dummy: {e:?}"))?,
            )
        } else {
            None
        };
        let dummy_ptr = dummy_gate_up
            .as_ref()
            .map(|tensor| tensor.buf.as_ptr() as u64)
            .unwrap_or(gate_up_base);
        let gate_up_ptrs: Vec<u64> = (0..n_exp)
            .map(|e| {
                if owns(e) {
                    gate_up_base + (local_of_global[e] * combined_stride) as u64
                } else {
                    dummy_ptr
                }
            })
            .collect();
        let gate_up_ptr_bytes: Vec<u8> = gate_up_ptrs
            .iter()
            .flat_map(|pointer| pointer.to_ne_bytes())
            .collect();
        let gate_up_ptr_tensor = gpu
            .alloc_tensor(&[2 * n_exp], DType::F32)
            .map_err(|e| format!("deepseek4: alloc gate_up ptr table {prefix}: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&gate_up_ptr_tensor.buf, &gate_up_ptr_bytes)
            .map_err(|e| format!("deepseek4: copy gate_up ptr table {prefix}: {e:?}"))?;
        layer.expert_gate_up_blob = Some(gate_up_tensor);
        layer.expert_gate_up_ptrs = Some(gate_up_ptr_tensor);
        layer.expert_gate_up_stride = combined_stride;
        layer.expert_gate_up_dummy = dummy_gate_up;
        Ok(())
    }

    fn upload_layer_routed_experts(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        prefix: &str,
        n_exp: usize,
        layer: &mut DeepseekV4LayerWeights,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
        keep: Option<&[u32]>,
    ) -> Result<(), String> {
        if !hfq.has_overlay() {
            return Self::upload_layer_routed_experts_parallel(
                hfq, gpu, prefix, n_exp, layer, shard, keep,
            );
        }
        // REAP keep-map: compact slot `e` loads ORIGINAL expert `src(e)`.
        // `keep = None` ⇒ identity (slot == original index), byte-identical
        // to the full load. `n_exp` is the COMPACT count (kept) when active.
        if keep.is_some() && shard.is_some() {
            return Err("deepseek4: REAP keep-map + EP sharding are mutually exclusive".into());
        }
        if let Some(k) = keep {
            if k.len() != n_exp {
                return Err(format!(
                    "deepseek4: {prefix} keep slice len {} != n_exp {n_exp}",
                    k.len()
                ));
            }
        }
        let src = |slot: usize| -> usize { keep.map(|k| k[slot] as usize).unwrap_or(slot) };
        // EP shard: precompute owned set + compact-slot mapping. `shard = None`
        // ⇒ every expert owned, `local_of_global[e] == e`, n_owned == n_exp →
        // identical layout to the unsharded path.
        let owns = |e: usize| {
            shard
                .map(|(s, rank)| s.owns_expert(rank, e))
                .unwrap_or(true)
        };
        let mut local_of_global = vec![usize::MAX; n_exp];
        let mut n_owned = 0usize;
        for e in 0..n_exp {
            if owns(e) {
                local_of_global[e] = n_owned;
                n_owned += 1;
            }
        }
        if n_owned == 0 {
            return Err(format!("deepseek4: {prefix} shard rank owns no experts"));
        }

        // w2 (down): pread each expert; pack ONLY owned into a layer-local host
        // Vec, then one upload. Non-owned experts are read for stride
        // validation, then dropped (never uploaded — the EP memory win).
        {
            let name0 = format!("{prefix}.ffn.experts.{}.w2.weight", src(0));
            let (info0, _b0) = hfq
                .tensor_data_pread(&name0)
                .ok_or_else(|| format!("deepseek4: missing {name0}"))?;
            let stride = info0.data_size;
            let shape0: Vec<usize> = info0.shape.iter().map(|&s| s as usize).collect();
            drop(_b0);

            let mut blob = Vec::with_capacity(stride * n_owned);
            for e in 0..n_exp {
                // EP shard: read+pack ONLY owned experts (each rank reads just
                // its 1/N of the file → faster load, less page-cache churn).
                // Non-owned experts are never touched — their pointer-table
                // slot reuses the compact base (rotate input 0 ⇒ output 0).
                if !owns(e) {
                    continue;
                }
                let name = format!("{prefix}.ffn.experts.{}.w2.weight", src(e));
                let (info, bytes) = hfq
                    .tensor_data_pread(&name)
                    .ok_or_else(|| format!("deepseek4: missing {name}"))?;
                if info.data_size != stride {
                    return Err(format!(
                        "deepseek4: {name} size {} != stride {}",
                        info.data_size, stride
                    ));
                }
                blob.extend_from_slice(&bytes);
            }
            let mut blob_shape = vec![n_owned];
            blob_shape.extend_from_slice(&shape0);
            let blob_tensor = gpu
                .upload_raw(&blob, &blob_shape)
                .map_err(|e| format!("deepseek4: upload blob {prefix}.w2: {e:?}"))?;
            drop(blob);
            let base_ptr = blob_tensor.buf.as_ptr() as u64;
            // Owned e → compact slot; non-owned e → base (rotate input 0 ⇒
            // output 0 regardless of which down weights are read).
            let ptrs: Vec<u64> = (0..n_exp)
                .map(|e| {
                    if owns(e) {
                        base_ptr + (local_of_global[e] * stride) as u64
                    } else {
                        base_ptr
                    }
                })
                .collect();
            let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
            let ptr_tensor = gpu
                .alloc_tensor(&[2 * n_exp], rdna_compute::DType::F32)
                .map_err(|e| format!("deepseek4: alloc ptr table {prefix}.w2: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&ptr_tensor.buf, &ptr_bytes)
                .map_err(|e| format!("deepseek4: copy ptr table {prefix}.w2: {e:?}"))?;
            layer.expert_w2_blob = Some(blob_tensor);
            layer.expert_w2_ptrs = Some(ptr_tensor);
            layer.expert_w2_stride = stride;
        }
        // gate_up (combined w1 ‖ w3): per-expert pread, pack ONLY owned, single
        // upload. Non-owned ptr → a shared ZEROED dummy gate_up buffer.
        {
            let w1_0 = format!("{prefix}.ffn.experts.{}.w1.weight", src(0));
            let w3_0 = format!("{prefix}.ffn.experts.{}.w3.weight", src(0));
            let (w1_info0, _b1) = hfq
                .tensor_data_pread(&w1_0)
                .ok_or_else(|| format!("deepseek4: missing {w1_0}"))?;
            let stride_w1 = w1_info0.data_size;
            drop(_b1);
            let (w3_info0, _b3) = hfq
                .tensor_data_pread(&w3_0)
                .ok_or_else(|| format!("deepseek4: missing {w3_0}"))?;
            let stride_w3 = w3_info0.data_size;
            drop(_b3);
            if stride_w1 != stride_w3 {
                return Err(format!(
                    "deepseek4: {prefix} w1/w3 stride mismatch: w1={} w3={}",
                    stride_w1, stride_w3
                ));
            }
            let combined_stride = stride_w1 + stride_w3;
            let mut combined = Vec::with_capacity(combined_stride * n_owned);
            for e in 0..n_exp {
                // EP shard: pack ONLY owned experts. Each read's `Ref` on the
                // shared pread buffer MUST be dropped before the next pread
                // (the buffer is reused; holding two `Ref`s panics with
                // "RefCell already borrowed").
                if !owns(e) {
                    continue;
                }
                let w1_name = format!("{prefix}.ffn.experts.{}.w1.weight", src(e));
                {
                    let (_, w1_bytes) = hfq
                        .tensor_data_pread(&w1_name)
                        .ok_or_else(|| format!("deepseek4: missing {w1_name}"))?;
                    combined.extend_from_slice(&w1_bytes);
                }
                let w3_name = format!("{prefix}.ffn.experts.{}.w3.weight", src(e));
                {
                    let (_, w3_bytes) = hfq
                        .tensor_data_pread(&w3_name)
                        .ok_or_else(|| format!("deepseek4: missing {w3_name}"))?;
                    combined.extend_from_slice(&w3_bytes);
                }
            }
            let combined_tensor = gpu
                .upload_raw(&combined, &[n_owned, combined_stride])
                .map_err(|e| format!("deepseek4: upload gate_up {prefix}: {e:?}"))?;
            drop(combined);
            let base_ptr = combined_tensor.buf.as_ptr() as u64;
            // Non-owned gate_up ptr → a shared zeroed dummy (only when actually
            // sharding with some experts non-owned); else the compact base.
            // Owned (not mem::forget-leaked): the zeroed buffer is threaded into
            // `layer.expert_gate_up_dummy` so the staging guard reclaims it if a
            // later layer/global fails to load, and `free_gpu` reclaims it on a
            // successful EP unload. GpuTensor has no Drop, so leaving it on the
            // stack here would leak its buffer. Must outlive the device pointer
            // table built just below that bakes its address. Mirrors the
            // minimax `dummy_gate_up` fix.
            let dummy_gate_up = if shard.is_some() && n_owned < n_exp {
                let z = gpu
                    .zeros(&[combined_stride / 4], rdna_compute::DType::F32)
                    .map_err(|e| format!("deepseek4: {prefix} zero gate_up dummy: {e:?}"))?;
                Some(z)
            } else {
                None
            };
            let dummy_gu = dummy_gate_up
                .as_ref()
                .map(|z| z.buf.as_ptr() as u64)
                .unwrap_or(base_ptr);
            let ptrs: Vec<u64> = (0..n_exp)
                .map(|e| {
                    if owns(e) {
                        base_ptr + (local_of_global[e] * combined_stride) as u64
                    } else {
                        dummy_gu
                    }
                })
                .collect();
            let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
            let ptr_tensor = gpu
                .alloc_tensor(&[2 * n_exp], rdna_compute::DType::F32)
                .map_err(|e| format!("deepseek4: alloc gate_up ptr table {prefix}: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&ptr_tensor.buf, &ptr_bytes)
                .map_err(|e| format!("deepseek4: copy gate_up ptr table {prefix}: {e:?}"))?;
            layer.expert_gate_up_blob = Some(combined_tensor);
            layer.expert_gate_up_ptrs = Some(ptr_tensor);
            layer.expert_gate_up_stride = combined_stride;
            // Store the owning handle (None on single-GPU / fully-owned shards).
            // Its device pointer is already baked into `ptr_tensor` above.
            layer.expert_gate_up_dummy = dummy_gate_up;
        }
        Ok(())
    }

    /// Upload an F16-on-disk HFQ tensor as F32 on GPU. Used for norms
    /// where the kernel side (rmsnorm_f32) expects F32 weight, but the
    /// quantizer stored F16 bytes. The conversion cost is one host-side
    /// f16→f32 pass; norms are tiny (~4 KB each) so this is negligible.
    fn upload_global_f16_as_f32(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = hfq
            .tensor_data_pread(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        let n: usize = shape.iter().product();
        if bytes.len() != n * 2 {
            return Err(format!(
                "deepseek4: '{name}' expected F16 bytes ({} = 2 × {}), got {}",
                n * 2,
                n,
                bytes.len()
            ));
        }
        let f32_vals: Vec<f32> = (0..n)
            .map(|i| {
                let lo = bytes[i * 2];
                let hi = bytes[i * 2 + 1];
                hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([lo, hi]))
            })
            .collect();
        gpu.upload_f32(&f32_vals, &shape)
            .map_err(|e| format!("deepseek4: upload f16→f32 '{name}' failed: {e:?}"))
    }

    /// REAP keep-map variant of `upload_quant_or_f16`: byte row-gather only
    /// the kept output rows (experts) before upload. Exact for row-major,
    /// row-independent quant (F16 / Q8 / MQ*-G256) — each row's quant blocks
    /// are self-contained, so a byte gather preserves the original encoding.
    fn upload_quant_or_f16_keep(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
        keep: &[u32],
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = hfq
            .tensor_data_pread(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape_usize: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        let (new_shape, sub) = hipfire_reap::gather::gather_rows(&shape_usize, &bytes, keep)?;
        let mut t = gpu
            .upload_raw(&sub, &new_shape)
            .map_err(|e| format!("deepseek4: upload keep-subset '{name}' failed: {e:?}"))?;
        if let Some(dtype) = dense_hfq_dtype(info.quant_type) {
            t.dtype = dtype;
        }
        Ok(t)
    }

    /// REAP keep-map variant of `upload_global_f16_as_f32`: gather kept rows
    /// of an F16 `[n_orig, ..]` (or `[n_orig]`) tensor, then decode to F32.
    fn upload_global_f16_as_f32_keep(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
        keep: &[u32],
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = hfq
            .tensor_data_pread(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let orig_rows = *info.shape.first().unwrap_or(&0) as usize;
        if orig_rows == 0 || bytes.len() % (orig_rows * 2) != 0 {
            return Err(format!(
                "deepseek4: '{name}' f16 keep-gather: {orig_rows} rows × 2B don't divide {} bytes",
                bytes.len()
            ));
        }
        let per_row = bytes.len() / (orig_rows * 2); // f16 elems per row
        let mut f32_vals: Vec<f32> = Vec::with_capacity(per_row * keep.len());
        for &oe in keep {
            let oe = oe as usize;
            if oe >= orig_rows {
                return Err(format!(
                    "deepseek4: '{name}' keep idx {oe} >= rows {orig_rows}"
                ));
            }
            let base = oe * per_row * 2;
            for j in 0..per_row {
                let lo = bytes[base + j * 2];
                let hi = bytes[base + j * 2 + 1];
                f32_vals.push(hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    lo, hi,
                ])));
            }
        }
        let mut shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        shape[0] = keep.len();
        gpu.upload_f32(&f32_vals, &shape)
            .map_err(|e| format!("deepseek4: upload f16→f32 keep '{name}' failed: {e:?}"))
    }

    pub fn load_weights_host_only_walk(
        hfq: &HfqFile,
        cfg: &DeepseekV4Config,
    ) -> Result<DeepseekV4Weights, String> {
        let n_layers = cfg.num_hidden_layers;
        let mut layers: Vec<DeepseekV4LayerWeights> = Vec::with_capacity(n_layers);

        // Global tensors.
        for name in &[
            "embed.weight",
            "head.weight",
            "norm.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
        ] {
            if hfq.find_tensor_info(name).is_none() {
                return Err(format!("deepseek4: missing global tensor '{name}'"));
            }
        }

        // Per-layer tensors.
        for l in 0..n_layers {
            // Attention LoRA + KV joint + norms.
            for suffix in &[
                "attn.wq_a.weight",
                "attn.wq_b.weight",
                "attn.wkv.weight",
                "attn.wo_a.weight",
                "attn.wo_b.weight",
                "attn.q_norm.weight",
                "attn.kv_norm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn.attn_sink",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing '{suffix}'"));
                }
            }

            // Main compressor — ratio > 0. Indexer sub-module — only on
            // ratio == 4 layers. DeepSeek V4 config records the ratio array;
            // layers 0, 1, and 43 (MTP) have ratio = 0.
            let ratio = *cfg.compress_ratios.get(l).unwrap_or(&0);
            if ratio > 0 {
                for suffix in &[
                    "attn.compressor.wkv.weight",
                    "attn.compressor.wgate.weight",
                    "attn.compressor.norm.weight",
                    "attn.compressor.ape",
                ] {
                    let name = format!("layers.{l}.{suffix}");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} (ratio={ratio}) missing '{suffix}'"
                        ));
                    }
                }
            }
            if ratio == 4 {
                for suffix in &[
                    "attn.indexer.wq_b.weight",
                    "attn.indexer.weights_proj.weight",
                    "attn.indexer.compressor.wkv.weight",
                    "attn.indexer.compressor.wgate.weight",
                    "attn.indexer.compressor.norm.weight",
                    "attn.indexer.compressor.ape",
                ] {
                    let name = format!("layers.{l}.{suffix}");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} (ratio=4) missing indexer '{suffix}'"
                        ));
                    }
                }
            }

            // Hyper-Connections per-layer.
            for suffix in &[
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing HC tensor '{suffix}'"));
                }
            }

            // FFN router. The first `num_hash_layers` layers are HASH-
            // ROUTED — they have `gate.weight` but NO `gate.bias`. The
            // hash-routing table (`tid2eid`) is an I64 tensor that we
            // skip at ingest time (see commit 8ccfa42's skip-I64 path)
            // and restore as raw bytes in forward bring-up. Layers
            // beyond `num_hash_layers` use the standard `noaux_tc`
            // scoring path with `gate.weight` + `gate.bias`.
            //
            // On DeepSeek V4: num_hash_layers=3 → layers 0, 1, 2 are hash;
            // layers 3..43 are score-routed.
            let is_hash_routed = l < cfg.num_hash_layers;
            let name = format!("layers.{l}.ffn.gate.weight");
            if hfq.find_tensor_info(&name).is_none() {
                return Err(format!("deepseek4: layer {l} missing 'ffn.gate.weight'"));
            }
            if !is_hash_routed {
                let name = format!("layers.{l}.ffn.gate.bias");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!(
                        "deepseek4: layer {l} (score-routed) missing 'ffn.gate.bias'"
                    ));
                }
            }
            // Shared expert.
            for suffix in &[
                "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w2.weight",
                "ffn.shared_experts.w3.weight",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing shared '{suffix}'"));
                }
            }
            // Routed experts: kept × {w1, w2, w3}. `n_routed_experts` is the
            // kept count under a REAP keep-map; remap slot → original index.
            let ep = cfg.reap_keep.as_ref().map(|r| r.expert_plan(l));
            for e in 0..cfg.n_routed_experts {
                let e_src = ep.as_ref().map(|p| p.src(e)).unwrap_or(e);
                for proj in &["w1", "w2", "w3"] {
                    let name = format!("layers.{l}.ffn.experts.{e_src}.{proj}.weight");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} expert {e_src} missing '{proj}'"
                        ));
                    }
                }
            }

            layers.push(DeepseekV4LayerWeights::new_empty(ratio));
        }

        Ok(DeepseekV4Weights {
            mq2r_backend: Mq2rBackend::Portable,
            token_embd: None,
            output_norm: None,
            head: None,
            hc_head_fn: None,
            hc_head_base: None,
            hc_head_scale: 1.0, // overwritten at load time
            layers,
            mtp_layer: None, // skipped by quantize per `mtp.` prefix; Phase 5 work.
            dspark: None,    // DSpark sidecar discovered+loaded in load_weights_inner.
            _scaffold: (),
        })
    }
}

impl Architecture for DeepseekV4 {
    type Weights = DeepseekV4Weights;
    type State = DeepseekV4State;
    type Config = DeepseekV4Config;

    fn arch_id() -> u32 {
        // 9 = DeepSeek V4 Flash. Next free slot after 8 = Qwen2-VL
        // (reserved). Registered in docs/architecture-ids.md.
        9
    }

    fn name() -> &'static str {
        "deepseek4"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        DeepseekV4Config::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Ok(Self::load_weights_inner(hfq, cfg, gpu, None, None, None, false)?.into_single())
    }

    fn new_state(_gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        DeepseekV4State::new(cfg)
    }
}

impl DeepseekV4 {
    /// Exact byte projection for the base MQ2R split before either device is
    /// allowed to allocate. It mirrors the upload contract: F16 norms/biases
    /// expand to F32, the scalar head-HC scale remains host-side, routed
    /// records pack into two blobs per layer, and two u64 pointer tables are
    /// added per routed layer.
    pub fn project_heterogeneous_gfx1100_gfx1151(
        hfq: &HfqFile,
        cfg: &DeepseekV4Config,
    ) -> Result<DeepseekV4HeterogeneousProjection, String> {
        Self::validate_mq2r_tensor_policy(hfq, cfg)?;
        let mut projection = DeepseekV4HeterogeneousProjection::default();
        for tensor in hfq.tensors() {
            if tensor.name.contains(".ffn.experts.") {
                projection.routed_record_count += 1;
                projection.routed_bytes += tensor.data_size;
                continue;
            }
            projection.dense_record_count += 1;
            if tensor.name == "hc_head_scale" {
                projection.host_only_record_count += 1;
                continue;
            }
            projection.dense_allocation_count += 1;
            projection.dense_bytes += tensor.data_size;
            if Self::heterogeneous_f16_expands_to_f32(&tensor.name) {
                if tensor.quant_type != 1 {
                    return Err(format!(
                        "deepseek4 heterogeneous preflight: '{}' expands F16->F32 but has qt={}",
                        tensor.name, tensor.quant_type
                    ));
                }
                projection.dense_bytes += tensor.data_size;
                projection.f16_expansion_bytes += tensor.data_size;
            }
        }
        let expected_routed = cfg
            .num_hidden_layers
            .checked_mul(cfg.n_routed_experts)
            .and_then(|count| count.checked_mul(3))
            .ok_or_else(|| "deepseek4 heterogeneous routed record-count overflow".to_string())?;
        if projection.routed_record_count != expected_routed {
            return Err(format!(
                "deepseek4 heterogeneous preflight: {} routed records, expected {expected_routed}",
                projection.routed_record_count
            ));
        }
        // The artifact SHA is checked by the public heterogeneous loader, and
        // this exact census makes the placement classifier fail closed if a
        // future artifact adds, drops, or silently reclassifies a record.
        const EXPECTED_DENSE_RECORDS: usize = 1_199;
        const EXPECTED_HOST_ONLY_RECORDS: usize = 1;
        if projection.dense_record_count != EXPECTED_DENSE_RECORDS
            || projection.host_only_record_count != EXPECTED_HOST_ONLY_RECORDS
        {
            return Err(format!(
                "deepseek4 heterogeneous preflight: dense/host-only census {}/{}; expected {EXPECTED_DENSE_RECORDS}/{EXPECTED_HOST_ONLY_RECORDS}",
                projection.dense_record_count, projection.host_only_record_count
            ));
        }
        // The packed implementation materializes exactly w2 + gate_up owners
        // and one u64 pointer per global expert for each owner.
        projection.routed_allocation_count = cfg.num_hidden_layers * 4;
        projection.pointer_table_bytes = cfg
            .num_hidden_layers
            .checked_mul(cfg.n_routed_experts)
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| "deepseek4 heterogeneous pointer-table overflow".to_string())?;
        projection.routed_bytes = projection
            .routed_bytes
            .checked_add(projection.pointer_table_bytes)
            .ok_or_else(|| "deepseek4 heterogeneous routed byte overflow".to_string())?;
        Ok(projection)
    }

    fn heterogeneous_f16_expands_to_f32(name: &str) -> bool {
        name == "norm.weight"
            || name.ends_with(".attn_norm.weight")
            || name.ends_with(".ffn_norm.weight")
            || name.ends_with(".attn.q_norm.weight")
            || name.ends_with(".attn.kv_norm.weight")
            || name.ends_with(".attn.attn_sink")
            || name.ends_with(".compressor.norm.weight")
            || name.ends_with(".compressor.ape")
            || name.ends_with(".ffn.gate.bias")
    }

    fn validate_mq2_family_tensor_policy(
        hfq: &HfqFile,
        cfg: &DeepseekV4Config,
        dense_qt: u8,
        sku: &str,
    ) -> Result<(), String> {
        const QT_Q8F16: u8 = 3;
        const QT_MQ2_LLOYD: u8 = 19;
        const EXPECTED_DENSE_TENSORS: usize = 554;

        if hfq.has_overlay() {
            return Err(format!(
                "deepseek4 {sku}: standalone product artifact refuses runtime REAP overlays"
            ));
        }

        let require_qt = |name: &str, expected: u8| -> Result<(), String> {
            let info = hfq
                .find_tensor_info(name)
                .ok_or_else(|| format!("deepseek4 {sku}: missing tensor '{name}'"))?;
            if info.quant_type != expected {
                return Err(format!(
                    "deepseek4 {sku}: '{name}' has qt={}, expected qt={expected}",
                    info.quant_type
                ));
            }
            Ok(())
        };

        require_qt("embed.weight", QT_Q8F16)?;
        require_qt("head.weight", dense_qt)?;
        let mut expected_dense = 1usize; // head

        for layer in 0..cfg.num_hidden_layers {
            for suffix in [
                "attn.wq_a.weight",
                "attn.wq_b.weight",
                "attn.wkv.weight",
                "attn.wo_a.weight",
                "attn.wo_b.weight",
                "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w2.weight",
                "ffn.shared_experts.w3.weight",
            ] {
                require_qt(&format!("layers.{layer}.{suffix}"), dense_qt)?;
                expected_dense += 1;
            }

            let ratio = cfg.compress_ratios.get(layer).copied().unwrap_or(0);
            if ratio > 0 {
                for suffix in ["attn.compressor.wkv.weight", "attn.compressor.wgate.weight"] {
                    require_qt(&format!("layers.{layer}.{suffix}"), dense_qt)?;
                    expected_dense += 1;
                }
            }
            if ratio == 4 {
                for suffix in [
                    "attn.indexer.wq_b.weight",
                    "attn.indexer.weights_proj.weight",
                    "attn.indexer.compressor.wkv.weight",
                    "attn.indexer.compressor.wgate.weight",
                ] {
                    require_qt(&format!("layers.{layer}.{suffix}"), dense_qt)?;
                    expected_dense += 1;
                }
            }

            require_qt(&format!("layers.{layer}.ffn.gate.weight"), dense_qt)?;
            expected_dense += 1;

            for expert in 0..cfg.n_routed_experts {
                for projection in ["w1", "w2", "w3"] {
                    require_qt(
                        &format!("layers.{layer}.ffn.experts.{expert}.{projection}.weight"),
                        QT_MQ2_LLOYD,
                    )?;
                }
            }
        }

        if expected_dense != EXPECTED_DENSE_TENSORS {
            return Err(format!(
                "deepseek4 {sku}: recipe resolved {expected_dense} dense tensors, expected {EXPECTED_DENSE_TENSORS}"
            ));
        }
        let actual_dense = hfq
            .tensors()
            .iter()
            .filter(|tensor| tensor.quant_type == dense_qt)
            .count();
        if actual_dense != EXPECTED_DENSE_TENSORS {
            return Err(format!(
                "deepseek4 {sku}: artifact carries {actual_dense} dense-tier tensors, expected {EXPECTED_DENSE_TENSORS}"
            ));
        }
        Ok(())
    }

    fn validate_mq2r_tensor_policy(hfq: &HfqFile, cfg: &DeepseekV4Config) -> Result<(), String> {
        Self::validate_mq2_family_tensor_policy(hfq, cfg, 35, "MQ2R")
    }

    fn validate_mq2rxt_tensor_policy(hfq: &HfqFile, cfg: &DeepseekV4Config) -> Result<(), String> {
        let metadata: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|error| format!("deepseek4 MQ2RXT: invalid metadata JSON: {error}"))?;
        if metadata
            .get("hipfire_quant_recipe")
            .and_then(serde_json::Value::as_str)
            != Some("deepseek4-mq2rxt-mq4-p3-v1")
        {
            return Err("deepseek4 MQ2RXT: missing exact product recipe identity".to_owned());
        }
        Self::validate_mq2_family_tensor_policy(hfq, cfg, 13, "MQ2RXT")?;
        let stale_e8 = hfq
            .tensors()
            .iter()
            .filter(|tensor| tensor.quant_type == 35)
            .count();
        if stale_e8 != 0 {
            return Err(format!(
                "deepseek4 MQ2RXT: artifact carries {stale_e8} stale E8 tensors; expected 0"
            ));
        }
        Ok(())
    }

    fn validate_mq2r_dspark_sidecar(sidecar: &HfqFile) -> Result<(), String> {
        let metadata: serde_json::Value = serde_json::from_str(&sidecar.metadata_json)
            .map_err(|error| format!("deepseek4 MQ2R DSpark: invalid metadata JSON: {error}"))?;
        let identity = metadata
            .get("mq2r_sidecar")
            .ok_or("deepseek4 MQ2R DSpark: missing mq2r_sidecar metadata identity")?;
        let target_recipe = identity
            .get("target_recipe")
            .and_then(serde_json::Value::as_str);
        if target_recipe != Some("deepseek4-mq2r-e8-p3-v1") {
            return Err(format!(
                "deepseek4 MQ2R DSpark: target_recipe={target_recipe:?}, \
                 expected deepseek4-mq2r-e8-p3-v1"
            ));
        }
        let draft_head = identity
            .get("draft_head")
            .and_then(serde_json::Value::as_str);
        if draft_head != Some("trunk_mfp4_e8_soa_b4") {
            return Err(format!(
                "deepseek4 MQ2R DSpark: draft_head={draft_head:?}, \
                 expected trunk_mfp4_e8_soa_b4"
            ));
        }
        if sidecar.find_tensor_info("draft_head.weight").is_some() {
            return Err(
                "deepseek4 MQ2R DSpark: v1 native-E8 sidecar must not carry draft_head.weight"
                    .to_owned(),
            );
        }
        Ok(())
    }

    fn validate_mq2rxt_dspark_sidecar(sidecar: &HfqFile) -> Result<(), String> {
        const QT_Q8F16: u8 = 3;
        const QT_MQ4: u8 = 13;
        const QT_MQ2_LLOYD: u8 = 19;
        const EXPECTED_MQ4: usize = 24;
        const EXPECTED_MQ2: usize = 2_304;
        const EXPECTED_Q8: usize = 7;
        const EXPECTED_F16: usize = 41;
        const EXPECTED_TOTAL: usize = 2_376;

        let metadata: serde_json::Value = serde_json::from_str(&sidecar.metadata_json)
            .map_err(|error| format!("deepseek4 MQ2RXT DSpark: invalid metadata JSON: {error}"))?;
        let identity = metadata
            .get("mq2rxt_sidecar")
            .ok_or("deepseek4 MQ2RXT DSpark: missing mq2rxt_sidecar metadata identity")?;
        if metadata
            .get("hipfire_quant_recipe")
            .and_then(serde_json::Value::as_str)
            != Some("deepseek4-mq2rxt-mq4-p3-v1")
        {
            return Err(
                "deepseek4 MQ2RXT DSpark: missing exact product recipe identity".to_owned(),
            );
        }
        if identity
            .get("target_recipe")
            .and_then(serde_json::Value::as_str)
            != Some("deepseek4-mq2rxt-mq4-p3-v1")
        {
            return Err("deepseek4 MQ2RXT DSpark: wrong target_recipe identity".to_owned());
        }
        if identity
            .get("draft_head")
            .and_then(serde_json::Value::as_str)
            != Some("trunk_mq4g256_b4")
        {
            return Err("deepseek4 MQ2RXT DSpark: wrong draft_head identity".to_owned());
        }
        if sidecar.find_tensor_info("draft_head.weight").is_some() {
            return Err("deepseek4 MQ2RXT DSpark: sidecar must reuse the trunk head".to_owned());
        }

        let require_qt = |name: &str, expected: u8| -> Result<(), String> {
            let info = sidecar
                .find_tensor_info(name)
                .ok_or_else(|| format!("deepseek4 MQ2RXT DSpark: missing tensor '{name}'"))?;
            if info.quant_type != expected {
                return Err(format!(
                    "deepseek4 MQ2RXT DSpark: '{name}' has qt={}, expected qt={expected}",
                    info.quant_type
                ));
            }
            Ok(())
        };

        for stage in 0..3 {
            for suffix in [
                "attn.wq_a.weight",
                "attn.wq_b.weight",
                "attn.wkv.weight",
                "attn.wo_a.weight",
                "attn.wo_b.weight",
                "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w2.weight",
                "ffn.shared_experts.w3.weight",
            ] {
                require_qt(&format!("mtp.{stage}.{suffix}"), QT_MQ4)?;
            }
            for expert in 0..256 {
                for projection in ["w1", "w2", "w3"] {
                    require_qt(
                        &format!("mtp.{stage}.ffn.experts.{expert}.{projection}.weight"),
                        QT_MQ2_LLOYD,
                    )?;
                }
            }
            require_qt(&format!("mtp.{stage}.ffn.gate.weight"), QT_Q8F16)?;
        }
        for name in [
            "mtp.0.main_proj.weight",
            "mtp.2.confidence_head.proj.weight",
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
        ] {
            require_qt(name, QT_Q8F16)?;
        }

        let count = |qt: u8| {
            sidecar
                .tensors()
                .iter()
                .filter(|tensor| tensor.quant_type == qt)
                .count()
        };
        for (qt, expected, label) in [
            (QT_MQ4, EXPECTED_MQ4, "MQ4"),
            (QT_MQ2_LLOYD, EXPECTED_MQ2, "MQ2-Lloyd"),
            (QT_Q8F16, EXPECTED_Q8, "Q8"),
        ] {
            let actual = count(qt);
            if actual != expected {
                return Err(format!(
                    "deepseek4 MQ2RXT DSpark: {actual} {label} tensors, expected {expected}"
                ));
            }
        }
        if count(35) != 0 {
            return Err("deepseek4 MQ2RXT DSpark: stale E8 payload present".to_owned());
        }
        if count(1) != EXPECTED_F16 {
            return Err(format!(
                "deepseek4 MQ2RXT DSpark: {} F16 tensors, expected {EXPECTED_F16}",
                count(1)
            ));
        }
        if sidecar.tensors().len() != EXPECTED_TOTAL {
            return Err(format!(
                "deepseek4 MQ2RXT DSpark: {} total tensors, expected {EXPECTED_TOTAL}",
                sidecar.tensors().len()
            ));
        }
        Ok(())
    }

    /// EP shard-aware load entry (mirrors `MiniMaxWeights::load`).
    ///
    /// Loads the full model but uploads only `rank`'s owned routed experts
    /// per layer (non-owned ptr → zeroed dummy), so an 81 GB model fits across
    /// N×32 GB cards under all-reduce EP. Non-expert weights (embed, head,
    /// attention, norms, shared expert, router) are replicated per rank.
    pub fn load_weights_sharded(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        gpu: &mut Gpu,
        shard: &hipfire_runtime::tp_shard::ShardConfig,
        rank: usize,
    ) -> Result<DeepseekV4Weights, String> {
        Ok(
            Self::load_weights_inner(hfq, cfg, gpu, Some((shard, rank)), None, None, false)?
                .into_single(),
        )
    }

    /// Load the fixed MQ2R artifact directly onto its two exact device owners:
    /// every non-routed tensor on gfx1100 and only packed routed-expert blobs
    /// plus pointer tables on gfx1151. The file is opened once by the caller;
    /// no full-model staging or post-load migration is performed.
    pub fn load_weights_heterogeneous_gfx1100_gfx1151(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        dense_gpu: &mut Gpu,
        routed_gpu: &mut Gpu,
    ) -> Result<DeepseekV4HeterogeneousWeights, String> {
        if dense_gpu.device_id == routed_gpu.device_id {
            return Err("deepseek4 heterogeneous load requires two distinct devices".into());
        }
        if dense_gpu.arch != "gfx1100" || routed_gpu.arch != "gfx1151" {
            return Err(format!(
                "deepseek4 heterogeneous exact admission requires dense=gfx1100 and routed=gfx1151; got dense={} (dev {}) routed={} (dev {})",
                dense_gpu.arch, dense_gpu.device_id, routed_gpu.arch, routed_gpu.device_id
            ));
        }
        if !cfg.mq2r || cfg.mq2rxt {
            return Err(
                "deepseek4 heterogeneous route admits only the frozen MQ2R P3 artifact".into(),
            );
        }
        if cfg.load_dspark {
            return Err(
                "deepseek4 heterogeneous G2 load excludes DSpark; enable it only after the base dual-device route is certified"
                    .into(),
            );
        }
        Self::load_weights_inner(hfq, cfg, dense_gpu, None, Some(routed_gpu), None, false)?
            .into_heterogeneous()
    }

    #[doc(hidden)]
    pub fn load_weights_heterogeneous_gfx1100_gfx1151_with_fault(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        dense_gpu: &mut Gpu,
        routed_gpu: &mut Gpu,
        fault: DeepseekV4HeterogeneousFault,
    ) -> Result<DeepseekV4HeterogeneousWeights, String> {
        if dense_gpu.device_id == routed_gpu.device_id
            || dense_gpu.arch != "gfx1100"
            || routed_gpu.arch != "gfx1151"
            || !cfg.mq2r
            || cfg.mq2rxt
            || cfg.load_dspark
        {
            return Err(
                "deepseek4 heterogeneous fault harness requires the exact admitted G2 route".into(),
            );
        }
        Self::load_weights_inner(
            hfq,
            cfg,
            dense_gpu,
            None,
            Some(routed_gpu),
            Some(fault),
            false,
        )?
        .into_heterogeneous()
    }

    fn validate_harmonic_worker_role(
        cfg: &DeepseekV4Config,
        gpu: &Gpu,
        owner: HarmonicOwner,
    ) -> Result<(), String> {
        owner.validate_arch(&gpu.arch).map_err(|error| {
            format!(
                "deepseek4 harmonic {:?} worker exact admission failed on device {}: {error}",
                owner, gpu.device_id
            )
        })?;
        if !cfg.mq2r || cfg.mq2rxt {
            return Err(
                "deepseek4 harmonic workers admit only the frozen MQ2R P3 artifact".to_owned(),
            );
        }
        if cfg.load_dspark {
            return Err("deepseek4 harmonic AR workers exclude DSpark".to_owned());
        }
        if cfg.num_hidden_layers != crate::harmonic::HARMONIC_LAYER_COUNT as usize
            || cfg.n_routed_experts != crate::harmonic::HARMONIC_EXPERT_COUNT as usize
            || cfg.num_experts_per_tok != crate::harmonic::HARMONIC_TOP_K
        {
            return Err(format!(
                "deepseek4 harmonic workers require layers/experts/top-k={}/{}/{}; got {}/{}/{}",
                crate::harmonic::HARMONIC_LAYER_COUNT,
                crate::harmonic::HARMONIC_EXPERT_COUNT,
                crate::harmonic::HARMONIC_TOP_K,
                cfg.num_hidden_layers,
                cfg.n_routed_experts,
                cfg.num_experts_per_tok
            ));
        }
        if cfg.reap_keep.is_some() {
            return Err("deepseek4 harmonic workers exclude REAP expert keep maps".to_owned());
        }
        Ok(())
    }

    /// Load only the canonical non-routed tower into an independently owned
    /// exact-gfx1100 worker. No routed expert payload or foreign `Gpu` is ever
    /// constructed in this process.
    pub fn load_weights_harmonic_dense_gfx1100(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        gpu: &mut Gpu,
    ) -> Result<DeepseekV4DenseWeights, String> {
        Self::validate_harmonic_worker_role(cfg, gpu, HarmonicOwner::DenseGfx1100)?;
        Self::load_weights_inner(hfq, cfg, gpu, None, None, None, true)?.into_dense()
    }

    /// Validate a proposed exact-expert replica set against the loaded frozen
    /// artifact without allocating VRAM or changing dispatch. The full routed
    /// payload remains resident on gfx1151; this only proves which byte-identical
    /// slots may later be additionally loaded on gfx1100.
    pub fn project_harmonic_hot_experts_gfx1100(
        hfq: &HfqFile,
        cfg: &DeepseekV4Config,
        gpu: &Gpu,
        plan: &HarmonicExpertResidencyPlan,
    ) -> Result<DeepseekV4HarmonicResidencyProjection, String> {
        Self::validate_harmonic_worker_role(cfg, gpu, HarmonicOwner::DenseGfx1100)?;
        Self::validate_mq2r_tensor_policy(hfq, cfg)?;

        let mut exact_bytes = 0_u64;
        let mut layer_slot_counts = [0_u16; HARMONIC_LAYER_COUNT as usize];
        for slot in plan.slots() {
            let mut slot_bytes = 0_u64;
            for projection in ["w1", "w2", "w3"] {
                let name = format!(
                    "layers.{}.ffn.experts.{}.{projection}.weight",
                    slot.layer, slot.expert
                );
                let info = hfq
                    .find_tensor_info(&name)
                    .ok_or_else(|| format!("deepseek4 harmonic residency missing {name}"))?;
                if info.quant_type != 19 {
                    return Err(format!(
                        "deepseek4 harmonic residency {name} has quant_type {}, expected frozen MQ2G256Lloyd qt=19",
                        info.quant_type
                    ));
                }
                slot_bytes = slot_bytes
                    .checked_add(info.data_size as u64)
                    .ok_or_else(|| {
                        format!("deepseek4 harmonic residency byte overflow at {name}")
                    })?;
            }
            if slot_bytes != plan.slot_bytes() {
                return Err(format!(
                    "deepseek4 harmonic residency layer {} expert {} is {slot_bytes} bytes, plan requires {}",
                    slot.layer,
                    slot.expert,
                    plan.slot_bytes()
                ));
            }
            exact_bytes = exact_bytes
                .checked_add(slot_bytes)
                .ok_or_else(|| "deepseek4 harmonic residency total byte overflow".to_owned())?;
            layer_slot_counts[slot.layer as usize] = layer_slot_counts[slot.layer as usize]
                .checked_add(1)
                .ok_or_else(|| {
                    format!(
                        "deepseek4 harmonic residency layer {} slot-count overflow",
                        slot.layer
                    )
                })?;
        }
        if exact_bytes != plan.required_bytes() {
            return Err(format!(
                "deepseek4 harmonic residency exact bytes {exact_bytes} != planned {}",
                plan.required_bytes()
            ));
        }
        let headroom_bytes = plan
            .budget_bytes()
            .checked_sub(exact_bytes)
            .ok_or_else(|| {
                format!(
                    "deepseek4 harmonic residency exceeds budget: {exact_bytes} > {}",
                    plan.budget_bytes()
                )
            })?;
        Ok(DeepseekV4HarmonicResidencyProjection {
            plan_identity: plan.identity(),
            slot_count: plan.slots().len(),
            exact_bytes,
            budget_bytes: plan.budget_bytes(),
            headroom_bytes,
            layer_slot_counts,
        })
    }

    /// Upload the already-projected exact replica set into compact per-layer
    /// blobs on gfx1100. The caller must retain the full routed model on
    /// gfx1151; this is an additive locality cache, never expert pruning.
    pub fn load_weights_harmonic_hot_experts_gfx1100(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        gpu: &mut Gpu,
        plan: &HarmonicExpertResidencyPlan,
    ) -> Result<DeepseekV4HarmonicReplicaWeights, String> {
        let projection = Self::project_harmonic_hot_experts_gfx1100(hfq, cfg, gpu, plan)?;
        if projection.layer_slot_counts.iter().any(|count| *count == 0) {
            return Err(
                "deepseek4 harmonic replica plan must select at least one expert per layer"
                    .to_owned(),
            );
        }
        hfq.drop_mmap();
        let mut staging = DeepseekV4RoutedWeightStaging::new(
            DeepseekV4RoutedWeights::new(cfg, Mq2rBackend::Portable),
            gpu,
        );
        for layer_index in 0..cfg.num_hidden_layers {
            let keep = plan.experts_in_layer(layer_index).collect::<Vec<_>>();
            let layer = staging
                .routed_mut()
                .layer_mut(layer_index)
                .ok_or_else(|| format!("deepseek4: replica owner missing layer {layer_index}"))?;
            Self::upload_layer_routed_experts(
                hfq,
                gpu,
                &format!("layers.{layer_index}"),
                keep.len(),
                layer,
                None,
                Some(&keep),
            )?;
        }
        hfq.shrink_pread_buf();
        Ok(DeepseekV4HarmonicReplicaWeights::new(
            staging.publish(),
            plan.identity(),
        ))
    }

    /// Load only the packed routed experts into an independently owned
    /// exact-gfx1151 worker. Dense weights, canonical state, routing, RMSNorm,
    /// and FWHT remain absent from this process by construction.
    pub fn load_weights_harmonic_experts_gfx1151(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        gpu: &mut Gpu,
    ) -> Result<DeepseekV4RoutedWeights, String> {
        Self::validate_harmonic_worker_role(cfg, gpu, HarmonicOwner::ExpertGfx1151)?;
        Self::validate_mq2r_tensor_policy(hfq, cfg)?;
        hfq.drop_mmap();

        let backend = Mq2rBackend::for_verified_mq2r(gpu);
        if !backend.is_gfx1151() {
            return Err(
                "deepseek4 harmonic expert worker did not select the exact gfx1151 backend"
                    .to_owned(),
            );
        }
        let mut staging =
            DeepseekV4RoutedWeightStaging::new(DeepseekV4RoutedWeights::new(cfg, backend), gpu);
        for layer_index in 0..cfg.num_hidden_layers {
            let layer = staging
                .routed_mut()
                .layer_mut(layer_index)
                .ok_or_else(|| format!("deepseek4: routed owner missing layer {layer_index}"))?;
            Self::upload_layer_routed_experts(
                hfq,
                gpu,
                &format!("layers.{layer_index}"),
                cfg.n_routed_experts,
                layer,
                None,
                None,
            )?;
        }
        hfq.shrink_pread_buf();
        Ok(staging.publish())
    }

    fn load_weights_inner(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        gpu: &mut Gpu,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
        mut routed_gpu: Option<&mut Gpu>,
        heterogeneous_fault: Option<DeepseekV4HeterogeneousFault>,
        dense_only: bool,
    ) -> Result<DeepseekV4WeightStaging, String> {
        // Model identity and route identity are intentionally separate.
        // `.mq2r` fixes the exact P3 tensor recipe on every architecture.
        // Native eligibility is installed on the returned DS4 weights after
        // verification; it is never written into the process-wide GPU.
        // This is not automatic Redline admission.
        if cfg.mq2rxt {
            Self::validate_mq2rxt_tensor_policy(hfq, cfg)?;
        } else if cfg.mq2r {
            Self::validate_mq2r_tensor_policy(hfq, cfg)?;
        }

        // Phase 1.5 host walk verifies every expected tensor is in the
        // HFQ index. We then upload all globals and per-layer
        // non-expert tensors. The 256 routed experts per layer are
        // default ON (most of the model's bytes — DeepSeek V4 is unusable
        // without them). Opt out with `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0`
        // for shared-only-FFN diagnostic loads.
        //
        // For VRAM-constrained partial-MoE testing, set
        //   HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END=N
        // to upload routed experts only for layers in [num_hash_layers,
        // N). Layers >= N fall back to shared-only FFN. Each layer's
        // expert blob is ~1.84 GB on the FP4-fixed HFQ (post-unpack
        // logical shape), so 22 layers ≈ 40 GB.
        let upload_experts = !dense_only
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS")
                .ok()
                .as_deref()
                != Some("0");
        let expert_layer_end: Option<usize> =
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END")
                .ok()
                .and_then(|s| s.parse().ok());

        // ── MTP addon HFQ discovery ──────────────────────────────────────
        // Resolves an optional second HFQ holding only `mtp.0.*` tensors so
        // users can opt into MTP / speculative decoding without re-quantizing
        // the 86 GB base. Resolution order (first match wins):
        //
        //   1. HIPFIRE_DEEPSEEK4_MTP_ADDON=<path>       — explicit override
        //   2. <base>.mtp-addon.hfq                     — `.mtp-addon.hfq`
        //      e.g. v4f.mq2lloyd-q8.hfq  →  v4f.mq2lloyd-q8.mtp-addon.hfq
        //   3. <stem>-mtp.<ext>                         — `-mtp` infix
        //      e.g. deepseek-v4-flash.mq2lloyd  →  deepseek-v4-flash-mtp.mq2lloyd
        //
        // When set, ALL `mtp.0.*` reads in the block below source from the
        // addon instead of the base. The MTP layer is present iff the addon
        // (or, for one-shot quants that put MTP in-band, the base) contains
        // `mtp.0.norm.weight`.
        let mut mtp_addon: Option<HfqFile> = if routed_gpu.is_some() || dense_only {
            // G2 owns only the frozen trunk. Auxiliary/speculative weights are
            // admitted after the base split route has passed full correctness.
            None
        } else {
            let env_path = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MTP_ADDON").ok();
            let resolved: Option<std::path::PathBuf> = if let Some(p) = env_path {
                Some(std::path::PathBuf::from(p))
            } else {
                let base = hfq.path();
                // Convention 1: append `.mtp-addon.hfq` (legacy).
                let stem = base.to_string_lossy();
                let conv1 = if let Some(s) = stem.strip_suffix(".hfq") {
                    std::path::PathBuf::from(format!("{s}.mtp-addon.hfq"))
                } else {
                    std::path::PathBuf::from(format!("{stem}.mtp-addon.hfq"))
                };
                // Convention 2: insert `-mtp` before the extension.
                let conv2 = match (base.parent(), base.file_stem(), base.extension()) {
                    (Some(parent), Some(file_stem), Some(ext)) => Some(parent.join(format!(
                        "{}-mtp.{}",
                        file_stem.to_string_lossy(),
                        ext.to_string_lossy()
                    ))),
                    _ => None,
                };
                if conv1.exists() {
                    Some(conv1)
                } else {
                    conv2.filter(|c| c.exists())
                }
            };
            match resolved {
                Some(p) => {
                    eprintln!("deepseek4: opening MTP addon HFQ {p:?}");
                    match HfqFile::open(&p) {
                        Ok(f) => Some(f),
                        Err(e) => {
                            return Err(format!(
                                "deepseek4: failed to open MTP addon HFQ {p:?}: {e:?}"
                            ));
                        }
                    }
                }
                None => None,
            }
        };

        let initial_weights = Self::load_weights_host_only_walk(hfq, cfg)?;
        let routed_weights = routed_gpu
            .as_ref()
            .map(|_| DeepseekV4RoutedWeights::new(cfg, Mq2rBackend::Portable));
        let mut weights = DeepseekV4WeightStaging::new(
            initial_weights,
            routed_weights,
            gpu,
            routed_gpu.as_deref_mut(),
        );
        if cfg.mq2r {
            weights.mq2r_backend = Mq2rBackend::for_verified_mq2r(gpu);
            if let (Some(routed), Some(expert_gpu)) =
                (weights.routed.as_mut(), routed_gpu.as_deref_mut())
            {
                routed.mq2r_backend = Mq2rBackend::for_verified_mq2r(expert_gpu);
            }
            let sku = if cfg.mq2rxt { "MQ2RXT" } else { "MQ2R" };
            let dense = if cfg.mq2rxt {
                "554 MQ4 tensors"
            } else {
                "554 E8 tensors"
            };
            match weights.mq2r_backend {
                Mq2rBackend::Gfx1151 => eprintln!(
                    "deepseek4: {sku} P3 tensor recipe verified; selected \
                     gfx1151 backend ({dense}; routed experts qt=19)"
                ),
                Mq2rBackend::Gfx942(_) => eprintln!(
                    "deepseek4: {sku} P3 tensor recipe verified; selected exact \
                     gfx942 backend ({dense}; routed experts qt=19)"
                ),
                Mq2rBackend::Portable => eprintln!(
                    "deepseek4: {sku} P3 tensor recipe verified; no native backend \
                     for {}, using portable dispatch",
                    gpu.arch
                ),
            }
            crate::forward::config_cache_log_gfx942_a2_levers(
                &gpu.arch,
                weights.mq2r_backend.is_gfx942(),
            );
        }

        // Drop the mmap BEFORE any tensor uploads. Every upload helper
        // below now uses `tensor_data_pread` (pread + FADV_DONTNEED)
        // which doesn't need the mmap alive. On unified-memory APUs
        // (Strix Halo etc.), holding the mmap during the upload pass
        // populates page cache that competes 1:1 with the upcoming
        // hipMalloc allocations — for the 88 GB deepseek4-q8-mtp build that
        // OOMs the 125 GB system at layer ~42. The earlier "drop after
        // dense pass" pattern (Phase B, 2026-05-19) was just one step
        // along that path; this completes the migration.
        // Also drop the addon's mmap on the same grounds.
        hfq.drop_mmap();
        if let Some(ref mut addon) = mtp_addon {
            addon.drop_mmap();
        }

        // Globals. Norms are F16 on disk but the kernels expect F32
        // weight; convert at upload time.
        //
        // `head.weight` MUST use `upload_quant_or_f16` so its dtype gets
        // tagged correctly (F16 / Q8_0 / Raw). With `upload_global_raw`
        // the dtype is always Raw, which makes `gemv_auto` dispatch to
        // the MQ4 fallback regardless of actual quant — Q8F16 bytes get
        // read as MQ4 blocks and produce NaN logits silently. Same
        // potential trap for `token_embd`, but the embedding_lookup_q8
        // kernel reads bytes layout-directly and doesn't gate on dtype,
        // so leaving it as raw upload is currently safe.
        weights.token_embd = Some(Self::upload_global_raw(hfq, gpu, "embed.weight")?);
        weights.output_norm = Some(Self::upload_global_f16_as_f32(hfq, gpu, "norm.weight")?);
        weights.head = Some(Self::upload_quant_or_f16(hfq, gpu, "head.weight")?);

        // Head HC mix tensors — F16 raw on GPU; scale is scalar host-side.
        weights.hc_head_fn = Some(Self::upload_global_raw(hfq, gpu, "hc_head_fn")?);
        weights.hc_head_base = Some(Self::upload_global_raw(hfq, gpu, "hc_head_base")?);
        {
            let (info, bytes) = hfq
                .tensor_data_pread("hc_head_scale")
                .ok_or_else(|| "deepseek4: hc_head_scale missing".to_string())?;
            if info.shape != vec![1] {
                return Err(format!(
                    "deepseek4: hc_head_scale unexpected shape {:?}",
                    info.shape
                ));
            }
            let scale =
                hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            weights.hc_head_scale = scale;
        }

        // Per-layer.
        for (l, layer) in weights.layers.iter_mut().enumerate() {
            // Norms (F16 on disk → F32 on GPU).
            layer.attn_norm = Some(Self::upload_global_f16_as_f32(
                hfq,
                gpu,
                &format!("layers.{l}.attn_norm.weight"),
            )?);
            layer.ffn_norm = Some(Self::upload_global_f16_as_f32(
                hfq,
                gpu,
                &format!("layers.{l}.ffn_norm.weight"),
            )?);
            layer.q_norm = Some(Self::upload_global_f16_as_f32(
                hfq,
                gpu,
                &format!("layers.{l}.attn.q_norm.weight"),
            )?);
            layer.kv_norm = Some(Self::upload_global_f16_as_f32(
                hfq,
                gpu,
                &format!("layers.{l}.attn.kv_norm.weight"),
            )?);
            layer.attn_sink = Some(Self::upload_global_f16_as_f32(
                hfq,
                gpu,
                &format!("layers.{l}.attn.attn_sink"),
            )?);

            // Attention LoRA + KV joint.
            // Attention projections — antirez recipe ships these as Q8_0
            // (8.5 bpw, 2× precision of MQ4G256). Dispatcher in
            // forward.rs branches on GpuTensor.dtype: Raw → MQ4 prerotated,
            // Q8_0 → gemv_q8_0 with plain RMSNorm'd input.
            layer.wq_a = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.attn.wq_a.weight"),
            )?);
            layer.wq_b = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.attn.wq_b.weight"),
            )?);
            layer.wkv = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.attn.wkv.weight"),
            )?);
            layer.wo_a = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.attn.wo_a.weight"),
            )?);
            layer.wo_b = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.attn.wo_b.weight"),
            )?);

            // Main-attention compressor — only when ratio > 0. Use the
            // dual-dtype helper so `--non-expert-f16` quants land as F32
            // (gemv_f32 path) while default MQ4G256 quants land as Raw
            // (gemv_mq4g256_prerotated path). gemv_auto in forward.rs
            // branches on GpuTensor.dtype to pick the right kernel.
            // Opt-in: keep F16-native parallel copies of the compressor
            // projections for the WMMA GEMM path. Doubles compressor
            // VRAM footprint but unlocks the 26× speedup measured in
            // microbench (gemm_f16_x_f16_wmma vs gemm_f32_register_tiled).
            let comp_f16_wmma = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_COMP_F16_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if layer.compress_ratio > 0 {
                let compressor_wkv_name = format!("layers.{l}.attn.compressor.wkv.weight");
                let compressor_wgate_name = format!("layers.{l}.attn.compressor.wgate.weight");
                layer.compressor_wkv =
                    Some(Self::upload_quant_or_f16(hfq, gpu, &compressor_wkv_name)?);
                layer.compressor_wgate =
                    Some(Self::upload_quant_or_f16(hfq, gpu, &compressor_wgate_name)?);
                // REAP overlays may replace either compressor projection with
                // a quantized format. Only retain the parallel F16 WMMA copy
                // when the overlay-resolved tensor is actually F16; otherwise
                // the regular dtype-aware GEMV/GEMM path below is authoritative.
                if comp_f16_wmma
                    && hfq
                        .find_tensor_info(&compressor_wkv_name)
                        .is_some_and(|info| info.quant_type == 1)
                {
                    layer.compressor_wkv_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &compressor_wkv_name,
                    )?);
                }
                if comp_f16_wmma
                    && hfq
                        .find_tensor_info(&compressor_wgate_name)
                        .is_some_and(|info| info.quant_type == 1)
                {
                    layer.compressor_wgate_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &compressor_wgate_name,
                    )?);
                }
                layer.compressor_norm = Some(Self::upload_global_f16_as_f32(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.compressor.norm.weight"),
                )?);
                // APE (Absolute Position Encoding) is added to the per-step
                // score in `compressor_forward_impl` via `add_inplace_f32`.
                // Convert F16 → F32 once at load so the per-step add is a
                // plain F32-F32 op. Shape is [ratio, proj_dim] — tiny
                // (max ratio=128 × proj_dim=1024 = 128k F32 = 512KB/layer).
                layer.compressor_ape = Some(Self::upload_global_f16_as_f32(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.compressor.ape"),
                )?);
            }

            // Indexer sub-module — only on layers with compress_ratio == 4.
            if layer.compress_ratio == 4 {
                let indexer_compressor_wkv_name =
                    format!("layers.{l}.attn.indexer.compressor.wkv.weight");
                let indexer_compressor_wgate_name =
                    format!("layers.{l}.attn.indexer.compressor.wgate.weight");
                layer.indexer_wq_b = Some(Self::upload_quant_or_f16(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.indexer.wq_b.weight"),
                )?);
                layer.indexer_weights_proj = Some(Self::upload_quant_or_f16(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.indexer.weights_proj.weight"),
                )?);
                layer.indexer_compressor_wkv = Some(Self::upload_quant_or_f16(
                    hfq,
                    gpu,
                    &indexer_compressor_wkv_name,
                )?);
                layer.indexer_compressor_wgate = Some(Self::upload_quant_or_f16(
                    hfq,
                    gpu,
                    &indexer_compressor_wgate_name,
                )?);
                if comp_f16_wmma
                    && hfq
                        .find_tensor_info(&indexer_compressor_wkv_name)
                        .is_some_and(|info| info.quant_type == 1)
                {
                    layer.indexer_compressor_wkv_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &indexer_compressor_wkv_name,
                    )?);
                }
                if comp_f16_wmma
                    && hfq
                        .find_tensor_info(&indexer_compressor_wgate_name)
                        .is_some_and(|info| info.quant_type == 1)
                {
                    layer.indexer_compressor_wgate_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &indexer_compressor_wgate_name,
                    )?);
                }
                layer.indexer_compressor_norm = Some(Self::upload_global_f16_as_f32(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.indexer.compressor.norm.weight"),
                )?);
                // Same F16 → F32 conversion as the main-attn APE; see
                // comment on `compressor_ape` above for rationale.
                layer.indexer_compressor_ape = Some(Self::upload_global_f16_as_f32(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.indexer.compressor.ape"),
                )?);
            }

            // Hyper-Connections (F16 small matrices).
            layer.hc_attn_base = Some(Self::upload_global_raw(
                hfq,
                gpu,
                &format!("layers.{l}.hc_attn_base"),
            )?);
            layer.hc_attn_fn = Some(Self::upload_global_raw(
                hfq,
                gpu,
                &format!("layers.{l}.hc_attn_fn"),
            )?);
            layer.hc_attn_scale = Some(Self::upload_global_raw(
                hfq,
                gpu,
                &format!("layers.{l}.hc_attn_scale"),
            )?);
            layer.hc_ffn_base = Some(Self::upload_global_raw(
                hfq,
                gpu,
                &format!("layers.{l}.hc_ffn_base"),
            )?);
            layer.hc_ffn_fn = Some(Self::upload_global_raw(
                hfq,
                gpu,
                &format!("layers.{l}.hc_ffn_fn"),
            )?);
            layer.hc_ffn_scale = Some(Self::upload_global_raw(
                hfq,
                gpu,
                &format!("layers.{l}.hc_ffn_scale"),
            )?);

            // FFN router. MUST use upload_quant_or_f16 (not upload_global_raw)
            // so the dtype tag matches the quant_type — same trap as head.weight.
            // With upload_global_raw, dtype=Raw always, and gemv_auto (in
            // moe_route) falls through to gemv_mq4g256_prerotated regardless
            // of actual quant. For Q8F16 routers (deepseek4-q8-mtp) that meant
            // reading Q8 bytes as MQ4 blocks → NaN logits at layer 3+
            // (the first non-hash layer that runs moe_route).
            // Per-layer keep slice (None ⇒ keep-all / no plan ⇒ full upload).
            let ep = cfg.reap_keep.as_ref().map(|r| r.expert_plan(l));
            let gate_name = format!("layers.{l}.ffn.gate.weight");
            layer.gate_weight = Some(match ep.as_ref().and_then(|p| p.keep()) {
                Some(keep) => Self::upload_quant_or_f16_keep(hfq, gpu, &gate_name, keep)?,
                None => Self::upload_quant_or_f16(hfq, gpu, &gate_name)?,
            });
            if l >= cfg.num_hash_layers {
                // Store F32 on GPU (was F16 on disk) so the bias can
                // either be added on-device or downloaded once for CPU
                // topk. Also cache host-side for the CPU-routing path.
                let bias_name = format!("layers.{l}.ffn.gate.bias");
                let bias_gpu = match ep.as_ref().and_then(|p| p.keep()) {
                    Some(keep) => Self::upload_global_f16_as_f32_keep(hfq, gpu, &bias_name, keep)?,
                    None => Self::upload_global_f16_as_f32(hfq, gpu, &bias_name)?,
                };
                layer.gate_bias_host = gpu
                    .download_f32(&bias_gpu)
                    .map_err(|e| format!("d2h gate_bias l{l}: {e:?}"))?;
                layer.gate_bias = Some(bias_gpu);
            } else {
                // Hash-routed layer: read `tid2eid` lookup table (I32 raw
                // bytes) if present. Pre-FP4-fix HFQs skipped this tensor
                // at quant time, in which case forward falls back to
                // shared-only on hash layers (current default behaviour).
                let tid_name = format!("layers.{l}.ffn.gate.tid2eid");
                if let Some((info, file_bytes)) = hfq.tensor_data_pread(&tid_name) {
                    // Under a REAP keep-map the hash table is REMAPPED (pruned
                    // experts redirected to kept ones, in 0..kept slot space);
                    // read the sidecar table instead of the file's original.
                    let bytes: Vec<u8> = match cfg.reap_keep.as_ref() {
                        Some(plan) => {
                            let p = crate::deepseek4::Ds4ReapHook
                                .sidecar_path(plan, &format!("tid2eid_l{l}.i32"));
                            std::fs::read(&p)
                                .map_err(|e| format!("deepseek4: REAP tid2eid read {p:?}: {e}"))?
                        }
                        None => file_bytes.to_vec(),
                    };
                    if bytes.len() % 4 == 0 {
                        let vals: Vec<u32> = bytes
                            .chunks_exact(4)
                            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                            .collect();
                        let expected = info.shape.iter().product::<u32>() as usize;
                        if vals.len() == expected {
                            // Upload to device for the GPU hash-router path.
                            // Reinterpret u32 bytes as raw bytes — keep dtype
                            // as F32 (raw) since the kernel reads `unsigned int*`
                            // and the buffer's bytes are what matters.
                            let shape: Vec<usize> =
                                info.shape.iter().map(|&s| s as usize).collect();
                            match gpu.upload_raw(&bytes, &shape) {
                                Ok(t) => layer.tid2eid_dev = Some(t),
                                Err(e) => eprintln!(
                                    "deepseek4: tid2eid l{l} upload failed: {e:?}; \
                                    fall back to host gather"
                                ),
                            }
                            layer.tid2eid_host = vals;
                        } else {
                            eprintln!(
                                "deepseek4: tid2eid l{l} size mismatch \
                                ({} vs expected {}); ignoring",
                                vals.len(),
                                expected
                            );
                        }
                    }
                }
            }

            // Shared expert.
            // Shared experts — antirez Q8_0 path (same dispatch logic).
            layer.shared_w1 = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.ffn.shared_experts.w1.weight"),
            )?);
            layer.shared_w2 = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.ffn.shared_experts.w2.weight"),
            )?);
            layer.shared_w3 = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.ffn.shared_experts.w3.weight"),
            )?);
        }

        // ── MTP layer (Multi-Token Prediction head, DeepSeek V3 style) ─
        // The MTP layer mirrors a main layer's attention + FFN structure
        // PLUS two input projections (e_proj, h_proj) and three extra
        // norms (enorm, hnorm, final norm). It has no compressor and no
        // indexer — its attention is SWA-only like a hash layer.
        //
        // Gated on `mtp.0.norm.weight` being present somewhere. The MTP
        // tensors source from the addon if it was opened above, else from
        // the base HFQ (in-band MTP, e.g. one-shot deepseek4-q8-mtp quants).
        // Files without MTP and no addon leave `mtp_layer = None`.
        let mtp_source: &HfqFile = mtp_addon.as_ref().unwrap_or(&*hfq);
        let mtp_present = mtp_source.find_tensor_info("mtp.0.norm.weight").is_some();
        if mtp_present {
            let load_mtp = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_LOAD_MTP")
                .map(|s| s != "0")
                .unwrap_or(true)
                && cfg.reap_keep.is_none();
            if !load_mtp {
                eprintln!(
                    "deepseek4: skipping MTP upload ({})",
                    if cfg.reap_keep.is_some() {
                        "REAP keep-map active — MTP unused for PPL/KLD and would \
                         need separate keep handling"
                    } else {
                        "HIPFIRE_DEEPSEEK4_LOAD_MTP=0"
                    }
                );
            } else {
                eprintln!(
                    "deepseek4: MTP layer present — uploading from {}.",
                    if mtp_addon.is_some() {
                        "addon HFQ"
                    } else {
                        "base HFQ"
                    }
                );
                let mut mtp = DeepseekV4LayerWeights::new_empty(0);
                // ── Standard layer fields under the `mtp.0.` prefix ──
                // All MTP reads source from `mtp_source` (addon if present, else base).
                mtp.attn_norm = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.attn_norm.weight",
                )?);
                mtp.ffn_norm = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.ffn_norm.weight",
                )?);
                mtp.q_norm = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.q_norm.weight",
                )?);
                mtp.kv_norm = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.kv_norm.weight",
                )?);
                mtp.attn_sink = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.attn_sink",
                )?);

                mtp.wq_a = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.wq_a.weight",
                )?);
                mtp.wq_b = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.wq_b.weight",
                )?);
                mtp.wkv = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.wkv.weight",
                )?);
                mtp.wo_a = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.wo_a.weight",
                )?);
                mtp.wo_b = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.attn.wo_b.weight",
                )?);

                // HC blocks (same shape as main layer).
                mtp.hc_attn_base = Some(Self::upload_global_raw(
                    mtp_source,
                    gpu,
                    "mtp.0.hc_attn_base",
                )?);
                mtp.hc_attn_fn = Some(Self::upload_global_raw(
                    mtp_source,
                    gpu,
                    "mtp.0.hc_attn_fn",
                )?);
                mtp.hc_attn_scale = Some(Self::upload_global_raw(
                    mtp_source,
                    gpu,
                    "mtp.0.hc_attn_scale",
                )?);
                mtp.hc_ffn_base = Some(Self::upload_global_raw(
                    mtp_source,
                    gpu,
                    "mtp.0.hc_ffn_base",
                )?);
                mtp.hc_ffn_fn = Some(Self::upload_global_raw(mtp_source, gpu, "mtp.0.hc_ffn_fn")?);
                mtp.hc_ffn_scale = Some(Self::upload_global_raw(
                    mtp_source,
                    gpu,
                    "mtp.0.hc_ffn_scale",
                )?);

                // FFN router (score-routed; MTP doesn't have hash routing).
                mtp.gate_weight = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.ffn.gate.weight",
                )?);
                let bias_gpu =
                    Self::upload_global_f16_as_f32(mtp_source, gpu, "mtp.0.ffn.gate.bias")?;
                mtp.gate_bias_host = gpu
                    .download_f32(&bias_gpu)
                    .map_err(|e| format!("d2h mtp gate_bias: {e:?}"))?;
                mtp.gate_bias = Some(bias_gpu);

                // Shared expert.
                mtp.shared_w1 = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.ffn.shared_experts.w1.weight",
                )?);
                mtp.shared_w2 = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.ffn.shared_experts.w2.weight",
                )?);
                mtp.shared_w3 = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.ffn.shared_experts.w3.weight",
                )?);

                // ── MTP-specific fields ──
                mtp.mtp_enorm = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.enorm.weight",
                )?);
                mtp.mtp_hnorm = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.hnorm.weight",
                )?);
                mtp.mtp_e_proj = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.e_proj.weight",
                )?);
                mtp.mtp_h_proj = Some(Self::upload_quant_or_f16(
                    mtp_source,
                    gpu,
                    "mtp.0.h_proj.weight",
                )?);
                mtp.mtp_final_norm = Some(Self::upload_global_f16_as_f32(
                    mtp_source,
                    gpu,
                    "mtp.0.norm.weight",
                )?);

                // MTP-specific head-HC matrices (mirrors the main-model globals
                // hc_head_fn / hc_head_base / hc_head_scale). Their presence
                // proves MTP was trained WITH head-HC mix on its lm_head path —
                // the v3 paper's "logits = OutHead @ norm(h_i^k)" should be
                // read with norm(h_i^k) = norm(head_hc_mix(streams)) on DeepSeek V4.
                mtp.mtp_hc_head_fn = Some(Self::upload_global_raw(
                    mtp_source,
                    gpu,
                    "mtp.0.hc_head_fn",
                )?);
                mtp.mtp_hc_head_base = Some(Self::upload_global_raw(
                    mtp_source,
                    gpu,
                    "mtp.0.hc_head_base",
                )?);
                {
                    let (info, bytes) = mtp_source
                        .tensor_data_pread("mtp.0.hc_head_scale")
                        .ok_or_else(|| "mtp.0.hc_head_scale missing".to_string())?;
                    if info.shape != vec![1] {
                        return Err(format!(
                            "mtp.0.hc_head_scale unexpected shape {:?}",
                            info.shape
                        ));
                    }
                    mtp.mtp_hc_head_scale =
                        hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                            bytes[0], bytes[1],
                        ]));
                }

                weights.mtp_layer = Some(mtp);
            }
        }

        // (Mmaps were dropped earlier, right after the host walk —
        // see the comment above `hfq.drop_mmap()` at the top of this
        // function. The previous "Phase B drop here" call is redundant
        // now that every upload helper uses tensor_data_pread, but is
        // left removed to make the lifecycle obvious.)
        //
        // Reclaim the pread reuse buffer's peak allocation before the
        // routed-expert pass. After the dense + MTP pass, pread_buf is
        // sitting at ~560 MB (size of head.weight at Q8F16) but the
        // routed-expert pass only ever reads ~9 MB at a time. On UMA
        // that 560 MB is the difference between fitting and OOM at
        // layer 42 of the 88 GB deepseek4-q8-mtp build.
        hfq.shrink_pread_buf();
        if let Some(ref addon) = mtp_addon {
            addon.shrink_pread_buf();
        }

        if (routed_gpu.is_some() || dense_only) && weights.mtp_layer.is_some() {
            return Err(
                "deepseek4: heterogeneous G2 admits the frozen trunk only; in-band MTP is not allowed"
                    .into(),
            );
        }
        if heterogeneous_fault == Some(DeepseekV4HeterogeneousFault::AfterDenseWeights) {
            return Err("deepseek4: injected heterogeneous failure after dense weights".into());
        }

        // Routed experts: 256 × 3 = 768 tensors per layer ×
        // 43 layers = 33,024 total. Per-expert hipMalloc takes ~10ms
        // (driver overhead) → 5+ min naive. Batch as ONE upload per
        // (layer, projection): 129 uploads total. Opt out with
        // HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0 (default ON; the experts
        // are ~40 GB, but DeepSeek V4 is architecturally MoE so a
        // shared-only run is diagnostic-only).
        // Per-layer gate: skip uploads when partial-MoE budget excludes
        // this layer (forward gracefully falls back to shared-only).
        //
        // Per-layer batched pread + single GPU upload. The pread bypasses
        // mmap entirely (no longer alive after the drop above); each pread
        // is followed by fadvise(DONTNEED) so the kernel reclaims file
        // pages as soon as they're consumed. Host peak per layer ≈
        // stride_w1 × n_exp + stride_w2 × n_exp ≈ 1.2 GB — bounded,
        // well below the pressure threshold.
        if upload_experts {
            for l in 0..weights.layers.len() {
                let upload_this_layer = expert_layer_end.is_none_or(|end| l < end);
                if !upload_this_layer {
                    continue;
                }
                let n_exp = cfg.n_routed_experts;
                let keep = cfg.reap_keep.as_ref().and_then(|r| r.expert_plan(l).keep());
                if let Some(expert_gpu) = routed_gpu.as_deref_mut() {
                    let layer = weights
                        .routed_layer_mut(l)
                        .ok_or_else(|| format!("deepseek4: routed owner missing layer {l}"))?;
                    Self::upload_layer_routed_experts(
                        hfq,
                        expert_gpu,
                        &format!("layers.{l}"),
                        n_exp,
                        layer,
                        shard,
                        keep,
                    )?;
                } else {
                    let layer = &mut weights.layers[l];
                    Self::upload_layer_routed_experts(
                        hfq,
                        gpu,
                        &format!("layers.{l}"),
                        n_exp,
                        layer,
                        shard,
                        keep,
                    )?;
                }
                if heterogeneous_fault == Some(DeepseekV4HeterogeneousFault::AfterRoutedLayer(l)) {
                    return Err(format!(
                        "deepseek4: injected heterogeneous failure after routed layer {l}"
                    ));
                }
            }
        }

        // Routed experts for the MTP layer (same upload logic, gated on
        // both `upload_experts` and the MTP layer existing). Reads from the
        // addon HFQ if present, else from the base (in-band MTP).
        if upload_experts {
            if let Some(mtp) = weights.mtp_layer.as_mut() {
                let mtp_expert_source: &HfqFile = mtp_addon.as_ref().unwrap_or(&*hfq);
                eprintln!(
                    "deepseek4: uploading MTP routed experts from {}.",
                    if mtp_addon.is_some() {
                        "addon HFQ"
                    } else {
                        "base HFQ"
                    }
                );
                if let Some(expert_gpu) = routed_gpu.as_deref_mut() {
                    Self::upload_layer_routed_experts(
                        mtp_expert_source,
                        expert_gpu,
                        "mtp.0",
                        cfg.n_routed_experts,
                        mtp,
                        shard,
                        None, // MTP not loaded under REAP keep-map (see load_mtp guard)
                    )?;
                } else {
                    Self::upload_layer_routed_experts(
                        mtp_expert_source,
                        gpu,
                        "mtp.0",
                        cfg.n_routed_experts,
                        mtp,
                        shard,
                        None, // MTP not loaded under REAP keep-map (see load_mtp guard)
                    )?;
                }
            }
        }

        // ── DSpark 3-stage drafter sidecar discovery ─────────────────────
        // Additive to the single-stage MTP load above. Mirrors the `-mtp`
        // addon resolution but for a `<stem>-dspark.<ext>` sidecar holding the
        // `mtp.{0,1,2}.*` DSpark stages (arch_id=9). Gated by `config.load_dspark`,
        // which the loader sets from the `speculation` selector (`dspark`/`auto`
        // → true, any other mechanism → false) so the 3×MoE sidecar is not paged
        // into VRAM when DSpark won't run. A missing sidecar is a silent no-op
        // (`weights.dspark` stays None).
        if cfg.load_dspark {
            if routed_gpu.is_some() {
                return Err(
                    "deepseek4: DSpark sidecar is not admitted on the G2 heterogeneous base-load route"
                        .into(),
                );
            }
            let base = hfq.path();
            let dspark_path: Option<std::path::PathBuf> =
                match (base.parent(), base.file_stem(), base.extension()) {
                    (Some(parent), Some(file_stem), Some(ext)) => Some(parent.join(format!(
                        "{}-dspark.{}",
                        file_stem.to_string_lossy(),
                        ext.to_string_lossy()
                    ))),
                    _ => None,
                };
            if let Some(p) = dspark_path.filter(|c| c.exists()) {
                eprintln!("deepseek4: opening DSpark sidecar HFQ {p:?}");
                let mut dspark_hfq = HfqFile::open(&p).map_err(|e| {
                    format!("deepseek4: failed to open DSpark sidecar {p:?}: {e:?}")
                })?;
                if cfg.mq2rxt {
                    Self::validate_mq2rxt_dspark_sidecar(&dspark_hfq)?;
                    eprintln!(
                        "deepseek4: MQ2RXT DSpark sidecar identity verified \
                         (24 MQ4 dense tensors; routed experts qt=19)"
                    );
                } else if cfg.mq2r {
                    Self::validate_mq2r_dspark_sidecar(&dspark_hfq)?;
                    eprintln!(
                        "deepseek4: MQ2R DSpark v1 sidecar identity verified \
                         (target=P3; draft head=trunk E8 B4)"
                    );
                }
                dspark_hfq.drop_mmap();
                weights.dspark = Self::load_dspark(&dspark_hfq, gpu, cfg)?;
            }
        }

        Ok(weights)
    }

    /// Load the dense per-stage tensors of one DSpark stage under `prefix`
    /// (`mtp.{s}`). Mirrors the single-stage MTP dense block but parameterized
    /// on the prefix and WITHOUT the MTP-only enorm/hnorm/e_proj/h_proj (those
    /// are absent on DSpark stages — their layer fields stay None). The
    /// per-stage hc_head / final-norm and the routed experts are loaded by the
    /// caller (`load_dspark`).
    fn load_dspark_stage_dense(
        source: &HfqFile,
        gpu: &mut Gpu,
        prefix: &str,
        layer: &mut DeepseekV4LayerWeights,
    ) -> Result<(), String> {
        // Norms (F16 on disk → F32 on GPU).
        layer.attn_norm = Some(Self::upload_global_f16_as_f32(
            source,
            gpu,
            &format!("{prefix}.attn_norm.weight"),
        )?);
        layer.ffn_norm = Some(Self::upload_global_f16_as_f32(
            source,
            gpu,
            &format!("{prefix}.ffn_norm.weight"),
        )?);
        layer.q_norm = Some(Self::upload_global_f16_as_f32(
            source,
            gpu,
            &format!("{prefix}.attn.q_norm.weight"),
        )?);
        layer.kv_norm = Some(Self::upload_global_f16_as_f32(
            source,
            gpu,
            &format!("{prefix}.attn.kv_norm.weight"),
        )?);
        layer.attn_sink = Some(Self::upload_global_f16_as_f32(
            source,
            gpu,
            &format!("{prefix}.attn.attn_sink"),
        )?);

        // Attention LoRA + KV joint (MQ-family / Q8F16 / F16).
        layer.wq_a = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.attn.wq_a.weight"),
        )?);
        layer.wq_b = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.attn.wq_b.weight"),
        )?);
        layer.wkv = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.attn.wkv.weight"),
        )?);
        layer.wo_a = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.attn.wo_a.weight"),
        )?);
        layer.wo_b = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.attn.wo_b.weight"),
        )?);

        // HC blocks (raw F16 matrices for the hc_* kernels).
        layer.hc_attn_base = Some(Self::upload_global_raw(
            source,
            gpu,
            &format!("{prefix}.hc_attn_base"),
        )?);
        layer.hc_attn_fn = Some(Self::upload_global_raw(
            source,
            gpu,
            &format!("{prefix}.hc_attn_fn"),
        )?);
        layer.hc_attn_scale = Some(Self::upload_global_raw(
            source,
            gpu,
            &format!("{prefix}.hc_attn_scale"),
        )?);
        layer.hc_ffn_base = Some(Self::upload_global_raw(
            source,
            gpu,
            &format!("{prefix}.hc_ffn_base"),
        )?);
        layer.hc_ffn_fn = Some(Self::upload_global_raw(
            source,
            gpu,
            &format!("{prefix}.hc_ffn_fn"),
        )?);
        layer.hc_ffn_scale = Some(Self::upload_global_raw(
            source,
            gpu,
            &format!("{prefix}.hc_ffn_scale"),
        )?);

        // FFN router (score-routed; gate weight + bias, bias host-cached).
        layer.gate_weight = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.ffn.gate.weight"),
        )?);
        let bias_gpu =
            Self::upload_global_f16_as_f32(source, gpu, &format!("{prefix}.ffn.gate.bias"))?;
        layer.gate_bias_host = gpu
            .download_f32(&bias_gpu)
            .map_err(|e| format!("d2h dspark {prefix} gate_bias: {e:?}"))?;
        layer.gate_bias = Some(bias_gpu);

        // Shared expert.
        layer.shared_w1 = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.ffn.shared_experts.w1.weight"),
        )?);
        layer.shared_w2 = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.ffn.shared_experts.w2.weight"),
        )?);
        layer.shared_w3 = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("{prefix}.ffn.shared_experts.w3.weight"),
        )?);

        Ok(())
    }

    /// Load the full DSpark 3-stage drafter from an already-opened sidecar
    /// `source`. Returns `None` when the sidecar carries no DSpark config
    /// (`DsparkConfig::from_metadata_json` absent). Probes the stage count by
    /// walking `mtp.{N}.attn_norm.weight` until absent, builds one
    /// `DeepseekV4LayerWeights` per stage (dense + routed experts), and on the
    /// LAST stage additionally loads the head-HC mix + final norm. The DSpark
    /// globals (`main_proj`/`main_norm` from stage 0, `markov_*` /
    /// `confidence_proj` from the last stage) are loaded after the stages.
    pub fn load_dspark(
        source: &HfqFile,
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
    ) -> Result<Option<DsparkWeights>, String> {
        let dspark_cfg = match DsparkConfig::from_metadata_json(&source.metadata_json) {
            Some(c) => c,
            None => return Ok(None),
        };

        // Guard: every target layer must index a real trunk layer. An
        // out-of-range id never matches in the capture hook
        // (`forward_prefill_batch_chunk`), so its capture slot stays
        // stale/zero, `main_hidden` degrades, and draft quality silently
        // collapses (acceptance craters; output stays greedy-correct). Fail
        // loud at load instead of shipping a lobotomized drafter.
        if let Some(&bad) = dspark_cfg
            .target_layer_ids
            .iter()
            .find(|&&l| l >= cfg.num_hidden_layers)
        {
            return Err(format!(
                "deepseek4: DSpark target_layer_id {bad} >= num_hidden_layers {} (sidecar/trunk mismatch)",
                cfg.num_hidden_layers
            ));
        }

        // Probe stage count: `mtp.{N}.attn_norm.weight` until absent.
        let mut n_stages = 0usize;
        while source
            .find_tensor_info(&format!("mtp.{n_stages}.attn_norm.weight"))
            .is_some()
        {
            n_stages += 1;
        }
        if n_stages == 0 {
            return Err("deepseek4: DSpark config present but no mtp.{N} stages found".into());
        }
        eprintln!("deepseek4: DSpark drafter present — uploading {n_stages} stages");

        let last = n_stages - 1;
        let mut stages: Vec<DeepseekV4LayerWeights> = Vec::with_capacity(n_stages);
        for s in 0..n_stages {
            let prefix = format!("mtp.{s}");
            let mut layer = DeepseekV4LayerWeights::new_empty(0);
            Self::load_dspark_stage_dense(source, gpu, &prefix, &mut layer)?;
            Self::upload_layer_routed_experts(
                source,
                gpu,
                &prefix,
                cfg.n_routed_experts,
                &mut layer,
                None,
                None,
            )?;
            if s == last {
                // Last stage carries the head-HC mix + final norm.
                layer.mtp_hc_head_fn = Some(Self::upload_global_raw(
                    source,
                    gpu,
                    &format!("{prefix}.hc_head_fn"),
                )?);
                layer.mtp_hc_head_base = Some(Self::upload_global_raw(
                    source,
                    gpu,
                    &format!("{prefix}.hc_head_base"),
                )?);
                {
                    let scale_name = format!("{prefix}.hc_head_scale");
                    let (info, bytes) = source
                        .tensor_data_pread(&scale_name)
                        .ok_or_else(|| format!("deepseek4: {scale_name} missing"))?;
                    if info.shape != vec![1] {
                        return Err(format!(
                            "deepseek4: {scale_name} unexpected shape {:?}",
                            info.shape
                        ));
                    }
                    layer.mtp_hc_head_scale =
                        hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                            bytes[0], bytes[1],
                        ]));
                }
                layer.mtp_final_norm = Some(Self::upload_global_f16_as_f32(
                    source,
                    gpu,
                    &format!("{prefix}.norm.weight"),
                )?);
            }
            stages.push(layer);
        }

        // DSpark globals. main_proj/main_norm live on stage 0; the Markov
        // head + confidence head live on the last stage.
        let main_proj = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            "mtp.0.main_proj.weight",
        )?);
        let main_norm = Some(Self::upload_global_f16_as_f32(
            source,
            gpu,
            "mtp.0.main_norm.weight",
        )?);
        let markov_w1 = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("mtp.{last}.markov_head.markov_w1.weight"),
        )?);
        let markov_w2 = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("mtp.{last}.markov_head.markov_w2.weight"),
        )?);
        let confidence_proj = Some(Self::upload_quant_or_f16(
            source,
            gpu,
            &format!("mtp.{last}.confidence_head.proj.weight"),
        )?);
        let draft_head = if source.find_tensor_info("draft_head.weight").is_some() {
            eprintln!(
                "deepseek4: DSpark sidecar draft_head.weight present — \
                 using it for draft logits only"
            );
            Some(Self::upload_quant_or_f16(source, gpu, "draft_head.weight")?)
        } else {
            None
        };

        Ok(Some(DsparkWeights {
            cfg: dspark_cfg,
            stages,
            main_proj,
            main_norm,
            markov_w1,
            markov_w2,
            confidence_proj,
            draft_head,
        }))
    }
}

// ── ModelSource (safetensors) load helpers ──────────────────────

impl DeepseekV4 {
    /// Determine whether a tensor's bytes represent F16 values or a
    /// quantized format by comparing the byte count against the
    /// expected sizes. Returns `(is_f16, is_q8_0)`.
    fn classify_tensor_bytes(bytes: &[u8], numel: usize, dtype: &str) -> (bool, bool) {
        // BF16 has 2 bytes/element just like F16, so explicitly exclude it
        // from the heuristic — the caller already knows the dtype.
        if dtype == "BF16" {
            return (false, false);
        }
        let is_f16 = bytes.len() == numel * 2;
        // Q8_0: 34 bytes per block of 32 elements:
        //   [f16 scale (2 bytes)] [32 × i8 (32 bytes)]
        let q8_0_expected = ((numel + 31) / 32) * 34;
        let is_q8_0 = !is_f16 && bytes.len() == q8_0_expected;
        (is_f16, is_q8_0)
    }

    /// Upload a tensor verbatim (raw bytes) from ModelSource to GPU.
    /// Mirrors `upload_global_raw` but sources from `&dyn ModelSource`.
    fn upload_global_raw_from_source(
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = source
            .tensor_data(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in source"))?;
        let shape: Vec<usize> = info.shape.clone();
        // BF16 and F16 share the same element size, but the GPU only
        // understands F16.  Decode BF16 → F16 on the host first.
        let upload_bytes = if info.dtype == "BF16" {
            bf16_bytes_to_f16(bytes)
        } else {
            bytes.to_vec()
        };
        gpu.upload_raw(&upload_bytes, &shape)
            .map_err(|e| format!("deepseek4: upload '{name}' failed: {e:?}"))
    }

    /// Upload a weight tensor, classifying it as F16, Q8_0, or Raw
    /// (MQ4-family) based on byte-count heuristics. Mirrors
    /// `upload_quant_or_f16` but sources from `&dyn ModelSource`.
    fn upload_quant_or_f16_from_source(
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = source
            .tensor_data(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in source"))?;
        let shape: Vec<usize> = info.shape.clone();
        let numel: usize = shape.iter().product();
        let (is_f16, is_q8_0) = Self::classify_tensor_bytes(bytes, numel, info.dtype.as_str());

        if is_f16 {
            if bytes.len() != numel * 2 {
                return Err(format!(
                    "deepseek4: '{name}' appears F16 but byte size {} != 2 × {numel}",
                    bytes.len()
                ));
            }
            let mut t = gpu
                .upload_raw(bytes, &shape)
                .map_err(|e| format!("deepseek4: upload f16-native '{name}' failed: {e:?}"))?;
            t.dtype = rdna_compute::DType::F16;
            return Ok(t);
        }

        let mut t = gpu
            .upload_raw(bytes, &shape)
            .map_err(|e| format!("deepseek4: upload '{name}' failed: {e:?}"))?;
        if is_q8_0 {
            t.dtype = rdna_compute::DType::Q8_0;
        }
        Ok(t)
    }

    /// Upload an F16-on-disk tensor as F32 on GPU. Mirrors
    /// `upload_global_f16_as_f32` but sources from `&dyn ModelSource`.
    fn upload_global_f16_as_f32_from_source(
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = source
            .tensor_data(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in source"))?;
        let shape: Vec<usize> = info.shape.clone();
        let n: usize = shape.iter().product();
        if bytes.len() != n * 2 {
            return Err(format!(
                "deepseek4: '{name}' expected F16 bytes ({} = 2 × {}), got {}",
                n * 2,
                n,
                bytes.len()
            ));
        }
        let f32_vals: Vec<f32> = (0..n)
            .map(|i| {
                let lo = bytes[i * 2];
                let hi = bytes[i * 2 + 1];
                hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([lo, hi]))
            })
            .collect();
        gpu.upload_f32(&f32_vals, &shape)
            .map_err(|e| format!("deepseek4: upload f16→f32 '{name}' failed: {e:?}"))
    }

    /// Upload an F16-on-disk tensor as F16 bytes on GPU (no conversion).
    /// Mirrors `upload_quant_as_f16_native` but sources from
    /// `&dyn ModelSource`. Errors if the tensor isn't F16.
    fn upload_quant_as_f16_native_from_source(
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = source
            .tensor_data(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in source"))?;
        let shape: Vec<usize> = info.shape.clone();
        let numel: usize = shape.iter().product();
        let (is_f16, _) = Self::classify_tensor_bytes(bytes, numel, info.dtype.as_str());
        if !is_f16 {
            return Err(format!(
                "deepseek4: '{name}' not F16 ({} bytes for {numel} elems); cannot upload as F16 native",
                bytes.len()
            ));
        }
        if bytes.len() != numel * 2 {
            return Err(format!(
                "deepseek4: '{name}' marked F16 but byte size {} != 2 × {numel}",
                bytes.len()
            ));
        }
        let mut t = gpu
            .upload_raw(bytes, &shape)
            .map_err(|e| format!("deepseek4: upload f16-native '{name}' failed: {e:?}"))?;
        t.dtype = rdna_compute::DType::F16;
        Ok(t)
    }

    /// Upload routed-expert blobs for one layer from a ModelSource.
    /// Mirrors `upload_layer_routed_experts` but sources from
    /// `&dyn ModelSource`.
    fn upload_layer_routed_experts_from_source(
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        prefix: &str,
        n_exp: usize,
        layer: &mut DeepseekV4LayerWeights,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
    ) -> Result<(), String> {
        // EP shard: precompute owned set + compact-slot mapping.
        let owns = |e: usize| {
            shard
                .map(|(s, rank)| s.owns_expert(rank, e))
                .unwrap_or(true)
        };
        let mut local_of_global = vec![usize::MAX; n_exp];
        let mut n_owned = 0usize;
        for e in 0..n_exp {
            if owns(e) {
                local_of_global[e] = n_owned;
                n_owned += 1;
            }
        }
        if n_owned == 0 {
            return Err(format!("deepseek4: {prefix} shard rank owns no experts"));
        }

        // w2 (down): read each expert, pack ONLY owned into blob.
        {
            let name0 = format!("{prefix}.ffn.experts.0.w2.weight");
            let (info0, _b0) = source
                .tensor_data(&name0)
                .ok_or_else(|| format!("deepseek4: missing {name0}"))?;
            // Guard: the indexed-MoE forward has no float-expert path — it
            // reinterprets the packed expert blob as quant blocks. A raw-HF
            // safetensors checkpoint ships bf16/f16/f32 experts, which would be
            // misread → silent garbage. Refuse cleanly (quantized experts only),
            // mirroring the lfm2moe Dir guard. (This Dir arm is otherwise
            // unvalidated — no deepseek_v4 checkpoint was available locally.)
            if matches!(info0.dtype.as_str(), "BF16" | "F16" | "F32") {
                return Err(format!(
                    "deepseek4: routed experts at {prefix} are raw float ({}); the \
                     indexed-MoE forward requires quantized experts. Quantize the \
                     checkpoint first or load the prebuilt HFQ.",
                    info0.dtype
                ));
            }
            let stride = info0.data_size;
            let shape0: Vec<usize> = info0.shape.clone();

            let mut blob = Vec::with_capacity(stride * n_owned);
            for e in 0..n_exp {
                if !owns(e) {
                    continue;
                }
                let name = format!("{prefix}.ffn.experts.{e}.w2.weight");
                let (info, bytes) = source
                    .tensor_data(&name)
                    .ok_or_else(|| format!("deepseek4: missing {name}"))?;
                if info.data_size != stride {
                    return Err(format!(
                        "deepseek4: {name} size {} != stride {}",
                        info.data_size, stride
                    ));
                }
                blob.extend_from_slice(bytes);
            }
            let mut blob_shape = vec![n_owned];
            blob_shape.extend_from_slice(&shape0);
            let blob_tensor = gpu
                .upload_raw(&blob, &blob_shape)
                .map_err(|e| format!("deepseek4: upload blob {prefix}.w2: {e:?}"))?;
            drop(blob);
            let base_ptr = blob_tensor.buf.as_ptr() as u64;
            let ptrs: Vec<u64> = (0..n_exp)
                .map(|e| {
                    if owns(e) {
                        base_ptr + (local_of_global[e] * stride) as u64
                    } else {
                        base_ptr
                    }
                })
                .collect();
            let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
            let ptr_tensor = gpu
                .alloc_tensor(&[2 * n_exp], rdna_compute::DType::F32)
                .map_err(|e| format!("deepseek4: alloc ptr table {prefix}.w2: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&ptr_tensor.buf, &ptr_bytes)
                .map_err(|e| format!("deepseek4: copy ptr table {prefix}.w2: {e:?}"))?;
            layer.expert_w2_blob = Some(blob_tensor);
            layer.expert_w2_ptrs = Some(ptr_tensor);
            layer.expert_w2_stride = stride;
        }

        // gate_up (combined w1 ‖ w3).
        {
            let w1_0 = format!("{prefix}.ffn.experts.0.w1.weight");
            let w3_0 = format!("{prefix}.ffn.experts.0.w3.weight");
            let (w1_info0, _b1) = source
                .tensor_data(&w1_0)
                .ok_or_else(|| format!("deepseek4: missing {w1_0}"))?;
            let stride_w1 = w1_info0.data_size;
            let (w3_info0, _b3) = source
                .tensor_data(&w3_0)
                .ok_or_else(|| format!("deepseek4: missing {w3_0}"))?;
            let stride_w3 = w3_info0.data_size;
            if stride_w1 != stride_w3 {
                return Err(format!(
                    "deepseek4: {prefix} w1/w3 stride mismatch: w1={} w3={}",
                    stride_w1, stride_w3
                ));
            }
            let combined_stride = stride_w1 + stride_w3;
            let mut combined = Vec::with_capacity(combined_stride * n_owned);
            for e in 0..n_exp {
                if !owns(e) {
                    continue;
                }
                let w1_name = format!("{prefix}.ffn.experts.{e}.w1.weight");
                {
                    let (_, w1_bytes) = source
                        .tensor_data(&w1_name)
                        .ok_or_else(|| format!("deepseek4: missing {w1_name}"))?;
                    combined.extend_from_slice(w1_bytes);
                }
                let w3_name = format!("{prefix}.ffn.experts.{e}.w3.weight");
                {
                    let (_, w3_bytes) = source
                        .tensor_data(&w3_name)
                        .ok_or_else(|| format!("deepseek4: missing {w3_name}"))?;
                    combined.extend_from_slice(w3_bytes);
                }
            }
            let combined_tensor = gpu
                .upload_raw(&combined, &[n_owned, combined_stride])
                .map_err(|e| format!("deepseek4: upload gate_up {prefix}: {e:?}"))?;
            drop(combined);
            let base_ptr = combined_tensor.buf.as_ptr() as u64;
            let dummy_gu = if shard.is_some() && n_owned < n_exp {
                let z = gpu
                    .zeros(&[combined_stride / 4], rdna_compute::DType::F32)
                    .map_err(|e| format!("deepseek4: {prefix} zero gate_up dummy: {e:?}"))?;
                let p = z.buf.as_ptr() as u64;
                std::mem::forget(z);
                p
            } else {
                base_ptr
            };
            let ptrs: Vec<u64> = (0..n_exp)
                .map(|e| {
                    if owns(e) {
                        base_ptr + (local_of_global[e] * combined_stride) as u64
                    } else {
                        dummy_gu
                    }
                })
                .collect();
            let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
            let ptr_tensor = gpu
                .alloc_tensor(&[2 * n_exp], rdna_compute::DType::F32)
                .map_err(|e| format!("deepseek4: alloc gate_up ptr table {prefix}: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&ptr_tensor.buf, &ptr_bytes)
                .map_err(|e| format!("deepseek4: copy gate_up ptr table {prefix}: {e:?}"))?;
            layer.expert_gate_up_blob = Some(combined_tensor);
            layer.expert_gate_up_ptrs = Some(ptr_tensor);
            layer.expert_gate_up_stride = combined_stride;
        }
        Ok(())
    }
}

// ── Top-level safetensors load entry point ──────────────────────

impl DeepseekV4 {
    /// Load model weights from a `&dyn ModelSource` (safetensors or HFQ
    /// wrapper). Mirrors `load_weights_inner` but reads tensor data via
    /// `ModelSource::tensor_data()` instead of `HfqFile::tensor_data_pread()`.
    ///
    /// Tensor names match those used in the HFQ path (the safetensors
    /// created by `hipfire-quantize` use the same naming convention).
    /// Quantization format is inferred from byte counts (F16 vs Q8_0 vs
    /// MQ4-family) matching the HFQ byte layout.
    ///
    /// Only `shard = None` is currently exposed — EP-shard-aware loading
    /// from safetensors is a future extension when multi-GPU deepseek4
    /// is brought up.
    pub fn load_weights_from_safetensors(
        source: &dyn ModelSource,
        cfg: &DeepseekV4Config,
        gpu: &mut Gpu,
    ) -> Result<DeepseekV4Weights, String> {
        let upload_experts = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS")
            .ok()
            .as_deref()
            != Some("0");
        let expert_layer_end: Option<usize> =
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END")
                .ok()
                .and_then(|s| s.parse().ok());
        let comp_f16_wmma = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_COMP_F16_WMMA")
            .map(|s| s != "0")
            .unwrap_or(true);

        // Build empty weight scaffold from config.
        let n_layers = cfg.num_hidden_layers;
        let mut layers: Vec<DeepseekV4LayerWeights> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ratio = *cfg.compress_ratios.get(l).unwrap_or(&0);
            layers.push(DeepseekV4LayerWeights::new_empty(ratio));
        }
        let mut weights = DeepseekV4Weights {
            // Safetensors loads do not pass the frozen HFQ tensor-policy
            // verifier, so they cannot acquire a native MQ2R backend.
            mq2r_backend: Mq2rBackend::Portable,
            token_embd: None,
            output_norm: None,
            head: None,
            hc_head_fn: None,
            hc_head_base: None,
            hc_head_scale: 1.0,
            layers,
            mtp_layer: None,
            dspark: None,
            _scaffold: (),
        };

        // ── Globals ────────────────────────────────────────────────────
        weights.token_embd = Some(Self::upload_global_raw_from_source(
            source,
            gpu,
            "embed.weight",
        )?);
        weights.output_norm = Some(Self::upload_global_f16_as_f32_from_source(
            source,
            gpu,
            "norm.weight",
        )?);
        weights.head = Some(Self::upload_quant_or_f16_from_source(
            source,
            gpu,
            "head.weight",
        )?);

        weights.hc_head_fn = Some(Self::upload_global_raw_from_source(
            source,
            gpu,
            "hc_head_fn",
        )?);
        weights.hc_head_base = Some(Self::upload_global_raw_from_source(
            source,
            gpu,
            "hc_head_base",
        )?);
        {
            let (info, bytes) = source
                .tensor_data("hc_head_scale")
                .ok_or_else(|| "deepseek4: hc_head_scale missing in source".to_string())?;
            if info.shape != vec![1] {
                return Err(format!(
                    "deepseek4: hc_head_scale unexpected shape {:?}",
                    info.shape
                ));
            }
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            let scale = if info.dtype == "BF16" {
                bf16_to_f32(raw)
            } else {
                hipfire_runtime::llama::f16_to_f32(raw)
            };
            weights.hc_head_scale = scale;
        }

        // ── Per-layer ──────────────────────────────────────────────────
        for (l, layer) in weights.layers.iter_mut().enumerate() {
            // Norms (F16 on disk → F32 on GPU).
            layer.attn_norm = Some(Self::upload_global_f16_as_f32_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn_norm.weight"),
            )?);
            layer.ffn_norm = Some(Self::upload_global_f16_as_f32_from_source(
                source,
                gpu,
                &format!("layers.{l}.ffn_norm.weight"),
            )?);
            layer.q_norm = Some(Self::upload_global_f16_as_f32_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.q_norm.weight"),
            )?);
            layer.kv_norm = Some(Self::upload_global_f16_as_f32_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.kv_norm.weight"),
            )?);
            layer.attn_sink = Some(Self::upload_global_f16_as_f32_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.attn_sink"),
            )?);

            // Attention LoRA + KV joint.
            layer.wq_a = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.wq_a.weight"),
            )?);
            layer.wq_b = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.wq_b.weight"),
            )?);
            layer.wkv = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.wkv.weight"),
            )?);
            layer.wo_a = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.wo_a.weight"),
            )?);
            layer.wo_b = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.attn.wo_b.weight"),
            )?);

            // Main-attention compressor — only when ratio > 0.
            if layer.compress_ratio > 0 {
                layer.compressor_wkv = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.compressor.wkv.weight"),
                )?);
                layer.compressor_wgate = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.compressor.wgate.weight"),
                )?);
                if comp_f16_wmma {
                    layer.compressor_wkv_f16 = Some(Self::upload_quant_as_f16_native_from_source(
                        source,
                        gpu,
                        &format!("layers.{l}.attn.compressor.wkv.weight"),
                    )?);
                    layer.compressor_wgate_f16 =
                        Some(Self::upload_quant_as_f16_native_from_source(
                            source,
                            gpu,
                            &format!("layers.{l}.attn.compressor.wgate.weight"),
                        )?);
                }
                layer.compressor_norm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.compressor.norm.weight"),
                )?);
                layer.compressor_ape = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.compressor.ape"),
                )?);
            }

            // Indexer sub-module — only on layers with compress_ratio == 4.
            if layer.compress_ratio == 4 {
                layer.indexer_wq_b = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.indexer.wq_b.weight"),
                )?);
                layer.indexer_weights_proj = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.indexer.weights_proj.weight"),
                )?);
                layer.indexer_compressor_wkv = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.indexer.compressor.wkv.weight"),
                )?);
                layer.indexer_compressor_wgate = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.indexer.compressor.wgate.weight"),
                )?);
                if comp_f16_wmma {
                    layer.indexer_compressor_wkv_f16 =
                        Some(Self::upload_quant_as_f16_native_from_source(
                            source,
                            gpu,
                            &format!("layers.{l}.attn.indexer.compressor.wkv.weight"),
                        )?);
                    layer.indexer_compressor_wgate_f16 =
                        Some(Self::upload_quant_as_f16_native_from_source(
                            source,
                            gpu,
                            &format!("layers.{l}.attn.indexer.compressor.wgate.weight"),
                        )?);
                }
                layer.indexer_compressor_norm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.indexer.compressor.norm.weight"),
                )?);
                layer.indexer_compressor_ape = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}.attn.indexer.compressor.ape"),
                )?);
            }

            // Hyper-Connections (F16 small matrices).
            layer.hc_attn_base = Some(Self::upload_global_raw_from_source(
                source,
                gpu,
                &format!("layers.{l}.hc_attn_base"),
            )?);
            layer.hc_attn_fn = Some(Self::upload_global_raw_from_source(
                source,
                gpu,
                &format!("layers.{l}.hc_attn_fn"),
            )?);
            layer.hc_attn_scale = Some(Self::upload_global_raw_from_source(
                source,
                gpu,
                &format!("layers.{l}.hc_attn_scale"),
            )?);
            layer.hc_ffn_base = Some(Self::upload_global_raw_from_source(
                source,
                gpu,
                &format!("layers.{l}.hc_ffn_base"),
            )?);
            layer.hc_ffn_fn = Some(Self::upload_global_raw_from_source(
                source,
                gpu,
                &format!("layers.{l}.hc_ffn_fn"),
            )?);
            layer.hc_ffn_scale = Some(Self::upload_global_raw_from_source(
                source,
                gpu,
                &format!("layers.{l}.hc_ffn_scale"),
            )?);

            // FFN router.
            layer.gate_weight = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.ffn.gate.weight"),
            )?);
            if l >= cfg.num_hash_layers {
                let bias_name = format!("layers.{l}.ffn.gate.bias");
                let bias_gpu = Self::upload_global_f16_as_f32_from_source(source, gpu, &bias_name)?;
                layer.gate_bias_host = gpu
                    .download_f32(&bias_gpu)
                    .map_err(|e| format!("d2h gate_bias l{l}: {e:?}"))?;
                layer.gate_bias = Some(bias_gpu);
            } else {
                // Hash-routed layer: read `tid2eid` lookup table (I32 raw bytes).
                let tid_name = format!("layers.{l}.ffn.gate.tid2eid");
                if let Some((info, bytes)) = source.tensor_data(&tid_name) {
                    if bytes.len() % 4 == 0 {
                        let vals: Vec<u32> = bytes
                            .chunks_exact(4)
                            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                            .collect();
                        let expected = info.shape.iter().product::<usize>();
                        if vals.len() == expected {
                            let shape: Vec<usize> = info.shape.clone();
                            match gpu.upload_raw(bytes, &shape) {
                                Ok(t) => layer.tid2eid_dev = Some(t),
                                Err(e) => eprintln!(
                                    "deepseek4: tid2eid l{l} upload failed: {e:?}; \
                                    fall back to host gather"
                                ),
                            }
                            layer.tid2eid_host = vals;
                        } else {
                            eprintln!(
                                "deepseek4: tid2eid l{l} size mismatch \
                                ({} vs expected {}); ignoring",
                                vals.len(),
                                expected
                            );
                        }
                    }
                }
            }

            // Shared expert.
            layer.shared_w1 = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.ffn.shared_experts.w1.weight"),
            )?);
            layer.shared_w2 = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.ffn.shared_experts.w2.weight"),
            )?);
            layer.shared_w3 = Some(Self::upload_quant_or_f16_from_source(
                source,
                gpu,
                &format!("layers.{l}.ffn.shared_experts.w3.weight"),
            )?);
        }

        // ── MTP layer ─────────────────────────────────────────────────
        // Check if the source has MTP tensors (same naming as HFQ path:
        // `mtp.0.norm.weight` as the canary).
        let mtp_present = source.tensor_info("mtp.0.norm.weight").is_some();
        if mtp_present {
            let load_mtp = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_LOAD_MTP")
                .map(|s| s != "0")
                .unwrap_or(true);
            if !load_mtp {
                eprintln!(
                    "deepseek4: source contains MTP layer but \
                    HIPFIRE_DEEPSEEK4_LOAD_MTP=0 — skipping MTP upload"
                );
            } else {
                eprintln!("deepseek4: MTP layer present — uploading from safetensors source.");
                let mut mtp = DeepseekV4LayerWeights::new_empty(0);

                // Standard layer fields under `mtp.0.` prefix.
                mtp.attn_norm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.attn_norm.weight",
                )?);
                mtp.ffn_norm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.ffn_norm.weight",
                )?);
                mtp.q_norm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.q_norm.weight",
                )?);
                mtp.kv_norm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.kv_norm.weight",
                )?);
                mtp.attn_sink = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.attn_sink",
                )?);

                mtp.wq_a = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.wq_a.weight",
                )?);
                mtp.wq_b = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.wq_b.weight",
                )?);
                mtp.wkv = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.wkv.weight",
                )?);
                mtp.wo_a = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.wo_a.weight",
                )?);
                mtp.wo_b = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.attn.wo_b.weight",
                )?);

                // HC blocks.
                mtp.hc_attn_base = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_attn_base",
                )?);
                mtp.hc_attn_fn = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_attn_fn",
                )?);
                mtp.hc_attn_scale = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_attn_scale",
                )?);
                mtp.hc_ffn_base = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_ffn_base",
                )?);
                mtp.hc_ffn_fn = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_ffn_fn",
                )?);
                mtp.hc_ffn_scale = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_ffn_scale",
                )?);

                // FFN router (score-routed).
                mtp.gate_weight = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.ffn.gate.weight",
                )?);
                let bias_gpu =
                    Self::upload_global_f16_as_f32_from_source(source, gpu, "mtp.0.ffn.gate.bias")?;
                mtp.gate_bias_host = gpu
                    .download_f32(&bias_gpu)
                    .map_err(|e| format!("d2h mtp gate_bias: {e:?}"))?;
                mtp.gate_bias = Some(bias_gpu);

                // Shared expert.
                mtp.shared_w1 = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.ffn.shared_experts.w1.weight",
                )?);
                mtp.shared_w2 = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.ffn.shared_experts.w2.weight",
                )?);
                mtp.shared_w3 = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.ffn.shared_experts.w3.weight",
                )?);

                // MTP-specific fields.
                mtp.mtp_enorm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.enorm.weight",
                )?);
                mtp.mtp_hnorm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.hnorm.weight",
                )?);
                mtp.mtp_e_proj = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.e_proj.weight",
                )?);
                mtp.mtp_h_proj = Some(Self::upload_quant_or_f16_from_source(
                    source,
                    gpu,
                    "mtp.0.h_proj.weight",
                )?);
                mtp.mtp_final_norm = Some(Self::upload_global_f16_as_f32_from_source(
                    source,
                    gpu,
                    "mtp.0.norm.weight",
                )?);

                // MTP-specific head-HC matrices.
                mtp.mtp_hc_head_fn = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_head_fn",
                )?);
                mtp.mtp_hc_head_base = Some(Self::upload_global_raw_from_source(
                    source,
                    gpu,
                    "mtp.0.hc_head_base",
                )?);
                {
                    let (info, bytes) = source
                        .tensor_data("mtp.0.hc_head_scale")
                        .ok_or_else(|| "mtp.0.hc_head_scale missing in source".to_string())?;
                    if info.shape != vec![1] {
                        return Err(format!(
                            "mtp.0.hc_head_scale unexpected shape {:?}",
                            info.shape
                        ));
                    }
                    mtp.mtp_hc_head_scale =
                        hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                            bytes[0], bytes[1],
                        ]));
                }

                weights.mtp_layer = Some(mtp);
            }
        }

        // ── Routed experts ────────────────────────────────────────────
        if upload_experts {
            for (l, layer) in weights.layers.iter_mut().enumerate() {
                let upload_this_layer = expert_layer_end.is_none_or(|end| l < end);
                if !upload_this_layer {
                    continue;
                }
                let n_exp = cfg.n_routed_experts;
                Self::upload_layer_routed_experts_from_source(
                    source,
                    gpu,
                    &format!("layers.{l}"),
                    n_exp,
                    layer,
                    None, // No EP shard in safetensors path yet.
                )?;
            }
        }
        if upload_experts {
            if let Some(mtp) = weights.mtp_layer.as_mut() {
                eprintln!("deepseek4: uploading MTP routed experts from safetensors source.");
                Self::upload_layer_routed_experts_from_source(
                    source,
                    gpu,
                    "mtp.0",
                    cfg.n_routed_experts,
                    mtp,
                    None,
                )?;
            }
        }

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek4_arch_id_is_nine() {
        assert_eq!(DeepseekV4::arch_id(), 9);
        assert_eq!(DeepseekV4::name(), "deepseek4");
    }

    #[test]
    fn dense_hfq_dtype_preserves_mfp4_e8_variants() {
        assert_eq!(dense_hfq_dtype(34), Some(DType::MFP4G32E8));
        assert_eq!(dense_hfq_dtype(35), Some(DType::MFP4G32E8SOA));
        assert_eq!(dense_hfq_dtype(3), Some(DType::Q8_0));
        assert_eq!(dense_hfq_dtype(19), None);
    }
}
