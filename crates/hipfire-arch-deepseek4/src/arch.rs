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

use crate::deepseek4::{
    DeepseekV4Config, DeepseekV4LayerWeights, DeepseekV4State, DeepseekV4Weights,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

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
    ///   - F16 (quant_type=1): decode to F32 on host, upload as F32, set
    ///     GpuTensor.dtype = F32. Forward routes to `gemv_f32` with plain
    ///     (non-FWHT) input.
    ///   - Q8F16 (quant_type=3): upload raw bytes, set GpuTensor.dtype =
    ///     Q8_0. Forward routes to `gemv_q8_0` with plain input.
    ///   - Otherwise (e.g. quant_type=13 MQ4G256 in hypothetical future
    ///     builds; not present in the canonical mq2lloyd file): upload raw
    ///     bytes, dtype stays Raw. Forward routes to
    ///     `gemv_mq4g256_prerotated` with FWHT-rotated input.
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
            t.dtype = rdna_compute::DType::F16;
            return Ok(t);
        }
        let mut t = gpu
            .upload_raw(&bytes, &shape)
            .map_err(|e| format!("deepseek4: upload '{name}' failed: {e:?}"))?;
        if info.quant_type == 3 {
            t.dtype = rdna_compute::DType::Q8_0;
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
    fn upload_layer_routed_experts(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        prefix: &str,
        n_exp: usize,
        layer: &mut DeepseekV4LayerWeights,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
    ) -> Result<(), String> {
        // EP shard: precompute owned set + compact-slot mapping. `shard = None`
        // ⇒ every expert owned, `local_of_global[e] == e`, n_owned == n_exp →
        // identical layout to the unsharded path.
        let owns = |e: usize| shard.map(|(s, rank)| s.owns_expert(rank, e)).unwrap_or(true);
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
            let name0 = format!("{prefix}.ffn.experts.0.w2.weight");
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
                let name = format!("{prefix}.ffn.experts.{e}.w2.weight");
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
            let w1_0 = format!("{prefix}.ffn.experts.0.w1.weight");
            let w3_0 = format!("{prefix}.ffn.experts.0.w3.weight");
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
                let w1_name = format!("{prefix}.ffn.experts.{e}.w1.weight");
                {
                    let (_, w1_bytes) = hfq
                        .tensor_data_pread(&w1_name)
                        .ok_or_else(|| format!("deepseek4: missing {w1_name}"))?;
                    combined.extend_from_slice(&w1_bytes);
                }
                let w3_name = format!("{prefix}.ffn.experts.{e}.w3.weight");
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
            let dummy_gu = if shard.is_some() && n_owned < n_exp {
                let z = gpu
                    .zeros(&[combined_stride / 4], rdna_compute::DType::F32)
                    .map_err(|e| format!("deepseek4: {prefix} zero gate_up dummy: {e:?}"))?;
                let p = z.buf.as_ptr() as u64;
                std::mem::forget(z); // leaked for model lifetime (process teardown reclaims)
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
            // Routed experts: 256 × {w1, w2, w3}.
            for e in 0..cfg.n_routed_experts {
                for proj in &["w1", "w2", "w3"] {
                    let name = format!("layers.{l}.ffn.experts.{e}.{proj}.weight");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!("deepseek4: layer {l} expert {e} missing '{proj}'"));
                    }
                }
            }

            layers.push(DeepseekV4LayerWeights::new_empty(ratio));
        }

        Ok(DeepseekV4Weights {
            token_embd: None,
            output_norm: None,
            head: None,
            hc_head_fn: None,
            hc_head_base: None,
            hc_head_scale: 1.0, // overwritten at load time
            layers,
            mtp_layer: None, // skipped by quantize per `mtp.` prefix; Phase 5 work.
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
        Self::load_weights_inner(hfq, cfg, gpu, None)
    }

    fn new_state(_gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        DeepseekV4State::new(cfg)
    }
}

impl DeepseekV4 {
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
        Self::load_weights_inner(hfq, cfg, gpu, Some((shard, rank)))
    }

    fn load_weights_inner(
        hfq: &mut HfqFile,
        cfg: &DeepseekV4Config,
        gpu: &mut Gpu,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
    ) -> Result<DeepseekV4Weights, String> {
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
        let upload_experts =
            std::env::var("HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS").ok().as_deref() != Some("0");
        let expert_layer_end: Option<usize> = std::env::var("HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END")
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
        let mut mtp_addon: Option<HfqFile> = {
            let env_path = std::env::var("HIPFIRE_DEEPSEEK4_MTP_ADDON").ok();
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

        let mut weights = Self::load_weights_host_only_walk(hfq, cfg)?;

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
            let comp_f16_wmma = std::env::var("HIPFIRE_DEEPSEEK4_COMP_F16_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if layer.compress_ratio > 0 {
                layer.compressor_wkv = Some(Self::upload_quant_or_f16(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.compressor.wkv.weight"),
                )?);
                layer.compressor_wgate = Some(Self::upload_quant_or_f16(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.compressor.wgate.weight"),
                )?);
                if comp_f16_wmma {
                    layer.compressor_wkv_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &format!("layers.{l}.attn.compressor.wkv.weight"),
                    )?);
                    layer.compressor_wgate_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &format!("layers.{l}.attn.compressor.wgate.weight"),
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
                    &format!("layers.{l}.attn.indexer.compressor.wkv.weight"),
                )?);
                layer.indexer_compressor_wgate = Some(Self::upload_quant_or_f16(
                    hfq,
                    gpu,
                    &format!("layers.{l}.attn.indexer.compressor.wgate.weight"),
                )?);
                if comp_f16_wmma {
                    layer.indexer_compressor_wkv_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &format!("layers.{l}.attn.indexer.compressor.wkv.weight"),
                    )?);
                    layer.indexer_compressor_wgate_f16 = Some(Self::upload_quant_as_f16_native(
                        hfq,
                        gpu,
                        &format!("layers.{l}.attn.indexer.compressor.wgate.weight"),
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
            layer.gate_weight = Some(Self::upload_quant_or_f16(
                hfq,
                gpu,
                &format!("layers.{l}.ffn.gate.weight"),
            )?);
            if l >= cfg.num_hash_layers {
                // Store F32 on GPU (was F16 on disk) so the bias can
                // either be added on-device or downloaded once for CPU
                // topk. Also cache host-side for the CPU-routing path.
                let bias_name = format!("layers.{l}.ffn.gate.bias");
                let bias_gpu = Self::upload_global_f16_as_f32(hfq, gpu, &bias_name)?;
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
                if let Some((info, bytes)) = hfq.tensor_data_pread(&tid_name) {
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
            let load_mtp = std::env::var("HIPFIRE_DEEPSEEK4_LOAD_MTP")
                .map(|s| s != "0")
                .unwrap_or(true);
            if !load_mtp {
                eprintln!(
                    "deepseek4: HFQ contains MTP layer but \
                    HIPFIRE_DEEPSEEK4_LOAD_MTP=0 — skipping MTP upload"
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
            for (l, layer) in weights.layers.iter_mut().enumerate() {
                let upload_this_layer = expert_layer_end.is_none_or(|end| l < end);
                if !upload_this_layer {
                    continue;
                }
                let n_exp = cfg.n_routed_experts;
                Self::upload_layer_routed_experts(
                    hfq,
                    gpu,
                    &format!("layers.{l}"),
                    n_exp,
                    layer,
                    shard,
                )?;
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
                Self::upload_layer_routed_experts(
                    mtp_expert_source,
                    gpu,
                    "mtp.0",
                    cfg.n_routed_experts,
                    mtp,
                    shard,
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
}
