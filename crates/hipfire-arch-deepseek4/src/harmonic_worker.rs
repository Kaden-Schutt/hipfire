// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Process-local gfx1151 routed-expert service for harmonic DS4 execution.
//!
//! This module has no peer pointer, foreign device ID, cross-device signal, or
//! GPU wait primitive. A supervisor publishes typed packets through
//! `harmonic_ipc`; this worker owns one exact gfx1151 HIP context and consumes
//! process-local aliases of HIP-registered payload pages. Any local HIP stall
//! is bounded by terminating this process, not by asking the peer GPU to wait
//! on it.

use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::deepseek4::{DeepseekV4Config, DeepseekV4RoutedWeights};
use crate::harmonic::{
    unpack_harmonic_x_rot, HarmonicCompletion, HarmonicContract, HarmonicOwner,
    HARMONIC_EXPERT_COUNT, HARMONIC_HIDDEN_SIZE, HARMONIC_LAYER_COUNT,
    HARMONIC_MOE_INTERMEDIATE_SIZE, HARMONIC_RESULT_EXTENT, HARMONIC_TOP_K, HARMONIC_X_ROT_BYTES,
};
use crate::harmonic_ipc::{harmonic_payload_fingerprint, HarmonicIntegrityMode, HarmonicWorkItem};
use crate::HarmonicRoutePacket;

/// All scratch and the one execution stream are owned by the gfx1151 worker
/// process. `Option` fields make partial construction transactionally
/// reclaimable without ever consulting another `Gpu`.
pub struct DeepseekV4HarmonicExpertService {
    allocation_generation: u64,
    x_rot: Option<GpuTensor>,
    topk_indices: Option<GpuTensor>,
    topk_weights: Option<GpuTensor>,
    gate_batch: Option<GpuTensor>,
    up_batch: Option<GpuTensor>,
    rot_batch: Option<GpuTensor>,
    down_expanded: Option<GpuTensor>,
    routed_partial: Option<GpuTensor>,
    topk_index_bytes: [u8; HARMONIC_TOP_K * std::mem::size_of::<u32>()],
    topk_weight_bytes: [u8; HARMONIC_TOP_K * std::mem::size_of::<u32>()],
    result_payload: Vec<u8>,
}

impl DeepseekV4HarmonicExpertService {
    pub fn new(
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
        allocation_generation: u64,
    ) -> Result<Self, String> {
        validate_worker_config(gpu, cfg, allocation_generation)?;
        if gpu.active_stream.is_some() {
            return Err(
                "deepseek4 harmonic expert worker requires an unclaimed local stream".to_owned(),
            );
        }
        let mut service = Self {
            allocation_generation,
            x_rot: None,
            topk_indices: None,
            topk_weights: None,
            gate_batch: None,
            up_batch: None,
            rot_batch: None,
            down_expanded: None,
            routed_partial: None,
            topk_index_bytes: [0; HARMONIC_TOP_K * std::mem::size_of::<u32>()],
            topk_weight_bytes: [0; HARMONIC_TOP_K * std::mem::size_of::<u32>()],
            result_payload: vec![0; HARMONIC_RESULT_EXTENT as usize],
        };
        let result = (|| {
            gpu.bind_thread()
                .map_err(|error| format!("harmonic expert bind: {error}"))?;
            gpu.active_stream = Some(
                gpu.hip
                    .stream_create()
                    .map_err(|error| format!("harmonic expert stream: {error}"))?,
            );
            service.x_rot = Some(
                gpu.alloc_tensor(&[HARMONIC_HIDDEN_SIZE], DType::F32)
                    .map_err(|error| format!("harmonic expert x_rot: {error}"))?,
            );
            service.topk_indices = Some(
                gpu.alloc_tensor(&[HARMONIC_TOP_K], DType::F32)
                    .map_err(|error| format!("harmonic expert top-k indices: {error}"))?,
            );
            service.topk_weights = Some(
                gpu.alloc_tensor(&[HARMONIC_TOP_K], DType::F32)
                    .map_err(|error| format!("harmonic expert top-k weights: {error}"))?,
            );
            service.gate_batch = Some(
                gpu.alloc_tensor(
                    &[HARMONIC_TOP_K, HARMONIC_MOE_INTERMEDIATE_SIZE],
                    DType::F32,
                )
                .map_err(|error| format!("harmonic expert gate batch: {error}"))?,
            );
            service.up_batch = Some(
                gpu.alloc_tensor(
                    &[HARMONIC_TOP_K, HARMONIC_MOE_INTERMEDIATE_SIZE],
                    DType::F32,
                )
                .map_err(|error| format!("harmonic expert up batch: {error}"))?,
            );
            service.rot_batch = Some(
                gpu.alloc_tensor(
                    &[HARMONIC_TOP_K, HARMONIC_MOE_INTERMEDIATE_SIZE],
                    DType::F32,
                )
                .map_err(|error| format!("harmonic expert rotation batch: {error}"))?,
            );
            service.down_expanded = Some(
                gpu.alloc_tensor(&[HARMONIC_TOP_K, HARMONIC_HIDDEN_SIZE], DType::F32)
                    .map_err(|error| format!("harmonic expert down expanded: {error}"))?,
            );
            service.routed_partial = Some(
                gpu.alloc_tensor(&[HARMONIC_HIDDEN_SIZE], DType::F32)
                    .map_err(|error| format!("harmonic expert result: {error}"))?,
            );
            Ok(())
        })();
        if let Err(error) = result {
            service.release_unsubmitted(gpu);
            return Err(error);
        }
        Ok(service)
    }

    pub const fn allocation_generation(&self) -> u64 {
        self.allocation_generation
    }

    pub fn result_payload(&self) -> &[u8] {
        &self.result_payload
    }

    /// Execute one already-routed expert request entirely on the local gfx1151
    /// owner. The surrounding worker process is the deadline boundary: this
    /// method performs one local stream synchronization only to materialize the
    /// CPU result, and never waits on another GPU or process.
    pub fn execute_work_item(
        &mut self,
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
        weights: &DeepseekV4RoutedWeights,
        work: &HarmonicWorkItem,
        now_tick: u64,
    ) -> Result<HarmonicCompletion, String> {
        validate_worker_config(gpu, cfg, self.allocation_generation)?;
        if !weights.mq2r_backend.is_gfx1151() {
            return Err("harmonic expert weights do not use the exact gfx1151 backend".to_owned());
        }
        if weights.layers.len() != HARMONIC_LAYER_COUNT as usize {
            return Err(format!(
                "harmonic expert layer residency is {}, expected {}",
                weights.layers.len(),
                HARMONIC_LAYER_COUNT
            ));
        }
        let contract = HarmonicContract::frozen(
            work.packet.source_allocation_generation,
            self.allocation_generation,
        );
        contract
            .validate(&work.packet, now_tick)
            .map_err(|error| format!("harmonic expert route contract: {error}"))?;
        if work.integrity_mode == HarmonicIntegrityMode::Fingerprint {
            let observed_fingerprint = harmonic_payload_fingerprint(&work.activation_payload);
            if observed_fingerprint != work.packet.activation_fingerprint {
                return Err(format!(
                    "harmonic expert activation fingerprint mismatch: got {observed_fingerprint:#x}, expected {:#x}",
                    work.packet.activation_fingerprint
                ));
            }
        }
        let x_rot_bytes = unpack_harmonic_x_rot(&work.packet, &work.activation_payload)
            .map_err(|error| format!("harmonic expert activation layout: {error}"))?;
        encode_words(&work.packet.expert_ids, &mut self.topk_index_bytes);
        encode_words(&work.packet.route_weight_bits, &mut self.topk_weight_bytes);

        gpu.bind_thread()
            .map_err(|error| format!("harmonic expert bind execute: {error}"))?;
        {
            let stream = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| "harmonic expert local stream missing".to_owned())?;
            gpu.hip
                .memcpy_htod_async(&self.x_rot.as_ref().unwrap().buf, x_rot_bytes, stream)
                .map_err(|error| format!("harmonic expert upload x_rot: {error}"))?;
            gpu.hip
                .memcpy_htod_async(
                    &self.topk_indices.as_ref().unwrap().buf,
                    &self.topk_index_bytes,
                    stream,
                )
                .map_err(|error| format!("harmonic expert upload top-k indices: {error}"))?;
            gpu.hip
                .memcpy_htod_async(
                    &self.topk_weights.as_ref().unwrap().buf,
                    &self.topk_weight_bytes,
                    stream,
                )
                .map_err(|error| format!("harmonic expert upload top-k weights: {error}"))?;
            let routed_partial = self.routed_partial.as_ref().unwrap();
            gpu.hip
                .memset_async(&routed_partial.buf, 0, routed_partial.byte_size(), stream)
                .map_err(|error| format!("harmonic expert zero result: {error}"))?;
        }

        let layer_index = work.packet.layer as usize;
        let layer = weights.resolve_layer(layer_index);
        let expert_gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().ok_or_else(|| {
            format!("harmonic expert gate/up pointer table missing l{layer_index}")
        })?;
        let expert_down_ptrs = layer
            .expert_w2_ptrs
            .as_ref()
            .ok_or_else(|| format!("harmonic expert down pointer table missing l{layer_index}"))?;
        let params = hipfire_dispatch::families::moe::MoeSelectedParams {
            hidden: HARMONIC_HIDDEN_SIZE,
            mi: HARMONIC_MOE_INTERMEDIATE_SIZE,
            k_top: HARMONIC_TOP_K,
            swiglu_limit: cfg.swiglu_limit,
            uses_atomic_moe_down: weights.mq2r_backend.uses_atomic_moe_down(),
            native_mq2_backend: weights.mq2r_backend.bias_aware_native_backend(),
            batch_size: 1,
            x_rot: self.x_rot.as_ref().unwrap(),
            ffn_out: self.routed_partial.as_ref().unwrap(),
            expert_gate_up_ptrs,
            expert_down_ptrs,
            topk_indices: self.topk_indices.as_ref().unwrap(),
            topk_weights: self.topk_weights.as_ref().unwrap(),
            gate_batch: self.gate_batch.as_ref().unwrap(),
            up_batch: self.up_batch.as_ref().unwrap(),
            rot_batch: self.rot_batch.as_ref().unwrap(),
            down_expanded: self.down_expanded.as_ref().unwrap(),
        };
        hipfire_runtime::llama::moe_family()
            .run_selected(gpu, &params)
            .map_err(|error| format!("harmonic selected experts l{layer_index}: {error}"))?;

        {
            let stream = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| "harmonic expert local stream missing after dispatch".to_owned())?;
            gpu.hip
                .memcpy_dtoh_async(
                    &mut self.result_payload,
                    &self.routed_partial.as_ref().unwrap().buf,
                    stream,
                )
                .map_err(|error| format!("harmonic expert download result: {error}"))?;
            gpu.hip
                .stream_synchronize(stream)
                .map_err(|error| format!("harmonic expert local completion: {error}"))?;
        }
        Ok(HarmonicCompletion {
            result_extent: HARMONIC_RESULT_EXTENT,
            result_fingerprint: if work.integrity_mode == HarmonicIntegrityMode::Fingerprint {
                harmonic_payload_fingerprint(&self.result_payload)
            } else {
                0
            },
        })
    }

    /// Execute one release/acquire packet directly from this process's mapped
    /// ring alias. The selected-expert kernels are identical to
    /// [`Self::execute_work_item`]; only the 16 KiB activation/result transport
    /// changes. The one local stream synchronization publishes completion only
    /// after the result copy reached the shared result slot.
    pub fn execute_mapped_packet(
        &mut self,
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
        weights: &DeepseekV4RoutedWeights,
        packet: &HarmonicRoutePacket,
        activation_payload: &DeviceBuffer,
        result_payload: &DeviceBuffer,
        now_tick: u64,
    ) -> Result<HarmonicCompletion, String> {
        validate_worker_config(gpu, cfg, self.allocation_generation)?;
        if !weights.mq2r_backend.is_gfx1151() {
            return Err("harmonic expert weights do not use the exact gfx1151 backend".to_owned());
        }
        if weights.layers.len() != HARMONIC_LAYER_COUNT as usize {
            return Err(format!(
                "harmonic expert layer residency is {}, expected {}",
                weights.layers.len(),
                HARMONIC_LAYER_COUNT
            ));
        }
        let contract = HarmonicContract::frozen(
            packet.source_allocation_generation,
            self.allocation_generation,
        );
        contract
            .validate(packet, now_tick)
            .map_err(|error| format!("harmonic expert mapped route contract: {error}"))?;
        if activation_payload.size() < HARMONIC_X_ROT_BYTES {
            return Err(format!(
                "harmonic mapped activation has {} bytes, expected at least {HARMONIC_X_ROT_BYTES}",
                activation_payload.size()
            ));
        }
        if result_payload.size() < HARMONIC_RESULT_EXTENT as usize {
            return Err(format!(
                "harmonic mapped result has {} bytes, expected at least {HARMONIC_RESULT_EXTENT}",
                result_payload.size()
            ));
        }
        encode_words(&packet.expert_ids, &mut self.topk_index_bytes);
        encode_words(&packet.route_weight_bits, &mut self.topk_weight_bytes);

        gpu.bind_thread()
            .map_err(|error| format!("harmonic mapped expert bind execute: {error}"))?;
        {
            let stream = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| "harmonic mapped expert local stream missing".to_owned())?;
            gpu.hip
                .memcpy_htod_async(
                    &self.topk_indices.as_ref().unwrap().buf,
                    &self.topk_index_bytes,
                    stream,
                )
                .map_err(|error| format!("harmonic mapped expert upload top-k indices: {error}"))?;
            gpu.hip
                .memcpy_htod_async(
                    &self.topk_weights.as_ref().unwrap().buf,
                    &self.topk_weight_bytes,
                    stream,
                )
                .map_err(|error| format!("harmonic mapped expert upload top-k weights: {error}"))?;
            let routed_partial = self.routed_partial.as_ref().unwrap();
            gpu.hip
                .memset_async(&routed_partial.buf, 0, routed_partial.byte_size(), stream)
                .map_err(|error| format!("harmonic mapped expert zero result: {error}"))?;
        }

        let x_rot = GpuTensor {
            // SAFETY: the process-local `HarmonicGpuMapping` owns this live HIP
            // registration across the synchronous call. The payload begins
            // with exactly one 4096-element F32 activation.
            buf: unsafe {
                DeviceBuffer::from_raw(activation_payload.as_ptr(), HARMONIC_X_ROT_BYTES)
            },
            shape: vec![HARMONIC_HIDDEN_SIZE],
            dtype: DType::F32,
        };
        let layer_index = packet.layer as usize;
        let layer = weights.resolve_layer(layer_index);
        let expert_gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().ok_or_else(|| {
            format!("harmonic mapped expert gate/up pointer table missing l{layer_index}")
        })?;
        let expert_down_ptrs = layer.expert_w2_ptrs.as_ref().ok_or_else(|| {
            format!("harmonic mapped expert down pointer table missing l{layer_index}")
        })?;
        let params = hipfire_dispatch::families::moe::MoeSelectedParams {
            hidden: HARMONIC_HIDDEN_SIZE,
            mi: HARMONIC_MOE_INTERMEDIATE_SIZE,
            k_top: HARMONIC_TOP_K,
            swiglu_limit: cfg.swiglu_limit,
            uses_atomic_moe_down: weights.mq2r_backend.uses_atomic_moe_down(),
            native_mq2_backend: weights.mq2r_backend.bias_aware_native_backend(),
            batch_size: 1,
            x_rot: &x_rot,
            ffn_out: self.routed_partial.as_ref().unwrap(),
            expert_gate_up_ptrs,
            expert_down_ptrs,
            topk_indices: self.topk_indices.as_ref().unwrap(),
            topk_weights: self.topk_weights.as_ref().unwrap(),
            gate_batch: self.gate_batch.as_ref().unwrap(),
            up_batch: self.up_batch.as_ref().unwrap(),
            rot_batch: self.rot_batch.as_ref().unwrap(),
            down_expanded: self.down_expanded.as_ref().unwrap(),
        };
        hipfire_runtime::llama::moe_family()
            .run_selected(gpu, &params)
            .map_err(|error| format!("harmonic mapped selected experts l{layer_index}: {error}"))?;

        {
            let stream = gpu.active_stream.as_ref().ok_or_else(|| {
                "harmonic mapped expert local stream missing after dispatch".to_owned()
            })?;
            gpu.hip
                .memcpy_dtod_async_at(
                    result_payload,
                    0,
                    &self.routed_partial.as_ref().unwrap().buf,
                    0,
                    HARMONIC_RESULT_EXTENT as usize,
                    stream,
                )
                .map_err(|error| format!("harmonic mapped expert publish result: {error}"))?;
            gpu.hip
                .stream_synchronize(stream)
                .map_err(|error| format!("harmonic mapped expert local completion: {error}"))?;
        }
        Ok(HarmonicCompletion {
            result_extent: HARMONIC_RESULT_EXTENT,
            result_fingerprint: 0,
        })
    }

    /// Reclaim a normally completed local service. The caller must have
    /// observed completion of the last work item. Faulted or timed-out workers
    /// are terminated by the supervisor and reclaimed by KFD instead.
    pub fn release_quiesced(mut self, gpu: &mut Gpu) {
        self.release_unsubmitted(gpu);
    }

    fn release_unsubmitted(&mut self, gpu: &mut Gpu) {
        fn free(gpu: &mut Gpu, tensor: &mut Option<GpuTensor>) {
            if let Some(tensor) = tensor.take() {
                let _ = gpu.free_tensor(tensor);
            }
        }
        gpu.bind_thread_or_warn();
        free(gpu, &mut self.x_rot);
        free(gpu, &mut self.topk_indices);
        free(gpu, &mut self.topk_weights);
        free(gpu, &mut self.gate_batch);
        free(gpu, &mut self.up_batch);
        free(gpu, &mut self.rot_batch);
        free(gpu, &mut self.down_expanded);
        free(gpu, &mut self.routed_partial);
        if let Some(stream) = gpu.active_stream.take() {
            let _ = gpu.hip.stream_destroy(stream);
        }
    }
}

fn validate_worker_config(
    gpu: &Gpu,
    cfg: &DeepseekV4Config,
    allocation_generation: u64,
) -> Result<(), String> {
    HarmonicOwner::ExpertGfx1151
        .validate_arch(&gpu.arch)
        .map_err(|error| format!("harmonic expert exact architecture: {error}"))?;
    if allocation_generation == 0 {
        return Err("harmonic expert allocation generation must be nonzero".to_owned());
    }
    if !cfg.mq2r
        || cfg.mq2rxt
        || cfg.load_dspark
        || cfg.num_hidden_layers != HARMONIC_LAYER_COUNT as usize
        || cfg.hidden_size != HARMONIC_HIDDEN_SIZE
        || cfg.moe_intermediate_size != HARMONIC_MOE_INTERMEDIATE_SIZE
        || cfg.n_routed_experts != HARMONIC_EXPERT_COUNT as usize
        || cfg.num_experts_per_tok != HARMONIC_TOP_K
    {
        return Err("harmonic expert worker requires the frozen MQ2R P3 shape".to_owned());
    }
    Ok(())
}

fn encode_words<const N: usize>(words: &[u32; N], bytes: &mut [u8]) {
    assert_eq!(bytes.len(), N * std::mem::size_of::<u32>());
    for (word, output) in words.iter().zip(bytes.chunks_exact_mut(4)) {
        output.copy_from_slice(&word.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn expert_worker_source_has_no_cross_device_primitives() {
        let source = include_str!("harmonic_worker.rs");
        for forbidden in [
            ["enable_peer_", "access"].concat(),
            ["memcpy_", "peer"].concat(),
            ["stream_wait_", "value32"].concat(),
            ["stream_write_", "value32"].concat(),
            ["device", "_id"].concat(),
        ] {
            assert!(!source.contains(&forbidden), "found {forbidden}");
        }
    }

    #[test]
    fn expert_worker_process_is_exact_pci_bound_and_has_no_peer_path() {
        let source = include_str!("bin/deepseek4_harmonic_expert_worker.rs");
        for required in [
            "Gpu::init_with_pci_bus_id",
            "GpuSelector::PciBusId",
            "DeepseekV4VerifiedArtifact::verify",
            "load_weights_harmonic_experts_gfx1151",
        ] {
            assert!(source.contains(required), "missing {required}");
        }
        for forbidden in [
            ["Gpu::init_with", "_device"].concat(),
            ["GpuSelector::", "Ordinal"].concat(),
            ["enable_peer", "_access"].concat(),
            ["memcpy_", "peer"].concat(),
            ["stream_wait_", "value32"].concat(),
            ["stream_write_", "value32"].concat(),
        ] {
            assert!(!source.contains(&forbidden), "found {forbidden}");
        }
    }
}
