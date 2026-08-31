// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::carrier::LlamaBundle;

impl ArchModel for LlamaBundle {
    fn dim(&self) -> usize {
        self.config.dim
    }

    fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "llama"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        Some(&mut self.kv)
    }

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.kv.compact_offset = 0;
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let LlamaBundle {
            config: _,
            weights,
            scratch,
            kv,
            manifest_plan: _,
            weight_store,
            weight_origin: _,
            mesh: _,
            dflash_extract_layers: _,
            dspark_weights: _,
            dspark_assets: _,
        } = *self;
        // Mirror the existing unload ordering: scratch → store/weights → kv.
        scratch.free_gpu(gpu);
        if let Some(store) = weight_store {
            // Attachment already checked the complete origin and created this
            // owner capability. There is no mismatch branch to leak the model:
            // an attached store can only be drained by this consuming owner.
            store.drain(gpu);
        }
        weights.free_gpu(gpu);
        let _ = kv.free_gpu(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_hardware::DeviceMesh;
    use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
    use hipfire_runtime::weight_store::{
        fulfill_manifest_single, WeightLoadTransaction, WeightOrigin, WeightStore,
    };

    #[test]
    fn attached_owner_drain_is_consuming_and_empty_drain_is_safe() {
        let Ok(gpu) = Gpu::init() else {
            return;
        };
        let mesh = DeviceMesh::single();
        let origin = WeightOrigin::for_single(&mesh, &gpu);
        let entry =
            WeightEntry::model("owned", vec![1], rdna_compute::DType::F32, ShardPolicy::Replicate);
        let transaction = fulfill_manifest_single(&[entry], &mesh, 1, &gpu, |_| {
            Ok((vec![0; 4], rdna_compute::DType::F32))
        })
        .unwrap();
        transaction.publish(origin).unwrap().drain(&gpu);
        WeightLoadTransaction::new(WeightStore::with_origin(origin))
            .publish(origin)
            .unwrap()
            .drain(&gpu);
    }
}
