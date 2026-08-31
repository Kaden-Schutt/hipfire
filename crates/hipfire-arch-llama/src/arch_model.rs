// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::weight_store::{
    WeightHandle, WeightOrigin, WeightStore, WeightStoreError,
};
use rdna_compute::Gpu;

use crate::carrier::LlamaBundle;

fn drain_weight_store(
    store: WeightStore,
    origin: WeightOrigin,
    gpu: &mut Gpu,
) -> Result<(), (WeightStore, WeightStoreError)> {
    for handle in store.take_all(origin)? {
        if let WeightHandle::Resident(tensor) = handle {
            let _ = gpu.free_tensor(tensor);
        }
    }
    Ok(())
}

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
        // Validate before destructuring the consuming owner. A mismatch must
        // leave the resident store attached to an owner that can be retried;
        // leaking the boxed owner is safer than dropping the only cleanup
        // authority.
        if let Some(store) = self.weight_store.as_ref() {
            if let Err(error) = store.validate_origin_value(self.weight_origin) {
                eprintln!("llama: refusing weight-store release: {error}");
                let _ = Box::into_raw(self);
                return;
            }
        }
        let LlamaBundle {
            config: _,
            weights,
            scratch,
            kv,
            manifest_plan: _,
            weight_store,
            weight_origin,
            mesh: _,
            dflash_extract_layers: _,
            dspark_weights: _,
            dspark_assets: _,
        } = *self;
        // Mirror the existing unload ordering: scratch → store/weights → kv.
        scratch.free_gpu(gpu);
        if let Some(store) = weight_store {
            if let Err((store, error)) = drain_weight_store(store, weight_origin, gpu) {
                // This is defensive because the pre-check above used the
                // same immutable origin. Never discard a rejected store.
                eprintln!("llama: refusing weight-store release: {error}");
                std::mem::forget(store);
            }
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
    use hipfire_runtime::weight_store::fulfill_manifest_single;

    #[test]
    fn arch_owner_unload_drains_residents_and_repeated_empty_unload_is_safe() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        let mesh = DeviceMesh::single();
        let origin = WeightOrigin::for_single(&mesh, &gpu);
        let entry = WeightEntry::model("owned", vec![1], rdna_compute::DType::F32, ShardPolicy::Replicate);
        let store = fulfill_manifest_single(&[entry], &mesh, 1, &gpu, |_| {
            Ok((vec![0; 4], rdna_compute::DType::F32))
        })
        .unwrap();
        assert!(drain_weight_store(store, origin, &mut gpu).is_ok());
        assert!(drain_weight_store(WeightStore::with_origin(origin), origin, &mut gpu).is_ok());
    }
}
