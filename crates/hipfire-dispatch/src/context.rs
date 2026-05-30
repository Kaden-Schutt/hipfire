// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;
use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::feature_flags::FeatureFlags;
use rdna_compute::Gpu;
use crate::resource::ResourceManager;

/// Per-session context resolved once at Gpu::init().
/// Shared immutably across all dispatch calls.
pub struct DispatchCtx {
    pub arch: ArchCaps,
    pub flags: Arc<FeatureFlags>,
    pub resources: ResourceManager,
}

impl DispatchCtx {
    pub fn new(gpu: &Gpu) -> Self {
        let flags = Arc::new(FeatureFlags::from_env(&gpu.arch));
        let arch = ArchCaps::new(&gpu.arch, flags.clone());
        Self {
            arch,
            flags,
            resources: ResourceManager::new(gpu),
        }
    }
}
