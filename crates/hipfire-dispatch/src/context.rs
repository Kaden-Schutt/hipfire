// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;
use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::feature_flags::FeatureFlags;
use crate::resource::ResourceManager;

/// Per-session context resolved once at Gpu::init().
/// Shared immutably across all dispatch calls.
pub struct DispatchCtx {
    pub arch: ArchCaps,
    pub flags: Arc<FeatureFlags>,
    pub resources: ResourceManager,
}
