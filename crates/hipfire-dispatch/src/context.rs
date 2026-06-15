// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
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
    /// Create a `DispatchCtx` from the GPU's current state. This is cheap
    /// enough to call per-layer (ArchCaps is a few dozen bools; FeatureFlags
    /// reads a handful of env vars), but callers in tight loops should prefer
    /// creating it once and reusing the reference.
    pub fn new(gpu: &Gpu) -> Self {
        let flags = Arc::new(FeatureFlags::from_env(&gpu.arch));
        let arch = ArchCaps::new(&gpu.arch, flags.clone());
        Self {
            arch,
            flags,
            resources: ResourceManager::new(gpu),
        }
    }

    /// Construct a `DispatchCtx` for the given arch string without a live GPU.
    /// Only for use in tests.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn for_test(arch: &str) -> Self {
        use rdna_compute::feature_flags::FeatureFlags;
        let flags = Arc::new(FeatureFlags::from_env_for_test(arch));
        let arch_caps = ArchCaps::new(arch, flags.clone());
        Self {
            arch: arch_caps,
            flags,
            resources: crate::resource::ResourceManager::for_test(),
        }
    }
}
