// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::resource::ResourceManager;
use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::feature_flags::FeatureFlags;
use rdna_compute::Gpu;
use std::sync::Arc;

/// Semantic workload carried through dispatch.
///
/// Most kernel choices are shape-driven. Speculative target verification is a
/// deliberate exception: the gfx12 wide-query attention route has repeatedly
/// regressed that short-batch regime, even when an opt-in overrides its normal
/// shape envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchWorkload {
    Standard,
    SpeculativeVerify,
}

/// Hardware snapshot plus call-site semantics shared immutably across a
/// dispatch sequence.
pub struct DispatchCtx {
    pub arch: ArchCaps,
    pub flags: Arc<FeatureFlags>,
    pub resources: ResourceManager,
    pub workload: DispatchWorkload,
}

impl DispatchCtx {
    /// Create a `DispatchCtx` from the GPU's current state. This is cheap
    /// enough to call per-layer (ArchCaps is a few dozen bools and the
    /// FeatureFlags snapshot is reference-counted), but callers in tight loops
    /// should prefer creating it once and reusing the reference.
    pub fn new(gpu: &Gpu) -> Self {
        let flags = gpu.flags.clone();
        let arch = ArchCaps::new(&gpu.arch, flags.clone());
        Self {
            arch,
            flags,
            resources: ResourceManager::new(gpu),
            workload: DispatchWorkload::Standard,
        }
    }

    /// Attach call-site semantics to this otherwise hardware-derived context.
    pub fn with_workload(mut self, workload: DispatchWorkload) -> Self {
        self.workload = workload;
        self
    }

    /// Construct a `DispatchCtx` for the given arch string without a live GPU.
    /// Only for use in tests.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn for_test(arch: &str) -> Self {
        use rdna_compute::feature_flags::FeatureFlags;
        let flags = Arc::new(FeatureFlags::for_test(arch));
        let arch_caps = ArchCaps::new(arch, flags.clone());
        Self {
            arch: arch_caps,
            flags,
            resources: crate::resource::ResourceManager::for_test(),
            workload: DispatchWorkload::Standard,
        }
    }
}
