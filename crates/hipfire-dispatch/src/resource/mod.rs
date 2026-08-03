// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use rdna_compute::Gpu;

pub struct ResourceManager {
    _priv: (),
}

impl ResourceManager {
    pub fn new(_gpu: &Gpu) -> Self {
        Self { _priv: () }
    }

    /// No-GPU construction for arch-string dispatch contexts.  Used by
    /// the Qwen35 Frozen preflight, whose selection contract forbids GPU
    /// allocation; the manager itself is a zero-sized placeholder.
    pub fn arch_only() -> Self {
        Self { _priv: () }
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn for_test() -> Self {
        Self { _priv: () }
    }
}
