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

    #[cfg(any(test, feature = "test-utils"))]
    pub fn for_test() -> Self {
        Self { _priv: () }
    }
}
