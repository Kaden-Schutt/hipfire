// SPDX-License-Identifier: MIT OR Apache-2.0
use rdna_compute::Gpu;

pub struct ResourceManager {
    _priv: (),
}

impl ResourceManager {
    pub fn new(_gpu: &Gpu) -> Self {
        Self { _priv: () }
    }
}
