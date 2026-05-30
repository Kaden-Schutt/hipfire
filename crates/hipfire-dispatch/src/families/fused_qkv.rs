// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::traits::KernelFamily;
use crate::types::{FusedQkvVariant};

pub struct FusedQkvFamily;

impl KernelFamily for FusedQkvFamily {
    fn name(&self) -> &'static str {
        "fused_qkv"
    }
}

pub struct FusedQkvParams {
    pub kind: FusedQkvVariant,
    pub weights: [*const u8; 4],
    pub x: *const u8,
    pub outputs: [*const u8; 4],
    pub m: [usize; 4],
    pub k: usize,
}

impl FusedQkvFamily {
    pub fn run(_params: &FusedQkvParams) -> Result<(), crate::types::DispatchError> {
        Ok(())
    }
}
