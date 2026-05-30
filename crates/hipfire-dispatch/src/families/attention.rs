// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::traits::KernelFamily;
use crate::types::AttentionVariant;

pub struct AttentionFamily;

impl KernelFamily for AttentionFamily {
    fn name(&self) -> &'static str {
        "attention"
    }
}

pub struct AttnParams {
    pub kind: AttentionVariant,
    pub q: *const u8,
    pub k: *const u8,
    pub v: *const u8,
    pub k_cache: *const u8,
    pub v_cache: *const u8,
    pub pos: *const u8,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
}

impl AttentionFamily {
    pub fn run(_params: &AttnParams) -> Result<(), crate::types::DispatchError> {
        Ok(())
    }
}
