// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;
use hip_bridge::DeviceBuffer;
use rdna_compute::{Gpu, GpuTensor};

pub struct AttnParams<'a> {
    pub kind: KernelKey,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    pub k_scales: Option<&'a GpuTensor>,
    pub v_scales: Option<&'a GpuTensor>,
    pub pos_buf: &'a DeviceBuffer,
    pub pos: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub physical_cap: usize,
    pub flash_partials: Option<&'a GpuTensor>,
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    pub flash_mode: Option<usize>,
    pub capture_mode: bool,
    pub output: &'a GpuTensor,
}

pub struct AttentionFamily {
    registry: KernelRegistry,
}

impl AttentionFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        super::super::tables::attention_table::populate(&mut registry);
        registry.validate().expect("attention kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    pub fn resolve(
        &self,
        key: KernelKey,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        self.registry.resolve(key, ctx, shape)
    }

    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &AttnParams,
    ) -> Result<(), DispatchError> {
        self.resolve(params.kind, ctx, None)?;
        dispatch_attention(gpu, params)
    }
}

impl KernelFamily for AttentionFamily {
    fn name(&self) -> &'static str {
        "attention"
    }
}

macro_rules! hip {
    ($e:expr) => {
        $e.map_err(|e| DispatchError::Hip(e.to_string()))
    };
}

fn dispatch_attention(gpu: &mut Gpu, params: &AttnParams) -> Result<(), DispatchError> {
    let seq_len = params.pos + 1;
    match params.kind {
        KernelKey::KvWriteF32 => {
            hip!(gpu.kv_cache_write(params.k_cache, params.k, params.pos_buf, params.kv_dim))?;
            hip!(gpu.kv_cache_write(params.v_cache, params.v, params.pos_buf, params.kv_dim))
        }
        KernelKey::KvWriteQ8_0 => {
            hip!(gpu.kv_cache_write_q8_0(params.k_cache, params.k, params.pos_buf, params.n_kv_heads, params.head_dim))?;
            hip!(gpu.kv_cache_write_q8_0(params.v_cache, params.v, params.pos_buf, params.n_kv_heads, params.head_dim))
        }
        KernelKey::KvWriteAsym4 => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym4_fused(
                params.k_cache, params.v_cache, params.k, params.v, params.pos_buf,
                ct, st, params.n_kv_heads, params.head_dim,
            ))
        }
        KernelKey::KvWriteAsym4Fwht => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht4_fused(
                params.k_cache, params.v_cache, params.k, params.v, params.pos_buf,
                ct, st, params.n_kv_heads, params.head_dim,
            ))
        }
        KernelKey::KvWriteAsym3 => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym3_fused(
                params.k_cache, params.v_cache, params.k, params.v, params.pos_buf,
                ct, st, params.n_kv_heads, params.head_dim,
            ))
        }
        KernelKey::KvWriteAsym3Fwht => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht3_fused(
                params.k_cache, params.v_cache, params.k, params.v, params.pos_buf,
                ct, st, params.n_kv_heads, params.head_dim,
            ))
        }
        KernelKey::KvWriteAsym2 => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym2_fused(
                params.k_cache, params.v_cache, params.k, params.v, params.pos_buf,
                ct, st, params.n_kv_heads, params.head_dim,
            ))
        }
        KernelKey::KvWriteAsym2Fwht => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht2_fused(
                params.k_cache, params.v_cache, params.k, params.v, params.pos_buf,
                ct, st, params.n_kv_heads, params.head_dim,
            ))
        }
        KernelKey::AttnF32 => {
            hip!(gpu.attention_f32(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap,
            ))
        }
        KernelKey::AttnFlashQ8_0 => {
            let fp = params.flash_partials.unwrap();
            hip!(gpu.attention_flash_q8_0(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym4 => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            let fp = params.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym4(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                ct, st, seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym4Fwht => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            let fp = params.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht4(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                ct, st, seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym3 => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            let fp = params.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym3(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                ct, st, seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym3Fwht => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            let fp = params.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht3(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                ct, st, seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym2 => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            let fp = params.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym2(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                ct, st, seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym2Fwht => {
            let ct = params.givens_cos.unwrap();
            let st = params.givens_sin.unwrap();
            let fp = params.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht2(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                ct, st, seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap, fp,
            ))
        }
        KernelKey::AttnGqaFused => {
            hip!(gpu.attention_flash_gqa_fused(
                params.q, params.k_cache, params.v_cache, params.output,
                seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap,
            ))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "attention",
            variant: "",
            arch: "",
            quant: "",
        }),
    }
}
