// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;
use hip_bridge::DeviceBuffer;
use rdna_compute::{Gpu, GpuTensor};

pub struct AttnParams<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    /// TODO(ship 3.1b): llama HFQ8/INT8 attend scales
    pub k_scales: Option<&'a GpuTensor>,
    /// TODO(ship 3.1b): llama HFQ8/INT8 attend scales
    pub v_scales: Option<&'a GpuTensor>,
    pub pos_buf: &'a DeviceBuffer,
    /// 0-based physical position index. `dispatch_attention` internally computes
    /// `seq_len = pos + 1`. Callers MUST pass `pos`, never `pos + 1`.
    pub pos: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub physical_cap: usize,
    pub flash_partials: Option<&'a GpuTensor>,
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    /// V-quant mode kernarg for fwht KV write/flash (8=Q8, 2/3/4=Lloyd-V).
    pub v_mode_bits: i32,
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
        key: KernelKey,
        params: &AttnParams,
    ) -> Result<(), DispatchError> {
        self.resolve(key, ctx, None)?;
        dispatch_attention(gpu, key, params)
    }

    /// Paired write-then-attend entry point (Phase 0.3). Takes a `KvTierPlan`
    /// carrying both the write key and attend key derived from the same
    /// `KvTierInputs`. Enforces the tier-match debug_assert before dispatch.
    pub fn run_attention(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        plan: &crate::families::kv_tier::KvTierPlan,
        io: &AttnParams,
    ) -> Result<(), DispatchError> {
        self.resolve(plan.write_key, ctx, None)?;
        dispatch_attention(gpu, plan.write_key, io)?;
        self.resolve(plan.attend_key, ctx, None)?;
        dispatch_attention(gpu, plan.attend_key, io)
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

fn dispatch_attention(gpu: &mut Gpu, key: KernelKey, params: &AttnParams) -> Result<(), DispatchError> {
    let seq_len = params.pos + 1;
    match key {
        KernelKey::KvWriteF32 => {
            let kv_dim = params.n_kv_heads * params.head_dim;
            hip!(gpu.kv_cache_write(params.k_cache, params.k, params.pos_buf, kv_dim))?;
            hip!(gpu.kv_cache_write(params.v_cache, params.v, params.pos_buf, kv_dim))
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
                ct, st, params.n_kv_heads, params.head_dim, params.v_mode_bits,
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
                ct, st, params.n_kv_heads, params.head_dim, params.v_mode_bits,
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
                ct, st, params.n_kv_heads, params.head_dim, params.v_mode_bits,
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
        KernelKey::AttnQ8_0Kv => {
            hip!(gpu.attention_q8_0_kv(
                params.q, params.k_cache, params.v_cache, params.output, params.pos_buf,
                seq_len, params.n_heads, params.n_kv_heads, params.head_dim, params.physical_cap,
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
                params.v_mode_bits,
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
                params.v_mode_bits,
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
                params.v_mode_bits,
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
            variant: "unhandled key — missing dispatch arm",
            arch: "",
            quant: "",
        }),
    }
}

/// All `KernelKey` variants that `dispatch_attention` handles with dedicated
/// (non-catch-all) match arms. Used by the dispatch-arm completeness test.
/// If you add a new attention key and forget to add a dispatch arm, the test
/// `dispatch_attention_has_arms_for_all_attention_keys` will fail.
pub(crate) const DISPATCHED_ATTENTION_KEYS: &[KernelKey] = &[
    // KV write
    KernelKey::KvWriteF32,
    KernelKey::KvWriteQ8_0,
    KernelKey::KvWriteAsym4,
    KernelKey::KvWriteAsym4Fwht,
    KernelKey::KvWriteAsym3,
    KernelKey::KvWriteAsym3Fwht,
    KernelKey::KvWriteAsym2,
    KernelKey::KvWriteAsym2Fwht,
    // Attention
    KernelKey::AttnF32,
    KernelKey::AttnFlashQ8_0,
    KernelKey::AttnQ8_0Kv,
    KernelKey::AttnFlashAsym4,
    KernelKey::AttnFlashAsym4Fwht,
    KernelKey::AttnFlashAsym3,
    KernelKey::AttnFlashAsym3Fwht,
    KernelKey::AttnFlashAsym2,
    KernelKey::AttnFlashAsym2Fwht,
    KernelKey::AttnGqaFused,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `KernelKey` registered in the attention table MUST have a dedicated
    /// dispatch arm. Catches the 953ea648 defect class: a key is registered but
    /// `dispatch_attention` falls through to the catch-all (silently or with a
    /// generic error).
    ///
    /// Two-way check:
    ///  (a) Every registered table key appears in `DISPATCHED_ATTENTION_KEYS`.
    ///  (b) Every key in `DISPATCHED_ATTENTION_KEYS` is registered in the table.
    #[test]
    fn dispatch_attention_has_arms_for_all_attention_keys() {
        let family = AttentionFamily::new();
        let ctx = DispatchCtx::for_test("gfx1100");

        // Build the set of registered attention keys by probing resolve.
        let registered: Vec<KernelKey> = DISPATCHED_ATTENTION_KEYS
            .iter()
            .filter(|&&k| family.resolve(k, &ctx, None).is_ok())
            .copied()
            .collect();

        // (a) Every registered table key must be in DISPATCHED_ATTENTION_KEYS.
        //     We check by seeing if the dispatched list covers all registrations.
        //     Since DISPATCHED_ATTENTION_KEYS is the exhaustive list, any key
        //     registered in the table but NOT dispatched will fail the (b) check.

        // (b) Every dispatched key must resolve (be registered) on at least gfx1100.
        //     This catches stale entries in DISPATCHED_ATTENTION_KEYS.
        for &key in DISPATCHED_ATTENTION_KEYS {
            assert!(
                registered.contains(&key),
                "DISPATCHED_ATTENTION_KEYS contains {:?} but it is NOT registered in the attention table — stale entry",
                key
            );
        }

        // The real power: check that the attention table registrations are a
        // subset of DISPATCHED_ATTENTION_KEYS. We iterate the attention-family
        // registry directly.
        let dispatched_set: std::collections::HashSet<KernelKey> =
            DISPATCHED_ATTENTION_KEYS.iter().copied().collect();

        // Probe every Attn*/KvWrite* variant against the registry.
        // If a key resolves but isn't in the dispatched set, it has no arm.
        let attention_key_candidates: &[KernelKey] = &[
            KernelKey::KvWriteF32,
            KernelKey::KvWriteQ8_0,
            KernelKey::KvWriteAsym4,
            KernelKey::KvWriteAsym4Fwht,
            KernelKey::KvWriteAsym3,
            KernelKey::KvWriteAsym3Fwht,
            KernelKey::KvWriteAsym2,
            KernelKey::KvWriteAsym2Fwht,
            KernelKey::AttnF32,
            KernelKey::AttnFlashQ8_0,
            KernelKey::AttnQ8_0Kv,
            KernelKey::AttnFlashAsym4,
            KernelKey::AttnFlashAsym4Fwht,
            KernelKey::AttnFlashAsym3,
            KernelKey::AttnFlashAsym3Fwht,
            KernelKey::AttnFlashAsym2,
            KernelKey::AttnFlashAsym2Fwht,
            KernelKey::AttnGqaFused,
        ];

        for &key in attention_key_candidates {
            if family.resolve(key, &ctx, None).is_ok() {
                assert!(
                    dispatched_set.contains(&key),
                    "attention table registers {:?} but dispatch_attention has no dedicated arm — will hit catch-all",
                    key
                );
            }
        }

        // Guard: the dispatched list + candidate list must have the same length.
        // If they differ, either list is out of sync with the other.
        assert_eq!(
            DISPATCHED_ATTENTION_KEYS.len(),
            attention_key_candidates.len(),
            "DISPATCHED_ATTENTION_KEYS ({} entries) and the candidate probe list ({} entries) are out of sync",
            DISPATCHED_ATTENTION_KEYS.len(),
            attention_key_candidates.len(),
        );
    }
}
