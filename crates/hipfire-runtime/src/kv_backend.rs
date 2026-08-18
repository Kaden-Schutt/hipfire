//! KV storage backend selection.
//!
//! This is deliberately separate from [`crate::kv_mode`]. A KV mode selects
//! the element encoding (q8, asym3, and so on); a backend selects how the
//! resulting bytes are allocated and grown.

use std::fmt;
use std::str::FromStr;


pub use saddle_core::kv::{
    KvBackend, KvChunkPlan, KvChunkPlanError, KvMapGrowth, ParseKvBackendError,
    DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES, KV_BACKEND_NAMES,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_stable_backend_names() {
        assert_eq!("contiguous".parse(), Ok(KvBackend::Contiguous));
        assert_eq!("vmm".parse(), Ok(KvBackend::Vmm));
    }

    #[test]
    fn default_is_the_existing_contiguous_path() {
        assert_eq!(KvBackend::default(), KvBackend::Contiguous);
    }

    #[test]
    fn rejects_unknown_or_empty_names() {
        for raw in ["", "auto", "paged", "VMM"] {
            let err = raw.parse::<KvBackend>().unwrap_err().to_string();
            assert!(err.contains(raw));
            assert!(err.contains("contiguous, vmm"));
        }
    }

    #[test]
    fn page_aligns_asym3_and_q8_growth_without_padding_the_dense_rows() {
        let k = KvChunkPlan::new(200, 4096, 64, 4096, 4096).unwrap();
        assert_eq!(k.growth_bytes(), 16_384);
        assert_eq!(k.mapped_bytes_for_tokens(64).unwrap(), 16_384);
        assert_eq!(k.token_capacity(16_384), 81);

        let v = KvChunkPlan::new(544, 4096, 64, 4096, 4096).unwrap();
        assert_eq!(v.growth_bytes(), 36_864);
        assert_eq!(v.mapped_bytes_for_tokens(64).unwrap(), 36_864);
        assert_eq!(v.token_capacity(36_864), 67);
    }

    #[test]
    fn emits_aligned_incremental_growth() {
        let plan = KvChunkPlan::new(544, 4096, 64, 4096, 4096).unwrap();
        let first = plan.growth(0, 1).unwrap().unwrap();
        assert_eq!(first.offset_bytes, 0);
        assert_eq!(first.size_bytes, 36_864);
        assert_eq!(first.token_capacity, 67);

        assert_eq!(plan.growth(first.size_bytes, 64).unwrap(), None);
        let second = plan.growth(first.size_bytes, 68).unwrap().unwrap();
        assert_eq!(second.offset_bytes, 36_864);
        assert_eq!(second.size_bytes, 36_864);
        assert_eq!(second.token_capacity, 135);
    }

    #[test]
    fn applies_a_physical_growth_target_without_padding_the_final_tail() {
        let k = KvChunkPlan::new(
            200,
            9216,
            DEFAULT_KV_CHUNK_TOKENS,
            4096,
            DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
        )
        .unwrap();
        assert_eq!(k.growth_bytes(), 2 * 1024 * 1024);
        let full_k = k.growth(0, 1).unwrap().unwrap();
        assert_eq!(full_k.size_bytes, k.reserve_bytes());
        assert_eq!(full_k.token_capacity, 9216);

        let v = KvChunkPlan::new(
            544,
            9216,
            DEFAULT_KV_CHUNK_TOKENS,
            4096,
            DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
        )
        .unwrap();
        let first_v = v.growth(0, 1).unwrap().unwrap();
        assert_eq!(first_v.size_bytes, 2 * 1024 * 1024);
        assert_eq!(first_v.token_capacity, 3855);
        let second_v = v.growth(first_v.size_bytes, 4096).unwrap().unwrap();
        assert_eq!(second_v.size_bytes, 2 * 1024 * 1024);
        assert_eq!(second_v.token_capacity, 7710);
        let final_v = v.growth(2 * second_v.size_bytes, 8192).unwrap().unwrap();
        assert_eq!(
            final_v.size_bytes,
            v.reserve_bytes() - 2 * second_v.size_bytes
        );
        assert_eq!(final_v.token_capacity, 9216);
    }

    #[test]
    fn validates_chunk_plan_bounds() {
        assert!(matches!(
            KvChunkPlan::new(0, 4096, 64, 4096, 4096),
            Err(KvChunkPlanError::Zero("bytes_per_token"))
        ));
        assert!(matches!(
            KvChunkPlan::new(200, 100, 64, 4096, 0),
            Err(KvChunkPlanError::Zero("minimum_growth_bytes"))
        ));
        let plan = KvChunkPlan::new(200, 100, 64, 4096, 4096).unwrap();
        assert_eq!(plan.reserve_bytes(), 20_480);
        assert!(matches!(
            plan.mapped_bytes_for_tokens(101),
            Err(KvChunkPlanError::RequiredTokens {
                required: 101,
                max: 100
            })
        ));
        assert!(matches!(
            plan.growth(1, 1),
            Err(KvChunkPlanError::InvalidMappedBytes { .. })
        ));
    }
}
