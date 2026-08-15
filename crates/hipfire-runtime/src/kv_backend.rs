//! KV storage backend selection.
//!
//! This is deliberately separate from [`crate::kv_mode`]. A KV mode selects
//! the element encoding (q8, asym3, and so on); a backend selects how the
//! resulting bytes are allocated and grown.

use std::fmt;
use std::str::FromStr;

pub const KV_BACKEND_NAMES: &[&str] = &["contiguous", "vmm"];
pub const DEFAULT_KV_CHUNK_TOKENS: usize = 64;

/// Default physical growth target. Standard GPU VMM drivers may account each
/// handle in 2 MiB pages even when the API accepts finer map alignment.
/// The final reserve tail may be smaller.
pub const DEFAULT_VMM_PHYSICAL_CHUNK_BYTES: usize = 2 * 1024 * 1024;

pub use saddle_core::kv::{KvBackend, ParseKvBackendError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvMapGrowth {
    pub offset_bytes: usize,
    pub size_bytes: usize,
    pub token_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvChunkPlan {
    bytes_per_token: usize,
    max_tokens: usize,
    target_chunk_tokens: usize,
    granularity: usize,
    reserve_bytes: usize,
    growth_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvChunkPlanError {
    Zero(&'static str),
    Overflow,
    RequiredTokens { required: usize, max: usize },
    InvalidMappedBytes { mapped: usize, reserve: usize },
}

impl fmt::Display for KvChunkPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero(field) => write!(f, "{field} must be greater than zero"),
            Self::Overflow => f.write_str("KV chunk byte calculation overflowed"),
            Self::RequiredTokens { required, max } => {
                write!(f, "required token count {required} exceeds maximum {max}")
            }
            Self::InvalidMappedBytes { mapped, reserve } => write!(
                f,
                "current mapped byte count {mapped} is not page-aligned or exceeds reserve {reserve}"
            ),
        }
    }
}

impl std::error::Error for KvChunkPlanError {}

impl KvChunkPlan {
    pub fn new(
        bytes_per_token: usize,
        max_tokens: usize,
        target_chunk_tokens: usize,
        granularity: usize,
        minimum_growth_bytes: usize,
    ) -> Result<Self, KvChunkPlanError> {
        for (name, value) in [
            ("bytes_per_token", bytes_per_token),
            ("max_tokens", max_tokens),
            ("target_chunk_tokens", target_chunk_tokens),
            ("granularity", granularity),
            ("minimum_growth_bytes", minimum_growth_bytes),
        ] {
            if value == 0 {
                return Err(KvChunkPlanError::Zero(name));
            }
        }

        let logical_bytes = bytes_per_token
            .checked_mul(max_tokens)
            .ok_or(KvChunkPlanError::Overflow)?;
        let target_bytes = bytes_per_token
            .checked_mul(target_chunk_tokens)
            .ok_or(KvChunkPlanError::Overflow)?;
        let reserve_bytes = checked_round_up(logical_bytes, granularity)?;
        let growth_bytes = checked_round_up(target_bytes.max(minimum_growth_bytes), granularity)?;

        Ok(Self {
            bytes_per_token,
            max_tokens,
            target_chunk_tokens,
            granularity,
            reserve_bytes,
            growth_bytes,
        })
    }

    pub const fn reserve_bytes(self) -> usize {
        self.reserve_bytes
    }

    pub const fn growth_bytes(self) -> usize {
        self.growth_bytes
    }

    pub const fn bytes_per_token(self) -> usize {
        self.bytes_per_token
    }

    pub const fn target_chunk_tokens(self) -> usize {
        self.target_chunk_tokens
    }

    pub const fn granularity(self) -> usize {
        self.granularity
    }

    pub fn mapped_bytes_for_tokens(
        self,
        required_tokens: usize,
    ) -> Result<usize, KvChunkPlanError> {
        if required_tokens > self.max_tokens {
            return Err(KvChunkPlanError::RequiredTokens {
                required: required_tokens,
                max: self.max_tokens,
            });
        }
        if required_tokens == 0 {
            return Ok(0);
        }

        let required_bytes = self
            .bytes_per_token
            .checked_mul(required_tokens)
            .ok_or(KvChunkPlanError::Overflow)?;
        Ok(checked_round_up(required_bytes, self.growth_bytes)?.min(self.reserve_bytes))
    }

    pub fn token_capacity(self, mapped_bytes: usize) -> usize {
        (mapped_bytes / self.bytes_per_token).min(self.max_tokens)
    }

    pub fn growth(
        self,
        mapped_bytes: usize,
        required_tokens: usize,
    ) -> Result<Option<KvMapGrowth>, KvChunkPlanError> {
        if mapped_bytes > self.reserve_bytes || !mapped_bytes.is_multiple_of(self.granularity) {
            return Err(KvChunkPlanError::InvalidMappedBytes {
                mapped: mapped_bytes,
                reserve: self.reserve_bytes,
            });
        }
        let target = self.mapped_bytes_for_tokens(required_tokens)?;
        if target <= mapped_bytes {
            return Ok(None);
        }
        Ok(Some(KvMapGrowth {
            offset_bytes: mapped_bytes,
            size_bytes: target - mapped_bytes,
            token_capacity: self.token_capacity(target),
        }))
    }
}

fn checked_round_up(value: usize, alignment: usize) -> Result<usize, KvChunkPlanError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(KvChunkPlanError::Overflow)
    }
}

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
