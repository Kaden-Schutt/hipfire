// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Partition-aware recipe wrappers for CU green-context assumptions.
//!
//! Recipes can declare the CU slice they were certified under without editing
//! the architecture-neutral catalog in [`crate::recipes`]. Wave 3 attaches
//! these wrappers to existing recipe records.
//!
//! Validation mirrors `redline-dispatch` partition rules: CU counts that do
//! not divide the device resource are a hard error — never a silent
//! non-partitioned fallback.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// CU partition context assumed by a recipe or certification row.
///
/// Corresponds to one `CuPartition` slice from dispatch (`cu_offset` /
/// `cu_count`) plus an `exclusive` flag for whether the recipe requires
/// exclusive ownership of that slice (green-ctx exclusive vs affinity-only).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PartitionContext {
    pub cu_offset: u32,
    pub cu_count: u32,
    pub exclusive: bool,
}

/// Generic wrapper so Wave 3 can attach partition context to any recipe
/// record type without touching `recipes.rs`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PartitionAwareRecipe<T> {
    pub partition: PartitionContext,
    pub inner: T,
}

/// Failures when a declared partition context cannot map onto a device.
///
/// Named variants match the fail-closed contract in the 7.14 leverage design:
/// unmappable policies are configuration errors, never silent downgrades.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PartitionError {
    /// `cu_count` is zero — empty partitions are never valid.
    #[error("partition cu_count must be non-zero")]
    ZeroCount,
    /// Device reports zero CUs — cannot form partitions.
    #[error("device CU count is zero")]
    ZeroDevice,
    /// Partition end exceeds the device CU resource (or offset+count overflowed).
    #[error(
        "partition cu_offset {cu_offset} + cu_count {cu_count} exceeds device CU count {device_cu_count}"
    )]
    OutOfBounds {
        cu_offset: u32,
        cu_count: u32,
        device_cu_count: u32,
    },
    /// Equal-parts layout: `parts` does not divide the device CU resource.
    #[error(
        "equal partition count {parts} does not divide device CU count {device_cu_count}"
    )]
    NonDividing {
        device_cu_count: u32,
        parts: u32,
    },
    /// Multi-slice coverage check failed (gaps, overlap, or sum mismatch).
    #[error(
        "partition set covers {covered} CUs but device has {device_cu_count} \
         (slices must be contiguous, non-overlapping, and complete)"
    )]
    CoverageMismatch { covered: u32, device_cu_count: u32 },
}

pub type PartitionResult<T> = std::result::Result<T, PartitionError>;

impl PartitionContext {
    /// Validate this single slice against a concrete device CU count.
    ///
    /// Single-slice rules (explicit partition membership):
    /// - `device_cu_count` must be non-zero
    /// - `cu_count` must be non-zero
    /// - `cu_offset + cu_count` must fit in `device_cu_count` (no overflow)
    ///
    /// Equal-grid alignment (`offset % count == 0` and `device % count == 0`) is
    /// **not** required here — heterogeneous explicit layouts from
    /// `redline-dispatch` (e.g. `[8, 24, 32]`) are valid. Use
    /// [`validate_partition_set`] to check a complete cover.
    pub fn validate(self, device_cu_count: u32) -> PartitionResult<()> {
        validate_partition_context(&self, device_cu_count)
    }
}

impl<T> PartitionAwareRecipe<T> {
    pub fn new(partition: PartitionContext, inner: T) -> Self {
        Self { partition, inner }
    }

    /// Validate the attached partition context against a device CU count.
    pub fn validate_partition(&self, device_cu_count: u32) -> PartitionResult<()> {
        validate_partition_context(&self.partition, device_cu_count)
    }

    pub fn map_inner<U>(self, f: impl FnOnce(T) -> U) -> PartitionAwareRecipe<U> {
        PartitionAwareRecipe {
            partition: self.partition,
            inner: f(self.inner),
        }
    }
}

/// Validate a single [`PartitionContext`] against `device_cu_count`.
///
/// Only nonzero count + in-bounds offset/count. Does **not** require the slice
/// size to divide the device (that would reject valid explicit heterogeneous
/// partitions from dispatch). Non-dividing equal-parts layouts still fail via
/// [`validate_equal_parts`].
pub fn validate_partition_context(
    partition: &PartitionContext,
    device_cu_count: u32,
) -> PartitionResult<()> {
    if device_cu_count == 0 {
        return Err(PartitionError::ZeroDevice);
    }
    if partition.cu_count == 0 {
        return Err(PartitionError::ZeroCount);
    }

    let end = partition
        .cu_offset
        .checked_add(partition.cu_count)
        .ok_or(PartitionError::OutOfBounds {
            cu_offset: partition.cu_offset,
            cu_count: partition.cu_count,
            device_cu_count,
        })?;
    if end > device_cu_count {
        return Err(PartitionError::OutOfBounds {
            cu_offset: partition.cu_offset,
            cu_count: partition.cu_count,
            device_cu_count,
        });
    }

    Ok(())
}

/// Validate a complete partition set: each slice in-bounds, no gaps/overlaps,
/// and total coverage exactly `device_cu_count` (mirrors dispatch Explicit).
pub fn validate_partition_set(
    partitions: &[PartitionContext],
    device_cu_count: u32,
) -> PartitionResult<()> {
    if device_cu_count == 0 {
        return Err(PartitionError::ZeroDevice);
    }
    if partitions.is_empty() {
        return Err(PartitionError::CoverageMismatch {
            covered: 0,
            device_cu_count,
        });
    }

    let mut ordered: Vec<&PartitionContext> = partitions.iter().collect();
    ordered.sort_by_key(|p| p.cu_offset);

    let mut cursor = 0u32;
    for p in &ordered {
        validate_partition_context(p, device_cu_count)?;
        if p.cu_offset != cursor {
            // Gap or overlap relative to contiguous cover from 0.
            return Err(PartitionError::CoverageMismatch {
                covered: cursor,
                device_cu_count,
            });
        }
        cursor = cursor
            .checked_add(p.cu_count)
            .ok_or(PartitionError::OutOfBounds {
                cu_offset: p.cu_offset,
                cu_count: p.cu_count,
                device_cu_count,
            })?;
    }
    if cursor != device_cu_count {
        return Err(PartitionError::CoverageMismatch {
            covered: cursor,
            device_cu_count,
        });
    }
    Ok(())
}

/// Validate an equal-parts layout the same way dispatch does for
/// `PartitionPolicy::Equal(parts)`.
///
/// `parts` is the partition count (not CUs per part). Returns
/// [`PartitionError::NonDividing`] when `device_cu_count` is not divisible by
/// `parts`, [`PartitionError::ZeroCount`] when `parts == 0`, and
/// [`PartitionError::ZeroDevice`] when the device has zero CUs.
pub fn validate_equal_parts(device_cu_count: u32, parts: u32) -> PartitionResult<u32> {
    if device_cu_count == 0 {
        return Err(PartitionError::ZeroDevice);
    }
    if parts == 0 {
        return Err(PartitionError::ZeroCount);
    }
    if device_cu_count % parts != 0 {
        return Err(PartitionError::NonDividing {
            device_cu_count,
            parts,
        });
    }
    Ok(device_cu_count / parts)
}

/// Build contiguous equal [`PartitionContext`] slices for `parts` workers.
///
/// Fails with the same non-divide / zero rules as [`validate_equal_parts`].
/// `exclusive` is applied uniformly to every produced context.
pub fn equal_partitions(
    device_cu_count: u32,
    parts: u32,
    exclusive: bool,
) -> PartitionResult<Vec<PartitionContext>> {
    let cu_count = validate_equal_parts(device_cu_count, parts)?;
    debug_assert!(cu_count > 0);
    Ok((0..parts)
        .map(|index| PartitionContext {
            cu_offset: index * cu_count,
            cu_count,
            exclusive,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{from_str, to_string, to_value, Value};

    #[test]
    fn partition_context_serde_round_trip() {
        let ctx = PartitionContext {
            cu_offset: 16,
            cu_count: 16,
            exclusive: true,
        };
        let json = to_string(&ctx).expect("serialize");
        let back: PartitionContext = from_str(&json).expect("deserialize");
        assert_eq!(back, ctx);

        let value = to_value(ctx).expect("to_value");
        assert_eq!(value["cu_offset"], 16);
        assert_eq!(value["cu_count"], 16);
        assert_eq!(value["exclusive"], true);
    }

    #[test]
    fn partition_aware_recipe_serde_round_trip() {
        #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
        struct DummyRecipe {
            id: String,
            revision: u32,
        }

        let wrapped = PartitionAwareRecipe {
            partition: PartitionContext {
                cu_offset: 0,
                cu_count: 32,
                exclusive: false,
            },
            inner: DummyRecipe {
                id: "demo".into(),
                revision: 1,
            },
        };

        let json = to_string(&wrapped).expect("serialize");
        let back: PartitionAwareRecipe<DummyRecipe> = from_str(&json).expect("deserialize");
        assert_eq!(back, wrapped);

        let value: Value = from_str(&json).expect("json value");
        assert_eq!(value["partition"]["cu_offset"], 0);
        assert_eq!(value["partition"]["cu_count"], 32);
        assert_eq!(value["partition"]["exclusive"], false);
        assert_eq!(value["inner"]["id"], "demo");
        assert_eq!(value["inner"]["revision"], 1);
    }

    #[test]
    fn validate_accepts_aligned_dividing_slice() {
        let ctx = PartitionContext {
            cu_offset: 16,
            cu_count: 16,
            exclusive: true,
        };
        assert_eq!(validate_partition_context(&ctx, 64), Ok(()));
        assert_eq!(ctx.validate(64), Ok(()));
    }

    #[test]
    fn validate_accepts_heterogeneous_explicit_slice() {
        // Dispatch Explicit([8, 24, 32]) — middle slice must not be rejected
        // solely because 24 does not divide 64.
        let mid = PartitionContext {
            cu_offset: 8,
            cu_count: 24,
            exclusive: true,
        };
        assert_eq!(validate_partition_context(&mid, 64), Ok(()));

        let set = [
            PartitionContext {
                cu_offset: 0,
                cu_count: 8,
                exclusive: true,
            },
            mid,
            PartitionContext {
                cu_offset: 32,
                cu_count: 32,
                exclusive: true,
            },
        ];
        assert_eq!(validate_partition_set(&set, 64), Ok(()));
    }

    #[test]
    fn validate_rejects_zero_count() {
        let ctx = PartitionContext {
            cu_offset: 0,
            cu_count: 0,
            exclusive: false,
        };
        assert_eq!(
            validate_partition_context(&ctx, 64),
            Err(PartitionError::ZeroCount)
        );
    }

    #[test]
    fn validate_rejects_zero_device() {
        let ctx = PartitionContext {
            cu_offset: 0,
            cu_count: 1,
            exclusive: false,
        };
        assert_eq!(
            validate_partition_context(&ctx, 0),
            Err(PartitionError::ZeroDevice)
        );
        assert_eq!(
            validate_equal_parts(0, 4),
            Err(PartitionError::ZeroDevice)
        );
        assert_eq!(
            equal_partitions(0, 4, false),
            Err(PartitionError::ZeroDevice)
        );
    }

    #[test]
    fn validate_rejects_out_of_bounds() {
        let ctx = PartitionContext {
            cu_offset: 48,
            cu_count: 32,
            exclusive: true,
        };
        assert_eq!(
            validate_partition_context(&ctx, 64),
            Err(PartitionError::OutOfBounds {
                cu_offset: 48,
                cu_count: 32,
                device_cu_count: 64,
            })
        );
    }

    #[test]
    fn validate_set_rejects_gap_or_incomplete() {
        let gap = [
            PartitionContext {
                cu_offset: 0,
                cu_count: 8,
                exclusive: false,
            },
            // skip 8..16
            PartitionContext {
                cu_offset: 16,
                cu_count: 48,
                exclusive: false,
            },
        ];
        assert!(matches!(
            validate_partition_set(&gap, 64),
            Err(PartitionError::CoverageMismatch { .. })
        ));

        let short = [PartitionContext {
            cu_offset: 0,
            cu_count: 32,
            exclusive: false,
        }];
        assert_eq!(
            validate_partition_set(&short, 64),
            Err(PartitionError::CoverageMismatch {
                covered: 32,
                device_cu_count: 64,
            })
        );
    }

    #[test]
    fn equal_parts_non_divide_is_error() {
        assert_eq!(
            validate_equal_parts(64, 3),
            Err(PartitionError::NonDividing {
                device_cu_count: 64,
                parts: 3,
            })
        );
        assert_eq!(validate_equal_parts(64, 0), Err(PartitionError::ZeroCount));
        assert_eq!(validate_equal_parts(64, 4), Ok(16));
    }

    #[test]
    fn equal_partitions_builds_contiguous_slices() {
        let parts = equal_partitions(64, 4, true).expect("divides");
        assert_eq!(
            parts,
            vec![
                PartitionContext {
                    cu_offset: 0,
                    cu_count: 16,
                    exclusive: true,
                },
                PartitionContext {
                    cu_offset: 16,
                    cu_count: 16,
                    exclusive: true,
                },
                PartitionContext {
                    cu_offset: 32,
                    cu_count: 16,
                    exclusive: true,
                },
                PartitionContext {
                    cu_offset: 48,
                    cu_count: 16,
                    exclusive: true,
                },
            ]
        );
        validate_partition_set(&parts, 64).expect("equal set covers device");
        for part in &parts {
            part.validate(64).expect("each slice valid");
        }
    }

    #[test]
    fn partition_aware_recipe_validate_delegates() {
        let recipe = PartitionAwareRecipe::new(
            PartitionContext {
                cu_offset: 60,
                cu_count: 8,
                exclusive: true,
            },
            "inner-id",
        );
        // 60+8 > 64 → OOB (single-slice no longer requires divide).
        assert_eq!(
            recipe.validate_partition(64),
            Err(PartitionError::OutOfBounds {
                cu_offset: 60,
                cu_count: 8,
                device_cu_count: 64,
            })
        );
    }

    #[test]
    fn map_inner_preserves_partition() {
        let recipe = PartitionAwareRecipe::new(
            PartitionContext {
                cu_offset: 0,
                cu_count: 32,
                exclusive: false,
            },
            7u32,
        );
        let mapped = recipe.map_inner(|n| n * 2);
        assert_eq!(mapped.inner, 14);
        assert_eq!(mapped.partition.cu_count, 32);
    }
}
