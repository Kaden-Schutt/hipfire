// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Pure per-tier partition of a top-k expert selection (CPU, unit-tested).
//!
//! For intra-layer mixed-tier MoE: a single layer's routed experts may live at
//! DIFFERENT quant tiers. [`bucket_topk_by_tier`] splits the `k` selected
//! experts into one bucket per distinct tier present, so the existing
//! single-tier indexed GEMV kernels can be called once per tier over that
//! tier's subset of the top-k.

use rdna_compute::DType;

/// One tier's slice of the selected experts: the tier's DType plus the
/// (rank, expert_index) pairs whose expert is at that tier. `rank` is the
/// position in the original top-k (0..k); `expert_index` is the routed-expert
/// id the kernel uses to index `expert_*_ptrs`.
#[derive(Debug, Clone, PartialEq)]
pub struct TierBucket {
    pub tier: DType,
    pub ranks: Vec<usize>, // positions in the top-k
    pub experts: Vec<u32>, // expert ids (parallel to ranks)
}

/// Partition the `k` selected experts (`topk` = expert ids, len k) by their
/// per-expert tier (`tier_of[expert_id]`). Returns one bucket per distinct
/// tier PRESENT among the selected experts, in first-seen order. A layer whose
/// selected experts are all one tier yields exactly ONE bucket (== uniform).
///
/// `topk` carries expert ids read back from the GPU router; this function does
/// NOT trust them blindly — an id `>= tier_of.len()` (corrupt routing output, a
/// kernel bug, or a mis-sized tier table) returns `Err(id)` instead of panicking
/// on an out-of-bounds `tier_of[id]`.
pub fn bucket_topk_by_tier(topk: &[u32], tier_of: &[DType]) -> Result<Vec<TierBucket>, u32> {
    let mut buckets: Vec<TierBucket> = Vec::new();
    for (rank, &e) in topk.iter().enumerate() {
        let tier = *tier_of.get(e as usize).ok_or(e)?;
        match buckets.iter_mut().find(|b| b.tier == tier) {
            Some(b) => {
                b.ranks.push(rank);
                b.experts.push(e);
            }
            None => buckets.push(TierBucket {
                tier,
                ranks: vec![rank],
                experts: vec![e],
            }),
        }
    }
    Ok(buckets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType::*;
    #[test]
    fn uniform_layer_is_single_bucket() {
        let topk = [3u32, 7, 1, 5];
        let tier_of = vec![MQ4G256; 8];
        let b = bucket_topk_by_tier(&topk, &tier_of).unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].tier, MQ4G256);
        assert_eq!(b[0].ranks, vec![0, 1, 2, 3]);
        assert_eq!(b[0].experts, vec![3, 7, 1, 5]);
    }
    #[test]
    fn mixed_layer_partitions_by_tier_preserving_rank() {
        // experts: 0,2->MQ4 ; 1,3->MQ6
        let tier_of = vec![MQ4G256, MQ6G256, MQ4G256, MQ6G256];
        let topk = [1u32, 0, 3, 2]; // ranks 0..3
        let b = bucket_topk_by_tier(&topk, &tier_of).unwrap();
        assert_eq!(b.len(), 2);
        // first-seen tier is MQ6 (expert 1 at rank 0)
        assert_eq!(b[0].tier, MQ6G256);
        assert_eq!(b[0].ranks, vec![0, 2]);
        assert_eq!(b[0].experts, vec![1, 3]);
        assert_eq!(b[1].tier, MQ4G256);
        assert_eq!(b[1].ranks, vec![1, 3]);
        assert_eq!(b[1].experts, vec![0, 2]);
    }
    #[test]
    fn out_of_range_expert_id_is_err_not_panic() {
        let tier_of = vec![MQ4G256, MQ6G256]; // only 2 experts
        let topk = [0u32, 5]; // expert 5 is out of range
        assert_eq!(bucket_topk_by_tier(&topk, &tier_of), Err(5));
    }
}
