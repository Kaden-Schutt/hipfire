// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Prefix reuse: how much of a session's existing KV a new turn can keep.
//
// Pure logic, deliberately knowing nothing about sessions, slots or the GPU.
// `lcp` takes two token slices rather than a session, so a future
// cross-session shared-prefix table can supply the other side without any
// change here (spec §5.1, §10).

/// Length of the longest common prefix of two token sequences.
pub fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// What a turn can reuse and what it must compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnPlan {
    /// Tokens of `prompt` already represented in the slot's KV.
    pub lcp: usize,
    /// Same as `lcp`; named separately because it is the quantity the slot's
    /// `seq_len` is rewound to, and conflating "how many tokens are cached"
    /// with "what bounds attention" is exactly the SP1 Critical defect
    /// (`positions[]` vs `desc.seq_len`).
    pub reused: usize,
    /// Tokens that must actually be prefilled: `prompt.len() - lcp`.
    pub to_prefill: usize,
}

/// Decide how much of `prompt` a slot holding `seq_len` valid tokens of
/// `cached` can skip.
///
/// Two caps, both load-bearing:
///
/// 1. **`seq_len`** — the session may remember more tokens than the slot's KV
///    actually holds (it was swapped out and partially restored, or rewound by
///    an earlier turn). Reusing beyond that reads KV that was never written.
/// 2. **`prompt.len() - 1`** — a turn whose prompt exactly equals the cached
///    conversation would otherwise have nothing to prefill, and the forward
///    would produce no logits to sample. Always leave the last token.
pub fn plan_turn(cached: &[u32], seq_len: usize, prompt: &[u32]) -> TurnPlan {
    let shared = lcp(cached, prompt);
    let capped = shared.min(seq_len).min(prompt.len().saturating_sub(1));
    TurnPlan {
        lcp: capped,
        reused: capped,
        to_prefill: prompt.len() - capped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_counts_the_shared_leading_run() {
        assert_eq!(lcp(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(lcp(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(lcp(&[], &[1, 2]), 0);
        assert_eq!(lcp(&[1, 2], &[]), 0);
        assert_eq!(lcp(&[9], &[1]), 0);
    }

    #[test]
    fn a_continuing_turn_reuses_everything_but_the_last_token() {
        // Turn 2 is turn 1's conversation plus new user text.
        let cached = [1, 2, 3, 4];
        let prompt = [1, 2, 3, 4, 5, 6];
        let p = plan_turn(&cached, 4, &prompt);
        assert_eq!(p.lcp, 4);
        assert_eq!(p.reused, 4);
        assert_eq!(p.to_prefill, 2, "only the new tokens are prefilled");
    }

    #[test]
    fn an_identical_prompt_still_leaves_one_token_to_prefill() {
        // Without this the forward gets an empty batch and produces no logits.
        let cached = [1, 2, 3, 4];
        let p = plan_turn(&cached, 4, &[1, 2, 3, 4]);
        assert_eq!(p.lcp, 3, "must stop one short of the whole prompt");
        assert_eq!(p.to_prefill, 1);
    }

    #[test]
    fn reuse_is_capped_by_what_the_slot_actually_holds() {
        // The session remembers 4 tokens but the slot only has 2 of them.
        let cached = [1, 2, 3, 4];
        let p = plan_turn(&cached, 2, &[1, 2, 3, 4, 5]);
        assert_eq!(p.lcp, 2, "must not claim KV the slot does not hold");
        assert_eq!(p.to_prefill, 3);
    }

    #[test]
    fn a_diverging_turn_falls_back_to_near_full_prefill() {
        let cached = [1, 2, 3, 4];
        let p = plan_turn(&cached, 4, &[9, 9, 9]);
        assert_eq!(p.lcp, 0);
        assert_eq!(p.reused, 0);
        assert_eq!(p.to_prefill, 3);
    }

    #[test]
    fn a_cold_session_prefills_the_whole_prompt() {
        let p = plan_turn(&[], 0, &[1, 2, 3]);
        assert_eq!(p.lcp, 0);
        assert_eq!(p.to_prefill, 3);
    }
}
