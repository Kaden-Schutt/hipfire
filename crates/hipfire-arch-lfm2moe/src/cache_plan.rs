// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Strict forward-extension prompt-cache planner for LFM2.5 (arch 11).
//!
//! Pure host-side decision only — no GPU, tokenizer, or daemon side effects.
//! The daemon renders the canonical conversation, then calls
//! [`plan_lfm_prompt_cache`]; on a miss it MUST reset LFM GPU recurrent state
//! (KV cursor + conv rolling state) and full-prefill `rendered`. On a hit it
//! MUST NOT reset and prefills only `new_tokens` at `start_pos == state.n_tokens`.
//!
//! HIT predicate (all required):
//! - eviction is inactive (`eviction_is_none`)
//! - `lcp == prior.len() == state_n_tokens`
//! - `rendered.len() > lcp` (non-empty strict suffix; exact full-match is a miss)
//!
//! Any divergence, stale `n_tokens` cursor, empty prior, empty suffix, or
//! exact full-match is a miss. There is no partial-prefix reuse and no
//! checkpoint resume: LFM's depthwise conv window chains back to token 0.

/// Outcome of [`plan_lfm_prompt_cache`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LfmPromptCachePlan {
    /// Full canonical conversation tokens for this turn. Caller stores this
    /// (plus any newly generated tokens) as `conversation_tokens` after the
    /// turn so the next request can LCP against it.
    pub rendered: Vec<u32>,
    /// Tokens to prefill: `rendered[start_pos..]` on a hit, the whole
    /// `rendered` on a miss.
    pub new_tokens: Vec<u32>,
    /// Absolute position the prefill starts at (== cached prefix length on a
    /// hit, 0 on a miss).
    pub start_pos: usize,
    /// Prefix length already resident in model state (== `start_pos`).
    pub cached_tokens: usize,
    /// `true` ⇒ reuse existing LFM KV + conv state `[0..start_pos)` and
    /// prefill only the suffix. `false` ⇒ caller must reset and full-prefill.
    pub cache_hit: bool,
    /// `true` when `rendered.len() + max_tokens > max_seq`. Caller must refuse
    /// the request before prefill (hit or miss); the planner still returns a
    /// well-formed miss/hit plan so telemetry stays consistent.
    pub over_capacity: bool,
}

impl LfmPromptCachePlan {
    /// Canonical miss over `rendered`: full prefill from position 0.
    #[inline]
    pub fn miss(rendered: Vec<u32>, over_capacity: bool) -> Self {
        let new_tokens = rendered.clone();
        Self {
            rendered,
            new_tokens,
            start_pos: 0,
            cached_tokens: 0,
            cache_hit: false,
            over_capacity,
        }
    }
}

/// Pure LCP prompt-cache decision for LFM2.5.
///
/// # Arguments
/// - `rendered`         — fully-rendered canonical conversation tokens for
///                        this turn (already built by the caller, typically
///                        via `build_cached_history_jinja` with asst-turn
///                        replay so the stream byte-matches the prior turn).
/// - `prior`            — `conversation_tokens` retained from the previous turn
///                        (prompt + generated tokens).
/// - `state_n_tokens`   — current `Lfm2MoeState::n_tokens` (must equal
///                        `prior.len()` for a hit; any skew is a stale cursor).
/// - `eviction_is_none` — `m.eviction.is_none()`. LFM currently has no eviction
///                        path; when eviction is ever wired this forces a miss
///                        because compact_offset would invalidate the
///                        conversation↔KV mirror the cache relies on.
/// - `cache_enabled`   — request-time kill switch (`HIPFIRE_QWEN_PROMPT_CACHE != "0"`).
///                        `false` forces the miss arm without bypassing the capacity
///                        calculation.
/// - `max_seq`          — KV / context capacity (`state.max_seq`).
/// - `max_tokens`       — requested generation budget for this turn.
///
/// # Hit rule
/// Strict forward extension only:
/// `lcp == prior.len() == state_n_tokens && rendered.len() > lcp` (and
/// eviction inactive, cache enabled, prior non-empty). Exact full-match
/// (`lcp == rendered.len()`) is intentionally a **miss** — there is no safe
/// forward progress and re-applying the last token would double-advance the
/// non-rewindable conv state.
pub fn plan_lfm_prompt_cache(
    rendered: Vec<u32>,
    prior: &[u32],
    state_n_tokens: usize,
    eviction_is_none: bool,
    cache_enabled: bool,
    max_seq: usize,
    max_tokens: usize,
) -> LfmPromptCachePlan {
    let over_capacity = rendered.len().saturating_add(max_tokens) > max_seq;

    // Empty render: nothing to prefill; treat as miss (caller errors upstream).
    if rendered.is_empty() {
        return LfmPromptCachePlan::miss(rendered, over_capacity);
    }

    // Cache disabled, eviction active, no prior, or GPU cursor doesn't mirror
    // prior tokens → never reuse. Stale n_tokens is the "cursor drifted /
    // partial abort" failure mode: prior bookkeeping and GPU state disagree.
    if !cache_enabled || !eviction_is_none || prior.is_empty() || state_n_tokens != prior.len() {
        return LfmPromptCachePlan::miss(rendered, over_capacity);
    }

    let prior_len = prior.len();
    let max_match = prior_len.min(rendered.len());
    let mut lcp = 0usize;
    while lcp < max_match && prior[lcp] == rendered[lcp] {
        lcp += 1;
    }

    // Strict forward extension: prior is a proper prefix of rendered AND the
    // GPU state cursor sits exactly at that prefix length.
    if lcp == prior_len && lcp == state_n_tokens && lcp < rendered.len() && lcp > 0 {
        return LfmPromptCachePlan {
            new_tokens: rendered[lcp..].to_vec(),
            start_pos: lcp,
            cached_tokens: lcp,
            cache_hit: true,
            over_capacity,
            rendered,
        };
    }

    // Divergence, exact full-match, or empty LCP → cold miss.
    LfmPromptCachePlan::miss(rendered, over_capacity)
}

/// Whether arch 11 can advertise `cache_capable` on the daemon `loaded`
/// response. Requires no eviction (LFM has none today) and a Jinja chat
/// template so multi-turn history can be rendered into a byte-stable
/// canonical stream for exact forward extension. Request-time conditions
/// (messages_history present, kill-switch off) are enforced by the planner
/// / generate path, not here.
#[inline]
pub fn lfm_cache_capable_advertised(eviction_is_none: bool, has_chat_template: bool) -> bool {
    eviction_is_none && has_chat_template
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_forward_extension_hit() {
        let p = plan_lfm_prompt_cache(
            vec![1, 2, 3, 4, 5],
            &[1, 2, 3],
            /*state_n_tokens=*/ 3,
            /*eviction_is_none=*/ true,
            /*cache_enabled=*/ true,
            /*max_seq=*/ 128,
            /*max_tokens=*/ 16,
        );
        assert!(p.cache_hit);
        assert_eq!(p.start_pos, 3);
        assert_eq!(p.cached_tokens, 3);
        assert_eq!(p.new_tokens, vec![4, 5]);
        assert_eq!(p.rendered, vec![1, 2, 3, 4, 5]);
        assert!(!p.over_capacity);
    }

    #[test]
    fn divergence_is_miss() {
        let p = plan_lfm_prompt_cache(vec![1, 2, 9, 10], &[1, 2, 3, 4], 4, true, true, 128, 16);
        assert!(!p.cache_hit);
        assert_eq!(p.start_pos, 0);
        assert_eq!(p.cached_tokens, 0);
        assert_eq!(p.new_tokens, vec![1, 2, 9, 10]);
        assert!(!p.over_capacity);
    }

    #[test]
    fn exact_full_match_is_miss() {
        // Re-sent identical prompt: no forward progress. Must NOT re-apply the
        // last token (would double-advance conv state).
        let p = plan_lfm_prompt_cache(vec![1, 2, 3], &[1, 2, 3], 3, true, true, 128, 16);
        assert!(!p.cache_hit);
        assert_eq!(p.start_pos, 0);
        assert_eq!(p.cached_tokens, 0);
        assert_eq!(p.new_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn stale_state_cursor_is_miss() {
        // Bookkeeping says prior has 3 tokens, but GPU n_tokens drifted to 2.
        let p = plan_lfm_prompt_cache(
            vec![1, 2, 3, 4, 5],
            &[1, 2, 3],
            /*state_n_tokens=*/ 2,
            true,
            true,
            128,
            16,
        );
        assert!(!p.cache_hit);
        assert_eq!(p.cached_tokens, 0);
        assert_eq!(p.new_tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn empty_suffix_via_exact_match_is_miss() {
        // Empty suffix is the exact-match edge; hit requires rendered.len() > lcp.
        let prior = vec![10u32, 20, 30];
        let p = plan_lfm_prompt_cache(prior.clone(), &prior, prior.len(), true, true, 64, 8);
        assert!(!p.cache_hit);
        assert!(p.new_tokens.len() == prior.len());
        assert_eq!(p.start_pos, 0);
    }

    #[test]
    fn capacity_guard_marks_over_capacity() {
        // rendered(5) + max_tokens(10) = 15 > max_seq(12).
        let p = plan_lfm_prompt_cache(vec![1, 2, 3, 4, 5], &[1, 2, 3], 3, true, true, 12, 10);
        assert!(p.over_capacity);
        // Capacity is orthogonal to the LCP decision — still a hit on tokens.
        assert!(p.cache_hit);
        assert_eq!(p.cached_tokens, 3);
        assert_eq!(p.new_tokens, vec![4, 5]);
    }

    #[test]
    fn capacity_guard_on_miss() {
        let p = plan_lfm_prompt_cache(vec![1, 2, 3, 4, 5], &[], 0, true, true, 8, 4);
        assert!(p.over_capacity);
        assert!(!p.cache_hit);
        assert_eq!(p.new_tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn empty_prior_is_miss() {
        let p = plan_lfm_prompt_cache(vec![1, 2, 3], &[], 0, true, true, 128, 16);
        assert!(!p.cache_hit);
        assert_eq!(p.cached_tokens, 0);
    }

    #[test]
    fn eviction_active_is_miss() {
        let p = plan_lfm_prompt_cache(
            vec![1, 2, 3, 4, 5],
            &[1, 2, 3],
            3,
            /*eviction_is_none=*/ false,
            /*cache_enabled=*/ true,
            128,
            16,
        );
        assert!(!p.cache_hit);
        assert_eq!(p.cached_tokens, 0);
    }

    #[test]
    fn cache_capable_advert_requires_template_and_no_eviction() {
        assert!(lfm_cache_capable_advertised(true, true));
        assert!(!lfm_cache_capable_advertised(false, true));
        assert!(!lfm_cache_capable_advertised(true, false));
        assert!(!lfm_cache_capable_advertised(false, false));
    }

    #[test]
    fn cache_disabled_is_miss_but_capacity_still_checked() {
        let p = plan_lfm_prompt_cache(vec![1, 2, 3, 4, 5], &[1, 2, 3], 3, true, false, 8, 4);
        assert!(!p.cache_hit);
        assert_eq!(p.cached_tokens, 0);
        assert_eq!(p.new_tokens, vec![1, 2, 3, 4, 5]);
        assert!(p.over_capacity);
    }
}
