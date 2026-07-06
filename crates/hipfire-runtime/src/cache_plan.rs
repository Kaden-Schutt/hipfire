// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure, side-effect-free prompt-cache planner.
//!
//! Unifies the two per-arch LCP cache decisions that previously lived
//! inline in `daemon.rs`:
//!
//! - **qwen35**: `plan_prompt_cache` (~lines 3811-3905 of `daemon.rs`)
//! - **deepseek4**: inline LCP block (~lines 9411-9482 of `daemon.rs`)
//!
//! Call [`plan_cache`] with the appropriate [`CachePolicy`] for the arch.
//! No GPU side-effects belong here; they move into the unified loop (T4).

/// Outcome of [`plan_cache`]: how much of the prior conversation is cached
/// and where the new prefill should start.
#[derive(Debug, PartialEq, Eq)]
pub struct CachePlan {
    /// Whether this turn can reuse the previous turn's recurrent state.
    pub cache_hit: bool,
    /// Token index in `rendered` at which the caller should begin prefilling.
    /// On a hit, `rendered[start_pos..]` is the suffix to feed to the model.
    pub start_pos: usize,
    /// Number of tokens from `rendered` that are already in the model's state
    /// (always equals `start_pos` — kept as a separate field for call-site
    /// clarity when accounting).
    pub cached_tokens: usize,
    /// For qwen35 resume-from-checkpoint: if `Some(p)`, the caller should
    /// rewind DeltaNet state to checkpoint `p` before prefilling the suffix.
    /// Always `None` for deepseek4 (no checkpoints).
    pub resume_from: Option<usize>,
}

impl CachePlan {
    /// Canonical miss: cold-prefill the entire rendered conversation.
    #[inline]
    pub fn miss() -> Self {
        CachePlan {
            cache_hit: false,
            start_pos: 0,
            cached_tokens: 0,
            resume_from: None,
        }
    }
}

/// How the planner handles an exact-match render (new render == prior, byte-for-byte).
#[derive(Debug, PartialEq, Eq)]
pub enum ExactMatch {
    /// qwen35: exact-match degrades to a miss (the 1-token DeltaNet
    /// over-advance that would result from advancing past the last token
    /// is not safe).
    Miss,
    /// deepseek4: step `lcp` back one so prefilling always processes ≥ 1
    /// token. The stepped-back `lcp` is then in the partial range
    /// `(0, prior_len)`, which the `allow_partial=false` guard immediately
    /// forces to a cold miss for DSA compressor-ring safety.
    StepBack,
}

/// Per-arch cache policy knobs.
///
/// Construct via [`CachePolicy::qwen35`] or [`CachePolicy::deepseek4`].
#[derive(Debug)]
pub struct CachePolicy {
    /// If `true`, a miss is forced when `rendered.len() < prior.len()`.
    ///
    /// **deepseek4** sets this `true` for DSA compressor-ring safety:
    /// `generate_deepseek4` (daemon.rs ~9413) checks
    /// `prompt_ids.len() < prior.len()` before computing LCP.
    /// **qwen35** sets this `false` — it has no such constraint.
    pub min_new_len_ge_prior: bool,
    /// What to do when the new render is byte-identical to the prior.
    ///
    /// See [`ExactMatch`] variants for the per-arch rationale.
    pub on_exact: ExactMatch,
    /// Whether a partial prefix match (`0 < lcp < prior_len`) is accepted
    /// as a cache hit.
    ///
    /// Both current arches set this `false`. The field is part of the
    /// documented policy surface so the unified loop (T4) can enable it
    /// for future arches without a new planner API.
    pub allow_partial: bool,
}

impl CachePolicy {
    /// Policy for **qwen35** (`plan_prompt_cache`, daemon.rs ~3811-3905).
    ///
    /// - No minimum-length constraint on the new render.
    /// - Exact-match (`lcp == rendered.len()`) → miss (avoids DeltaNet over-advance).
    /// - Partial divergence without a usable checkpoint → miss.
    /// - Resume-from-checkpoint is a call-site toggle, not a policy knob.
    pub fn qwen35() -> Self {
        CachePolicy {
            min_new_len_ge_prior: false,
            on_exact: ExactMatch::Miss,
            allow_partial: false,
        }
    }

    /// Policy for **deepseek4** (inline LCP, daemon.rs ~9411-9482).
    ///
    /// - Rendered must be at least as long as the prior (DSA compressor-ring safety).
    /// - Exact-match (`lcp == rendered.len()`) → step back one, which then
    ///   falls into the partial-cold guard and becomes a miss.
    /// - Any partial hit (`0 < lcp < prior_len`) → forced cold (DSA ring safety).
    pub fn deepseek4() -> Self {
        CachePolicy {
            min_new_len_ge_prior: true,
            on_exact: ExactMatch::StepBack,
            allow_partial: false,
        }
    }
}

/// Compute the prompt-cache plan for one turn.
///
/// # Arguments
/// - `rendered`        — the fully-rendered canonical conversation tokens for
///                       this turn (already built by the caller).
/// - `prior`           — `m.conversation_tokens` from the previous turn.
/// - `policy`          — per-arch knobs; see [`CachePolicy::qwen35`] /
///                       [`CachePolicy::deepseek4`].
/// - `checkpoints`     — ascending DeltaNet checkpoint positions
///                       (`m.dflash_checkpoints`); pass `&[]` for deepseek4.
/// - `resume_enabled`  — whether to attempt resume-from-checkpoint on
///                       divergence; `false` for deepseek4.
///
/// # Returns
/// A [`CachePlan`] whose `start_pos` is the index into `rendered` at which
/// the caller should begin prefilling. `cached_tokens == start_pos` always.
pub fn plan_cache(
    rendered: &[u32],
    prior: &[u32],
    policy: &CachePolicy,
    checkpoints: &[usize],
    resume_enabled: bool,
) -> CachePlan {
    // 1. No prior → miss.
    if prior.is_empty() {
        return CachePlan::miss();
    }
    // 2. ds4 ring-safety: new render must be at least as long as prior.
    if policy.min_new_len_ge_prior && rendered.len() < prior.len() {
        return CachePlan::miss();
    }
    // 3. Raw longest common prefix, bounded by both lengths.
    let max_match = prior.len().min(rendered.len());
    let mut lcp = 0usize;
    while lcp < max_match && prior[lcp] == rendered[lcp] {
        lcp += 1;
    }
    // 4. Exact-match edge: lcp consumed the WHOLE new render.
    if lcp == rendered.len() && lcp > 0 {
        match policy.on_exact {
            ExactMatch::Miss => return CachePlan::miss(),
            ExactMatch::StepBack => lcp -= 1, // falls into partial-cold below
        }
    }
    // 5. Pure forward extension → hit.
    if lcp == prior.len() && lcp < rendered.len() && lcp > 0 {
        return CachePlan {
            cache_hit: true,
            start_pos: lcp,
            cached_tokens: lcp,
            resume_from: None,
        };
    }
    // 6. Partial divergence (0 < lcp < prior_len).
    if lcp > 0 && lcp < prior.len() {
        if policy.allow_partial {
            return CachePlan {
                cache_hit: true,
                start_pos: lcp,
                cached_tokens: lcp,
                resume_from: None,
            };
        }
        if resume_enabled {
            if let Some(&ckpt) = checkpoints
                .iter()
                .filter(|&&p| p <= lcp && p < rendered.len())
                .max()
            {
                return CachePlan {
                    cache_hit: true,
                    start_pos: ckpt,
                    cached_tokens: ckpt,
                    resume_from: Some(ckpt),
                };
            }
        }
        return CachePlan::miss();
    }
    // 7. Otherwise miss (lcp == 0: total divergence).
    CachePlan::miss()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q() -> CachePolicy {
        CachePolicy::qwen35()
    }

    fn d() -> CachePolicy {
        CachePolicy::deepseek4()
    }

    fn check(plan: CachePlan, hit: bool, start: usize, resume: Option<usize>) {
        assert_eq!(plan.cache_hit, hit);
        assert_eq!(plan.start_pos, start);
        assert_eq!(
            plan.cached_tokens, start,
            "cached_tokens must equal start_pos"
        );
        assert_eq!(plan.resume_from, resume);
    }

    #[test]
    fn t01_empty_prior_miss() {
        // branch: step 1 — no prior → miss
        let plan = plan_cache(&[1, 2, 3], &[], &q(), &[], false);
        check(plan, false, 0, None);
    }

    #[test]
    fn t02_qwen35_forward_extension_hit() {
        // branch: step 5 — pure forward extension
        let plan = plan_cache(&[1, 2, 3, 4, 5], &[1, 2, 3], &q(), &[], false);
        check(plan, true, 3, None);
    }

    #[test]
    fn t03_ds4_forward_extension_hit() {
        // branch: step 5 — pure forward extension (ds4 policy)
        let plan = plan_cache(&[1, 2, 3, 4, 5], &[1, 2, 3], &d(), &[], false);
        check(plan, true, 3, None);
    }

    #[test]
    fn t04_qwen35_exact_match_miss() {
        // branch: step 4 — exact-match → ExactMatch::Miss → miss
        let plan = plan_cache(&[1, 2, 3], &[1, 2, 3], &q(), &[], false);
        check(plan, false, 0, None);
    }

    #[test]
    fn t05_ds4_exact_match_stepback_then_partial_cold() {
        // branch: step 4 → StepBack (lcp=2), then step 6 partial-cold → miss
        let plan = plan_cache(&[1, 2, 3], &[1, 2, 3], &d(), &[], false);
        check(plan, false, 0, None);
    }

    #[test]
    fn t06_qwen35_partial_no_resume_miss() {
        // branch: step 6 — partial divergence, resume_enabled=false → miss
        let plan = plan_cache(&[1, 2, 9, 9, 9], &[1, 2, 3, 4], &q(), &[], false);
        check(plan, false, 0, None);
    }

    #[test]
    fn t07_qwen35_partial_resume_latest_ckpt() {
        // branch: step 6 — partial divergence, resume finds latest ckpt ≤ lcp(2)
        let plan = plan_cache(&[1, 2, 9, 9, 9], &[1, 2, 3, 4], &q(), &[0, 1], true);
        check(plan, true, 1, Some(1));
    }

    #[test]
    fn t08_qwen35_partial_resume_ckpt_beyond_lcp_miss() {
        // branch: step 6 — resume_enabled but ckpt=3 > lcp(2), filtered → miss
        let plan = plan_cache(&[1, 2, 9, 9, 9], &[1, 2, 3, 4], &q(), &[3], true);
        check(plan, false, 0, None);
    }

    #[test]
    fn t09_ds4_partial_cold_miss() {
        // branch: step 6 — ds4 partial → allow_partial=false, no resume → miss
        let plan = plan_cache(&[1, 2, 9, 9, 9], &[1, 2, 3, 4], &d(), &[], false);
        check(plan, false, 0, None);
    }

    #[test]
    fn t10_ds4_rendered_shorter_than_prior_miss() {
        // branch: step 2 — min_new_len_ge_prior triggers miss
        let plan = plan_cache(&[1, 2, 3], &[1, 2, 3, 4, 5], &d(), &[], false);
        check(plan, false, 0, None);
    }

    #[test]
    fn t11_qwen35_rendered_shorter_full_prefix_exact_miss() {
        // branch: step 4 — rendered shorter, all 3 rendered tokens match →
        // lcp == rendered.len() → exact → ExactMatch::Miss
        let plan = plan_cache(&[1, 2, 3], &[1, 2, 3, 4, 5], &q(), &[], false);
        check(plan, false, 0, None);
    }

    #[test]
    fn t12_ds4_forward_extension_cached_tokens_eq_start_pos() {
        // branch: step 5 — forward extension; assert cached_tokens == start_pos == 3
        let plan = plan_cache(&[1, 2, 3, 4], &[1, 2, 3], &d(), &[], false);
        assert!(plan.cache_hit);
        assert_eq!(plan.start_pos, 3);
        assert_eq!(
            plan.cached_tokens, plan.start_pos,
            "cached_tokens must equal start_pos"
        );
    }
}
