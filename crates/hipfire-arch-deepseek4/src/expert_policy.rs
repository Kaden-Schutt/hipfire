// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline eviction-policy simulator for the routed-expert cache.
//!
//! This is the P0 gate from `docs/specs/2026-07-19-weight-pager-eviction-policy.md`
//! (Kaden Schutt): before changing the pager's eviction policy, measure whether
//! the current one actually costs anything. That spec cites SpecMD
//! (arXiv:2602.03921), which found MoE expert access does NOT follow temporal
//! locality — reuse is predictable from routing STRUCTURE rather than recency,
//! so LRU/LFU underperform a staleness-aware policy.
//!
//! The spec's stop rule is the point of this module: **if LRU is within ~2% of
//! Belady at a realistic cap, record that and stop.** Policy work is only
//! warranted if the gap is real on our routing distributions, and that is a
//! question about traces, not about hardware — so it is answered here, offline,
//! with no GPU and no model load.
//!
//! MEASURED on a full 43-layer trace (137,742 accesses, 86 buckets, all 256
//! experts, 256 decode tokens, 8 GB cache):
//!
//! ```text
//!  slots   Belady     LRU     LFU  LeastStale   LRU vs Belady
//!      8    46.6%   66.6%   68.4%      93.4%     +20.0 pp
//!     16    33.0%   49.9%   55.5%      83.2%     +16.9 pp
//!     28    24.2%   38.3%   43.4%      64.3%     +14.1 pp   <- operating point
//!     64    14.3%   23.0%   24.6%      24.8%      +8.7 pp
//!    128    10.9%   13.1%   13.3%      13.4%      +2.2 pp
//!    192    10.4%   11.0%   11.0%      11.0%      +0.6 pp
//! ```
//!
//! Two conclusions, and they pull in opposite directions:
//!
//! 1. **The stop rule does NOT fire.** LRU takes 1.58x the misses Belady does
//!    at 28 slots. There is real headroom for a better policy — unlike the
//!    earlier 6-layer measurement, which showed a 0.3 pp gap and was WRONG:
//!    layers 0-3 are hash-routed with repetitive routing, so a trace confined
//!    to the first 6 layers is mostly hash layers and says nothing about the
//!    39 score-routed ones.
//!
//! 2. **Neither candidate replacement captures it.** Least-Stale is far worse
//!    (64.3% vs 38.3%) and LFU is worse (43.4%), so SpecMD's finding does not
//!    transfer to ds4 routing. Implementing P1 as specified would be a
//!    regression.
//!
//! So the gap is open, not closed: it needs a policy neither recency nor
//! frequency nor observed-staleness expresses. One untested idea is to use the
//! router's own gate scores — an expert selected with a low weight is less
//! likely to recur than one selected strongly — which is information the
//! pager currently throws away.
//!
//! Simulation is per `(layer, role)` bucket because that is how the real pager
//! is organised: each bucket owns `slots` cache entries and evicts within
//! itself. A trace is replayed independently per bucket and the totals summed.

use std::collections::HashMap;

use crate::expert_pager::ExpertBlobRole;

/// One expert access, in dispatch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertAccess {
    pub token: u32,
    pub layer: u16,
    pub role: ExpertBlobRole,
    pub expert: u16,
}

/// Eviction policies to compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// What the pager ships today.
    Lru,
    /// Evict the least frequently used. SpecMD's other recency-ish baseline.
    Lfu,
    /// Evict the entry whose next use is furthest away, using the actual
    /// future. Not implementable online — this is the optimal lower bound on
    /// misses, and the yardstick the stop rule is written against.
    Belady,
    /// SpecMD's Least-Stale: evict the entry with the longest PREDICTED time
    /// to next use, estimated from observed reuse intervals rather than from
    /// recency. Online-implementable.
    LeastStale,
}

/// Miss/hit counts for one simulated run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimResult {
    pub hits: u64,
    pub misses: u64,
}

impl SimResult {
    pub fn total(&self) -> u64 {
        self.hits + self.misses
    }
    pub fn miss_rate(&self) -> f64 {
        if self.total() == 0 {
            0.0
        } else {
            self.misses as f64 / self.total() as f64
        }
    }
}

/// Replay `trace` through a cache of `slots` entries per (layer, role) bucket.
///
/// Compulsory misses (first touch of an expert) are counted, matching the real
/// pager: it starts with empty slots and reads on first use.
pub fn simulate(trace: &[ExpertAccess], slots: usize, policy: Policy) -> SimResult {
    let mut buckets: HashMap<(u16, ExpertBlobRole), Vec<usize>> = HashMap::new();
    for (i, a) in trace.iter().enumerate() {
        buckets.entry((a.layer, a.role)).or_default().push(i);
    }
    let mut out = SimResult::default();
    for idxs in buckets.values() {
        let seq: Vec<u16> = idxs.iter().map(|&i| trace[i].expert).collect();
        let r = simulate_bucket(&seq, slots, policy);
        out.hits += r.hits;
        out.misses += r.misses;
    }
    out
}

/// Single-bucket replay. `seq` is that bucket's expert accesses in order.
fn simulate_bucket(seq: &[u16], slots: usize, policy: Policy) -> SimResult {
    let mut out = SimResult::default();
    if slots == 0 {
        out.misses = seq.len() as u64;
        return out;
    }
    // Next-use index per position, for Belady.
    let next_use: Vec<usize> = if policy == Policy::Belady {
        let mut nu = vec![usize::MAX; seq.len()];
        let mut last: HashMap<u16, usize> = HashMap::new();
        for i in (0..seq.len()).rev() {
            nu[i] = *last.get(&seq[i]).unwrap_or(&usize::MAX);
            last.insert(seq[i], i);
        }
        nu
    } else {
        Vec::new()
    };

    let mut resident: Vec<u16> = Vec::with_capacity(slots);
    // Per-policy bookkeeping.
    let mut last_used: HashMap<u16, usize> = HashMap::new();
    let mut freq: HashMap<u16, u64> = HashMap::new();
    // Least-Stale: exponential moving average of the interval between uses.
    let mut mean_interval: HashMap<u16, f64> = HashMap::new();

    for (i, &e) in seq.iter().enumerate() {
        let hit = resident.contains(&e);
        if hit {
            out.hits += 1;
        } else {
            out.misses += 1;
            if resident.len() == slots {
                let victim_pos = match policy {
                    Policy::Lru => resident
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, r)| last_used.get(r).copied().unwrap_or(0))
                        .map(|(p, _)| p)
                        .unwrap_or(0),
                    Policy::Lfu => resident
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, r)| freq.get(r).copied().unwrap_or(0))
                        .map(|(p, _)| p)
                        .unwrap_or(0),
                    Policy::Belady => {
                        // Furthest next use among resident entries.
                        let mut best = (0usize, 0usize); // (pos, next_use)
                        for (p, r) in resident.iter().enumerate() {
                            let nu = seq[i..]
                                .iter()
                                .position(|x| x == r)
                                .map(|d| i + d)
                                .unwrap_or(usize::MAX);
                            if nu >= best.1 {
                                best = (p, nu);
                            }
                        }
                        best.0
                    }
                    Policy::LeastStale => {
                        // Longest predicted time-to-next-use: time since last
                        // use minus its typical reuse interval. An entry well
                        // past its usual interval is the least likely to be
                        // needed soon.
                        let mut best = (0usize, f64::NEG_INFINITY);
                        for (p, r) in resident.iter().enumerate() {
                            let since = i.saturating_sub(last_used.get(r).copied().unwrap_or(0));
                            let mi = mean_interval.get(r).copied().unwrap_or(f64::INFINITY);
                            let staleness = if mi.is_finite() {
                                since as f64 - mi
                            } else {
                                // Never reused: treat as maximally stale.
                                f64::MAX
                            };
                            if staleness > best.1 {
                                best = (p, staleness);
                            }
                        }
                        best.0
                    }
                };
                resident.swap_remove(victim_pos);
            }
            resident.push(e);
        }
        // Update reuse statistics.
        if let Some(&prev) = last_used.get(&e) {
            let gap = (i - prev) as f64;
            let m = mean_interval.entry(e).or_insert(gap);
            *m = 0.5 * *m + 0.5 * gap;
        }
        last_used.insert(e, i);
        *freq.entry(e).or_insert(0) += 1;
    }
    let _ = &next_use;
    out
}

/// Parse a trace file written by `HIPFIRE_DEEPSEEK4_EXPERT_TRACE`.
/// Format: one `token,layer,role,expert` record per line, role `g`|`d`.
pub fn parse_trace(text: &str) -> Vec<ExpertAccess> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut f = line.split(',');
        let (Some(t), Some(l), Some(r), Some(e)) = (f.next(), f.next(), f.next(), f.next()) else {
            continue;
        };
        let (Ok(token), Ok(layer), Ok(expert)) =
            (t.trim().parse(), l.trim().parse(), e.trim().parse())
        else {
            continue;
        };
        let role = match r.trim() {
            "g" => ExpertBlobRole::GateUp,
            "d" => ExpertBlobRole::Down,
            _ => continue,
        };
        out.push(ExpertAccess {
            token,
            layer,
            role,
            expert,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc(token: u32, expert: u16) -> ExpertAccess {
        ExpertAccess {
            token,
            layer: 0,
            role: ExpertBlobRole::GateUp,
            expert,
        }
    }

    #[test]
    fn every_first_touch_is_a_compulsory_miss() {
        let t: Vec<ExpertAccess> = (0..5).map(|i| acc(i, i as u16)).collect();
        for p in [Policy::Lru, Policy::Lfu, Policy::Belady, Policy::LeastStale] {
            let r = simulate(&t, 8, p);
            assert_eq!(r.misses, 5, "{p:?}");
            assert_eq!(r.hits, 0, "{p:?}");
        }
    }

    #[test]
    fn a_cache_big_enough_never_evicts() {
        // 4 experts, repeated: 4 compulsory misses then all hits.
        let mut t = Vec::new();
        for round in 0..5u32 {
            for e in 0..4u16 {
                t.push(acc(round, e));
            }
        }
        for p in [Policy::Lru, Policy::Lfu, Policy::Belady, Policy::LeastStale] {
            let r = simulate(&t, 4, p);
            assert_eq!(r.misses, 4, "{p:?} should only take compulsory misses");
            assert_eq!(r.hits, 16, "{p:?}");
        }
    }

    #[test]
    fn belady_is_never_worse_than_any_online_policy() {
        // The defining property — if this fails the simulator is wrong and no
        // conclusion drawn from it means anything.
        let seq: Vec<u16> = (0..400u16).map(|i| (i * 7 % 23) as u16).collect();
        let t: Vec<ExpertAccess> = seq
            .iter()
            .enumerate()
            .map(|(i, &e)| acc(i as u32, e))
            .collect();
        for slots in [2usize, 3, 5, 8, 13] {
            let bel = simulate(&t, slots, Policy::Belady);
            for p in [Policy::Lru, Policy::Lfu, Policy::LeastStale] {
                let o = simulate(&t, slots, p);
                assert!(
                    bel.misses <= o.misses,
                    "Belady ({}) worse than {p:?} ({}) at {slots} slots",
                    bel.misses,
                    o.misses
                );
            }
        }
    }

    #[test]
    fn lru_loses_badly_on_a_cyclic_pattern() {
        // The textbook LRU pathology, and the shape SpecMD says MoE routing
        // has: a cycle one larger than the cache evicts exactly the entry
        // about to be used. Establishes the simulator can SEE a policy gap —
        // a simulator that reported "no difference" everywhere would be
        // useless as a stop rule.
        let cycle = 6u16;
        let seq: Vec<u16> = (0..600u32).map(|i| (i % cycle as u32) as u16).collect();
        let t: Vec<ExpertAccess> = seq
            .iter()
            .enumerate()
            .map(|(i, &e)| acc(i as u32, e))
            .collect();
        let slots = 5; // one less than the cycle
        let lru = simulate(&t, slots, Policy::Lru);
        let bel = simulate(&t, slots, Policy::Belady);
        assert!(
            lru.miss_rate() > 0.9,
            "LRU should thrash on a cycle of {cycle} with {slots} slots, got {:.3}",
            lru.miss_rate()
        );
        assert!(
            bel.miss_rate() < lru.miss_rate() / 2.0,
            "Belady {:.3} should be far better than LRU {:.3}",
            bel.miss_rate(),
            lru.miss_rate()
        );
    }

    #[test]
    fn buckets_are_simulated_independently() {
        // Layer 0 and layer 1 each get their own `slots` entries, exactly as
        // the real pager does. Interleaving two layers must not halve capacity.
        let mut t = Vec::new();
        for round in 0..4u32 {
            for l in 0..2u16 {
                for e in 0..3u16 {
                    t.push(ExpertAccess {
                        token: round,
                        layer: l,
                        role: ExpertBlobRole::GateUp,
                        expert: e,
                    });
                }
            }
        }
        let r = simulate(&t, 3, Policy::Lru);
        assert_eq!(
            r.misses, 6,
            "3 experts x 2 layers compulsory, then all hits"
        );
    }

    #[test]
    fn roles_are_separate_buckets() {
        let mut t = Vec::new();
        for round in 0..3u32 {
            for role in [ExpertBlobRole::GateUp, ExpertBlobRole::Down] {
                for e in 0..2u16 {
                    t.push(ExpertAccess {
                        token: round,
                        layer: 0,
                        role,
                        expert: e,
                    });
                }
            }
        }
        let r = simulate(&t, 2, Policy::Lru);
        assert_eq!(r.misses, 4, "2 experts x 2 roles compulsory");
    }

    #[test]
    fn trace_parsing_round_trips_and_skips_junk() {
        let text = "# comment\n0,3,g,17\n1,3,d,4\n\ngarbage\n2,0,x,9\n3,1,g,not_a_number\n";
        let t = parse_trace(text);
        assert_eq!(t.len(), 2, "only the two well-formed rows: {t:?}");
        assert_eq!(t[0].layer, 3);
        assert_eq!(t[0].role, ExpertBlobRole::GateUp);
        assert_eq!(t[0].expert, 17);
        assert_eq!(t[1].role, ExpertBlobRole::Down);
        assert_eq!(t[1].expert, 4);
    }
}
