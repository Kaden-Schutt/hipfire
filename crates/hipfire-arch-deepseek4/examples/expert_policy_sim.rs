// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Replay a routed-expert access trace through candidate eviction policies.
//!
//! The P0 gate from `docs/specs/2026-07-19-weight-pager-eviction-policy.md`:
//! decide whether the pager's eviction policy is worth changing by measuring
//! the LRU-vs-Belady gap on real routing, offline. The spec's stop rule is
//! "if LRU is within ~2% of Belady, record that and stop".
//!
//! Capture a trace with `HIPFIRE_DEEPSEEK4_EXPERT_TRACE=<path>`, then:
//!
//!   cargo run --release -p hipfire-arch-deepseek4 --example expert_policy_sim -- <path> [caps...]

use hipfire_arch_deepseek4::expert_policy::{parse_trace, simulate, Policy};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: expert_policy_sim <trace.csv> [slots...]");
            std::process::exit(2);
        }
    };
    let caps: Vec<usize> = {
        let v: Vec<usize> = args.filter_map(|a| a.parse().ok()).collect();
        if v.is_empty() {
            vec![4, 8, 16, 25, 32, 64, 128]
        } else {
            v
        }
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(2);
        }
    };
    let trace = parse_trace(&text);
    let mut buckets: std::collections::BTreeSet<(u16, &str)> = Default::default();
    let mut experts: std::collections::BTreeSet<u16> = Default::default();
    for a in &trace {
        buckets.insert((
            a.layer,
            match a.role {
                hipfire_arch_deepseek4::expert_pager::ExpertBlobRole::GateUp => "g",
                hipfire_arch_deepseek4::expert_pager::ExpertBlobRole::Down => "d",
            },
        ));
        experts.insert(a.expert);
    }
    println!(
        "trace: {} accesses, {} buckets (layer x role), {} distinct experts",
        trace.len(),
        buckets.len(),
        experts.len()
    );
    println!();
    println!(
        "{:>6}  {:>9}  {:>9}  {:>9}  {:>9}   {:>12}",
        "slots", "Belady", "LRU", "LFU", "LeastStale", "LRU vs Belady"
    );
    println!("{}", "-".repeat(70));
    for &c in &caps {
        let bel = simulate(&trace, c, Policy::Belady);
        let lru = simulate(&trace, c, Policy::Lru);
        let lfu = simulate(&trace, c, Policy::Lfu);
        let ls = simulate(&trace, c, Policy::LeastStale);
        // Gap in miss-rate percentage points, and as excess reads over optimal.
        let gap_pp = (lru.miss_rate() - bel.miss_rate()) * 100.0;
        let excess = lru.misses as f64 / bel.misses.max(1) as f64;
        println!(
            "{:>6}  {:>8.1}%  {:>8.1}%  {:>8.1}%  {:>8.1}%   {:>+6.1} pp  {:.2}x",
            c,
            bel.miss_rate() * 100.0,
            lru.miss_rate() * 100.0,
            lfu.miss_rate() * 100.0,
            ls.miss_rate() * 100.0,
            gap_pp,
            excess
        );
    }
    println!();
    println!("Stop rule (spec P0): if LRU is within ~2 pp of Belady, record and stop.");
}
