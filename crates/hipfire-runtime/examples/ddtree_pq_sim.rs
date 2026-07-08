//! Phase-0 decision gate for the q-exploiting tree-verify plan
//! (docs/plans/2026-06-26-ddtree-q-exploiting-verify.md).
//!
//! Reads a `(p,q)` dump produced by `HIPFIRE_DDTREE_DUMP_PQ` (one JSONL record
//! per decode cycle; each slot is a reduced categorical `{child_1..child_k, TAIL}`
//! over the target `p` and draft conditional `q`) and Monte-Carlo simulates the
//! accept-LENGTH of four verify schemes on the SAME real trees:
//!
//!   greedy     — argmax descent (the control; temperature-invariant).
//!   naive      — sample x~target, accept the child it lands on (the LANDED baseline).
//!   swor       — q-exploiting sampling-without-replacement (Sequoia/SpecTr family,
//!                approximate residual; the candidate scheme).
//!   optimistic — accept child c w.p. min(1,p(c)/q(c)) with NO residual depletion;
//!                an UPPER BOUND on any q-exploiting scheme's acceptance.
//!
//! naive ≤ swor ≤ optimistic brackets the upside. Decision rule (from the plan):
//! if even `optimistic` barely beats `naive`, the real SWOR cannot justify the
//! draft-sampling + full-q GPU cost — stop. Run: `ddtree_pq_sim <dump.jsonl>`.

use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone)]
struct Slot {
    argmax_child: i64,
    children: Vec<(usize, f32, f32)>, // (child_slot, q, p)
}

struct Cycle {
    slots: HashMap<usize, Slot>,
}

#[inline]
fn xorshift_unit(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    ((s >> 40) as f32) * (1.0 / 16_777_216.0)
}

fn parse(path: &str) -> Vec<Cycle> {
    let txt = std::fs::read_to_string(path).expect("read dump");
    let mut cycles = Vec::new();
    for line in txt.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).expect("parse jsonl");
        let mut slots = HashMap::new();
        for sl in v["slots"].as_array().unwrap() {
            let s = sl["s"].as_u64().unwrap() as usize;
            let argmax_child = sl["argmax_child"].as_i64().unwrap();
            let children = sl["children"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    let a = c.as_array().unwrap();
                    (
                        a[0].as_u64().unwrap() as usize,
                        a[1].as_f64().unwrap() as f32,
                        a[2].as_f64().unwrap() as f32,
                    )
                })
                .collect();
            slots.insert(
                s,
                Slot {
                    argmax_child,
                    children,
                },
            );
        }
        cycles.push(Cycle { slots });
    }
    cycles
}

// Each scheme returns the next accepted child slot at `s`, or None to stop.
fn step_greedy(sl: &Slot) -> Option<usize> {
    if sl.argmax_child >= 0 {
        Some(sl.argmax_child as usize)
    } else {
        None
    }
}

fn step_naive(sl: &Slot, rng: &mut u64) -> Option<usize> {
    // Sample x~target over {children, TAIL}; land on a child or stop.
    let u = xorshift_unit(rng);
    let mut acc = 0.0f32;
    for &(cs, _q, p) in &sl.children {
        acc += p;
        if u < acc {
            return Some(cs);
        }
    }
    None // tail
}

fn step_optimistic(sl: &Slot, rng: &mut u64) -> Option<usize> {
    // Accept child c w.p. min(1, p/q), no residual depletion → upper bound.
    for &(cs, q, p) in &sl.children {
        let a = if q > 0.0 { (p / q).min(1.0) } else { 0.0 };
        if xorshift_unit(rng) < a {
            return Some(cs);
        }
    }
    None
}

fn step_swor(sl: &Slot, rng: &mut u64) -> Option<usize> {
    // Sampling-without-replacement q-exploiting acceptance on the reduced support
    // {children, TAIL} with an approximate residual (per-child renorm). Brackets
    // between naive and optimistic; the exact Sequoia/SpecTr residual is for the
    // GPU build (MC-validated), not this gate.
    let k = sl.children.len();
    let mut p_res: Vec<f32> = sl.children.iter().map(|&(_, _, p)| p).collect();
    let p_tail: f32 = (1.0 - p_res.iter().sum::<f32>()).max(0.0);
    p_res.push(p_tail); // TAIL bucket at index k
    let mut remaining_q = 1.0f32;
    for j in 0..k {
        let (cs, q, _p) = sl.children[j];
        let prop = if remaining_q > 1e-6 {
            q / remaining_q
        } else {
            q
        };
        let ratio = if prop > 0.0 {
            (p_res[j] / prop).min(1.0)
        } else {
            0.0
        };
        if xorshift_unit(rng) < ratio {
            return Some(cs);
        }
        // Reject: deplete residual target by the proposed mass, renormalize.
        p_res[j] = (p_res[j] - prop).max(0.0);
        let s: f32 = p_res.iter().sum();
        if s > 0.0 {
            for x in p_res.iter_mut() {
                *x /= s;
            }
        }
        remaining_q -= q;
        if remaining_q <= 1e-6 {
            break;
        }
    }
    None
}

fn walk<F: FnMut(&Slot, &mut u64) -> Option<usize>>(
    cyc: &Cycle,
    mut step: F,
    rng: &mut u64,
) -> usize {
    let mut len = 0usize;
    let mut s = 0usize;
    loop {
        let Some(sl) = cyc.slots.get(&s) else { break };
        if sl.children.is_empty() {
            break;
        }
        match step(sl, rng) {
            Some(cs) => {
                len += 1;
                s = cs;
            }
            None => break,
        }
    }
    len
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ddtree_pq_sim <dump.jsonl>");
    let cycles = parse(&path);
    let trials = 4000u32;
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;

    // greedy is deterministic; the rest are MC-averaged.
    let mut g = 0.0f64;
    for c in &cycles {
        g += walk(c, |sl, _| step_greedy(sl), &mut rng) as f64;
    }
    let mean = |total: f64| total / cycles.len() as f64;
    let g = mean(g);

    let mut mc = |mut step: Box<dyn FnMut(&Slot, &mut u64) -> Option<usize>>, rng: &mut u64| {
        let mut tot = 0.0f64;
        for c in &cycles {
            for _ in 0..trials {
                tot += walk(c, &mut step, rng) as f64;
            }
        }
        tot / (cycles.len() as f64 * trials as f64)
    };
    let naive = mc(Box::new(|sl, r| step_naive(sl, r)), &mut rng);
    let swor = mc(Box::new(|sl, r| step_swor(sl, r)), &mut rng);
    let opt = mc(Box::new(|sl, r| step_optimistic(sl, r)), &mut rng);

    let n_cyc = cycles.len();
    println!("ddtree (p,q) Phase-0 simulation — {n_cyc} cycles, {trials} trials/cycle");
    println!("mean accepted children / cycle (τ ≈ this + 1):");
    println!("  greedy      {g:.3}");
    println!(
        "  naive       {naive:.3}   ({:+.1}% vs greedy)",
        100.0 * (naive - g) / g
    );
    println!(
        "  swor        {swor:.3}   ({:+.1}% vs naive)",
        100.0 * (swor - naive) / naive
    );
    println!(
        "  optimistic  {opt:.3}   ({:+.1}% vs naive)  [UPPER BOUND]",
        100.0 * (opt - naive) / naive
    );
    println!();
    let upside = 100.0 * (opt - naive) / naive;
    if upside < 10.0 {
        println!("DECISION: optimistic upper bound is +{upside:.1}% over naive (<10%). The real");
        println!("  SWOR is below this — the draft-sampling + full-q GPU build is NOT justified.");
    } else {
        println!("DECISION: optimistic upper bound is +{upside:.1}% over naive (≥10%). Real SWOR");
        println!(
            "  uplift (between naive and optimistic) may justify the GPU build — investigate."
        );
    }
}
