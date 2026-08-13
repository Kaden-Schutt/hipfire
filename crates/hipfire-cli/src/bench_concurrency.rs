// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Concurrency sweep for `hipfire bench`: drives both concurrent backends
//! (this branch's in-process `SlotEngine` and beta's in-daemon continuous
//! batching) over a range of concurrent stream counts and reports aggregate
//! throughput for each.
//!
//! The two backends are mutually exclusive in production — `complete_request`
//! returns into `complete_request_slots` whenever a slot engine is present and
//! never reaches the daemon — so the only way to choose between them on
//! evidence is to measure both here, on the same model and the same clock.

use anyhow::{bail, Result};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendSel {
    Slots,
    Batch,
    Both,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkloadSel {
    Stateless,
    Multiturn,
    Both,
}

/// Parse `--concurrency 1,2,3,4` into sorted unique positive counts.
pub fn parse_concurrency(s: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let n: usize = part
            .parse()
            .map_err(|_| anyhow::anyhow!("--concurrency: not a number: {part}"))?;
        if n == 0 {
            bail!("--concurrency values must be positive");
        }
        out.push(n);
    }
    if out.is_empty() {
        bail!("--concurrency needs at least one value");
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Interleaved visiting order: `1,2,3,4,1,2,3,4,...`, never `1,1,2,2,...`.
///
/// Blocked ordering lets a thermal drift over the sweep read as a
/// concurrency effect. On this hardware that is not hypothetical: a blocked
/// single-run sweep reported 46.63 tok/s at 4 slots — below its own 1-slot
/// figure — where an interleaved 3-round repeat put the same point at
/// 108–118 tok/s.
pub fn sweep_order(points: &[usize], runs: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(points.len() * runs);
    for _ in 0..runs {
        out.extend_from_slice(points);
    }
    out
}

/// Median of the samples. Sorts in place. `None` when empty.
pub fn median(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    let mid = xs.len() / 2;
    if xs.len() % 2 == 1 {
        Some(xs[mid])
    } else {
        Some((xs[mid - 1] + xs[mid]) / 2.0)
    }
}

/// One measured run of one arm at one concurrency point.
#[derive(Clone, Copy, Debug)]
pub struct ArmResult {
    /// Generated tokens summed across all streams.
    pub tokens: u64,
    /// Wall clock from first submit to last completion.
    pub wall_ms: f64,
    /// Streams the backend refused. Their tokens are NOT counted.
    pub rejected: usize,
    /// Prefix-cache hits observed during this run (SlotEngine only).
    pub prefix_hits: usize,
}

impl ArmResult {
    /// Aggregate throughput delivered to the caller.
    ///
    /// This is the ONLY metric compared across backends. The SlotEngine's
    /// internal `ms/step` is deliberately excluded: the daemon path has no
    /// comparable figure and pairing them would flatter whichever side
    /// reported the friendlier decomposition.
    pub fn aggregate_tok_s(&self) -> f64 {
        if self.wall_ms <= 0.0 {
            return 0.0;
        }
        self.tokens as f64 / (self.wall_ms / 1000.0)
    }
}

/// All samples for one (backend, workload, k).
pub struct Point {
    pub backend: &'static str,
    pub workload: &'static str,
    pub k: usize,
    pub samples: Vec<ArmResult>,
}

/// Render the comparison table, medians per point.
pub fn render_table(points: &[Point]) -> String {
    let mut out = String::new();
    out.push_str(
        "\n  backend  workload    k   aggregate tok/s   per-stream   prefix hits   rejected\n",
    );
    for p in points {
        let mut aggs: Vec<f64> = p.samples.iter().map(ArmResult::aggregate_tok_s).collect();
        let Some(med) = median(&mut aggs) else {
            continue;
        };
        let per_stream = if p.k > 0 { med / p.k as f64 } else { 0.0 };
        let hits: usize = p.samples.iter().map(|s| s.prefix_hits).sum();
        let rej: usize = p.samples.iter().map(|s| s.rejected).sum();
        out.push_str(&format!(
            "  {:<8} {:<10} {:>2}   {:>15.2}   {:>10.2}   {:>11}   {:>8}\n",
            p.backend, p.workload, p.k, med, per_stream, hits, rej
        ));
    }
    out.push_str(
        "\n  NOTE: SlotEngine runs in-process; the daemon path pays JSONL pipe\n\
         \x20 encode/decode per token. That difference is inherent to where each\n\
         \x20 backend lives and is NOT a property of the batching algorithm.\n\
         \x20 Each backend is held at max concurrency and k varied, so the KV\n\
         \x20 arena is sized for the maximum at every point: this is a\n\
         \x20 concurrency curve, not a memory-footprint curve.\n",
    );
    out
}

/// One concurrent backend, started once at max concurrency.
pub trait ConcurrencyBackend {
    fn label(&self) -> &'static str;
    fn max_concurrency(&self) -> usize;
    /// Run `k` concurrent streams to completion.
    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult>;
}

/// Reject `k` above the started maximum instead of clamping. A clamped run
/// would report a k=4 row that was really measured at k=2.
pub fn check_k(b: &dyn ConcurrencyBackend, k: usize) -> Result<()> {
    if k > b.max_concurrency() {
        bail!(
            "concurrency {k} exceeds {} backend maximum {}",
            b.label(),
            b.max_concurrency()
        );
    }
    Ok(())
}

/// Fixed prompts, one per stream. Distinct so slots cannot alias each
/// other's KV undetected — identical prompts would hide a cross-slot leak.
pub const STREAM_PROMPTS: &[&str] = &[
    "The capital of France is",
    "Once upon a time, in a distant galaxy,",
    "The recipe for a good cup of tea starts with",
    "In machine learning, gradient descent works by",
    "The history of the printing press begins",
    "A well-designed database index avoids",
    "The difference between TCP and UDP is",
    "To sort a list in linear time you must",
];

/// Second user turn for the multi-turn arm.
pub const FOLLOWUP_PROMPT: &str = "Explain that again in one sentence.";

/// FNV-1a over a user turn — the same identity function
/// `complete_request_slots` uses, so sessions match the production path.
pub fn turn_hash(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct SlotDriver {
    engine: hipfire_arch_qwen35::serve_engine::SlotEngine,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    max_concurrency: usize,
}

impl SlotDriver {
    pub fn start(model: &Path, max_concurrency: usize, cap_tokens: usize) -> Result<Self> {
        let hfq = hipfire_runtime::hfq::HfqFile::open(model)
            .map_err(|e| anyhow::anyhow!("slots: open {}: {e}", model.display()))?;
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .map_err(|e| anyhow::anyhow!("slots: tokenizer: {e}"))?;
        drop(hfq);
        let engine = hipfire_arch_qwen35::serve_engine::SlotEngine::spawn(
            hipfire_arch_qwen35::serve_engine::EngineConfig {
                model_path: model.to_path_buf(),
                n_slots: max_concurrency,
                cap_tokens,
                host_budget_bytes: 16 * 1024 * 1024 * 1024,
                swap_dir: std::env::temp_dir().join("hipfire-bench-swap"),
            },
        )
        .map_err(|e| anyhow::anyhow!("slots: {e}"))?;
        Ok(Self {
            engine,
            tokenizer,
            max_concurrency,
        })
    }

    /// Render one stream's prompt tokens plus its conversation identity.
    fn render(&self, idx: usize) -> (Vec<u32>, Vec<u64>) {
        use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame, Role};
        let user = STREAM_PROMPTS[idx % STREAM_PROMPTS.len()];
        let frame = ChatFrame {
            tokenizer: &self.tokenizer,
            system: None,
            user,
            // Benchmarks run answer-mode: a reasoning model cannot close a
            // think span inside a benchmark budget.
            assistant_prefix: AssistantPrefix::ClosedThink,
            raw: false,
        };
        let history: Vec<(Role, &str)> = Vec::new();
        let tokens = frame.build_multi_turn(&history);
        (tokens, vec![turn_hash(user)])
    }
}

impl ConcurrencyBackend for SlotDriver {
    fn label(&self) -> &'static str {
        "slots"
    }
    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult> {
        use hipfire_runtime::serve::{Event, SubmitRequest};
        check_k(self, k)?;

        let hits_before = self.engine.stats().prefix_hits;
        let started = Instant::now();
        let mut rxs = Vec::with_capacity(k);
        for i in 0..k {
            let (tx, rx) = mpsc::channel::<Event>();
            let (prompt_tokens, convo) = self.render(i);
            self.engine
                .submit(SubmitRequest {
                    session: None,
                    prompt_tokens,
                    convo,
                    continuation: Vec::new(),
                    max_tokens: max_tokens as usize,
                    reply: tx,
                })
                .map_err(|e| anyhow::anyhow!("slots submit: {e}"))?;
            rxs.push(rx);
        }

        let mut tokens = 0u64;
        let mut rejected = 0usize;
        let mut sessions: Vec<Option<u64>> = vec![None; k];
        for (i, rx) in rxs.into_iter().enumerate() {
            while let Ok(ev) = rx.recv() {
                match ev {
                    Event::Accepted { session } => sessions[i] = Some(session),
                    Event::Token { .. } => tokens += 1,
                    Event::Done { .. } => break,
                    Event::Rejected { .. } => {
                        rejected += 1;
                        break;
                    }
                }
            }
        }

        // Multi-turn arm: a second turn on each surviving session, which is
        // what exercises the prefix cache.
        if matches!(workload, WorkloadSel::Multiturn) {
            use hipfire_runtime::prompt_frame::{continuation_suffix, AssistantPrefix};
            let mut rxs2 = Vec::new();
            for (i, session) in sessions.iter().enumerate() {
                let Some(session) = session else { continue };
                let (tx, rx) = mpsc::channel::<Event>();
                let first = STREAM_PROMPTS[i % STREAM_PROMPTS.len()];
                let convo = vec![turn_hash(first), turn_hash(FOLLOWUP_PROMPT)];
                let continuation = continuation_suffix(
                    &self.tokenizer,
                    FOLLOWUP_PROMPT,
                    AssistantPrefix::ClosedThink,
                );
                self.engine
                    .submit(SubmitRequest {
                        session: Some(*session),
                        prompt_tokens: Vec::new(),
                        convo,
                        continuation,
                        max_tokens: max_tokens as usize,
                        reply: tx,
                    })
                    .map_err(|e| anyhow::anyhow!("slots submit turn2: {e}"))?;
                rxs2.push(rx);
            }
            for rx in rxs2 {
                while let Ok(ev) = rx.recv() {
                    match ev {
                        Event::Token { .. } => tokens += 1,
                        Event::Done { .. } => break,
                        Event::Rejected { .. } => {
                            rejected += 1;
                            break;
                        }
                        Event::Accepted { .. } => {}
                    }
                }
            }
        }

        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let prefix_hits = self.engine.stats().prefix_hits.saturating_sub(hits_before);
        Ok(ArmResult {
            tokens,
            wall_ms,
            rejected,
            prefix_hits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_concurrency_sorts_dedups_and_rejects_zero() {
        assert_eq!(parse_concurrency("4,1,2,1").unwrap(), vec![1, 2, 4]);
        assert!(parse_concurrency("0").is_err());
        assert!(parse_concurrency("").is_err());
        assert!(parse_concurrency("x").is_err());
    }

    /// The sweep must interleave. A blocked order (1,1,1,2,2,2) lets thermal
    /// drift masquerade as a concurrency effect — which it did in this repo.
    #[test]
    fn sweep_order_is_interleaved_not_blocked() {
        assert_eq!(
            sweep_order(&[1, 2, 3, 4], 3),
            vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        );
    }

    #[test]
    fn median_handles_odd_even_and_empty() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn aggregate_tok_s_is_tokens_over_wall_clock() {
        let r = ArmResult {
            tokens: 256,
            wall_ms: 2000.0,
            rejected: 0,
            prefix_hits: 0,
        };
        assert!((r.aggregate_tok_s() - 128.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_tok_s_is_zero_when_no_time_elapsed() {
        let r = ArmResult {
            tokens: 10,
            wall_ms: 0.0,
            rejected: 0,
            prefix_hits: 0,
        };
        assert_eq!(r.aggregate_tok_s(), 0.0);
    }

    #[test]
    fn render_table_reports_median_and_per_stream() {
        let points = vec![Point {
            backend: "slots",
            workload: "stateless",
            k: 2,
            samples: vec![
                ArmResult {
                    tokens: 200,
                    wall_ms: 1000.0,
                    rejected: 0,
                    prefix_hits: 0,
                },
                ArmResult {
                    tokens: 100,
                    wall_ms: 1000.0,
                    rejected: 0,
                    prefix_hits: 0,
                },
            ],
        }];
        let table = render_table(&points);
        // median of {200, 100} tok/s = 150; per-stream = 75
        assert!(table.contains("150.00"), "aggregate median missing: {table}");
        assert!(table.contains("75.00"), "per-stream missing: {table}");
        assert!(table.contains("slots"), "backend label missing: {table}");
    }

    /// A backend must refuse k above what it was started with rather than
    /// silently clamping — a clamped run would publish a k=4 number that was
    /// actually measured at k=2.
    #[test]
    fn run_rejects_k_above_max_concurrency() {
        struct Fake {
            max: usize,
        }
        impl ConcurrencyBackend for Fake {
            fn label(&self) -> &'static str {
                "fake"
            }
            fn max_concurrency(&self) -> usize {
                self.max
            }
            fn run(&mut self, _w: WorkloadSel, k: usize, _m: u64) -> Result<ArmResult> {
                check_k(self, k)?;
                Ok(ArmResult {
                    tokens: 0,
                    wall_ms: 1.0,
                    rejected: 0,
                    prefix_hits: 0,
                })
            }
        }
        let mut f = Fake { max: 2 };
        assert!(f.run(WorkloadSel::Stateless, 2, 8).is_ok());
        let err = f.run(WorkloadSel::Stateless, 3, 8).unwrap_err().to_string();
        assert!(err.contains("exceeds"), "unexpected error: {err}");
    }
}
