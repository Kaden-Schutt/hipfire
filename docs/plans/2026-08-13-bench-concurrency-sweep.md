# Bench Concurrency Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `hipfire bench --concurrency 1,2,3,4` sweeping both concurrent backends (this branch's in-process `SlotEngine` and beta's in-daemon continuous batching) so the choice between them is measured rather than argued.

**Architecture:** One `ConcurrencyBackend` trait with two drivers, so the sweep loop, the clock, and the reporting are written once and neither backend can be timed on a different basis. `SlotDriver` calls `SlotEngine::submit` on `k` threads against one in-process engine. `DaemonDriver` pipelines `Engine::send` × k before any `recv`, because `Engine::generate` is request/response and would serialise the thing under test. Each backend is started once at `max(concurrency)`; the sweep varies `k`.

**Tech Stack:** Rust, `hipfire-cli` (clap), `hipfire-runtime::serve` (`SlotEngine`, `SubmitRequest`, `Event`, `EngineStats`), `hipfire-client::Engine`, `std::sync::mpsc`, `std::thread`.

**Spec:** `docs/specs/2026-08-13-bench-concurrency-sweep.md`

## Global Constraints

- **Purely additive.** `--concurrency` absent ⇒ `bench` behaves byte-identically to today. Task 1 pins this with a test.
- **Answer mode everywhere.** All generated requests go through `bench_generate_request` / `bench_generate_request_reasoning`, which already inject `max_think_tokens=1`, `assistant_prefix=closed_think`, `reasoning_effort=none`. A reasoning model cannot close a `<think>` span inside a benchmark budget, and the daemon fails such a turn closed as a validation error.
- **Interleaved sweep, medians.** Points run `1,2,3,4,1,2,3,4,…`, never blocked. A blocked single-run sweep in this repo produced a fake 4-slot cliff (46.63 tok/s) that an interleaved 3-round repeat put at 108–118 tok/s.
- **Aggregate tok/s is the only cross-backend metric.** Do not compare `ms/step`; the daemon path has no equivalent.
- **Build/run env:** `export PATH=/opt/rocm-7.2.2/bin:$PATH`. Pin the daemon with `export HIPFIRE_DAEMON_BIN=$PWD/target/release/examples/daemon` — `find_daemon` prefers the stale `~/.hipfire/bin/daemon` otherwise.
- **All new code lives in `crates/hipfire-cli/src/bench_concurrency.rs`**, a new module. `main.rs` is ~15k lines already; adding ~600 lines of driver code to it is the wrong call. `main.rs` gains only the flags, the `mod` declaration, and one dispatch call.

---

### Task 1: Module scaffold, flags, and the additive guarantee

**Files:**
- Create: `crates/hipfire-cli/src/bench_concurrency.rs`
- Modify: `crates/hipfire-cli/src/main.rs` (add `mod bench_concurrency;` near the other `mod` lines; add three fields to `struct BenchArgs` after `reasoning_on`)
- Test: inline `#[cfg(test)] mod tests` in `bench_concurrency.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `pub enum BackendSel { Slots, Batch, Both }`, `pub enum WorkloadSel { Stateless, Multiturn, Both }`, `pub fn parse_concurrency(s: &str) -> anyhow::Result<Vec<usize>>`, `pub fn sweep_order(points: &[usize], runs: usize) -> Vec<usize>`, `pub fn median(xs: &mut [f64]) -> Option<f64>`.

- [ ] **Step 1: Write the failing tests**

In a new file `crates/hipfire-cli/src/bench_concurrency.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Concurrency sweep for `hipfire bench`: drives both concurrent backends
//! (this branch's in-process `SlotEngine` and beta's in-daemon continuous
//! batching) over a range of concurrent stream counts and reports aggregate
//! throughput for each.

use anyhow::{bail, Result};

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
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: FAIL — `file not found for module 'bench_concurrency'`, because `main.rs` has no `mod bench_concurrency;` yet.

- [ ] **Step 3: Wire the module and the flags**

In `crates/hipfire-cli/src/main.rs`, next to the other top-level `mod` declarations, add:

```rust
mod bench_concurrency;
```

In `struct BenchArgs`, immediately after the `reasoning_on` field, add:

```rust
    /// Sweep concurrent stream counts, e.g. `1,2,3,4`. Absent leaves bench
    /// on its single-stream path, unchanged.
    #[arg(long)]
    concurrency: Option<String>,
    /// Which concurrent backend to drive: slots, batch, or both.
    #[arg(long, value_parser = ["slots", "batch", "both"], default_value = "both")]
    backend: String,
    /// Which workload arm to run: stateless, multiturn, or both.
    #[arg(long, value_parser = ["stateless", "multiturn", "both"], default_value = "both")]
    workload: String,
```

Then update every `BenchArgs { .. }` literal in the file (there are two: in `profile_command` and in the `bench_generate_request_includes_numeric_first_attempt` region's harness if present) by adding:

```rust
            concurrency: None,
            backend: "both".to_owned(),
            workload: "both".to_owned(),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: PASS — 3 passed.

- [ ] **Step 5: Prove the additive guarantee still holds**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_ --test-threads=2`
Expected: PASS — the existing `bench_generate_request_is_answer_mode_by_default`, `bench_reasoning_on_opts_back_into_thinking`, and `bench_generate_request_includes_numeric_first_attempt` all still pass, proving the single-stream request shape is untouched.

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-cli/src/bench_concurrency.rs crates/hipfire-cli/src/main.rs
git commit -m "feat(bench): concurrency sweep scaffold — flags, parsing, interleaved order"
```

---

### Task 2: Result types and the report table

**Files:**
- Modify: `crates/hipfire-cli/src/bench_concurrency.rs`
- Test: inline `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `median` from Task 1.
- Produces: `pub struct ArmResult { pub tokens: u64, pub wall_ms: f64, pub rejected: usize, pub prefix_hits: usize }`, `pub struct Point { pub backend: &'static str, pub workload: &'static str, pub k: usize, pub samples: Vec<ArmResult> }`, `impl ArmResult { pub fn aggregate_tok_s(&self) -> f64 }`, `pub fn render_table(points: &[Point]) -> String`.

- [ ] **Step 1: Write the failing tests**

Append to `bench_concurrency.rs`, inside the existing `mod tests`:

```rust
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
                ArmResult { tokens: 200, wall_ms: 1000.0, rejected: 0, prefix_hits: 0 },
                ArmResult { tokens: 100, wall_ms: 1000.0, rejected: 0, prefix_hits: 0 },
            ],
        }];
        let table = render_table(&points);
        // median of {200, 100} tok/s = 150; per-stream = 75
        assert!(table.contains("150.00"), "aggregate median missing: {table}");
        assert!(table.contains("75.00"), "per-stream missing: {table}");
        assert!(table.contains("slots"), "backend label missing: {table}");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: FAIL — `cannot find struct ArmResult in this scope`.

- [ ] **Step 3: Implement the types and renderer**

Append to `bench_concurrency.rs` (before `#[cfg(test)]`):

```rust
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-cli/src/bench_concurrency.rs
git commit -m "feat(bench): concurrency result types and comparison table"
```

---

### Task 3: The backend trait and the SlotEngine driver

**Files:**
- Modify: `crates/hipfire-cli/src/bench_concurrency.rs`
- Test: inline `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `ArmResult` (Task 2).
- Produces: `pub trait ConcurrencyBackend { fn label(&self) -> &'static str; fn max_concurrency(&self) -> usize; fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult>; }`, `pub struct SlotDriver`, `impl SlotDriver { pub fn start(model: &std::path::Path, max_concurrency: usize, cap_tokens: usize) -> Result<Self> }`.

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
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
                Ok(ArmResult { tokens: 0, wall_ms: 1.0, rejected: 0, prefix_hits: 0 })
            }
        }
        let mut f = Fake { max: 2 };
        assert!(f.run(WorkloadSel::Stateless, 2, 8).is_ok());
        let err = f.run(WorkloadSel::Stateless, 3, 8).unwrap_err().to_string();
        assert!(err.contains("exceeds"), "unexpected error: {err}");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: FAIL — `cannot find trait ConcurrencyBackend in this scope`.

- [ ] **Step 3: Implement the trait, the guard, and the SlotEngine driver**

Add these imports at the top of `bench_concurrency.rs`:

```rust
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
```

Append before `#[cfg(test)]`:

```rust
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
        let prefix_hits = self
            .engine
            .stats()
            .prefix_hits
            .saturating_sub(hits_before);
        Ok(ArmResult {
            tokens,
            wall_ms,
            rejected,
            prefix_hits,
        })
    }
}
```

Add `hipfire-arch-qwen35` to `crates/hipfire-cli/Cargo.toml` `[dependencies]` if it is not already there:

```toml
hipfire-arch-qwen35 = { path = "../hipfire-arch-qwen35" }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-cli/src/bench_concurrency.rs crates/hipfire-cli/Cargo.toml
git commit -m "feat(bench): ConcurrencyBackend trait and in-process SlotEngine driver"
```

---

### Task 4: The daemon continuous-batch driver

**Files:**
- Modify: `crates/hipfire-cli/src/bench_concurrency.rs`
- Test: inline `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `ConcurrencyBackend`, `check_k`, `ArmResult`, `STREAM_PROMPTS` (Task 3).
- Produces: `pub struct DaemonDriver`, `impl DaemonDriver { pub fn start(engine: hipfire_client::Engine, max_concurrency: usize, capable: bool) -> Result<Self> }`, `pub fn batch_request(prompt: &str, max_tokens: u64, id: &str) -> serde_json::Value`.

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
    /// The daemon batch route only engages when the request opts in AND the
    /// request stays answer-mode. A request missing `serve_continuous_batch`
    /// silently runs sequentially, which would report batching numbers for a
    /// non-batched run.
    #[test]
    fn batch_request_opts_into_the_batch_route_and_answer_mode() {
        let r = batch_request("hello", 64, "bench-c-1");
        assert_eq!(
            r.get("serve_continuous_batch").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(r.get("type").and_then(|v| v.as_str()), Some("generate"));
        assert_eq!(r.get("id").and_then(|v| v.as_str()), Some("bench-c-1"));
        assert_eq!(r.get("max_tokens").and_then(|v| v.as_u64()), Some(64));
        // Answer mode, same contract as the single-stream bench path.
        assert_eq!(
            r.get("max_think_tokens").and_then(|v| v.as_u64()),
            Some(1)
        );
        assert_eq!(
            r.get("assistant_prefix").and_then(|v| v.as_str()),
            Some("closed_think")
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: FAIL — `cannot find function batch_request in this scope`.

- [ ] **Step 3: Implement the daemon driver**

Append before `#[cfg(test)]`:

```rust
/// One batch-route generate request.
///
/// `serve_continuous_batch` is what puts the daemon on
/// `drive_qwen_continuous_batch`; without it the request runs sequentially and
/// the resulting numbers would describe a non-batched run. Answer-mode fields
/// mirror `bench_generate_request` so both backends measure the same turn.
pub fn batch_request(prompt: &str, max_tokens: u64, id: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "generate",
        "id": id,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens,
        "attempt_id": 1,
        "serve_continuous_batch": true,
        "max_think_tokens": 1,
        "assistant_prefix": "closed_think",
        "reasoning_effort": "none",
    })
}

pub struct DaemonDriver {
    engine: hipfire_client::Engine,
    max_concurrency: usize,
}

impl DaemonDriver {
    /// `capable` is the daemon's own `continuous_batch_capable` from the load
    /// reply. Refusing here turns "this daemon cannot batch" into a reported
    /// result rather than a run that silently measures sequential execution.
    pub fn start(
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        capable: bool,
    ) -> Result<Self> {
        if !capable {
            bail!("daemon does not advertise continuous_batch_capable");
        }
        Ok(Self {
            engine,
            max_concurrency,
        })
    }
}

impl ConcurrencyBackend for DaemonDriver {
    fn label(&self) -> &'static str {
        "batch"
    }
    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult> {
        check_k(self, k)?;
        // The batch route rejects multi-turn (`batch_messages_are_single_user`),
        // so this arm is sequential by construction on this backend. Report it
        // rather than pretending it batched.
        let _ = workload;

        let started = Instant::now();
        // PIPELINE: every request goes out before any reply is read.
        // `Engine::generate` would block per request and serialise exactly the
        // behaviour under test.
        let mut ids = Vec::with_capacity(k);
        for i in 0..k {
            let id = format!("bench-conc-{i}");
            let prompt = STREAM_PROMPTS[i % STREAM_PROMPTS.len()];
            self.engine
                .send(&batch_request(prompt, max_tokens, &id))
                .map_err(|e| anyhow::anyhow!("batch send: {e}"))?;
            ids.push(id);
        }

        let mut tokens = 0u64;
        let mut rejected = 0usize;
        let mut done = 0usize;
        while done < k {
            let frame = self
                .engine
                .recv()
                .map_err(|e| anyhow::anyhow!("batch recv: {e}"))?;
            match frame.get("type").and_then(serde_json::Value::as_str) {
                Some("token") => tokens += 1,
                Some("done") => done += 1,
                Some("error") => {
                    rejected += 1;
                    done += 1;
                }
                _ => {}
            }
        }

        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(ArmResult {
            tokens,
            wall_ms,
            rejected,
            prefix_hits: 0,
        })
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-cli/src/bench_concurrency.rs
git commit -m "feat(bench): pipelined daemon continuous-batch driver"
```

---

### Task 5: Sweep loop, prefix-hit validity gate, and wiring into bench_command

**Files:**
- Modify: `crates/hipfire-cli/src/bench_concurrency.rs`
- Modify: `crates/hipfire-cli/src/main.rs` (`bench_command`, just after the `--exp` guard block)
- Test: inline `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `pub fn arm_is_valid(workload: WorkloadSel, backend: &str, r: &ArmResult) -> std::result::Result<(), String>`, `pub fn run_sweep(...) -> Result<()>`.

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
    /// A faster second turn is not evidence of prefix reuse — it could be
    /// cache warmth. If the multi-turn arm recorded no prefix hits on the
    /// slot backend, the run is invalid and must not publish a number.
    #[test]
    fn multiturn_without_prefix_hits_is_invalid_on_slots() {
        let no_hits = ArmResult { tokens: 100, wall_ms: 500.0, rejected: 0, prefix_hits: 0 };
        let hits = ArmResult { tokens: 100, wall_ms: 500.0, rejected: 0, prefix_hits: 2 };

        assert!(arm_is_valid(WorkloadSel::Multiturn, "slots", &no_hits).is_err());
        assert!(arm_is_valid(WorkloadSel::Multiturn, "slots", &hits).is_ok());
        // The batch backend cannot reuse a prefix at all, so zero is expected.
        assert!(arm_is_valid(WorkloadSel::Multiturn, "batch", &no_hits).is_ok());
        // Stateless never reuses anything.
        assert!(arm_is_valid(WorkloadSel::Stateless, "slots", &no_hits).is_ok());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: FAIL — `cannot find function arm_is_valid in this scope`.

- [ ] **Step 3: Implement the validity gate and the sweep loop**

Append before `#[cfg(test)]`:

```rust
/// Reject a multi-turn slot run that recorded no prefix hits.
///
/// The whole point of the multi-turn arm is that `SlotEngine` reuses the
/// conversation's KV. A faster turn 2 proves nothing on its own — warm caches
/// do that too. `EngineStats::prefix_hits` is the event itself, so gate on it.
pub fn arm_is_valid(
    workload: WorkloadSel,
    backend: &str,
    r: &ArmResult,
) -> std::result::Result<(), String> {
    if matches!(workload, WorkloadSel::Multiturn) && backend == "slots" && r.prefix_hits == 0 {
        return Err(
            "multi-turn arm recorded no prefix hits — turn 2 did not reuse the \
             session KV, so this number does not measure what the arm claims"
                .to_string(),
        );
    }
    Ok(())
}

/// Run the interleaved sweep over one backend and collect points.
pub fn sweep_backend(
    backend: &mut dyn ConcurrencyBackend,
    arms: &[WorkloadSel],
    points: &[usize],
    runs: usize,
    max_tokens: u64,
    out: &mut Vec<Point>,
) -> Result<()> {
    let label = backend.label();
    for &arm in arms {
        let arm_label = match arm {
            WorkloadSel::Stateless => "stateless",
            WorkloadSel::Multiturn => "multiturn",
            WorkloadSel::Both => continue,
        };
        // One untimed warmup per (backend, arm) so first-call kernel
        // compilation never lands inside a timed sample.
        let _ = backend.run(arm, points[0], max_tokens.min(8));

        let mut by_k: std::collections::BTreeMap<usize, Vec<ArmResult>> =
            std::collections::BTreeMap::new();
        for k in sweep_order(points, runs) {
            let r = backend.run(arm, k, max_tokens)?;
            match arm_is_valid(arm, label, &r) {
                Ok(()) => by_k.entry(k).or_default().push(r),
                Err(why) => eprintln!("  [invalid] {label}/{arm_label} k={k}: {why}"),
            }
            eprint!(".");
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }
        eprintln!();
        for (k, samples) in by_k {
            out.push(Point {
                backend: label,
                workload: arm_label,
                k,
                samples,
            });
        }
    }
    Ok(())
}
```

In `crates/hipfire-cli/src/main.rs`, inside `bench_command`, immediately after the `--exp` / `--reasoning-on` guard block and **before** `if args.exp { return bench_experimental(...) }`, add:

```rust
    if let Some(spec) = args.concurrency.clone() {
        return bench_concurrency_command(paths, &args, &spec);
    }
```

Then add this function next to `bench_command`:

```rust
/// Concurrency sweep across both concurrent backends. Only reached when
/// `--concurrency` is given; the single-stream path above is untouched.
fn bench_concurrency_command(paths: &Paths, args: &BenchArgs, spec: &str) -> Result<()> {
    use crate::bench_concurrency::{
        parse_concurrency, render_table, sweep_backend, BackendSel, ConcurrencyBackend,
        DaemonDriver, Point, SlotDriver, WorkloadSel,
    };

    let points = parse_concurrency(spec)?;
    let max_k = *points.iter().max().expect("non-empty");
    let backend_sel = match args.backend.as_str() {
        "slots" => BackendSel::Slots,
        "batch" => BackendSel::Batch,
        _ => BackendSel::Both,
    };
    let arms: Vec<WorkloadSel> = match args.workload.as_str() {
        "stateless" => vec![WorkloadSel::Stateless],
        "multiturn" => vec![WorkloadSel::Multiturn],
        _ => vec![WorkloadSel::Stateless, WorkloadSel::Multiturn],
    };

    eprintln!("hipfire bench — concurrency sweep");
    eprintln!("  model:       {}", args.model);
    eprintln!("  concurrency: {points:?}");
    eprintln!("  runs/point:  {}", args.runs);
    eprintln!("  max_tokens:  {}", args.max_tokens);

    let mut out: Vec<Point> = Vec::new();

    if matches!(backend_sel, BackendSel::Slots | BackendSel::Both) {
        let registry = load_registry(&paths.registry).registry;
        let model_path = find_model_path(paths, &registry, &args.model)
            .ok_or_else(|| anyhow!("model not found: {}", args.model))?;
        match SlotDriver::start(&model_path, max_k, 8192) {
            Ok(mut d) => {
                eprintln!("  slots backend up ({max_k} slots)");
                sweep_backend(
                    &mut d as &mut dyn ConcurrencyBackend,
                    &arms,
                    &points,
                    args.runs,
                    args.max_tokens as u64,
                    &mut out,
                )?;
            }
            // A backend that cannot run this model is a RESULT, not a crash.
            Err(e) => eprintln!("  slots backend unavailable: {e}"),
        }
    }

    if matches!(backend_sel, BackendSel::Batch | BackendSel::Both) {
        let mut batch_args = args.clone();
        batch_args.concurrency = None;
        let (engine, loaded, _, _) = open_bench_engine_batched(paths, &batch_args, max_k)?;
        let capable = loaded
            .get("continuous_batch_capable")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        match DaemonDriver::start(engine, max_k, capable) {
            Ok(mut d) => {
                eprintln!("  batch backend up (continuous_batch_size={max_k})");
                sweep_backend(
                    &mut d as &mut dyn ConcurrencyBackend,
                    &arms,
                    &points,
                    args.runs,
                    args.max_tokens as u64,
                    &mut out,
                )?;
            }
            Err(e) => eprintln!("  batch backend unavailable: {e}"),
        }
    }

    println!("{}", render_table(&out));
    Ok(())
}
```

Add `#[derive(Clone)]` to `struct BenchArgs` if it is not already derived (the `args.clone()` above needs it).

Add a variant of `open_bench_engine` that sets the batch size, next to `open_bench_engine`:

```rust
/// `open_bench_engine`, but loading with `continuous_batch_size` set so the
/// daemon allocates batch lanes and advertises `continuous_batch_capable`.
/// The value is fixed per load, which is why the sweep holds it at max.
fn open_bench_engine_batched(
    paths: &Paths,
    args: &BenchArgs,
    batch_size: usize,
) -> Result<(
    Engine,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
)> {
    std::env::set_var("HIPFIRE_BENCH_CONTINUOUS_BATCH", batch_size.to_string());
    let r = open_bench_engine(paths, args, None);
    std::env::remove_var("HIPFIRE_BENCH_CONTINUOUS_BATCH");
    r
}
```

And in `open_bench_engine`, immediately before `let loaded = engine.load(&path, params)?;`, add:

```rust
    if let Ok(n) = std::env::var("HIPFIRE_BENCH_CONTINUOUS_BATCH") {
        if let Ok(n) = n.parse::<u64>() {
            params["continuous_batch_size"] = serde_json::json!(n);
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_concurrency --test-threads=2`
Expected: PASS — 9 passed.

- [ ] **Step 5: Verify the additive guarantee once more**

Run: `cargo test --release -p hipfire-cli --bin hipfire -- bench_ --test-threads=2`
Expected: PASS — all bench tests, including the three request-shape tests from the earlier work.

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-cli/src/bench_concurrency.rs crates/hipfire-cli/src/main.rs
git commit -m "feat(bench): wire the concurrency sweep into bench_command"
```

---

### Task 6: End-to-end run on real hardware

**Files:**
- Modify: `docs/specs/2026-08-13-bench-concurrency-sweep.md` (append a Results section)

**Interfaces:**
- Consumes: the whole feature.
- Produces: measured numbers.

- [ ] **Step 1: Build both binaries**

```bash
export PATH=/opt/rocm-7.2.2/bin:$PATH
cargo build --release -p hipfire-cli --bin hipfire
cargo build --release -p hipfire-runtime --example daemon
```

Expected: both finish with no errors.

- [ ] **Step 2: Confirm no daemon holds the singleton**

```bash
pgrep -af "hipfire/bin/daemon|examples/daemon" || echo "clear"
```

Expected: `clear`. The daemon refuses to start while another lives; stop any leftover before running.

- [ ] **Step 3: Run the sweep**

```bash
export HIPFIRE_DAEMON_BIN=$PWD/target/release/examples/daemon
./target/release/hipfire bench qwen3.6:35b-a3b-mq4r \
  --concurrency 1,2,3,4 --runs 3 --max-tokens 64
```

Expected: a table with rows for `slots`/`batch` × `stateless`/`multiturn` × k∈{1,2,3,4}, each with a positive aggregate tok/s. The `slots`/`multiturn` rows must show non-zero prefix hits; if they show `[invalid]` lines instead, the multi-turn arm is not reusing KV and that must be investigated before trusting any multiturn number.

Sanity anchor: `slots`/`stateless` should land near the already-measured curve on this box — 1:~65, 2:~80, 3:~97, 4:~116 tok/s. A large deviation means the driver is measuring something other than the demo did.

- [ ] **Step 4: Record results in the spec**

Append a `## Results` section to `docs/specs/2026-08-13-bench-concurrency-sweep.md` with the table, the hardware (gfx1151), the model, and the date. State plainly which backend won on which arm, and repeat the in-process-vs-pipe caveat next to the comparison.

- [ ] **Step 5: Commit**

```bash
git add docs/specs/2026-08-13-bench-concurrency-sweep.md
git commit -m "docs(bench): record the first two-backend concurrency sweep"
```

---

## Self-Review

**Spec coverage:** CLI surface → Task 1. Result types/metrics/caveat text → Task 2. Trait + SlotEngine driver + stateless & multiturn arms → Task 3. Daemon driver + pipelining + capability refusal → Task 4. Sweep loop, interleaving, medians, prefix-hit gate, backend-unavailable handling, wiring → Task 5. Measured results → Task 6. Every error-handling row in the spec's table maps to code in Tasks 3–5.

**Placeholder scan:** none — every code step carries complete code.

**Type consistency:** `ArmResult` fields (`tokens`, `wall_ms`, `rejected`, `prefix_hits`) are identical in Tasks 2–5. `ConcurrencyBackend::run(&mut self, WorkloadSel, usize, u64) -> Result<ArmResult>` is identical in Tasks 3, 4, 5. `check_k` takes `&dyn ConcurrencyBackend` in both its definition (Task 3) and uses (Tasks 3, 4).

**Known risk carried from the spec:** the daemon driver counts a `"token"` frame as one token. If the batch route emits tokens for several lanes without per-request attribution, the aggregate is still correct (total tokens over total wall clock) but a per-stream breakdown would not be. Task 6 Step 3's sanity anchor is what catches a driver that is measuring the wrong thing.
