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
    /// The multi-slot engine: k streams advance together.
    Slots,
    /// No slots at all — k requests one after another through the daemon.
    /// This is what a deployment does today with `serve.multi_slot` off, and
    /// it is the baseline the slot engine has to beat.
    Sequential,
    /// Beta's in-daemon continuous batching (see `DaemonDriver::start` for why
    /// this cannot currently be driven).
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

/// A stream's prompt, made unique per (run, stream) so no prompt is ever
/// submitted twice in a sweep.
///
/// This is load-bearing for fairness, not cosmetic. The two backends cache on
/// different keys and a repeated prompt is exactly the case where they
/// diverge:
///
/// - the daemon holds a token-level longest-common-prefix cache, so resending
///   an identical prompt lets it skip prefill entirely;
/// - the slot engine keys on conversation identity and `find_continuation`
///   returns `None` outright for a single-turn `convo`
///   (`session_table.rs:170`), so a repeat can never hit.
///
/// Cycling a fixed prompt list therefore hands the daemon free prefills and
/// denies the slot engine any — an asymmetry worth ~a prefill per request,
/// all of it favouring the daemon. Varying the leading token defeats the LCP
/// match at position 0 and costs the slot engine nothing it could have used.
pub fn stream_prompt(run: usize, stream: usize) -> String {
    format!(
        "Q{run}-{stream}. {}",
        STREAM_PROMPTS[stream % STREAM_PROMPTS.len()]
    )
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

#[cfg(feature = "multi-slot")]
pub struct SlotDriver {
    engine: hipfire_arch_qwen35::serve_engine::SlotEngine,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    max_concurrency: usize,
    /// Monotonic run counter, so every prompt in a sweep is unique. See
    /// `stream_prompt` for why that is load-bearing.
    seq: usize,
}

#[cfg(not(feature = "multi-slot"))]
pub struct SlotDriver {
    max_concurrency: usize,
    seq: usize,
}

#[cfg(feature = "multi-slot")]
impl SlotDriver {
    pub fn start(model: &Path, max_concurrency: usize, cap_tokens: usize) -> Result<Self> {
        let hfq = hipfire_runtime::hfq::HfqFile::open(model)
            .map_err(|e| anyhow::anyhow!("slots: open {}: {e}", model.display()))?;
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .map_err(|e| anyhow::anyhow!("slots: tokenizer: {e}"))?;
        drop(hfq);
        // Host swap budget is deliberately SMALL here, unlike the serve
        // default of 16 GiB. A sweep opens a fresh session per stream per run
        // and the engine keeps finished sessions resident for multi-turn
        // reuse, so evicted snapshots are the one thing that grows without
        // bound over a long sweep. On a box with no swap, letting that reach
        // 16 GiB of host RAM is how a benchmark takes the machine down.
        // Beyond this budget the swap manager spills to disk, which is the
        // correct trade for a throughput measurement.
        let engine = hipfire_arch_qwen35::serve_engine::SlotEngine::spawn(
            hipfire_arch_qwen35::serve_engine::EngineConfig {
                model_path: model.to_path_buf(),
                n_slots: max_concurrency,
                cap_tokens,
                prefill_chunk: 1024,
                host_budget_bytes: 2 * 1024 * 1024 * 1024,
                swap_dir: std::env::temp_dir().join("hipfire-bench-swap"),
            },
        )
        .map_err(|e| anyhow::anyhow!("slots: {e}"))?;
        Ok(Self {
            engine,
            tokenizer,
            max_concurrency,
            seq: 0,
        })
    }

    /// Render one stream's prompt tokens plus its conversation identity.
    fn render(&self, run: usize, idx: usize) -> (Vec<u32>, Vec<u64>) {
        use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame, Role};
        let owned = stream_prompt(run, idx);
        let user = owned.as_str();
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

#[cfg(not(feature = "multi-slot"))]
impl SlotDriver {
    pub fn start(_model: &Path, _max_concurrency: usize, _cap_tokens: usize) -> Result<Self> {
        bail!("multi-slot feature disabled: rebuild with --features multi-slot")
    }
}

#[cfg(feature = "multi-slot")]
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
        self.seq += 1;
        let run = self.seq;
        let started = Instant::now();
        let mut rxs = Vec::with_capacity(k);
        for i in 0..k {
            let (tx, rx) = mpsc::channel::<Event>();
            let (prompt_tokens, convo) = self.render(run, i);
            self.engine
                .submit(SubmitRequest {
                    session: None,
                    prompt_tokens,
                    convo,
                    continuation: Vec::new(),
                    image_bytes: None,
                    vis_span: None,
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
                let first = stream_prompt(run, i);
                let convo = vec![turn_hash(&first), turn_hash(FOLLOWUP_PROMPT)];
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
                        image_bytes: None,
                        vis_span: None,
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

#[cfg(not(feature = "multi-slot"))]
impl ConcurrencyBackend for SlotDriver {
    fn label(&self) -> &'static str {
        "slots"
    }
    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }
    fn run(&mut self, _workload: WorkloadSel, _k: usize, _max_tokens: u64) -> Result<ArmResult> {
        bail!("multi-slot feature disabled")
    }
}

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

/// The no-slots baseline: k requests run strictly one after another through
/// the ordinary daemon generate path.
///
/// This is what a box serves today with `serve.multi_slot` off, so it is the
/// number the slot engine has to beat. It needs no multi-inflight support --
/// `Engine::generate` is blocking and one-at-a-time, which is exactly the
/// behaviour under measurement -- so unlike `DaemonDriver` it is unaffected by
/// the client's lack of a concurrent API.
///
/// Timed on the SAME clock as every other backend: wall from the first submit
/// to the last completion, with generated tokens summed across all k. That is
/// the only way "slots vs no slots" means anything.
pub struct SequentialDriver {
    engine: hipfire_client::Engine,
    max_concurrency: usize,
    /// Monotonic run counter — see `stream_prompt`. Without this the daemon's
    /// longest-common-prefix cache serves a repeated prompt from KV and the
    /// arm measures cached prefill rather than the real thing.
    seq: usize,
}

impl SequentialDriver {
    pub fn start(engine: hipfire_client::Engine, max_concurrency: usize) -> Result<Self> {
        Ok(Self {
            engine,
            max_concurrency,
            seq: 0,
        })
    }
}

impl ConcurrencyBackend for SequentialDriver {
    fn label(&self) -> &'static str {
        "noslots"
    }
    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult> {
        check_k(self, k)?;
        // Multi-turn on the sequential path is just two turns back to back;
        // there is no session to reuse, which is precisely the gap the slot
        // engine's prefix cache exists to close.
        let turns = if matches!(workload, WorkloadSel::Multiturn) {
            2
        } else {
            1
        };

        self.seq += 1;
        let run = self.seq;
        let started = Instant::now();
        let mut tokens = 0u64;
        let mut rejected = 0usize;
        for i in 0..k {
            for turn in 0..turns {
                let base = stream_prompt(run, i);
                let prompt = if turn == 0 {
                    base
                } else {
                    format!("{base} {FOLLOWUP_PROMPT}")
                };
                let req = batch_request(&prompt, max_tokens, &format!("bench-seq-{i}-{turn}"));
                // Drop the batch-route opt-in: this arm is deliberately the
                // plain sequential path.
                let mut req = req;
                if let Some(obj) = req.as_object_mut() {
                    obj.remove("serve_continuous_batch");
                }
                let mut n = 0u64;
                match self.engine.generate(&req, |ev| {
                    if ev.get("type").and_then(serde_json::Value::as_str) == Some("token") {
                        n += 1;
                    }
                    Ok(())
                }) {
                    Ok(_) => tokens += n,
                    Err(_) => rejected += 1,
                }
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

pub struct DaemonDriver {
    engine: hipfire_client::Engine,
    max_concurrency: usize,
}

impl DaemonDriver {
    /// `capable` is the daemon's own `continuous_batch_capable` from the load
    /// reply. Refusing here turns "this daemon cannot batch" into a reported
    /// result rather than a run that silently measures sequential execution.
    ///
    /// # Currently blocked on a client-side limitation
    ///
    /// Driving this backend needs several generates in flight at once, and
    /// `hipfire_client::Engine` cannot express that. `Engine::spawn` starts a
    /// reader thread (`hipfire-client/src/lib.rs:447`) that consumes ALL of
    /// the daemon's stdout and routes each lifecycle frame to a channel keyed
    /// by `(id, attempt_id)`; a frame whose key has no registered sender is
    /// **silently dropped**. Registration happens only inside `generate`,
    /// which is blocking and single-attempt, and the `pending` map is private.
    ///
    /// So a hand-rolled `send` × k followed by `recv` deadlocks: the daemon
    /// generates, the reader thread bins every token and `done` for want of a
    /// registered channel, and the driver waits forever. That was observed --
    /// the daemon reported `continuous batch staged: slots=4` and then neither
    /// side moved.
    ///
    /// Unblocking it needs ONE of:
    ///   1. a public multi-inflight API on `Engine` (e.g. `submit_streaming(req)
    ///      -> Receiver<Value>` that registers the pending channel and returns
    ///      it), which is a small, genuinely reusable addition; or
    ///   2. driving this backend over HTTP through `hipfire serve` with
    ///      `serve.continuous_batch_size` set, as `scripts/serve_concurrency_gate.sh`
    ///      does -- at the cost of folding HTTP into the measurement, which the
    ///      spec deliberately excluded.
    ///
    /// Failing here is deliberate: a benchmark that hangs is worse than one
    /// that says why it cannot run.
    pub fn start(
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        capable: bool,
    ) -> Result<Self> {
        if !capable {
            bail!("daemon does not advertise continuous_batch_capable");
        }
        let _ = (&engine, max_concurrency);
        bail!(
            "hipfire_client::Engine has no public multi-inflight API: its reader thread \
             drops lifecycle frames for unregistered (id, attempt_id) keys, so pipelined \
             generates deadlock. Needs Engine::submit_streaming or an HTTP driver — see \
             DaemonDriver::start docs"
        );
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
        for i in 0..k {
            let id = format!("bench-conc-{i}");
            let prompt = STREAM_PROMPTS[i % STREAM_PROMPTS.len()];
            self.engine
                .send(&batch_request(prompt, max_tokens, &id))
                .map_err(|e| anyhow::anyhow!("batch send: {e}"))?;
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
                // The daemon stages a turn and WAITS for the client to commit
                // it before sending `done`. `Engine::generate` performs this
                // handshake internally; a hand-rolled send/recv pipeline must
                // do it explicitly or both sides block forever -- the daemon
                // holding a staged turn, the driver waiting for a `done` that
                // will never come. Echo id/attempt_id back so the daemon can
                // match the commit to the right in-flight request.
                Some("commit_ready") => {
                    let commit = serde_json::json!({
                        "type": "commit",
                        "id": frame.get("id").cloned().unwrap_or(serde_json::Value::Null),
                        "attempt_id": frame
                            .get("attempt_id")
                            .cloned()
                            .unwrap_or(serde_json::json!(1)),
                    });
                    self.engine
                        .send(&commit)
                        .map_err(|e| anyhow::anyhow!("batch commit: {e}"))?;
                }
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
        assert!(
            table.contains("150.00"),
            "aggregate median missing: {table}"
        );
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
        assert_eq!(r.get("max_think_tokens").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(
            r.get("assistant_prefix").and_then(|v| v.as_str()),
            Some("closed_think")
        );
    }

    /// A faster second turn is not evidence of prefix reuse — it could be
    /// cache warmth. If the multi-turn arm recorded no prefix hits on the
    /// slot backend, the run is invalid and must not publish a number.
    #[test]
    fn multiturn_without_prefix_hits_is_invalid_on_slots() {
        let no_hits = ArmResult {
            tokens: 100,
            wall_ms: 500.0,
            rejected: 0,
            prefix_hits: 0,
        };
        let hits = ArmResult {
            tokens: 100,
            wall_ms: 500.0,
            rejected: 0,
            prefix_hits: 2,
        };

        assert!(arm_is_valid(WorkloadSel::Multiturn, "slots", &no_hits).is_err());
        assert!(arm_is_valid(WorkloadSel::Multiturn, "slots", &hits).is_ok());
        // The batch backend cannot reuse a prefix at all, so zero is expected.
        assert!(arm_is_valid(WorkloadSel::Multiturn, "batch", &no_hits).is_ok());
        // Stateless never reuses anything.
        assert!(arm_is_valid(WorkloadSel::Stateless, "slots", &no_hits).is_ok());
    }
}
