# SP7 Concurrent Serve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Several coding agents hit `hipfire serve` at once and are served concurrently by the multi-slot engine, instead of serialising one at a time.

**Architecture:** `hipfire serve` already exposes an OpenAI-compatible `/v1/chat/completions` over HTTP — which is what coding agents speak, and HTTP is already concurrent. The bottleneck is behind it: a single `Engine` (one daemon process) guarded by `Mutex<ServeRuntime>`, so concurrent requests queue on that mutex. SP7 adds a `SlotEngine` — the SP1–SP6 multi-slot rig owned by one background thread, fed by a channel — that serves N requests at once, and dispatches the HTTP handler to it behind a config flag.

**Why not a new socket protocol:** an earlier draft of this plan proposed one. It was wrong. Agents connect to the OpenAI HTTP surface that already exists; a bespoke socket would have been something nothing connects to.

**Why not remove the Mutex directly:** `ServeRuntime` also carries model switching, registry state and KV overrides, all of which assume exclusive access. `SlotEngine` is an alternative backend rather than a rewrite of that state machine, so the existing path stays exactly as it is.

**Tech Stack:** Rust, `std::sync::mpsc`, `std::thread`, `hipfire-runtime` (`SessionTable`, `AdmissionController`, `prefix`, `swap`), `hipfire-arch-qwen35` (`Scheduler`, `forward_batch_slots_graphed`).

## Global Constraints

- **The engine thread owns the GPU rig exclusively.** Nothing else touches `Gpu`, `SlotPool`, the arenas or the DeltaNet states; all interaction is by message.
- **Tokens are authoritative.** Any swap failure marks the session `Cold` and it re-prefills. No failure path may produce wrong output.
- **Never preempt a generating session.** Eviction picks LRU *idle* only; when every resident session is busy, a new request is rejected with a reason rather than queued indefinitely.
- Sampling happens only once a request's prompt is fully consumed — sampling mid-prefill and appending injects a generated token into the middle of the prompt.
- A dead client must not wedge the engine: a closed reply channel ends that request and frees its slot.
- `daemon.rs` and the existing `serve` path must keep working unchanged; the new backend is opt-in.
- Formatting: `BASE_REF=origin/beta ./scripts/fmt-changed.sh`. GPU runs under `./scripts/run-bounded.sh`.
- Existing gates stay green: `test_forward_slots_golden`, `test_prefix_cache_equivalence`, `test_swap_roundtrip`, `test_swap_equivalence`, `kernel_resource_gate.sh`, `attn_legacy_baseline.sh`.

## File Structure

| File | Responsibility |
|---|---|
| `crates/hipfire-runtime/src/serve/engine.rs` (create) | `SlotEngine`: thread, channel API, admission + eviction + scheduling loop |
| `crates/hipfire-runtime/src/serve/mod.rs` (create) | module registration, `SubmitRequest`/`Event` types |
| `crates/hipfire-runtime/examples/test_serve_concurrent.rs` (create) | the SP7 gate: 6 concurrent submissions on 4 slots |
| `crates/hipfire-cli/src/main.rs` (modify) | dispatch `/v1/chat/completions` to `SlotEngine` when `serve.multi_slot` is set |

---

### Task 1: `SlotEngine` — types and the request lifecycle

**Files:**
- Create: `crates/hipfire-runtime/src/serve/mod.rs`, `crates/hipfire-runtime/src/serve/engine.rs`
- Modify: `crates/hipfire-runtime/src/lib.rs`

**Interfaces:**
- Produces:
  - `pub struct SubmitRequest { pub session: Option<u64>, pub prompt_tokens: Vec<u32>, pub max_tokens: usize, pub reply: std::sync::mpsc::Sender<Event> }`
  - `pub enum Event { Accepted { session: u64 }, Token { id: u32 }, Done { reason: DoneReason }, Rejected { reason: String } }`
  - `pub enum DoneReason { Eos, MaxTokens, ClientGone }`
  - `pub struct SlotEngine { tx: Sender<SubmitRequest>, handle: JoinHandle<()> }` with `SlotEngine::submit(&self, req) -> Result<(), String>` and `SlotEngine::stats() -> EngineStats`.
  - `pub struct EngineStats { pub admitted: usize, pub rejected: usize, pub evictions: usize, pub restores: usize }`

Task 1 covers the types plus the non-GPU request bookkeeping, which is what can be unit-tested without a model. The GPU loop is Task 2.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn a_request_whose_client_vanished_is_detected_before_work_is_done() {
        let (tx, rx) = channel::<Event>();
        drop(rx);
        assert!(
            send_event(&tx, Event::Token { id: 1 }).is_err(),
            "a dropped receiver must be visible so the engine can free the slot"
        );
    }

    #[test]
    fn a_live_client_receives_its_events_in_order() {
        let (tx, rx) = channel::<Event>();
        send_event(&tx, Event::Accepted { session: 4 }).unwrap();
        send_event(&tx, Event::Token { id: 7 }).unwrap();
        send_event(&tx, Event::Done { reason: DoneReason::Eos }).unwrap();
        assert!(matches!(rx.recv().unwrap(), Event::Accepted { session: 4 }));
        assert!(matches!(rx.recv().unwrap(), Event::Token { id: 7 }));
        assert!(matches!(rx.recv().unwrap(), Event::Done { reason: DoneReason::Eos }));
    }

    #[test]
    fn stats_start_at_zero_and_count_each_outcome() {
        let mut s = EngineStats::default();
        s.note_admitted();
        s.note_admitted();
        s.note_rejected();
        assert_eq!(s.admitted, 2);
        assert_eq!(s.rejected, 1);
        assert_eq!(s.evictions, 0);
    }
}
```

- [ ] **Step 2: Run to verify they fail.** `cargo test -p hipfire-runtime --lib serve` → `send_event` not found.

- [ ] **Step 3: Implement** the types above plus
  `pub fn send_event(tx: &Sender<Event>, e: Event) -> Result<(), String>` mapping `SendError` to a message, and `EngineStats::{default, note_admitted, note_rejected, note_eviction, note_restore}`. Register `pub mod serve;` in `lib.rs`.

- [ ] **Step 4: Run to verify they pass.** Expected `test result: ok. 3 passed`.

- [ ] **Step 5: Commit**

```bash
BASE_REF=origin/beta ./scripts/fmt-changed.sh
git add crates/hipfire-runtime/src/serve crates/hipfire-runtime/src/lib.rs
git commit -m "feat(serve): SlotEngine request/event types and lifecycle bookkeeping"
```

---

### Task 2: The engine loop

**Files:**
- Modify: `crates/hipfire-runtime/src/serve/engine.rs`

**Interfaces:**
- Consumes: Task 1's types; `SessionTable::{open, begin_turn, touch, lru_idle_victim, mark_swapped, mark_cold, mark_resident, close}`; `swap::{SwapManager, snapshot::{capture_slot, restore_slot, SnapshotStamp}}`; `Scheduler`, `PendingWork`, `forward_batch_slots_graphed`, `sample_per_slot`.
- Produces: `SlotEngine::spawn(cfg: EngineConfig) -> Result<SlotEngine, String>` where `EngineConfig { model_path: PathBuf, n_slots: usize, cap_tokens: usize, host_budget_bytes: u64, swap_dir: PathBuf }`.

**Loop shape:** one iteration = (drain pending submissions) → (one batched forward across all in-flight requests) → (post events).

- [ ] **Step 1: Implement `spawn` and the loop**

The thread builds the rig exactly as `examples/test_swap_equivalence.rs` does (model load, `SlotPool`, arenas, `DeltaNetState`s, `SlotDescStaging`, `PrefillBatchScratch`, `Qwen35Scratch`, logits, sample params), then loops:

1. `rx.try_recv()` until empty. For each `SubmitRequest`:
   - Admit via `SessionTable::open`. On failure, try `lru_idle_victim`; if one exists, `capture_slot` + `SwapManager::park` + `mark_swapped`, then retry. If still none, `send_event(Rejected)` and drop.
   - A continuing session that is `Swapped`: get a slot the same way, `unpark`, `restore_slot`, `mark_resident`. **Any `SwapError` → `mark_cold`, then re-prefill the whole conversation from `session.tokens`** — slow, never wrong.
   - `begin_turn` to compute the reusable prefix, seed `PendingWork` with `prompt[plan.reused..]` at `next_pos = plan.reused`, `send_event(Accepted)`.
2. If nothing is in flight, `rx.recv()` blocks until the next submission (so an idle engine does not spin).
3. Otherwise build the `PendingWork` vector, `scheduler.next_batch`, `forward_batch_slots_graphed`, `sample_per_slot`, download token ids.
4. For each in-flight request whose prompt is fully consumed: append the sampled token, `send_event(Token)`, `sessions.touch(id)`. On EOS or `max_tokens`, `send_event(Done)`, `sessions.close(...)`, free the slot. **If `send_event` fails the client is gone: close the session and free the slot immediately** rather than generating into a void.

- [ ] **Step 2: Build**

`cargo build --release -p hipfire-runtime --features deltanet,arch-qwen35`
Expected: builds clean.

- [ ] **Step 3: Commit**

```bash
BASE_REF=origin/beta ./scripts/fmt-changed.sh
git add crates/hipfire-runtime/src/serve/engine.rs
git commit -m "feat(serve): SlotEngine loop with admission, eviction and restore"
```

---

### Task 3: The SP7 gate — more concurrent requests than slots

**Files:**
- Create: `crates/hipfire-runtime/examples/test_serve_concurrent.rs`

**Why:** the programme's headline claim is "3–4 agents at once on one GPU". This is the first test that submits genuinely concurrent requests and checks each gets its own stream.

- [ ] **Step 1: Write the gate**

Spawn one `SlotEngine` with `n_slots = 4`, then submit **6 requests from 6 threads**, each with a distinct prompt, each reading its own `Receiver` until `Done`.

```rust
for (i, out) in outputs.iter().enumerate() {
    assert!(!out.tokens.is_empty(), "client {i} received no tokens");
    assert!(out.accepted || out.rejected, "client {i} got neither accept nor reject");
}
// Distinct prompts must give distinct continuations. Identical output from
// every client would mean the sessions are sharing state.
let distinct: HashSet<Vec<u32>> = served.iter().map(|o| o.tokens.clone()).collect();
assert!(distinct.len() > 1, "every client produced identical tokens — sessions are not isolated");
let s = engine.stats();
assert!(s.admitted >= 4, "at least the four slots should have been used");
assert!(
    s.evictions > 0 || s.rejected > 0,
    "6 concurrent requests on 4 slots must force an eviction or a rejection"
);
println!("ALL CHECKS PASS");
```

- [ ] **Step 2: Build and run**

```bash
cargo build --release -p hipfire-runtime --features deltanet,arch-qwen35 --example test_serve_concurrent
HIPFIRE_MEM_CAP=30G ./scripts/run-bounded.sh ./target/release/examples/test_serve_concurrent ~/.hipfire/models/qwen3.6-35b-a3b.mq4r
```
Expected: six streams, at least one eviction or rejection, distinct outputs, `ALL CHECKS PASS`, exit 0, no OOM.

- [ ] **Step 3: Re-run every existing gate**

```bash
for g in test_forward_slots_golden test_prefix_cache_equivalence test_swap_roundtrip test_swap_equivalence; do
  HIPFIRE_MEM_CAP=30G ./scripts/run-bounded.sh ./target/release/examples/$g ~/.hipfire/models/qwen3.6-35b-a3b.mq4r 2>&1 | tail -1
done
./scripts/kernel_resource_gate.sh > /tmp/kr.txt 2>&1 && diff -q /tmp/kr.txt scripts/kernel_resource_gate.beta.txt && echo GATE_OK
```
Expected: `ALL CHECKS PASS` four times, then `GATE_OK`.

- [ ] **Step 4: Commit**

```bash
BASE_REF=origin/beta ./scripts/fmt-changed.sh
git add crates/hipfire-runtime/examples/test_serve_concurrent.rs
git commit -m "test(serve): six concurrent requests on four slots stream independently"
```

---

### Task 4: Dispatch `/v1/chat/completions` to the engine

**Files:**
- Modify: `crates/hipfire-cli/src/main.rs`

**Interfaces:**
- Consumes: `hipfire_runtime::serve::{SlotEngine, SubmitRequest, Event}`.

- [ ] **Step 1: Add the config knob and the optional engine**

Add `serve.multi_slot` (bool, default `false`) alongside the existing `serve.host`/`serve.port` config reads. When set, `serve_foreground` builds a `SlotEngine` and stores it in `ServeShared` as `Option<SlotEngine>` — **outside** the `Mutex<ServeRuntime>`, which is the entire point: the mutex is what serialises requests today.

- [ ] **Step 2: Dispatch in the handler**

In the `(&Method::Post, "/v1/chat/completions")` arm, before the existing path:

```rust
if let Some(engine) = shared.slot_engine.as_ref() {
    // Concurrent path: no runtime lock is taken, so requests overlap.
    let (tx, rx) = std::sync::mpsc::channel();
    engine.submit(SubmitRequest { session: None, prompt_tokens, max_tokens, reply: tx })?;
    // stream rx -> HTTP response using the existing SSE writer
    return respond_streaming(request, rx);
}
```

The existing path is untouched and remains the default.

- [ ] **Step 3: Verify both paths**

```bash
cargo build --release -p hipfire-cli
# default path unchanged:
hipfire serve --help >/dev/null && echo CLI_OK
```
Expected: `CLI_OK`, and with `serve.multi_slot=false` the serve path behaves exactly as before.

- [ ] **Step 4: Commit**

```bash
BASE_REF=origin/beta ./scripts/fmt-changed.sh
git add crates/hipfire-cli/src/main.rs
git commit -m "feat(serve): dispatch chat completions to SlotEngine behind serve.multi_slot"
```

---

## Self-Review

**Spec coverage (SP4 §3.3 and §5 criteria 1–5).** Criterion 1 (3–4 concurrent clients each streamed) → Tasks 2–4. Criterion 2 (admission refuses with a reason rather than OOMing) → Task 2 step 1, asserted in Task 3. Criterion 3 (second turn reuses prefix) → the engine calls the same `begin_turn` already gated by `test_prefix_cache_equivalence`. Criterion 4 (single-session path unaffected) → `daemon.rs` untouched and the existing serve path is the default; verified by the unchanged gates. Criterion 5 (budget enforced) → `AdmissionController` in the admit path.

**Closes SP6's deferred item:** "wiring eviction into a request loop" is Task 2 step 1.

**Still deferred:** async write-back (`park` stays synchronous); multi-turn reuse over HTTP (the engine supports `session`, but OpenAI chat completions are stateless, so mapping conversations to sessions needs a keying decision — prompt-prefix matching or a client-supplied id — that belongs in its own increment).

**Type consistency.** `Event::{Accepted, Token, Done, Rejected}` and `DoneReason::{Eos, MaxTokens, ClientGone}` are defined in Task 1 and constructed with those names in Tasks 2–3. `SubmitRequest`'s four fields match between Task 1's definition and Tasks 2–4's construction. `EngineStats`'s four counters match between definition and Task 3's assertions.

---

## Execution record (2026-08-08)

**Tasks 1-3 done.** Protocol types, the `SlotEngine` loop, and the concurrency
gate are built, committed and green.

Gate result, 6 clients against 4 slots:

```
clients 0-3: 12 tokens each, 4 DISTINCT token streams
clients 4-5: REJECTED ("all slots busy")
admitted=4 rejected=2 evictions=0
```

The rejections are the specified policy rather than a shortfall — eviction
takes only *idle* sessions and all four residents were mid-generation. It does
mean **`evictions=0`, so the engine's eviction path is not exercised by this
gate.** It is covered at the SP6 level by `test_swap_equivalence`, but not yet
through the engine. Closing that needs a gate where a session goes idle between
turns, which is the same multi-turn shape Task 4 needs.

**Task 4 is now partly done — see the update below.**

~~**Task 4 (dispatch `/v1/chat/completions` to the engine) is NOT done.**~~ It is
the piece that makes any of this reachable by an agent, so its absence is the
gap that matters. Two things need deciding first rather than coding through:

1. **Session keying.** OpenAI chat completions are stateless: the client re-sends
   the whole conversation each turn. To get prefix reuse, the engine must map a
   request back to an existing session — either by longest-prefix match against
   live sessions (transparent, and the reuse machinery already computes exactly
   this) or by a client-supplied id (explicit, but no standard agent sends one).
   Prefix matching is almost certainly right, and it makes `SessionTable` the
   lookup index rather than a plain map.
2. **Streaming shape.** The handler must turn `Event`s into SSE deltas, and the
   existing serve path builds its response through `ServeRuntime`. The two need
   a shared writer, or the new path needs its own.

Neither is hard; both are decisions rather than typing, and getting them wrong
in a 12,000-line file that is the daily driver is expensive. The engine is
deliberately usable without them: `SlotEngine::spawn` + `submit` is the whole
API surface.


## Task 4 update (2026-08-08)

`hipfire serve` now answers `/v1/chat/completions` from the `SlotEngine` behind
`serve.multi_slot`. Verified against a real HTTP client:

```
"What is the capital of France?"    -> "The capital of France is Paris."
"Who described the laws of motion?" -> "...Philosophiae Naturalis Principia"
```

Three bugs surfaced only by running it end to end:

1. **The terminal callback was never invoked.** It stages the response body and
   signals the handler; without it every request returned "generation worker
   disconnected" — a worker that finished without reporting.
2. **`build_multi_turn` panics on System/Tool inside history.** System text
   belongs in `ChatFrame.system`, the final user turn in `.user`; history is
   only the prior User/Assistant exchange.
3. **DeltaNet state was not reset when a slot was reused.** `seq_len = 0` clears
   the KV, but DN state lives outside the KV arena, so every request after the
   first inherited the previous conversation's recurrent state and echoed or
   degenerated. The same trap SP6 documented for the swap unit, in a new place —
   worth noting that writing it down did not prevent hitting it again.

Plus a framing fix: thinking models need `AssistantPrefix::OpenThink`, as the
daemon's `spec_assistant_prefix` uses. With `Plain` the model loops on
`</think>` and re-opens user turns.

### RESOLVED (2026-08-08): concurrent, 1.82x

Widening the gate works. Measured against a live server, four
`/v1/chat/completions` at 48 tokens:

```
4 sequential : 3.114 s
4 concurrent : 1.705 s   -> 1.82x, 4/4 distinct correct answers
```

The earlier "hang" report was wrong, and both symptoms were the test harness:

- The hang was a bare `wait`, which waited on the backgrounded `serve` job as
  well as the `curl` jobs.
- The follow-up "connection refused" runs were hitting a server already killed
  when the invocation that started it ended — while `pgrep -f "hipfire serve"`
  reported it UP by matching its own shell command line.

Two harness rules worth keeping: `pgrep -f` matching a pattern that appears in
the checking command's own argv is a false positive, and a server backgrounded
in one invocation does not survive into the next. Start it and measure it in the
same invocation, waiting on explicit PIDs.

### Superseded: the earlier "still serialised" finding

Measured 4 concurrent against 4 sequential: **1.01x**. The requests reach the
engine correctly and produce 4/4 distinct correct answers, but they do not
overlap, because the HTTP admission gate is a single busy flag — one in-flight
request, which is what protected the single daemon.

`AdmissionState.busy: bool` is now `in_flight: usize` with an
`Admission::with_concurrency` constructor, and that much is in place. Wiring it
to the slot count made requests hang (empty responses, engine still up), so the
wiring is reverted rather than shipped broken. Diagnosing that hang is the next
step, and it is the only thing between here and genuine concurrency.
