# Low-risk Crate Integrations — Evaluation and Adoption Plan

> **For agentic workers:** each candidate is an independent measured screen.
> Do not bundle candidates into one diff. A candidate that misses its kill gate
> is removed completely before the next candidate begins.

**Status:** proposed  
**Lifecycle:** `active` implementation plan  
**Goal:** Adopt only small, maintained Rust dependencies that measurably improve
hipfire's host performance or operator experience without changing GPU math,
model output, or portability by accident.

## 1. Principles

- Prefer ordinary Cargo dependencies pinned by `Cargo.lock`; do not vendor
  source unless hipfire must carry a reviewed patch.
- One dependency, one question, one commit, one revert point.
- Product behavior and output parity are mandatory; microbench wins alone do
  not justify adoption.
- Never move Python/JavaScript into the user-facing control plane.
- Do not add nightly Rust to shipped crates.
- Record license/attribution before merging external code.
- Measure fresh-process load, steady-state serve, CPU, RSS, and decoded output
  where relevant.

## 2. Already adopted during the campaign

The native server now uses:

- `hyper` 1;
- `hyper-util`;
- `http-body-util`;
- `bytes`;
- `tokio`;
- `tokio-util`.

Commit `158ff5ec6` removed the local tiny_http fork. Retained measurements on the
same Qwen3.8 fixture showed `/health` p50 about 2.4× faster and one-token request
latency unchanged/slightly lower. These dependencies are not candidates in this
plan; they are the baseline.

## 3. Candidate A — TCP_NODELAY (no new crate)

Set `TCP_NODELAY` on accepted Tokio streams before passing them to Hyper.

### Question

Does disabling Nagle reduce SSE/token-chunk tail latency for remote clients
without increasing CPU or packet overhead materially?

### Work

- [ ] Set `stream.set_nodelay(true)` on accepted TCP streams.
- [ ] Add a focused failure diagnostic; inability to set the option must not
      silently change server startup.
- [ ] Measure local and remote SSE first-token/chunk p50/p95/p99.
- [ ] Measure short non-stream requests and CPU.

### Kill gate

Remove the change if remote streaming tails do not improve beyond run noise or
if CPU/packet overhead regresses materially. This screen lands separately from
all crate candidates.

## 4. Candidate B — `crossbeam-channel`

### Scope

Only the high-frequency daemon lifecycle/event channels in `hipfire-client` are
candidates. Do not globally replace every `std::sync::mpsc` channel.

### Question

Does Crossbeam reduce host overhead or tail latency when routing token events
at high token rates or high concurrency?

### Work

- [ ] Inventory request-control, pending-attempt, token-event, abort/drain, and
      test-only channels.
- [ ] Identify channels that require disconnect semantics, blocking receive,
      timeout receive, or multiple selection.
- [ ] Replace one complete event path as a clean cutover; no adapter layer.
- [ ] Preserve exact request/attempt correlation and terminal ordering.
- [ ] Retain the existing cancellable receive polling bound.
- [ ] Add channel-close, abort/drain, commit-boundary, stale-event, and pending
      map cleanup tests.

### Measurements

- synthetic event routing at 30, 500, and multi-stream aggregate events/s;
- real small-model streaming throughput and CPU;
- Qwen3.8 one-token and normal generation latency;
- concurrent serve cancellation tails.

### Kill gate

Remove `crossbeam-channel` unless it improves an actual serve/host metric beyond
noise. A channel microbench alone is insufficient. No decoded output may change.

## 5. Candidate C — `serde_path_to_error`

### Scope

Use only at human-facing JSON boundaries:

- OpenAI request bodies;
- registry/cache documents;
- model/local configuration documents where Serde owns decoding.

### Question

Can hipfire report the exact malformed field path without obscuring existing
status/error classes?

### Work

- [ ] Wrap typed deserialization at one boundary first.
- [ ] Preserve request size limits and 400/413 status selection.
- [ ] Return a stable message containing the failing JSON path and underlying
      type error.
- [ ] Add nested tool-schema, reasoning, sampling, and registry fixture tests.

### Kill gate

Remove if it requires broad request representation changes, worsens error status
compatibility, or adds measurable large-body overhead. This is primarily an
operator-experience candidate, not a throughput claim.

## 6. Candidate D — allocator A/B (`mimalloc`)

### Policy

This begins as a Linux-only, feature-gated experiment. It is not a default
allocator change.

### Question

Does mimalloc reduce model-load time, host allocation churn, or concurrent serve
CPU/RSS on hipfire's real workloads?

### Work

- [ ] Add an opt-in Cargo feature selecting the allocator in the final binary.
- [ ] Keep the default system allocator unchanged during the screen.
- [ ] Compare fresh-process model load/prewarm, short-request churn, multi-slot
      serving, long-session cache use, peak RSS, and process teardown.
- [ ] Test Linux x86_64 and supported non-Linux builds for feature isolation.

### Kill gate

Do not promote unless at least one product wall-time/CPU metric improves
repeatably, RSS remains acceptable, and shutdown/model-unload behavior remains
correct. Remove the dependency if only allocator microbenchmarks improve.

## 7. Candidate E — `hyper-rustls` (product decision required)

### Question

Should `hipfire serve` terminate TLS itself, or should deployment continue to
use a reverse proxy/service mesh?

Do not implement until the product decision is explicit.

If approved:

- [ ] add a feature-gated TLS listener using the existing Hyper router;
- [ ] keep plain local HTTP available;
- [ ] define certificate/key reload and failure policy;
- [ ] cover streaming, disconnect cancellation, client auth expectations, and
      HTTP status parity;
- [ ] measure handshake/resumption overhead.

### Kill gate

Reject if native TLS duplicates the supported deployment model without a clear
operator benefit. Do not add TLS merely because Hyper makes it convenient.

## 8. Explicit rejections / deferrals

### Gigatoken — rejected

Prior hipfire measurement showed about 0.1 tok/s at roughly 20K context. Its
bulk dataset design, nightly `portable_simd`, Python/PyO3 dependency graph,
mutable caches, and >=1 MiB parallel threshold do not fit hipfire's interactive
inference tokenizer. Do not revisit for performance unless a distinct tokenizer
correctness defect appears.

### Global hasher migration — rejected

Do not replace every `HashMap` hasher. Redline PM4 dispatch already receives a
resolved kernel; visible kernel-name lookup is setup-time/linear. Profile a
specific trusted internal map before considering `rustc-hash` or `ahash`.

### PM4 `ArrayVec` dependency — deferred

First implement the Redline fixed-buffer packet writer with a plain bounded
array. Add a crate only if the dependency-free representation proves materially
worse.

### `simd-json` — deferred

Request and lifecycle JSON objects are small. Do not trade API/unsafe complexity
for bulk parsing claims without a profile proving JSON is a significant host
cost at target concurrency.

### Tower wholesale — rejected for now

Hipfire admission is model-aware and cancellation-correlated. Generic timeout /
concurrency middleware must not replace those contracts. Adopt individual
middleware only for a demonstrated need.

### HTTP/2/3 — deferred

The new HTTP/1 transport is measured and cancellation-safe. Multiplexed
protocols require their own stream cancellation, framing, and performance gates.

## 9. Execution order

1. TCP_NODELAY screen (no dependency).
2. `crossbeam-channel` measured cutover.
3. `serde_path_to_error` operator-error improvement.
4. Feature-gated allocator A/B.
5. `hyper-rustls` only after an explicit product decision.

Each step starts from a clean master commit and ends in either one promoted
commit or complete removal. Do not stack unproven candidates.

## 10. Success criteria

This plan is complete when every candidate has a recorded promote/reject
outcome, promoted crates have focused behavioral tests and license inventory,
and no dependency remains solely because a synthetic microbenchmark looked
faster.
