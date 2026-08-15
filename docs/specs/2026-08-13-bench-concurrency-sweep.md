# `hipfire bench` concurrency sweep — both batching backends

- **Date:** 2026-08-13
- **Status:** design, awaiting approval
- **Branch:** `feat/batched-attn-impl`

## Problem

This tree now carries **two** concurrent-execution backends, and nothing
measures either from the standard benchmark surface:

| backend | lives in | selected by |
|---|---|---|
| `SlotEngine` (this branch, SP1–SP7) | in-process, `hipfire-arch-qwen35::serve_engine` | `serve.multi_slot` |
| continuous batching (merged from beta) | inside the daemon, `ContinuousBatchScheduler` | `serve.continuous_batch_size` + per-request `serve_continuous_batch` |

They are mutually exclusive at runtime: `complete_request`
(`crates/hipfire-cli/src/main.rs:5156`) returns into `complete_request_slots`
whenever a slot engine is present and never reaches the daemon. So a
deployment runs one or the other, and today there is no way to ask which is
faster on a given workload.

`hipfire bench` is single-stream only. Every existing flag (`--runs`, `--pp`,
`--ctx`, `--tg`, `--max-tokens`, `--matrix`, `--redline`, `--spec`,
`--reasoning-on`) drives one request at a time. The only concurrency harnesses
that exist are `demo_multislot_generate` (env-driven `N_SLOTS`, one process per
count, SlotEngine only) and `scripts/serve_concurrency_gate.sh` (a pass/fail
gate at one slot count, not a sweep). Neither can compare the two backends.

**Goal:** one command that sweeps concurrency across both backends on the same
model and workload, so the choice between them is measured rather than argued.

## Non-goals

- Changing either backend's behaviour. This is measurement only.
- Deciding which backend wins, or removing one. The redundancy in
  `attention_q8_0_kv_batched` (which now carries both the descriptor and
  lane addressing schemes) is out of scope.
- Replacing `serve_concurrency_gate.sh`. That gate asserts correctness
  (HTTP 200, distinct answers, concurrent-faster-than-sequential); this
  reports throughput. Different jobs.
- Benchmarking through HTTP. Both drivers talk to their backend directly, so
  the numbers exclude the serve layer.

## CLI surface

```
hipfire bench <model> --concurrency 1,2,3,4 [--backend slots|batch|both]
                      [--workload stateless|multiturn|both]
                      [--runs N] [--max-tokens N]
```

- `--concurrency` absent ⇒ **bench behaves exactly as today**. The existing
  single-stream path is untouched; this is purely additive.
- `--backend` defaults to `both`.
- `--workload` defaults to `both`.
- `--runs` applies per concurrency point (default: existing value).
- `--max-tokens` is the per-stream token budget, reusing the existing flag.

## Architecture

One trait, two drivers, so the sweep loop and the reporting are written once
and neither backend can be measured on a different clock:

```rust
trait ConcurrencyBackend {
    /// Spawn once at max concurrency. Returns after the model is resident.
    fn start(model: &Path, max_concurrency: usize, cap_tokens: usize) -> Result<Self>;
    /// Run `k` streams to completion. Returns per-stream token counts and the
    /// wall-clock span from first submit to last completion.
    fn run(&mut self, arm: &Workload, k: usize, max_tokens: u64) -> Result<ArmResult>;
    /// Backend-specific evidence (prefix hits, admissions, rejections).
    fn stats(&self) -> BackendStats;
}
```

**`SlotBackendDriver`** — `SlotEngine::spawn(EngineConfig { n_slots:
max_concurrency, cap_tokens, host_budget_bytes, swap_dir })`, then `k` calls to
`submit(SubmitRequest { .. })`, each with its own `reply` channel, draining
`Event::Token` until `Event::Done`. `EngineStats` supplies
`prefix_hits`/`evictions`/`restores`.

**`DaemonBatchDriver`** — loads the model with `continuous_batch_size =
max_concurrency` in the **load params** (`main.rs:5356` is the existing call
site; the daemon answers `continuous_batch_capable` in the load reply). It is
fixed per load, so varying it means reloading the model — which is what forces
the fixed-max sweep below. Then **pipelines**: `Engine::send()` all `k` requests
before reading, then `recv()` interleaved frames, correlating by request id.
`Engine::generate()` is unusable here — it is request/response and would
serialise the very thing under test. Each request carries
`serve_continuous_batch: true`.

The sweep holds each backend at `max(concurrency)` and varies `k`. One model
load per backend for the whole sweep. **Consequence to state in the output:**
the KV arena is sized for the maximum at every point, so this is a concurrency
curve, not a memory-footprint curve.

## Workload arms

**`stateless`** — single user turn, plain string, no tools/stop/images. The
only shape *both* backends accept: beta's `is_batch_eligible_request`
(`main.rs:2748`) rejects tools, `tool_choice`, images, `stop`, speculation
keys, and anything other than exactly one user message. This arm is beta's
best case and does not exercise prefix reuse, swap, or eviction.

**`multiturn`** — turn 1, then turn 2 continuing the same conversation.
- SlotEngine: `session: Some(id)` + `convo` hashes + `continuation` tokens
  from `prompt_frame::continuation_suffix`, hitting the prefix cache.
- Daemon batch: ineligible (multi-turn fails `batch_messages_are_single_user`),
  so it falls back to sequential.

This arm is **not** a like-for-like batching comparison and must be labelled as
such: it compares batched-with-reuse against sequential. It is included because
it is the shape agent traffic actually takes, and the stateless arm alone would
overstate beta's applicability.

**Correctness assertion, not a latency proxy:** the multi-turn arm asserts
`EngineStats.prefix_hits` increased on turn 2. A faster second turn is *not*
evidence of reuse — it could be cache warmth. If `prefix_hits` did not move,
the arm reports the run as invalid rather than publishing a number.

## Metrics

Reported per (backend, workload, k):

- **aggregate tok/s** — total generated tokens ÷ wall-clock from first submit
  to last completion. The primary number.
- **per-stream tok/s** — aggregate ÷ k.
- **wall-clock ms** to last completion.
- backend evidence: `prefix_hits`, `evictions`, `restores`, `rejected`.

Aggregate throughput is the only metric both backends can report on equal
terms, so it is the only one compared. The SlotEngine's internal `ms/step` is
deliberately **excluded**: the daemon path has no comparable figure, and
pairing them would flatter whichever side happened to report the friendlier
decomposition.

### Declared unfairness

`SlotEngine` runs in-process. The daemon path pays JSONL encode/decode over a
pipe per token. That difference is inherent to where each backend lives and
cannot be normalised away — it is a real cost of the daemon architecture, but
it is not a property of the *batching* algorithm. The output prints this
caveat next to the comparison table rather than leaving the reader to assume
the numbers are pure.

## Methodology

- **Interleaved sweep.** Points run `1,2,3,4,1,2,3,4,…` per `--runs`, never
  blocked as `1,1,1,2,2,2`. Report the **median** per point.

  This is not theoretical caution. During this session a blocked single-run
  sweep produced 46.63 tok/s at 4 slots — below the 1-slot figure — and an
  interleaved 3-round repeat put the same point at 108–118 tok/s. The apparent
  cliff was thermal drift on a shared-TDP APU. A blocked sweep would have
  shipped that as a finding.

- **Warmup discarded.** One untimed stream per (backend, k) before timing,
  matching `demo_multislot_generate`'s discarded warmup step.

- **Answer mode.** All requests inherit `bench_generate_request`'s answer-mode
  default, so neither backend dies on an open-think validation terminal.

- **Identical prompts** across backends, fixed in the source, with per-stream
  variation so slots cannot alias each other's KV undetected.

## Error handling

| condition | behaviour |
|---|---|
| model's dtypes fail the slot gates (`require_batchable_*`) | report backend unavailable with the gate's own error, continue with the other backend |
| daemon does not advertise `continuous_batch_capable` | same — report and continue |
| a stream is `Rejected` | record it in `BackendStats.rejected`; do not count its tokens |
| `k` exceeds a backend's configured max | error before running, not silently clamped |
| multi-turn arm shows no `prefix_hits` increase | mark that arm invalid; print why; do not report its tok/s |

A backend being unavailable is a **result**, not a crash — "SlotEngine cannot
run this model" is exactly the kind of finding this tool exists to surface.

## Testing

- Unit: sweep-order generation is interleaved, not blocked.
- Unit: median selection over per-point repeats.
- Unit: `--concurrency` absent produces a request identical to today's
  single-stream path (guards the additive claim).
- Unit: `k > max_concurrency` errors rather than clamping.
- Unit: an arm with no `prefix_hits` movement is marked invalid.
- Integration (GPU, gated): 1..4 on both backends against
  `qwen3.6-35b-a3b.mq4r`, asserting every point produces a positive
  aggregate and the multi-turn SlotEngine arm records prefix hits.

## Risks

- **`continuous_batch_size` is fixed per model load.** Holding it at max and
  varying `k` is the design's core compromise. If beta's scheduler behaves differently
  at `max_batch=4, k=2` than at `max_batch=2, k=2`, this sweep will not see it.
  A `--reconfigure-per-point` mode is the fallback, at one model load per
  point; deliberately deferred as YAGNI until the fixed-max numbers look wrong.
- **Pipelined `send`/`recv` correlation.** If the daemon interleaves frames
  from different requests without stable ids, the driver cannot attribute
  tokens. `AttemptKey` exists for this, but the driver must be verified
  against real interleaved output before its numbers are trusted.
- **Thermals.** Medians over interleaved repeats mitigate but do not eliminate
  this. Any single-run comparison from this tool should be treated as
  indicative only.

## Verified before writing this spec

- SlotEngine runs `qwen3.6-35b-a3b.mq4r` today (MoE gates admit uniform
  MQ4G256 attention projections plus an admissible MoE FFN,
  `forward_slots.rs:261-345`). An earlier claim in this session that it could
  not was wrong — it came from `demo_multislot_generate`'s stale header
  comment, contradicted by the same file's module doc.
- Measured curve, gfx1151, 3 interleaved rounds, medians:
  1 slot 64.71 · 2 slots 80.15 · 3 slots 97.07 · 4 slots 116.23 tok/s.

## Results — 2026-08-13, gfx1151, `qwen3.6-35b-a3b.mq4r`

`hipfire bench qwen3.6:35b-a3b-mq4r --concurrency 1,2,3,4 --runs 3
--max-tokens 64 --backend slots`, medians of 3 interleaved rounds:

| workload | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| stateless | 33.55 | 57.08 | 70.14 | **84.69** |
| multiturn | 36.17 | 59.41 | 66.11 | **76.19** |

Prefix hits were 3 / 6 / 9 / 12 — exactly `k` per run across 3 runs — so every
multi-turn stream reused its session KV. Zero `[invalid]` arms, zero rejections.

**These are NOT comparable to the 64.71–116.23 figures above.** `ArmResult`
measures wall clock from first submit to last completion, so prefill, session
setup and the second turn are all inside the denominator; the earlier numbers
are `demo_multislot_generate`'s decode-only `ms/step`. Both are internally
consistent; only compare within a table. The shape is what carries: throughput
rises monotonically to k=4 in both arms.

The multi-turn arm is *lower* than stateless at k≥3 despite reusing KV, because
it runs two turns per sample and pays a second prefill and scheduling round.
Reuse makes turn 2 cheaper than a cold render, not free.

### slots vs no-slots — measured, but NOT a clean comparison

`--backend noslots` runs k requests strictly one after another through the
ordinary daemon path: what a box serves today with `serve.multi_slot` off.
Same clock, same prompts, same 64-token budget, medians of 3 interleaved runs.

| backend | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| noslots (sequential daemon) | 52.90 | 85.64 | 89.47 | **91.53** |
| slots (multi-slot engine) | 33.55 | 57.08 | 70.14 | **84.69** |

Taken at face value the slot engine loses at every point. The two arms are not
running the same pipeline, but **only one of the two suspected differences is
real** — an earlier revision of this document claimed both, wrongly:

- ~~the daemon path engages the **MTP speculative head**~~ — **NO.** The
  sidecar loads and logs `MTP head loaded`, but engagement is gated behind
  `HIPFIRE_QWEN_MTP=1` (`daemon.rs:1494`; the path at `:20871` documents
  itself as "only reached when HIPFIRE_QWEN_MTP=1"). That variable was never
  set in any run here, so MTP was **loaded but never used**. The log line is
  not evidence of speculation. This was a misread.
- the daemon path **does** enable **redline retained/PM4 replay**, and the
  in-process `SlotEngine` cannot: redline lives in the daemon, and the slot
  path never goes through it.

Redline is not switchable for this model — `mq4r_redline_default`
(`hipfire-runtime/src/config.rs:14`) hardcodes it on for a `.mq4r` extension on
gfx1100/1151/1201 at pp=1, tp=1, with no env or config override. So it cannot
be turned off to isolate it; it is a genuine property of the daemon path rather
than a benchmark artifact. The redline campaign took mq4r from ~110 to ~204
tok/s single-stream, so it is more than large enough to account for the gap.

Two further reasons to distrust the shape:

1. **The no-slots curve should be FLAT.** Strictly sequential execution means
   k requests take k× as long, so aggregate throughput is independent of k.
   It instead rises 52.90 → 91.53 and plateaus, which is a fixed per-run cost
   (first-generate warmup) being amortised over more tokens. Its steady-state
   number is the plateau, ~91, not the k=1 figure.
2. **64 tokens is too short.** At this budget prefill and per-turn overhead
   dominate the wall clock, and batching's win is in decode. A decode-dominated
   budget (512+) is where the slot engine should show its advantage — and the
   decode-only measurement earlier in this document has slots at 116 tok/s
   aggregate at 4 slots against ~98 single-stream.

### Decode-dominated re-run: same ordering

`--concurrency 1,4 --runs 2 --max-tokens 512 --workload stateless`, with
`--spec off` on the no-slots arm:

| backend | k=1 | k=4 |
|---|---|---|
| noslots | 54.28 | **99.39** |
| slots | 40.64 | **81.43** |

At 512 tokens decode dominates the wall clock, so the ordering is not a
prefill artifact: **the sequential daemon path beats the slot engine by
~20–25% at both concurrencies, on two independent token budgets.**

One thing this run did NOT establish, and it matters:

1. **The intra-arm k=1 → k=4 rise is not trustworthy at `--runs 2`.** A
   strictly sequential arm must have flat aggregate throughput in k, yet
   noslots goes 54.28 → 99.39, which a 512-token budget is far too long for
   warmup amortisation to explain. With `sweep_order` visiting `1,4,1,4` and
   the median of two samples being their mean, the first (cold) k=1 run
   contaminates that point. Compare ACROSS arms at matched k; do not read the
   within-arm curve from this run.

### Definitive run — `--runs 5`, 512 tokens

The `--runs 2` results above have a cold-sample problem (median of two = their
mean). Re-run at `--runs 5`, where a single cold sample cannot own the median:

| backend | k=1 | k=4 |
|---|---|---|
| noslots (sequential daemon) | 51.68 | **95.15** |
| slots (multi-slot engine) | 34.37 | **78.45** |

**noslots wins by 50% at k=1 and 21% at k=4.** Consistent with both earlier
runs; the ranking has now held across three independent measurements and two
token budgets, so it is not a sampling artifact.

The narrowing gap is the one hint that batching is doing real work: the slot
engine recovers from −50% at k=1 to −21% at k=4, i.e. it scales better with
concurrency (2.28× from k=1→k=4, versus 1.84× for sequential) even while
losing on absolute throughput. Extrapolating that trend past k=4 is not
supported by this data — it is two points.

The `noslots` k=1→k=4 rise (51.68 → 95.15) is still larger than a strictly
sequential path permits, and `--runs 5` did not remove it. Per-run fixed cost
is the remaining explanation, but it is no longer a cold-median artifact and
is not fully accounted for. Treat the cross-arm comparison at matched k as
sound and the within-arm curve as still unexplained.

### Corrected run — unique prompts, and the slot decode graph switched on

Every measurement above shares a bias found later: the sweep cycled a fixed
prompt list, and the two backends cache on different keys.

- The daemon holds a **token-level longest-common-prefix** cache, so a repeated
  prompt can skip prefill.
- The slot engine keys on **conversation identity**, and `find_continuation`
  returns `None` immediately when `convo.len() < 2`
  (`session_table.rs:170`) — it matches *continuations*, never *repeats*. A
  resent identical prompt can never hit.

So repeats handed the daemon free prefills and gave the slot engine nothing.
`stream_prompt(run, stream)` now prefixes every prompt with a unique `Q{run}-{stream}`
tag so neither side can cache.

Separately: **the slot engine's own hipGraph decode capture was off in every
run above.** `forward_batch_slots_graphed` (`forward_slots.rs:2312`) is gated on
`HIPFIRE_SLOTS_DECODE_GRAPH=1`, which defaults false (`feature_flags.rs:498`,
`:715`); the earlier demo output confirms it — `decode graph: 0 capture(s), 0
replay(s)`. So the slot arm ran with no graph replay against a daemon arm that
had redline replay on automatically.

Re-run with both corrections, `--runs 5`, 512 tokens:

| backend | k=1 | k=4 |
|---|---|---|
| noslots, unique prompts | 48.78 | **94.49** |
| slots, unique prompts + decode graph | 33.28 | **84.58** |

**The prompt-cache confound was real but small:** noslots moved 51.68 → 48.78
at k=1 (−5.6%) and 95.15 → 94.49 at k=4 (−0.7%). It never explained the gap.

**noslots still wins**, by 47% at k=1 and 12% at k=4 — but the k=4 gap has
closed from 21% to 12%.

#### What the decode graph is actually worth — one-variable control

The run above moved two variables at once, so a control was run with unique
prompts and the graph OFF, everything else identical:

| slots, unique prompts | k=1 | k=4 |
|---|---|---|
| decode graph ON | 33.28 | 84.58 |
| decode graph OFF | 33.38 | 81.22 |
| **delta** | −0.3% (noise) | **+4.1%** |

**The decode graph is worth ~4% at k=4 and nothing at k=1.** Real, repeatable,
and far too small to close a 12% gap. The larger improvement credited to it in
the first corrected run was mostly the prompt change, not the graph.

The zero at k=1 is mildly surprising — a single-slot pure-decode step is the
easiest thing to capture — and suggests capture overhead cancels the replay
saving when there is only one slot's worth of launches to amortise. Note also
that engagement is still not *directly* observed: `SlotDecodeGraph::stats()`
exposes `(captures, replays)` and the sweep never prints it. The A/B is
consistent with the feature firing at k=4, but a counter in the output would
settle it properly.

### What this actually says

The slot engine loses to the plain daemon path today, consistently. The one
substantiated difference between the arms is **redline retained/PM4 replay**,
which the daemon gets automatically for `.mq4r` and the in-process `SlotEngine`
structurally cannot reach. MTP is NOT part of the explanation — it was never
engaged. If redline is the cause, the lever is bringing that replay path to the
slot engine rather than tuning the batching, and the slot engine's own
decode-only figure (116 tok/s aggregate at 4 slots) indicates the headroom.

**This remains a hypothesis, not a demonstrated cause.** Redline cannot be
disabled for `.mq4r`, so it cannot be isolated on this model. The clean test is
to re-run both arms on a SKU whose extension does not trigger the default (e.g.
`.mq4p`), where the daemon runs plain HIP too: if the gap closes, redline
explains it; if it persists, something in the slot path itself does. That test
has not been run.

### The batch backend is NOT measured

`--backend batch` cannot run. `hipfire_client::Engine` has no public
multi-inflight API: its reader thread drops lifecycle frames whose
`(id, attempt_id)` has no registered channel, registration happens only inside
the blocking single-attempt `generate`, and the `pending` map is private. A
pipelined `send`×k then `recv` therefore deadlocks — observed twice, with the
daemon reporting `continuous batch staged: slots=4` and neither side moving.
`DaemonDriver::start` now fails with that explanation instead of hanging.

Unblocking needs either a public `Engine::submit_streaming` returning a
registered receiver, or an HTTP driver through `hipfire serve` (folding HTTP
into the measurement, which this spec deliberately excluded). Until then the
two backends cannot be compared, and the question that motivated this work —
whether `SlotEngine` beats beta's continuous batching — remains open.

### Operational note

The first end-to-end attempt exhausted host RAM. `bench_concurrency_command`
held the slots engine resident while the daemon loaded a second copy of the
weights. Fixed in `bfc4dca9a` (scoped drop plus `preflight_headroom_for_model`).
The sweep's `cap_tokens` is 2048 and its host swap budget 2 GiB — both smaller
than the serve defaults (8192 / 16 GiB), because a sweep opens a fresh session
per stream per run and the engine keeps finished sessions resident for reuse.
Peak host usage for a slots-only sweep is ~88 GB of 125 GB; run it under a
memory watchdog.
