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

**`DaemonBatchDriver`** — spawns the daemon with `continuous_batch_size =
max_concurrency` in `ProcessConfig` (it is fixed at configure time, so it
cannot vary per point), then **pipelines**: `Engine::send()` all `k` requests
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

- **`continuous_batch_size` is configure-time.** Holding it at max and varying
  `k` is the design's core compromise. If beta's scheduler behaves differently
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
