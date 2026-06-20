# Plan: read-only training monitoring for operator UIs

Status: **proposed** - 2026-06-20.

## Goal

Add first-class training monitoring to hipfire's optional operator clients: the
browser WebUI and `hipfire-tui`. Both clients should consume the same typed
operator API and render the same training run state differently.

This plan extends `docs/plans/2026-06-20-operator-ui-clients.md` and aligns with
the daemon-training direction in:

- `docs/plans/2026-06-19-train-as-daemon-op.md`
- `docs/plans/2026-06-19-training-via-daemon-forward.md`

The first implementation pass is read-only. It observes training jobs, progress,
metrics, checkpoints, and export/admission state without launching, cancelling,
or mutating runs.

## Non-goals

- Do not make the WebUI or TUI own trainer lifecycle in this phase.
- Do not duplicate training interpretation logic between the clients.
- Do not scrape terminal logs as the primary state source.
- Do not put Python in the inference hot path.
- Do not require the TUI or WebUI to use hipfire training.
- Do not assume the daemon is locally owned; it may be supervised by systemd,
  launchd, a container, or a remote process.

## Shared Training State

Training monitoring should be based on durable structured state plus a live event
stream.

### Run Summary

Each run should expose:

- `id`: stable run identifier.
- `kind`: drafter, recovery, QAT, calibration, export, or future trainer class.
- `status`: queued, capturing, training, evaluating, checkpointing, exporting,
  completed, failed, cancelled, or unknown.
- `target_model`: teacher or base model identifier.
- `artifact`: expected output artifact path or ID.
- `created_at`, `started_at`, `updated_at`, `completed_at`.
- `progress`: current phase, current step, total steps when known, percent when
  meaningful, and ETA when available.
- `metrics`: latest loss, eval metric, best eval metric, learning rate,
  throughput, and wall-clock timing.
- `checkpoint`: latest checkpoint path, age, resume source, and checkpoint state.
- `handoff`: export/admission target, admission verdict, and evidence artifact
  links when present.
- `last_error`: structured error or warning summary.

### Run Events

Events should be append-only JSONL and streamable through the operator API:

- `run_started`
- `phase_started`
- `capture_progress`
- `train_progress`
- `eval_progress`
- `metric`
- `checkpoint_written`
- `export_started`
- `export_done`
- `admission_started`
- `admission_done`
- `warning`
- `error`
- `run_done`

Events should be resilient to older trainers. Unknown event types must be
preserved and rendered as generic events, not dropped.

### Durable Location

Prefer a stable local layout:

```text
~/.hipfire/training/runs/<run-id>/status.json
~/.hipfire/training/runs/<run-id>/events.jsonl
~/.hipfire/training/runs/<run-id>/artifacts/
```

Daemon-integrated training can write this directly. Standalone training tools can
write the same files, allowing the operator clients to monitor both daemon-owned
and tool-owned runs.

## Operator API

Add read-only endpoints under the operator namespace:

```text
GET /operator/training/runs
GET /operator/training/runs/{id}
GET /operator/training/runs/{id}/events
```

The event endpoint may start as bounded JSONL and later support SSE or a watch
mode. The response vocabulary should stay the same across JSONL, SSE, and any
future WebSocket transport.

## Feature Targets

### Shared Data Model

Target goals:

- Define serializable Rust structs for run summaries, progress, metrics,
  checkpoints, handoff state, errors, and events.
- Support unknown event kinds without losing raw payloads.
- Load missing or partially written run files gracefully.
- Include unit tests for serialization, default handling, and corrupt-file
  tolerance.

Done when: a no-GPU test can create sample run directories, load them through the
shared model, and produce stable JSON snapshots.

### Training Run Discovery

Target goals:

- Discover run directories under `~/.hipfire/training/runs`.
- Sort active and recent runs predictably: active first, then most recently
  updated.
- Distinguish daemon-owned, standalone-tool-owned, and unknown-owner runs when
  metadata exists.
- Surface stale runs whose status has not updated recently.

Done when: `/operator/training/runs` returns useful summaries for empty,
completed, active, failed, and stale run directories.

### Training Event Reader

Target goals:

- Read `events.jsonl` incrementally without assuming the file is complete.
- Preserve malformed lines as structured read errors tied to line offsets.
- Expose latest N events for detail views.
- Leave room for tail/follow semantics without forcing it into the first pass.

Done when: a test fixture with mixed valid, unknown, and malformed events renders
a deterministic response and does not fail the whole run.

### Operator API Endpoints

Target goals:

- Add the three read-only training endpoints to `hipfire-server`.
- Keep endpoint payloads stable and documented through tests.
- Return empty lists cleanly when no training directory exists.
- Avoid coupling endpoint handlers to a specific trainer implementation.

Done when: route tests cover list, detail, missing ID, and events responses
without requiring a GPU.

### WebUI Training View

Target goals:

- Add a Training surface to the browser operator UI.
- Show active/recent run list, selected-run details, current phase, progress,
  loss/eval metrics, throughput, ETA, checkpoint status, latest warnings/errors,
  and export/admission state.
- Render empty, offline, loading, stale, failed, and completed states clearly.
- Use only operator API data; do not parse training files in browser code.

Done when: the WebUI can inspect sample run fixtures through the server and show
the same facts as the API response without any trainer running.

### TUI Training Tab

Target goals:

- Add a `Training` tab to `hipfire-tui`.
- Show a dense run list with status, phase, updated time, progress, best metric,
  and artifact.
- Show selected-run detail with latest metrics, checkpoint, handoff, and recent
  events.
- Keep keyboard behavior consistent with existing tabs.
- Keep the tab useful over SSH and on narrow terminals.

Done when: `cargo run -p hipfire-tui` can show sample training runs through the
same operator API vocabulary, with graceful fallback when the server is offline.

### Log and Error Display

Target goals:

- Treat structured events as the primary source.
- Show raw log excerpts only when a run provides them as artifacts or fallback
  metadata.
- Make the latest actionable error visible in both clients.
- Preserve the link from error to run, phase, checkpoint, and artifact.

Done when: a failed-run fixture exposes the same error summary in API, WebUI, and
TUI without requiring users to open a raw log first.

### Export and Admission Handoff

Target goals:

- Represent exported `.hfq` artifacts and sidecars using the canonical artifact
  naming convention.
- Show whether a run has pending, running, passed, failed, or missing admission
  evidence.
- Link to eval/admission artifacts when available.
- Avoid declaring a model promoted from UI state alone; promotion remains an
  explicit workflow.

Done when: completed-run fixtures can show "exported but not admitted",
  "admission failed", and "admission passed" states in both clients.

### Tests and Gates

Target goals:

- Add focused no-GPU tests for data loading, endpoint JSON, and UI state mapping.
- Run `cargo check` for affected crates.
- Run `./tests/no-gpu-ci.sh` before handoff if the change stays workflow/API/UI
  only.
- Do not run GPU coherence gates unless the implementation touches kernels,
  quant formats, dispatch, fusion, rotation, rmsnorm, or spec-decode paths.

Done when: no-GPU checks cover the training monitoring surface and the handoff
notes state exactly which gates were run.

## Implementation Sequence

1. Add the shared training monitoring structs and fixture loader.
2. Add run discovery and event reading from `~/.hipfire/training/runs`.
3. Add read-only operator API endpoints and tests.
4. Add the TUI `Training` tab against the shared API vocabulary.
5. Replace or extend the browser operator page with a Training view.
6. Add sample fixtures and no-GPU coverage for active, completed, failed, stale,
   and malformed runs.
7. Later phase: add live watch mode and typed actions for launch, cancel,
   checkpoint, resume, and admission.

## Open Questions

- Should the first live stream be SSE, JSONL follow, or both over the same event
  vocabulary?
- Should run state live in a new shared crate or an existing serving/operator
  boundary?
- How much local process ownership should the clients expose once mutation
  actions are added?
- Should standalone training tools write status/events directly, or should they
  report through the daemon when available?
