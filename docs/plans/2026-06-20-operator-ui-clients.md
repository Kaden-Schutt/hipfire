# Plan: optional operator UIs over the daemon/service API

Status: **proposed** - 2026-06-20.

## Goal

Build hipfire's operator experience as optional clients over daemon/service APIs,
not as a mandatory TUI runtime. The TUI and a future WebUI should both connect to
the daemon dynamically, observe the same state, and issue the same typed actions.

The UI surface should cover:

- model inventory, aliases, downloads, active model, sidecars, and load state;
- server lifecycle, health, logs, resource locks, idle eviction, and request
  activity;
- config and per-model overrides, including source/default/override metadata;
- eval runs, admission evidence, batteries/suites, and result artifacts;
- training feedback: capture progress, live loss/quality metrics, timing/ETA,
  checkpoint/resume status, and export/admission handoff.

## Non-goals

- Do not require users to run a TUI to use hipfire.
- Do not put product logic in the TUI or WebUI that is unavailable to CLI/API
  callers.
- Do not make the UI own daemon lifetime when hipfire is running as a systemd,
  launchd, container, or manually managed service.
- Do not make a browser UI a separate backend with its own model/config/eval
  interpretation.

## Client Model

The daemon/server should expose a small operator API in addition to inference
routes. Both TUI and WebUI consume that API:

- discovery: probe configured host/port, then optional well-known local fallbacks;
- identity: report daemon version, PID/service identity, uptime, build features,
  GPU/NPU inventory, and active resource leases;
- health: structured readiness and degraded states, not just HTTP 200/500;
- watch: stream operator events over SSE/WebSocket/JSONL so UIs do not poll heavy
  endpoints;
- actions: typed requests for load/unload, start/stop where locally owned,
  config update, eval launch/cancel, training launch/cancel/checkpoint, and log
  tail.

If the daemon is not reachable, the UI should show an offline state and offer
only actions that are valid in that context. Starting a local daemon can be an
optional convenience, but the UI must not assume it owns the process.

## Shared State Surfaces

### Models

Use shared model/registry code for local `.hfq` discovery, canonical artifact
names, aliases, sidecars, size/VRAM metadata, and loadability checks. The UI
should not parse a parallel registry format.

### Server

Expose server and daemon status through typed status structs: current model,
request queue, active sessions, scheduler state, health, idle eviction countdown,
resource locks, and recent errors. The TUI/WebUI should render the same status
data differently, not derive it from logs.

### Config

Centralize config field metadata: key, type, default, allowed values, global vs
per-model mutability, source, validation, and restart/reload impact. This enables
CLI `config`, TUI settings, and WebUI forms to share behavior.

The config system should support declarations near the code that consumes a
field, then aggregate those declarations into a global JSON/TOML/Markdown schema.
Required fields must allow predicates such as "required when vision is enabled",
not only a boolean. Resolution must be layered and explainable across compiled
defaults, global config, profiles, host/node config, pool policy, per-model
overrides, environment variables, CLI flags, and request overrides. See
`docs/plans/2026-06-20-config-schema-registry.md`.

### Eval

Eval status should be observable as runs with IDs, batteries, suite/tier, row
status, artifacts, comparison baseline, admission verdict, and failure logs. The
UI should link to evidence artifacts rather than reimplement eval interpretation.

### Training Feedback

Training jobs need first-class progress events:

- label/capture phase progress and ETA;
- per-step/per-epoch loss, eval-Spearman or task metric, and learning rate;
- throughput, wall-clock time, and overall ETA;
- checkpoint path, last checkpoint age, resume source, and SIGINT checkpoint
  state;
- export/admission handoff status when a trainer emits a daemon-servable `.hfq`.

The plain terminal log remains the fallback for CI and non-TTY runs. TUI and WebUI
should subscribe to structured events when available.

## Implementation Sequence

1. Define shared operator state structs in a library crate or existing serving
   core boundary.
2. Define the config schema registry and layered override resolver, including
   host/node/pool scopes for future local-network daemon pools.
3. Add read-only daemon/server operator endpoints for status, models, config,
   eval runs, and training jobs.
4. Refactor `hipfire-tui` to consume shared APIs instead of local duplicate
   config/registry/status parsing.
5. Add a WebUI as a thin client of the same operator API. Prefer static assets
   served by `hipfire serve` or an explicitly separate dev server, but keep the
   daemon API as the authority.
6. Add typed actions incrementally: config write, model load/unload, eval
   launch/cancel, training checkpoint/cancel/resume.

## Open Questions

- Whether the operator watch stream should be SSE, WebSocket, JSONL, or all three
  behind the same event vocabulary.
- Whether WebUI assets belong in `hipfire-server` by default or behind a feature
  flag.
- How much local process control is appropriate when the daemon is supervised by
  systemd/launchd/container infrastructure.
