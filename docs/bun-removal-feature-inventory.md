# Bun Removal Feature Inventory

This document records the Bun/TypeScript feature surface removed from the active
tree so the missing pieces can be rebuilt intentionally in the Rust Clap CLI or a
future TUI.

## Removed Runtime Surfaces

### `hipfire serve` (Bun HTTP server)

Removed source: `cli/index.ts::serve`.

Features to recreate or explicitly retire:

- Positional serve parsing: `hipfire serve [host] [port]`, `host:port`, IPv6
  `[host]:port`, and model-tag guardrails.
- Detached/background mode: `-d`, `--detach`, `--background`.
- Detached PID/log handling: `~/.hipfire/serve.pid`, `~/.hipfire/serve.log`,
  `hipfire stop`, readiness polling against `/health`.
- `--tp N` flag that sets `HIPFIRE_TP` for expert-parallel daemon loads.
- Startup pre-warm using `HIPFIRE_MODEL` or configured `default_model`.
- Daemon restart/recovery after pre-warm failure or interrupted generation.
- Single-request lock around daemon I/O.
- Idle model unload driven by `idle_timeout`.
- HTTP endpoints: `/health`, `/v1/models`, `/v1/chat/completions`.
- OpenAI chat compatibility details in the Bun path:
  - streaming and non-streaming responses;
  - tool-call parsing/repair and OpenAI tool-call chunk shape;
  - `<think>` stripping and optional preservation;
  - `stream_options.include_usage`;
  - image request support via base64 data URL;
  - sampling, stop, system, priority, reasoning, and chat-template knobs;
  - timings/usage fields, including cache read/write counters.

Rust status: the native Axum server already owns `/health`,
`/v1/models`, `/v1/chat/completions`, `/v1/responses`, `/v1/files`, and
`/v1/batches`. Missing Bun ergonomics should be added to
`crates/hipfire-cli`/`crates/hipfire-server` instead of resurrecting TS.

### `hipfire stop`

Removed source: `cli/index.ts` command switch.

Features to recreate:

- Read detached serve PID from `~/.hipfire/serve.pid`.
- Gracefully SIGTERM the serve process, wait briefly, then SIGKILL if needed.
- Remove stale PID file.

Rust status: not present in the current Clap surface.

### `hipfire run`

Removed source: `cli/index.ts::run` and HTTP fallback helpers.

Features to recreate:

- Model tag/path resolution against registry, local catalog, aliases, and direct
  `.hfq` paths.
- Auto-use of a running local serve when possible.
- One-shot daemon generation fallback.
- Flags: `--temp`, `--top-p`, `--repeat-penalty`, `--max-tokens`, `--image`,
  `--system`.
- Default prompt handling and image prompt default.
- Visible thinking stripping in terminal output.

Rust status: `crates/hipfire-cli` has a `run` command, but parity with every Bun
flag and fallback behavior should be audited before calling it complete.

### `hipfire chat`

Removed sources: `cli/chat.ts`, `cli/chat_pure.ts`.

Features to recreate:

- Interactive terminal chat UI.
- ANSI/OSC stripping and markdown/code-fence rendering.
- Paste sanitization and bracketed-paste handling.
- Token/window estimation and context trimming.
- Rolling tok/s display.
- Multi-turn message history and optional color disabling.

Rust status: not present in the current Clap surface.

### `hipfire pull`

Removed source: `cli/index.ts::pull`.

Features to recreate:

- Registry-backed model downloads from Hugging Face.
- Auth header support via HF token environment.
- Atomic partial-download handling.
- Local model catalog refresh.
- User-facing remote model list with download status.

Rust status: not present in the current Clap surface.

### `hipfire list`

Removed source: `cli/index.ts::list`.

Features to recreate or verify:

- Local `.hfq` discovery.
- Registry tag mapping.
- User alias display from `~/.hipfire/models.json`.
- Optional remote registry listing.

Rust status: `crates/hipfire-cli` has `list`, but Bun's alias and remote-display
behavior should be audited.

### `hipfire ps`

Removed source: `cli/index.ts::ps`.

Features to recreate:

- Process scan for daemon, serve, quantize, and HF uploads.
- RSS/elapsed-time display.
- Default serve port status and detached PID reporting.

Rust status: not present in the current Clap surface.

### `hipfire profile`

Removed source: `cli/index.ts::profile`.

Features to recreate or verify:

- Wrapper around `hipfire-host-profile`.
- Optional model load trigger.
- `--json` output mode.
- `--kernel` filter.

Rust status: `crates/hipfire-cli` forwards `host-profile`; compare flag parity.

### `hipfire update`

Removed source: `cli/index.ts::update`.

Features to recreate:

- Dependency probing for `git`, `cargo`, `hipcc`, and PATH augmentation.
- Guardrails around non-master branches, dirty worktrees, and local commits.
- Fetch/reset to upstream master.
- Rebuild daemon/runtime/eval/quantizer tools.
- Copy installed binaries into `~/.hipfire/bin`.
- Kernel cache cleanup and precompile.

Rust status: not present in the current Clap surface. Any replacement must use
the active branch policy instead of the old master-only rule.

### `hipfire diag`

Removed source: `cli/index.ts::diag`.

Features to recreate:

- Platform detection for Linux, WSL2, and Windows.
- PCI/DRM/KFD/amdgpu/ROCm probing.
- Daemon binary discovery.
- Local model listing.
- Kernel cache inspection.
- Live daemon `diag` probe.
- Actionable install/permission guidance.
- Config-drift display.

Rust status: not present in the current Clap surface.

### `hipfire bench`

Removed source: `cli/index.ts::bench`.

Features to recreate:

- Multi-run decode and prefill benchmark harness.
- Optional RDNA2 experimental kernel variant sweep.
- Summary statistics for tok/s and timings.
- Timeout handling.

Rust status: not present in the current Clap surface.

### `hipfire rm`

Removed source: `cli/index.ts::rm`.

Features to recreate:

- Resolve model tag/alias/direct file.
- Delete local model file.
- Helpful "model not found" guidance.

Rust status: not present in the current Clap surface.

### `hipfire quantize`

Removed source: `cli/index.ts::quantize`.

Features to recreate:

- Frontend wrapper around `hipfire-quantize`.
- HF model download into local cache for safetensors sources.
- GGUF vs safetensors default format selection.
- Format aliases: `mq4`, `mq6`, `q8`, `q8f16`, `hf4`, `hf6`, `hfq4`,
  `hfq4g256`, `hfq6`, `hfq6g256`.
- Multi-format output, `--both`, `--output`, `--output-dir`, `--stem`.
- HF upload and create-repo flow.
- Local install and alias registration.

Rust status: core quantizer exists as `hipfire-quantize`; this user-facing
orchestration wrapper is removed.

### `hipfire sidecar-gen`

Removed source: `cli/index.ts::sidecar-gen`.

Features to recreate:

- Resolve model tag/path.
- Default sidecar path next to the model.
- Wrapper around `triattn_validate`.
- Flags: `--corpus`, `--max-tokens`, `--chunk-len`, `--gpu-calib`,
  `--cpu-calib`, `--output`, `--skip-validation`.
- Cargo fallback build of `triattn_validate` if no installed binary exists.

Rust status: not present in the current Clap surface.

### `hipfire config`

Removed source: `cli/index.ts::config` and config TUI helpers.

Features to recreate:

- Global config TUI.
- Per-model config picker/TUI.
- Scripted `list`, `get`, `set`, `reset`.
- Per-model override storage and migration into `~/.hipfire/models.json`.
- CASK profile bundles: off, triattn, cask-low, cask-balanced,
  cask-aggressive.
- Field validation and help text for all Bun-era config keys.
- Model-aware restrictions for A3B CASK behavior.

Rust status: not present in the current Clap surface.

## Removed Pure TypeScript Helper Modules

These modules were testable policy/control-plane helpers. Recreate in Rust when
their behavior is still required:

- `batch_api.ts`: batch JSONL validation, Responses batch normalization, batch
  error/output artifact builders.
- `dummy_model.ts`: dummy model tag/path helpers and load-message construction.
- `eval.ts`: human-friendly `hipfire eval` argument normalization and local
  speed-baseline writing.
- `generate_batch_prefill_protocol.ts`: generate-batch-prefill capability probe
  interpretation and dispatch status helpers.
- `host_profile.ts`: host-profile binary discovery and command construction.
- `model_worker_routing.ts`: resident/serving worker selection policy.
- `prefill_batch_health.ts`: prefill and batch health payload construction.
- `resource_lock.ts`: filesystem lease naming, CPU core parsing, and serve lock
  planning.
- `scheduler_policy.ts`: scheduler priority parsing, policy thresholds, and
  opportunistic dispatch rules.
- `server_prefill_batch.ts`: server prefill batching policy and session creation.
- `server_prefill_request_path.ts`: request-path queue decision.
- `session_state.ts`: worker-key normalization and prefill session compatibility.
- `state_cache.ts`: prefix checkpoint manifests, attachability, compatibility,
  and eviction selection.
- `worker_scheduler.ts`: prefill/decode worker scheduling.

## Removed Bun Test Coverage

The deleted `cli/*.test.ts` files covered the TypeScript helpers above plus
tool-call parsing and TUI-pure rendering behavior. Before rebuilding any of
those features, port the corresponding assertions to Rust unit tests in the
owning crate.

## Removed Nix Support

Removed files:

- `flake.nix`
- `nix/package.nix`
- `nix/module.nix`
- `nix/dev-shell.nix`
- `nix/kernels.nix`

Feature surface removed:

- Flake package build.
- Nix development shell.
- NixOS service module.
- Nix kernel precompile derivation.
- Overlay exports.

No replacement is planned in this tree.
