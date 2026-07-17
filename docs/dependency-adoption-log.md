# Dependency adoption log

This log records dependency and developer-tool changes that materially affect
Hipfire's build, runtime, or maintenance surface. Add entries when a dependency
is introduced, removed, centralized, or given a new production use.

## 2026-07-16 — beta dependency refresh

- Updated the Bun CI/runtime pin to 1.3.14, `@types/bun` to 1.3.14, and
  TypeScript to 7.0.2 in `cli/`. This keeps local and no-GPU CI behavior aligned
  and adopts TypeScript's current nullability checks.
- Updated `crossterm` to 0.29, `ratatui` to 0.30, and `ureq` to 3 in
  `hipfire-tui`. The associated source changes are API migrations, not feature
  additions.
- Updated `md5` to 0.8 in `hipfire-detect`.
- Updated `libloading` to 0.9 in `hip-bridge`, `hsa-bridge`, `redline`,
  `redline-dispatch`, and `redline-rocr`. All direct consumers share the
  workspace version so the beta-only Redline crates cannot drift from the
  bridge crates.

Implemented by commit `77dca1da5`.

## 2026-07-16 — support crates

- Added `clap` to `hipfire-quantize` and replaced the main quantizer's manual
  argument parsing. The derive-based schema makes validation and help text
  consistent while reducing parsing code.
- Added `safetensors` to `hipfire-quantize` and `hipfire-runtime`. Quantizer
  tools now share one bounds-checked safetensors reader instead of maintaining
  several handwritten header parsers.
- Added `half` to `hipfire-quantize` and `hipfire-runtime` for standard F16 and
  BF16 decoding. The quantizer retains its custom byte-exact F32-to-F16 encoder
  where the crate's rounding behavior would change emitted model bytes.
- Added `tracing` to `hipfire-runtime` and `tracing-subscriber` to the daemon
  example's development dependencies. Runtime libraries emit structured events;
  the daemon owns subscriber/filter initialization so stdout remains reserved
  for its JSON protocol.
- Enabled `cargo-deny` in no-GPU CI for license, source, duplicate-version, and
  advisory visibility. Advisory publication is informational; license/source
  policy remains blocking.

Implemented by commit `b15e6f884`.

## 2026-07-16 — property testing and version unification

- Added `proptest` 1.11 as a dev-only workspace dependency for
  `redline-dispatch` and `redline-rocr`. Default features are disabled and only
  `std` is enabled, avoiding the fork, timeout, tempfile, and bit-set dependency
  branches.
- Property tests cover device-region overlap/intersection symmetry, resource
  non-aliasing, kernarg overlap rejection, deterministic ABI hashing, and the
  requirement that arbitrary/truncated ELF input never panics the ROCr metadata
  parser.
- Extended `proptest` into `CompiledPlan` and replay bindings: generated DAGs
  preserve dependency order and deterministic fingerprints, semantic changes
  perturb fingerprints, unordered RAW/WAR/WAW overlaps are rejected until
  ordered, recorder-owned IDs cannot cross recorder boundaries, cycle-forming
  back edges are rejected without corrupting the graph, and resource/scalar
  binding sizes enforce their exact acceptance boundaries. These cases live in
  `redline-dispatch` only and remain excluded from release artifacts.
- Centralized `libloading`, `half`, and `safetensors` in
  `[workspace.dependencies]` because each has multiple direct consumers.

## Deferred after inspection

- `lexopt`: not adopted; maintained command-line tools should converge on
  `clap` instead of carrying two parser conventions.
- `bytemuck`/`zerocopy`: not adopted for Redline. Pointer-bearing AQL ABI
  structures retain explicit layout assertions and audited byte publication.
- `object`: deferred until ELF format support expands or property tests expose
  a maintenance problem in the small current parser.
- `approx`: deferred to a future CPU numerical-test cleanup; explicit GPU error
  metrics remain the correctness contract.

## Future beta follow-ups

- Add a `QuantizeArgs::try_parse_from` regression using the live argument shape
  from `scripts/mi300x_v3_matrix.sh` so script/CLI drift fails in unit tests.
- Add a generated safetensors fixture that checks tensor names, shapes, absolute
  data offsets, and emitted HFQ byte parity through the shared reader.
- Add a localhost HTTP/SSE test for the `ureq` 3 TUI migration, including
  streamed deltas, HTTP error bodies, and timeout behavior.
- Run one lightweight live serve smoke with structured logging enabled to prove
  tracing remains on stderr and cannot contaminate daemon JSON on stdout. A
  full GPU architecture matrix is not required for this dependency-only work.
- At the end of the beta run, return to the README update and the maintainer's
  additional beta ideas before promoting or merging the staging branch.

## Validation recorded for this adoption

- `bun install --frozen-lockfile`, `bun run typecheck`, and all 288 Bun tests
  pass with Bun 1.3.14 and TypeScript 7.
- `cargo check` passes for every crate changed by the broad refresh, including
  the five `libloading` consumers and `hipfire-tui`.
- `cargo check --locked -p hipfire-runtime --lib` and the shipped daemon
  example both pass with `safetensors`, `half`, and `tracing` enabled.
- All three `hipfire-quantize` library tests pass, including exhaustive
  non-NaN F16 decode coverage and the byte-exact encoder regression.
- All 80 `redline-dispatch` and `redline-rocr` library tests pass with the new
  property cases enabled.
- The complete `cargo test --lib --workspace --locked` no-GPU CI suite passes.
- `cargo audit` reports no vulnerabilities in the 277-package resolved graph.
- `cargo-deny` is enabled in GitHub CI. It was not run locally because the
  `cargo-deny` executable is not installed on this host.
