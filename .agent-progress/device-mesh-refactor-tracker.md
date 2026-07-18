# Device-Mesh Refactor Tracker

## Authority Rule

This file is the authoritative source of current status for the device-mesh refactor. [PR #527](https://github.com/Kaden-Schutt/hipfire/pull/527) mirrors the active task IDs for contributor visibility. If the PR, a handover, a task report, a design note, or any other status document disagrees with this tracker, this tracker wins. Historical documents remain evidence, not status authorities.

## Task ID Migration

This immutable table records the sole bootstrap correction to IDs published before implementation began. The old IDs are retired aliases: they must never be reused for another task, and all current or future references must use the corrected IDs.

| Initial published ID | Meaning in `7115135e` and the initial PR mirror | Corrected ID | Correction date | Correction commit |
|---|---|---|---|---|
| `PAR-003` | Optional TP x EP composition scope decision | `COMP-001` | 2026-07-12 | `754e68bc` |
| `COMP-001` | Final validation and merge gate | `DOC-002` | 2026-07-12 | `754e68bc` |

The alias rows are historical provenance only. They do not define active dependencies, and they must not be copied into the PR checklist when it is next synchronized. Because the correction was simultaneous, `COMP-001` has one explicit bootstrap collision: its retired `final validation and merge gate` meaning must never be reused, while the current `COMP-001` ID refers only to optional TP x EP composition. `DOC-002` is the only current final-gate ID.

## Completion Definition

The refactor is complete only when every active task below is `complete`, including all physical-hardware gates. Completion means:

- RCCL expert-parallel serving is validated for DeepSeek4 and MiniMax on distinct physical GPUs.
- Dense PP, Qwen35 PP, and TP teardown are validated on distinct physical GPUs with correct placement, transfer, output, and bounded post-unload VRAM.
- Request metadata, reset ownership, parser finalization, and session-state ownership are total and regression-tested.
- Ordinary AR, speculative/MTP, VL, Step/manifest, and required PP/TP/EP model-family paths use the shared architecture and mesh abstractions without legacy duplicate orchestration.
- Every supported model family has an explicit tested PP/TP/EP support decision; TP x EP is either implemented for a concrete requirement or explicitly remains out of scope.
- Stale status documentation points here, the full required validation matrix passes, evidence is recorded, and PR #527 is synchronized and merge-ready.

Emulation can prove structure and byte parity, but it cannot satisfy an acceptance criterion that explicitly requires distinct physical GPUs.

## Current Status

**Foundation implemented; refactor incomplete.** `COR-004` and `COR-003` are complete; no task is currently marked `in progress`. The mesh, manifest, Step execution, generic AR dispatch, model-parallel ownership, god-struct foundation, and COR-003 terminal lifecycle work are substantial and tested. Remaining architecture migrations and the separate physical PP/TP/EP topology tasks tracked below remain open. No open item is implicitly waived by earlier emulated validation.

Contributor validation on two gfx1201 R9700s (2026-07-14, commit `4df03537`) confirmed balanced Qwen35 PP allocation and peer access, but did not close either physical PP gate: dense LLaMA forward hit an unclassified illegal access and Qwen35 PP=2 diverged at token 58/100. The evidence and bounded follow-up are recorded under HW-003 and HW-004; neither changes the current execution queue or relaxes exact-parity requirements.

## Execution Priority

This is the implementation queue. The dependency graph below remains the
authoritative constraint; a task is marked `in progress` only when work begins.

1. `COR-002` — implement the total reset contract now that `COR-004` resolved the
   ownership boundary.
2. `STEP-001` — adopt Step/manifest for DeltaNet with single-device parity.
3. `PAR-001` — publish and enforce the model-family PP/TP/EP support matrix.
4. `COMP-001` — decide and enforce the TP x EP scope boundary.
5. `COR-005` — make generic LLaMA/Qwen3 spec-target loading transactional.
6. `COR-006` — align eviction physical-cap metadata with KV allocation.

After those ownership, execution, and support decisions, schedule their
dependent work by the dependency graph. Hardware tasks remain blocked until
the required distinct-GPU topology is available; emulation does not advance
them toward completion.

## Completed Foundation Evidence

- Hardware and mesh foundation: `ff709bdc` (`hipfire-hardware` extraction), `0b95b89c` (`DeviceMesh`), `5f4b581c` (`resolve_mesh`), and `e66d6f94` (PP stage/band helpers).
- Manifest and placement foundation: `a6a0acb9` (manifest types), `41b63cdb` (placement), `69c61c05` (collective schedule), plus store-backed llama validation recorded in `.agent-progress/device-mesh-status.md`.
- Model-parallel ownership: `3e99918c` (owning enums), `8c3d7f85` (TP), `a4211e3c` (dense PP), `a4583dbc` (EP), and `0fe02058` (Qwen35 arch-resident PP).
- Session/meta collapse: `a7082ee9`, `4b1a2fe8`, `e16e7c01`, `8be7bf63`, and `9c57148d` established `SessionState`, `PersistState`, reset routing, and `ModelMeta` readers.
- Generic generation foundation: the live `ar_generate` path and StreamParser/ArchDispatch folds are documented with parity evidence in `.agent-memory/notes/daemon-god-struct-archdispatch-design.md` and `.superpowers/sdd/progress.md`.
- Teardown and peer-order fixes: `eafd8663` and `17fc1c4c` closed the known emulated TP/PP unload leak and corrected peer-access ordering; physical teardown remains tracked below.
- Scope and authority design: `a1ad8a46` defines this tracker, its PR synchronization rule, and the non-goal of treating emulation as production hardware proof.

## Active Tasks

### HW-001 DeepSeek4 RCCL EP Validation

- **Status:** blocked
- **Dependencies:** STEP-002
- **Goal:** Validate the production RCCL expert-parallel path for DeepSeek4 without the peer-all-reduce fallback.
- **Acceptance criteria:** Pin the DeepSeek4 model artifact SHA-256 and prompt-file MD5 before testing; capture the existing peer-all-reduce `ep_decode_parity` committed-token hash as the oracle; on at least two distinct GPUs, the RCCL run must produce the identical committed-token hash, pass the same multi-turn assertions, complete four load/generate/reset/unload cycles without hangs or invalid access, and return each GPU to within 64 MiB of its post-first-unload baseline with no monotonic growth across cycles 2-4.
- **Validation:** Run `ep_decode_parity` and its multi-turn serving fixture first with `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` to capture the oracle, then with RCCL enabled and `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` unset; record artifact/prompt digests, topology, GPU architecture, ROCm/RCCL versions, exact commands, token hashes, and per-cycle VRAM.
- **Hardware:** At least two distinct RCCL-capable AMD GPUs with enough aggregate VRAM for the pinned DeepSeek4 fixture.
- **Evidence:** Pending

### HW-002 MiniMax RCCL EP Validation

- **Status:** blocked
- **Dependencies:** STEP-002
- **Goal:** Validate the production RCCL expert-parallel path for MiniMax without the peer-all-reduce fallback.
- **Acceptance criteria:** Pin the MiniMax model artifact SHA-256 and deterministic prompt-file MD5 before testing; capture the emulated/peer EP committed-token hashes for cold prefill, LCP reuse, and the Tokyo-then-Germany multi-turn fixture as oracles; RCCL on at least two distinct GPUs must match every hash, complete four load/generate/unload cycles, and return each GPU to within 64 MiB of its post-first-unload baseline with no monotonic growth across cycles 2-4.
- **Validation:** Run the existing MiniMax EP deterministic capital/code, LCP, and Tokyo-then-Germany multi-turn fixtures with the peer path to capture oracles, then repeat with RCCL and the peer fallback disabled; record digests, topology, versions, commands, hashes, and per-cycle VRAM.
- **Hardware:** At least two distinct RCCL-capable AMD GPUs with enough aggregate VRAM for the pinned MiniMax fixture.
- **Evidence:** Pending

### HW-003 Physical Dense PP Validation

- **Status:** blocked
- **Dependencies:** None
- **Goal:** Prove dense pipeline placement and boundary transfer on physically separate devices.
- **Acceptance criteria:** Using `qwen3-0.6b-llama.mq4` in `llama_store_pp`, PP=2 must preserve the established single-device oracle of `max |delta| = 0` across logits; the 28 layers must remain banded 14/14 with embed on stage 0 and output norm/lm_head on stage 1; allocation inspection must show no stage-owned weight page on the wrong GPU; four load/forward/unload cycles must return each GPU to within 64 MiB of its post-first-unload baseline with no monotonic growth across cycles 2-4.
- **Validation:** Before treating the reported gfx1201 failure as a device-mesh defect, classify it with the canonical `qwen3-0.6b-llama.mq4` artifact on this branch and upstream/master across PP=1, emulated PP=2, and a physical PP=2-capable harness. Record model and binary SHA-256 values, topology, ROCm/driver versions, and the first failing HIP launch after synchronization (kernel, stage/device, launch dimensions, tensor shape/dtype, and HIP error); a later logits-download error is insufficient. `llama_store_pp` alone cannot prove physical execution when it forces emulation. Then run the acceptance validation on two distinct devices: capture the 311-tensor placement inventory, zero logit delta, an explicit boundary-copy trace, per-device peak VRAM, and per-cycle post-unload VRAM; run the dense PP serving smoke with a pinned prompt MD5 and compare its committed-token hash to single-device generation.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs; a homogeneous pair is preferred for the first proof.
- **Evidence:** External report from `taniguchi-taku-softm`, 2026-07-14, at `4df035373669369484797abdd274f3f710c4c061`: two gfx1201 R9700s, ROCm 7.2.4, RCCL 2.27.7.70204, bidirectional P2P. Noncanonical `qwen3-0.6b.hf4` SHA-256 `7760b19dfb940f8b33078eb524602b4f2b5e6825c6e10c466e6e99bcfc133838` produced correct 155/156 emulated placement but an illegal-memory-access surfaced at logits download. This is preliminary classification evidence only: the canonical artifact was unavailable and no first failing launch or physical dense-PP forward was captured.

### HW-004 Physical Qwen35 PP Validation

- **Status:** blocked
- **Dependencies:** GEN-001
- **Goal:** Prove Qwen35 arch-resident pipeline execution and teardown on physically separate devices.
- **Acceptance criteria:** Before the physical run, pin the Qwen35 model SHA-256 and prompt-file MD5 and capture single-device committed-token hashes for cold generation and a two-turn recurrent-reset fixture; PP=2 on distinct GPUs must match both hashes, place every hybrid attention/recurrent weight and state allocation on its assigned stage, use the peer boundary path, and return each GPU to within 64 MiB of its post-first-unload baseline after four cycles with no monotonic growth across cycles 2-4.
- **Validation:** After GEN-001, first reproduce on physical hardware with deterministic mode explicitly enabled and the pinned model, binary, and prompt hashes recorded. Compare PP=1, emulated PP=2, and physical PP=2 for cold and two-turn reset fixtures. If parity differs, capture the first numerical difference: top-k logits and margins, boundary-residual checksum, recurrent/conv-state checksums, copy byte counts and source/destination devices, and stream/event ordering. Then run Qwen35 single-device oracle capture followed by PP=2 deterministic cold, two-turn reset, placement, explicit boundary-transfer, and four-cycle load/unload tests; record artifact/prompt digests, topology, exact commands, hashes, allocation inventory, and VRAM traces.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for the pinned Qwen35 fixture.
- **Evidence:** External report from `taniguchi-taku-softm`, 2026-07-14, at `4df035373669369484797abdd274f3f710c4c061`: physical PP=2 on two gfx1201 R9700s for `qwen3.5-9b.mq4` SHA-256 `ba83acf5bfd5d4e334b0afc26d779734e31623bb7f74e807c3581dfecb3128ad` allocated 2.638 GiB of weights, 0.134 GiB of KV, and 0.006 GiB of DeltaNet state per card; peer access was verified. PP=1 and PP=2 greedy output matched 58/100 tokens and first diverged at index 58. This is a hard parity failure, not accepted numerical variance; the run did not record a prompt MD5 or explicitly force deterministic mode, so it does not localize the cause or satisfy HW-004.

### HW-005 Physical TP Teardown Validation

- **Status:** blocked
- **Dependencies:** None
- **Goal:** Confirm TP teardown frees allocations, pools, streams, and communicator resources on real multi-GPU hardware.
- **Acceptance criteria:** Pin the TP-capable model SHA-256 and prompt-file MD5 and capture its single-device committed-token hash before testing; at least four TP=2 load/generate/unload cycles on distinct GPUs must reproduce that hash, leave no live model stream or communicator after unload, return each GPU to within 64 MiB of its post-first-unload baseline, and show no monotonic VRAM growth across cycles 2-4.
- **Validation:** Capture the single-device oracle, then run four TP=2 cycles while recording exact commands, hashes, per-device VRAM before load and after unload, and stream/communicator diagnostics; report the baseline and maximum absolute drift.
- **Hardware:** At least two supported AMD GPUs usable by the production TP path.
- **Evidence:** Pending

### COR-001 Wire `mtp_k` Metadata

- **Status:** complete
- **Dependencies:** None
- **Goal:** Make the configured/load-message `mtp_k` value the deliberate source used by generation, or remove the unsupported knob rather than silently ignoring it.
- **Acceptance criteria:** `ModelMeta` receives the configured value exactly once; native/spec generation reads that value with documented environment precedence; no stale flat field or self-assignment remains; CLI metadata exposes the setting; tests cover default, configured, and environment-override behavior.
- **Validation:** Run targeted Rust metadata/generation tests, `cli/config_meta.test.ts`, and searches proving generation no longer bypasses `meta.mtp_k`.
- **Hardware:** None
- **Completion blockers:** None.
- **Evidence:** `bun test cli/mtp_k_config.test.ts` (10 passed); `bun test cli/config_meta.test.ts` (1 passed); `nix develop --command bash -lc 'cargo test -p hipfire-loader --lib --locked && cargo test -p hipfire-runtime --example daemon mtp_k_tests --locked'` (13 loader and 15 daemon tests passed); `nix develop --command cargo test --workspace --locked` (passed); `nix develop --command ./scripts/coherence-gate-dflash.sh` (no hard errors; `/tmp/coherence-dflash-20260713-105546.md`); and `nix develop --command bash scripts/coherence-gate-deepseek4-mtp.sh --full` (all six DeepSeek MTP cases passed at K=2 and K=3; `/tmp/coherence-deepseek4-mtp-20260713-113736.md`). Generation reads `ModelMeta::mtp_k`; direct environment values are resolved only during model load.

### COR-002 Make Reset Total

- **Status:** in-progress
- **Dependencies:** COR-004
- **Goal:** Define and implement the single authoritative reset contract: request-owned state is cleared by `SessionState`, architecture-owned state is reset through exhaustive dispatch, and speculative state is reset by the same entry point.
- **Acceptance criteria:** One reset entry point and ownership contract cover abort, overflow, reset command, normal completion, VL, single, PP, TP, EP, speculative, recurrent, and conv state; adding a model-state variant cannot silently omit its reset arm. Integration tasks do not redefine reset semantics: they only implement their architecture adapter and prove conformance to COR-002.
- **Validation:** Run reset-contract unit tests, exhaustiveness/ownership checks, `serve-multiturn-gate.sh`, architecture-specific multi-turn tests, and abort/overflow/reset-command regressions for single and mesh paths.
- **Hardware:** A supported AMD GPU; distinct GPUs are additionally required for integration proof, not for defining or implementing the reset contract.
- **Evidence:** Reset ownership and lifecycle coverage includes dense TP/PP cache-miss routing, MiniMax EP cache-miss routing, Cohere cold-prefill routing, DSpark retry-safe hidden-buffer freeing, VL cold-reset behavior, and the two-row DFlash prompt-cache miss regression. Qwen cache misses now reset before capacity validation; VL/dots.ocr dirty abort/error paths defer their terminal envelope until fallible reset completes, and reset failures poison/terminate the daemon rather than serving on unknown GPU state. On this workspace's gfx1151, the updated Cohere cold + one-token warm-prefix parity gate passed with retained evidence at `/tmp/hipfire-cor-002/cohere-OhH5hG/`, and the updated DFlash two-row gate passed with retained evidence at `/tmp/hipfire-cor-002/dflash-joTHnh/`. After those focused gates, serialized `./scripts/serve-multiturn-gate.sh` passed at `/tmp/serve-multiturn-20260717-220023.md`, and serialized `./scripts/coherence-gate-dflash.sh` reported no hard errors at `/tmp/coherence-dflash-20260717-220054.md`. The reports are retained for review; no separate physical distinct-device TP/PP/EP proof or full physical VL/dots.ocr multi-turn proof was available in this environment and neither is claimed. Focused CPU tests and the workspace suite passed.

### COR-003 Finalize Parser On Pending EOS

- **Status:** complete
- **Dependencies:** None
- **Goal:** Ensure EOS and request termination always finalize buffered parser output exactly once.
- **Acceptance criteria:** Terminal `StopQuarantine`/`EosFilter` and `StreamParser` finalization are idempotent; Cohere recovery, generic AR/spec normal-versus-discard policy, sealed Qwen speculative turn authority/cache/reset behavior, Qwen PP sealed-boundary/reset behavior, DeepSeek AR discard reset/cache zeroing, and native Qwen/DeepSeek/DSpark MTP in-flight cancellation all preserve the no-late-output and no-cross-turn-residue contract. Injected EOS remains pre-commit where required.
- **Validation:** `nix develop --command cargo test --workspace --locked` passed on 2026-07-16 (CPU; GPU tests ignored as applicable); `nix develop --command ./scripts/coherence-gate-dflash.sh` passed with no hard or soft warnings; `nix develop --command ./scripts/serve-multiturn-gate.sh` passed; and `git diff --check` passed.
- **Hardware:** None for unit tests; a supported AMD GPU for end-to-end parity gates.
- **Evidence:** Current implementation includes terminal stop/finalization, sealed speculative-turn ownership, architecture discard/reset paths, and native Qwen/DeepSeek/DSpark MTP cancellation with production-owned lifecycle tests. Fresh evidence on 2026-07-16: `nix develop --command cargo test --workspace --locked` passed (CPU; GPU tests ignored as applicable); `nix develop --command ./scripts/coherence-gate-dflash.sh` passed with no hard or soft warnings, report `/tmp/coherence-dflash-20260716-110721.md`; `nix develop --command ./scripts/serve-multiturn-gate.sh` passed, report `/tmp/serve-multiturn-20260716-110919.md`; and `git diff --check` passed. COR-003 is complete. Remaining architecture migrations and separate physical PP/TP/EP hardware tasks remain tracked independently.

### COR-004 Decide Eviction Ownership

- **Status:** complete
- **Dependencies:** None
- **Goal:** Decide and enforce whether eviction is resettable request state in `SessionState` or persistent/model-owned state.
- **Acceptance criteria:** The ownership decision is documented with lifecycle rationale; the field is moved or explicitly retained accordingly; reset, reuse, and speculative commit semantics follow that decision; tests prevent cross-request eviction bleed and accidental loss of intentionally persistent state.
- **Validation:** Run ownership/reset unit tests plus multi-turn and speculative eviction scenarios; inspect `LoadedModel` so no duplicate eviction authority remains.
- **Hardware:** None for ownership tests; a supported AMD GPU for end-to-end eviction behavior.
- **Evidence:** Implementation: `62050f7c`. Decision: `LoadedModel.eviction` owns the calibrated policy and
  reusable GPU scratch until unload; `KvCache::compact_offset`, physical cursor,
  target recurrent state, and the DFlash mirror are request state. Qwen35
  DFlash construction, sidecar loading, snapshots, and eviction scratch now
  roll back unpublished GPU allocations; failed speculative transitions drop
  their target guard and rejoin `model_reset_context`. Validation: `nix develop
  --command cargo test -p hipfire-runtime --lib --locked` (346 passed, 1
  ignored); `nix develop --command cargo test -p hipfire-loader --lib --locked`
  (14 passed, 2 ignored); `nix develop --command cargo test -p hipfire-runtime
  --example daemon --locked` (37 passed); Qwen35 lib tests (141 passed, 5
  ignored); release daemon build with `deltanet`; default and sidecar-enabled
  `serve-multiturn-gate.sh` passes (`/tmp/serve-multiturn-20260714-081913.md`,
  `/tmp/serve-multiturn-20260714-081935.md`); deterministic lifecycle pass
  (`/tmp/nix-shell.vsoRuA/qwen35-eviction-lifecycle.GBTT1t`, A=104 > 40,
  reset B token-identical to clean B). Fixtures SHA-256: target
  `70dcd063a493af20a519e3afd0f341910b97bfd1af76aba45fe4742aed14fd15`, draft
  `bd8c4f07ae80fe1385bf2606af9a7ba0daa18ca8daec50916f2a489054c44e70`, sidecar
  `d6cb8026841830cfeb82d2709453aa753f65b5596bfb9cc9c085c808fda6ad22`.

### COR-005 Transactional LLaMA Spec-Target Loading

- **Status:** ready
- **Dependencies:** None
- **Goal:** Make generic LLaMA/Qwen3 speculative-target loading and DFlash
  construction transactional so every fallible load path returns a normal error
  without orphaning target, draft, scratch, or verification GPU allocations.
- **Acceptance criteria:** Generic carrier loading retains ownership of a
  partially loaded LLaMA target until its generic DFlash scratch and target
  verification resources are fully published; all failure paths free every
  earlier allocation exactly once; success and unload preserve the existing
  explicit teardown contract; no global `Drop` for GPU buffers is introduced.
- **Validation:** Add deterministic fault injection for generic target load and
  generic DFlash scratch/verify allocation. For each injected failure, drain
  the pool and require exact VRAM baseline recovery; run generic DFlash
  success/unload and repeated load/generate/unload cycles on a supported GPU.
- **Hardware:** A supported AMD GPU with a generic LLaMA/Qwen3 target and
  compatible DFlash draft fixture.
- **Evidence:** Pending

### COR-006 Align Eviction Physical-Cap Allocation

- **Status:** ready
- **Dependencies:** None
- **Goal:** Make the physical capacity derived for TriAttention/CASK size the
  actual Qwen35 KV allocation rather than only eviction metadata and scratch.
- **Acceptance criteria:** With a sidecar, the Qwen35 KV cache allocation uses
  the resolved physical capacity; loading rejects impossible budget/beta/cap
  combinations before allocation; configured long context retains the intended
  bounded VRAM behavior; non-eviction loading remains byte-identical.
- **Validation:** Add loader and GPU allocation-inventory tests for plain
  TriAttention and CASK; record KV allocation bytes, physical cap, budget, and
  beta; run repeated long-context eviction and unload/reload cycles.
- **Hardware:** A supported AMD GPU with a Qwen35 target and TriAttention
  sidecar.
- **Evidence:** Pending

### GEN-001 Complete Qwen35 Arch-Resident PP

- **Status:** ready
- **Dependencies:** COR-002, STEP-001, STEP-003
- **Goal:** Complete Qwen35 PP through the arch-resident `ModelParallel::Pp(PipelineImpl::ArchResident)` path for hybrid attention and DeltaNet layers.
- **Acceptance criteria:** Load, prefill, decode, recurrent/conv state, sampling, and unload use the generic PP ownership and stage interfaces; the Qwen35 adapter implements the COR-002 reset contract without creating a second reset authority; no legacy `pp`/`pp_gpus` side channel or duplicate Qwen35 PP loop remains; emulated PP parity is byte- or token-identical before physical validation.
- **Validation:** Run Qwen35 single-versus-emulated-PP deterministic parity, COR-002 conformance and recurrent multi-turn/reset tests, placement assertions, and repeated unload tests; then hand off to HW-004.
- **Hardware:** One supported AMD GPU for emulated PP; physical closure is HW-004.
- **Evidence:** Pending

### GEN-002 Add DeepSeek4 Single-GPU Fallback

- **Status:** ready
- **Dependencies:** COR-002
- **Goal:** Provide an ordinary single-GPU DeepSeek4 generation path when EP is not selected or available.
- **Acceptance criteria:** DeepSeek4 selects a single-device ArchDispatch/AR path without constructing EP state; DSML grammar/parser behavior matches the EP path; its adapter implements and proves the COR-002 reset contract; deterministic output, tool calls, and unload are coherent; unsupported model sizes fail explicitly on insufficient VRAM.
- **Validation:** Run deterministic prose/code/tool-call and multi-turn parity against the accepted DeepSeek4 behavior, COR-002 reset conformance, load/unload, and low-VRAM failure tests.
- **Hardware:** One supported AMD GPU with enough VRAM for the selected DeepSeek4 fixture.
- **Evidence:** Pending

### SPEC-001 Unify AR And Speculative Orchestration

- **Status:** ready
- **Dependencies:** COR-001, COR-002, COR-003
- **Goal:** Share request framing, reset, prefill, parser, streaming, accounting, and finalization above AR and speculative strategies.
- **Acceptance criteria:** AR and speculative/MTP execution are strategies under one request lifecycle; accepted-token commit semantics remain strategy-specific; duplicate request orchestration is removed; Qwen35's RAII spec-target guard is represented safely; `ArchDispatch::as_spec_target` is either implemented with a fitting contract or deleted with all dead scaffolding and TODOs removed; strategy adapters conform to COR-002 rather than owning reset semantics.
- **Validation:** Run AR-versus-spec lifecycle tests, DFlash coherence, deterministic accepted-token accounting, parser finalization, COR-002 reset conformance, abort, and multi-turn tests; search for orphaned `as_spec_target` implementations and duplicate request loops.
- **Hardware:** A supported AMD GPU with paired target/draft fixtures for DFlash validation.
- **Evidence:** Pending

### SPEC-002 Native Qwen MTP

- **Status:** ready
- **Dependencies:** COR-001, SPEC-001
- **Goal:** Integrate native Qwen MTP as a first-class speculative strategy using model metadata and the shared lifecycle.
- **Acceptance criteria:** Native Qwen MTP loads only when compatible weights are present; uses configured `mtp_mode` and `mtp_k`; commits only accepted target tokens; falls back explicitly to AR when disabled or unavailable; its adapter implements the COR-002 contract for all MTP scratch/state; quality and performance reporting uses fixed fixtures.
- **Validation:** Run MTP-off/auto/on selection tests, deterministic acceptance/accounting tests, AR fallback, COR-002 reset conformance, unload loops, coherence gate, and fixed-prompt performance measurements with prompt and binary hashes.
- **Hardware:** A supported AMD GPU with a Qwen model containing native MTP weights.
- **Evidence:** Native Qwen MTP in-flight cancellation and production-owned lifecycle tests are implemented as part of COR-003. The 2026-07-16 workspace, DFlash coherence, and multi-turn serving evidence passed; the exact reports are recorded under COR-003. Broader SPEC-002 MTP selection, fixed-fixture quality/performance, and unload coverage remain task scope. Transactional target loading is not part of this completion claim and remains deferred to `SPEC-003`.

### SPEC-003 Transactional Qwen MTP Loading And Allocation Safety

- **Status:** deferred
- **Dependencies:** COR-001
- **Goal:** Make native Qwen MTP loading and per-request scratch allocation transactional, so malformed or incompatible heads and every fallible allocation path return a normal error without leaking GPU memory, panicking, or silently changing serving behavior.
- **Acceptance criteria:** Head preflight validates actual on-disk payload length, metadata, GQA geometry, vocab-map bounds, trunk/head compatibility, supported dense and MoE tensor layouts, and reports errors without panics; `mtp_mode=on` is rejected explicitly on unsupported Qwen load paths while `auto` remains AR-only; one native head has one GPU owner; all fallible steps after trunk/head/vision/CASK allocation and all MTP scratch allocations roll back every owned GPU tensor on error; no direct allocation relies on `Drop`; fixed failure-injection tests prove every staged resource is explicitly freed; MTP-off/auto/on policy and the 1..8 K range are consistent across CLI, TUI, loader, and documentation.
- **Validation:** Run CPU malformed-container, physical-truncation, GQA/vocab-map, head/trunk mismatch, dense/MoE preflight, and staged rollback tests; run MTP-off/auto/on tests for single, PP, and safetensors routes; run GPU fault-injection for head upload, CASK/vision post-head setup, and MTP scratch allocation while checking VRAM before/after; run repeated load/generate/unload cycles plus coherence and multi-turn reset tests on a fixed native-MTP fixture.
- **Hardware:** A supported AMD GPU with a native-MTP Qwen fixture; CPU tests cover preflight and staged-owner contracts.
- **Evidence:** Deferred by priority decision on 2026-07-13. Native Qwen MTP allocation safety predates the device-mesh work; COR-001 metadata wiring exposed it but did not introduce it. The task remains mandatory before final merge, but does not block higher-priority lifecycle, mesh, and architecture work.

### VL-001 Adopt Shared Lifecycle For Qwen35-VL

- **Status:** ready
- **Dependencies:** COR-002, COR-003
- **Goal:** Route Qwen35-VL post-prefill AR generation through the shared request lifecycle while preserving image-conditioned prefill.
- **Acceptance criteria:** This task is AR-only: vision preprocessing and multimodal prefill remain architecture-owned; post-prefill AR parsing, accounting, COR-002 reset conformance, and finalization use shared orchestration; image state cannot bleed across requests; text-only Qwen35 behavior is unchanged. VL target/draft or native-MTP speculation is out of scope until a model-specific quality fixture exists and must be added as a separate SPEC/VL follow-up depending on SPEC-001.
- **Validation:** Run image-plus-text deterministic fixtures, repeated different-image requests, text-only parity, COR-002 reset/abort conformance, and parser finalization; verify unsupported VL speculative modes are rejected explicitly rather than silently selected.
- **Hardware:** A supported AMD GPU with enough VRAM for the canonical Qwen35-VL fixture.
- **Evidence:** Pending

### VL-002 Adopt Shared Lifecycle For dots.ocr

- **Status:** ready
- **Dependencies:** COR-002, COR-003, SPEC-001
- **Goal:** Route dots.ocr post-image-prefill AR and existing model-free n-gram decoding through the shared request lifecycle without changing its custom framing or vision tower.
- **Acceptance criteria:** Image encoding and custom prompt framing remain dots.ocr-owned; post-prefill AR and existing n-gram selection, parser finalization, accounting, COR-002 reset conformance, and unload use shared orchestration; OCR output preserves the canonical fixture quality; image state is request-local. Target/draft and native-MTP VL speculation are out of scope and require a separate follow-up with a dots.ocr quality oracle.
- **Validation:** Run the canonical dots.ocr image fixture and F1 comparison in AR and existing n-gram modes, repeated-image isolation, text-decoder parity, COR-002 reset/abort conformance, and unload tests; verify other speculative modes are rejected explicitly.
- **Hardware:** A supported AMD GPU for the canonical dots.ocr fixture.
- **Evidence:** The request-state transition now records the preprocessed image
  sentinel and requires a cold reset when a prior image turn exists even if
  `seq_pos == 0`; daemon unit coverage includes image-A→image-B versus fresh-B
  state parity. The daemon-level
  `scripts/dots-ocr-image-reset-gate.sh` compares image-A→image-B output with
  fresh-B output when distinct image fixtures are supplied. The canonical
  dots.ocr F1 oracle and physical VL gate remain pending because this
  environment has only one dots.ocr image fixture and no distinct-device VL
  hardware; those gaps are intentionally not claimed closed.

### STEP-001 Adopt Step/Manifest For DeltaNet

- **Status:** ready
- **Dependencies:** None
- **Goal:** Represent Qwen35 DeltaNet weights, state, and forward execution through manifests and the Step spine.
- **Acceptance criteria:** The Qwen35 weight manifest covers layer-type-specific fused projections, norms, convolution, recurrent parameters, and dense/MoE variants; placement derives from policy; DeltaNet forward emits/executes Steps without a parallel bespoke layer loop; single-device output remains identical.
- **Validation:** Run manifest coverage/placement tests, source-to-store byte/dtype checks, Step-versus-legacy deterministic parity during migration, and Qwen35 coherence tests.
- **Hardware:** None for manifest tests; a supported AMD GPU for forward parity.
- **Evidence:** Pending

### STEP-002 Adopt Step/Manifest For MoE

- **Status:** ready
- **Dependencies:** PAR-001
- **Goal:** Fold routed-expert execution and its EP collectives into the common Step/manifest path.
- **Acceptance criteria:** Expert ownership, compact shard layout, routing, zero/dummy handling, and collective hints derive from the manifest/mesh; DeepSeek4, MiniMax, and Qwen35 MoE variants no longer require an independent executor; single and already-supported EP behavior preserve accepted output. This task adopts existing architecture forwards and does not add a new PP/TP/EP support cell.
- **Validation:** Run manifest shard tests, emulated EP deterministic parity for each covered family, expert-routing edge cases, transactional load failure, and EP coherence tests; physical RCCL closure remains HW-001/HW-002.
- **Hardware:** One supported AMD GPU for emulated EP; physical RCCL validation requires the hardware in HW-001 and HW-002.
- **Evidence:** Pending

### STEP-003 Adopt Step/Manifest For Recurrent And Conv State

- **Status:** ready
- **Dependencies:** COR-002, STEP-001
- **Goal:** Represent recurrent and convolution operations/state in Step execution with mesh-aware placement and reset.
- **Acceptance criteria:** Recurrent and conv state manifests encode layer ownership; Step execution handles prefill/decode state updates on the owning stage/device; boundary movement is explicit; the adapter implements the COR-002 reset contract; bespoke recurrent/conv forward loops are removed after parity.
- **Validation:** Run state placement tests, multi-token prefill/decode parity, COR-002 conformance, repeated multi-turn tests, PP emulation, and Qwen35 recurrent coherence tests.
- **Hardware:** A supported AMD GPU; physical PP closure is HW-004.
- **Evidence:** Pending

### STEP-004 Migrate Remaining Forward Paths

- **Status:** ready
- **Dependencies:** STEP-001, STEP-002, STEP-003, PAR-001
- **Goal:** Adopt Step/manifest for every remaining architecture forward path that already has a supported Single/PP/TP/EP cell, or record a justified non-decoder exception.
- **Acceptance criteria:** An inventory names every architecture and forward entry point; existing supported decoder paths use Step/manifest; encode-only or vision-only exceptions have explicit boundaries and ownership; obsolete executors and duplicate placement logic are deleted; each migration has parity evidence. This task does not create support for a new parallel axis; PAR-002 owns those implementations.
- **Validation:** Run an inventory search against architecture registration and forward symbols, per-family deterministic parity/coherence tests for already-supported cells, workspace tests, and checks that no unapproved bespoke decoder executor remains.
- **Hardware:** Supported AMD GPU coverage for each migrated, already-supported path; exact models/topologies follow PAR-001 decisions.
- **Evidence:** Pending

### PAR-001 Decide Model-Family PP/TP/EP Support

- **Status:** ready
- **Dependencies:** None
- **Goal:** Define the supported parallel axes and explicit refusal behavior for every registered model family.
- **Acceptance criteria:** A maintained matrix covers Single, PP, TP, and EP for every family; each cell is supported, planned with a task dependency, or explicitly unsupported with a technical reason; runtime selection and errors enforce the matrix; tests prevent accidental claims or silent fallback.
- **Validation:** Compare the matrix with architecture registration and load dispatch; run selection/refusal tests for every family and axis; verify docs and CLI report the same capabilities.
- **Hardware:** None for decisions and refusal tests; supported cells inherit their implementation task's hardware gates.
- **Evidence:** Pending

### PAR-002 Implement Required Additional PP/TP/EP Paths

- **Status:** blocked
- **Dependencies:** COR-002, PAR-001, STEP-004
- **Goal:** Implement only the new model-family PP/TP/EP support cells that PAR-001 marks required for this refactor.
- **Acceptance criteria:** Every newly required matrix cell has mesh-derived placement, reuses the architecture's STEP-004-adopted forward path, implements the COR-002 reset contract, covers lifecycle/unload and explicit unsupported combinations, and has deterministic parity. Architecture-forward migration itself remains STEP-004 scope.
- **Validation:** Run per-new-cell unit, emulated topology, coherence/parity, COR-002 conformance, and teardown tests; require physical topology evidence before marking any newly supported multi-GPU cell production-ready.
- **Hardware:** Determined by the new cells in PAR-001; physical multi-GPU closure is mandatory for production support.
- **Evidence:** Pending

### COMP-001 Gate Optional TP x EP Composition

- **Status:** ready
- **Dependencies:** None
- **Goal:** Make an unconditional scope decision for TP x EP composition in this refactor.
- **Acceptance criteria:** Record one decision: either TP x EP is out of scope and `TP>1 && EP>1` is explicitly rejected, or a concrete deployment requirement names the model, topology, owner, and measurable success target. In the latter case, create a new conditional COMP task for design/implementation/physical validation; COMP-001 itself completes when the decision and refusal-or-follow-up are recorded and never waits on implementation or hardware.
- **Validation:** Review the requirement record and support matrix; for the out-of-scope decision, run configuration/refusal tests; for the required decision, verify the new follow-up ID exists with dependencies and acceptance criteria.
- **Hardware:** None
- **Evidence:** Pending

### DOC-001 Consolidate Stale Status Documentation

- **Status:** complete
- **Dependencies:** None
- **Goal:** Prevent historical device-mesh reports from presenting stale plans as current status.
- **Acceptance criteria:** Complete: every stale handover/status/phase, follow-up, review, pivot, ArchDispatch, god-struct, and SDD progress document named in `docs/superpowers/specs/2026-07-12-device-mesh-tracking-design.md` carries an appropriate superseded or chronological-evidence notice linking here; historical evidence remains preserved; conclusively closed findings are labeled accurately.
- **Validation:** Complete in `7115135e`: all named documents were checked for authority links and stale current-status claims; the focused diff preserved forensic history while adding banners and status corrections.
- **Hardware:** None
- **Evidence:** `7115135e` (`docs(device-mesh): establish canonical completion tracker`); acceptance checks completed in the committed documentation diff.

### DOC-002 Final Validation And Merge Gate

- **Status:** blocked
- **Dependencies:** HW-001, HW-002, HW-003, HW-004, HW-005, COR-001, COR-002, COR-003, COR-004, COR-005, COR-006, GEN-001, GEN-002, SPEC-001, SPEC-002, SPEC-003, VL-001, VL-002, STEP-001, STEP-002, STEP-003, STEP-004, PAR-001, PAR-002, COMP-001, DOC-001
- **Goal:** Establish that the completed refactor is correct, production-honest, documented, and ready to merge.
- **Acceptance criteria:** Every listed dependency and every conditional follow-up created by COMP-001 is `complete` with evidence; every row in the Final Validation Matrix passes against its named fixture/oracle; HW-001 through HW-005 meet the 64 MiB/no-monotonic-growth thresholds; no stale active checklist conflicts with this tracker; PR #527 mirrors all IDs, required CI checks pass, and no blocking review finding remains.
- **Validation:** Execute and archive every row in the Final Validation Matrix, rerun tracker schema and documentation-link checks, inspect the final branch diff and PR checks/reviews, and attach the physical PP/TP/EP reports with artifact/prompt digests and per-cycle VRAM.
- **Hardware:** The union of hardware required by HW-001 through HW-005 and each supported model-family validation cell.
- **Evidence:** Pending

## Terminal lifecycle migration matrix

COR-003 establishes the terminal lifecycle contract; this matrix is mandatory
for every remaining and future architecture. COR-003 completion does not close
the architecture migrations listed here or the separate physical PP/TP/EP
tasks (`HW-001` through `HW-005`). A row is not production-ready
until its normal-finalization and discard/reset evidence exists for the named
driver. `VL-001` and `VL-002` are the downstream multimodal adopters. Generic
AR/spec adoption belongs to `SPEC-001`; native Qwen MTP uses `SPEC-002` (its
in-flight cancellation is implemented), while transactional Qwen target
loading remains deferred to `SPEC-003`.

| Architecture | Driver entry point / owner | Normal finalization | Abort/error discard/reset | Forced/injected EOS | Cache/cross-turn isolation | Required focused evidence | Unsupported/refused mode |
|---|---|---|---|---|---|---|---|
| DeepSeek4 | `ArchDispatch` DeepSeek4 AR/MTP adapters; `GEN-002`, `SPEC-001` | Bespoke AR and native MTP emit pending output exactly once on normal completion. | DeepSeek AR discard resets request state and zeros decode cache; native MTP cancellation restores guards/PBS and resets before terminal envelope. | Carrier/model EOS remains distinct from user stop and does not finalize early. | No discarded turn enters assistant cache; reset prevents decode-cache or turn residue. | Production-owned cancellation/reset tests, DeepSeek AR normal/discard tests, MTP coherence and real multi-turn hardware gate. | Qwen-style DFlash is not a DeepSeek4 mode; unsupported combinations must refuse explicitly. |
| MiniMax | MiniMax `ArchDispatch`/Step adapter; `STEP-002`, `STEP-004` | Shared lifecycle finalizes parser/emitter once after normal AR completion. | Abort/error drops pending output, resets request-owned state, and emits no late event. | Carrier EOS injection must remain non-terminal until the actual terminal outcome. | Multi-turn cache and parser state must be request-local and reset on discard. | MiniMax tool/stream, abort, reset, unload, and Tokyo-then-Germany multi-turn fixtures; EP evidence remains `HW-002`. | Unmigrated speculative/MTP or unsupported parallel cells refuse rather than silently fall back. |
| LFM2 | LFM2 architecture adapter and `ArchDispatch`; `PAR-001`, `STEP-004` | Shared generic AR finalization once the dense LFM2 loader/forward path is admitted. | Discard/reset must clear parser and request state even when the current model is refused. | No forced EOS may turn an unsupported or incomplete LFM2 path into a successful terminal response. | No cache reuse across refused, aborted, or reset requests. | Dense LFM2 support/refusal tests, parser terminal-policy tests, and a fixed multi-turn fixture after loader support lands. | Current dense LFM2 path is unsupported/refused; no speculative mode is admitted until an explicit support decision. |
| Qwen35 | Qwen35 `ArchDispatch`/AR, PP, and native MTP adapters; `GEN-001`, `SPEC-001`, `SPEC-002` | Generic AR/spec and Qwen PP use the sealed boundary and finalize exactly once; native MTP cancellation is implemented. | Sealed Qwen turns discard beyond the boundary, reset/cache-invalidate as required; MTP cancellation restores guards and resets before abort/error. | Injected EOS remains pre-commit; carrier framing and user stops remain separate. | Only the sealed turn may feed replay/fingerprint/cache; uncacheable cuts force reset/cold next turn. | Qwen AR/spec parser and sealed-turn tests, PP reset tests, native MTP lifecycle tests, coherence and physical multi-turn gate. | `SPEC-003` transactional target loading is deferred; unsupported Qwen MTP load paths must reject `on` and remain AR-only in `auto`/`off`. |
| Qwen35-VL | Qwen35-VL image prefill owner plus shared post-prefill AR lifecycle; `VL-001` (future speculation depends on `SPEC-001`) | Vision prefill remains architecture-owned; shared post-prefill AR finalizes once. | Abort/error discards parser and image/request state, then resets before the terminal envelope. | Image/carrier framing and injected EOS must not bypass the shared terminal policy. | Image state, parser state, and any cache are request-local; different-image turns cannot reuse discarded state. | Canonical image-plus-text, different-image isolation, abort/reset, text-only parity, and physical VL fixture. | VL target/draft/native-MTP speculation is refused until a model-specific quality fixture and follow-up exist. |
| dots.ocr | dots.ocr vision/prompt-framing owner plus shared post-prefill AR/n-gram lifecycle; `VL-002` (shared spec adoption through `SPEC-001`) | Custom image framing and prefill finish, then shared AR/n-gram output finalizes once. | Discard/reset clears parser and image state; no OCR/tool output follows abort/error. | Custom framing remains distinct from injected EOS and must not finalize twice. | Image state and OCR/cache state are request-local across repeated images and turns. | Canonical `dots_ocr_smoke_001_vllm.json`/demo-image F1, AR/n-gram parity, repeated-image isolation, abort/reset, unload, and physical gate. | Target/draft and native-MTP VL speculation is explicitly refused pending a dots.ocr quality oracle. |
| Future architecture onboarding | New architecture owner with `ArchDispatch`; adopt `GEN-*`, `SPEC-*`, `VL-*`, and `STEP-004` as applicable | Implement the shared normal-completion epilogue before claiming support. | Implement explicit discard/reset before any parser, cache, or GPU state is published. | Declare carrier/model EOS versus user stop and test injected EOS as pre-commit where applicable. | Name every cache/state owner; prove no discarded or prior-turn state crosses the boundary. | Add focused terminal lifecycle tests, deterministic parity, refusal tests, and required model/coherence/multi-turn hardware evidence before adding a support cell. | Every unsupported axis, speculative mode, or loader path must return a documented refusal; no silent fallback. |

## Final Validation Matrix

DOC-002 cannot complete from a generic “tests pass” statement. Its evidence must enumerate these rows with exact command, commit, fixture digest, result, and report path:

| Area | Required fixture or oracle | Pass condition |
|---|---|---|
| Workspace | `cargo build --workspace --features hipfire-runtime/deltanet` plus workspace tests with the same required feature set | Exit 0 and zero test failures. |
| DeepSeek4 EP | HW-001 pinned model/prompt; peer `ep_decode_parity` committed-token hash | RCCL hash identical to peer oracle; four-cycle VRAM threshold passes. |
| MiniMax EP | HW-002 pinned model/prompt; capital/code, LCP, and Tokyo-then-Germany peer hashes | Every RCCL hash identical to its oracle; four-cycle VRAM threshold passes. |
| Dense PP | `qwen3-0.6b-llama.mq4`, `llama_store_pp`, 311 tensors, established `max |delta| = 0` oracle | Physical PP preserves zero logit delta, 14/14 layer placement, and HW-003 VRAM threshold. |
| Qwen35 PP | HW-004 pinned Qwen35 model/prompt and captured single-device cold/two-turn hashes | Physical PP hashes identical, placement inventory exact, and HW-004 VRAM threshold passes. |
| TP teardown | HW-005 pinned TP model/prompt and captured single-device hash | Four physical TP cycles reproduce the hash, leave no live stream/communicator, and meet the VRAM threshold. |
| Reset | COR-002 reset-contract tests and `serve-multiturn-gate.sh` across Single, PP, TP, EP, spec/MTP, recurrent/conv, and VL adapters | Every adapter proves the central contract; abort, overflow, reset-command, and normal-completion cases pass. |
| Parser | COR-003 pending UTF-8/reasoning/tool-call/injected-EOS/stop/budget/abort fixtures | Final output is emitted exactly once with no cross-turn residue. |
| AR/spec | Canonical DFlash fixtures from `scripts/coherence-gate-dflash.sh`; fixed prompt and binary hashes | Coherence gate passes; accepted-token accounting and AR fallback tests pass. |
| VL | Canonical Qwen35-VL fixture captured by VL-001; dots.ocr canonical `dots_ocr_smoke_001_vllm.json`/demo image oracle | Qwen35-VL AR parity/reset passes; dots.ocr preserves its recorded F1 oracle in AR and n-gram modes; unsupported VL speculation rejects explicitly. |
| Step/manifest | Architecture inventory produced by STEP-004 with one pinned parity fixture per registered decoder family | Every existing supported forward cell has manifest coverage and deterministic Step parity, or a documented non-decoder exception. |
| Axis matrix | PAR-001 support matrix and PAR-002 required new cells | Every cell selects the documented path or returns the documented refusal; every supported multi-GPU cell has physical evidence. |
| Documentation/PR | DOC-001 named-document list, tracker schema check, PR #527 checklist | Every stale document links here, IDs/fields validate, PR IDs match, and required CI/reviews are green. |

## Dependency Order

1. HW-003 and HW-005 can start immediately when suitable machines are available; HW-001 and HW-002 wait for STEP-002 so physical RCCL validation exercises the final Step/manifest EP path; HW-004 waits for GEN-001.
2. COR-001, COR-003, COR-004, COR-005, COR-006, PAR-001, STEP-001, and DOC-001 are independent starting points.
3. COR-004 feeds COR-002; COR-001 through COR-003 feed SPEC-001; SPEC-001 feeds SPEC-002; VL-001 depends only on COR-002 and COR-003. SPEC-003 is deferred until final closure and blocks DOC-002 only.
4. STEP-001 feeds STEP-003; PAR-001 feeds STEP-002; STEP-001, STEP-002, STEP-003, and PAR-001 feed STEP-004.
5. COR-002 plus STEP-001/STEP-003 feed GEN-001; GEN-001 feeds physical Qwen35 validation HW-004.
6. PAR-001 and STEP-004 define PAR-002. COMP-001 independently decides TP x EP scope; if required, it creates a conditional COMP implementation follow-up with its own dependencies.
7. DOC-002 is the only final closure task and cannot complete while any dependency is open.

## Parallel Streams

- **Physical validation:** HW-003 and HW-005 can run independently; HW-001 and HW-002 follow STEP-002; HW-004 follows GEN-001.
- **Correctness ownership:** COR-001, COR-003, COR-004, COR-005, and COR-006 initially; COR-002 follows the eviction decision.
- **Generation/spec:** GEN-002 can proceed alongside SPEC-001; SPEC-002 follows metadata and shared orchestration. SPEC-003 is deferred to final closure.
- **Multimodal:** VL-001 follows only COR-002/COR-003 and can proceed independently of SPEC-001; VL-002 waits for SPEC-001 because it adopts the existing n-gram strategy through shared speculative orchestration.
- **Execution/placement:** STEP-001 and PAR-001 can start together; STEP-002 and STEP-003 then proceed largely independently before STEP-004.
- **Documentation:** DOC-001 can proceed without runtime or hardware work.

## Update Protocol

1. After the one-time bootstrap correction recorded in the immutable Task ID Migration table, task IDs must never be renamed, renumbered, or reused. Add a new stable prefixed ID for newly discovered work.
2. Before implementation, set only the selected task to `in progress`; record a newly discovered blocker or follow-up immediately.
3. Do not set `complete` until every acceptance criterion and validation item passes. Emulation never closes a physical-hardware criterion.
4. Replace `Evidence: Pending` with commit hashes, exact commands/results, hardware topology, GPU architecture, ROCm/RCCL versions, and artifact/report links as applicable. Use `None` only when evidence or a field genuinely does not apply.
5. Update this tracker in the same change that alters task status. After pushing, synchronize the matching checklist IDs in PR #527.
6. If the PR and tracker diverge, correct the PR mirror; never edit historical evidence to manufacture agreement.
