# Device-Mesh Refactor Tracker

## Authority Rule

This file is the authoritative source of current status for the device-mesh refactor. [PR #527](https://github.com/fivetide/hipfire/pull/527) mirrors the active task IDs for contributor visibility. If the PR, a handover, a task report, a design note, or any other status document disagrees with this tracker, this tracker wins. Historical documents remain evidence, not status authorities.

## Completion Definition

The refactor is complete only when every active task below is `Complete`, including all physical-hardware gates. Completion means:

- RCCL expert-parallel serving is validated for DeepSeek4 and MiniMax on distinct physical GPUs.
- Dense PP, Qwen35 PP, and TP teardown are validated on distinct physical GPUs with correct placement, transfer, output, and bounded post-unload VRAM.
- Request metadata, reset ownership, parser finalization, and session-state ownership are total and regression-tested.
- Ordinary AR, speculative/MTP, VL, Step/manifest, and required PP/TP/EP model-family paths use the shared architecture and mesh abstractions without legacy duplicate orchestration.
- Every supported model family has an explicit tested PP/TP/EP support decision; TP x EP is either implemented for a concrete requirement or explicitly remains out of scope.
- Stale status documentation points here, the full required validation matrix passes, evidence is recorded, and PR #527 is synchronized and merge-ready.

Emulation can prove structure and byte parity, but it cannot satisfy an acceptance criterion that explicitly requires distinct physical GPUs.

## Current Status

**Foundation implemented; refactor incomplete.** The mesh, manifest, Step execution, generic AR dispatch, model-parallel ownership, and god-struct foundation are substantial and tested. Physical topology evidence and the integration gaps tracked below remain open. No open item is implicitly waived by earlier emulated validation.

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

- **Status:** Blocked
- **Dependencies:** STEP-002
- **Goal:** Validate the production RCCL expert-parallel path for DeepSeek4 without the peer-all-reduce fallback.
- **Acceptance criteria:** Pin the DeepSeek4 model artifact SHA-256 and prompt-file MD5 before testing; capture the existing peer-all-reduce `ep_decode_parity` committed-token hash as the oracle; on at least two distinct GPUs, the RCCL run must produce the identical committed-token hash, pass the same multi-turn assertions, complete four load/generate/reset/unload cycles without hangs or invalid access, and return each GPU to within 64 MiB of its post-first-unload baseline with no monotonic growth across cycles 2-4.
- **Validation:** Run `ep_decode_parity` and its multi-turn serving fixture first with `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` to capture the oracle, then with RCCL enabled and `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` unset; record artifact/prompt digests, topology, GPU architecture, ROCm/RCCL versions, exact commands, token hashes, and per-cycle VRAM.
- **Hardware:** At least two distinct RCCL-capable AMD GPUs with enough aggregate VRAM for the pinned DeepSeek4 fixture.
- **Evidence:** Pending

### HW-002 MiniMax RCCL EP Validation

- **Status:** Blocked
- **Dependencies:** STEP-002
- **Goal:** Validate the production RCCL expert-parallel path for MiniMax without the peer-all-reduce fallback.
- **Acceptance criteria:** Pin the MiniMax model artifact SHA-256 and deterministic prompt-file MD5 before testing; capture the emulated/peer EP committed-token hashes for cold prefill, LCP reuse, and the Tokyo-then-Germany multi-turn fixture as oracles; RCCL on at least two distinct GPUs must match every hash, complete four load/generate/unload cycles, and return each GPU to within 64 MiB of its post-first-unload baseline with no monotonic growth across cycles 2-4.
- **Validation:** Run the existing MiniMax EP deterministic capital/code, LCP, and Tokyo-then-Germany multi-turn fixtures with the peer path to capture oracles, then repeat with RCCL and the peer fallback disabled; record digests, topology, versions, commands, hashes, and per-cycle VRAM.
- **Hardware:** At least two distinct RCCL-capable AMD GPUs with enough aggregate VRAM for the pinned MiniMax fixture.
- **Evidence:** Pending

### HW-003 Physical Dense PP Validation

- **Status:** Blocked
- **Dependencies:** None
- **Goal:** Prove dense pipeline placement and boundary transfer on physically separate devices.
- **Acceptance criteria:** Using `qwen3-0.6b-llama.mq4` in `llama_store_pp`, PP=2 must preserve the established single-device oracle of `max |delta| = 0` across logits; the 28 layers must remain banded 14/14 with embed on stage 0 and output norm/lm_head on stage 1; allocation inspection must show no stage-owned weight page on the wrong GPU; four load/forward/unload cycles must return each GPU to within 64 MiB of its post-first-unload baseline with no monotonic growth across cycles 2-4.
- **Validation:** Run `llama_store_pp` on two distinct devices, capture the 311-tensor placement inventory, logit delta, boundary-copy path, per-device peak VRAM, and per-cycle post-unload VRAM; then run the dense PP serving smoke with a pinned prompt MD5 and compare its committed-token hash to single-device generation.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs; a homogeneous pair is preferred for the first proof.
- **Evidence:** Pending

### HW-004 Physical Qwen35 PP Validation

- **Status:** Blocked
- **Dependencies:** GEN-001
- **Goal:** Prove Qwen35 arch-resident pipeline execution and teardown on physically separate devices.
- **Acceptance criteria:** Before the physical run, pin the Qwen35 model SHA-256 and prompt-file MD5 and capture single-device committed-token hashes for cold generation and a two-turn recurrent-reset fixture; PP=2 on distinct GPUs must match both hashes, place every hybrid attention/recurrent weight and state allocation on its assigned stage, use the peer boundary path, and return each GPU to within 64 MiB of its post-first-unload baseline after four cycles with no monotonic growth across cycles 2-4.
- **Validation:** Run Qwen35 single-device oracle capture followed by PP=2 deterministic cold, two-turn reset, placement, boundary-transfer, and four-cycle load/unload tests; record artifact/prompt digests, topology, exact commands, hashes, allocation inventory, and VRAM traces.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for the pinned Qwen35 fixture.
- **Evidence:** Pending

### HW-005 Physical TP Teardown Validation

- **Status:** Blocked
- **Dependencies:** None
- **Goal:** Confirm TP teardown frees allocations, pools, streams, and communicator resources on real multi-GPU hardware.
- **Acceptance criteria:** Pin the TP-capable model SHA-256 and prompt-file MD5 and capture its single-device committed-token hash before testing; at least four TP=2 load/generate/unload cycles on distinct GPUs must reproduce that hash, leave no live model stream or communicator after unload, return each GPU to within 64 MiB of its post-first-unload baseline, and show no monotonic VRAM growth across cycles 2-4.
- **Validation:** Capture the single-device oracle, then run four TP=2 cycles while recording exact commands, hashes, per-device VRAM before load and after unload, and stream/communicator diagnostics; report the baseline and maximum absolute drift.
- **Hardware:** At least two supported AMD GPUs usable by the production TP path.
- **Evidence:** Pending

### COR-001 Wire `mtp_k` Metadata

- **Status:** Ready
- **Dependencies:** None
- **Goal:** Make the configured/load-message `mtp_k` value the deliberate source used by generation, or remove the unsupported knob rather than silently ignoring it.
- **Acceptance criteria:** `ModelMeta` receives the configured value exactly once; native/spec generation reads that value with documented environment precedence; no stale flat field or self-assignment remains; CLI metadata exposes the setting; tests cover default, configured, and environment-override behavior.
- **Validation:** Run targeted Rust metadata/generation tests, `cli/config_meta.test.ts`, and searches proving generation no longer bypasses `meta.mtp_k`.
- **Hardware:** None
- **Evidence:** Pending

### COR-002 Make Reset Total

- **Status:** Ready
- **Dependencies:** COR-004
- **Goal:** Define and implement the single authoritative reset contract: request-owned state is cleared by `SessionState`, architecture-owned state is reset through exhaustive dispatch, and speculative state is reset by the same entry point.
- **Acceptance criteria:** One reset entry point and ownership contract cover abort, overflow, reset command, normal completion, VL, single, PP, TP, EP, speculative, recurrent, and conv state; adding a model-state variant cannot silently omit its reset arm. Integration tasks do not redefine reset semantics: they only implement their architecture adapter and prove conformance to COR-002.
- **Validation:** Run reset-contract unit tests, exhaustiveness/ownership checks, `serve-multiturn-gate.sh`, architecture-specific multi-turn tests, and abort/overflow/reset-command regressions for single and mesh paths.
- **Hardware:** A supported AMD GPU; distinct GPUs are additionally required for integration proof, not for defining or implementing the reset contract.
- **Evidence:** Pending

### COR-003 Finalize Parser On Pending EOS

- **Status:** Ready
- **Dependencies:** None
- **Goal:** Ensure EOS and request termination always finalize buffered parser output exactly once.
- **Acceptance criteria:** Every stop mode invokes parser finalization when bytes, reasoning markers, tool-call fragments, or forced tokens remain pending; injected EOS semantics remain pre-commit where required; no output is duplicated, dropped, or leaked across turns.
- **Validation:** Add focused parser tests for pending UTF-8, reasoning, tool-call, injected-EOS, stop-sequence, budget, and abort cases; rerun architecture byte-parity and parser/coherence gates.
- **Hardware:** None for unit tests; a supported AMD GPU for end-to-end parity gates.
- **Evidence:** Pending

### COR-004 Decide Eviction Ownership

- **Status:** Ready
- **Dependencies:** None
- **Goal:** Decide and enforce whether eviction is resettable request state in `SessionState` or persistent/model-owned state.
- **Acceptance criteria:** The ownership decision is documented with lifecycle rationale; the field is moved or explicitly retained accordingly; reset, reuse, and speculative commit semantics follow that decision; tests prevent cross-request eviction bleed and accidental loss of intentionally persistent state.
- **Validation:** Run ownership/reset unit tests plus multi-turn and speculative eviction scenarios; inspect `LoadedModel` so no duplicate eviction authority remains.
- **Hardware:** None for ownership tests; a supported AMD GPU for end-to-end eviction behavior.
- **Evidence:** Pending

### GEN-001 Complete Qwen35 Arch-Resident PP

- **Status:** Ready
- **Dependencies:** COR-002, STEP-001, STEP-003
- **Goal:** Complete Qwen35 PP through the arch-resident `ModelParallel::Pp(PipelineImpl::ArchResident)` path for hybrid attention and DeltaNet layers.
- **Acceptance criteria:** Load, prefill, decode, recurrent/conv state, sampling, and unload use the generic PP ownership and stage interfaces; the Qwen35 adapter implements the COR-002 reset contract without creating a second reset authority; no legacy `pp`/`pp_gpus` side channel or duplicate Qwen35 PP loop remains; emulated PP parity is byte- or token-identical before physical validation.
- **Validation:** Run Qwen35 single-versus-emulated-PP deterministic parity, COR-002 conformance and recurrent multi-turn/reset tests, placement assertions, and repeated unload tests; then hand off to HW-004.
- **Hardware:** One supported AMD GPU for emulated PP; physical closure is HW-004.
- **Evidence:** Pending

### GEN-002 Add DeepSeek4 Single-GPU Fallback

- **Status:** Ready
- **Dependencies:** COR-002
- **Goal:** Provide an ordinary single-GPU DeepSeek4 generation path when EP is not selected or available.
- **Acceptance criteria:** DeepSeek4 selects a single-device ArchDispatch/AR path without constructing EP state; DSML grammar/parser behavior matches the EP path; its adapter implements and proves the COR-002 reset contract; deterministic output, tool calls, and unload are coherent; unsupported model sizes fail explicitly on insufficient VRAM.
- **Validation:** Run deterministic prose/code/tool-call and multi-turn parity against the accepted DeepSeek4 behavior, COR-002 reset conformance, load/unload, and low-VRAM failure tests.
- **Hardware:** One supported AMD GPU with enough VRAM for the selected DeepSeek4 fixture.
- **Evidence:** Pending

### SPEC-001 Unify AR And Speculative Orchestration

- **Status:** Ready
- **Dependencies:** COR-001, COR-002, COR-003
- **Goal:** Share request framing, reset, prefill, parser, streaming, accounting, and finalization above AR and speculative strategies.
- **Acceptance criteria:** AR and speculative/MTP execution are strategies under one request lifecycle; accepted-token commit semantics remain strategy-specific; duplicate request orchestration is removed; Qwen35's RAII spec-target guard is represented safely; `ArchDispatch::as_spec_target` is either implemented with a fitting contract or deleted with all dead scaffolding and TODOs removed; strategy adapters conform to COR-002 rather than owning reset semantics.
- **Validation:** Run AR-versus-spec lifecycle tests, DFlash coherence, deterministic accepted-token accounting, parser finalization, COR-002 reset conformance, abort, and multi-turn tests; search for orphaned `as_spec_target` implementations and duplicate request loops.
- **Hardware:** A supported AMD GPU with paired target/draft fixtures for DFlash validation.
- **Evidence:** Pending

### SPEC-002 Native Qwen MTP

- **Status:** Ready
- **Dependencies:** COR-001, SPEC-001
- **Goal:** Integrate native Qwen MTP as a first-class speculative strategy using model metadata and the shared lifecycle.
- **Acceptance criteria:** Native Qwen MTP loads only when compatible weights are present; uses configured `mtp_mode` and `mtp_k`; commits only accepted target tokens; falls back explicitly to AR when disabled or unavailable; its adapter implements the COR-002 contract for all MTP scratch/state; quality and performance reporting uses fixed fixtures.
- **Validation:** Run MTP-off/auto/on selection tests, deterministic acceptance/accounting tests, AR fallback, COR-002 reset conformance, unload loops, coherence gate, and fixed-prompt performance measurements with prompt and binary hashes.
- **Hardware:** A supported AMD GPU with a Qwen model containing native MTP weights.
- **Evidence:** Pending

### VL-001 Adopt Shared Lifecycle For Qwen35-VL

- **Status:** Ready
- **Dependencies:** COR-002, COR-003
- **Goal:** Route Qwen35-VL post-prefill AR generation through the shared request lifecycle while preserving image-conditioned prefill.
- **Acceptance criteria:** This task is AR-only: vision preprocessing and multimodal prefill remain architecture-owned; post-prefill AR parsing, accounting, COR-002 reset conformance, and finalization use shared orchestration; image state cannot bleed across requests; text-only Qwen35 behavior is unchanged. VL target/draft or native-MTP speculation is out of scope until a model-specific quality fixture exists and must be added as a separate SPEC/VL follow-up depending on SPEC-001.
- **Validation:** Run image-plus-text deterministic fixtures, repeated different-image requests, text-only parity, COR-002 reset/abort conformance, and parser finalization; verify unsupported VL speculative modes are rejected explicitly rather than silently selected.
- **Hardware:** A supported AMD GPU with enough VRAM for the canonical Qwen35-VL fixture.
- **Evidence:** Pending

### VL-002 Adopt Shared Lifecycle For dots.ocr

- **Status:** Ready
- **Dependencies:** COR-002, COR-003, SPEC-001
- **Goal:** Route dots.ocr post-image-prefill AR and existing model-free n-gram decoding through the shared request lifecycle without changing its custom framing or vision tower.
- **Acceptance criteria:** Image encoding and custom prompt framing remain dots.ocr-owned; post-prefill AR and existing n-gram selection, parser finalization, accounting, COR-002 reset conformance, and unload use shared orchestration; OCR output preserves the canonical fixture quality; image state is request-local. Target/draft and native-MTP VL speculation are out of scope and require a separate follow-up with a dots.ocr quality oracle.
- **Validation:** Run the canonical dots.ocr image fixture and F1 comparison in AR and existing n-gram modes, repeated-image isolation, text-decoder parity, COR-002 reset/abort conformance, and unload tests; verify other speculative modes are rejected explicitly.
- **Hardware:** A supported AMD GPU for the canonical dots.ocr fixture.
- **Evidence:** Pending

### STEP-001 Adopt Step/Manifest For DeltaNet

- **Status:** Ready
- **Dependencies:** None
- **Goal:** Represent Qwen35 DeltaNet weights, state, and forward execution through manifests and the Step spine.
- **Acceptance criteria:** The Qwen35 weight manifest covers layer-type-specific fused projections, norms, convolution, recurrent parameters, and dense/MoE variants; placement derives from policy; DeltaNet forward emits/executes Steps without a parallel bespoke layer loop; single-device output remains identical.
- **Validation:** Run manifest coverage/placement tests, source-to-store byte/dtype checks, Step-versus-legacy deterministic parity during migration, and Qwen35 coherence tests.
- **Hardware:** None for manifest tests; a supported AMD GPU for forward parity.
- **Evidence:** Pending

### STEP-002 Adopt Step/Manifest For MoE

- **Status:** Ready
- **Dependencies:** PAR-001
- **Goal:** Fold routed-expert execution and its EP collectives into the common Step/manifest path.
- **Acceptance criteria:** Expert ownership, compact shard layout, routing, zero/dummy handling, and collective hints derive from the manifest/mesh; DeepSeek4, MiniMax, and Qwen35 MoE variants no longer require an independent executor; single and already-supported EP behavior preserve accepted output. This task adopts existing architecture forwards and does not add a new PP/TP/EP support cell.
- **Validation:** Run manifest shard tests, emulated EP deterministic parity for each covered family, expert-routing edge cases, transactional load failure, and EP coherence tests; physical RCCL closure remains HW-001/HW-002.
- **Hardware:** One supported AMD GPU for emulated EP; physical RCCL validation requires the hardware in HW-001 and HW-002.
- **Evidence:** Pending

### STEP-003 Adopt Step/Manifest For Recurrent And Conv State

- **Status:** Ready
- **Dependencies:** COR-002, STEP-001
- **Goal:** Represent recurrent and convolution operations/state in Step execution with mesh-aware placement and reset.
- **Acceptance criteria:** Recurrent and conv state manifests encode layer ownership; Step execution handles prefill/decode state updates on the owning stage/device; boundary movement is explicit; the adapter implements the COR-002 reset contract; bespoke recurrent/conv forward loops are removed after parity.
- **Validation:** Run state placement tests, multi-token prefill/decode parity, COR-002 conformance, repeated multi-turn tests, PP emulation, and Qwen35 recurrent coherence tests.
- **Hardware:** A supported AMD GPU; physical PP closure is HW-004.
- **Evidence:** Pending

### STEP-004 Migrate Remaining Forward Paths

- **Status:** Ready
- **Dependencies:** STEP-001, STEP-002, STEP-003, PAR-001
- **Goal:** Adopt Step/manifest for every remaining architecture forward path that already has a supported Single/PP/TP/EP cell, or record a justified non-decoder exception.
- **Acceptance criteria:** An inventory names every architecture and forward entry point; existing supported decoder paths use Step/manifest; encode-only or vision-only exceptions have explicit boundaries and ownership; obsolete executors and duplicate placement logic are deleted; each migration has parity evidence. This task does not create support for a new parallel axis; PAR-002 owns those implementations.
- **Validation:** Run an inventory search against architecture registration and forward symbols, per-family deterministic parity/coherence tests for already-supported cells, workspace tests, and checks that no unapproved bespoke decoder executor remains.
- **Hardware:** Supported AMD GPU coverage for each migrated, already-supported path; exact models/topologies follow PAR-001 decisions.
- **Evidence:** Pending

### PAR-001 Decide Model-Family PP/TP/EP Support

- **Status:** Ready
- **Dependencies:** None
- **Goal:** Define the supported parallel axes and explicit refusal behavior for every registered model family.
- **Acceptance criteria:** A maintained matrix covers Single, PP, TP, and EP for every family; each cell is supported, planned with a task dependency, or explicitly unsupported with a technical reason; runtime selection and errors enforce the matrix; tests prevent accidental claims or silent fallback.
- **Validation:** Compare the matrix with architecture registration and load dispatch; run selection/refusal tests for every family and axis; verify docs and CLI report the same capabilities.
- **Hardware:** None for decisions and refusal tests; supported cells inherit their implementation task's hardware gates.
- **Evidence:** Pending

### PAR-002 Implement Required Additional PP/TP/EP Paths

- **Status:** Blocked
- **Dependencies:** COR-002, PAR-001, STEP-004
- **Goal:** Implement only the new model-family PP/TP/EP support cells that PAR-001 marks required for this refactor.
- **Acceptance criteria:** Every newly required matrix cell has mesh-derived placement, reuses the architecture's STEP-004-adopted forward path, implements the COR-002 reset contract, covers lifecycle/unload and explicit unsupported combinations, and has deterministic parity. Architecture-forward migration itself remains STEP-004 scope.
- **Validation:** Run per-new-cell unit, emulated topology, coherence/parity, COR-002 conformance, and teardown tests; require physical topology evidence before marking any newly supported multi-GPU cell production-ready.
- **Hardware:** Determined by the new cells in PAR-001; physical multi-GPU closure is mandatory for production support.
- **Evidence:** Pending

### PAR-003 Gate Optional TP x EP Composition

- **Status:** Ready
- **Dependencies:** None
- **Goal:** Make an unconditional scope decision for TP x EP composition in this refactor.
- **Acceptance criteria:** Record one decision: either TP x EP is out of scope and `TP>1 && EP>1` is explicitly rejected, or a concrete deployment requirement names the model, topology, owner, and measurable success target. In the latter case, create a new conditional PAR task for design/implementation/physical validation; PAR-003 itself completes when the decision and refusal-or-follow-up are recorded and never waits on implementation or hardware.
- **Validation:** Review the requirement record and support matrix; for the out-of-scope decision, run configuration/refusal tests; for the required decision, verify the new follow-up ID exists with dependencies and acceptance criteria.
- **Hardware:** None
- **Evidence:** Pending

### DOC-001 Consolidate Stale Status Documentation

- **Status:** Ready
- **Dependencies:** None
- **Goal:** Prevent historical device-mesh reports from presenting stale plans as current status.
- **Acceptance criteria:** The stale handover/status/phase, follow-up, review, pivot, ArchDispatch, god-struct, and SDD progress documents named in `docs/superpowers/specs/2026-07-12-device-mesh-tracking-design.md` carry an appropriate superseded or chronological-evidence notice linking here; historical evidence is preserved; conclusively closed findings are labeled accurately.
- **Validation:** Search the named documents for unqualified authority/current-status claims; verify every notice links to this file; inspect the diff to confirm no forensic history was deleted or rewritten.
- **Hardware:** None
- **Evidence:** Pending

### COMP-001 Final Validation And Merge Gate

- **Status:** Blocked
- **Dependencies:** HW-001, HW-002, HW-003, HW-004, HW-005, COR-001, COR-002, COR-003, COR-004, GEN-001, GEN-002, SPEC-001, SPEC-002, VL-001, VL-002, STEP-001, STEP-002, STEP-003, STEP-004, PAR-001, PAR-002, PAR-003, DOC-001
- **Goal:** Establish that the completed refactor is correct, production-honest, documented, and ready to merge.
- **Acceptance criteria:** Every listed dependency and every conditional follow-up created by PAR-003 is `Complete` with evidence; every row in the Final Validation Matrix passes against its named fixture/oracle; HW-001 through HW-005 meet the 64 MiB/no-monotonic-growth thresholds; no stale active checklist conflicts with this tracker; PR #527 mirrors all IDs, required CI checks pass, and no blocking review finding remains.
- **Validation:** Execute and archive every row in the Final Validation Matrix, rerun tracker schema and documentation-link checks, inspect the final branch diff and PR checks/reviews, and attach the physical PP/TP/EP reports with artifact/prompt digests and per-cycle VRAM.
- **Hardware:** The union of hardware required by HW-001 through HW-005 and each supported model-family validation cell.
- **Evidence:** Pending

## Final Validation Matrix

COMP-001 cannot complete from a generic “tests pass” statement. Its evidence must enumerate these rows with exact command, commit, fixture digest, result, and report path:

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
2. COR-001, COR-003, COR-004, PAR-001, STEP-001, and DOC-001 are independent starting points.
3. COR-004 feeds COR-002; COR-001 through COR-003 feed SPEC-001; SPEC-001 feeds SPEC-002 and VL-002 only. VL-001 depends only on COR-002 and COR-003.
4. STEP-001 feeds STEP-003; PAR-001 feeds STEP-002; STEP-001, STEP-002, STEP-003, and PAR-001 feed STEP-004.
5. COR-002 plus STEP-001/STEP-003 feed GEN-001; GEN-001 feeds physical Qwen35 validation HW-004.
6. PAR-001 and STEP-004 define PAR-002. PAR-003 independently decides TP x EP scope; if required, it creates a conditional implementation follow-up with its own dependencies.
7. COMP-001 is the only final closure task and cannot complete while any dependency is open.

## Parallel Streams

- **Physical validation:** HW-003 and HW-005 can run independently; HW-001 and HW-002 follow STEP-002; HW-004 follows GEN-001.
- **Correctness ownership:** COR-001, COR-003, and COR-004 initially; COR-002 follows the eviction decision.
- **Generation/spec:** GEN-002 can proceed alongside SPEC-001; SPEC-002 follows metadata and shared orchestration.
- **Multimodal:** VL-001 follows only COR-002/COR-003 and can proceed independently of SPEC-001; VL-002 waits for SPEC-001 because it adopts the existing n-gram strategy through shared speculative orchestration.
- **Execution/placement:** STEP-001 and PAR-001 can start together; STEP-002 and STEP-003 then proceed largely independently before STEP-004.
- **Documentation:** DOC-001 can proceed without runtime or hardware work.

## Update Protocol

1. Keep task IDs stable and unique. Add a new prefixed ID rather than renumbering existing tasks.
2. Before implementation, set only the selected task to `In Progress`; record a newly discovered blocker or follow-up immediately.
3. Do not set `Complete` until every acceptance criterion and validation item passes. Emulation never closes a physical-hardware criterion.
4. Replace `Evidence: Pending` with commit hashes, exact commands/results, hardware topology, GPU architecture, ROCm/RCCL versions, and artifact/report links as applicable. Use `None` only when evidence or a field genuinely does not apply.
5. Update this tracker in the same change that alters task status. After pushing, synchronize the matching checklist IDs in PR #527.
6. If the PR and tracker diverge, correct the PR mirror; never edit historical evidence to manufacture agreement.
