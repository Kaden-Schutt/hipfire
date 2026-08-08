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

**Foundation implemented; refactor incomplete.** `COR-001` through `COR-006`, `DOC-001`, `STEP-001`, `STEP-002`, `STEP-002R`, `PAR-001`, `CAP-001`, and `COMP-001` are complete. The mesh, manifest, Step execution, generic AR dispatch, model-parallel ownership, god-struct foundation, COR-002 terminal reset, COR-003 terminal lifecycle, COR-005 transactional loading, COR-006 eviction physical-cap KV alignment, STEP-001 DeltaNet migration, STEP-002 MoE Step/manifest adoption, STEP-002R owner-preserving bundle teardown, PAR-001 policy characterization, CAP-001 capability admission, and COMP-001 TP×EP refusal work are substantial and tested. `GEN-001` was briefly mis-marked complete in `b1d54d5c2`; corrected 2026-08-08 to blocked — Qwen35 arch-resident PP remains `generate_multi` (see the task's blocked reason). Remaining architecture migrations and the separate physical PP/TP/EP topology tasks tracked below remain open. No open item is implicitly waived by earlier emulated validation.

Contributor validation on two gfx1201 R9700s (2026-07-14, commit `4df03537`) confirmed balanced Qwen35 PP allocation and peer access, but did not close either physical PP gate: dense LLaMA forward hit an unclassified illegal access and Qwen35 PP=2 diverged at token 58/100. The evidence and bounded follow-up are recorded under HW-003 and HW-004; neither changes the current execution queue or relaxes exact-parity requirements.

## Execution Priority

This is the implementation queue. The dependency graph below remains the
authoritative constraint; a task is marked `in progress` only when work begins.

1. `STEP-003` — adopt Step/manifest for recurrent and conv state (ready; deps COR-002, STEP-001 complete). Unblocks `STEP-004`, which gates `AXIS-001` through `AXIS-004`.
2. `GEN-002` — DeepSeek4 single-GPU fallback (ready; deps COR-002 complete). Independent stream; can proceed alongside STEP-003.
3. `SPEC-001` — unify AR and speculative orchestration (ready; deps COR-001..003 complete). Unblocks `SPEC-002` and `VL-002`.
4. `VL-001` — shared lifecycle for Qwen35-VL (ready; deps COR-002, COR-003 complete). Independent multimodal stream.

`COMP-001` (TP×EP refusal) and `CAP-001` (capability contract + dense-EP normalization) are complete and are no longer queue items. `STEP-002R` (owner-preserving bundle teardown) is complete as of 2026-08-08 (PR #18 mirror, commits `064e26e0d` + `c7f142af8`). `GEN-001` is blocked on STEP-003. The `AXIS-001` through `AXIS-004`, `PAR-002`, and `HW-001` through `HW-013` tasks remain dependency-blocked until their capability, family, implementation, and topology prerequisites are satisfied. Hardware tasks remain blocked until the required distinct-GPU topology is available; emulation does not advance them toward completion.

## Completed Foundation Evidence

- Hardware and mesh foundation: `ff709bdc` (`hipfire-hardware` extraction), `0b95b89c` (`DeviceMesh`), `5f4b581c` (`resolve_mesh`), and `e66d6f94` (PP stage/band helpers).
- Manifest and placement foundation: `a6a0acb9` (manifest types), `41b63cdb` (placement), `69c61c05` (collective schedule), plus store-backed llama validation recorded in `.agent-progress/device-mesh-status.md`.
- Model-parallel ownership: `3e99918c` (owning enums), `8c3d7f85` (TP), `a4211e3c` (dense PP), `a4583dbc` (EP), and `0fe02058` (Qwen35 arch-resident PP).
- Session/meta collapse: `a7082ee9`, `4b1a2fe8`, `e16e7c01`, `8be7bf63`, and `9c57148d` established `SessionState`, `PersistState`, reset routing, and `ModelMeta` readers.
- Generic generation foundation: the live `ar_generate` path and StreamParser/ArchDispatch folds are documented with parity evidence in `.agent-memory/notes/daemon-god-struct-archdispatch-design.md` and `.superpowers/sdd/progress.md`.
- Teardown and peer-order fixes: `eafd8663` and `17fc1c4c` closed the known emulated TP/PP unload leak and corrected peer-access ordering; physical teardown remains tracked below.
- Eviction physical-cap KV alignment: `770d89ec` added `kv_physical_cap` to `LoadCtx`, threaded it into Qwen35 `KvDims` via the carrier, added pre-allocation budget/beta rejection, and added three VRAM-allocation GPU tests.
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

### HW-003 Physical Standard-Attention PP Validation

- **Status:** blocked
- **Dependencies:** AXIS-001
- **Goal:** Prove physical PP placement, boundary transfer, parity, and lifecycle for every admitted standard-attention family cell.
- **Acceptance criteria:** For each admitted LLaMA, plain Qwen, and Qwen2-VibeThinker PP cell, pin the model artifact SHA-256 and prompt-file MD5; PP on distinct GPUs must match the single-device oracle, place every stage-owned weight/state allocation on the assigned device, preserve the explicit boundary transfer, and complete four load/forward/reset/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic growth across cycles 2-4. The canonical `qwen3-0.6b-llama.mq4` 14/14 placement and `max |delta| = 0` oracle remain the dense reference case.
- **Validation:** Run PP=1, emulated PP, and physical PP for every admitted standard-attention family with deterministic mode; record artifact/binary/prompt digests, topology, ROCm/driver versions, first failing HIP launch when applicable, placement inventory, boundary-copy trace, logits/token parity, per-device peak VRAM, and per-cycle post-unload VRAM. `llama_store_pp` alone cannot prove physical execution when it forces emulation.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs; a homogeneous pair is preferred for the first proof.
- **Evidence:** External report from `taniguchi-taku-softm`, 2026-07-14, at `4df035373669369484797abdd274f3f710c4c061`: two gfx1201 R9700s, ROCm 7.2.4, RCCL 2.27.7.70204, bidirectional P2P. Noncanonical `qwen3-0.6b.hf4` SHA-256 `7760b19dfb940f8b33078eb524602b4f2b5e6825c6e10c466e6e99bcfc133838` produced correct 155/156 emulated placement but an illegal-memory-access surfaced at logits download. This is preliminary classification evidence only: the canonical artifact was unavailable and no first failing launch or physical dense-PP forward was captured.

### HW-004 Physical Qwen35 PP Validation

- **Status:** blocked
- **Dependencies:** GEN-001
- **Goal:** Prove physical PP placement, boundary transfer, parity, and lifecycle for every admitted Qwen35 dense and MoE PP cell owned by GEN-001.
- **Acceptance criteria:** Before each physical run, pin the Qwen35 model SHA-256 and prompt-file MD5 and capture single-device committed-token hashes for cold generation and a two-turn recurrent-reset fixture; every admitted dense and MoE PP cell on distinct GPUs must match both hashes, place every hybrid attention/recurrent weight and state allocation on its assigned stage, use the peer boundary path, and return each GPU to within 64 MiB of its post-first-unload baseline after four cycles with no monotonic growth across cycles 2-4. A real distinct-GPU FWHT/asym-KV prefill fixture must prove per-band/device Givens cos/sin consumption.
- **Validation:** With GEN-001 complete and deterministic mode explicit, compare PP=1, emulated PP, and physical PP for every admitted dense and MoE Qwen35 cell; record artifact/prompt/binary digests, topology, exact commands, token hashes, placement/allocation inventory, boundary transfers, first numerical difference if any, and per-cycle VRAM. Run the FWHT/asym-KV prefill fixture through `forward_prefill_chunk` and compare it with the pinned single-device oracle.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for the pinned Qwen35 fixture.
- **Evidence:** External report from `taniguchi-taku-softm`, 2026-07-14, at `4df035373669369484797abdd274f3f710c4c061`: physical PP=2 on two gfx1201 R9700s for `qwen3.5-9b.mq4` SHA-256 `ba83acf5bfd5d4e334b0afc26d779734e31623bb7f74e807c3581dfecb3128ad` allocated 2.638 GiB of weights, 0.134 GiB of KV, and 0.006 GiB of DeltaNet state per card; peer access was verified. PP=1 and PP=2 greedy output matched 58/100 tokens and first diverged at index 58. This is a hard parity failure, not accepted numerical variance; the run did not record a prompt MD5 or explicitly force deterministic mode, so it does not localize the cause or satisfy HW-004.

### HW-005 Physical TP Teardown Validation

- **Status:** blocked
- **Dependencies:** HW-006, HW-007, HW-008, HW-009, HW-010, HW-013
- **Goal:** Aggregate physical TP teardown evidence from every admitted standard-attention, Qwen35, non-Qwen35, and vision TP cell.
- **Acceptance criteria:** Each per-family TP report from HW-006, HW-007, HW-008, HW-009, HW-010, and HW-013 must pin its model artifact SHA-256 and prompt-file MD5, reproduce its single-device committed-token oracle on distinct GPUs, leave no live model stream or communicator after unload, and show four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4; HW-005 completes only when all six reports satisfy those conditions.
- **Validation:** Review the six named per-family TP physical reports, verify exact commands, hashes, topology, placement, per-device VRAM before load and after unload, stream/communicator diagnostics, and maximum absolute drift; reject aggregate closure when any family report is missing or fails parity/lifecycle.
- **Hardware:** At least two supported AMD GPUs usable by the production TP path.
- **Evidence:** Pending

### HW-006 Physical Standard-Attention TP Validation

- **Status:** blocked
- **Dependencies:** AXIS-001
- **Goal:** Prove physical TP parity, placement, and lifecycle for every admitted standard-attention LLaMA, plain Qwen, and Qwen2-VibeThinker TP cell.
- **Acceptance criteria:** For each admitted family, pin the model artifact SHA-256 and prompt-file MD5; TP on distinct GPUs must reproduce the single-device committed-token oracle, place tensor shards and state on the declared devices, pass reset/abort/multi-turn behavior, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4.
- **Validation:** Run single-device, emulated TP, and physical TP with deterministic mode; record per-family parity hashes, placement/allocation inventory, shard and collective traces, topology, exact commands, artifact/prompt/binary digests, and per-cycle VRAM and lifecycle diagnostics.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for each admitted fixture.
- **Evidence:** Pending

### HW-007 Physical Qwen35 TP Validation

- **Status:** blocked
- **Dependencies:** AXIS-002
- **Goal:** Prove physical TP parity, placement, state ownership, and lifecycle for admitted dense and MoE Qwen35 TP cells.
- **Acceptance criteria:** For every admitted dense and MoE Qwen35 TP cell, pin the model artifact SHA-256 and prompt-file MD5; physical TP must match the single-device cold and two-turn reset oracles, place hybrid attention/recurrent and expert shard/state allocations on their declared devices, pass reset/abort/multi-turn behavior, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4.
- **Validation:** Compare single-device, emulated TP, and physical TP in deterministic mode; record per-family token hashes, stage/shard/state placement, collective traces, topology, exact commands, artifact/prompt/binary digests, and per-cycle VRAM. This gate does not close Qwen35 EP, which is covered by HW-011.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for the pinned dense and MoE Qwen35 fixtures.
- **Evidence:** Pending

### HW-008 Physical DeepSeek4 PP/TP Validation

- **Status:** blocked
- **Dependencies:** AXIS-003, STEP-004
- **Goal:** Prove physical PP/TP parity, placement, and lifecycle for newly enabled DeepSeek4 PP/TP cells; existing DeepSeek4 EP remains HW-001.
- **Acceptance criteria:** For each admitted DeepSeek4 PP or TP cell, pin the model artifact SHA-256 and prompt-file MD5; physical execution must match the single-device or accepted EP-independent oracle, place trunk/state/shard allocations correctly, pass reset/abort/multi-turn behavior, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4.
- **Validation:** Run deterministic single-device, emulated mesh, and physical PP/TP fixtures; capture parity hashes, placement and boundary/collective traces, topology, exact commands, artifact/prompt/binary digests, first numerical divergence, and per-cycle VRAM/lifecycle diagnostics.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for each admitted DeepSeek4 fixture.
- **Evidence:** Pending

### HW-009 Physical MiniMax PP/TP Validation

- **Status:** blocked
- **Dependencies:** AXIS-003, STEP-004
- **Goal:** Prove physical PP/TP parity, placement, and lifecycle for newly enabled MiniMax PP/TP cells; existing MiniMax EP remains HW-002.
- **Acceptance criteria:** For each admitted MiniMax PP or TP cell, pin the model artifact SHA-256 and deterministic prompt-file MD5; physical execution must match the accepted single-device/EP oracle for cold, LCP, and Tokyo-then-Germany multi-turn fixtures, place model/state/shard allocations correctly, pass reset/abort/multi-turn behavior, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4.
- **Validation:** Run deterministic single-device, emulated mesh, and physical PP/TP fixtures; record capital/code, LCP, and Tokyo-then-Germany token hashes, placement and boundary/collective traces, topology, exact commands, artifact/prompt/binary digests, and per-cycle VRAM/lifecycle diagnostics.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for each admitted MiniMax fixture.
- **Evidence:** Pending

### HW-010 Physical LFM2 Dense+MoE/Cohere2-MoE PP/TP/EP Validation

- **Status:** blocked
- **Dependencies:** AXIS-003, STEP-004
- **Goal:** Prove physical parity, placement, and lifecycle for newly enabled dense LFM2 PP/TP, LFM2-MoE PP/TP/EP, and Cohere2-MoE PP/TP/EP cells.
- **Acceptance criteria:** For every admitted dense LFM2, LFM2-MoE, or Cohere2-MoE cell, pin the per-family model artifact SHA-256 and prompt-file MD5; physical execution must match the single-device or emulated oracle, place routed experts and PP/TP state/shards correctly, pass reset/abort/multi-turn behavior, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4. EP evidence applies only to MoE variants; dense LFM2 `EP>1` is normalized to single via CAP-001 and creates no EP support claim.
- **Validation:** Run per-family deterministic single-device, emulated mesh, and physical PP/TP fixtures for dense LFM2 and physical PP/TP/EP fixtures for LFM2-MoE and Cohere2-MoE; record capability selection/refusal, token hashes, expert/stage/shard placement, boundary and collective traces, topology, exact commands, artifact/prompt/binary digests, and per-cycle VRAM/lifecycle diagnostics.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM and RCCL/collective support for each admitted fixture.
- **Evidence:** Pending

### HW-011 Physical Qwen35 MoE EP Validation

- **Status:** blocked
- **Dependencies:** AXIS-002
- **Goal:** Prove physical EP parity, expert placement, and lifecycle for admitted Qwen35 MoE EP cells.
- **Acceptance criteria:** For each admitted Qwen35 MoE EP cell, pin the model artifact SHA-256 and prompt-file MD5; physical EP must match the single-device or emulated EP committed-token oracle, place experts and routed state on the declared devices, use the production collective path, pass reset/abort/multi-turn behavior, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4. Dense Qwen35 EP remains a CAP-001 normalization/no-claim case, not an EP production cell.
- **Validation:** Run deterministic single-device, emulated EP, and physical EP fixtures; record artifact/prompt/binary digests, topology, RCCL/ROCm versions, expert ownership and collective traces, token hashes, exact commands, and per-cycle VRAM/lifecycle diagnostics.
- **Hardware:** At least two distinct RCCL-capable AMD GPUs with enough aggregate VRAM for the pinned Qwen35 MoE fixture.
- **Evidence:** Pending

### HW-012 Physical Vision-Family PP Validation

- **Status:** blocked
- **Dependencies:** AXIS-004, VL-001, VL-002
- **Goal:** Prove physical PP parity, image isolation, placement, and lifecycle for admitted Qwen35-VL and dots.ocr PP cells.
- **Acceptance criteria:** For each admitted vision PP cell, pin the model artifact SHA-256, image fixture digest, and prompt-file MD5; physical PP must match the single-device and emulated PP text/image oracles, isolate image state, place vision/text weights and state on the declared stages, pass reset/abort/unload, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4.
- **Validation:** Run deterministic single-device, emulated PP, and physical PP image-plus-text and text-only fixtures; record image/prompt/model/binary digests, placement and boundary traces, text parity, image isolation, topology, exact commands, and per-cycle VRAM/lifecycle diagnostics.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for the pinned Qwen35-VL and dots.ocr fixtures.
- **Evidence:** Pending

### HW-013 Physical Vision-Family TP Validation

- **Status:** blocked
- **Dependencies:** AXIS-004, VL-001, VL-002
- **Goal:** Prove physical TP parity, placement, image isolation, and lifecycle for admitted Qwen35-VL and dots.ocr TP cells.
- **Acceptance criteria:** For each admitted vision TP cell, pin the model artifact SHA-256, image fixture digest, and prompt-file MD5; physical TP must match the single-device and emulated TP text/image oracles, place vision/text shards and state on the declared devices, pass reset/abort/unload, and complete four load/generate/unload cycles within 64 MiB of the post-first-unload baseline with no monotonic VRAM growth across cycles 2-4.
- **Validation:** Run deterministic single-device, emulated TP, and physical TP image-plus-text and text-only fixtures; record image/prompt/model/binary digests, shard/collective traces, text parity, image isolation, topology, exact commands, and per-cycle VRAM/lifecycle diagnostics.
- **Hardware:** At least two mutually peer-accessible supported AMD GPUs with enough aggregate VRAM for the pinned Qwen35-VL and dots.ocr fixtures.
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

- **Status:** complete
- **Dependencies:** COR-004
- **Goal:** Define and implement the single authoritative reset contract: request-owned state is cleared by `SessionState`, architecture-owned state is reset through exhaustive dispatch, and speculative state is reset by the same entry point.
- **Acceptance criteria:** One reset entry point and ownership contract cover abort, overflow, reset command, normal completion, VL, single, PP, TP, EP, speculative, recurrent, and conv state; adding a model-state variant cannot silently omit its reset arm. Integration tasks do not redefine reset semantics: they only implement their architecture adapter and prove conformance to COR-002.
- **Validation:** Run reset-contract unit tests, exhaustiveness/ownership checks, `serve-multiturn-gate.sh`, architecture-specific multi-turn tests, and abort/overflow/reset-command regressions for single and mesh paths.
- **Hardware:** A supported AMD GPU; distinct GPUs are additionally required for integration proof, not for defining or implementing the reset contract.
- **Evidence:** Completed by commit `68a3c18c` (`refactor(runtime): make context reset total`). Reset ownership and lifecycle coverage includes dense TP/PP cache-miss routing, MiniMax EP cache-miss routing, Cohere cold-prefill routing, DSpark retry-safe hidden-buffer freeing, VL cold-reset behavior, and the two-row DFlash prompt-cache miss regression. Qwen cache misses now reset before capacity validation; VL/dots.ocr dirty abort/error paths defer their terminal envelope until fallible reset completes, and reset failures poison/terminate the daemon rather than serving on unknown GPU state. `cargo test --workspace --locked` passed, `git diff --check` passed, the updated Cohere cold + one-token warm-prefix parity gate passed with retained evidence at `/tmp/hipfire-cor-002/cohere-OhH5hG/`, and the updated DFlash two-row parity gate passed with retained evidence at `/tmp/hipfire-cor-002/dflash-joTHnh/`. Serialized `./scripts/serve-multiturn-gate.sh` passed at `/tmp/serve-multiturn-20260717-220023.md`, and serialized `./scripts/coherence-gate-dflash.sh` reported no hard errors at `/tmp/coherence-dflash-20260717-220054.md`. Distinct-device TP/PP/EP proof and physical VL/dots.ocr multi-turn proof remain downstream integration evidence unavailable in this environment; they are not blockers to the COR-002 core reset contract and are not claimed closed here.

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

- **Status:** complete
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
- **Evidence:** 2026-07-19 on UMA gfx1151 (HIP 7.2), using
  `~/.hipfire/models/qwen3-8b.q8f16.hfq` with
  `~/.hipfire/models/qwen3-8b-dflash.hfq`:
  `HIPFIRE_GENERIC_DFLASH_TARGET=...qwen3-8b.q8f16.hfq
  HIPFIRE_GENERIC_DFLASH_DRAFT=...qwen3-8b-dflash.hfq cargo test -p
  hipfire-loader --lib --features dflash-fault-inject
  registry_tests::generic_dflash_load_rolls_back_each_completed_resource --
  --ignored --exact` passed in 1154.75s. The gate exercised every generic
  target/DFlash fault stage, allocation sweep, and repeated load/prefill/step/
  unload cycles. It synchronizes before measurement and drains the pool; ROCm
  driver graph/allocation accounting varied by up to 24.5 MiB across equivalent
  histories, so it rejects post-cleanup free VRAM more than 64 MiB below the
  warm baseline rather than requiring byte-exact `hipMemGetInfo` equality.
  Feature-gated runtime tests: 401 passed, 3 ignored; loader tests: 28 passed,
  11 ignored. Final adversarial review approved.

### COR-006 Align Eviction Physical-Cap Allocation

- **Status:** complete
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
- **Evidence:** Commit `770d89ec` — added `kv_physical_cap: Option<usize>` to
  `LoadCtx`, threaded it through `Qwen35Carrier::load` into the `KvDims`
  passed to `load_bundle`, and added pre-allocation rejection of `budget=0`
  and `budget+beta+4 > max_seq`. Three `#[ignore]`'d GPU tests
  (`qwen35_cask_physical_cap_reduces_kv_allocation`,
  `qwen35_cask_rejects_impossible_budget_beta`,
  `qwen35_no_eviction_load_unchanged`) validate the VRAM differential,
  fast-reject paths, and non-eviction byte-identity respectively. The
  `KvCache` capped-filtered constructors (`new_gpu_q8_capped_filtered` et al.)
  already supported `physical_cap < max_seq` — the gap was only that the
  derived capacity never reached `KvDims`.

### GEN-001 Complete Qwen35 Arch-Resident PP

- **Status:** blocked
- **Dependencies:** COR-002, STEP-001, STEP-002, STEP-003
- **Goal:** Own and complete Qwen35 PP for both dense and MoE cells through the arch-resident `ModelParallel::Pp(PipelineImpl::ArchResident)` path for hybrid attention and DeltaNet layers.
- **Acceptance criteria:** Dense and MoE Qwen35 load, prefill, decode, recurrent/conv state, sampling, and unload use the generic PP ownership and stage interfaces; the Qwen35 adapter implements the COR-002 reset contract without creating a second reset authority; no legacy `pp`/`pp_gpus` side channel or duplicate Qwen35 PP loop remains; dense and MoE emulated PP parity is byte- or token-identical before physical validation.
- **Validation:** Run dense and MoE Qwen35 single-versus-emulated-PP deterministic parity, COR-002 conformance and recurrent multi-turn/reset tests, placement assertions, and repeated unload tests; then hand off both PP families to HW-004.
- **Hardware:** One supported AMD GPU for emulated PP; physical closure is HW-004.
- **Blocked reason:** Status correction 2026-08-08 — the `complete` flip in `b1d54d5c2` was premature. STEP-003 (recurrent/conv Step/manifest adoption) is not complete, PR #527's architecture table still records Qwen35 PP as `Partial: Pp(ArchResident), still generate_multi`, and `docs/MODELS.md` still labels PP "partial arch-resident (GEN-001; not production-supported)". The arch-resident PP owner exists but the generation path remains `generate_multi`, not the unified PP loop; the acceptance criteria and validation are unproven.
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

### SPEC-004 PP+MTP Bounded Carry-Forward

- **Status:** ready
- **Dependencies:** GEN-001, SPEC-002, SPEC-003
- **Goal:** Reintroduce the reviewable PP+MTP capability from #449 / commit `6d86816f` into the unified mesh/generation architecture, not by merging the stale branch: provide bounded PP prefill capabilities, PP-safe compressed-MTP stepping, per-device MTP mirror/snapshot/rollback state, explicit compressed `.mtp` head loading, daemon dispatch/unload, and bounded cycle/depth termination.
- **Acceptance criteria:** PP+MTP uses the unified mesh and generation lifecycle with no duplicate legacy PP orchestration; MTP-versus-DFlash selection and refusal are explicit and correct; the implementation conforms to COR-002 and COR-003 lifecycle ownership; all PP+MTP resource publication, rollback, and unload paths preserve SPEC-003-safe transactional ownership; bounded PP prefill and cycle/depth termination cannot leave uncommitted or unowned state.
- **Validation:** Run deterministic emulated PP=2 versus single-device greedy output with committed-token and acceptance accounting; prove MTP-disabled AR parity; run MTP/DFlash selection and refusal tests; run repeated load/generate/unload cycles and require recovery within the documented 64 MiB no-monotonic-growth threshold; run the DFlash coherence regression; use a fixed PP+MTP fixture/path with artifact, prompt, and binary digests.
- **Hardware:** One supported AMD GPU for emulated PP=2; physical PP closure remains HW-004.
- **Evidence:** Maintainer comment [#4999241024](https://github.com/Kaden-Schutt/hipfire/pull/527#issuecomment-4999241024) carries the intent; implementation evidence Pending.

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

- **Status:** complete
- **Dependencies:** None
- **Goal:** Represent Qwen35 DeltaNet weights, state, and forward execution through manifests and the Step spine.
- **Acceptance criteria:** The Qwen35 weight manifest covers layer-type-specific fused projections, norms, convolution, recurrent parameters, and dense/MoE variants; placement derives from policy; DeltaNet forward emits/executes Steps without a parallel bespoke layer loop; single-device output remains identical.
- **Validation:** Run manifest coverage/placement tests, source-to-store byte/dtype checks, Step-versus-legacy deterministic parity during migration, and Qwen35 coherence tests.
- **Hardware:** None for manifest tests; a supported AMD GPU for forward parity.
- **Evidence:** Qwen35 manifest/resolver/transactional fulfillment is complete, covering the six semantic Steps through the canonical Step paths. Raw-vs-Step GPU parity passed on gfx1151 with HIP 7.2 using `qwen3.5-0.8b`. DFlash coherence passed with report `/tmp/coherence-dflash-20260720-222434.md`; serve multiturn passed with report `/tmp/serve-multiturn-20260720-222552.md`; final cargo checks/tests passed. Paro MoE remains intentionally on the legacy loader pending representation of separate gate/up projections plus shared sidecars; Paro GPU fixture parity was not run.

### STEP-002 Adopt Step/Manifest For MoE

- **Status:** ready
- **Dependencies:** PAR-001
- **Goal:** Fold routed-expert execution and its EP collectives into the common Step/manifest path.
- **Acceptance criteria:** Expert ownership, compact shard layout, routing, zero/dummy handling, and collective hints derive from the manifest/mesh; routed-expert placements and all derived resources have permanent ownership in the immutable `WeightStore`; architecture consumers use private read-only typed projections with no tensor extraction or typed freeing; rank-branded, non-forgeable allocation tokens enforce origin mesh epoch/rank/physical device/pool epoch; raw-pointer `WeightStoreView` values cannot satisfy acceptance. DeepSeek4/MiniMax preserve accepted Single behavior and named structural EP regression examples (`ep_deepseek4` and `ep_minimax`); their full-model EP>1 parity is deferred and non-blocking; Qwen35 requires a canonical Qwen35-MoE 35B fixture before acceptance: record model SHA-256, prompt MD5, binary digest, exact command/topology, and deterministic pass condition: the emulated EP harness uses Single as the sole baseline (EP=1 is its alias and is not a second required run); honest FP32 partition reduction changes addition order, so bitwise-exact logits are NOT required — the gate is exact final-prefill argmax, the first token emitted after prefill, exact decode/generated token IDs, finite logits, and a strict measured and pinned maximum-absolute-logit-delta bound, with multi-turn/reset; report first logit divergence if tokens differ. This task adopts existing architecture forwards and does not add a new PP/TP/EP support cell.
- **Validation:** Run manifest shard tests, expert-routing edge cases, transactional load failure, and EP coherence tests; assert token-origin rejection and failed-free token retention, immutable projection/no-extraction/no-typed-free boundaries, and permanent store ownership for placements and derived resources; for DeepSeek4/MiniMax, run deterministic Single parity and the named structural EP regression examples `ep_deepseek4` and `ep_minimax` only; for Qwen35 only, run full deterministic emulated EP>1 parity and assert the invariant that Qwen35Moe EP remains `Planned`/refused before allocation throughout STEP-002, with no `EpArch::Qwen35` and no daemon admission; AXIS-002 remains the sole Qwen admission owner; physical RCCL closure remains HW-001/HW-002 for DeepSeek4/MiniMax and HW-011 after AXIS-002 for Qwen35.
- **Hardware:** One supported AMD GPU for emulated EP; physical RCCL validation for DeepSeek4/MiniMax requires HW-001/HW-002. Qwen35Moe EP remains refused before allocation through STEP-002; no Qwen `EpArch` or daemon admission is permitted here, and Qwen35 production EP requires HW-011 after AXIS-002, with AXIS-002 as the sole admission owner.
- **Evidence:** Complete — final Oracle Gate 3 returned
  `GATE3_APPROVED_WITH_DEFERRED_DEBT` on 2026-08-06. Approved pieces (Oracle Gate B,
  2026-08-03): single-target frozen-store facade with retained target
  identity; private ID-only Qwen35 MoE projection with borrowed forward
  bindings; direct Frozen MoE staging into `SingleWeightStoreBuilder`; exact
  C2 indexed dispatch admission; source-bound preflight with Legacy fallback;
  and checked published/unpublished unload plus exact-domain retry backlog.
  Qwen35Moe EP remains `Planned`/refused before allocation throughout
  (AXIS-002 sole owner; no `EpArch::Qwen35`; no daemon admission) and the
  residency boundary script asserts that invariant. Phase B did not follow
  test-first ordering; that TDD violation is an explicit process failure and
  no strict-TDD claim is made.   Phase C boundary/evidence:
  `scripts/check_moe_residency_boundary.sh` is green on this tree and its
  `--self-test` mode catches a controlled violation fixture. CPU evidence:
  runtime weight_store 110 passed / 13 GPU ignored; runtime compile-fail
  doctests 5 passed; qwen35 380 passed / 18 ignored (final merged Oracle
  remediation added 5 CPU tests + 4 GPU-ignored tests and repaired the
  stale GPU-ignored `frozen_moe_resident_build_and_bind` fixture); loader
  132 passed / 10 ignored; dispatch 172 passed / 1 ignored (E8
  routed_indexable test added); dispatch-tests 70 passed; rdna-compute 61
  passed;
  `cargo check -p hipfire-loader --all-targets` and
  `cargo check --workspace --all-targets` finish with pre-existing warnings;
  `git diff --check` passes. Final merged Oracle remediation (2026-08-03,
  strict TDD): the model-wide MQ6 fence is now defined as ANY MoE FFN
  projection in any layer — router / shared_expert_gate / shared gate/up/
  down plus every routed expert gate_up/down (uniform or graded) — for BOTH
  Legacy and Frozen through ONE shared metadata predicate
  (`MoeFfnMetaView::has_mq6`, generic over the projection key so CPU tests
  use fabricated projections). `layers_have_mq6_moe`, the Legacy assembly
  seam (`assembled_legacy_layers_have_mq6`), and the Frozen resident
  publication all consume it, so the storage kinds cannot diverge; the old
  snapshot predicate missed graded (non-uniform) routed MQ6 and the Frozen
  path missed structural MQ6 entirely. The Legacy assembly's own inline
  routed-only `moe_has_mq6` scan (missed structural MQ6) is replaced by the
  shared seam — assembly-level regression: real Legacy fixture with
  shared-only MQ6 layer + pure MQ4 layer → published `moe_has_mq6` true
  (GPU-ignored `legacy_assembly_derives_model_wide_mq6_fence`, passes on
  gfx1151); CPU seam test `assembled_layers_mq6_seam_shared_only_layer_and_pure_layer`.
  Dead `MoeDtypeSnapshot::has_mq6` and `MoeFfnMetaView::proj` deleted
  (snapshot `has_mq6` test assertions removed with the method). O(1) Frozen
  binding — decode materializes routed-expert refs only for the Legacy
  CPU-top-K fallback (`routed_expert_refs_for_params` seam; call-count
  seam proves Frozen = zero resolutions, Legacy = exactly one); Frozen
  prefill's unused `routed_experts` Vec and 21 further dead
  migration-extraction variables removed; `Qwen35BundleBuildError`/
  `BundleBuildTransaction` made `pub(crate)` (no external consumer);
  `MoeResolution::routed_indexable()` now includes admitted E8 (new field,
  E8 test); latent Frozen common-assembly `derived_plans[layer]`
  index-OOB (panic on any real Frozen load) fixed by gating the plan read
  to Legacy mode. GPU-ignored tests `frozen_routed_expert_refs_seam_resolves_zero`
  and `frozen_publication_derives_model_wide_mq6_fence` (routed-MQ6,
  pure-MQ4, and structural-MQ6 fixtures through the real publication seam)
  pass on gfx1151 with `--ignored`. The final merged Oracle gate APPROVED on
  2026-08-03 with zero Critical or Important findings outside accepted
  STEP-002R.
  Current Qwen35-MoE evidence (2026-08-04): the canonical fixture is pinned —
  `~/.hipfire/models/qwen3.6-35b-a3b.mq4`, 22,855,051,520 bytes, SHA-256
  `1dc1c7964de415e0040a540a4300b9518e11b00c13d99c23f576f2b9fe1e8bca`, MD5
  `edde51ec1dac0f2bd42cff5ef1cb8944`; prompt
  `benchmarks/prompts/qwen35_moe_ep_parity.txt` (MD5
  `1aacd3c05cf9695cc799acc59581938d`) is currently untracked/uncommitted with
  the test-only harness lane (never committed); the test-only emulated-EP2
  parity harness (`.agent-progress/run-ep-parity.sh` + `ep_decode_parity`
  example + `ep2_harness`/`store_ep2` modules, uncommitted) satisfies the
  honest-FP32 parity contract — finite logits, identical final-prefill
  argmax/first token/generated token IDs, second-turn/reset true, Q8
  resolved, no first divergence, max delta within the pin (bitwise-exact
  logits NOT required). Five fresh logs
  `.agent-progress/ep-parity-confirm-probe-{1..5}.log` are bit-stable at
  Dmax `0.7081642` with an identical token vector; the pinned tolerance is
  `0.708164275`, the next representable f32 (the accept log may print the
  shortened f32 `0.7081643`); `.agent-progress/ep-parity-confirm-accept.log`
  recorded the acceptance pass; `.agent-progress/ep-parity-confirm-structural.log`
  and `.agent-progress/ep-parity-confirm-cleanup.log` each report 1 passed,
  exit 0; full feature suite 442 passed / 21 ignored; parity binary SHA-256
  `f4b82b109e779f8518332dd86e31371e9a46a5cf50c0ec87360d3a75c95dbd6f`. The
  quant-independent/config-derived/checked conv resolver fix was TDD
  RED/GREEN with
  `conv_physical_shape_alias_is_quant_independent_and_uses_config_kernel` and
  `synthetic_conv_q8_geometry_resolves_through_real_resolver`; it unblocked
  canonical Q8 physical `[channels,1,kernel]` loading. Production AR confirm
  (`.agent-progress/ep-production-ar-confirm.log`): coherent `The capital of
  **France** is **Paris**.<|im_end|>`, 15 tokens, 5.1 tok/s. DFlash regression
  report `/tmp/coherence-dflash-20260804-122132.md`: 4/4 OK, no hard errors,
  no soft/tier3 warnings — it uses the separate canonical qwen3.6-27b
  target/draft and is a production regression gate, NOT direct A3B DFlash
  evidence. Single-shot success-path cleanup/one-page warmed evidence is
  satisfied, but full repeated multi-cycle transactional lifecycle and
  failed-free owner retention remain explicitly deferred to STEP-002R and are
  not claimed. Task 8 Step 3 production/common-plan migration is complete for
  Qwen, MiniMax, and DeepSeek; migrated-tree peer-direct EP2 runs matched the
  unchanged DeepSeek `0x26a13602bedf9926` and MiniMax `0x887c2e7717e9c3bf`
  pins with coherent output. STEP-002R remains accepted separate debt and does
  not block STEP-002 closure.

### STEP-002R Make Qwen35 Frozen Construction Rollback Owner-Preserving

- **Status:** complete
- **Dependencies:** STEP-002
- **Goal:** Replace best-effort Qwen35 Frozen pre-publication rollback with
  origin-preserving, retryable transactions for common weights and every
  auxiliary allocation.
- **Acceptance criteria:** Common manifest fulfillment/conversion/assembly,
  DeltaNet state, Qwen scratch and PrefillBatchScratch, KV construction and
  reconfiguration, and MTP-head construction preserve every allocation as
  published, successfully freed, or returned in an exact retry owner; direct
  uploads retain direct-allocation provenance rather than being fabricated as
  pooled `GpuTensor`s; replacement transitions cannot double-free; complete
  `Qwen35CleanupFailure` aggregates retain both tensor and Frozen owners; no
  `Drop`, logging, string conversion, or best-effort free is a correctness
  mechanism.
- **Validation:** Deterministically inject allocation and cleanup failure at
  every production transaction boundary and assert each allocation identity and
  origin appears exactly once in published, freed, or retained output. Run
  repeated failed-load/retry/unload GPU tests and prove post-retry VRAM recovery.
- **Hardware:** CPU fault-ledger coverage plus one supported AMD GPU with the
  canonical Qwen35-MoE HFQ fixture.
- **Evidence:** Implemented and validated by PR #18 (fivetide mirror of this
  branch; commits `064e26e0d` `feat(loader): owner-preserving teardown for
  every arch bundle` and `c7f142af8` `fix(loader): retained-owner backlog
  closes the String-error teardown gap`, 2026-08-07). The generic core lives
  in `hipfire-runtime/src/gpu_cleanup.rs`: `RetainedGpuTensor`,
  `GpuCleanupFailure` (tensors + boxed `RetryableOwner` category; frozen
  stores travel whole, never flattened), `BundleTeardown` trait,
  `retain_free!` macro, checked frees (`free_tensor_retained`,
  `free_weight_all_checked`, `free_weight_sidecars_checked`,
  `retain_kv_failures`), and a process-local retained-owner backlog
  (`enqueue_cleanup_failure` / `retry_backlog` / `backlog_pending`) so owners
  surviving a terminal retry are enqueued and drained at the next
  load/unload boundary. Qwen35: `Qwen35BundleLoadError` carries cleanup
  owners, checked rollback (`KvCache::free_checked`,
  `Qwen35Scratch::abort_checked`, `PrefillBatchScratch::abort_checked`,
  `Qwen35Weights::free_gpu_checked`), `construct_kv_cache` returns the built
  KV on reconfiguration failure, transactional MTP-head staging
  (`MtpHeadStaging`), store rollback keeps real dtype/shape, and
  rejected-replacement buffers thread into `staging_retained`. Every other
  arch bundle (qwen2, llama incl. dspark, lfm2moe, minimax, cohere2moe,
  deepseek4, dots-ocr) implements `BundleTeardown`; `unload_model`
  dispatches exhaustively through `ModelState::free_checked` and reports
  only after a same-call retry. Fault injection: `frozen-fault-inject`
  feature + `HIPFIRE_FROZEN_FAIL_STAGE` /
  `HIPFIRE_FROZEN_FAIL_FREE` (continuous-while-set). Forensic fixes en
  route: dspark sidecar pread `RefCell` borrow panic, packed-MQ4 expert
  interior-view pooling corruption (`free_moe_ffn_checked`),
  VMM-arena release skipped by `free_tensor_checked`. Validation: GPU fault
  battery 13/13; per-arch load→unload VRAM teardown 8/8 (exact-zero for
  llama/dots-ocr/VMM); retained-backlog double-failure test (620 owners
  enqueued → drained) 5/5 runs; serve_harness battery (qwen3.5:9b) 5/5
  turns, empty=0, attractor=0; CPU suites runtime 821 / loader 106 / qwen35
  424 / qwen2 8 / llama 1 / lfm2moe 2 / minimax 28 / cohere2moe 9 /
  deepseek4 117 / dots-ocr 27; clippy (new symbols) clean; independent
  review 0 blockers (2 majors + 3 minors, all resolved). Residual debt is
  explicitly tracked, not waived: Qwen35 mid-constructor leaks
  (`DeltaNetState::new_with_quant` / `Qwen35Scratch::new_with_kv_max` /
  `PrefillBatchScratch::new_opt` / KV constructors / mid-`fulfill_manifest_gpu`
  — same class as the EP constructor leak; deferred to the allocation-
  tracking refactor) and `load_qwen35_pp` load-error rollback (owned by
  GEN-001/HW-004). Cohere2Moe GPU verification remains env-gated
  (`HIPFIRE_TEARDOWN_COHERE2MOE`). See NEXT-STEPS.md "STEP-002R" section
  and PR #18 body for the full report. Prior recovery checkpoints:
  `/tmp/opencode/qwen35-post-second-recovery-20260727.patch`,
  `/tmp/opencode/qwen35-phase-b-20260727.patch`, and
  `/tmp/opencode/qwen35-phase-b-remediated-20260727.patch` (SHA-256
  `5a4f3d0b64405a3830871eaca7f5eec0afbf1b976126bbd7ae622d93c220a2e2`).

### STEP-003 Adopt Step/Manifest For Recurrent And Conv State

- **Status:** in progress
- **Dependencies:** COR-002, STEP-001
- **Goal:** Represent recurrent and convolution operations/state in Step execution with mesh-aware placement and reset.
- **Acceptance criteria:** Recurrent and conv state manifests encode layer ownership; Step execution handles prefill/decode state updates on the owning stage/device; boundary movement is explicit; the adapter implements the COR-002 reset contract; bespoke recurrent/conv forward loops are removed after parity.
- **Validation:** Run state placement tests, multi-token prefill/decode parity, COR-002 conformance, repeated multi-turn tests, PP emulation, and Qwen35 recurrent coherence tests.
- **Hardware:** A supported AMD GPU; physical PP closure is HW-004.
- **Evidence:** Increment 1 landed 2026-08-08 in `9dcb1862a` (`refactor(qwen35): manifest-derived recurrent/conv state placement`): the hand-built DeltaNet `la_to_device` sidecar is defined out of existence. `qwen35_la_devices(cfg, gpus)` derives the compact LA-layer → device map by filtering `state_manifest` `StateKind::Recurrent` entries (global-layer keyed) through `Gpus::device_for_layer`; `DeltaNetState::new_with_quant_multi` returns `Self` only and `free_gpu_multi` takes `&Qwen35Config`; `LoadedModel.pp_dn_la_to_device` and the `skeleton_pp` param are deleted; loader `validate_qwen35_pipeline_layout` / `reset_qwen35_pipeline_recurrent` / `validate_reset_layout` derive placement from config + mesh (single-model rejection keys off `pp_gpus`); daemon `reset_qwen35_recurrent`, both `generate_multi` reset blocks, and `reset_pp_uncommitted_state!` derive the map from bundle config + gpus. New GPU-gated test `qwen35_la_devices_matches_mesh_placement` proves the manifest map equals the mesh placement and state allocations land on their owning devices (passes with `HIPFIRE_EMULATE_GPUS=2`; the pointer-placement assertion is gated to distinct physical devices because emulated ranks alias one device). CPU suites: loader 106 passed / 12 ignored, qwen35 424 passed / 21 ignored, daemon example 192 passed; `cargo check --workspace --all-targets` clean; `git diff --check` clean. Remaining acceptance scope (not yet closed): single-GPU decode/prefill already execute the six DeltaNet Steps (`build_delta_net_decode_steps` / `build_delta_net_batch_steps` / `build_delta_net_tree_steps`), but the PP multi-GPU forward (`forward_scratch_layers_multi`) still runs bespoke raw kernels and `execute_steps_mesh` still asserts a 1×1 mesh — porting PP recurrent/conv Step execution on the owning stage/device and removing the bespoke loop after parity is the remaining work (overlaps GEN-001 arch-resident PP folding). Emulated PP parity gate (`pp_parity_chatml_50_decode`, `HIPFIRE_HAVE_2_GPU=1` + emulation) diverges at decode step 1 identically with and without this change — pre-existing on this box, HW-004-class, not introduced here.

### STEP-004 Migrate Remaining Forward Paths

- **Status:** ready
- **Dependencies:** STEP-001, STEP-002, STEP-003, PAR-001
- **Goal:** Adopt Step/manifest for every remaining architecture forward path that already has a supported Single/PP/TP/EP cell, or record a justified non-decoder exception.
- **Acceptance criteria:** An inventory names every architecture and forward entry point; existing supported decoder paths use Step/manifest; encode-only or vision-only exceptions have explicit boundaries and ownership; obsolete executors and duplicate placement logic are deleted; each migration has parity evidence. This task does not create support for a new parallel axis; CAP-001 owns capability plumbing, while AXIS-001 through AXIS-004 own concrete family implementations and PAR-002 aggregates their closure.
- **Validation:** Run an inventory search against architecture registration and forward symbols, per-family deterministic parity/coherence tests for already-supported cells, workspace tests, and checks that no unapproved bespoke decoder executor remains.
- **Hardware:** Supported AMD GPU coverage for each migrated, already-supported path; exact models/topologies follow PAR-001 decisions.
- **Evidence:** Pending

### PAR-001 Decide Model-Family PP/TP/EP Support

- **Status:** complete
- **Dependencies:** None
- **Goal:** Publish the policy matrix for every registered model family and characterize current refusal behavior for each PP/TP/EP axis.
- **Acceptance criteria:** A maintained matrix covers Single, PP, TP, and EP for every family; each cell is supported, planned with a named implementation and hardware dependency, explicitly unsupported with a technical reason, or, for dense EP, marked `normalized-to-single(CAP-001)` as the future state. Dense LFM2 is currently refused for its single/PP/TP admission path and is explicitly planned under AXIS-003; dense LFM2 EP is normalized by CAP-001 rather than admitted. Current loader/daemon selection and refusal behavior is recorded without claiming new runtime enforcement; CAP-001 and AXIS-001 through AXIS-004 own subsequent implementation.
- **Validation:** Compare the matrix with architecture registration and current load dispatch; run characterization tests for existing selection/refusal behavior for every family and axis; verify the policy matrix and current refusal report agree. This task does not implement the capability contract or dense-EP normalization.
- **Hardware:** None for policy and current-refusal characterization; supported cells inherit their implementation task's hardware gates.
- **Evidence:** Commit `4e8bd0a2` (`docs(device-mesh): define parallel capability policy`); `cargo test -p hipfire-loader reset_ownership_tests --lib` passed (11 tests); `git diff --check` passed; final integration review approved; PR #527 mirror synchronized. Scoped `cargo clippy -p hipfire-loader --lib --no-deps -- -D warnings` reported nine pre-existing errors outside this diff and was explicitly accepted by the user for this commit; clippy did not pass and is not claimed as passed.

### CAP-001 Architecture Capability Contract And Dense-EP Normalization

- **Status:** complete
- **Dependencies:** PAR-001
- **Goal:** Implement the central architecture capability contract used by loader selection, ownership validation, and daemon dispatch; canonicalize dense `EP>1` to effective `EP=1` before mesh, device, allocation, or collective construction.
- **Acceptance criteria:** Each shipped family declares PP/TP/EP capabilities; planned cells refuse early before GPU allocation with the axis, family, and owning task ID; unsupported cells refuse early before GPU allocation with the axis, family, and technical reason; dense `EP>1` is canonicalized to `normalized-to-single(CAP-001)` before mesh, device, allocation, or collective construction, creates no extra logical device, allocation, collective, or EP support claim, and matches one-replica output.
- **Validation:** Run capability-table selection/refusal tests, dense-EP allocation-inventory tests, one-replica parity tests, and loader/daemon tests; prove canonicalization occurs before mesh/device/allocation/collective setup. No hardware required.
- **Hardware:** None.
- **Evidence:** See `.agent-progress/cap-001-evidence-2026-07-22.md` for exact commands, named pre-mesh proofs, fixture/prompt/binary digests, parsed-token parity output, and fresh validation results.

### PAR-002 Implement Required Additional PP/TP/EP Paths

- **Status:** blocked
- **Dependencies:** CAP-001, STEP-004, GEN-001, AXIS-001, AXIS-002, AXIS-003, AXIS-004, HW-003, HW-004, HW-005, HW-006, HW-007, HW-008, HW-009, HW-010, HW-011, HW-012, HW-013
- **Goal:** Aggregate closure for the PP/TP/EP cells that PAR-001 marks planned after CAP-001 and the AXIS implementation tasks are complete.
- **Acceptance criteria:** This is an aggregate closure task with no family-specific code ownership; every PAR-001 planned cell has implementation evidence from its AXIS task, deterministic emulated parity, and the required named physical evidence from HW-003 through HW-013; PAR-002 completes only when all such cells and physical gates are complete.
- **Validation:** Review the PAR-001 matrix, CAP-001 contract evidence, AXIS-001 through AXIS-004 implementation/emulation evidence, and every required HW-003 through HW-013 report; reject closure for any missing cell, parity result, placement/lifecycle result, or physical gate.
- **Hardware:** The union of hardware required by AXIS-001 through AXIS-004 and HW-003 through HW-013; no additional family-specific hardware scope is owned here.
- **Evidence:** Pending

### AXIS-001 Implement Standard-Attention PP/TP Cells

- **Status:** blocked
- **Dependencies:** CAP-001, COR-002, STEP-004
- **Goal:** Implement PP and TP cells for standard-attention LLaMA, plain Qwen, and Qwen2-VibeThinker families only.
- **Acceptance criteria:** Each admitted standard-attention PP/TP cell has mesh-derived placement and ownership, deterministic emulated parity, and implementation-level lifecycle/reset/abort/multi-turn/unload coverage; production support additionally requires HW-003 for PP and HW-006 for TP.
- **Validation:** Run per-family loader/dispatch selection, emulated PP/TP placement and deterministic parity, reset/abort/multi-turn/unload, and allocation-inventory tests. This task owns implementation and emulated parity only; physical evidence is owned by HW-003 and HW-006.
- **Hardware:** A supported AMD GPU for implementation and emulated validation; physical production closure is HW-003 and HW-006.
- **Evidence:** Pending

### AXIS-002 Implement Qwen35 TP And MoE EP Cells

- **Status:** blocked
- **Dependencies:** CAP-001, GEN-001
- **Goal:** Implement Qwen35 dense and MoE TP cells and Qwen35-MoE EP cells only.
- **Acceptance criteria:** Qwen35 dense and MoE TP and Qwen35-MoE EP have mesh stage/state or expert ownership, deterministic emulated parity, and lifecycle/reset/unload coverage; this task explicitly does not implement PP, which GEN-001 owns, and does not implement dense EP, which CAP-001 normalizes without an EP support claim; production closure requires HW-007 and HW-011.
- **Validation:** Run Qwen35 dense/MoE TP and Qwen35-MoE EP selection, emulated placement, deterministic parity, reset, lifecycle, unload, and refusal tests. Do not add PP implementation or dense-EP behavior here; physical evidence is owned by HW-007 and HW-011.
- **Hardware:** A supported AMD GPU for implementation and emulated validation; physical production closure is HW-007 and HW-011.
- **Evidence:** Pending

### AXIS-003 Implement Additional Non-Qwen35 PP/TP/EP Cells

- **Status:** blocked
- **Dependencies:** CAP-001, COR-002, STEP-002, STEP-004
- **Goal:** Admit the dense LFM2 single path and implement its PP/TP cells, plus LFM2-MoE PP/TP/EP, Cohere2-MoE PP/TP/EP, and the planned PP/TP cells for DeepSeek4 and MiniMax.
- **Acceptance criteria:** Dense LFM2 is currently refused but has planned single-path admission and PP/TP ownership here; its admitted cells have mesh placement, deterministic emulated parity, reset/lifecycle/unload, and explicit refusal coverage. LFM2-MoE and Cohere2-MoE may additionally admit EP; dense LFM2 `EP>1` normalizes to single via CAP-001 and is not an EP cell. Existing DeepSeek4 and MiniMax EP remains owned by STEP-002 and physically closed by HW-001/HW-002; newly enabled production cells require HW-008, HW-009, and HW-010.
- **Validation:** Run dense LFM2 single-path admission and PP/TP selection/refusal, emulated placement, deterministic parity, reset, lifecycle, and unload tests; run LFM2-MoE and Cohere2-MoE PP/TP/EP selection/refusal, emulated placement, deterministic parity, reset, lifecycle, and unload tests; run the planned DeepSeek4/MiniMax PP/TP coverage without reimplementing their existing EP. Physical evidence for newly enabled cells is owned by HW-008/HW-009/HW-010.
- **Hardware:** A supported AMD GPU for implementation and emulated validation; physical production closure is HW-008, HW-009, and HW-010, while existing DeepSeek4/MiniMax EP uses HW-001/HW-002.
- **Evidence:** Pending

### AXIS-004 Implement Vision-Family PP/TP Cells

- **Status:** blocked
- **Dependencies:** CAP-001, VL-001, VL-002, STEP-004
- **Goal:** Implement PP and TP cells for Qwen35-VL and dots.ocr only.
- **Acceptance criteria:** Each admitted vision PP/TP cell preserves image isolation and text parity and has deterministic emulated PP/TP, reset, abort, and unload coverage; dense EP normalization is CAP-001 behavior and outside AXIS-004 scope; production closure requires HW-012 and HW-013.
- **Validation:** Run image-isolation and text-parity fixtures, deterministic emulated PP/TP placement/parity, reset/abort/unload, and admitted PP/TP dispatch tests. This task owns implementation and emulated parity only; dense EP normalization is not tested or implemented here; physical VL topology evidence is owned by HW-012/HW-013.
- **Hardware:** A supported AMD GPU for implementation and emulated multimodal validation; physical production closure is HW-012 and HW-013.
- **Evidence:** Pending

### COMP-001 Gate Optional TP x EP Composition

- **Status:** complete
- **Dependencies:** None
- **Goal:** Make an unconditional scope decision for TP x EP composition in this refactor.
- **Acceptance criteria:** Record one decision: either TP x EP is out of scope and `TP>1 && EP>1` is explicitly rejected, or a concrete deployment requirement names the model, topology, owner, and measurable success target. In the latter case, create a new conditional COMP task for design/implementation/physical validation; COMP-001 itself completes when the decision and refusal-or-follow-up are recorded and never waits on implementation or hardware.
- **Decision:** TP×EP is out of scope for this refactor. Requests with `tp > 1 && ep > 1` are explicitly refused at every boundary.
- **Validation:** Review the requirement record and support matrix; for the out-of-scope decision, run configuration/refusal tests; for the required decision, verify the new follow-up ID exists with dependencies and acceptance criteria.
- **Hardware:** None
- **Evidence:** Implementation in working tree: `config::validate_parallel_axes` rejects the pair with a `Result<(), String>`, the daemon emits its JSON error envelope before the EP-wins remap, and `preflight_manifest` returns a `FulfillError` for a composed Tp×Ep mesh; `cargo test -p hipfire-runtime --lib config::tests` (11 passed), `cargo test -p hipfire-runtime --example daemon load_rejects_tp_ep_before_ep_wins_remap` (1 passed), `cargo test -p hipfire-runtime --lib weight_store::tests` (12 passed); PR #527 mirror synchronized.

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
- **Dependencies:** HW-001, HW-002, HW-003, HW-004, HW-005, HW-006, HW-007, HW-008, HW-009, HW-010, HW-011, HW-012, HW-013, COR-001, COR-002, COR-003, COR-004, COR-005, COR-006, GEN-001, GEN-002, SPEC-001, SPEC-002, SPEC-003, SPEC-004, VL-001, VL-002, STEP-001, STEP-002, STEP-003, STEP-004, PAR-001, CAP-001, PAR-002, AXIS-001, AXIS-002, AXIS-003, AXIS-004, COMP-001, DOC-001
- **Goal:** Establish that the completed refactor is correct, production-honest, documented, and ready to merge.
- **Acceptance criteria:** Every listed dependency and every conditional follow-up created by COMP-001 is `complete` with evidence; every row in the Final Validation Matrix passes against its named fixture/oracle; HW-001 through HW-013 meet the 64 MiB/no-monotonic-growth thresholds; no stale active checklist conflicts with this tracker; PR #527 mirrors all IDs, required CI checks pass, and no blocking review finding remains.
- **Validation:** Execute and archive every row in the Final Validation Matrix, rerun tracker schema and documentation-link checks, inspect the final branch diff and PR checks/reviews, and attach the physical PP/TP/EP reports with artifact/prompt digests and per-cycle VRAM.
- **Hardware:** The union of hardware required by HW-001 through HW-013 and each supported model-family validation cell.
- **Evidence:** Pending

## Terminal lifecycle migration matrix

COR-003 establishes the terminal lifecycle contract; this matrix is mandatory
for every remaining and future architecture. COR-003 completion does not close
the architecture migrations listed here or the separate physical PP/TP/EP
tasks (`HW-001` through `HW-013`). CAP-001 supplies shared capability
plumbing, AXIS-001 through AXIS-004 own concrete implementation, and the
named hardware tasks own production closure. A row is not production-ready
until its normal-finalization and discard/reset evidence exists for the named
driver. `VL-001` and `VL-002` are the downstream multimodal adopters. Generic
AR/spec adoption belongs to `SPEC-001`; native Qwen MTP generally uses
`SPEC-002` (its in-flight cancellation is implemented), PP-specific MTP
integration belongs to `SPEC-004`, while transactional Qwen target loading
remains deferred to `SPEC-003`.

| Architecture | Driver entry point / owner | Normal finalization | Abort/error discard/reset | Forced/injected EOS | Cache/cross-turn isolation | Required focused evidence | Unsupported/refused mode |
|---|---|---|---|---|---|---|---|
| DeepSeek4 | `ArchDispatch` DeepSeek4 AR/MTP adapters; `GEN-002`, `SPEC-001`, `CAP-001`, `AXIS-003` | Bespoke AR and native MTP emit pending output exactly once on normal completion. | DeepSeek AR discard resets request state and zeros decode cache; native MTP cancellation restores guards/PBS and resets before terminal envelope. | Carrier/model EOS remains distinct from user stop and does not finalize early. | No discarded turn enters assistant cache; reset prevents decode-cache or turn residue. | Production-owned cancellation/reset tests, DeepSeek AR normal/discard tests, MTP coherence, HW-001 EP evidence, and HW-008 newly enabled PP/TP evidence. | Qwen-style DFlash is not a DeepSeek4 mode; CAP-001 must refuse unsupported combinations explicitly. |
| MiniMax | MiniMax `ArchDispatch`/Step adapter; `STEP-002`, `STEP-004`, `CAP-001`, `AXIS-003` | Shared lifecycle finalizes parser/emitter once after normal AR completion. | Abort/error drops pending output, resets request-owned state, and emits no late event. | Carrier EOS injection must remain non-terminal until the actual terminal outcome. | Multi-turn cache and parser state must be request-local and reset on discard. | MiniMax tool/stream, abort, reset, unload, and Tokyo-then-Germany multi-turn fixtures; HW-002 EP evidence and HW-009 newly enabled PP/TP evidence. | CAP-001 must refuse unsupported speculative or parallel cells rather than silently fall back. |
| LFM2 | LFM2 architecture adapter and `ArchDispatch`; `CAP-001`, `AXIS-003`, `STEP-004` | Shared generic AR finalization once AXIS-003 admits the dense LFM2 single path or an LFM2-MoE path. | Discard/reset must clear parser and request state even when the current model is refused. | No forced EOS may turn an unsupported or incomplete LFM2 path into a successful terminal response. | No cache reuse across refused, aborted, or reset requests. | Current dense LFM2 refusal characterization; planned single-path admission, PP/TP emulated parity, LFM2-MoE/Cohere2-MoE PP/TP/EP lifecycle tests, fixed multi-turn fixture after loader support lands, and HW-010 per-family physical evidence. | Dense LFM2 single/PP/TP is currently refused pending CAP-001 and AXIS-003; dense LFM2 `EP>1` normalizes to single via CAP-001; LFM2-MoE/Cohere2-MoE EP remains planned until AXIS-003/HW-010 closure. |
| Qwen35 | Qwen35 `ArchDispatch`/AR, PP, TP, and native MTP adapters; `GEN-001`, `CAP-001`, `AXIS-002`, `SPEC-001`, `SPEC-002`, `SPEC-004` | Generic AR/spec and Qwen PP use the sealed boundary and finalize exactly once; native MTP cancellation is implemented, while PP+MTP termination is required under `SPEC-004`. | Sealed Qwen turns discard beyond the boundary, reset/cache-invalidate as required; MTP cancellation restores guards and resets before abort/error. | Injected EOS remains pre-commit; carrier framing and user stops remain separate. | Only the sealed turn may feed replay/fingerprint/cache; uncacheable cuts force reset/cold next turn. | Qwen AR/spec parser and sealed-turn tests, GEN-001 dense+MoE PP reset tests, AXIS-002 TP/EP lifecycle tests, HW-004 dense+MoE PP, HW-007 TP, HW-011 MoE EP, native MTP lifecycle tests, fixed PP+MTP fixture/path, and coherence gate. | `SPEC-003` transactional target loading is deferred; CAP-001 handles planned/unsupported capability outcomes, and dense EP is normalized to `normalized-to-single(CAP-001)` without an EP support claim. |
| Qwen35-VL | Qwen35-VL image prefill owner plus shared post-prefill AR lifecycle; `VL-001`, `CAP-001`, `AXIS-004` | Vision prefill remains architecture-owned; shared post-prefill AR finalizes once. | Abort/error discards parser and image/request state, then resets before the terminal envelope. | Image/carrier framing and injected EOS must not bypass the shared terminal policy. | Image state, parser state, and any cache are request-local; different-image turns cannot reuse discarded state. | Canonical image-plus-text, different-image isolation, abort/reset, text-only parity, HW-012 PP, and HW-013 TP physical VL evidence. | Dense EP normalization is CAP-001 behavior and outside AXIS-004 scope; VL target/draft/native-MTP speculation is refused until a model-specific quality fixture and follow-up exist. |
| dots.ocr | dots.ocr vision/prompt-framing owner plus shared post-prefill AR/n-gram lifecycle; `VL-002`, `CAP-001`, `AXIS-004` | Custom image framing and prefill finish, then shared AR/n-gram output finalizes once. | Discard/reset clears parser and image state; no OCR/tool output follows abort/error. | Custom framing remains distinct from injected EOS and must not finalize twice. | Image state and OCR/cache state are request-local across repeated images and turns. | Canonical `dots_ocr_smoke_001_vllm.json`/demo-image F1, AR/n-gram parity, repeated-image isolation, abort/reset, unload, HW-012 PP, and HW-013 TP physical evidence. | Dense EP normalization is CAP-001 behavior and outside AXIS-004 scope; target/draft and native-MTP VL speculation is explicitly refused pending a dots.ocr quality oracle. |
| Future architecture onboarding | New architecture owner with `ArchDispatch`; adopt `GEN-*`, `SPEC-*`, `VL-*`, and `STEP-004` as applicable | Implement the shared normal-completion epilogue before claiming support. | Implement explicit discard/reset before any parser, cache, or GPU state is published. | Declare carrier/model EOS versus user stop and test injected EOS as pre-commit where applicable. | Name every cache/state owner; prove no discarded or prior-turn state crosses the boundary. | Add focused terminal lifecycle tests, deterministic parity, refusal tests, and required model/coherence/multi-turn hardware evidence before adding a support cell. | Every unsupported axis, speculative mode, or loader path must return a documented refusal; no silent fallback. |

## Final Validation Matrix

DOC-002 cannot complete from a generic “tests pass” statement. Its evidence must enumerate these rows with exact command, commit, fixture digest, result, and report path:

| Area | Required fixture or oracle | Pass condition |
|---|---|---|
| Workspace | `cargo build --workspace --features hipfire-runtime/deltanet` plus workspace tests with the same required feature set | Exit 0 and zero test failures. |
| DeepSeek4 EP | HW-001 pinned model/prompt; peer `ep_decode_parity` committed-token hash | RCCL hash identical to peer oracle; four-cycle VRAM threshold passes. |
| MiniMax EP | HW-002 pinned model/prompt; capital/code, LCP, and Tokyo-then-Germany peer hashes | Every RCCL hash identical to its oracle; four-cycle VRAM threshold passes. |
| Dense PP | `qwen3-0.6b-llama.mq4`, `llama_store_pp`, 311 tensors, established `max |delta| = 0` oracle | Physical PP preserves zero logit delta, 14/14 layer placement, and HW-003 VRAM threshold. |
| Qwen35 PP | HW-004 pinned Qwen35 dense+MoE model/prompt fixtures and captured single-device cold/two-turn hashes | Physical dense and MoE PP hashes identical, placement inventory exact, and HW-004 VRAM threshold passes. |
| TP teardown | HW-006, HW-007, HW-008, HW-009, HW-010, and HW-013 per-family TP physical reports aggregated by HW-005 | Every per-family TP report has pinned fixtures, physical parity/placement/lifecycle evidence, no live stream/communicator, and the aggregate HW-005 threshold passes. |
| Reset | COR-002 reset-contract tests and `serve-multiturn-gate.sh` across Single, PP, TP, EP, spec/MTP, recurrent/conv, and VL adapters | Every adapter proves the central contract; abort, overflow, reset-command, and normal-completion cases pass. |
| Parser | COR-003 pending UTF-8/reasoning/tool-call/injected-EOS/stop/budget/abort fixtures | Final output is emitted exactly once with no cross-turn residue. |
| AR/spec | Canonical DFlash fixtures from `scripts/coherence-gate-dflash.sh`; fixed PP+MTP fixture/path, prompt, and binary hashes | Coherence gate passes; accepted-token accounting, MTP/DFlash selection/refusal, MTP-disabled AR parity, and PP+MTP fallback tests pass. |
| VL | Canonical Qwen35-VL fixture captured by VL-001; dots.ocr canonical `dots_ocr_smoke_001_vllm.json`/demo image oracle | Qwen35-VL AR parity/reset passes; dots.ocr preserves its recorded F1 oracle in AR and n-gram modes; unsupported VL speculation rejects explicitly. |
| Step/manifest | Architecture inventory produced by STEP-004 with one pinned parity fixture per registered decoder family | Every existing supported forward cell has manifest coverage and deterministic Step parity, or a documented non-decoder exception. |
| Axis matrix | PAR-001 policy matrix/current-refusal characterization, including dense LFM2 current refusal with planned AXIS-003 ownership and `normalized-to-single(CAP-001)` for dense EP; CAP-001 capability contract, AXIS-001 through AXIS-004 implementations, and HW-003 through HW-013 named closure reports | Every cell selects the documented path, records the documented refusal with its planned owner where applicable, or records dense EP as `normalized-to-single(CAP-001)`; every supported multi-GPU cell has the named physical evidence; dense EP creates no EP support claim. |
| Documentation/PR | DOC-001 named-document list, tracker schema check, PR #527 checklist | Every stale document links here, IDs/fields validate, PR IDs match, and required CI/reviews are green. |

## Dependency Order

1. HW-001 and HW-002 wait for STEP-002 so physical RCCL validation exercises the final Step/manifest EP path. HW-003 and HW-006 wait for AXIS-001; HW-004 waits for GEN-001, which owns dense and MoE Qwen35 PP; HW-007/HW-011 wait for AXIS-002; HW-008/HW-009/HW-010 wait for AXIS-003 and STEP-004, including dense LFM2 admission and its PP/TP cells; HW-012/HW-013 wait for AXIS-004 and the VL lifecycle adopters; HW-005 aggregates HW-006, HW-007, HW-008, HW-009, HW-010, and HW-013.
2. COR-001, COR-003, COR-004, COR-005, COR-006, PAR-001, STEP-001, and DOC-001 are independent starting points.
3. COR-004 feeds COR-002; COR-001 through COR-003 feed SPEC-001; SPEC-001 feeds SPEC-002; GEN-001, SPEC-002, and SPEC-003 feed SPEC-004; VL-001 depends only on COR-002 and COR-003. SPEC-003 remains deferred by priority and, together with SPEC-004, blocks DOC-002.
4. STEP-001 feeds STEP-003; PAR-001 feeds STEP-002; STEP-001, STEP-002, STEP-003, and PAR-001 feed STEP-004.
5. COR-002 plus STEP-001/STEP-002/STEP-003 feed GEN-001; GEN-001 owns dense and MoE Qwen35 PP and feeds physical Qwen35 validation HW-004.
6. PAR-001 owns policy and current-refusal characterization; CAP-001 owns the shared capability contract and dense-EP normalization; AXIS-001 through AXIS-004 own concrete family implementations and emulated parity; HW-003 through HW-013 own named physical closure; PAR-002 aggregates their closure without family-specific code. COMP-001 independently decides TP x EP scope; if required, it creates a conditional COMP implementation follow-up with its own dependencies.
7. DOC-002 is the only final closure task and cannot complete while any dependency is open.

## Parallel Streams

- **Physical validation:** HW-001 and HW-002 follow STEP-002; HW-003/HW-006 follow AXIS-001; HW-004 follows GEN-001 for dense and MoE Qwen35 PP; HW-007/HW-011 follow AXIS-002; HW-008/HW-009/HW-010 follow AXIS-003 for planned non-Qwen35 cells including dense LFM2 admission; HW-012/HW-013 follow AXIS-004; HW-005 follows its six aggregate TP reports.
- **Correctness ownership:** COR-001, COR-003, COR-004, COR-005, and COR-006 initially; COR-002 follows the eviction decision.
- **Generation/spec:** GEN-002 can proceed alongside SPEC-001; SPEC-002 follows metadata and shared orchestration; SPEC-003 remains deferred by priority; SPEC-004 follows GEN-001, SPEC-002, and SPEC-003 for PP+MTP integration.
- **Multimodal:** VL-001 follows only COR-002/COR-003 and can proceed independently of SPEC-001; VL-002 waits for SPEC-001 because it adopts the existing n-gram strategy through shared speculative orchestration.
- **Execution/placement:** STEP-001 and PAR-001 can start together; STEP-002 and STEP-003 then proceed largely independently before STEP-004; CAP-001 follows PAR-001, and AXIS implementation follows CAP-001 plus its named architecture prerequisites.
- **Documentation:** DOC-001 can proceed without runtime or hardware work.

## Update Protocol

1. After the one-time bootstrap correction recorded in the immutable Task ID Migration table, task IDs must never be renamed, renumbered, or reused. Add a new stable prefixed ID for newly discovered work.
2. Before implementation, set only the selected task to `in progress`; record a newly discovered blocker or follow-up immediately.
3. Do not set `complete` until every acceptance criterion and validation item passes. Emulation never closes a physical-hardware criterion.
4. Replace `Evidence: Pending` with commit hashes, exact commands/results, hardware topology, GPU architecture, ROCm/RCCL versions, and artifact/report links as applicable. Use `None` only when evidence or a field genuinely does not apply.
5. Update this tracker in the same change that alters task status. After pushing, synchronize the matching checklist IDs in PR #527.
6. If the PR and tracker diverge, correct the PR mirror; never edit historical evidence to manufacture agreement.
