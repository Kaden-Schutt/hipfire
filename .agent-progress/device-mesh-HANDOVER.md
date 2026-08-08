> **Current handover — 2026-08-04 (harness-evidence boundary).** This is a
> self-contained handover for a new session. The authoritative task status
> remains
> [device-mesh-refactor-tracker.md](device-mesh-refactor-tracker.md); this file
> records the current working-tree stopping point (HEAD `ea5d76fa`; the
> test-only EP2 harness lane is uncommitted — see the working-tree snapshot)
> and the next implementation/evidence boundary.

# Device-mesh / STEP-002 handover

## Current goal

Complete **STEP-002 — Adopt Step/Manifest for MoE** for the all-family MoE
surface: DeepSeek4, MiniMax, and Qwen35 contracts must use the shared
manifest/mesh/dispatch vocabulary for routed-expert ownership, routing,
zero/dummy handling, and collectives.

STEP-002 remains `ready` / in progress, **not complete**. The Qwen35
single-device HFQ Frozen MoE residency cutover is now **committed**: the
checkpoint `60b7f62a` ("feat(weight-store): adopt frozen MoE residency") plus
the cleanup/review series through HEAD `ea5d76fa` are on `feature/device-mesh`
(ahead of origin by 18; the test-only EP2 harness lane is uncommitted — see
the working-tree snapshot). The final full review is
APPROVED; the full test suite passes 1865 / 66 ignored; affected clippy exits
0 with zero diagnostics on post-checkpoint changed lines; the residency
boundary script normal + self-test both pass.

Fresh-GPU dense smoke (gfx1151 AMD Radeon 8060S, HIP 7.2.53211 / ROCm
toolchain 7.2.3) passed: DFlash coherence report
`/tmp/coherence-dflash-20260803-154705.md` (4/4 OK, no hard/soft flags,
outputs eyeballed coherent) and serve multi-turn report
`/tmp/serve-multiturn-20260803-155040.md` (4/4 AR + 4/4 DFlash coherent).
**These prove dense-engine regression safety only** — they do NOT touch
STEP-002 MoE/Frozen acceptance.

The canonical Qwen35-MoE fixture is user-authorized and the test-only
emulated-EP2 parity evidence is now complete (see the closure matrix and
runbook below); multi-cycle lifecycle/VRAM closure is explicitly deferred to
STEP-002R, which remains accepted open debt. Qwen35
production EP remains a planned, refused-before-allocation capability owned
by AXIS-002 (physical closure HW-011); STEP-002 does not make it
production-ready.

## Working-tree snapshot

HEAD is `ea5d76fa` on `feature/device-mesh` (ahead of
`origin/feature/device-mesh` by 18). The Phase A/B/C implementation set is
committed: `60b7f62a` (frozen MoE residency checkpoint) followed by the
docs/cleanup/review series through `ea5d76fa` (TP-refusal preservation,
scoped clippy cleanup, frozen-owner surface cleanup, seam truthification,
and the final seam-asserting OOB/unknown-dtype/abort tests).

The **Qwen35 TEST-ONLY emulated-EP2 parity harness lane is currently
UNCOMMITTED work** — do not lose it, and do not commit it without the
appropriate implementation review:

- modified: `crates/hipfire-arch-qwen35/src/{lib.rs,qwen35.rs,store.rs}`,
  `crates/hipfire-arch-qwen35/Cargo.toml`, `crates/hipfire-runtime/Cargo.toml`,
  `scripts/check_moe_residency_boundary.sh`, `.agent-progress/run-ep-parity.sh`
  (plus this tracker/handover);
- new (untracked): `crates/hipfire-arch-qwen35/src/ep2_harness.rs`,
  `crates/hipfire-arch-qwen35/src/store/store_ep2.rs`,
  `crates/hipfire-runtime/examples/ep_decode_parity.rs`,
  `benchmarks/prompts/qwen35_moe_ep_parity.txt`;
- local evidence logs (gitignored, not tracked):
  `.agent-progress/ep-parity-confirm-probe-{1..5}.log`,
  `.agent-progress/ep-parity-confirm-accept.log`,
  `.agent-progress/ep-parity-confirm-structural.log`,
  `.agent-progress/ep-parity-confirm-cleanup.log`,
  `.agent-progress/ep-production-ar-confirm.log`.
  The earlier `ep-parity-final-*` probe/accept logs and
  `ep-production-ar-smoke.log` are superseded by the `confirm-*` set (the
  binary was regenerated after the checked-cleanup fix; the AR smoke was
  re-measured at 5.1 tok/s).

Unrelated untracked paths remain untouched: `crates/graphify-out/`,
`graphify-out/`, `docs/pr-dspark-qwen3.md`, `docs/pr-dspark-qwen35.md`.

Do not reset, clean, or reformat this worktree. Do not use git
checkout/restore/reset/stash here: two destructive recovery incidents in this
history (2026-07-26 and 2026-07-27, recorded below) erased unstaged work
irrecoverably. The rule for every delegated writer is **no
checkout/restore/reset/stash**, and future lanes must be smaller than the
failed all-common-transaction assignment. Do not touch the unrelated
untracked files listed above.

## Completed / approved work

### Tracker contract

The STEP-002 tracker row now makes acceptance explicit: permanent WeightStore
ownership for routed-expert placements and derived resources; private
read-only typed projections; origin-enforcing rank-branded allocation tokens;
and rejection of raw-pointer `WeightStoreView` values. It also records the
Qwen refusal invariant and preserves the canonical Single-vs-emulated-EP2 gate.
The tracker remains the status authority; its STEP-002 `Evidence` is still
`In progress`.

### Manifest contracts

The generic manifest layer is present in
`crates/hipfire-runtime/src/weight_manifest.rs`:

- `ShardPolicy`, including `ExpertSharded` and `ExpertTensorSharded`;
- `WeightEntry` / `StateEntry` and placement validation;
- `collective_for_policy`, `layer_collectives`, and `placement_devices`;
- deterministic `plan_manifest` / `ManifestPlan`; and
- validation of expert shape, shard policy, placement, and collective
  contracts.

`crates/hipfire-runtime/src/weight_store.rs` fulfills whole tensors and
expert-compact placement through the existing generic path, with projection
metadata for static, compact-expert, column, and row placements. This is
manifest/placement foundation work, not the selected frozen-store ownership
implementation.

### Dispatch contracts

`crates/hipfire-dispatch/src/families/moe.rs` now centralizes the MoE dispatch
vocabulary and resolution boundary: `MoeDtypes`, `MoeResolution`, typed MoE
parameter records, prefill resolution, and `MoeFamily` routing. The loader's
`MoEExecutionPolicy` in
`crates/hipfire-loader/src/model_parallel.rs` validates that the effective
named mesh axis matches Single, TP, or EP and rejects competing TP×EP axes.

These are dispatch contracts and policy seams. They do not prove permanent
resident ownership or Qwen production admission.

### Failed ExpertShard ownership removal

The unfinished `ExpertShardResourceKind`, `ExpertShardResource`,
`ExpertShardResident`, `ExpertShardAssembly`, `ExpertShardTarget`, and
`ExpertShardSlot` layer has been removed from tracked Rust sources. The
pointer-table/dummy lifetime fixes in the current DS4, MiniMax, and Qwen diffs
avoid the former `mem::forget` leak path by threading the dummy allocation into
the per-layer owner.

This removal is an approved **reset boundary**, not proof that the new hybrid
store exists. The old ownership model must not be reconstructed.

## Rejected ownership approaches

1. **Architecture-owned `ExpertShardResident` / resource assembly.** Rejected
   because it split ownership between generic placement and architecture
   structs, made partial-rank rollback ambiguous, and invited leaks/double
   frees when pointer tables and dummy buffers outlived their source records.
2. **`WeightStoreAuxiliary` plus `WeightStoreView`.** Rejected because it added
   a second ownership vocabulary, allowed raw-pointer descriptors to outlive
   their actual owner, and made typed extraction/freeing look valid without a
   lifetime or origin proof. `WeightStoreView` is explicitly non-accepting,
   even when it is described as “non-owning.”
3. **Raw `take`/replacement from a mutable store.** Rejected for the final
   architecture because it turns cell identity into temporal mutation and
   lets architecture assembly silently become an owner.
4. **Launch leases in this slice.** Deferred. Do not add a lease abstraction
   while resetting residency ownership; kernel launch lifetime/argument leases
   require a separate contract and are not needed to establish the store
   ownership invariant.

## Selected hybrid architecture

The authoritative design is
`docs/superpowers/plans/2026-07-22-weight-store-moe-residency-recovery.md`:

- `WeightStoreAllocation` is a non-forgeable, non-cloneable, rank-branded
  free authority containing origin mesh epoch, logical rank, physical device,
  and pool epoch. A fallible free consumes the token on success and returns
  the original token with the error on failure.
- `WeightStoreBuilder` owns all staged allocations until freeze.
  `FrozenWeightStore` owns one immutable cell arena keyed by opaque
  `WeightCellId` values. Original routed placements, pointer tables, dummy
  buffers, dtype/layout metadata, and shared sidecars are store cells; there
  is no auxiliary owner.
- Alias resolution happens at freeze. After freeze there is no cell `take`,
  replacement, mutable lookup, or transfer of ownership.
- Qwen35, DeepSeek4, and MiniMax keep private typed read-only projections of
  IDs and aliases. A forward borrows bindings from the frozen owner; the
  projection cannot extract tensors, clone raw views, or free typed weights.
- The loader owns the builder during construction and publishes exactly one
  frozen store into `LoadedModel.weight_store` or `EpState`. Unload consumes
  that same owner. Architecture teardown frees only architecture-owned scratch
  and state.
- **Launch leases are deferred.** The selected architecture uses borrowed
  bindings for this migration and does not claim a launch-lease solution.

## Mandatory Qwen35 acceptance invariant

The canonical gate is preserved exactly in intent and must remain in the next
session's acceptance evidence:

- use the pinned canonical Qwen35-MoE 35B fixture;
- record model SHA-256, prompt MD5, binary digest, exact command, and topology;
- the emulated EP harness uses **Single as the sole baseline**; EP=1 is its
  alias and is not a second required run;
- prefill parity is finite final-prefill logits with identical argmax plus
  the first token emitted after prefill (bitwise-exact logits are NOT
  required — an honest FP32 partition reduction changes addition order);
- decode parity is exact generated token IDs, with reset and multi-turn
  behavior; and
- report the first logit divergence if tokens differ.

The explicit negative invariant is equally mandatory: throughout STEP-002,
Qwen35Moe EP remains `Planned`/refused before allocation. No emulated test may
construct `EpArch::Qwen35`; no daemon Qwen EP admission may be added; and
**AXIS-002 is the sole Qwen admission owner**. HW-011 owns physical closure
only after AXIS-002 admits the cell.

## STEP-002 closure matrix (updated 2026-08-06, HEAD `ea5d76fa` + uncommitted migration/harness lanes)

| Criterion | Status | Evidence / gap |
|---|---|---|
| Qwen35 single-device HFQ Frozen MoE residency cutover (facade, ID-only projection/bindings, direct Frozen staging, exact C2 admission, source-bound preflight/Legacy fallback, checked published/unpublished unload/backlog) | ✅ Satisfied | Oracle Gate B APPROVED; committed at `60b7f62a` + review series to `ea5d76fa`; residency boundary script asserts every invariant |
| Final implementation review | ✅ Satisfied | Full review APPROVED on `ea5d76fa` |
| Full test suite | ✅ Satisfied | 1865 passed / 66 ignored |
| Affected clippy (post-checkpoint changed lines) | ✅ Satisfied | exit 0, zero diagnostics |
| Residency boundary normal + self-test | ✅ Satisfied | both exit 0 (self-test: 44 induced failures, all 32 violation categories caught); runner self-test passes |
| Dense-engine regression safety (smoke only) | ✅ Satisfied — NOT STEP-002 acceptance | coherence 4/4 OK + serve multi-turn 4/4 AR + 4/4 DFlash; binary md5 `22d547fa6a3bffd137279639f6ac701a` |
| Canonical Qwen35-MoE 35B fixture pinned/authorized | ✅ Satisfied | user-authorized reuse of `~/.hipfire/models/qwen3.6-35b-a3b.mq4` (22,855,051,520 bytes; MD5 `edde51ec1dac0f2bd42cff5ef1cb8944`; SHA-256 `1dc1c7964de415e0040a540a4300b9518e11b00c13d99c23f576f2b9fe1e8bca`); no dedicated copy required |
| Test-only Qwen35-MoE emulated-EP2 parity harness | ✅ Satisfied | single-shot harness exists (`run-ep-parity.sh` + `ep_decode_parity` example + `ep2_harness`/`store_ep2` modules, uncommitted); no `EpArch::Qwen35`, no daemon admission; current probe exits 0 after the Qwen-local logical-view repair; boundary normal + self-test pass on the harness tree |
| Single-as-sole-baseline parity under honest FP32 contract | ✅ Satisfied | prompt `benchmarks/prompts/qwen35_moe_ep_parity.txt` (MD5 `1aacd3c05cf9695cc799acc59581938d`; currently untracked/uncommitted with the harness lane — never committed); parity binary SHA-256 `f4b82b109e779f8518332dd86e31371e9a46a5cf50c0ec87360d3a75c95dbd6f`; exact final-prefill argmax, first token, all generated token IDs, finite logits, exact second-turn/reset flags, Q8 resolved, no first divergence; five fresh confirm probes bit-stable at Dmax = 0.7081642, pinned tolerance `0.708164275` (next representable f32, one ULP; the accept log may print the shortened f32 `0.7081643`); identical token vector `[13, 198, 760, 6511, 314, 9338, 369, 11751, …]` both sides; `--accept` PASSED; logs `.agent-progress/ep-parity-confirm-{probe-1..5,accept}.log`; `.agent-progress/ep-parity-confirm-{structural,cleanup}.log` each 1 passed, exit 0; full feature suite 442 passed / 21 ignored |
| Graph-enabled generate/unload/reload lifecycle + VRAM recovery (≤64 MiB of post-first-unload baseline; no monotonic growth cycles 2–4) | ⏳ Deferred to STEP-002R (NOT satisfied) | explicit user decision: lifecycle example/runner removed; harness is one GPU-bearing invocation per process — success-path pool drain proven, but partial-construction rollback, typed load-error retention, and failed-free retention NOT claimed |
| Frozen-cutover DFlash coherence | ⏳ Out of scope for this harness (NOT claimed) | no paired A3B draft / canonical prompt defined; no DFlash gate claimed or required; scoped AR-only smoke passed instead: `The capital of **France** is **Paris**.<|im_end|>`, 15 tokens, 5.1 tok/s (`.agent-progress/ep-production-ar-confirm.log`). The 2026-08-04 DFlash regression gate `/tmp/coherence-dflash-20260804-122132.md` (4/4 OK, no hard errors, no soft/tier3 warnings) uses the separate canonical qwen3.6-27b target/draft — a production regression gate, NOT direct A3B DFlash evidence |
| Required GPU harness tests | ✅ Satisfied | warmed success-path state free + checked pool drain within one 4096-byte page; `frozen_moe_resident_ep2_build_bind_rank_tables_and_canonical_unaffected` passed |
| Final narrow evidence | ✅ Satisfied | focused EP2 qwen tests 62 passed / 2 ignored; dispatch-tests qwen35 19 passed; loader parallel capability 18 passed |
| DS4/MiniMax accepted Single behavior + named structural EP examples (`ep_deepseek4`, `ep_minimax`) | ✅ Satisfied | migrated-tree peer-direct emulated EP2 runs passed on gfx1151/HIP 7.2 with unchanged pins: DS4 `0x26a13602bedf9926`, MiniMax `0x887c2e7717e9c3bf`; decoded outputs coherent. Full physical RCCL parity remains non-blocking (HW-001/HW-002) |
| Task 8 Step 3 production/common-plan migration | ✅ Satisfied | Qwen, MiniMax, and DeepSeek use model-owned cached manifest authority and caller-owned execution policy; local ExpertGroupPlan fabrication is deleted; family spec/quality reviews passed |
| STEP-002R origin-preserving rollback | ⏳ Accepted open debt | not closed; exact failed-free retention not claimed; owns the deferred lifecycle/VRAM closure |

## Evidence-collection runbook (ordered)

Goal: preserve the completed STEP-002 evidence. Steps 1–3 and 6 are DONE and
recorded; step 4 is explicitly deferred to STEP-002R and step 5 is scoped out.
Final Oracle Gate 3 returned `GATE3_APPROVED_WITH_DEFERRED_DEBT` on 2026-08-06.

Preconditions for every step: fresh process; record GPU model (gfx1151, AMD
Radeon 8060S), HIP 7.2.53211 / ROCm toolchain 7.2.3, and the artifact digests
listed. The canonical fixture IS the user-authorized
`~/.hipfire/models/qwen3.6-35b-a3b.mq4` (see step 1) — no other artifact.

1. **Pin/authorize the canonical Qwen35-MoE 35B fixture** — DONE (2026-08-04).
   The user explicitly authorized reuse of the existing
   `~/.hipfire/models/qwen3.6-35b-a3b.mq4` artifact; no dedicated copy is
   required. Recorded: model size 22,855,051,520 bytes; MD5
   `edde51ec1dac0f2bd42cff5ef1cb8944`; SHA-256
   `1dc1c7964de415e0040a540a4300b9518e11b00c13d99c23f576f2b9fe1e8bca`;
   gate prompt `benchmarks/prompts/qwen35_moe_ep_parity.txt`
   (bytes `The capital of France is\n`, MD5 `1aacd3c05cf9695cc799acc59581938d`;
   currently untracked/uncommitted with the harness lane — never committed);
   parity binary SHA-256 `f4b82b109e779f8518332dd86e31371e9a46a5cf50c0ec87360d3a75c95dbd6f`;
   exact replay command/topology/identity/version printed by the runner.
2. **Create/re-establish the test-only Qwen35-MoE emulated-EP2 parity
   harness** — DONE (2026-08-04). Single-shot harness:
   `.agent-progress/run-ep-parity.sh` (`--probe` / `--accept` / `--self-test`)
   driving `crates/hipfire-runtime/examples/ep_decode_parity.rs` with the
   `ep2_harness`/`store_ep2` modules; one GPU, two sequential logical
   stride-2 ranks; topology `single-GPU logical-EP2 (2 ranks, stride-2),
   HIP_VISIBLE_DEVICES=0`; HIP `7.2.53211-9999`. It never constructs
   `EpArch::Qwen35` and adds no daemon Qwen EP admission:
   `bash scripts/check_moe_residency_boundary.sh` passes (normal + self-test
   on the harness tree, 32 categories incl. EP2 staging/resident/harness
   surfaces). AXIS-002 remains the sole Qwen admission owner.
3. **Single-as-sole-baseline parity under the honest FP32 contract** — DONE
   (2026-08-04). Contract: bitwise logits NOT required; exact final-prefill
   argmax, first token, all generated token IDs, finite logits, exact
   second-turn/reset flags, Q8 resolved, no first divergence, strict
   max-abs-delta pin. EP=1 is the alias of Single, not a second required run.
   Results: five fresh confirm probes bit-stable at Dmax = 0.7081642; pinned
   tolerance `0.708164275` is the next representable f32 (one ULP; the accept
   log may print the shortened f32 `0.7081643`); identical token vector
   both sides: `[13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760,
   6511, 314, 9338, 369, 11751]`; `bash .agent-progress/run-ep-parity.sh
   --accept` PASSED. Logs: `.agent-progress/ep-parity-confirm-probe-{1..5}.log`
   (bit-stable), `.agent-progress/ep-parity-confirm-accept.log`;
   `.agent-progress/ep-parity-confirm-structural.log` and
   `.agent-progress/ep-parity-confirm-cleanup.log` each 1 passed, exit 0;
   full feature suite 442 passed / 21 ignored. The quant-independent/
   config-derived/checked conv resolver fix was TDD RED/GREEN
   (`conv_physical_shape_alias_is_quant_independent_and_uses_config_kernel`,
   `synthetic_conv_q8_geometry_resolves_through_real_resolver`) and unblocked
   canonical Q8 physical `[channels,1,kernel]` loading. Required GPU
   tests: warmed success-path state free + checked pool drain within one
   4096-byte page; `frozen_moe_resident_ep2_build_bind_rank_tables_and_canonical_unaffected`
   passed. Narrow evidence: focused EP2 qwen tests 62 passed / 2 ignored;
   dispatch-tests qwen35 19 passed; loader parallel capability 18 passed;
   boundary normal passed; self-test 44 induced failures / all 32 categories
   caught; runner self-test passed.
4. **Graph-enabled generate/unload/reload lifecycle** — **DEFERRED, NOT
   SATISFIED.** By explicit user decision the full multi-cycle lifecycle/VRAM
   closure was moved to STEP-002R; the lifecycle example/runner were removed.
   The harness is one GPU-bearing invocation per process: success-path pool
   drain is proven, but partial-construction rollback, typed load-error
   retention, and failed-free retention are NOT claimed. Do not re-add a
   lifecycle runner in this lane. If/when run under STEP-002R, expected pass:
   per-cycle VRAM recovery within 64 MiB of the post-first-unload baseline
   with no monotonic growth across cycles 2–4.
5. **Frozen-cutover DFlash coherence** — **OUT OF SCOPE for this harness
   (NOT claimed).** No paired A3B draft or canonical prompt is defined; no
   DFlash gate is claimed or required for the harness. The dense DFlash gate
   (`/tmp/coherence-dflash-20260803-154705.md`, 2026-08-03) and the current
   production DFlash regression gate `/tmp/coherence-dflash-20260804-122132.md`
   (2026-08-04, 4/4 OK, no hard errors, no soft/tier3 warnings, using the
   separate canonical qwen3.6-27b target/draft) remain dense-engine regression
   records only — they are NOT direct A3B DFlash evidence. Recorded instead:
   production AR-only smoke (no draft/speculation, Q8 KV, FP32 state, temp 0)
   on gfx1151 — output `The capital of **France** is **Paris**.<|im_end|>`,
   15 tokens, 5.1 tok/s
   (`.agent-progress/ep-production-ar-confirm.log`). This is a scoped AR
   sanity check, not a coherence gate.
6. **Re-run accepted DeepSeek4/MiniMax Single behavior and named structural
   examples** `ep_deepseek4` and `ep_minimax` — **DONE (2026-08-06).** This is
   independent of the Qwen
   fixture, but requires the corresponding DS4/MiniMax model artifacts and a
   topology that can hold them. Record model/prompt digests and run the
   source-documented command shapes (substitute the pinned paths/rank count):
   ```bash
   HIP_VISIBLE_DEVICES=0,1,2,3 cargo run --release \
     -p hipfire-arch-deepseek4 --example ep_deepseek4 -- \
     --model "$DS4_MODEL" --tp 4 --max 48 \
     --prompt "The capital of France is"

   HIP_VISIBLE_DEVICES=0,1,2,3 cargo run --release --features deltanet \
     -p hipfire-arch-minimax --example ep_minimax -- \
     --model "$MINIMAX_MODEL" --tp 4 --max 32 \
     --prompt "The capital of France is"
   ```
   `HIPFIRE_EMULATE_GPUS=N` may replace physical ranks only when the pinned
   fixture fits one GPU; do not treat emulation as physical RCCL evidence.
   Result: both named examples reproduced their deterministic pins under the
   documented single-gfx1151 peer-direct EP2 topology
   (`HIPFIRE_EMULATE_GPUS=2`, `HIPFIRE_DETERMINISTIC=1`,
   `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1`): DeepSeek
   `0x26a13602bedf9926`, MiniMax `0x887c2e7717e9c3bf`; both outputs were
   coherent. Full EP>1 / physical RCCL parity remains non-blocking
   (HW-001/HW-002). Downstream HW-001/HW-002 references
   to the historical `ep_decode_parity` (deleted in `38def37a`; the current
   `ep_decode_parity` example is the new test-only harness, not the deleted
   production example) are stale and need repair; that repair is
   NOT a STEP-002 blocker.

**Tracker update:** the Qwen harness evidence, production common-plan migration,
MiniMax authority repair, and DS4/MiniMax named EP2 runs are recorded. STEP-002
is complete under `GATE3_APPROVED_WITH_DEFERRED_DEBT`; STEP-002R lifecycle and
rollback remain separate accepted debt and do not keep STEP-002 open.

## Updated stopping point — direct-builder prerequisite complete

This handover was originally written after the unsafe-foundation reset, when
`WeightStoreAllocation`, `WeightCellId`, `WeightStoreBuilder`, and
`FrozenWeightStore` did not yet exist. Tasks 2–4 of the direct-builder
fulfillment plan have since been implemented and passed applicable Oracle gates.
The current tree now contains:

- `WeightStoreAllocation` with live-origin-gated free and retry ownership
  (Task 2).
- `WeightStoreBuilder` with `for_target` full-binding capture,
  `stage_bytes`/`stage_alias` keyed placement, and private adoption surface
  (Task 3, after full-binding remediation).
- `fulfill_manifest_builder` with transactional rollback, retry-owning freeze,
  global-device/local-shard rank separation, whole-arena structural validation,
  and panic-free shard helpers (Task 4, after rank/slicing/arena/no-panic
  remediations).

Legacy `WeightStore`, `WeightHandle`, `WeightStoreAssembly`, and
`fulfill_manifest*` remain untouched for unmigrated callers. Historical
architecture-file changes (Qwen/DS4/MiniMax/depth) are already committed and
were unrelated to direct-builder Tasks 2–4.

Both boundary scripts pass:

```text
MOE residency boundary check passed: no forbidden ownership symbols in tracked Rust sources under crates/.
weight_store hybrid foundation boundary: passed
```

Historical CPU evidence (2026-07-23 direct-builder prerequisite, superseded by
the Phase C numbers below): `cargo test -p hipfire-runtime weight_store --lib`
102 passed / 9 GPU ignored; `cargo test -p hipfire-runtime weight_manifest
--lib` 40 passed; `cargo test -p hipfire-hardware --lib` 20 CPU passed / 10 GPU
ignored (GPU tests executed separately as allocation-domain identity evidence,
NOT direct-builder GPU proof). GPU upload/freeze/rollback fixtures are
explicitly ignored — unavailable/unexecuted direct-path evidence.

## Phase B → C checkpoint (2026-08-03)

### Oracle Gate B approval

Oracle Gate B **APPROVED** (2026-08-03) with zero Critical or Important
findings after iterative remediation. The final state: the exact dispatch
snapshot is bound into a sealed source-owning plan; the target is verified
before vision or Frozen allocation; Legacy fallback is preserved only for
preflight Ineligible outcomes; complete cleanup aggregates are retained;
exact-domain retry and pool drain are serialized; and an abort-on-unwind guard
protects the post-selection vision owner. Focused evidence at the gate: Qwen
374 passed / 15 ignored, four unwind-guard tests passed.

### STEP-002R debt (accepted, not closed)

By explicit user decision on 2026-07-27, best-effort rollback of
pre-publication common and auxiliary allocations is accepted for this slice
and tracked as STEP-002R. It does **not** count as exact failed-free retention
evidence. The relaxation applies only to failed common/auxiliary construction
rollback; it does not relax sole Frozen MoE ownership after publication,
complete Frozen failure propagation where already available, checked
unload/backlog ordering, dispatch admission, refusal boundaries, or honest
GPU-evidence reporting. STEP-002R remains open.

### Phase B TDD ordering violation

Phase B did not follow test-first ordering. This is an explicit process
failure; no strict-TDD claim is permitted for Phase B. The violation is
preserved in the tracker and deepwork records.

### Two destructive recovery incidents

1. **First recovery incident (2026-07-26):** the lifecycle remediation writer
   ran `git checkout HEAD --` on dirty `qwen35.rs`, `store.rs`, `carrier.rs`,
   loader `lib.rs`, and loader `carriers.rs`, erasing all unstaged Phase 2
   work in those files. No blob, stash, index entry, reflog entry, or
   temporary copy was recoverable. The user chose fresh reimplementation;
   surviving dirty changes are checkpointed at
   `/tmp/opencode/qwen35-recovery-survivors-20260726.patch`.
2. **Second recovery incident (2026-07-27):** the delegated common-transaction
   writer violated the no-checkout rule again with `git checkout --
   crates/hipfire-arch-qwen35/src/store.rs`, erasing the full unstaged file.
   Recovery used the pre-incident snapshot `/tmp/opencode/oracle-gate-a.diff`;
   its `store.rs` patch base exactly matched HEAD (`f95095dc`) and restored
   blob `f59537b5`. The full worktree was then checkpointed at
   `/tmp/opencode/qwen35-post-second-recovery-20260727.patch`.

Rule for every delegated writer: **no git checkout/restore/reset/stash** in
this worktree, and future lanes must be smaller than the failed
all-common-transaction assignment.

### Implementation state (post-Gate B, Phase C)

- Single-target frozen-store facade with retained target identity
  (`SingleFrozenWeightStore` + `SingleWeightStoreBuilder` in
  `crates/hipfire-runtime/src/weight_store.rs`).
- Private ID-only Qwen35 MoE projection (`Qwen35MoeResident` +
  `Qwen35MoeLayerProjection<WeightCellId>` + borrowed `MoeFfnBindings`) in
  `crates/hipfire-arch-qwen35/src/store.rs`; no `GpuTensor` fields, no raw
  views, no typed-free-authority exposure.
- Direct Frozen MoE staging (`build_frozen_moe_resident`) with exact C2
  indexed dispatch admission and routed-down AWQ (routed gate-up AWQ rejected
  before upload).
- Source-bound preflight with Legacy fallback; Frozen refused at every
  multi-device entry (`reject_frozen_multi` in all three multi forward
  entries).
- Qwen35Moe EP remains `Planned`/refused before allocation: `EpArchKind`
  (loader) has no Qwen35 variant, `validate_ep_layout` refuses non-DS4/MiniMax
  architectures, the capability matrix keeps `(Qwen35Moe, Ep) => Planned`
  owned by AXIS-002, and there is no `EpArch::Qwen35` or daemon admission.
- Residency boundary script now asserts all of the above
  (`scripts/check_moe_residency_boundary.sh`, Phase C).

### Exact verification commands and results (2026-08-03, Phase C)

```text
$ bash scripts/check_moe_residency_boundary.sh
MOE residency boundary check passed: no forbidden ownership symbols in tracked Rust sources under crates/.
Also passed: ID-only projection fields, Frozen staging-path purity (no from_raw/alias),
from_raw legacy whitelist, no public ownership-surface exposure, multi-device Frozen refusal,
and Qwen35 EP Planned/refused admission (no EpArch::Qwen35, no daemon admission).
exit 0

$ bash scripts/check_moe_residency_boundary.sh --self-test
24 assertion failure(s) total; all 16 expected category/categories caught.
exit 0  — every expected violation category (ExpertShard family, WeightCellId::for_test,
Check 4 public WeightStoreAllocation/raw/adoption surface, ID-only projection, Frozen
from_raw/alias, from_raw whitelist, multi-entry refusal, EP arch/layout/matrix) caught
independently; the script fails closed at startup if a required tool is missing

$ cargo test -p hipfire-runtime weight_store --lib
110 passed; 0 failed; 13 ignored

$ cargo test -p hipfire-runtime --doc
5 passed; 0 failed  (all five weight_store compile-fail doctests)

$ cargo test -p hipfire-arch-qwen35 --lib
374 passed; 0 failed; 15 ignored

$ cargo test -p hipfire-loader --lib
132 passed; 0 failed; 10 ignored

$ cargo test -p hipfire-dispatch --lib
171 passed; 0 failed; 1 ignored

$ cargo test -p hipfire-dispatch-tests
70 passed; 0 failed

$ cargo test -p rdna-compute --lib dispatch   (narrow)
34 passed; 0 failed; 27 filtered

$ cargo test -p rdna-compute --lib            (full, covers pool-affected)
61 passed; 0 failed; 0 ignored

$ cargo check -p hipfire-loader --all-targets
Finished; warnings only (3 pre-existing warnings in lib test)

$ cargo check --workspace --all-targets
Finished; warnings only (pre-existing; no errors)

$ rustfmt --edition 2021 --check --config skip_children=true <29 changed .rs files>
exit 1 — pre-existing formatting drift in mtp_head.rs, qwen35.rs, weight_store.rs
(Phase A/B worktree drift; NOT formatted/churned per scoping rule)

$ git diff --check
exit 0
```

Rust files changed by Phase C itself: none (boundary script is bash; docs are
Markdown). The rustfmt failure above is entirely pre-existing worktree drift.

### Pending GPU evidence and final gate

> Phase C snapshot (2026-08-03), partially superseded 2026-08-04 — the
> closure matrix and runbook above carry the current status. Of the items
> below: fixture/parity/digest evidence is now DONE (user-authorized
> fixture, harness lane complete); VRAM/lifecycle is explicitly deferred to
> STEP-002R; Frozen-cutover DFlash coherence remains unclaimed (no paired
> A3B draft / canonical prompt; scoped AR-only smoke passed instead).

Required evidence that was **unavailable/skipped** at Phase C (no compatible
canonical fixture + time):

- graph-enabled generate/unload/reload on the canonical Qwen35-MoE fixture;
- post-unload VRAM recovery with no monotonic growth across cycles;
- DFlash coherence on the Frozen cutover;
- final-prefill logits / first token / decode token IDs / reset / multi-turn
  parity against a pinned Single baseline;
- model SHA-256, prompt MD5, and binary digest for the canonical fixture.

Hardware/fixture facts (2026-08-03): fresh GPU gfx1151 AMD Radeon 8060S with
HIP 7.2.53211 / ROCm toolchain 7.2.3 (dense coherence and serve multi-turn
smoke passed there — regression safety only).
`~/.hipfire/models/qwen3.6-35b-a3b.mq4` (22,855,051,520 bytes) is
present with MD5 `edde51ec1dac0f2bd42cff5ef1cb8944` and SHA-256
`1dc1c7964de415e0040a540a4300b9518e11b00c13d99c23f576f2b9fe1e8bca`;
on 2026-08-04 the user explicitly authorized reuse of this existing artifact
as the canonical STEP-002 fixture (no dedicated copy required), and the
test-only emulated-EP2 parity lane completed with the pinned prompt/binary
digests (see closure matrix/runbook above). The final merged Oracle gate
APPROVED the complete ownership/lifecycle cutover on 2026-08-03 with zero
Critical or Important findings outside accepted STEP-002R and named this
evidence gap explicitly.

## Next work — accepted downstream debt after STEP-002 closure

Qwen harness steps 1–3 and the DS4/MiniMax named structural EP2 examples are
DONE. Step 4 (lifecycle/VRAM) is deferred to STEP-002R by user decision; step 5
(A3B DFlash coherence) is out of scope for this harness with scoped AR-only
smoke recorded instead. Remaining downstream work:

- **STEP-002R** — origin-preserving, retryable common/auxiliary construction
  rollback (accepted debt; exact failed-free retention not claimed) plus the
  deferred multi-cycle lifecycle/VRAM closure.
- **Stale `ep_decode_parity` references** in HW-001/HW-002 (the historical
  example deleted in `38def37a`; the current `ep_decode_parity` example is
  the new test-only harness) need repair before those tasks run; not a
  STEP-002 blocker.
- **HW-001/HW-002** physical DS4/MiniMax RCCL, **HW-011** Qwen physical EP,
  **AXIS-002** admission, **SPEC-003** MTP — separate downstream/deferred.

Do not claim STEP-002 completed, Qwen EP admission, or production GPU
validation. Do not begin adding another view, auxiliary ledger, or launch
lease. Do not use git checkout/restore/reset/stash in this worktree; do not
touch the unrelated untracked files (`crates/graphify-out/`, `graphify-out/`,
`docs/pr-dspark-qwen3.md`, `docs/pr-dspark-qwen35.md`).

## Final merged Oracle remediation (2026-08-03, fresh narrow lane)

All four final merged Oracle findings are fixed with strict TDD (focused RED
tests first), plus the final Legacy assembly Important finding:

1. **FROZEN MODEL-WIDE MQ6 FENCE (final Oracle Important gap, widened)** —
   the fence is defined as ANY MoE FFN projection in any layer: router /
   shared_expert_gate / shared gate/up/down plus every routed expert
   gate_up/down (uniform or graded), for BOTH Legacy and Frozen, through
   ONE shared metadata predicate `MoeFfnMetaView::has_mq6` (generic over
   the projection key; metadata-only, no tensor lookup). `layers_have_mq6_moe`
   (Legacy) and the Frozen resident publication both consume it, so the
   two storage kinds cannot diverge. This closes two gaps: the old
   snapshot predicate required uniform routed experts (graded MQ6 was
   missed) and the Frozen path covered only routed experts (structural
   MQ6 was missed). Derived BEFORE publication/attachment at the Phase 6
   seam of `load_qwen35_hfq_weights_frozen_prepared`. The gfx1151 prefill
   fence (`force_mq4_grouped_fp16 = model_has_mq6_moe && is_gfx1151 &&
   moe_grouped_i8.is_none()` in `prefill_moe_ffn_body_batched`) reads the
   published field. RED: `moe_view_has_mq6_detects_graded_routed_mq6`
   failed behaviorally (snapshot missed graded MQ6); the store.rs shared-
   field table tests could not compile (non-generic meta view). GREEN:
   `projection_layers_mq6_fence_mixed_true_pure_mq4_false` (routed, preserved),
   `moe_ffn_meta_view_mq6_fence_covers_every_shared_field` (cross-layer:
   layer A pure MQ4 + layer B each shared field MQ6 → true; pure all-MQ4 →
   false), `moe_ffn_meta_view_mq6_fence_covers_graded_routed_experts`,
   `moe_view_has_mq6_detects_graded_routed_mq6` /
   `moe_view_has_mq6_detects_every_shared_projection` (Legacy), and the
   GPU-ignored `frozen_publication_derives_model_wide_mq6_fence` extended
   with a structural-MQ6 fixture (layer-1 shared projections MQ6, routed
   MQ4 → true) — all pass on gfx1151 with `--ignored`. Also repaired the
   stale GPU-ignored `frozen_moe_resident_build_and_bind` (used a k=1
   config that Frozen admission refuses, and its panic poisoned
   `GPU_TEST_LOCK` for the other GPU tests); it now uses the valid
   k=8/MQ4 fixture.
   **Legacy assembly gap (final Important finding):** the Legacy
   assembly (`assemble_qwen35_weights_inner_with_mode`) still derived
   `moe_has_mq6` from its own inline routed-only scan — structural MQ6
   (router/shared) in a Legacy checkpoint was missed. It now publishes the
   fence via the shared CPU-testable seam
   `assembled_legacy_layers_have_mq6` (per-layer
   `MoeFfnMetaView::Legacy(ffn).has_mq6()`; Frozen markers → false, the
   resident publication derives later); `layers_have_mq6_moe` delegates to
   the same seam so there is exactly one layer-scan implementation. RED:
   `legacy_assembly_derives_model_wide_mq6_fence` failed behaviorally
   (shared-only MQ6 fixture published `moe_has_mq6=false`); CPU seam test
   was compile-RED (seam missing). GREEN: the same GPU test passes on
   gfx1151 (shared-only MQ6 → true, pure MQ4 → false through the real
   `load_qwen35_hfq_weights` legacy loader), and CPU
   `assembled_layers_mq6_seam_shared_only_layer_and_pure_layer` covers the
   seam (shared-MQ6 + pure-MQ4 layers → true; all-MQ4 → false; Frozen
   markers → false). Dead `MoeDtypeSnapshot::has_mq6` and
   `MoeFfnMetaView::proj` deleted (snapshot `has_mq6` test assertions
   removed with the method).
2. **O(1) FROZEN BINDING** — Frozen decode no longer calls/materializes
   `routed_expert_refs()` (C2 guarantees the indexed GPU route). New seam
   `routed_expert_refs_for_params`: Frozen → empty slice (dispatch's
   `check_moe_decode_supported` rejects empty refs on the CPU fallback, so
   no fake refs/aliases), Legacy → materializes exactly as before. Frozen
   prefill's entirely unused `routed_experts` Vec removed. Indexed dispatch
   inputs (gate_up/down pointer tables, AWQ table, dtype tags) preserved.
   Call-count seam: `#[cfg(test)] routed_ref_seam` (instrumented counter +
   serialized `SeamGuard`). GREEN: CPU
   `routed_ref_seam_legacy_materializes_once_and_retains_behavior` (exactly
   one resolution per Legacy call; Ok(empty) retained) and GPU-ignored
   `frozen_routed_expert_refs_seam_resolves_zero` (published Frozen resident
   → ZERO resolutions). Existing `check_moe_decode_supported` /
   `cpu-topk-fallback-needs-resident-experts` guard tests retain the
   empty-ref error behavior.
3. **Minor** — `Qwen35BundleBuildError` + `BundleBuildTransaction` are
   `pub(crate)` (no external consumer; `lib.rs` re-exports only
   `Qwen35Bundle`); `MoeResolution::routed_indexable()` now includes the new
   `routed_indexable_e8` field consistently (E8 test
   `moe_res_e8_routed_indexable_consistent_when_admitted` — behavioral RED
   first); dead prefill extraction variables caused by the migration removed
   (22 unused variables including the prefill `routed_experts`). While making
   the publication seam executable, a latent Frozen common-assembly
   index-OOB panic was found and fixed: both MoE layer arms read
   `derived_plans[layer]` unconditionally although Frozen mode never builds
   the plan Vec — the read is now gated to `MoeAssemblyMode::Legacy`.
4. **Verification (2026-08-03, fresh runs)** — qwen35 380 passed / 18
   ignored; dispatch 172 passed / 1 ignored; dispatch-tests 70 passed;
   loader 132 passed / 10 ignored; runtime weight_store 110 passed / 13
   ignored; runtime compile-fail doctests 5 passed;
   `cargo check -p hipfire-loader --all-targets` and
   `cargo check --workspace --all-targets` finish with pre-existing warnings;
   `bash scripts/check_moe_residency_boundary.sh` exit 0; `--self-test` exit
   0 (24 assertion failures, all 16 categories caught); scoped `rustfmt
   --check` on changed files: my changed regions clean (store.rs fully
   canonical; remaining qwen35.rs drift is pre-existing Phase A/B worktree
   drift); `git diff --check` exit 0. GPU (gfx1151, HIP 7.2): all four
   GPU-ignored frozen/legacy tests pass with `--ignored` (parallel and
   serial). Tracker STEP-002 Evidence and this handover updated; deepwork
   record appended. Final-remediation checkpoint:
   `/tmp/opencode/qwen35-final-remediation-20260803.patch` (refreshed for
   the Legacy assembly fix).

## Verification already passed

- `bash scripts/check_moe_residency_boundary.sh` passed on the Phase C tree;
  `--self-test` caught all 16 expected violation categories independently
  (exit 0), and the script fails closed at startup if a required tool is
  missing. Both re-pass at `ea5d76fa`; on the harness tree the self-test
  expanded to 44 induced failures / all 32 categories caught (EP2
  staging/resident/harness surfaces included) and the runner self-test
  (`bash .agent-progress/run-ep-parity.sh --self-test`) passes.
- `bash scripts/check-weight-store-hybrid-boundary.sh` passed (unchanged).
- Full test suite at `ea5d76fa`: **1865 passed / 66 ignored**; affected
  clippy exits 0 with zero diagnostics on post-checkpoint changed lines.
- Dense coherence (fresh GPU gfx1151, HIP 7.2.53211 / ROCm 7.2.3): report
  `/tmp/coherence-dflash-20260803-154705.md` — commit `ea5d76fa`,
  qwen3.6-27b.mq4 + qwen36-27b-dflash-mq4.hfq, 4/4 OK, no hard/soft flags,
  outputs eyeballed coherent. Binary md5
  `22d547fa6a3bffd137279639f6ac701a` (sha256
  `c96f8db68a2565401fa61a53a5b9bddf1ace2ff2edb4155096fd2b65bcb87741`).
- DFlash production regression gate (2026-08-04): report
  `/tmp/coherence-dflash-20260804-122132.md` — 4/4 cases OK, no hard errors,
  no soft/tier3 warnings. It uses the separate canonical qwen3.6-27b
  target/draft and is a production regression gate, NOT direct A3B DFlash
  evidence.
- Serve multi-turn: report `/tmp/serve-multiturn-20260803-155040.md` — 4/4 AR
  + 4/4 DFlash coherent.
- The two GPU reports prove dense-engine regression safety only; they do NOT
  touch STEP-002 MoE/Frozen acceptance.
- Qwen35 emulated-EP2 harness (2026-08-04): `bash .agent-progress/run-ep-parity.sh --accept`
  PASSED — five fresh confirm probes bit-stable at Dmax = 0.7081642, pinned
  tolerance `0.708164275` (next representable f32; the accept log may print
  the shortened f32 `0.7081643`), identical token vector `[13, 198, 760, 6511,
  314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751]` both
  sides, no first divergence. Logs:
  `.agent-progress/ep-parity-confirm-probe-{1..5}.log`,
  `.agent-progress/ep-parity-confirm-accept.log`;
  `.agent-progress/ep-parity-confirm-structural.log` and
  `.agent-progress/ep-parity-confirm-cleanup.log` each 1 passed, exit 0;
  full feature suite 442 passed / 21 ignored. Model SHA-256
  `1dc1c7964de415e0040a540a4300b9518e11b00c13d99c23f576f2b9fe1e8bca` (user
  authorized reuse); prompt MD5 `1aacd3c05cf9695cc799acc59581938d` (prompt
  file currently untracked with the harness lane); binary
  SHA-256 `f4b82b109e779f8518332dd86e31371e9a46a5cf50c0ec87360d3a75c95dbd6f`.
- Harness GPU tests (2026-08-04): warmed success-path state free + checked
  pool drain within one 4096-byte page;
  `frozen_moe_resident_ep2_build_bind_rank_tables_and_canonical_unaffected`
  passed. Narrow evidence: focused EP2 qwen tests 62 passed / 2 ignored;
  dispatch-tests qwen35 19 passed; loader parallel capability 18 passed;
  boundary normal passed; self-test 44 / 32; runner self-test passed.
- Production AR-only smoke (2026-08-04, gfx1151, no draft/speculation, Q8 KV,
  FP32 state, temp 0): `The capital of **France** is **Paris**.<|im_end|>`,
  15 tokens, 5.1 tok/s (`.agent-progress/ep-production-ar-confirm.log`) —
  scoped AR sanity check, NOT a DFlash coherence gate; no DFlash gate is
  claimed or required for this harness.
- The tracker records STEP-001 manifest/Step parity, Qwen35 coherence, and
  serve-multiturn evidence as complete; those are prior evidence, not STEP-002
  completion.
- The tracker records the existing manifest, dispatch, model-parallel, and
  failed-ownership-reset work as the current foundation; STEP-002 evidence
  remains `In progress` (Qwen harness evidence appended 2026-08-04;
  DS4/MiniMax reruns, Task 8 Step 3 production migration, and STEP-002R
  incl. deferred lifecycle/VRAM remain open).

Before handing off implementation, run `git diff --check` and inspect only the
intended source changes. Do not commit from this handover session.

## Active risks

- The current mutable store can still be mistaken for the selected immutable
  store; keep the Phase 0 boundary script active until every old API is gone.
- `GpuTensor` has no ordinary freeing `Drop`; any temporary or derived
  allocation not registered under the eventual token owner can leak on a
  later-rank failure.
- Pointer tables bake physical addresses. Borrowed binding construction must
  prove the cell/store lifetime and must not recreate raw cloneable views.
- DS4/MiniMax EP has physical RCCL gates HW-001/HW-002; emulation is not
  production hardware evidence.
- Qwen's canonical 35B fixture is now user-authorized and pinned (model
  SHA-256 `1dc1c796…`, prompt MD5 `1aacd3c0…`, parity binary SHA-256
  `f4b82b10…`); the remaining acceptance gaps are the DS4/MiniMax reruns,
  Task 8 Step 3's production migration, and STEP-002R (incl. deferred
  lifecycle/VRAM). Missing evidence is a failed/incomplete gate, not an
  invitation to substitute a smaller model.
- Downstream HW-001/HW-002 references to the historical `ep_decode_parity`
  (deleted in `38def37a`) are stale — the current `ep_decode_parity` example
  is the new test-only harness — and need repair before those tasks run; not
  a STEP-002 blocker.
- The harness lane (code + runner + scripts) is UNCOMMITTED; do not lose it,
  do not commit without the appropriate implementation review, and review its
  full diff before integration.

## Files to read first

1. `.agent-progress/device-mesh-refactor-tracker.md` — authoritative status,
   especially STEP-002 (status at line 414; acceptance/validation/hardware
   contract at lines 417–419) and AXIS-002 at line 573.
2. `docs/superpowers/plans/2026-07-22-weight-store-moe-residency-recovery.md` —
   selected hybrid phases, exact paths, TDD tasks, and old-task supersession.
3. `crates/hipfire-runtime/src/weight_store.rs` — current mutable store and
   the exact Phase-0-to-Phase-1 seam.
4. `crates/hipfire-runtime/src/weight_manifest.rs` — placement and collective
   contracts.
5. `crates/hipfire-dispatch/src/families/moe.rs` — centralized MoE dispatch
   contracts.
6. `crates/hipfire-loader/src/model_parallel.rs` — named-axis MoE execution
   policy and refusal checks.
7. `crates/hipfire-arch-qwen35/src/{store.rs,qwen35.rs,paro_moe.rs}` — raw
   projection/ownership surfaces to migrate only after the frozen store.
8. `crates/hipfire-arch-deepseek4/src/arch.rs` and
   `crates/hipfire-arch-minimax/src/minimax.rs` — routed placement consumers.
9. `scripts/check_moe_residency_boundary.sh` and
   `scripts/check-weight-store-hybrid-boundary.sh` — reset and hybrid boundary
   checks.
10. `.agent-progress/run-ep-parity.sh` — the test-only emulated-EP2 parity
    runner (`--probe` / `--accept` / `--self-test`; acceptance pins enforced),
    and `crates/hipfire-runtime/examples/ep_decode_parity.rs` with
    `crates/hipfire-arch-qwen35/src/{ep2_harness.rs,store/store_ep2.rs}` — the
    harness lane (uncommitted).
11. `docs/superpowers/plans/2026-07-22-step-002-all-moe-spine.md` — Task 8
    harness checkboxes (Steps 1, 2, 4, 5 done; Step 3 remains the
    production/common-plan migration boundary).
