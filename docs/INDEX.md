# Documentation index

Single hub for navigation, lifecycle classification, and ownership.
Domain prose lives in the linked owners; this file does not duplicate it.

| Field | Value |
|---|---|
| Inventory date | 2026-07-22 |
| Working branch | `beta` |
| Audited source ref | `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Comparison base | `origin/beta` @ `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Integrated commit / tree / source hashes | Supplied externally by Git/CI after cutover. Never self-referenced here. |

## Truth states

Exactly four **active** truth states apply to a current claim or surface.
Pick exactly one:

| State | Meaning |
|---|---|
| **shipped / ref-pinned** | Present on the audited source ref and treated as current product or contributor authority for that concern. |
| **branch-implemented** | Implemented or documented on a named working branch but not present in the audited beta ref. |
| **measured** | Observation tied to named fixture, binary/model identity, and date. Not a default, floor, or admission. |
| **planned** | Intent only. No executable authority and no implied schedule. |

**Historical** is a **lifecycle / evidence disposition**, not a fifth active
truth state. It means a retained record useful for provenance — not current
procedure, baseline, or product authority — unless a shipped owner re-states
the claim under one of the four active states above.

Directory and collection rows use **exactly one** canonical classification:
one of the four active states, a historical disposition, or an explicit
fail-closed marker below. When members vary, put the directory policy
classification in the State cell and explain member exceptions in
Notes/Policy. Never invent another truth state or a slash-combined policy
state. Each individual claim still takes exactly one active label, a
historical disposition, or fails closed.

Lack of authority is not an active truth state:

| Marker | Meaning |
|---|---|
| **blocked** | No single canonical owner, missing executable route, or unresolved contradiction. Fail closed. |
| **superseded / rejected** | Replaced or declined; keep only as history. Do not promote. |
| **unknown** | Not classified. Fail closed — do not invent an owner, admission, or gate. |
| **historical** (disposition) | Retained evidence or archive. Not an active authority label. |

**Fail-closed:** if a concern has no row below, or a row is `blocked` /
`unknown`, do not treat any nearby doc, skill, measurement, or runtime default
as authority. Open an explicit owner or keep the claim out of product language.

Admissions are machine-recorded only in [`admissions.yml`](admissions.yml) (schema v2). Exactly one evidence-bound row is admitted; every other route stays unadmitted. Runtime gating or a passing measurement alone never admits a route.

## How to use this index

1. Find the concern in [Ownership](#ownership).
2. Open that owner for mutable facts.
3. For validation or promotion evidence, use only [`VALIDATION.md`](VALIDATION.md).
4. Treat collection rows as directory policy; do not promote a file inside a historical collection by recency alone.

## Ownership

Exactly one canonical owner (or explicit `BLOCKED`) per concern.

| Concern | Canonical owner | State | Notes |
|---|---|---|---|
| Docs navigation, lifecycle labels, ownership map | [`docs/INDEX.md`](INDEX.md) | shipped / ref-pinned | This file. |
| Human validation route selection | [`docs/VALIDATION.md`](VALIDATION.md) | shipped / ref-pinned | Sole route selector. |
| Machine admission registry | [`docs/admissions.yml`](admissions.yml) | shipped / ref-pinned | Schema v2; exactly one earned record (LFM2.5-350M MQ4 gfx1201 retained-PM4). |
| Upstream device-mesh port tracker | [`docs/device-mesh-port-tracker.json`](device-mesh-port-tracker.json) | branch-implemented | PR-00/G0 authority for 68 domain obligations, G0..G15 grouped change sets, topology-class evidence campaigns, and FCP-00; canonical landing G1(hipfire-pr-673@8e4f1daa92439f74b285657dd1b09f623107651d) → G3(hipfire-pr-675@04de36cf8249797cb34d2aeb2b303e669342fd8c) → G5(hipfire-pr-677@26fe251394499721679ae6a05a221d8f4997e3ab), G2 sibling hipfire-pr-674@46a469d08c258f01292e287c7fbfc3003e3e8209 after G1 (semantic gate before complete production multi-device enablement, not prerequisite for G3); public branch `review-pr-668` @ `514fd489326b70d1d796849d19929c8e85c76b5d` (TEMPORARY - parent will pin implementation commit in second metadata-only commit after implementation); does not replace VALIDATION or admissions ownership. |
| Product onboarding | [`docs/GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | |
| CLI surface and model lifecycle commands | [`docs/CLI.md`](CLI.md) | shipped / ref-pinned | |
| Daemon / user config keys | [`docs/CONFIG.md`](CONFIG.md) | shipped / ref-pinned | |
| Copyable TOML config profiles | [`docs/configs/`](configs/README.md) | shipped / ref-pinned | User, developer, and retained-PM4 examples. |
| Environment variables | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | Env coverage enforced by [`scripts/check-env-docs.py`](../scripts/check-env-docs.py). |
| Registry models, VRAM, sampling, sidecars | [`docs/MODELS.md`](MODELS.md) | shipped / ref-pinned | Registry presence ≠ runtime admission. |
| Serve HTTP API | [`docs/SERVE.md`](SERVE.md) | shipped / ref-pinned | |
| Chat UX and daemon attach behavior | [`docs/CHAT.md`](CHAT.md) | shipped / ref-pinned | |
| Crate layout and request lifecycle (overview) | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | shipped / ref-pinned | Runtime source wins on conflict. |
| Architecture id table | [`docs/architecture-ids.md`](architecture-ids.md) | shipped / ref-pinned | |
| Quantization formats and math | [`docs/QUANTIZATION.md`](QUANTIZATION.md) | shipped / ref-pinned | |
| `hipfire quantize` operator guide | [`docs/QUANTIZE.md`](QUANTIZE.md) | shipped / ref-pinned | |
| Multi-GPU operator guide | [`docs/multi-gpu.md`](multi-gpu.md) | shipped / ref-pinned | |
| Container install / run | [`docs/CONTAINER.md`](CONTAINER.md) | shipped / ref-pinned | |
| NixOS notes | [`docs/NIXOS.md`](NIXOS.md) | shipped / ref-pinned | |
| Perf claim protocol (warmup, fresh-process, noise) | [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) | shipped / ref-pinned | Numbers live in measured owners, not here. |
| Bench-suite layout | [`docs/methodology/bench-suite.md`](methodology/bench-suite.md) | shipped / ref-pinned | |
| Arch-port validation procedure (channel / speed) | [`docs/methodology/arch-port-validation.md`](methodology/arch-port-validation.md) | shipped / ref-pinned | Does not restore retired coherence-gate batteries. |
| Perf-arch working discipline | [`docs/methodology/perf-arch-discipline.md`](methodology/perf-arch-discipline.md) | shipped / ref-pinned | |
| Kernel Atlas methodology | [`docs/methodology/kernel-atlas.md`](methodology/kernel-atlas.md) | shipped / ref-pinned | |
| Redline contributor certification and route-proof policy | [`docs/REDLINE.md`](REDLINE.md) | shipped / ref-pinned | Normative contributor policy. |
| Golden Redline fixture reproduction and client setup | [`docs/GOLDEN-REDLINE.md`](GOLDEN-REDLINE.md) | branch-implemented | Sealed measured fixtures; does not create admission. |
| Thin Redline skill hook (workflow only) | [`.agents/skills/redline-retained-replay/`](../.agents/skills/redline-retained-replay/) | shipped / ref-pinned | Must not fork policy from `REDLINE.md`. |
| Executable agent skills root | [`.agents/skills/`](../.agents/skills/) | shipped / ref-pinned | Sole executable skill root. |
| Speculation feature inventory | [`docs/speculation-support-inventory.md`](speculation-support-inventory.md) | historical | Inventory snapshot; verify in source before product claims. |
| Spec-decode durability note (2026-06-23) | [`docs/spec-decode-durability-2026-06-23.md`](spec-decode-durability-2026-06-23.md) | measured | Dated fixture/tables report. |
| Published benchmark tables | [`docs/BENCHMARKS.md`](BENCHMARKS.md) | historical | Unmanifested / incomplete evidence identity on tables → historical only; not live floors or admissions. |
| Dated perf campaign checkpoints | [`docs/perf-checkpoints/`](perf-checkpoints/) | measured | Immutable bodies; amend only via new dated files. |
| Dependency adoption log | [`docs/dependency-adoption-log.md`](dependency-adoption-log.md) | historical | |
| Upstream merge journal | [`docs/upstream-merge-journal.md`](upstream-merge-journal.md) | historical | |
| DeepSeek4 PR body archive | [`docs/deepseek4-pr-body.md`](deepseek4-pr-body.md) | historical | |
| Multi-GPU bring-up lessons | [`docs/multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md) | historical | |
| HFP4 format note | [`docs/quant-formats/hfp4.md`](quant-formats/hfp4.md) | shipped / ref-pinned | Current HFP4/MFP4 owners are shipped / ref-pinned; some reserved aliases remain planned. Broader quant authority stays `QUANTIZATION.md`. |
| MoE AWQ working notes | [`docs/moe-awq/`](moe-awq/) | historical | |
| MI300X rental runbook | [`docs/rental/MI300X-RENTAL-RUNBOOK.md`](rental/MI300X-RENTAL-RUNBOOK.md) | historical | |
| Relicense / governance records | [`docs/governance/`](governance/) | historical | Legal/historical; do not rewrite. |
| Design-time architecture drafts | [`docs/design/`](design/) | planned | Directory policy is planned intent. Member exceptions: some branch-implemented LFM/Redline designs and measured baseline notes may coexist; none are product defaults. Historical member bodies stay disposition-only. |
| Implementation plans and PRDs | [`docs/plans/`](plans/) | planned | Directory policy is planned intent. Member exceptions: measured results ledgers may appear; recency ≠ authority. |
| Narrow specs | [`docs/specs/`](specs/) | planned | Intent/spec records; promote only via shipped owners. |
| Investigations | [`docs/investigations/`](investigations/) | historical | Discovery trails including measured research; not product defaults. |
| Reviews | [`docs/reviews/`](reviews/) | historical | Review archives. |
| Lessons learned | [`docs/lessons_learned/`](lessons_learned/) | historical | Postmortems. |
| Superpowers nested plans/specs | [`docs/superpowers/`](superpowers/) | planned | Nested plans/specs only. Not an executable skill root. |
| Universal replacement gate for all GPU changes | — | **BLOCKED** | No universal gate. Route per [`VALIDATION.md`](VALIDATION.md). |
| Inferred model/route admissions without registry rows | — | **BLOCKED** | [`admissions.yml`](admissions.yml) schema v2 forbids inferred/wildcard rows; only the exact earned record is admitted. |
| Stitching manual Redline capture to product timed-arm proof | — | **BLOCKED** | See `REDLINE.md` certification ladder; fail closed. |

## Top-level page classification

Every current top-level page, exactly once.

| Page | State | Owner role |
|---|---|---|
| [`INDEX.md`](INDEX.md) | shipped / ref-pinned | Navigation, lifecycle, ownership. |
| [`VALIDATION.md`](VALIDATION.md) | shipped / ref-pinned | Validation route selector. |
| [`admissions.yml`](admissions.yml) | shipped / ref-pinned | Admission registry (schema v2; exactly one record). |
| [`device-mesh-port-tracker.json`](device-mesh-port-tracker.json) | branch-implemented | Domain obligations, G0..G15 change sets, evidence campaigns, and final closure authority; canonical landing G1→G3→G5, G2 semantic gate; public branch `review-pr-668` @ `514fd489326b70d1d796849d19929c8e85c76b5d` (TEMPORARY - parent will pin after implementation); statuses do not claim merged GitHub PRs. |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | Onboarding. |
| [`CLI.md`](CLI.md) | shipped / ref-pinned | CLI. |
| [`CONFIG.md`](CONFIG.md) | shipped / ref-pinned | Config keys. |
| [`env-vars.md`](env-vars.md) | shipped / ref-pinned | Environment variables. |
| [`MODELS.md`](MODELS.md) | shipped / ref-pinned | Models. |
| [`SERVE.md`](SERVE.md) | shipped / ref-pinned | Serve API. |
| [`CHAT.md`](CHAT.md) | shipped / ref-pinned | Chat. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | shipped / ref-pinned | Architecture overview. |
| [`architecture-ids.md`](architecture-ids.md) | shipped / ref-pinned | Arch id table. |
| [`QUANTIZATION.md`](QUANTIZATION.md) | shipped / ref-pinned | Quant design. |
| [`QUANTIZE.md`](QUANTIZE.md) | shipped / ref-pinned | Quantize tool. |
| [`multi-gpu.md`](multi-gpu.md) | shipped / ref-pinned | Multi-GPU ops. |
| [`CONTAINER.md`](CONTAINER.md) | shipped / ref-pinned | Containers. |
| [`NIXOS.md`](NIXOS.md) | shipped / ref-pinned | NixOS. |
| [`REDLINE.md`](REDLINE.md) | shipped / ref-pinned | Redline certification policy. |
| [`GOLDEN-REDLINE.md`](GOLDEN-REDLINE.md) | branch-implemented | Sealed fixture reproduction and OpenAI-client setup. |
| [`BENCHMARKS.md`](BENCHMARKS.md) | historical | Dated/historical bench tables; incomplete manifests stay historical-only. |
| [`speculation-support-inventory.md`](speculation-support-inventory.md) | historical | Speculation inventory snapshot. |
| [`spec-decode-durability-2026-06-23.md`](spec-decode-durability-2026-06-23.md) | measured | Dated durability fixture/tables. |
| [`dependency-adoption-log.md`](dependency-adoption-log.md) | historical | Dependency log. |
| [`upstream-merge-journal.md`](upstream-merge-journal.md) | historical | Merge journal. |
| [`deepseek4-pr-body.md`](deepseek4-pr-body.md) | historical | PR body archive. |
| [`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md) | historical | Bring-up narrative. |

## Collection classification

Every current top-level collection, exactly once. Directory policy applies to members that lack their own stronger metadata. One canonical policy state per collection; member exceptions go in Policy only.

| Collection | State | Policy |
|---|---|---|
| [`methodology/`](methodology/) | shipped / ref-pinned | Active methodology owners. Link; do not copy matrices into root guides. **Member exceptions:** [`astrea-atlas-pareto-workflow.md`](methodology/astrea-atlas-pareto-workflow.md) and [`astrea-model-policy.md`](methodology/astrea-model-policy.md) are **planned / blocked** (proposed intent; not current INDEX/VALIDATION owners; `candidate_id` join unsupported in current Atlas schema). |
| [`perf-checkpoints/`](perf-checkpoints/) | measured | Immutable dated evidence. Correct only with a new dated amendment file. |
| [`design/`](design/) | planned | Approved intent and drafts. Member exceptions: branch-implemented LFM/Redline docs and measured baseline notes may appear; directory is not runtime truth. |
| [`plans/`](plans/) | planned | Execution plans/PRDs. Member exceptions: measured results ledgers may appear. Recency ≠ authority. |
| [`specs/`](specs/) | planned | Narrow specs and intent; promote only via shipped owners. |
| [`investigations/`](investigations/) | historical | Historical discovery plus measured research trails. |
| [`reviews/`](reviews/) | historical | Review archives. |
| [`lessons_learned/`](lessons_learned/) | historical | Postmortems. |
| [`moe-awq/`](moe-awq/) | historical | MoE AWQ session notes. |
| [`quant-formats/`](quant-formats/) | shipped / ref-pinned | Format notes under quant umbrella; HFP4/MFP4 current surface is shipped / ref-pinned while some aliases stay planned. `QUANTIZATION.md` remains primary. |
| [`rental/`](rental/) | historical | Rental runbooks. |
| [`governance/`](governance/) | historical | Legal/governance records; bodies stay untouched. |
| [`superpowers/`](superpowers/) | planned | Nested plans/specs only; not an executable skill root. |

## Audit scope

- **Audited ref** (`202282de8759dfa6963ea5184ad2bf2b9259cef6`): beta behavior pin used for this inventory refresh.
- **Comparison base** (`origin/beta` @ `202282de8759dfa6963ea5184ad2bf2b9259cef6`): use when separating branch-only work from current beta facts.
- Branch-implemented surfaces must not be phrased as beta product facts.
- Historical, legal, and measured checkpoint bodies are not rewritten by this index.

## Explicit non-goals

- No mutable fact duplication across owners.
- No self-referencing “final” commit or tree hash in this file.
- No universal GPU/coherence replacement gate.
- No admissions inferred from benches, harness exits, or registry tags.
