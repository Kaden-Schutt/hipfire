# Documentation index

Single hub for navigation, lifecycle classification, and ownership.
Domain prose lives in the linked owners; this file does not duplicate it.

| Field | Value |
|---|---|
| Inventory date | 2026-07-19 |
| Working branch | `lfm-redline` |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| Comparison base | `origin/beta` @ `9ffb18da9d1377dfbf759db82641ea039b2e522e` |
| Integrated commit / tree / source hashes | Supplied externally by Git/CI after cutover. Never self-referenced here. |

## Truth states

Exactly one label applies to an active claim or surface:

| State | Meaning |
|---|---|
| **shipped / ref-pinned** | Present on the audited source ref and treated as current product or contributor authority for that concern. |
| **branch-implemented** | Implemented or documented on `lfm-redline` (or another named working branch) but not a fact of `origin/beta` at the comparison base. |
| **measured** | Observation tied to named fixture, binary/model identity, and date. Not a default, floor, or admission. |
| **planned** | Intent only. No executable authority and no implied schedule. |
| **historical** | Retained record. Useful for provenance; not current procedure or baseline unless a shipped owner re-states it. |

Directory rows may say **mixed (member metadata)** when members span more than one defined state. That is collection policy, not a sixth truth state — each claim still takes exactly one label from the table above (or fail closed).

Lack of authority is not a fifth positive state:

| Marker | Meaning |
|---|---|
| **blocked** | No single canonical owner, missing executable route, or unresolved contradiction. Fail closed. |
| **superseded / rejected** | Replaced or declined; keep only as history. Do not promote. |
| **unknown** | Not classified. Fail closed — do not invent an owner, admission, or gate. |

**Fail-closed:** if a concern has no row below, or a row is `blocked` / `unknown`, do not treat any nearby doc, skill, measurement, or runtime default as authority. Open an explicit owner or keep the claim out of product language.

Admissions are machine-recorded only in [`admissions.yml`](admissions.yml). An empty `records` list means no inferred admissions. Runtime gating or a passing measurement alone never admits a route.

## How to use this index

1. Find the concern in [Ownership](#ownership).
2. Open that owner for mutable facts.
3. For validation or promotion evidence, use only [`VALIDATION.md`](VALIDATION.md).
4. Treat collection rows as directory policy; do not promote a file inside a historical collection by recency alone.

## Ownership

Exactly one canonical owner (or explicit `BLOCKED`) per concern.

| Concern | Canonical owner | State | Notes |
|---|---|---|---|
| Docs navigation, lifecycle labels, ownership map | [`docs/INDEX.md`](INDEX.md) | branch-implemented | This file. Not yet on audited ref / integrated tree. |
| Human validation route selection | [`docs/VALIDATION.md`](VALIDATION.md) | branch-implemented | Sole route selector. Not yet on audited ref / integrated tree. |
| Machine admission registry | [`docs/admissions.yml`](admissions.yml) | branch-implemented | Schema only; `records: []`. Not yet on audited ref / integrated tree. |
| Product onboarding | [`docs/GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | |
| CLI surface and model lifecycle commands | [`docs/CLI.md`](CLI.md) | shipped / ref-pinned | |
| Daemon / user config keys | [`docs/CONFIG.md`](CONFIG.md) | shipped / ref-pinned | |
| Environment variables | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | Machine-checked against source where `scripts/check-env-docs.py` applies. |
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
| Redline contributor certification and route-proof policy | [`docs/REDLINE.md`](REDLINE.md) | branch-implemented | Normative on this branch; not an `origin/beta` fact at the comparison base. |
| Thin Redline skill hook (workflow only) | [`docs/skills/redline-retained-replay.md`](skills/redline-retained-replay.md) | branch-implemented | Must not fork policy from `REDLINE.md`. |
| Executable agent skills root | [`.agents/skills/`](../.agents/skills/) | shipped / ref-pinned | Sole executable skill root. `docs/skills/` is non-executable reference only. |
| Speculation feature inventory | [`docs/speculation-support-inventory.md`](speculation-support-inventory.md) | historical | Inventory snapshot; verify in source before product claims. |
| Spec-decode durability note (2026-06-23) | [`docs/spec-decode-durability-2026-06-23.md`](spec-decode-durability-2026-06-23.md) | measured | Dated fixture/tables report. |
| Published benchmark tables | [`docs/BENCHMARKS.md`](BENCHMARKS.md) | measured | Historical/measured snapshots; not live floors. |
| Dated perf campaign checkpoints | [`docs/perf-checkpoints/`](perf-checkpoints/) | measured | Immutable bodies; amend only via new dated files. |
| Dependency adoption log | [`docs/dependency-adoption-log.md`](dependency-adoption-log.md) | historical | |
| Upstream merge journal | [`docs/upstream-merge-journal.md`](upstream-merge-journal.md) | historical | |
| DeepSeek4 PR body archive | [`docs/deepseek4-pr-body.md`](deepseek4-pr-body.md) | historical | |
| Multi-GPU bring-up lessons | [`docs/multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md) | historical | |
| HFP4 format note | [`docs/quant-formats/hfp4.md`](quant-formats/hfp4.md) | mixed (member metadata) | Member text: v1/v1.5 shipped / ref-pinned claims, v2/v3 planned. Broader quant authority remains `QUANTIZATION.md`. |
| MoE AWQ working notes | [`docs/moe-awq/`](moe-awq/) | historical | |
| MI300X rental runbook | [`docs/rental/MI300X-RENTAL-RUNBOOK.md`](rental/MI300X-RENTAL-RUNBOOK.md) | historical | |
| Relicense / governance records | [`docs/governance/`](governance/) | historical | Legal/historical; do not rewrite. |
| Design-time architecture drafts | [`docs/design/`](design/) | mixed (member metadata) | Branch-implemented LFM/Redline designs, measured baselines, and planned intent may coexist; not product defaults. |
| Implementation plans and PRDs | [`docs/plans/`](plans/) | mixed (member metadata) | Plans/PRDs plus measured results ledgers. Recency ≠ authority. |
| Narrow specs | [`docs/specs/`](specs/) | mixed (member metadata) | Intent/spec records; promote only via shipped owners. |
| Investigations | [`docs/investigations/`](investigations/) | mixed (member metadata) | Discovery trails including measured research; not product defaults. |
| Reviews | [`docs/reviews/`](reviews/) | historical | Review archives. |
| Lessons learned | [`docs/lessons_learned/`](lessons_learned/) | historical | Postmortems. |
| Superpowers nested plans/specs | [`docs/superpowers/`](superpowers/) | mixed (member metadata) | Members are planned intent and historical nested plans/specs only. Not an executable skill root. |
| Universal replacement gate for all GPU changes | — | **BLOCKED** | No universal gate. Route per [`VALIDATION.md`](VALIDATION.md). |
| Inferred model/route admissions without registry rows | — | **BLOCKED** | [`admissions.yml`](admissions.yml) stays empty until earned records exist. |
| Stitching manual Redline capture to product timed-arm proof | — | **BLOCKED** | See `REDLINE.md` certification ladder; fail closed. |
| `docs/skills/` as executable skill root | — | **BLOCKED** | Non-executable reference/history only. Do not add executable skill definitions here. |

## Top-level page classification

Every current top-level page, exactly once.

| Page | State | Owner role |
|---|---|---|
| [`INDEX.md`](INDEX.md) | branch-implemented | Navigation, lifecycle, ownership. |
| [`VALIDATION.md`](VALIDATION.md) | branch-implemented | Validation route selector. |
| [`admissions.yml`](admissions.yml) | branch-implemented | Admission registry (empty records). |
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
| [`REDLINE.md`](REDLINE.md) | branch-implemented | Redline certification (branch-only vs `origin/beta`). |
| [`BENCHMARKS.md`](BENCHMARKS.md) | measured | Dated/historical bench tables. |
| [`speculation-support-inventory.md`](speculation-support-inventory.md) | historical | Speculation inventory snapshot. |
| [`spec-decode-durability-2026-06-23.md`](spec-decode-durability-2026-06-23.md) | measured | Dated durability fixture/tables. |
| [`dependency-adoption-log.md`](dependency-adoption-log.md) | historical | Dependency log. |
| [`upstream-merge-journal.md`](upstream-merge-journal.md) | historical | Merge journal. |
| [`deepseek4-pr-body.md`](deepseek4-pr-body.md) | historical | PR body archive. |
| [`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md) | historical | Bring-up narrative. |

## Collection classification

Every current top-level collection, exactly once. Directory policy applies to members that lack their own stronger metadata.

| Collection | State | Policy |
|---|---|---|
| [`methodology/`](methodology/) | shipped / ref-pinned | Active methodology owners. Link; do not copy matrices into root guides. |
| [`perf-checkpoints/`](perf-checkpoints/) | measured | Immutable dated evidence. Correct only with a new dated amendment file. |
| [`design/`](design/) | mixed (member metadata) | Approved branch-only LFM/Redline docs, measured baselines, and planned intent. Member label controls; directory is not runtime truth. |
| [`plans/`](plans/) | mixed (member metadata) | Execution plans/PRDs and measured results ledgers. Recency ≠ authority. |
| [`specs/`](specs/) | mixed (member metadata) | Narrow specs and intent; promote only via shipped owners. |
| [`investigations/`](investigations/) | mixed (member metadata) | Historical discovery plus measured research trails. |
| [`reviews/`](reviews/) | historical | Review archives. |
| [`lessons_learned/`](lessons_learned/) | historical | Postmortems. |
| [`moe-awq/`](moe-awq/) | historical | MoE AWQ session notes. |
| [`quant-formats/`](quant-formats/) | mixed (member metadata) | Format notes under quant umbrella (e.g. HFP4 ships some versions, plans others). `QUANTIZATION.md` remains primary. |
| [`rental/`](rental/) | historical | Rental runbooks. |
| [`governance/`](governance/) | historical | Legal/governance records; bodies stay untouched. |
| [`skills/`](skills/) | mixed (member metadata) | Members are branch-implemented reference hooks and historical skill prose. Non-executable; sole executable root is `.agents/skills/`. |
| [`superpowers/`](superpowers/) | mixed (member metadata) | Members are planned intent and historical nested plans/specs only. |

## Branch scope

- **Audited ref** (`692a726dde53508cb53de1a74c720e75a7c9f33e`): pin for greenfield active prose derived from current source on this branch.
- **Comparison base** (`origin/beta` @ `9ffb18da9d1377dfbf759db82641ea039b2e522e`): use when labeling branch-only vs already-on-beta facts.
- Branch-only Redline and LFM surfaces must not be phrased as `origin/beta` product facts.
- Historical, legal, and measured checkpoint bodies are not rewritten by this index.

## Explicit non-goals

- No mutable fact duplication across owners.
- No self-referencing “final” commit or tree hash in this file.
- No universal GPU/coherence replacement gate.
- No admissions inferred from benches, harness exits, or registry tags.
