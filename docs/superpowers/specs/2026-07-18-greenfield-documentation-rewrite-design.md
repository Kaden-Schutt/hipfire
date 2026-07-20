# Greenfield Documentation Rewrite Design

**Status:** Approved design intent  
**Lifecycle:** `planned`  
**Allowed claim states:** Planned design requirements; cited audit examples retain their explicitly labeled states  
**Canonical domain:** Documentation-governance greenfield rewrite design, not operational runtime or validation policy  
**Audited reference:** `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`  
**Comparison base:** `origin/beta@9ffb18da9d1377dfbf759db82641ea039b2e522e`  
**Integrated verification reference:** External Git/CI full commit SHA and tree SHA of the final integrated rewrite tree; tracked content MUST NOT embed or self-attest those object IDs  
**Last checked:** 2026-07-18  
**Supersedes:** `docs/superpowers/specs/2026-07-18-documentation-reliability-cutover-design.md`  
**Replacement:** Becomes `superseded` and non-operational after `docs/INDEX.md` and `docs/VALIDATION.md` assume their defined authority on the verified integrated tree

This document uses **MUST**, **MUST NOT**, **SHOULD**, and **MAY** normatively. It unconditionally supersedes the selective cutover design as the operational documentation-reliability design. The superseded design remains historical provenance with its body intact and MUST NOT compete with this design, `docs/INDEX.md`, or `docs/VALIDATION.md`.

Facts present only on the audited branch (24 commits ahead of the comparison base) are branch facts, not shipped `origin/beta` facts.

## Intent

Replace every **active** documentation surface body at its established path with content re-derived from pinned machine or policy owners. Do **not** wipe `docs/`, rewrite historical/evidence/legal bodies, or introduce an `old-docs/` mirror. Blank files remove copied prose; they do not invent truth state. Unknown and blocked claims fail closed.

No runtime, admission, benchmark, or release status changes. No bespoke candidate/attestation commit pair, field-allowlisted direct-child protocol, or tracked self-naming of verification object IDs.

## Truth model (summary)

Every factual claim has exactly one primary truth state. States classify claims, not files; they are not a promotion ladder.

| State | Meaning | Forbidden inference |
|---|---|---|
| **Shipped / integration-ref-pinned** | Behavior in a release or named integration ref. | Not measured performance, full-device certification, or wider support. |
| **Branch-implemented** | Behavior on `branch@full-commit`, labeled **Branch-only; not shipped**. | Not released, default, or present on the comparison base. |
| **Measured** | One durable fixture/method observation with complete evidence. | Not admission, default, route certification, transfer, or current baseline. |
| **Planned** | Approved design/plan intent at a named revision. | Not implemented, measured, admitted, shipped, or default-on. |

**Unknown** and **blocked** are lack-of-authority conditions, not truth states:

- Unknown → omit from product prose or record as an open question.
- Blocked → name the missing proof/policy; remain non-authoritative.
- Document conflicts stay blocked until the canonical owner or executable source resolves them. Recency, confidence, or rewrite breadth MUST NOT break the tie.

File lifecycle (`current`, `branch-only`, `planned`, `historical`, `superseded`, `rejected`, `blocked`) is separate from claim truth. Transitions (planned → branch-implemented → shipped; measurement ≠ admission) follow source/tests and `docs/admissions.yml` only—documentation edits cannot promote.

Audited branch limits that MUST survive in active prose when cited:

- LFM optimized path: exact 350M dense MQ4, gfx1201, `HIPFIRE_LFM2_PREFILL_BATCH=1` opt-in at the audited ref only; Q8-first, later cohorts, and default-on remain Planned.
- Redline timing: positive timed-arm route proof in the same timed report; separate artifacts MUST NOT be stitched.
- Rejected LFM Stage-A remains rejected (below gate, non-Redline, no PM4).
- Retired `scripts/coherence-gate*.sh` are never acceptance evidence.

## Exact active rewrite surface

Bodies at these paths are replaced with newly derived content. Paths stay; Git retains prior revisions.

### Authority and validation

- `docs/INDEX.md`
- `docs/VALIDATION.md`
- `docs/admissions.yml` — machine-readable decision registry; initially empty unless a qualifying immutable decision is independently proved

### Root routing

- `README.md`
- `CONTRIBUTING.md`
- `AGENTS.md`
- `CLAUDE.md`

### Product

- `docs/GETTING_STARTED.md`
- `docs/CLI.md`
- `docs/MODELS.md`
- `docs/CONFIG.md`
- `docs/SERVE.md`
- `docs/env-vars.md`
- `docs/CONTAINER.md`

### Architecture, quant, performance, specialized policy

- `docs/ARCHITECTURE.md`
- `docs/architecture-ids.md`
- `docs/QUANTIZATION.md`
- `docs/QUANTIZE.md`
- `docs/BENCHMARKS.md`
- `docs/REDLINE.md`
- `docs/methodology/perf-benchmarking.md`
- `docs/methodology/bench-suite.md`
- `docs/methodology/perf-arch-discipline.md`
- `docs/methodology/arch-port-validation.md`
- `docs/methodology/kernel-atlas.md`
- `docs/methodology/kernel-atlas-architecture.md`

### Executable agent skills (sole root: `.agents/skills/`)

Recreate from `docs/skills/` only at:

- `.agents/skills/redline-retained-replay/SKILL.md`
- `.agents/skills/agent-memory/SKILL.md`
- `.agents/skills/serve-restart/SKILL.md`
- `.agents/skills/gfx-kernel-metadata/SKILL.md`

Re-derive the existing active set (not light patch):

- `.agents/skills/hipfire-arch-port/SKILL.md`
- `.agents/skills/hipfire-arch-port/contributor-onboarding.md`
- `.agents/skills/hipfire-arch-port/playbook.md`
- `.agents/skills/hipfire-arch-port/speculation.md`
- `.agents/skills/hipfire-arch-port/validation.md`
- `.agents/skills/hipfire-arch-port/wmma-matrix.md`
- `.agents/skills/hipfire-kernel-tuning/SKILL.md`
- `.agents/skills/hipfire-kernel-tuning/case-studies.md`
- `.agents/skills/hipfire-kernel-tuning/playbook.md`
- `.agents/skills/rebase-onto-modular/SKILL.md`
- `.agents/skills/hipfire-tester/SKILL.md`
- `.agents/skills/hipfire-tester/guide.md`
- `.agents/skills/hipfire-kernel-atlas/SKILL.md`
- `.agents/skills/hipfire-diag/SKILL.md`
- `.agents/skills/hipfire-autoheal/SKILL.md`

Remove `docs/skills/` sources and the empty tree only in the same authority switch that updates every reference. MUST NOT introduce `.skills/` as alias, symlink, wrapper, or compatibility root.

### Active-layer removals allowed

Copied validation matrices/universal gates outside VALIDATION; copied inventories outside canonical projections; executable skill bodies outside `.agents/skills/`; dead `.skills/` refs; coherence-gate acceptance claims; unsupported present-tense, inferred admissions, unlabeled branch claims; obsolete mirrors once owner + routes + enforcement land atomically. Remove `scripts/check-env-docs.py` only if a replacement covers all of its behavior.

## Retained collections

MUST retain; greenfield is not archive erasure.

| Collection | Rule |
|---|---|
| `docs/perf-checkpoints/` | Every pre-existing file byte-for-byte; corrections = separate dated amendments linked to unchanged originals |
| Rejected dispositions | Including rejected LFM Stage-A and its limitations |
| Plans, designs, investigations, specs, reviews, lessons, superpowers/workflow | Bodies untouched; collection-level classification only |
| Legal/provenance | `PRIOR-ART.md`, `NOTICE`, licenses, graft records; attribution meaning preserved |
| Governance/evidence records | Durable; not active product prose |
| Superseded selective design | Retained as provenance; not deleted |
| Runtime/source/test history | Machine owners used to derive new pages |

Historical boundary banners (add without modifying bodies beneath):

- `docs/design/README.md`
- `docs/plans/README.md`
- `docs/investigations/README.md`
- `docs/specs/README.md`
- `docs/reviews/README.md`
- `docs/lessons_learned/README.md`
- `docs/superpowers/README.md`
- `docs/perf-checkpoints/README.md`

## Canonical owners (greenfield destinations)

| Concern | Owner |
|---|---|
| Navigation, lifecycle, SoT map | `docs/INDEX.md` |
| Validation-route selection | `docs/VALIDATION.md` (scripts own behavior) |
| Executable agent workflows | `.agents/skills/<skill>/` only |
| Product onboarding | `docs/GETTING_STARTED.md` |
| CLI surface | parser/help → `docs/CLI.md` |
| Models | `cli/registry.json` → `docs/MODELS.md` |
| Config | runtime schema → `docs/CONFIG.md` |
| Env inventory | source scan → `docs/env-vars.md` |
| Architecture IDs | load/dispatch registry → `docs/architecture-ids.md` |
| Engine narrative | `docs/ARCHITECTURE.md` |
| Redline policy | `docs/REDLINE.md` when on published ref |
| Perf method | `docs/methodology/perf-benchmarking.md`, `bench-suite.md` |
| Dated numerical evidence | `docs/perf-checkpoints/` (append-only) |
| Admissions | `docs/admissions.yml` |
| Current benchmark claims | `docs/BENCHMARKS.md` only with qualifying admission + evidence; else historical/blocked |
| Plans/designs/investigations | Own immutable records + INDEX lifecycle |
| Attribution | `PRIOR-ART.md`, `NOTICE`, graft records |
| Executable behavior | Runtime source/tests at pinned ref (always strongest) |

Secondary surfaces link; they MUST NOT reproduce normative tables or validation matrices. INDEX owns routing/lifecycle metadata, not domain prose.

## Archaeology and omission ledger

Before discarding active text, complete a semantic migration ledger for every normative statement in `AGENTS.md`, `CLAUDE.md`, and every executable skill.

Per statement record: subject, predicate, modal strength, negation, branch/device/model scope, exceptions, legal/provenance class, and disposition (`retain` | `redirect:<owner>` | `historical` | `omit` | `block`).

Also freeze: audited commit, comparison base, HEAD, post-audit delta; skill/path/coherence reference inventory; checkpoint blob IDs; product-fact → machine-owner map.

Independent semantic review MUST sign every removed normative statement to retain, redirect, historical, omit, or block. Unsigned removal fails closed.

## Machine-derived facts

Facts code can decide MUST be generated or CI-checked against the integrated tree. A page MUST NOT claim “generated” without a generator and reproducible source ref. Ungeneratable facts carry `last checked` + source metadata, are not duplicated, and fail closed rather than guessed.

Minimum checks (fail closed; pre-commit + no-GPU CI before authority switch):

- model registry parity (`cli/registry.json`)
- config keys/defaults/enums
- env-var inventory
- CLI commands/flags/defaults/help
- validation scripts/args vs VALIDATION routes
- repository paths, internal links, anchors
- architecture IDs and paths
- sole-root skill topology + unique skill IDs
- lifecycle/reference coverage
- checkpoint immutability + rejected dispositions
- evidence-schema completeness
- branch-only labeling vs comparison base
- absence of a second canonical owner

Semantic review still rejects: promotion without admission; plan tense as present capability; stitched route attribution; undated historical numbers; unknowns as confidence; opt-in as default; skill thinning that drops required action sequencing.

## Atomic authority switch

1. Freeze refs, checkpoint hashes, inventories, and the complete archaeology ledger. No active authority changes yet.
2. Freeze INDEX/VALIDATION/admissions schemas, historical classifications, and checker behavior before drafting domain pages.
3. Derive every active surface privately from its pinned owner; historical/evidence/legal bodies untouched.
4. Prove structural, fact, evidence, branch-scope, archaeology, and archive-immutability reviews on the integrated tree. No dual-authority publish state.
5. Land one integrated tree containing every substantive rewrite, skill move/removal, routing change, checker, tests, hook/CI enforcement, collection banner, and owner mapping. Only INDEX and VALIDATION hold defined active authority; stale coherence-gate authority, second skill roots, and unlabeled branch claims MUST NOT remain.
6. Bind acceptance to external Git/CI **commit SHA** and **tree SHA** of that integrated tree. Ordinary integration MAY rebase, squash, amend, or cherry-pick; any rewrite yields a new identity and MUST re-run full verification. No tracked attestation child commit and no self-named object ID in docs.

Rollback removes new owners, routes, and enforcement as one unit. MUST NOT reactivate known-invalid coherence-gate claims, `.skills/`, unlabeled branch claims, or dual authority. Checkpoints and rejections are never deleted to simplify rollback.

## CI artifact contract

Every verification run that can accept the rewrite MUST emit a durable CI artifact (workflow upload or equivalent immutable job output) with **all** of the following fields. Missing any field fails closed.

| Field | Type | Requirement |
|---|---|---|
| `final_integrated_commit_sha` | full 40-hex Git commit object ID | Commit under test after final integration (post any rebase/squash/amend/cherry-pick) |
| `final_integrated_tree_sha` | full 40-hex Git tree object ID | `git rev-parse <commit>^{tree}` for that commit |
| `comparison_ref` | string | Explicit comparison base (e.g. `origin/beta@9ffb18da9d1377dfbf759db82641ea039b2e522e`) |
| `source_refs` | list of strings | Every source/policy ref used to derive or check active pages (audited branch ref, release/integration pin, generator source commits as applicable) |
| `checker_results` | map name → `{status, detail}` | One entry per required machine check (registry, config, env, CLI, validation routes, paths/links, arch IDs, skill topology, lifecycle, checkpoint immutability, evidence schema, branch scope, second-owner absence). `status` is `pass` or `fail` |
| `semantic_acceptance_matrix` | map criterion_id → `{status, notes}` | One entry per binary acceptance criterion below (rows 1–N). `status` is `pass` or `fail`; failures name the missing authority or residual contradiction |
| `archaeology_ledger_status` | `{complete, unsigned_removals}` | Ledger complete boolean; count of unsigned normative removals (MUST be 0) |
| `workflow_run_id` | string | Hosting CI run identity for external audit |
| `artifact_generated_at` | ISO-8601 UTC | Emission time |

Rules:

- Artifact values for commit/tree SHA MUST match the checkout Git actually verified; docs MUST NOT copy those SHAs into tracked prose as self-attestation.
- After history rewrite, a **new** artifact MUST be emitted for the new commit/tree; prior artifacts do not transfer.
- CI MUST fail the job if any required field is absent, any checker is not `pass`, any matrix row is not `pass`, or `unsigned_removals ≠ 0`.

## Binary acceptance criteria

Accepted only if every row is `pass` on the final integrated commit/tree per the CI artifact. Safe fallback: ref-pinned, limited, or blocked wording—not a broader claim.

| # | Pass condition |
|---:|---|
| 1 | INDEX names exactly one owner or explicit blocked record per concern. |
| 2 | VALIDATION is the only human-facing route-selection owner; every route has an executable path or explicit block. |
| 3 | No active page treats `scripts/coherence-gate*.sh` as acceptance evidence. |
| 4 | `.agents/skills/` is the sole executable skill root; no active `.skills/`; no executable skill under `docs/skills/`. |
| 5 | `AGENTS.md` and `CLAUDE.md` are thin routing/rule surfaces (no copied matrices, baselines, or skill bodies). |
| 6 | Active product pages show release/integration scope; branch facts show `branch@full-commit` and **Not shipped**. |
| 7 | New numerical claims have durable manifest, raw samples, correctness, disposition. |
| 8 | `docs/perf-checkpoints/` is append-only dated evidence; no checkpoint is an automatic current baseline. |
| 9 | Pre-existing checkpoints and rejected dispositions are byte-identical; corrections are separate dated amendments. |
| 10 | Plans/designs/investigations/specs/reviews/workflow are excluded from active product truth unless independently proved at the product ref. |
| 11 | Unknown/blocked claims are omitted or explicitly labeled; none promoted by inference. |
| 12 | Model, config, env, CLI, validation command, arch-ID, path, and link facts are generated or checked where a machine source exists. |
| 13 | No active claim of generic LFM batched prefill, completed LFM Q8 prefill, multi-cohort LFM admission, or Phase-4 default-on. |
| 14 | Any LFM branch claim is limited to audited 350M dense MQ4 / gfx1201 / opt-in and absent from the comparison base. |
| 15 | No Redline timing called route-certified without positive timed-arm proof in the same timed report. |
| 16 | Rejected LFM Stage-A stays rejected, below gate, non-Redline wherever mentioned. |
| 17 | All active links, anchors, skill paths, script paths, and command names resolve at the integrated commit. |
| 18 | Checks fail on fact drift, second owner, dropped branch label, or silent checkpoint rewrite. |
| 19 | README/BENCHMARKS label historical perf by date/fixture; current claims need admissions.yml + complete evidence, else historical/blocked. |
| 20 | No unresolved active contradiction among INDEX, VALIDATION, product docs, routing surfaces, guides, and executable sources. |
| 21 | Every path in the exact active rewrite surface is present and re-derived from its canonical owner. |
| 22 | Eight historical banners exist; retained collection bodies unmodified except allowed amendments/banners. |
| 23 | Archaeology ledger complete; every removed normative AGENTS/CLAUDE/skill statement signed retain/redirect/historical/omit/block. |
| 24 | CI artifact present with every required field; commit/tree SHAs match the verified checkout; post-rewrite runs emit a fresh artifact. |
| 25 | No bespoke C/A pair, direct-child attestation protocol, or tracked self-naming of verification object IDs is required or used for acceptance. |

## Principal risks

Semantic omission; truth-state collapse; evidence laundering; dual authority; immediate drift without mandatory checks; large-diff review failure; skill sequencing regression; identity drift after rebase without re-verify; archive erasure by scope creep.
