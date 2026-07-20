# Documentation Reliability Cutover Design

**Status:** Superseded design intent
**Lifecycle:** `superseded`
**Allowed claim states:** Planned design requirements; cited audit examples retain their explicitly labeled states  
**Canonical domain:** Documentation-governance cutover design, not operational runtime or validation policy  
**Audited reference:** `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`  
**Comparison base:** `origin/beta@9ffb18da9d1377dfbf759db82641ea039b2e522e`  
**Cutover candidate reference (`C`):** Unset until closeout; MUST be one immutable full commit containing every substantive cutover change
**Attestation reference (`A`):** External Git/CI identity of the final direct child of `C`; tracked content MUST NOT attempt to name `A`
**Last checked:** 2026-07-18  
**Replacement:** Superseded by `docs/superpowers/specs/2026-07-18-greenfield-documentation-rewrite-design.md`

The audited branch is 24 commits ahead of the comparison base. Facts introduced only in that delta are branch facts, not shipped `origin/beta` facts. This document uses **MUST**, **MUST NOT**, **SHOULD**, and **MAY** normatively.

This document governs cutover design intent only. After the replacement condition above is met, it remains historical provenance and MUST NOT compete with `docs/INDEX.md` or `docs/VALIDATION.md`.

## Problem

Hipfire’s documentation currently lets the same question resolve to incompatible answers depending on the entry point:

- `CLAUDE.md` retires `scripts/coherence-gate*.sh` as acceptance evidence, while `AGENTS.md`, `README.md`, `docs/BENCHMARKS.md`, methodology pages, and multiple executable skills still call a coherence script canonical or mandatory. One referenced script, `scripts/coherence-gate.sh`, is absent; the presence of other coherence scripts does not resolve the policy conflict.
- Contributor skill links point at a missing `.skills/` tree, executable guidance is split between `.agents/skills/` and `docs/skills/`, and neither `AGENTS.md` nor `CLAUDE.md` provides a complete, current routing surface.
- `AGENTS.md` says `docs/perf-checkpoints/` was archived in April, while the repository continued adding and linking dated checkpoint records in May and July. The directory’s actual role—fixture-bound evidence rather than current defaults—is not stated consistently.
- Product facts drift from their machine sources: model support and sizes diverge from `cli/registry.json`; configuration defaults conflict across product pages; environment-variable coverage lags runtime source; CLI, script, and path references are copied by hand.
- Plans, designs, investigations, results, reviews, and `docs/superpowers/` artifacts sit beside product documentation without a reliable lifecycle boundary. `docs/plans/` alone contains 127 tracked records and no directory index; location, filename, or checkbox state does not establish whether a record is active, shipped, rejected, or superseded.
- Branch-only LFM and Redline work is easy to read as released behavior. In particular, the active LFM plan, its frozen design, current branch implementation, and available measurements are not at the same state.
- Redline timing can be observed without being route-certified. Current tooling cannot prove a timed product arm merely by combining a product timing report with a separate manual retained-launch capture. Treating those artifacts as one proof would create a claim the evidence does not support.

The reliability failure is structural, not a reason to rewrite every document. The repository needs a single ownership map, an explicit truth model, a hard active-versus-historical boundary, and checks that make copied facts fail closed.

## Goals

1. Establish one canonical owner for each documentation concern and make conflicts mechanically visible.
2. Limit unqualified product documentation to behavior shipped in a release or explicitly pinned to an integration reference.
3. Require branch-only implementation facts to name the branch and immutable commit and to say that they are not shipped.
4. Separate implementation, measurement, admission, and intent so that none is inferred from another.
5. Make `docs/VALIDATION.md` the sole human-facing owner of validation-route selection, backed by executable scripts and path-specific policy.
6. Make `docs/INDEX.md` the lifecycle, navigation, and ownership hub for the documentation tree.
7. Make `.agents/skills/` the sole root for executable agent skills; reduce `AGENTS.md` and `CLAUDE.md` to thin routing and repository-rule surfaces.
8. Define `docs/perf-checkpoints/` as an append-only ledger of dated, fixture-bound evidence, never as a source of current defaults or an automatically current baseline.
9. Generate or check model, configuration, environment-variable, command, path, architecture-ID, and link facts against their machine sources wherever feasible.
10. Preserve useful historical evidence without allowing it to become current product truth through proximity, aggregation, or paraphrase.
11. Resolve unknown or blocked claims by omission or explicit blocking labels, never by choosing the most convenient document.
12. Cut over selectively: remove dangerous mirrors and stale assertions while leaving archive and worklog bodies intact.

## Non-goals

- This design does not change runtime behavior, model admission, kernel dispatch, benchmark methodology, or release status.
- It does not certify any existing benchmark, rerun a measurement, or invent a current performance baseline.
- It does not turn the current LFM branch implementation into generic LFM batched-prefill support.
- It does not claim a completed LFM Q8 path, multi-cohort admission, or a Phase-4 default-on state.
- It does not certify any Redline timing that lacks positive timed-arm route proof in the same report.
- It does not rehabilitate retired coherence-gate scripts as valid acceptance evidence.
- It does not flatten the 127 plan records or other worklog trees into a rewritten narrative.
- It does not infer completion from a design marked frozen, a checked task box, an existing symbol, emitted tokens, a benchmark number, or an executable script’s mere presence.
- It does not make `docs/INDEX.md` a second copy of every guide. The index owns routing and lifecycle metadata, not domain prose.
- It does not make `docs/VALIDATION.md` an implementation of validation. Executable scripts remain the authority for what their commands actually do.
- It does not assign implementation tasks, owners, worktrees, or commit order. The cutover phases below define valid repository states, not an implementation plan.

## Truth model

Every factual claim MUST have exactly one primary truth state. The states classify claims, not files, and they are not a promotion ladder.

| State | Meaning | Minimum authority | Permitted use | Forbidden inference |
|---|---|---|---|---|
| **Shipped / integration-ref-pinned** | The behavior exists in a released artifact or in the explicitly named integration reference against which the documentation is published. An arbitrary feature branch does not qualify merely because it has a commit SHA. | Runtime source and tests at the release tag or immutable integration commit; generated facts from the same reference. | Unqualified present tense is allowed only for the stable shipped scope. Ref-pinned pages MUST display the reference. | Shipped does not imply measured performance, certification on every device, or support beyond the stated predicate. |
| **Branch-implemented** | Executable behavior exists on a named non-integrated branch at an immutable commit. | Runtime source and applicable tests at `branch@full-commit`. | Development notes, branch-scoped callouts, and design reconciliation. The label **Branch-only; not shipped** is mandatory. | It MUST NOT be restated as released, default, generally supported, or present on the comparison base. |
| **Measured** | A result was observed for one durable, identified fixture and method. | A complete evidence manifest, raw samples, correctness evidence, and a durable immutable record. | Dated evidence, experiment reports, and explicitly scoped performance discussion. | Measurement does not imply admission, default activation, route certification, transfer to another cohort, or a current baseline. |
| **Planned** | An approved design, plan, or frozen contract states intended behavior. | The plan or design at a named revision and its lifecycle status. | `docs/design/`, `docs/plans/`, `docs/superpowers/`, and links explicitly labeled **Intent, not runtime fact**. | Planned work MUST NOT be described as implemented, measured, admitted, shipped, or default-on. |

### Unknown and blocked claims

**Unknown** and **blocked** are not truth states. They mean the repository lacks enough authority to make the claim.

- An unknown claim MUST be omitted from product prose or recorded as an open question in a design, audit, or investigation.
- A blocked claim MUST name the missing proof or policy decision and MUST remain non-authoritative.
- A conflict between two documents is blocked until the canonical owner or executable source resolves it. Recency, confidence, repetition, or the breadth of a rewrite MUST NOT break the tie.
- A local, ignored, remote-only, or `/tmp` artifact MAY point investigators toward evidence but cannot by itself resolve an unknown into an authoritative claim.

### State transition rules

1. Planned intent becomes branch-implemented only when current source at a named branch commit implements the exact predicate.
2. Branch-implemented becomes shipped only through integration into the declared product reference. Claims of product admission, default enablement, certification, or promoted performance additionally require a qualifying immutable decision in `docs/admissions.yml`. Documentation edits cannot perform either transition.
3. A measured claim remains measured even when its candidate is rejected. Rejection changes disposition, not the historical observation.
4. Measurement attaches to an exact shipped or branch fixture; it does not promote that fixture.
5. A claim that changes predicate, model artifact, binary, prompt, route, environment, or timing method is a new claim and needs new evidence.

## Canonical ownership table

When sources disagree, the owner in this table wins within its stated scope. Secondary surfaces MUST link to the owner and MUST NOT reproduce normative tables or validation matrices.

| Concern | Canonical owner | Secondary surfaces | Ownership contract |
|---|---|---|---|
| Executable behavior and dispatch predicates | Runtime source and tests at a pinned release or commit | All prose | Source establishes what executes. Prose MUST state the reference when it is not the stable shipped reference. |
| Documentation navigation, lifecycle, and source-of-truth map | `docs/INDEX.md` | `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md` | The index lists owners, truth scope, lifecycle, and historical boundaries. It links rather than mirrors domain content. |
| Human validation-route selection | `docs/VALIDATION.md` | Root entry points, methodology pages, skills, specialized guides | It maps claim type and code path to executable evidence. Scripts own actual behavior. No other page may declare a conflicting universal gate. |
| Executable agent workflows | `.agents/skills/<skill>/` | `docs/INDEX.md`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md` | `.agents/skills/` is the only executable skill root. Routing surfaces contain descriptions and links only. |
| Product onboarding | `docs/GETTING_STARTED.md` | `README.md` | Defaults and commands are generated or checked against their machine owners. README remains a concise product front door. |
| CLI command and flag surface | CLI parser/help output, projected into `docs/CLI.md` | README and product guides | Command names, flags, defaults, and examples MUST be checked against the target binary/reference. |
| Model tags, default artifacts, and declared sizes | `cli/registry.json` and generated registry output; projected into `docs/MODELS.md` | README model teaser | Static prose MUST NOT override the registry. Runtime support predicates remain source facts and may be narrower than registry availability. |
| Configuration keys and defaults | Runtime configuration definitions/schema, projected into `docs/CONFIG.md` | GETTING_STARTED, SERVE, CLI | Product pages link to CONFIG for details and may show only checked examples. |
| Environment-variable inventory | Runtime/source extraction, projected into `docs/env-vars.md` | CONFIG, VALIDATION, skills, designs | Generated inventory and manual policy annotations MUST be visually distinct and carry source reference plus generation/review date. |
| Architecture IDs | Runtime load/dispatch registry, projected into `docs/architecture-ids.md` | ARCHITECTURE, skills | IDs and names are checked facts; architecture narratives link to the registry. |
| Engine architecture narrative | `docs/ARCHITECTURE.md` | CLAUDE and CONTRIBUTING summaries | The narrative is ref-pinned and checked for crate/path existence; root guidance does not maintain a competing crate map. |
| Redline procedure and claim language | `docs/REDLINE.md`, when present on the published reference | `docs/VALIDATION.md`, a thin skill under `.agents/skills/`, README | Runtime source remains stronger for behavior. The guide owns certification policy, including timed-arm route proof. On the audited branch it is branch-local relative to `origin/beta`. |
| Performance measurement protocol | `docs/methodology/perf-benchmarking.md` and `docs/methodology/bench-suite.md`, with route choice delegated to VALIDATION | AGENTS, CLAUDE, BENCHMARKS, skills | Warmup, repetition, identity, and interpretation live here; copied noise bands and gate tables are prohibited. |
| Dated numerical evidence | New immutable files under `docs/perf-checkpoints/` or the producing commit/PR when policy requires it | README/BENCHMARKS links | Checkpoints are append-only exact-fixture records, not current defaults. Existing records are not rewritten to match later runs. |
| Product admission decisions | `docs/admissions.yml`, a ref-pinned decision registry whose superseding decisions are added as new records | README, BENCHMARKS, specialized guides | Each record names the exact predicate, source ref, evidence, disposition, default state, date, and superseded record if any. Runtime gating alone does not establish policy admission. |
| Current product benchmark claims | `docs/BENCHMARKS.md`, conditional on a qualifying `docs/admissions.yml` record and complete evidence | README | Without that admission record, the concern is explicitly blocked and BENCHMARKS remains historical. No current summary exists by implication. |
| Plans, designs, investigations, reviews, and handoffs | Their own immutable record plus `docs/INDEX.md` lifecycle classification | Product docs may link under a labeled history/design section | These are intent or history, never product truth. Filename and checkbox state have no authority. |
| Attribution and provenance | `PRIOR-ART.md`, `NOTICE`, and applicable graft records | AGENTS and README legal summaries | AGENTS retains a thin provenance route but not a second provenance narrative. |

## Information architecture

The post-cutover navigation model is:

```text
README.md                         Product front door; shipped/ref-pinned summary
CONTRIBUTING.md                   Contributor entry; routes to canonical owners
AGENTS.md                         Thin agent rules, provenance route, docs/skill links
CLAUDE.md                         Thin maintainer/agent routing and hard repository rules
.agents/skills/                   Sole executable skill root

docs/
├── INDEX.md                      Ownership, lifecycle, navigation, archive boundary
├── VALIDATION.md                 Human validation-route owner
├── GETTING_STARTED.md            Product onboarding
├── CLI.md                        Checked CLI surface
├── MODELS.md                     Registry-backed model narrative
├── CONFIG.md                     Checked configuration surface
├── env-vars.md                   Generated/checked environment inventory
├── ARCHITECTURE.md               Ref-pinned engine narrative
├── architecture-ids.md           Checked architecture-ID registry
├── REDLINE.md                    Specialized Redline policy on refs that contain it
├── methodology/                  Measurement and engineering method
├── perf-checkpoints/             Append-only dated fixture evidence
├── design/                       Planned/frozen design intent
├── plans/                        Planned work and execution history
├── investigations/              Historical discovery
├── specs/                        Specifications with explicit lifecycle state
├── superpowers/                  Workflow records, not product documentation
├── reviews/ and lessons_learned/ Historical review/lesson records
└── governance/                   Durable governance decisions
```

### Lifecycle taxonomy

Document lifecycle and claim truth are separate fields. For example, a historical checkpoint can contain a valid **Measured** claim, and a current design can contain only **Planned** claims.

| Lifecycle | Meaning | Navigation treatment |
|---|---|---|
| `current` | Normative for the displayed release or integration reference. | Listed in the active section of `docs/INDEX.md`. |
| `branch-only` | Normative or implemented only on a named branch commit. | Listed separately with **Not shipped** and the comparison base. |
| `planned` | Approved or proposed intent that has not become runtime truth. | Listed under designs/plans; excluded from product navigation. |
| `historical` | Immutable chronology, evidence, investigation, or prior procedure. | Searchable through the historical index; never used as an unlabeled current source. |
| `superseded` | Replaced by a named current document or decision. | Retained for provenance with a `replaced-by` link. |
| `rejected` | Evaluated and explicitly not promoted. | Retained with its failed criterion and disposition. |
| `blocked` | Missing a policy decision, durable evidence, or executable proof. | Listed only where the blocking condition helps contributors; not linked as current authority. |

New or materially revised normative documents MUST carry, directly or through an unambiguous index record:

- lifecycle status;
- claim truth state or allowed truth states;
- canonical owner/domain;
- release tag or full source commit;
- last checked date;
- replacement relationship where applicable.

Legacy archive bodies do not need bulk frontmatter edits. A directory-level classification in `docs/INDEX.md` and a short directory banner are sufficient until a legacy file is materially revised.

### Active-versus-historical boundary

- Active product navigation includes only `current` shipped/ref-pinned material and separately labeled `branch-only` material.
- Plans, designs, investigations, checkpoints, reviews, lessons, and workflow artifacts are historical or intent collections by default, even if recently modified.
- Product pages MAY cite a historical artifact only under a visibly labeled **Dated evidence**, **Design intent**, or **Historical context** heading.
- No search result, relative link, or same-directory placement may elevate a historical file into an active owner.
- A superseded document MUST point forward. The current owner SHOULD NOT copy its historical body.

## Cutover mechanics

The reliability cutover is a sequence of repository validity states. These phases define what must be true at each boundary; they do not prescribe task ownership or implementation order.

### Phase 1: Authority boundary exists

`docs/INDEX.md` and `docs/VALIDATION.md` exist and declare their scopes. The index records the target release or full integration commit and separates branch-only material from shipped material. Validation policy no longer depends on contradictory root-file prose.

This phase is not complete if a canonical owner is merely named while a competing page still claims the same authority.

### Phase 2: Routing surfaces are cut over

`README.md`, `CONTRIBUTING.md`, `AGENTS.md`, and `CLAUDE.md` link to canonical owners. Duplicated validation matrices, benchmark acceptance rules, skill bodies, model inventories, configuration tables, and environment inventories are removed from routing surfaces.

All executable skills reside under `.agents/skills/`. Content currently acting as a skill under `docs/skills/` is either consolidated into `.agents/skills/` or converted into ordinary non-executable documentation. The missing `.skills/` path is not retained as an alias, symlink, or compatibility root.

The coherence-policy conflict is resolved fail-closed: retired `scripts/coherence-gate*.sh` commands are not acceptance evidence. A script’s continued presence does not authorize it. Path-specific validation routes come only from `docs/VALIDATION.md` and their executable backends.

### Phase 3: Copied facts become projections

Machine-readable or executable sources own model, configuration, environment, CLI, architecture-ID, script-command, path, and link facts. Documentation either:

1. contains a generated block tied to the source reference; or
2. contains a manually authored explanation whose factual tokens are checked in CI.

A page MUST NOT call itself generated unless a generator and reproducible source reference exist. If a fact cannot yet be generated or checked, it carries `last checked` and cutover-target source metadata and is not duplicated elsewhere.

### Phase 4: Historical boundary is explicit

`docs/perf-checkpoints/` is declared an append-only dated evidence ledger. New dated records MAY be added; an existing checkpoint file MUST NOT be modified or deleted. A correction is a new, separately dated amendment file that links to the unchanged original; an amendment never authorizes mutation of the original.

Plans, designs, investigations, reviews, specs, and workflow records receive directory-level lifecycle classification. The cutover does not rewrite their bodies and does not infer active state for unclassified legacy files.

### Phase 5: Enforcement is mandatory

Drift, ownership, link, skill-root, lifecycle, checkpoint-immutability, and evidence-schema checks are required on documentation changes. If authority cannot be established, the claim MUST be omitted or explicitly marked blocked. Branch and source-ref metadata are permitted only after the exact claim is verified against its required source and tests; unavailable automation does not lower that authority requirement.

The critical authority switch is atomic at completion: there MUST NOT be a final state in which the new owners exist while stale coherence-gate authority, a second executable skill root, or unlabeled branch claims remain active.

## Automation and checks

Automation MUST validate facts that code or repository structure can decide. Human review remains responsible for semantic scope and claim strength.

| Check | Machine source | Checked projection | Failure condition |
|---|---|---|---|
| Model registry parity | `cli/registry.json` and generated registry artifact | Model tags, default artifact, declared download/VRAM fields in MODELS and README | A copied value differs, an entry is presented as supported without a matching scoped runtime fact, or a branch-only model fact is unlabeled. |
| Configuration parity | Runtime configuration definitions/schema | Keys and defaults in CONFIG; examples elsewhere | Unknown key, wrong default, invalid enum, or duplicated normative table. |
| Environment inventory | Source scan of recognized environment variables | Generated inventory in `docs/env-vars.md` | Runtime variable missing from inventory, removed variable presented as active, or generated/manual sections are indistinguishable. |
| CLI surface | Parser and target binary help output | `docs/CLI.md` command/flag blocks | Command, flag, default, or subcommand path cannot be reproduced at the page’s pinned ref. |
| Validation routes | Executable script inventory and help/argument surface | Route table in `docs/VALIDATION.md` | Missing script, stale argument, route without evidence output, or another active page claims a conflicting route. |
| Repository paths | Target-ref filesystem | Paths in INDEX, product docs, skills, and validation docs | Missing path, wrong skill root, or link to `.skills/`. |
| Markdown links and anchors | Target-ref documentation graph | All active documents | Broken relative link, missing anchor, or active link that bypasses the canonical owner. |
| Skill topology | Repository tree | `.agents/skills/` plus routing indexes | Executable skill found outside `.agents/skills/`, duplicated skill ID, or root guidance contains a copied executable workflow. |
| Lifecycle coverage | `docs/INDEX.md` ownership registry and directory banners | Active and historical collections | Active page lacks reference/status, archive collection is presented as product docs, or superseded file lacks a forward link. |
| Checkpoint immutability | Version-control diff plus checkpoint schema | `docs/perf-checkpoints/` | An existing checkpoint file is modified or deleted, or a checkpoint is labeled a current default. Corrections are separate dated amendment files and do not relax this rule. |
| Evidence completeness | Evidence-record schema | New numerical claims and checkpoint entries | Required identity, raw samples, correctness, disposition, or durable location is absent. |
| Branch scope | Comparison of documentation target against integration base | Product and branch-only callouts | A fact introduced only on a feature branch appears as unqualified shipped behavior. |
| Architecture narrative paths | Runtime crate/module inventory | ARCHITECTURE and architecture IDs | Named current crate/path/ID is absent at the pinned ref. |

Generated blocks SHOULD be stable and reviewable: deterministic ordering, explicit source path, source commit, generation command, and a marker forbidding manual edits. Check-only blocks SHOULD report the exact divergent tokens rather than regenerate unrelated prose.

Semantic review MUST additionally reject:

- promotion language without an admission decision;
- present-tense capability copied from a plan;
- route attribution assembled from separate reports;
- historical numbers presented without date and fixture;
- unknowns rewritten as confident prose;
- an opt-in branch predicate presented as a default.

## Evidence policy

### Evidence classes

Documentation MUST distinguish:

1. **Runtime fact:** behavior proved by source/tests at a pinned reference.
2. **Certification policy:** rules owned by VALIDATION or a specialized policy guide.
3. **Dated measurement:** exact-fixture observation with durable evidence.
4. **Design intent:** approved or frozen future contract.
5. **Inference:** a reasoned interpretation explicitly labeled as such.
6. **Unknown/blocked:** a missing authority or proof that cannot be promoted.

Only the first three can support active operational claims, and each remains limited to its own predicate. A certification policy says what proof is required; it does not supply the proof.

### Required measurement identity

A numerical claim MUST record, as applicable:

- full source commit and dirty/clean state;
- comparison base and candidate identity;
- binary or daemon digest;
- model artifact path plus content digest;
- GPU/device identity and relevant architecture capability;
- driver and runtime version;
- deterministic prompt or token-stream digest;
- route-affecting configuration and environment;
- warmup policy;
- process freshness, run order, run count, and aggregation method;
- every raw sample, not only a summary statistic;
- correctness or parity evidence appropriate to the path;
- route proof when the claim attributes timing to a route;
- durable immutable raw-artifact location;
- disposition: accepted, rejected, experimental, or blocked.

A path under `/tmp`, an ignored `.remember/`, `.codeinsight+research/`, or `.superpowers/` directory, a workstation-only directory, or a remote-only ledger is a discovery pointer, not a durable canonical record. If a durable copy is absent, the claim remains local or blocked even when the artifact is currently readable.

### Measurement is not admission

- A benchmark above a threshold does not admit a route unless the applicable policy records admission.
- Correct output does not prove the intended optimized route ran.
- Kernel-symbol presence does not by itself prove the timed product arm used that route.
- A microkernel win does not imply end-to-end product improvement.
- A result for one model, quant, GPU, prompt length, or environment does not transfer to an adjacent cohort without separate evidence.
- A rejected candidate’s record remains immutable and MUST retain its rejection.

### Performance checkpoints

`docs/perf-checkpoints/` is an **append-only dated evidence ledger**:

- Every record is historical from the moment it is written and is bound to its recorded fixture.
- New records do not update, overwrite, or silently supersede older records.
- A checkpoint MUST NOT declare itself the current baseline merely because it is the newest file.
- README and BENCHMARKS MAY cite a checkpoint only with date, fixture, lifecycle label, and disposition.
- A current product baseline requires a separate, ref-pinned owner and an explicit admission/revalidation policy. It cannot be inferred from the checkpoint tree.
- Commit or PR evidence remains subject to the same manifest standard; storage in a commit body does not repair missing identity or raw samples.

### Validation evidence

`docs/VALIDATION.md` MUST map claim classes to path-specific routes. At minimum, the architecture recognizes distinct roles for:

- `scripts/gates.sh` as the live wrapper for the routes it actually invokes, not as a universal proof;
- `scripts/serve_harness.py` for user-visible serve behavior within its supported modes;
- `scripts/lfm_serve_harness.py` for LFM framing/thinking/semantic behavior within its supported fixture;
- `scripts/redline_daemon_harness.py` for its Redline retained-launch evidence;
- specialized parity, state, performance, and route-proof requirements named by the applicable design or policy.

Automatic pre-commit checks and manual acceptance evidence MUST be listed separately. Passing the commit hook MUST NOT imply that manual model, route, or performance validation ran.

Retired coherence scripts MUST NOT appear as current acceptance evidence. A campaign-scoped instruction not to use them, such as the LFM prefill campaign rule, MUST NOT be generalized into a replacement universal gate; the correct replacement is a path-specific route in VALIDATION.

## LFM and Redline examples

### LFM: plan, implementation, measurement, and admission remain separate

The audited LFM state demonstrates why the truth model is required:

- The active session plan specifies a Q8-first 350M vertical slice, later cohort admissions, and a default-on flip only after Phase 4 certification. Its displayed baseline table is explicitly exploratory and non-reportable until reproduced under the reportable protocol. Those are **Planned** claims.
- The current audited branch runtime instead gates the optimized path to the exact frozen 350M dense MQ4 fixture, on gfx1201, behind explicit opt-in `HIPFIRE_LFM2_PREFILL_BATCH=1`. This is a **Branch-implemented** claim at `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`, not a fact about `origin/beta@9ffb18da9d1377dfbf759db82641ea039b2e522e` and not a product-admission decision.
- The branch implementation order therefore diverges from the Q8-first plan. Documentation MUST preserve both facts rather than rewriting either one to make them appear aligned.
- Commit-bound MQ4 measurements lack parts of the complete durable performance ledger required for a current canonical benchmark. They MAY be cited as limited historical evidence with that limitation; they MUST NOT become current throughput baselines or proof for Q8, 1.2B, 8B-A1B, another quant, or another GPU.
- The plan’s Phase-4 default-on state has not been established by the supplied evidence. Product documentation MUST retain opt-in wording for the branch predicate and MUST NOT announce the planned default flip.
- The LFM campaign’s no-coherence-gate directive is campaign-scoped. Applicable evidence is the specified numerical/state parity, final-token behavior, LFM serve harness, and fresh-process performance protocol—not a retired coherence script and not a project-wide replacement rule.

Allowed branch-scoped wording:

> **Branch-only; not shipped.** On `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`, the runtime contains an opt-in batched-prefill path gated to the exact 350M dense MQ4 fixture on gfx1201. This statement does not apply to `origin/beta`, Q8, other LFM cohorts, other GPUs, a product-admission decision, or a default-on configuration.

Forbidden wording includes “LFM has batched prefill,” “LFM Q8 prefill is complete,” “LFM batched prefill is the default,” or any multi-cohort performance generalization.

### Redline: route evidence cannot be stitched

On the audited branch, `docs/REDLINE.md` defines the normative Redline procedure and separates runtime admission from certification. The file and its discovery hook are themselves branch-only relative to `origin/beta`, so repository-wide or release-facing pages MUST name the branch until integration.

A Redline timing attribution requires positive timed-arm route proof in the same report as the timing:

- A product timing report plus a separate manual capture or parity report MUST NOT be stitched into a route-certified result.
- A fresh local report stored under ignored or `/tmp` paths remains candidate evidence until it has a durable record and the guide’s route proof.
- A changed retained-launch tape, launch count, artifact, or route predicate is a distinct fixture; a newer count MUST NOT overwrite an older checkpoint’s identity.
- If the current product tooling cannot expose the required positive timed-arm proof, the Redline attribution is **blocked**, even if the underlying timing and manual capture are independently credible.
- The recorded LFM Stage-A launch-count reduction remains a rejected standalone promotion and non-Redline result: it missed the predeclared wall-time promotion gate and no PM4 route was run. Documentation MUST preserve that disposition rather than headline the structural reduction as a shipped optimization.

Allowed wording:

> The candidate has separate product timing and manual route evidence, but current tooling cannot provide the required positive timed-arm proof in one report; Redline attribution remains blocked.

Forbidden wording includes “route-certified Redline timing,” “Redline caused the measured gain,” or “the product arm used retained replay” when the same timed report does not prove that route.

## Migration and rollback

### Migration rules

- Canonical content is moved or consolidated, not copied. A former mirror becomes a link or is removed.
- Executable skill content is consolidated under `.agents/skills/`; references are updated in the same cutover. No `.skills/` or `docs/skills/` compatibility execution root remains.
- `AGENTS.md` and `CLAUDE.md` retain only hard repository rules that genuinely apply at that surface, provenance routing, and links to `docs/INDEX.md`, `docs/VALIDATION.md`, and `.agents/skills/`. They do not retain benchmark tables or validation playbooks.
- Stale product facts are replaced by checked projections from cutover candidate `C`. Where no reliable projection exists, the claim is removed or marked blocked rather than guessed.
- Existing perf checkpoint files are preserved byte-for-byte. Corrections are added only as separate, newly dated amendment files that link to the unchanged originals. Migration does not aggregate or normalize their numbers.
- Legacy plans and investigations are classified at the directory/index level. Their bodies are not rewritten merely to add modern terminology.
- Branch-only LFM and Redline material carries full-ref labels until the integration target contains it. The labels are removed only after checking the new integration ref, not automatically when a branch merges.
- Broken links to missing roots or scripts are removed; they are not redirected to a superficially similar command.

### Rollback rules

Rollback is fail-closed and preserves evidence:

1. If an ownership or generation cutover fails, product pages revert to the last known-good ref-pinned wording or remove the disputed claim.
2. The owner page, all routing changes, and its enforcing check roll back as one unit; a competing mirror MUST NOT be reactivated as a temporary authority.
3. Rollback MUST NOT restore known-invalid coherence-gate claims, the dead `.skills/` root, or unlabeled branch behavior.
4. Append-only checkpoints and rejection records are never deleted to simplify rollback.
5. If branch integration changes the relevant predicates, LFM or Redline claims return to `blocked` or `branch-only` until rechecked against the new full commit.
6. A failed generator may temporarily force a manually checked, ref-pinned block with explicit `last checked` metadata; it may not leave stale generated labeling in place.

## Verification

Verification evaluates the documentation cutover itself. It does not certify the runtime campaigns discussed by the documents.

### Structural verification

- From fresh detached checkouts of candidate `C` and attestation `A`, every active page and anchor linked by README, CONTRIBUTING, AGENTS, CLAUDE, INDEX, and VALIDATION resolves with identical substantive results.
- `.agents/skills/` is the only tree containing executable skill definitions; no active link targets `.skills/`, and no executable skill remains under `docs/skills/`.
- INDEX names exactly one canonical owner per concern and classifies all top-level documentation collections.
- VALIDATION contains exactly one route-selection table and every listed executable path exists at the pinned ref.
- Root routing surfaces contain no copied validation matrices, numerical baselines, environment inventories, model tables, or executable skill bodies.

### Fact verification

- Regenerated or checked model facts match `cli/registry.json` and the registry artifact at candidate `C`.
- Configuration examples use keys, enums, and defaults accepted by the runtime definitions at `C`.
- The environment inventory includes every recognized runtime variable and marks manual annotations separately.
- CLI command and validation-script examples match parser/help output built from `C`.
- Architecture and repository paths exist at `C`.
- Branch-diff checking identifies facts absent from the integration base and verifies that each is labeled branch-only.

### Evidence verification

- A schema check rejects new numerical records without required identity, samples, correctness, durable location, and disposition.
- A version-control check unconditionally rejects modification or deletion of an existing checkpoint file. Corrections must be separate dated amendment files linked to the unchanged original.
- A semantic review maps at least one shipped/ref-pinned claim, one branch-implemented claim, one measured claim, one planned claim, and one blocked claim to their stated authorities.
- No checkpoint or historical benchmark table is labeled a current default without a qualifying immutable admission record in `docs/admissions.yml`.

### Campaign-boundary verification

- LFM product prose is checked against the integration ref and does not inherit Q8-first, multi-cohort, or Phase-4 intent from the plan.
- Any branch LFM callout names the exact 350M dense MQ4, gfx1201, opt-in predicate and full branch commit; it does not generalize.
- Any Redline timing attribution links a single report containing both the timing and positive timed-arm route proof. Without it, wording is blocked or candidate-only.
- No active documentation treats any retired coherence-gate script as acceptance evidence.

### Candidate and attestation identity

Closeout uses two commits because tracked content cannot contain its own Git object ID:

1. Candidate `C` contains every substantive cutover change. Only the predeclared closeout values may remain pending: the INDEX candidate reference is null, this design remains `planned`, and the acceptance record is absent.
2. Attestation `A` has exactly one parent, and `parent(A) = C`. Neither `C` nor tracked content in `A` names `A`.
3. The `C..A` diff is field-allowlisted. It may only fill the human and machine INDEX candidate-reference fields with `C`, update this design's predeclared candidate/lifecycle/replacement metadata without changing its body or criteria, and add one schema-valid acceptance record whose evidence is bound to `C`.
4. `A` MUST NOT change code, checks, tests, workflows, admissions, product truth, owner mappings, validation policy, historical evidence, or any other file or field. Attestation evidence cannot make a substantive predicate pass.
5. All substantive predicates run against fresh detached checkouts of both `C` and `A` with identical results. Attestation-only checks on `A` prove parentage, exact recorded references, the field-level diff allowlist, acceptance-record integrity, final lifecycle metadata, and external CI bound to `A`.
6. `C` and `A` are an inseparable closeout pair. Rebasing, amending, squashing, or cherry-picking either invalidates the attestation and requires a new pair plus complete re-verification.

## Binary acceptance criteria

The cutover is accepted only if every row passes on attestation `A`. Substantive predicates MUST also pass identically on candidate `C`; no `A`-only metadata may make them pass. `docs/INDEX.md`, this design's closeout metadata, and the acceptance record in `A` MUST all name the same full `C`; the audited reference, comparison base, and `A` cannot substitute for it.

| # | Pass condition |
|---:|---|
| 1 | `docs/INDEX.md` exists and names exactly one owner or one explicit blocked record for every concern in the canonical ownership table. |
| 2 | `docs/VALIDATION.md` is the only human-facing owner of validation-route selection, and every route points to an executable path or an explicit blocked condition. |
| 3 | No active page calls `scripts/coherence-gate*.sh` canonical, mandatory, current, or valid acceptance evidence. |
| 4 | `.agents/skills/` is the sole executable skill root; `.skills/` has no active references and `docs/skills/` contains no executable skill definition. |
| 5 | `AGENTS.md` and `CLAUDE.md` are thin routing/rule surfaces with no copied validation matrix, benchmark baseline, or skill body. |
| 6 | Active product pages display their release/integration scope; every feature-branch fact displays `branch@full-commit` and **Not shipped**. |
| 7 | Every new numerical claim has the required durable manifest, raw samples, correctness evidence, and disposition. |
| 8 | `docs/perf-checkpoints/` is declared append-only dated evidence, and no checkpoint is presented as an automatically current baseline or default. |
| 9 | Every pre-existing checkpoint file and rejected disposition is byte-for-byte unchanged; any correction is a separate dated amendment linked to the original. |
| 10 | Plans, designs, investigations, specs, reviews, and workflow artifacts are excluded from active product truth unless an exact claim is independently proved at the product reference. |
| 11 | Unknown or blocked claims are omitted from product prose or explicitly labeled with the missing authority; none is promoted by inference. |
| 12 | Model, configuration, environment, CLI command, validation command, architecture-ID, repository-path, and link facts are generated or checked where a machine source exists. |
| 13 | No product or active agent document claims generic LFM batched prefill, completed LFM Q8 prefill, multi-cohort LFM admission, or a Phase-4 default-on state. |
| 14 | Any LFM branch claim is limited to the exact audited 350M dense MQ4, gfx1201, opt-in predicate and states that it is absent from the comparison base. |
| 15 | No Redline performance statement is called route-certified unless its own timed report contains positive timed-arm route proof; separate artifacts are never stitched. |
| 16 | The rejected LFM Stage-A result remains rejected, below its promotion gate, and non-Redline in every summary that mentions it. |
| 17 | All active internal links, anchors, skill paths, script paths, and command names resolve identically at candidate `C` and attestation `A`. |
| 18 | Documentation checks fail when a machine-owned fact drifts, a second canonical owner appears, a branch label disappears, or an existing checkpoint is silently rewritten. |
| 19 | README and BENCHMARKS label historical performance snapshots by date and fixture; a current claim requires a matching `docs/admissions.yml` record and complete evidence, otherwise the concern is explicitly historical or blocked. |
| 20 | The final repository contains no unresolved active contradiction between INDEX, VALIDATION, product docs, routing surfaces, specialized guides, and executable sources. |

Failure of any criterion leaves the documentation cutover incomplete. The safe fallback is ref-pinned, explicitly limited, or blocked wording—not a broader claim.