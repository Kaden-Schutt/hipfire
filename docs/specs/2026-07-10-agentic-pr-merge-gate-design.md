# Agentic PR Merge-Gate — Design Spec

> Status: **DRAFT / approved-in-brainstorm** · Date: 2026-07-10 · Branch: `feat/rdna-kernel-oracle`
> Author: Kaden Schutt
>
> An agent-orchestrated, GPU-backed PR pipeline that reads a PR, tests the
> behaviors it touches on the RDNA fleet (gfx1100 / gfx1151 / gfx1201) via
> `serve_harness`, renders a deterministic parity/perf/coherence verdict, and —
> when clean, non-clobbering, and meaningfully helpful — lands it (auto for
> Kaden's PRs; agent-executed on a maintainer's `@claude /merge`), stacking
> approved PRs on a `staging` merge-train so PR debt never accumulates.

## 1. Goal

Replace "Kaden manually approves every merge" with an agent that certifies each
PR on real hardware and either lands it or returns an actionable **Bill of Debt
(BOD)**. First real use: certify the `feat/rdna-kernel-oracle → master` merge
across all three archs before it lands.

## 2. Non-goals

- **No new perf-search.** This gate *certifies* PRs; it does not hunt kernels
  (that is the autoresearch loop). It reuses the loop's `certify/` engine.
- **No maintainer permission change.** The **agent** (Claude App, write scope)
  performs every merge; maintainer GitHub permissions stay exactly as they are.
- **No Vulkan / no spec-decode targeting** — inherits the repo's standing rules;
  the gate is arch/behavior-agnostic and does not privilege any lever.
- **No running untrusted fork code with secrets** — the trust boundary from
  `claude-review.yml` is preserved (fork PRs never auto-run on GPU hardware).

## 3. Where it sits (tier architecture)

Completes the pipeline `ci.yml` and `claude-review.yml` already reference:

| Tier | Workflow | Runner | Role | Status |
|---|---|---|---|---|
| 1 | `ci.yml` | ubuntu | build + `test --lib` | exists, required |
| 2 | `claude-review.yml` | ubuntu | **dispatcher + interpreter** (GPU-free): read diff → select tests → trigger Tier 3 → interpret results → verdict/merge/BOD | exists (re-enable + elevate) |
| **3** | **`gpu-gates.yml`** | **self-hosted matrix** | **deterministic `ar gate`**: fit → cross-arch → parity → perf → coherence → non-clobber merge | **new** |

**Division of labor (agentic-first).** Claude is the *brain*; **codex on the
owning box is the hands**; the deterministic `gate_cell` math is the *ruler*. The
per-PR flow, actor by actor:

```
CLAUDE classify + route (§8)  →  CLAUDE writes codex directions (§8)
  →  codex exec -p "[directions]"  on the box that OWNS the affected arch:
       cargo build --release --features deltanet -p hipfire-runtime --example daemon
         @base_ref AND @head_ref  (cache by sha),
       run scripts/serve_harness.py parity/perf/coherence per (model, arch),
       invoke the deterministic gate_cell to GRADE the measurements  →  results/<arch>.json
  →  CLAUDE decide_pr (§8/§12)  →  CLAUDE PR comment (§9)
  →  codex exec  Gate-4 non-clobber merge + merge-fix (§10)  →  staging train (§11)
  →  CLAUDE automerge (§12, Kaden-only, behind GATE_AUTOMERGE)
```

Two things that keep this honest: **codex RUNS `gate_cell`, it never *judges*
parity** — the parity/perf/coherence math is deterministic (`certify/verdict.py`,
reused from the loop), so codex reports token-ids / tok-s / detector output and the
math decides PASS/FAIL. And **serve_harness is `scripts/serve_harness.py`
(committed) — codex does NOT build it**; the only on-box build is the daemon,
per ref. **Claude orchestrates; the deterministic core decides the raw verdict.**

## 4. The gate pipeline (per eligible arch, cheapest-first, short-circuiting)

| Gate | What it does | Reuses | Fail → |
|---|---|---|---|
| **0 · fit/smoke** | target model loads + runs one generation on that arch's GPU | `LiveServeRunner` spawn | **N/A skip** (not a failure) — unless the change *requires* that arch, then hard-fail |
| **1 · cross-arch** | each changed `kernels/src/*.hip` does not alter another arch's device codegen (byte-exact preprocessor invariance) | `cross_arch.check_cross_arch` | reject — "arch-bleed; `#if`-gate the gfxNNNN path" |
| **2 · parity** | token-exact greedy vs base (FP32 + deterministic) | `verdict.parity_result` | reject |
| **3 · perf** | no significant, replicated regression (see §6) | `verdict.perf_result` + `perf.mwu` | reject + offender recommendation |
| **3b · coherence** | attractor / validator / McNemar vs base | `verdict.coherence_result` | reject |
| **4 · non-clobber merge** | merge PR into the **staging tip**, re-run 3 + 3b on the merged tree | §10 | codex merge-fix → re-fire; still fail → **BOD** |

Gate 0 is per `(model, arch)` cell; a cell that does not fit is skipped as N/A,
never a failure. "If any *eligible* arch regresses → reject."

### 4.1 Arch → box deferral (neither box parity-fails on an arch it can't run)

The GPU fleet is **two** boxes: **hipx owns {gfx1100, gfx1151}**, **hiptrx owns
{gfx1201}**. Claude's `decide` step computes the PR's **affected archs** from the
diff. Each box gates only `affected_archs ∩ box_archs`:

- Non-empty → the box builds + runs the real gate for those archs.
- **Empty → the box DEFERS: a no-op PASS, not a run.** hipx defers a gfx1201-only
  change to hiptrx; hiptrx defers a gfx1100/1151 change to hipx. A box never spawns a
  daemon for an arch its GPUs don't provide, so it can **never spuriously parity-fail**
  on a non-owned arch (the 70 ms empty-output REJECT the scaffold produced).

The **cross-arch leak** check (Gate 1) is orthogonal and still fires everywhere: a
gfx1201 change that alters the gfx1100 device codegen is a *real* REJECT on hipx —
that is arch-bleed, not a deferral. Deferral skips the *battery* for a non-owned
arch; it never skips the isolation check for an owned one.

## 5. Model × arch fit map (config-driven)

`autoresearch/config/pr_gate.toml` declares, per SKU, which archs it fits. The
gate runs only fitting cells.

| SKU | gfx1100 (24 GB) | gfx1151 (96 GB) | gfx1201 (32 GB) | Notes |
|---|---|---|---|---|
| `qwen3.6-27b` (dense) | ✅ | ✅ | ✅ | canonical |
| `qwen3.6-a3b` (MoE) | ✅ | ✅ | ✅ | canonical |
| `deepseek4` (EP) | ✗ | ✅ | (EP-4 multi) | example: DS4 change → gfx1151 only |

The **canonical battery** (`qwen3.6-27b` + `qwen3.6-a3b`) always runs on every
arch it fits; a change that touches a specific model/arch **adds** that cell
(Claude selects it — §8). Auto-merge requires clean + non-clobber on the
canonical battery **and** on whatever the change touches.

## 6. Perf governance (two mechanisms, two reference frames)

**Baseline B — per-arch × SKU high-water master perf** (3-run-avg decode tok/s).
Stored in `autoresearch/state/pr_gate_baselines.json` (tracked). Moves **up only**
on a confirmed improvement that also passes parity + coherence; **never down**.
It is the restoration target for the drift guard.

### 6.1 Per-PR perf gate — reject any significant, replicated regression

The mirror of the loop's WIN gate. Uses `verdict.perf_result` (conjunctive
Mann-Whitney): the loop declares WIN iff tok/s↑ AND duration↓ both significant;
the gate declares **REGRESSION** iff **tok/s↓ AND duration↑ both mwu-significant
(α=0.05)**, beyond the existing `perf.FLOOR = 0.15%`, measured **head-vs-base
interleaved in the same run** (base = the staging tip, §11), **adaptive-sampled**
(min ~3 up to `perf.CAP = 16`) so significance requires a real effect at a
practical sample budget. (The stored baseline B and the drift-guard master
measurement, §6.2, use a fixed **3-run average** — the stable "last measured
master" number; the per-PR *significance* test is the adaptive interleaved A/B.)

- **Not significant** (within the ±1–3% noise band) → **PASS** (no magnitude
  threshold; significance is the discriminator, so noise passes automatically).
- **Significant regression on the first pass** → **confirmation rerun** (fresh,
  warmed). If the rerun is not significant → noise → **PASS**. If it replicates →
  **REJECT** + offender recommendation (§9).
- **Improvement** (mwu WIN) + parity + coherence → **PASS**; on landing this
  raises B (§6.2).
- The `0.15%` floor is belt-and-suspenders against large-N nitpicking; the
  `CAP=16` sample budget already means only effects big enough to be significant
  at n≤16 (realistically ≳0.5%) ever flag.

### 6.2 Longitudinal drift guard — −3% cumulative vs high-water B

Every *significant* single-PR regression is already rejected, so the only thing
that can erode master is the slow bleed of *individually-insignificant*
regressions. On every landing (§11), the merged master tip is re-measured per
arch:

- If a new high → **B resets up** (drift budget resets).
- If the tip has drifted **≤ −3% below B** → **auto-open an investigation**:
  identify the PRs merged since the last B-reset that each passed at
  sub-significance (the accumulators), and post a recommendation to **restore to
  B** (named bisect/revert candidates). Runs as an `ar`-driven job; output is a
  tracking issue + a BOD-style report, not an automatic revert.

## 7. Parity + coherence = the ar loop (unchanged)

Parity is token-exact greedy (FP32 + `HIPFIRE_DETERMINISTIC=1`). Coherence is
`detect_attractor` (tiered DFlash thresholds) + `run_validators` (genre-specific)
+ `mcnemar_worse` (paired base-vs-PR failure test). A perf improvement is
accepted **only if** parity + coherence also pass — the loop's exact contract.

## 8. Dispatch — Claude classifies + authors the behavior test

**Claude classifies the PR — not a path regex.** The dispatcher (Claude) reads the
diff **and the contributor's PR description** and *understands what it does*: a
rename vs a real behavior change, risk hiding behind an innocent-looking path,
whether the change is even reachable through the serve path at all. The
deterministic `classify_pr` (path taxonomy) is **only a floor** — it pins a
*minimum* risk tier that Claude may escalate above but never de-escalate below, so
a kernel/dispatch/forward-pass change can never be mis-classified as trivial even
if Claude errs or is unavailable. Claude's semantic read is the authority; the
regex is the safety net.

Claude's test plan has **two parts**:

**1. serve_harness floor (deterministic, always runs).** The canonical
coherence+perf battery over the **serve path** — parity / perf / attractor /
McNemar on `qwen3.6-27b.mq4` + `a3b.mq4r`/`a3b.mq4p`, per fitting arch (gates 2/3/3b
of §4). This is the hard regression gate: it catches anything that breaks or slows
the serve path, and it runs regardless of Claude's plan. Claude may *add* coverage
but never remove this floor.

**2. Bespoke behavior test (Claude-authored, codex-executed) — the fix for the
serve_harness blind spot.** `serve_harness` only exercises the *serve path*; it
**cannot** reach a new CLI command, a tokenizer / quant-format change, a
build/tooling change, or any behavior that never hits `serve`. For those, Claude
writes a **bespoke test prompt** describing exactly what to verify ("this PR adds
`hipfire foo --bar`; build it, run it on <input>, confirm <X>"), and **codex
executes it generally on-box** — build, run the new path, check the output — **not
limited to serve_harness**. Without this, a non-serve behavior would be untested
and the gate would false-pass; this closes that hole. Claude passes this prompt
**in addition to** the serve_harness bench, never instead of it.

Plan shape:
```
{ risk: "trivial|low|moderate|high-risk",
  serve_floor: { models, archs, genres, long_context, session_files, perf_ab },
  behavior_tests: [ { what, prompt, expect, harness, model, effort } ],
  reason }
```

**Verdict combination.** The serve_harness floor is the deterministic PASS/FAIL and
is never overridden. Each bespoke behavior test contributes a codex pass/fail for
its specific behavior. A PR passes iff the floor is green **AND** every bespoke
test passes. (The floor's rigor is deterministic; the bespoke tests are the best
available agentic check for behaviors that are not deterministically gateable.)

**Degraded-coverage safety.** If Claude/codex errors or times out, Tier 3 falls
back to the serve_harness floor + the `classify_pr` floor tier and **flags**
degraded coverage — it never blocks and never false-passes on the floor.

### 8.1 Executor routing (Claude tiers codex by the risk it read)

Having classified the diff semantically, Claude sets `(harness, model, effort)`
for each bespoke behavior test's on-box executor via `agent_exec`. The
`classify_pr` floor pins the **minimum** tier; Claude escalates from there. The
table lives in `pr_gate.toml`, re-tuned from the ledger's per-tier
false-pass/false-reject rates:

| risk (Claude's read; floored by `classify_pr`) | codex tier |
|---|---|
| trivial | *none* — serve_harness floor only |
| low | codex **luna high** |
| moderate | codex **terra high** |
| high-risk | codex **sol xhigh** (+ **grok** on the gfx1201 arm as a diversity second-opinion; disagreement → human, no auto-merge) |

codex authenticates **box-local** on hipx/hiptrx (not a GitHub secret). pi.dev
slots in as a fourth harness when added (§Phase 6).

**Codex usage-limit resilience.** codex exec can refuse a round when the box's
codex account is out of usage. `agent_exec.run_round_resilient` (the seam every
gate codex round runs through — behavior tests §8 *and* the Gate-4 merge-fix §11)
distinguishes that refusal (nonzero exit **plus** a usage-limit marker in codex's
output) from a genuine task failure and reacts by tier:

- **non-sol** (luna/terra) round → **fall back to grok** immediately with the
  identical prompt (grok is a fine substitute at these tiers; `$GATE_GROK_MODEL`,
  default `grok-4.5`). grok exists on hiptrx; on a box without grok the fallback
  round simply fails and the PR punts — never a false pass.
- **sol** (`gpt-5.6-sol`, high-risk / merge-conflict) round → the top
  intelligence tier is *required*, so we do **not** degrade to grok. We **wait**
  `CODEX_RESET_POLL_SECS` (default 900s) and retry codex, up to `CODEX_MAX_WAITS`
  (default 8) times, until codex resets. Exhausting the wait budget returns
  codex's failing rc (the PR punts) — a sol requirement is never silently
  downgraded to a lower tier.

A genuine codex error (nonzero exit **without** a usage marker) is returned as-is
and never masked by a model swap.

## 9. Offender location → contributor recommendation

`certify` runs per kernel, so a regression is attributed to the specific kernel
whose perf arm went negative. Claude interprets that attribution and writes,
e.g.:

> ❌ **gpu-gate / gfx1201 — perf regression −3.2%** (confirmed, 2 runs).
> Offending change: `kernels/src/gemv_hfq4g256_moe_down…hip`. Parity ✅
> coherence ✅ — this is perf-only. Please adjust it to be perf-neutral to
> gfx1201 (arch-gate the gfx1201 path, or restore the prior schedule).

When the diff touches several kernels, codex/grok on-box bisects by re-running
`ar gate` over kernel subsets to isolate the offender before Claude writes the
recommendation.

## 10. Gate 4 — post-merge anti-clobber validation, codex merge-fix, BOD

**A git-clean merge is NOT clobber-free.** Two PRs can merge with *zero textual
conflict* yet **semantically clobber** — PR-A changes behavior X, PR-B still assumes
old X → they merge clean and run broken. Textual-clean ≠ functional-clean. But Gate 4
does **not re-derive work already done.** The PR's own gate (earlier in the pipe)
already validated and **recorded** its behaviors — token-exact parity, the perf delta
vs master, coherence, and the §8 behavior-test verdicts; the master baseline is known.
Post-merge, Gate 4 **recalls those recorded behaviors and confirms they REPRODUCE on
the merged tree** — it re-runs only the PR's already-established behaviors on the
merged result, **never the full PR gate and never a fresh master measurement**.

Three reference points, all **recall-based** (confirm reproduction, don't re-derive):

1. **vs its PR** — re-run the PR's *recorded* behavior tests + coherence on the merged
   tree; they must REPRODUCE (same pass) — the merge didn't break what the PR does.
2. **vs master** — measure the merged tree's perf against the **already-known**
   master/staging baseline (no re-measurement of master); it must not regress.
3. **vs the staging stack** — every folded PR's recorded behaviors reproduce
   *together* on the merged result (no PR clobbers another).

A recorded behavior that fails to reproduce on the merged tree = semantic clobber. The
merge stands only if all three reproduce.

- **Clobber (any of the three fails)** → Claude dispatches a **targeted codex
  merge-fix**, resolved on the agent-owned staging (§11), → **re-runs the functional
  gate on the re-merged tree** (re-validating all three points).
- **Fix succeeds** (clean at all three) → the PR folds onto staging.
- **Fix fails** → a **Bill of Debt (BOD)**: the itemized blockers — conflicting
  hunks, the regressing kernel, the failed coherence row, or *the PR whose behavior
  the merge broke* — the contributor must clear.
- **Documented-clobber exception:** when the post-merge check surfaces a behavioral
  interaction *resolved by a new command/feature* one PR introduces (not a true
  regression), the gate flags it for **documentation** rather than rejection; Claude
  records it in the landing ledger and requests a doc line.

**Landing re-validates too.** Because a single landing flushes the *whole* train to
master, the gate **re-runs the functional check on the landed master** before the
landing is final (§11) — the post-merge master is proven clobber-free vs every folded
PR and vs the prior master, never approved on a textual-clean git result alone.

## 11. Staging merge-train (debt prevention)

`origin/staging` = `master` + {all currently-approved-but-unlanded PRs}, a
**derived** branch, deterministically rebuilt from `master + the approved set`
(so a stale/dropped approval self-heals).

1. A PR that passes gates 0–3b is **folded onto staging**. Gate 4 trial-merges it
   against the **staging tip** (= master + prior approved stack). Clean → it folds.
   **On a clobber the pipeline RESOLVES it — it does not punt.** Because staging is
   a **derived, agent-owned** branch, the agent resolves **on staging**: it rebases
   the PR's commits onto the staging tip (a *mechanical* conflict — adjacent edits,
   a change master already made — auto-resolves this way, as the dry-run showed for
   #479), and escalates a *semantic* conflict to the codex merge-fix (§10). This
   works for **fork PRs too** — the agent never needs to push to the PR's own branch
   to stack it (the fork limitation applies only to Gate-4's *in-PR* fix). Only a
   conflict codex genuinely **cannot** resolve becomes a BOD, itemized with the split
   reason: *"rebase on master"* (stale) / *"conflicts with approved PR #X"* (a real
   stack interaction) — and the codex-authored resolution of a fork PR is recorded in
   the landing ledger so the contributor can review it.
2. Approved PRs **accumulate** on staging (they stay "open" on GitHub; content
   rides staging).
3. **Any landing event flushes the whole train:** Kaden's auto-merge **or** a
   maintainer `/merge` lands the **entire staging stack** on master in one
   non-clobber merge — merging one PR carries **all** currently-approved PRs with
   it. Folded PRs then **close behind it**.
4. staging **re-syncs to master** after every landing (§13) → never stale.

**Debt = only a conflict codex could not resolve (or a fork it cannot push a fix
to) — not any conflict.** A clobbering fold first goes through the Gate-4 merge-fix;
the pipeline **resolves before it punts**. Approved, resolvable PRs are always on
the train. The perf baseline for an individual PR is the staging tip, so the *stack
as a whole* cannot regress; on landing the master tip is re-measured for the drift
guard (§6.2).

**GitHub close semantics:** landing is a real (non-squash) merge so folded
commits become master ancestors; each folded PR is closed with a
`landed via staging stack → <master-sha>` comment + link. Where GitHub detects
the ancestry it shows "merged"; otherwise "closed (landed via stack)".

### 11.1 Backlog sweep — COLD START ONLY

The sweep is a **one-time cold-start** to clear the *standing* open-PR backlog that
accumulated before the gate existed — **not** a steady-state pipeline component. Once
the backlog is drained, the pipeline is per-PR (classify → codex gate → decide →
codex Gate-4 → staging → automerge); no recurring/scheduled sweep runs. `ar gate
--sweep` (via `gate-sweep.yml`, `workflow_dispatch`) drains the backlog in one pass
onto a **collection branch** — `feat/rdna-kernel-oracle` for this batch — which a
maintainer then lands as one non-clobbering stack. For each open eligible PR:

- **Gate it. A REJECT is *punted, not resolved*.** In particular a **perf regression
  is skipped and the sweep moves to the next PR** — perf is never auto-fixed. Only
  *merge conflicts* get the Gate-4 codex merge-fix during the fold; a functional gate
  failure (perf regression / parity / coherence) punts with a BOD.
- **Fold the approved onto the collection branch** — Gate-4 resolve-not-punt for
  conflicts, §10 recall-reproduce for post-merge validation.
- **Perf-supersede.** If a fold LOSES a perf gain another PR won ("perf superseded,
  but lost in merge"), the **perf-preserving branch wins** and supersedes the loser:
  gracefully (the merge-fix keeps both) if possible, else the **higher perf-delta**
  branch stays on the train and the loser is dropped (`superseded`).

The sweep returns `{train, punted, superseded}`. The train lands on master via §11's
`land_train` once the collection branch is **proven non-clobbering** (§10 recall-reproduce
on the landed result). A superseded loser is **deferred, not lost** — it can re-enter a
later sweep once rebased. (Core: `gate/sweep.py`, unit-tested; the GPU/git/codex bindings
are prod wiring.)

## 12. Authority & triggers

| Author | Gate acts? | On green pass |
|---|---|---|
| **Kaden-Schutt** | yes (open, non-draft) | agent **auto-merges** — only Kaden auto-merges, for now (behind the `GATE_AUTOMERGE` kill-switch). |
| **fivetide / unverbraucht / nwoolmer** | yes (open, non-draft) | agent posts **"✅ green — comment `/merge` and I'll land it"**; the maintainer comments **`/merge`** → the **agent merges on their behalf** (re-confirming green first). Agent never merges them unprompted. |
| **non-maintainer** | **only if a maintainer runs `/gate`** | on pass, the **invoking maintainer** comments `/merge` → agent merges on their behalf |
| **Draft (any author)** | no | on invocation → **verdict + BOD only**, never merges |

The gate **acts only** on maintainer-authored PRs, or when a maintainer runs
`/gate` on any PR — nothing else triggers it. Maintainer list + the
`auto_merge_authors` (Kaden only, for now) live in `pr_gate.toml`. Fork PRs never
auto-run (secrets withheld — the `claude-review` trust boundary).

**The `/merge` command (`gate-merge.yml`).** Maintainer GitHub permissions are
**unchanged** — a maintainer's `/merge` comment is their explicit approval, and the
**agent executes the merge on their behalf**. The handler re-confirms the
`gpu-gate / *` check-runs on the current head are ALL green before merging (never
relies on branch protection alone), so a maintainer can never land a red/stale
gate. Optionally also make `gpu-gate / *` **required status checks** on master as
belt-and-suspenders.

**"Meaningfully helpful" (the auto-merge judgment).** After all deterministic
gates pass, Claude gates auto-merge on a helpfulness check to keep no-op churn
off master. A PR is helpful iff it passes all gates **and** it is one of: a
measured perf/behavior improvement; a bug/correctness fix; a feature or
capability the contributor's description states; or scaffolding whose stated
purpose (enabling later work) is served and which is provably off the hot path
(no perf A/B needed per §8). It is **not** helpful if it is a pure no-op, a
revert-churn, or a change whose only effect is a (permitted, sub-significance)
regression. Helpfulness is a *merge* gate, not a *pass* gate — an unhelpful PR
still gets a green verdict + a "no-op: clarify intent" note rather than a BOD.

**Permissions:** the agent holds `pull-requests: write` + `contents: write` (in
the elevated `interpret` job and `gate-merge.yml`) — used to (a) auto-merge
Kaden's green PRs, (b) merge a maintainer's green PR **on their behalf when they
comment `/merge`**, and (c) post verdicts. Maintainers gain **no** new GitHub
permission — their `/merge` comment is the approval, the agent executes. Auto-merge
(the only *unprompted* merge) sits behind a repo kill-switch — the `GATE_AUTOMERGE`
variable (must equal `on`; default off posts "would auto-merge").

## 13. `ar gate` engine (the reusable core)

New `autoresearch.ar` subcommand, sibling to `certify`:

```
python -m autoresearch.ar gate \
  --arch <gfx> --base <sha> --head <sha> --pr <n> \
  --dev <N> --card <N> [--harness codex|grok] [--models qwen3.6-27b,qwen3.6-a3b,…]
```

On-box, per arch: resolve `pr_gate.toml` → for each fitting `(model, arch)` cell
run gates 0–3b via `orchestrator.certify(runner, base_daemon, var_daemon, …)`
with a `LiveServeRunner`, then Gate 4. Emits a self-describing verdict row
(`verdict.make_row` + the orchestrator's identity block: `gpu_arch, base_sha,
variant_sha, prompt_md5, kv, tok/dur deltas, measurement_hash`).

Everything except `LiveServeRunner` + rocprof is pure/no-GPU, so the engine is
**unit-tested with a mock `ServeRunner`** + injected `cross_arch.preprocess`
against captured-diff fixtures (extends `autoresearch/ar/tests`, runs under
`no-gpu-ci.sh`). Topology-agnostic: the same `ar gate` runs under an ssh-fanout
fallback if self-hosted runners ever prove painful.

## 14. `gpu-gates.yml` workflow

- **Trigger:** `pull_request` (opened/synchronize/reopened/ready_for_review) for
  in-repo branches with `HAS_TOKEN`; **+ `issue_comment`** for `/gate` and
  `@claude /merge` commands. Draft PRs gated out of auto-run.
- **Matrix:** `strategy.matrix.arch: [gfx1100, gfx1151, gfx1201]`,
  `runs-on: [self-hosted, <arch>]` → GH schedules each on the owning box (gfx1100
  + gfx1151 → hipx; gfx1201 → hiptrx). Each job runs `ar gate` and publishes a
  check-run `gpu-gate / <arch>`.
- **Concurrency:** GH group per PR (`cancel-in-progress`) **+ on-box
  `gpu-lock.sh`** so gate jobs serialize with the autoresearch loop and with each
  other (hipx runs both gfx1100 and gfx1151 → they queue on hipx's per-box lock,
  keeping co-resident perf measurements from perturbing each other). The gate is
  a lock *citizen* (never `rm`s the lockfile) but a **priority** one — it can
  preempt the background loop (§17) so a human waiting on a PR is never starved
  behind a 12h loop run.
- **Aggregator / interpret job** (`workflow_run: completed`): re-invokes Claude
  (Tier 2) to read the per-arch rows + drift state, post one PR comment (verdict
  table + ledger rows), set the aggregate status, and take the merge/tag/BOD
  action.
- **Freshness:** on every landing, sync `master` + rebuild `staging` on hipx and
  hiptrx (validators) and on the k9lin dev checkout (repo sync only; k9lin stays
  zero-validation).

## 15. Ledger

Every landing appends a self-describing row to
`autoresearch/ledger/pr_gate_merges.jsonl` (tracked; same shape as the swarm
ledgers): `pr`, `author`, `archs`, `models`, `base_sha`, `staging_sha`,
`master_sha`, `stacked_prs:[…]`, per-arch `{tok_delta_pct, dur_delta_pct, tok_p,
dur_p}`, `parity`, `coherence`, `verdict`, `helpful`, `measurement_hash`,
`landed_via` (`kaden-auto` | `maintainer-merge` | `gate-invoke`). This is the
durable record of what landed, on what evidence.

## 16. Runner bring-up contract (sub-project 0)

Register self-hosted GitHub Actions runners; this spec names what each must
provide (bring-up is Phase 0):

| Runner | Labels | Archs | Executors | Tools required |
|---|---|---|---|---|
| **hipx** | `self-hosted, gfx1100, gfx1151` | gfx1100, gfx1151 | codex | rocprofv3, passwordless `sudo -n` (clock pin), `MODELS_DIR`, cargo, bun, `gpu-lock.sh` |
| **hiptrx** | `self-hosted, gfx1201` | gfx1201 | codex + grok | same + `GROK_BIN` |

Per-box `gpu-lock.sh` serializes the gate against the loop. Secrets:
`CLAUDE_CODE_OAUTH_TOKEN` (Claude App, for Tier 2 dispatch/interpret/merge);
codex/grok auth on-box.

## 17. Error handling & edge cases

- **GPU noise / clock skew** → `INCONCLUSIVE`/`VOID` → **neutral** status + one
  auto-retry, then human — never a hard fail on noise.
- **GPU lock contention (the whole resolution).** The gate never hangs, never
  false-fails, and never starves:
  - *Holder crashed / killed / OOM'd* → `flock(1)` on an open fd means the kernel
    **auto-releases** the lock on holder death — no stale lock is possible — so
    the gate acquires on its next 5s poll. Nothing to do (never `rm` the
    lockfile).
  - *Holder is a peer gate job on the same box* (hipx: gfx1100 + gfx1151) → they
    serialize on the per-box lock **by design**, so neither perturbs the other's
    perf measurement; the reentrancy guard (`HIPFIRE_GPU_LOCK_OWNER`) keeps a
    gate's own nested sub-measurements from self-deadlocking.
  - *Holder is the healthy long-running autoresearch loop* → the gate drops a
    **`gate-priority` marker** before waiting; the loop honors it **between
    rounds** and pauses re-acquiring until the gate clears it, so the gate is
    served within one loop-round instead of after the loop's 12h TTL. This
    priority hook is a **Phase-0 change to the loop** (`driver.py`/`watcher.py`
    check the marker at each round boundary).
  - *Backstop* → the gate's `GPU_LOCK_TIMEOUT` is set well under the GH job
    timeout; on timeout the check is **neutral (retry-later)** with a "GPU busy —
    will re-run" note and the aggregator re-fires it. A busy GPU can never red a
    PR, and no job hangs to the GH ceiling.
- **Build fail at base or head** on an arch → `BUILD_FAIL` (real fail — the PR
  broke that arch's daemon build; Tier 1 only covers the generic workspace).
- **Agent round failure** (dispatch/interpret/merge-fix) → fall back to the
  canonical battery / post a manual-review request; degrade coverage, never
  block or false-pass.
- **staging rebuild conflict** (an approved PR now clobbers after master moved) →
  that PR drops from the stack + gets a re-run request; staging self-heals.

## 18. Testing strategy

- **`ar gate` core:** no-GPU unit tests (mock `ServeRunner`, injected
  `preprocess`, captured-diff fixtures) asserting each gate's verdict; runs under
  `no-gpu-ci.sh`.
- **Perf governance:** unit-test the reject/pass/floor/drift logic against
  synthetic sample vectors (reuse `perf.mwu`).
- **Staging train:** unit-test the derived-branch rebuild + close-behind
  bookkeeping with a mock git (`gitpilot` seam).
- **Workflow:** first PRs adding `gpu-gates.yml` are docs/CI-only → Claude's plan
  says "no GPU needed" → the gate gates itself trivially. Then a `/gate` dry-run
  on a trivial in-repo PR.
- **First real use:** `/gate` on `kernel-oracle → master` → the gate's debut and
  the merge's safety net.

## 19. Rollout

1. **Phase 0** — runners on hipx/hiptrx (§16) + the loop `gate-priority` hook
   (`driver.py`/`watcher.py` honor the marker between rounds, §17) so the gate can
   preempt a running loop.
2. **Phase 1** — `ar gate` engine (gates 0–3b) + no-GPU unit tests.
3. **Phase 2** — Gate 4 non-clobber + codex merge-fix + BOD.
4. **Phase 3** — `gpu-gates.yml` + re-enable/elevate `claude-review.yml`
   (dispatch/interpret) + triggers. **Kaden auto-merge live**; maintainer
   gated-merge (`@claude /merge`) live.
5. **Phase 4** — perf governance (high-water B + drift guard + ledger).
6. **Phase 5** — staging merge-train + freshness sync (the most complex phase;
   may split into its own plan).
7. **Phase 6 (later)** — pi.dev as a third executor (one `agent_exec` branch).

## 20. Security considerations

- Auto-merge is the one genuinely sensitive capability — an agent with merge
  authority. It is gated by (author ∈ maintainers) ∧ (all deterministic gates
  green) ∧ (Claude judges helpful), sits behind a kill-switch, and never runs on
  untrusted fork code (secrets withheld on fork PRs).
- codex merge-fix pushes only to **in-repo** PR branches; fork clobbers → BOD.
- The agent holds write scope; maintainers do not — every merge is agent-executed
  and ledger-recorded, so there is a durable audit trail of what landed and why.

## 21. Open questions

- Exact GitHub API path for "close PR as landed-via-stack" that maximizes the
  "merged" badge (ancestry detection vs manual close) — resolve during Phase 5.
- Whether the drift-guard investigation should ever auto-revert vs
  always-recommend (spec says recommend-only for now).
- pi.dev executor auth/flags (Phase 6).
