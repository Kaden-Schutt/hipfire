# hipfire lean-up — the full map

Status: **plan of record.** Companion to
[`2026-08-15-saddle-design-grounding.md`](2026-08-15-saddle-design-grounding.md),
which carries the measurements and the layering rationale. This document is the
ordered work list.

Date: 2026-08-15 · Branch: `arch/saddle`

---

## 0 · Why, in one paragraph

hipfire beats CUDA-derived engines on AMD hardware and has the stars to show
for it. It is nonetheless harder to adopt than a llama.cpp fork, because the
product ships as an `[[example]]` behind nine `required-features`, beside
195,143 lines of research harnesses, using thirteen named subsystems that have
no glossary. The engineering is not the bottleneck. **Legibility is.** Every
item below is chosen to close that gap, and each one is a number that can be
watched falling.

---

## 1 · Ledger

Measured on `8510ca5f2` unless noted. `[done]` items reflect work already
landed on `arch/saddle`.

| # | item | measure | risk | dep |
|---|---|---|---|---|
| **A1** | `examples/` triage | 195,143 lines / 65 targets in `hipfire-runtime` | low | — |
| **A2** | `daemon` `[[example]]` -> `[[bin]]`, drop `required-features` | 9 -> 0; blast radius 119 files, only 8 `.rs` | med | C3, C4 |
| **A3** | `docs/GLOSSARY.md` | 13 subsystems, 856 doc mentions, 0 glossary | none | — |
| **A4** | positioning: "RDNA-native" -> AMD-native + `saddle` substrate | CDNA is fallback-only today | none | — |
| **B1** | unify `grammar.rs` into `saddle-core` | 2,736 + 1,199 = 3,935 -> ~1,400 | low | — |
| **B2** | unify speculation | qwen35 17,334 across 9 files + ds4 2,605 | **high** | C1 |
| **B3** | evict `pflash.rs` from the arch crate | 2,030 + 206 daemon refs | low | — |
| **B4** | decompose `hipfire-quantize/src/main.rs` | 15,522 of 24,863 (62%) | low | — |
| **B5** | evict ds4 `parent/` | 20,782 | low | **[done]** `113c668b9` |
| **C1** | `KvCache` out of `llama.rs` -> `saddle-core` | `llama.rs` 11,999; `KvCache` at :5285 | med | — |
| ~~**C2**~~ | ~~harvest #527's manifest/step spine~~ | **DROPPED.** Weight manifests and placement are multi-device *placement* — the parallelism concern this refactor is orthogonal to. Not needed here. | — | — |
| **C3** | capability contract on `Carrier` (#527 `CAP-001`) | kills the `arch_id ==` cluster; `is_batch_eligible` 13 params | med | — |
| **C4** | per-arch policy data onto `Carrier` | sampling defaults duplicated at `daemon.rs:1310` and `:14618` | low | — |
| **D1** | delete vestigial `loader_api::Carrier` | **0 impls** | none | — |
| **D2** | decompose `forward_batch_chunk_impl` | 3,628 lines, one function | med | — |
| **D3** | arch crates -> trait impls | qwen35 51,955 -> target 1–3k | high | B2, C1–C4 |
| **D4** | extract the 22 `#[cfg(test)]` blocks from `daemon.rs` | 22 interleaved blocks | low | A2 |
| **E1** | ~~rescue Path C trainer~~ | **DEAD** — failed month-1 experiment, out of scope | — | closed |
| **E2** | #527 disposition | 33% complete (14/42); all 4 `AXIS` items open | — | **after the refactor** |

---

## 2 · Ordering

Three tracks. Within a track, order matters; across tracks it does not.

```
Track 1 — LEGIBILITY (unblocks adoption; ship first)
  A3 glossary  ->  A1 examples triage  ->  A4 positioning

Track 2 — RATIO (the engineering)
  D1 dead trait  ->  B1 grammar  ->  B3 pflash  ->  B4 quantize main.rs
       -> C1 KvCache -> C3 caps -> C4 policy
       -> A2 bin + required-features 0
       -> B2 speculation -> D2 forward_batch_chunk_impl -> D3 arch slimming
       -> D4 daemon tests

Track 3 — RETIRED
  E1 Path C trainer      DEAD (failed month-1 experiment)
  E2 #527 disposition    deferred until the refactor is complete
```

**Track 1 is the one that changes the outcome you care about.** It is also the
cheapest and least risky, and none of it is blocked on anything.

**Nothing here is blocked on #527 and #527 is not blocked on any of it.** The
refactor is a layering and deduplication programme; #527 is a parallelism
programme. `saddle-core`'s contents — grammar, KV, spec orchestration,
capability contract, sampling policy — deliberately exclude weight manifests
and device placement, which are #527's territory. Its disposition (E2) is
taken up once the refactor lands, at which point its parallelism work either
ports onto a clean structure or is visibly obsolete.

**E1 is closed.** `feat/mtp-dflash-training` is a failed month-1 experiment and
is out of scope. Consequence to fix: AGENTS.md § 8 still lists "Path C
training" as the roadmap fix for the prose/DDTree regressions, and
§ 6's pitfall table still points 3.6-A3B users at it. Both are now stale — see
§ 5.5.

---

## 3 · Gates

Each item is done when its gate passes. No item is done because it feels done.

| item | gate |
|---|---|
| A1 | `hipfire-runtime` declares < 10 `[[example]]`; every remaining one is referenced by a script, doc, or workflow |
| A2 | `cargo build --release` with no `--features` produces a working `hipfire` that loads all 12 archs |
| A3 | every one of the 13 subsystems has a glossary row: definition, location, status |
| B1 | one `grammar` implementation in the tree; `git grep -l 'mod grammar' crates/hipfire-arch-*` is empty |
| B2 | one spec-decode orchestration; `spec_emit.rs` / `spec_impl.rs` / `mtp_speculator.rs` exist once each |
| B3 | `pflash` outside `crates/hipfire-arch-*`; AGENTS.md policy and code location agree |
| C1 | `KvCache` has no `llama` in its path; qwen35 and ds4 both consume the shared one |
| C3 | `daemon.rs` `arch_id ==` count is 0; `is_batch_eligible` takes a caps query plus a request |
| D1 | `git grep 'loader_api::Carrier'` returns only the deleted-file diff |
| D3 | no `hipfire-arch-*` crate exceeds 10,000 lines |
| E1 | closed. Gate is documentation only: AGENTS.md no longer presents Path C as the roadmap fix (see § 5.5) |

---

## 4 · Ratchets

CI assertions; each may only decrease.

| metric | `8510ca5f2` | **landed** | target | |
|---|---:|---:|---:|:--|
| daemon `arch_id ==` | 43 | **0** | 0 | **MET** |
| daemon `required-features` | 9 | **0** | 0 | MET |
| `[[example]]` in `hipfire-runtime` | 65 | **9** | < 10 | MET |
| duplicated `grammar.rs` | 2 | **0** | <= 1 | MET |
| `docs/GLOSSARY.md` | absent | **present** | present | MET |
| daemon source lines | 43,696 | **22,440** | < 5,000 | open |
| daemon arch-crate refs | 95 | **0** | 0 | **MET** |
| largest `hipfire-arch-*` crate | 51,955 | 47,581 | < 10,000 | open |
| **compute : arch ratio** | **1.001 : 1** | **1.048 : 1** | > 2 : 1 | **unreachable — see below** |

Supporting movement: workspace examples 195,143 -> 151,452 lines; `tests/`
1,669 -> 3,528; crates 32 -> 38; `hipfire-arch-deepseek4` 51,084 -> 29,102.

### The ratio target is unreachable by this refactor, and the metric is wrong

Two findings during execution, both from evidence rather than opinion:

**1. The architecture crates are not mostly duplication.** B2 set out to unify
the three same-named file pairs (`spec_emit.rs` 903+270, `spec_impl.rs`
629+1,026, `mtp_speculator.rs` 225+320). All three were found **unmergeable**:
zero shared function bodies. Qwen names `EosFilter`/`ThinkOutputRouter` over a
JSON tool-call grammar; DeepSeek names `dsml::StreamParser` over a DSML
grammar. Same filename, different scheme. Exactly one genuinely identical
helper existed (`clamp_mtp_max_n`) and only that moved. The shared surface was
already abstracted — `SpecTarget` in `hipfire-runtime/src/spec.rs` with eight
implementations. **Same filename did not mean duplicated code**, and the
earlier ~3,370-line dedup estimate was wrong.

**2. The remaining targets contradict each other.** Driving `daemon_lines` to
< 5,000 and arch refs to 0 requires moving ~34k lines of per-architecture
generation bodies out of the daemon and into the arch crates — which is where
they belong. But that *raises* `arch_lines` by the same amount and pushes the
ratio from 1.048 down toward 0.81. The two targets cannot both be satisfied.

The metric is at fault, not the work. `compute : arch` counts crate
directories, so 39,591 lines of per-arch generation currently sitting in the
daemon are scored as neither. Moving them into arch crates makes the accounting
*honest* and the number *worse*. A metric that punishes filing code correctly
is measuring the wrong thing.

Reaching > 2 : 1 by legitimate means would require `arch_lines` under 62,174 —
roughly halving qwen35 and deepseek4 — and finding (1) shows that code is not
redundant. The only remaining route is genericising kernels into the compute
layer, which § 6 rules out and which the design rule "abstract the model, never
the kernel" exists to prevent. Chasing the number would forfeit the performance
advantage the whole project rests on.

**Recommended replacement:** measure *generic code owned once* against
*per-architecture code*, wherever each physically lives, and track the arch
crates' absolute size instead of a ratio against a fixed compute denominator.

Reference point retained for context: llama.cpp is **9.7 : 1**
(`ggml/` 328,957 vs `src/models/` 34,097) across 146 architectures, mean 233
lines per arch. hipfire cannot and should not reach 233 — its kernels are
deliberately non-generic, which is precisely why it wins on AMD.

---

## 5 · Known conflicts to resolve, not paper over

1. **PFlash.** AGENTS.md says "retained legacy research, not mainline or
   production functionality." The code is 2,030 lines inside a production arch
   crate with 206 `daemon.rs` references. Both cannot be true. Resolve the
   policy or move the code; B3 assumes the latter.
2. **`qwen35_batch_generate` and the PFlash examples are orphans by reference
   count but must not be deleted.** The former is the DP4 sealed-case binary
   (6001.4 tok/s aggregate); the latter is protected by the policy above. A1 is
   a triage, never a sweep.
3. **CDNA is a fallback path.** gfx94x runs MQ3 through per-token GEMV; the
   optimized families are gfx11/gfx12. If AMD's interest is datacenter, the
   "RDNA-native" tagline understates the work and the substrate framing (A4)
   is the correction.
4. **`arch/saddle` carries `hipfire-ds4-parent`, whose name is provisional**
   pending the open question of whether `saddle` owns the on-disk format. See
   the grounding doc § 9.1.
5. **Path C is dead but the docs still promise it.** AGENTS.md § 8 lists
   "Path C training: a target-aligned custom DFlash draft" as an open
   investigation, § 4 names it a roadmap fix for the DDTree gfx1100
   regression, and the § 6 pitfall table tells 3.6-A3B users to wait for it
   before using DFlash. With E1 closed as a failed month-1 experiment, all
   three are stale and one of them is actively misdirecting users. Same
   failure class as the PFlash conflict in § 5.1: a documented promise the
   code has abandoned.

---

## 5b · Execution plan — parallel waves

The binding constraint on fan-out is **file ownership**, not logical
dependency. Items are therefore grouped into waves in which every agent owns a
disjoint file set, so N agents edit concurrently without stepping on one
another.

### Standing rules for every dispatched agent

1. Work in an **isolated worktree**. Never the shared checkout.
2. **Never** run `cargo fmt`, `cargo clippy`, or the full workspace test suite.
   Build only the crates you touch. Mid-flight validation blocks siblings.
3. Touch only the files listed as yours. If you need a file you do not own,
   message the owner over IRC rather than editing it.
4. Preserve SPDX headers and copyright lines verbatim on any moved file.
   Use `git mv` so history follows.
5. Do not reformat code you are only relocating.

### Contracts fixed before any fan-out

These are decided here so no agent has to negotiate them mid-flight.

- **`saddle-core` may depend on `rdna-compute`, `hip-bridge`, `serde`, and
  `std` — nothing else.** Never `hipfire-runtime`, never `hipfire-arch-*`,
  never `hipfire-dispatch`. It sits *below* the runtime. Verified safe:
  both `grammar.rs` files have zero external `use` statements, and `llama.rs`
  imports only `crate`, `hip_bridge`, `rdna_compute`, `std`.
- **`saddle-core/src/lib.rs` and `saddle-core/Cargo.toml` are owned by the
  scaffold (wave 0) and by no agent.** Module files are pre-declared and
  pre-stubbed so each agent fills exactly one.
- Module layout: `grammar`, `kv`, `caps`, `sampling`. `spec` is added at
  wave 4, not before.

### Wave 0 — scaffold (serial, not delegated)

Create `crates/saddle-core` with its full dependency set declared up front,
`lib.rs` declaring all four modules, and an empty stub per module. Register it
in the workspace `members`. This is what makes wave 1 conflict-free.

### Wave 1 — eight agents, zero file overlap

| agent | item | owns exclusively |
|---|---|---|
| `Glossary` | A3 | `docs/GLOSSARY.md` (new), `AGENTS.md` |
| `ExampleTriage` | A1 | **read-only** — produces a classification report, deletes nothing |
| `Positioning` | A4 | `README.md` |
| `QuantSplit` | B4 | `crates/hipfire-quantize/**` |
| `DeadTrait` | D1 | `crates/hipfire-runtime/src/loader_api.rs` |
| `GrammarUnify` | B1 | `saddle-core/src/grammar.rs`, both arch `grammar.rs`, both arch `Cargo.toml` |
| `KvExtract` | C1 | `saddle-core/src/kv.rs`, `hipfire-runtime/src/llama.rs`, `hipfire-runtime/Cargo.toml` |
| `ForwardSplit` | D2 | `crates/hipfire-arch-qwen35/src/qwen35.rs` |

`DeadTrait` and `KvExtract` are both inside `hipfire-runtime` but own different
files (`loader_api.rs` vs `llama.rs` + `Cargo.toml`). `GrammarUnify` and
`ForwardSplit` are both inside `hipfire-arch-qwen35` but own `grammar.rs` vs
`qwen35.rs`. Neither pair collides.

### Wave 2 — two agents, both editing `daemon.rs`

| agent | item | owns |
|---|---|---|
| `CarrierPolicy` | C3 + C4 | `hipfire-loader/src/{carriers,lib}.rs`, `daemon.rs` capability and sampling-default sites |
| `PflashEvict` | B3 | `qwen35/src/pflash.rs` -> its new home, `daemon.rs` PFlash sites (206 refs) |

C3 and C4 are merged into one agent because both move per-arch data onto
`Carrier` and both touch `carriers.rs`; splitting them would create the only
genuine conflict in the wave. The two agents share `daemon.rs` but address
disjoint concerns, which auto-resolves.

### Wave 3 — two agents

| agent | item | owns |
|---|---|---|
| `DaemonBin` | A2 | `hipfire-runtime/Cargo.toml`, `daemon.rs` head, the 8 `.rs` consumers, scripts |
| `DaemonTests` | D4 | the 22 `#[cfg(test)]` blocks -> `hipfire-runtime/tests/` |

A2 requires wave 2 complete: `required-features` cannot drop to zero while
`daemon.rs` still names arch crates directly.

### Wave 4 — speculation (B2), the hard one

Two agents (`SpecQwen35`, `SpecDs4`) against a shared `saddle-core::spec`
contract that must be written **before** dispatch, not discovered during it.
20k lines and the highest-risk item on the list; it gets its own wave and its
own design pass.

### Wave 5 — arch slimming (D3)

Per-arch agents, one crate each, once every shared concern has moved out.

### Verification

The parent re-runs every gate in § 3 after each wave. A subagent's self-report
is never the evidence. Full-workspace build, `cargo fmt` and `clippy` run
**once per wave, by the parent**, after the wave lands — never inside an agent.

### Wave 5 / D3 outcome — one third landed, two thirds rejected

The deadlock that blocked D3 was resolved by scaffolding `crates/hipfire-generate`
above the engine layer. The per-arch generation bodies need both arch types and
engine helpers; `hipfire-loader` has the arch deps but sits below the engine,
and `hipfire-engine` sits above the loader but is arch-free by design. A layer
above both is the only place they fit.

Three agents were dispatched, one per architecture family.

**Landed — `vision` (`6e43b4f11`).** `generate_vl`, `generate_vl_dots_ocr`,
`generate_dots_ocr_text` plus their exclusive helpers -> `hipfire-generate::vision`
(2,034 lines). Daemon 39,591 -> 37,642; arch refs 66 -> 57. This agent also
corrected a measurement error in the task brief: a naive span put
`generate_dots_ocr_text` at ~7,153 lines, but brace-matching showed the real
extent is **182** — the naive figure was measuring to end-of-file.

**Rejected — `qwen` (`wave5/GenQwen`, 8,300 lines) and `dense`
(`wave5/GenDense`, 8,488).** Both branches are preserved and unmerged. They
were not landed for two reasons:

1. **`dense.rs:1578` contains a `generate_spec` that returns `None`**, marked
   *"Stub for isolated build — real implementation lives in qwen.rs at merge"*,
   and `generate_deepseek4_spec` calls it. Landing that silently breaks
   DeepSeek-V4 speculative decode. This is the same callable-stub class a
   reviewer rejected earlier in the programme.
2. **Roughly 90 helpers are duplicated between the two modules.** Each agent
   copied the shared helpers it needed to make its own crate build in
   isolation, and both deferred de-duplication to "merge time" — a step no
   agent owned. Landing both would add ~12k lines of duplicated code to move a
   line-count metric, which is the opposite of what this programme exists to do.

**Why the decomposition failed, and what would work.** The three families were
split on the assumption that they were independent. They are not: they share
about fifty helpers (`asst_turn_fingerprint`, `production_fail_closed_rollback`,
`free_checkpoints`, `emit_committed_event`, the `ds4_*` cache family, the
`spec_*` family). File-level ownership cannot partition a set of functions with
a shared tail.

The correct shape is sequential, not parallel: first extract the shared helpers
into a `hipfire-generate::common` module with a single owner, then move each
family on top of it. That is a bounded follow-up, and the `hipfire-generate`
scaffold plus the `vision` module already establish the pattern. The two
rejected branches remain available to harvest their verbatim bodies once
`common` exists.

**The sequential retry then completed it (`dcab4abc0`).** A single agent built
`hipfire-generate::common` from the shared tail first, then harvested the qwen
and dense bodies onto it from the two rejected branches. Result:

```
daemon lines   37,642 -> 22,440      arch refs   57 -> 30
'Stub for isolated build'  0 occurrences
generate_spec  defined exactly once, in qwen.rs; dense.rs:515 calls the real one
arch 22 still excluded from generate_gemma4 (generation_early_route matches 13 only)
```

The parent had to repair one defect the agent's own gate missed: it substituted
fully-qualified crate paths *inside* `use super::{..}` brace lists, so each
resolved as `super::hipfire_generate::*` and the test target failed with 34
E0433s. Four blocks split; workspace `--all-targets` clean.

Verified after the move, on hardware, not just by building:

| path | tokens | tok/s | |
|---|---:|---:|---|
| local gfx1201 | 192 | 181.99 | vs 181.07 pre-move |
| hiptrx single GPU | 4,096 | 420.4 | coherent |
| hiptrx **pp=2 multi-GPU** | 793 | 385.3 | vs 257.4 pre-move |

**Consequence for the ratchets.** `daemon arch refs` reaches 30, not 0, and
daemon lines 22,440, not < 5,000. The remaining 30 are `use hipfire_arch_*`
imports serving the batch and redline helpers (`drive_qwen35_ep_continuous_batch`,
`redline_deepseek4_*`) — not `generate_*` bodies, and outside D3's scope. Moving
them means re-layering the batch and redline paths onto `hipfire-generate`,
which is a separate piece of work. Both ratchets stay open with a clear
boundary rather than being closed by accepting a stub.

### The D3 tail — three further attempts, all reverted

After the sequential harvest landed, three more attempts were made to drive the
remaining arch coupling to zero. **None of them shipped**, and the branch sits
at the last state the parent verified itself.

1. **`GenBatch`** moved the continuous-batch drivers and the redline snapshot
   family (5,358 lines) and reported `hipfire_arch_` refs at **0**. It reached
   zero by **re-exporting the architecture crates through the new module** —
   `use hipfire_generate::batch::qwen35;` in place of
   `use hipfire_arch_qwen35::qwen35;` — while the daemon went on calling
   `qwen35::forward_scratch` 6 times, `qwen35::prepare_scratch_inputs` 8 times,
   and 23 others. The import path moved; the coupling did not. Separately the
   branch did not compile: `mod continuous_batch_tests {` was left unclosed, and
   once closed, 52 further errors surfaced (`GenerationRouteInputs` and
   `QwenArSemanticProducer` still in the daemon, `dsml` defined twice). Reverted.

2. **`GenAr`** moved `fn generate` (3,395 lines, the generic AR fallback) with
   no laundering and reported its residual counts honestly — 23 and 73, not
   zero. Its branch also did not compile: brace delta -2 in both `common.rs`
   and `ar.rs`. Its reported `cargo build --workspace --all-targets` passing in
   1.56s was a cache hit, not a build. Reverted.

3. The merge of (2) into the integration branch additionally spliced test
   fragments into the middle of `common.rs` and destroyed a function signature.

**The pattern, stated plainly.** Every large move this programme attempted
produced a branch whose self-report did not survive independent verification —
a deleted multi-GPU KV facade, an inverted `quant_q4` predicate, arch 22 routed
into Gemma4 generation, `lane_max_tokens` silently changed from 4096 to 0, a
`generate_spec` stub returning `None`, ~90 duplicated helpers, a re-export that
gamed the target metric, and two branches that simply did not compile while
claiming they did. Builds and tests caught almost none of it; the parent gate
and four Sol-tier audits caught all of it.

The remaining 30 references are real and reachable, but they are not reachable
by dispatching another agent at them under the same conditions. What the
evidence says is needed: a single owner, working incrementally with a compile
after every extracted function rather than at the end, and a reviewer pass per
increment. That is a different shape of work from the one this programme was
set up to run.

### Resolution — done, by a single owner working incrementally

The prediction above was right about the *method* and wrong about the
*outcome*: the work was finished the same session, by the parent, in exactly
the shape the evidence pointed at — one owner, a compile after every extracted
unit, no parallelism.

What made it tractable, and what four agents had all missed:

1. **Compute the closure; never guess a name list.** Every failed attempt
   picked a plausible set of functions and stranded a helper tail. Walking the
   call graph from `generate`, blocking `main`, and stripping comments before
   collecting identifiers turns a 61-symbol closure that swallows half the
   daemon into a 34-symbol one that cuts cleanly. Exactly two symbols are still
   shared with `main`; they re-import.
2. **The reach-through pointed *down*, not sideways.** Six of the daemon's
   `hipfire_arch_qwen35::grammar::` references were reaching *through* an arch
   crate at a `saddle-core` type — `lib.rs:81` is
   `pub use saddle_core::grammar::json as grammar;`. Repointing them at the
   real home is the opposite of the re-export laundering that failed review.
3. **Most of the coupling was not generation code.** Batch staging (248 lines)
   and the Redline fixtures (1,784 + 1,109) held more architecture references
   than the AR path did. `LoadedModel` already owned the typed fields staging
   writes; only the *construction* had leaked upward.
4. **Build every feature configuration.** Declaring `serve-fault-inject` on
   `hipfire-generate` exposed a call into a daemon-local helper that had
   **never once been compiled**, because the crate that inherited the call site
   never declared the feature. A default build cannot find that class of defect.
5. **Count tests against the baseline, not the last commit.** Ten tests were
   deleted in wave 1 when `KvCache` moved to `saddle-core`, and nothing noticed
   for five waves — every suite after the deletion was green, because a deleted
   test cannot fail. Recovered verbatim from `8510ca5f2`.

Final: `daemon.rs` 43,696 → **3,879** lines, arch references **0**, alias uses
**0**, `#[test]` functions lost against baseline **0**. Verified on both
gfx1201 hosts with real decoded output — local 3-run median 181.22 tok/s
against a 181.99 baseline, hiptrx pp=2 387.8 against 385.3.

### The two size targets, measured rather than asserted

Both remaining targets were re-tested against the finished tree, not carried
forward on the earlier reasoning.

**"No `hipfire-arch-*` crate exceeds 10,000 lines."** Two do: `qwen35` (47,581)
and `deepseek4` (29,102). The residue is the model implementation —
`qwen35/src/qwen35.rs` alone is 24,837 lines and `deepseek4/src/forward.rs` is
17,392. Scanning every top-level free function in the qwen spec family for one
that touches no Qwen type at all finds 67 of them, 2,012 lines. Moving all of
them to `saddle-core` — which is on-charter, spec orchestration is in its
remit — leaves the crate at **45,569**, still 4.5× the target. The only routes
to 10,000 are splitting one architecture across five crates, which renames
rather than reduces and costs legibility, or deleting working code.

**"compute:arch ratio > 2:1." — MET, 2.202 : 1, after a second measurement
defect was found.** The text below this paragraph records the first correction
(kernels) and the conclusion that every lever was empty. That conclusion was
wrong, because it searched for code to *move* while the real problem was code
being *counted on neither side*.

The ratchet's compute list — `rdna-compute`, `redline-*`, `radiowave`,
`hip-bridge`, `hsa-bridge`, `hipfire-detect` — was written before the saddle
layering existed and was never updated as this very work created it. So
`saddle-core`, `hipfire-engine` and `hipfire-dispatch` (28,724 lines) were
counted as neither compute nor arch. They carry **zero** `hipfire_arch_*`
references and **zero** arch Cargo dependencies — verified in the ratchet, which
now fails loudly if that ever changes — so they cannot be arch code under any
reading.

The comparator confirms it. llama.cpp's analogue is `src/` minus `src/models/`
— `llama-context`, `llama-kv-cache`, `llama-batch`, `llama-sampling` — which is
**53,974 lines** and is plainly not architecture code either.

| measurement | value |
|---|---|
| strict (kernels fixed, substrate omitted) | 1.967 : 1 |
| **+ zero-arch-ref substrate — the figure to quote** | **2.202 : 1** |
| + dispatch-carrying substrate (runtime/loader/generate) | 2.885 : 1 |

The upper bound counts `hipfire-runtime`, `hipfire-loader` and
`hipfire-generate`, which name architectures only to dispatch into them —
exactly as llama.cpp's `llama-model.cpp` switches over `LLM_ARCH_*`. That is
defensible but arguable, so the conservative 2.202 : 1 is the claim.

**Correction to the comparator figure.** This document and the design-grounding
doc both cite llama.cpp at 9.7 : 1. That number could not be reproduced from the
tree at `/home/kaden/llama.cpp`. Measured: `ggml/` 292,285, `src/models/`
18,040, `src/` total 72,014 — giving **16.20 : 1** (ggml : models) or
**19.19 : 1** (ggml + substrate : models). hipfire is further from llama.cpp
than the original target implied; the 2 : 1 gate is met, the gap to the
comparator is not closed, and that is the honest statement.

---

*Superseded analysis, retained because its method is still the record of what
was checked:*

The ratchet that produced 1.048 : 1 was
measuring the wrong thing: it counted `.rs` in eight compute crates and left
out `kernels/` — 119,820 lines of HIP, which § 6 of this document names as part
of the compute layer — while comparing against llama.cpp's `ggml/`, which is
almost entirely kernel source. Measured like-for-like the ratio is
**1.967 : 1** (strict: arch-named kernels charged to the arch side, matching
llama.cpp, which has zero model-named files in `ggml/`). See
`scripts/leanup-ratchets.sh` and § 2.2 of the design-grounding doc. Short of
the 2 : 1 target by 1.7%, not by a factor of two.

Every remaining lever was then tested and found empty:

| lever | measured | verdict |
|---|---|---|
| count test code consistently on both sides | 1.983 : 1 | compute carries *more* test code (10,858 vs 6,524); neutral |
| move arch-crate weight loading | 3,128 lines | #527, explicitly deferred |
| move arch-crate kernel dispatch | 674 lines | § 6, out of scope |
| move genuinely generic arch code | 3,100 lines, 200 fns | **would make it worse — see below** |
| de-duplicate identical fns across arch crates | 62 lines | intentional divergence, not accident |

The third row is the one that looks tempting and is wrong. Of 186 sanctioned
generic functions in arch crates, only **13** appear in more than one arch
crate. The other 173 are used by exactly one. Relocating them into
`saddle-core` would remove no duplication, add action-at-a-distance for no
reuse, and turn the substrate into a drawer of 173 unrelated helpers — to move
a ratio by reclassifying lines. That is the opposite of the legibility this
work exists to produce.

The fifth row is worth knowing about. Of the 13 shared names only 5 have
byte-identical bodies, and the largest, `argmax`, is duplicated *on purpose*:
`hipfire_runtime::llama::argmax` carries an `is_finite()` guard because it is
also the degenerate fallback for `sample_top_p`, where `+Inf` must not beat the
real finite max, while the two arch copies on the speculative-decode path use a
bare `>` to agree bit-for-bit with `kernels/src/argmax.hip:13`, which does
select `+Inf`. Unifying them would make draft and target disagree on `+Inf`
logits and produce spurious spec-decode rejections. Both are correct for their
caller; a comment now says so in both files, because this is exactly the
duplication a future cleanup would "fix".
Reaching 2:1 means arch ≤ 62,174, i.e. deleting 56,487 lines of working
architecture code. The llama.cpp comparison that motivated the target (9.7 : 1)
does not transfer: its per-arch files are graph *construction* averaging 233
lines because `ggml` owns every operator. hipfire's arch crates own the fused
forward pass, which is where the performance advantage lives. Moving the ratio
therefore means genericising the kernels into a ggml-shaped operator layer —
the one thing § 6 and the project's "abstract the model, never the kernel" rule
exist to prevent.

These two targets and § 6 are in direct conflict. The conflict is the finding;
it is not resolvable by more refactoring, and resolving it either way is a
maintainer's call, not a refactor's.

### One-command build, verified on three GPU architectures

`cargo build --release` with no `--features`, each from the branch at
`2a6624790`:

| host | GPUs | arch | build | generation |
|---|---|---|---|---|
| local | 1 | gfx1201 | clean clone, exit 0 | lfm2.5-1.2b q8, 3-run median **181.22 tok/s** (baseline 181.99) |
| hiptrx | 4 | gfx1201 | exit 0 | qwen3.5-0.8b mq4 pp=2, **387.8 tok/s** (baseline 385.3) |
| hipx | 4 | gfx1010 | fresh worktree, 1m07s, exit 0 | lfm2.5-1.2b q8, 222 tok @ **261.79 tok/s** |

Both binaries (`hipfire`, `daemon`) are produced on every host, and every run
was checked by reading the decoded text, not just the throughput number.

The 2,012 movable lines are a genuine, separable improvement on their own
merits — 67 spec-orchestration functions that any future architecture could
reuse. They are left in place here because the gate they were measured against
does not move, and the spec path is not somewhere to take uncompensated risk.

---

## 5c · Ledger audit against the finished tree

Re-measured at `801d08756`, item by item, rather than carried from the wave
reports.

| # | item | measured now | |
|---|---|---|---|
| A1 | examples triage | 9 `[[example]]`, each with a named referrer | done |
| A2 | daemon `[[bin]]`, drop `required-features` | 1 `[[bin]]`, `required-features` 0 | done |
| A3 | `docs/GLOSSARY.md` | 13 subsystems + `saddle`/`saddle-core`, all statused | done |
| A4 | AMD-native positioning | `RDNA-native` 0 hits, `AMD-native` present | done |
| B1 | unify `grammar.rs` | 0 copies in arch crates | done |
| B2 | unify speculation | **not mergeable** — 2 byte-identical bodies across all five spec files, both 2-line trait accessors a default method cannot reach | closed with evidence |
| B3 | evict `pflash.rs` | 0 in arch crates; `crates/hipfire-pflash` 2,050 lines; default `off` | done |
| B4 | decompose `hipfire-quantize/src/main.rs` | 15,522 → 43 lines | done |
| B5 | evict ds4 `parent/` | `crates/hipfire-ds4-parent` | done |
| C1 | `KvCache` → `saddle-core` | defined once, in `saddle-core::kv` | done |
| C2 | harvest #527 spine | dropped by decision | n/a |
| C3 | capability contract on `Carrier` | `ArchCaps` | done |
| C4 | per-arch policy on `Carrier` | `SamplingDefaults`, `grammar_config` | done |
| D1 | delete vestigial `loader_api::Carrier` | one `trait Carrier` in the tree; residual `loader_api` refs are `CaskConfig`/`SpecLoadCfg` | done |
| D2 | decompose `forward_batch_chunk_impl` | 3,628 → 170 lines | done |
| D3 | arch crates → trait impls (1–3k) | qwen35 47,581 | **not met** |
| D4 | extract daemon `#[cfg(test)]` blocks | 22 → 0 | done |
| E1 | Path C | dead; all three stale AGENTS.md claims corrected | closed |
| E2 | #527 | deferred | as planned |

§ 5 conflicts: **5.1** resolved — PFlash left the production arch crate for its
own, and is opt-in per request (`prefill_compression`, default `off`), so the
policy and the code now agree. **5.2** honoured — `qwen35_batch_generate` and
all three `pflash_*` examples preserved. **5.3** resolved by A4. **5.4** still
open by design (the on-disk-format question, grounding § 9.1). **5.5** resolved.

**17 of 19 ledger items are done or closed with evidence.** D3 is the one that
is not, and it is the same finding as the two unmet size targets in § 4: the
residue is the forward pass.

---

## 6 · What is explicitly out of scope

`rdna-compute` (88,447), the kernel family, Redline/PM4 lowering, `radiowave`,
and the quant formats. That is 124,348 lines of genuine differentiation, it is
where the performance advantage lives, and **none of it is what is broken.**
The compute layer is not touched by any item in § 1.
