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
| daemon `arch_id ==` | 43 | **0** | 0 | MET |
| daemon `required-features` | 9 | **0** | 0 | MET |
| `[[example]]` in `hipfire-runtime` | 65 | **9** | < 10 | MET |
| duplicated `grammar.rs` | 2 | **0** | <= 1 | MET |
| `docs/GLOSSARY.md` | absent | **present** | present | MET |
| daemon source lines | 43,696 | **22,440** | < 5,000 | open |
| daemon arch-crate refs | 95 | **30** | 0 | open |
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

---

## 6 · What is explicitly out of scope

`rdna-compute` (88,447), the kernel family, Redline/PM4 lowering, `radiowave`,
and the quant formats. That is 124,348 lines of genuine differentiation, it is
where the performance advantage lives, and **none of it is what is broken.**
The compute layer is not touched by any item in § 1.
