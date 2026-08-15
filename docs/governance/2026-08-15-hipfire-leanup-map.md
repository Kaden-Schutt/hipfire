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
| **B2** | unify speculation | qwen35 17,334 across 9 files + ds4 2,605 | **high** | C2 |
| **B3** | evict `pflash.rs` from the arch crate | 2,030 + 206 daemon refs | low | — |
| **B4** | decompose `hipfire-quantize/src/main.rs` | 15,522 of 24,863 (62%) | low | — |
| **B5** | evict ds4 `parent/` | 20,782 | low | **[done]** `113c668b9` |
| **C1** | `KvCache` out of `llama.rs` -> `saddle-core` | `llama.rs` 11,999; `KvCache` at :5285 | med | — |
| **C2** | harvest #527's manifest/step spine | `weight_manifest` 4,662 + `weight_store` 6,829 + `moe_plan` 11,563 | med | — |
| **C3** | capability contract on `Carrier` (#527 `CAP-001`) | kills the `arch_id ==` cluster; `is_batch_eligible` 13 params | med | — |
| **C4** | per-arch policy data onto `Carrier` | sampling defaults duplicated at `daemon.rs:1310` and `:14618` | low | — |
| **D1** | delete vestigial `loader_api::Carrier` | **0 impls** | none | — |
| **D2** | decompose `forward_batch_chunk_impl` | 3,628 lines, one function | med | — |
| **D3** | arch crates -> trait impls | qwen35 51,955 -> target 1–3k | high | B2, C1–C4 |
| **D4** | extract the 22 `#[cfg(test)]` blocks from `daemon.rs` | 22 interleaved blocks | low | A2 |
| **E1** | rescue Path C trainer | `feat/mtp-dflash-training` 21 ahead / **1,832 behind**, +6,747 | **urgent** | — |
| **E2** | #527 disposition | 33% complete (14/42); all 4 `AXIS` items open | — | — |

---

## 2 · Ordering

Three tracks. Within a track, order matters; across tracks it does not.

```
Track 1 — LEGIBILITY (unblocks adoption; ship first)
  A3 glossary  ->  A1 examples triage  ->  A4 positioning

Track 2 — RATIO (the engineering)
  D1 dead trait  ->  B1 grammar  ->  B3 pflash  ->  B4 quantize main.rs
       -> C1 KvCache -> C2 harvest 527 -> C3 caps -> C4 policy
       -> A2 bin + required-features 0
       -> B2 speculation -> D2 forward_batch_chunk_impl -> D3 arch slimming
       -> D4 daemon tests

Track 3 — RESCUE (time-critical, independent)
  E1 Path C trainer   (decaying now)
  E2 #527 harvest decision
```

**Track 1 is the one that changes the outcome you care about.** It is also the
cheapest and least risky, and none of it is blocked on anything.

**E1 is the only item with a deadline.** `feat/mtp-dflash-training` is 1,832
commits behind — **2.2x the drift that made #527 unlandable** — and holds a
KL-topk loss backward pass and target-init loader across 41 files. It is
smaller and more self-contained than #527 was. It gets rescued now or it gets
harvested later at a much worse exchange rate.

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
| E1 | branch merged or explicitly harvested with a written disposition |

---

## 4 · Ratchets

CI assertions; each may only decrease.

| metric | 8510ca5f2 | now | target |
|---|---:|---:|---:|
| `daemon.rs` lines | 43,696 | 43,696 | < 5,000 |
| `daemon.rs` `arch_id ==` | 43 | 43 | 0 |
| `daemon.rs` arch-crate refs | 95 | 95 | 0 |
| `daemon` `required-features` | 9 | 9 | 0 |
| `[[example]]` in `hipfire-runtime` | 65 | 65 | < 10 |
| largest `hipfire-arch-*` crate | 51,955 | 51,955 | < 10,000 |
| duplicated `grammar.rs` | 2 | 2 | 1 |
| **compute : arch ratio** | **0.70 : 1** | **0.85 : 1** | **> 2 : 1** |

Reference point: llama.cpp's compute:arch ratio is **9.7 : 1**
(`ggml/` 328,957 vs `src/models/` 34,097), at 146 architectures and a mean of
233 lines per arch. hipfire will not and should not reach 233 — its kernels are
deliberately non-generic — but the *ratio* is the honest target.

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

---

## 6 · What is explicitly out of scope

`rdna-compute` (88,447), the kernel family, Redline/PM4 lowering, `radiowave`,
and the quant formats. That is 124,348 lines of genuine differentiation, it is
where the performance advantage lives, and **none of it is what is broken.**
The compute layer is not touched by any item in § 1.
