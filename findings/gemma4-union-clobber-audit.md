# gemma4 union merge — clobber audit

**Date:** 2026-06-10
**Branch:** `fix/gemma4-union-fixups` (UNION = 818a8936)
**Merge commit audited:** 72f94ab1
**Parents:** OURS = f63751a5 (`gemma4-rz`) × KEVIN = aa66352f (`unverbraucht/feat/dispatch-unification-gemma4`)
**Base for diff:** 12d2fc57 (origin/integration/dispatch-unification, Ship 5.2 + ds4/jinja)

---

## Purpose

35-agent three-way merge audit conducted 2026-06-10. The union merge at
72f94ab1 reconciled two independent gemma4 bring-up branches. Both touched
the same files; some decisions made by OURS were overwritten by KEVIN's
resolution and vice versa. This document catalogs every confirmed and
suspected clobber, records which version was correct, and states the fix
required for each item.

---

## Headline finding: the G128/G256 PARO clobber is FALSE

The suspected `PARO4G128` / `PARO4G256_T` quantizer-side clobber is a
false alarm. Both variants are **enum declaration only** — zero dispatch
sites exist; `paro: None` on every gemma4 model load path. There was no
live PARO logic to clobber.

The real contested arm was the **expert / dense down-proj fallback**:

- OURS (`f63751a5:6196-6202`) used unconditional `quantize_hfq4g128` —
  the live form of the ragged-K floor-stride bug for 26B expert K=704 and
  dense FFN K=2112.
- KEVIN (`aa66352f`) added `supports_g128` guard + Q8F16 ragged fallback
  and the `quantize_fallback_g128_or_q8` helper (10 call sites).

The union kept KEVIN's version (818a8936:6280-6290). **The merge discarded
our bug, not his work.** This is the correct outcome.

---

## Ragged-K status in union

Ragged-K is **not live for gemma4 in the union build**:

- 12B / QAT dims are all `%256 == 0`.
- 26B ragged dims (K=704, K=2112) route to Q8 via KEVIN's
  `quantize_fallback_g128_or_q8` guard.

Protection is quantizer-side only (Option C from
`findings/gemma4_ragged_kernel.md`). The kernel-side per-row re-quant fix
(Option A) remains pending. The new regression test pinned in this PR
guards the quantizer-side path.

---

## Genuine merge defects fixed in this branch

### 1. Cargo.toml duplicate workspace member (cosmetic)

`hipfire-arch-gemma4` appeared twice in the workspace `[members]` array.
`cargo` silently deduplicates workspace members, so this caused no build
failure, but it is a hygiene issue. Fixed by removing the duplicate entry.

### 2. Daemon `cache_capable` missing arch 13

The daemon's `cache_capable` match did not list `arch_id == 13`
(gemma4_unified). **Resolution: DEFER (documented, not wired).**
Investigation showed Kevin's inclusion of gemma4 in `cache_capable` was
aspirational: `generate_gemma4` has no LCP prefix-reuse logic (compare
`generate_minimax`, which does), so advertising `cache_capable: true`
would make the client skip per-turn resets and send delta prompts the
server cannot align — silent multi-turn KV corruption. This branch adds
a comment above the `cache_capable` line marking the omission deliberate.
Wire `| 13` only after `generate_gemma4` gains an LCP block and the
change is validated multi-turn on hiptrx.

### 3. Dropped credit / SPDX headers

Several files lost their copyright / SPDX headers during merge resolution:

- `crates/hipfire-runtime/src/tokenizer.rs` — 'Kate' author credit
- `crates/hipfire-arch-gemma4/src/arch.rs` and `crates/hipfire-arch-gemma4/src/lib.rs` header blocks
- Two `.hip` kernel files: SPDX-License-Identifier lines absent in union

These are restored in this branch.

---

## False alarms — for the record

**HIGH severity 'MoE dead code / 26B unreachable' (×2):** Wrong. The
daemon reaches `lowered::load_weights` and `init_scratch_constants`
directly via `daemon.rs:4185-4277`, bypassing the `Architecture` trait
dispatch. The 26B-MoE forward is reachable from the daemon even though
the Architecture trait impl shows no 26B arm.

Kevin's WMMA-prefill / 26B-MoE / ring-KV stack and our EAGLE / tokenizer /
daemon stacks are both intact in the union.

---

## Item-by-item table

| # | Item | OURS | KEVIN | Union | Clobber? | Action |
|---|------|------|-------|-------|----------|--------|
| 1 | arch_id 13 (dense) / 22 (drafter) | 13/22 | 12/missing | 13/22 | Partial | Verify GGUF drafter arm present |
| 2 | `use_f32_passthrough` | Present | Removed | Present (OURS won) | No | None |
| 3 | Vision/audio skip guard | `arch_id==13` | Removed | Re-added (~5713) | Partial | Confirm not applied to arch 22 |
| 4 | Attn Q8 guard (arch 13+22) | Present | Removed | Present (OURS won) | No | Verify GGUF vs safetensors path |
| 5 | Ragged-K Q8 fallback | Partial (expert only) | Systematic (10 sites) | Systematic (KEVIN won) | No | None (improvement) |
| 6 | `deltanet` feature gate in Cargo.toml | Present | Removed | Present (OURS won) | No | None |
| 7 | `dump_gemma4_hidden_states.rs` | Present | Deleted | Deleted | Yes (low) | Accepted loss; new oracle tools present |
| 8 | `lowered.rs` | Absent | New | Added + 2 bugs fixed | No | None |
| 9 | `config.rs` `LayerType` enum | Present | Simpler | Present (clean) | No | None |
| 10 | `attention.rs` 8k window gate | Present | Divergent | Present | No | None |

### Items 6 and 9: verified clean at HEAD

Earlier audit passes rated these high-risk; direct verification at the
union tip shows both are non-issues. Item 6: the `[features]` block
(`default = ["deltanet"]`, `deltanet = [...]`) is intact at
`crates/hipfire-arch-gemma4/Cargo.toml:10-14` — the union kept OURS's
feature gate, and `rope_partial_halved_f32` is referenced from
`forward.rs`/`drafter.rs`/`lib.rs`/`lowered.rs`. Item 9:
`pub enum LayerType` with `Sliding`/`Full` is defined at
`config.rs:23`, parsed at `config.rs:183-188`, and dispatched in
`gemma4.rs` (295/394/658/662). No action needed for either.

### Partially-recovered items requiring verification

**Item 1** — `"gemma4_unified_assistant" => 22` must appear in both the
safetensors pipeline and the GGUF pipeline model-type match arm
(`main.rs:~4321`). Bare `arch_id == 12` must not survive outside comments.

**Item 3** — Vision/audio skip guard re-added at `main.rs:~5713`. Confirm
condition is `arch_id == 13` only and is NOT applied to the drafter
(`arch_id == 22`), which uses flat `model.*` tensor names with no
`model.language_model.` prefix.

**Item 4** — Attn Q8 guard at `main.rs:~6507` (`if arch_id == 13 || arch_id == 22`).
Confirm it is guarded to the safetensors pipeline only (GGUF has its own Q8 path).

### Clean items (no action)

Items 2, 5, 8, 10 are confirmed clean. Item 7 is an accepted loss.

---

## References

- `findings/gemma4_ragged_kernel.md` — root-cause model for ragged-K bug
- `findings/phase2_gate_report.md` — Phase 2 gate pass, window-gate fix
- `findings/gemma4_dispatch_devlog.md` — full session-by-session bring-up log
- CLAUDE.md MEMORY entry `project_gemma4_union_validated_2026_06_10.md`
  (EAGLE 0.63→2.0–2.3× AR, hiptrx PASS, ragged-K G128 latent bug confirmed)
- CLAUDE.md MEMORY entry `project_kevin_gemma4_fork_review_2026_06_10.md`
  (Kevin's fork review, union plan, arch 12→13 renumber)
