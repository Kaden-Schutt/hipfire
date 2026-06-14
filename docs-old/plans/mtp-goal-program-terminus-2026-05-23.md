# MTP decode-acceleration program (Goals A→B→C) — terminus & joint rulings

**Date:** 2026-05-23
**Branch:** `mtp-hiptrx-rocprof`. Hardware: k9lin / 7900 XTX (gfx1100), Qwen3.6-27B MQ4 trunk.
**Method:** Codex/Claude mutual-verification loop — Codex independent source root-cause, Claude empirical bench, neither lands a claim unverified.

## One-line outcome

Goal A delivered a real native-MTP-solo runtime gain (1.12x → 1.56x AR, committed) but its ≥100 tok/s floor is structurally unreachable. Goals B and C are mutually-ruled asymptotic: MTP+DFlash composition cannot beat DFlash-solo because MTP is non-orthogonal to DFlash's already-tile-saturated block draft. All three outcomes reduce to ONE structural fact (below).

## The unifying structural fact

There is a single bandwidth-bound trunk-verify forward per speculative cycle (~14 ms hard floor: 13.5 GB MQ4 weights ÷ ~960 GB/s). Throughput = committed-tokens-per-cycle ÷ cycle-wall, and the lever is how many accepted tokens fill that verify forward's 16-wide WMMA tile. DFlash's block-diffusion drafter fills it near-optimally (τ≈10.9 at the perfmax regime — a *batched, non-autoregressive* draft that produces ~16 candidates in one cheap forward). MTP — whether solo or as a composition tail — has no mechanism to add committed tokens per trunk pass:
- **Solo:** filling the tile requires generating candidate nodes, and the native MTP head generates them *autoregressively* at ~1.3 ms/node (serial); a batched MTP node-gen is architecturally blocked (self-only attention; logical-position and KV-slot indexing are coupled, so tree siblings can't share a position). The serial node-gen tax cancels the tile-fill benefit → Goal-A asymptote.
- **Composition tail:** MTP's tail tokens, conditioned on a DFlash-verified prefix, are non-orthogonal — they almost never match the trunk's continuation that DFlash's drafter missed → Goal-B/C asymptote.

DFlash is the architecturally-correct tile-filler for this trunk; MTP is strictly dominated by it.

## Goal A — DELIVERED (gains) but floor unreachable. [committed: 86fcf4ff fix, c758ab62 ruling]

Real runtime/dispatch wins, zero head retraining:
```
AR baseline 44.6 → MTP K=2 pre-fix ~53 (1.19x) → +GDN-tape replay fix ~60 (1.35x) → +K=3 ~68 → +p_min=0.50 ~70 (1.56x)
```
- **GDN-tape replay fix** (Codex diagnosis, Claude-verified +16.5% back-to-back A/B, τ-invariant): the non-full-accept cycle was doing a full trunk forward replay to repair DeltaNet state; replaced with DFlash's cheap branch-aware GDN innovation-tape replay. The session's blindspot — visible in the cycle trace, dismissed as "~3 ms" when it was the ~15-20 ms dominant non-verify cost.
- Cheap rollback then unlocked K=3 (deeper chain, previously penalized by the per-cycle full replay); p_min=0.50 the refined tuning point.
- **Asymptote (mutual):** native-MTP-solo caps ~70-85 tok/s. Confirmed two ways — Codex (source: batched MTP node-gen architecturally blocked) and Claude (bench: linear high-K sweep, τ climbs 3.1→3.4 at K=6 but tok/s flat at ~70 — the serial node-gen tax cancelling the deeper-acceptance gain). ≥100 needs a batched non-AR drafter (DFlash), excluded from Goal A by definition.

## Goal B/C — joint asymptote ruling: composition cannot beat DFlash-solo.

Goal-A→B gate opened on the mutual asymptote agreement. Findings:
- **Corrected DFlash-solo bar:** ~219.6 tok/s, τ=10.93 on the blessed config (merge_sort_thinking_off, ctx 4096) — tight 0.7% spread, matches the repo speed-baseline anchor and a human-blessed prior run. (4.9× the same-prompt AR ~44.8.)
- **Current composition (`spec_step_dflash_mtp`) is a pessimal hybrid** (Codex source verdict): placement is the correct additive-tail (MTP commits only after DFlash full-accept), but latency is an *unconditional* every-cycle tax — the MTP fanout and a widened B+K verify run regardless of full-accept. Empirically this regresses to ~140-148 tok/s (−33% vs the 219.6 bar).
- **The orthogonality verdict (linear tail) — non-orthogonal in every τ regime (Claude empirical):** the MTP additive-tail commit rate is τ_mtp ≈ 0.11/cycle at high DFlash τ (tile full) and ≈ 0.04-0.056/cycle in the tile-slack regime (DFlash starved, ~half the tile empty). Near-zero whether or not there is room — the slots DFlash leaves empty are empty because the continuation is genuinely unpredictable, and the native MTP head can't predict it either.
- **Restructure pre-screened futile (the airtight pre-screen logic):** Codex's conditional-additive-tail restructure changes only *when* MTP work is paid (cost), not the *acceptance criterion*. So the measured τ_mtp ~0.04-0.11 is a faithful *upper bound on the additive-tail prize independent of the restructure*. The best the (coherence-risky split-verify) restructure could do is convert the −33% regression into ≈ DFlash-solo + 0.05-0.1 tokens/cycle = parity within noise — never a win. Building it would be wasted effort; the existing pessimal-but-acceptance-faithful code served as a perfectly good measurement instrument to establish this without building anything.
- **The tree topology (`spec_step_dflash_mtp_tree`) — the loop's reverse catch and its resolution.** Codex's reciprocal audit correctly refuted the *universal* "composition is dead" claim: the linear upper bound covers only the tail-after-the-whole-block topology; the per-slot tree topology attaches MTP children to each DFlash slot and could in principle commit at *partial*-accept frontiers (the common case), which the linear measurement never touched. Claude pre-screened the existing tree demo on the blessed prompt: it is a *catastrophic regression* — ~7 tok/s (below AR), because the tree-mode verify destroys DFlash's own base acceptance (tau_dflash collapses 10.7 → 0.45 — the documented tree-mode RoPE-phase-skew, the attractor-prone machinery the coherence gate exists for), with MTP-child commit rate τ_mtp = 0.015 (even lower than the linear tail). The broken base means this run can't *fairly* measure a healthy-frontier prize — but that prize is bounded conceptually: a tree child only adds tokens by extending *past* the free trunk-argmax bonus DFlash already commits at the frontier, which is exactly native MTP-solo drafting from that prefix — re-entering the Goal-A MTP-solo asymptote, plus a multiplied serial-node-gen tax across slots. **Codex co-signed this conceptual bound** (verdict: CO-SIGN — "no mechanistic escape in the current tree topology; it just relocates MTP-solo onto DFlash's accepted prefix").

- **Goal C** ("+25-100% over current-best via composition") inherits the same impossibility — composition can't even reach DFlash-solo, let alone exceed it.

**Joint ruling status — CLOSED, mutually co-signed (Codex + Claude).** Tiered: linear composition asymptote affirmed by upper bound; existing tree composition affirmed by catastrophic empirical regression; the lone theoretical loophole (a repaired-tree healthy-frontier orthogonality) is bounded by the Goal-A MTP-solo asymptote and gated behind a high-coherence-risk tree-mode-verify repair — low expected value, not worth funding absent a *specific new mechanism that cheaply generates deeper frontier-conditioned tokens without native MTP's serial tax — which would be a different batched/non-AR drafter project, not MTP+DFlash composition.*

## Methodology lessons (the failure log that narrows future search)

- **Head retraining is a dead end for runtime speedup** (v1-v7 prior session: offline argmax-agreement +15-18pp, runtime flat-to-negative). Wins are in dispatch/runtime structure, not weights.
- **Harness trust — the "84 tok/s" fiasco.** A DFlash-solo bar was first mismeasured at ~84 tok/s on the wrong prompt at default ctx=512. Jointly investigated: Codex source audit (default ctx hard-stops the decode loop early; adaptive-B shrinks the block on low-τ prompts — both faithful readings of a starved regime, not a measurement bug) + Claude arithmetic (both numbers fit one cycle-cost model differing only in τ) + a direct reproducibility replay (the 84 reproduces as the *center of a chaotic ~72-95 band* — ~30% run-to-run variance from adaptive-B's feedback instability in the low-acceptance regime). Verdict: harness sound; the fault was driving it into an ill-conditioned regime.
- **Mandatory bench pins** (keep the harness in its well-conditioned, reproducible regime — 0.7% spread vs ~30% chaos): pin the blessed prompt + `--ctx 4096`, `--temp 0.0 --no-chatml --kv-mode q8`, `HIPFIRE_NORMALIZE_PROMPT=1`, `HIPFIRE_VERIFY_GRAPH` and `HIPFIRE_DPM_WARMUP_SECS` set consistently, and pin adaptive-B off anywhere the regime isn't well-conditioned (moderate/low-τ).
- **Prompt-structure τ sensitivity bit hard:** speculative speedups only materialize on prompts/ctx where the drafter sustains acceptance; the same DFlash binary reads τ≈3.3 on a starved LRU/small-ctx point and τ≈10.9 on the blessed config. Every cross-comparison must pin byte-identical prompt + ctx.

## What the program proves about the architecture

The native MTP head is a genuine but modest decode accelerator (~1.5x on fast consumer hardware where the AR baseline is already efficient — consistent with the 1.4-3x range in published MTP work, whose higher ratios come from slower baselines). DFlash's block-diffusion architecture is the real tile-filling lever (~5x). The two do not compose because they amortize the same single trunk-verify forward and DFlash already saturates it. For tok/s beyond DFlash-solo on this trunk, the lever is a *better tile-filler* (a stronger or larger non-AR drafter), not an MTP augmentation.
