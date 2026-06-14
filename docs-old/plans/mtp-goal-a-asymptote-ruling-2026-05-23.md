# Goal A asymptote ruling — native MTP-solo decode caps at ~70-85 tok/s

**Date:** 2026-05-23
**Branch:** `mtp-hiptrx-rocprof`
**Hardware:** k9lin / 7900 XTX (gfx1100), Qwen3.6-27B MQ4 trunk.
**Objective:** native MTP-solo decode to ≥100 tok/s (target 120), no DFlash / no separate drafter.
**Outcome:** floor structurally unreachable on the native head. Mutual Codex/Claude asymptote agreement — the Goal-A→B gate condition.

## What landed (the real Goal-A wins — all runtime/dispatch, zero head retraining)

```
AR baseline:                       44.6 tok/s   1.00x
MTP K=2, session-start complaint:  ~53          1.19x   (felt like ~1.1x against a fast AR baseline)
+ GDN-tape replay fix (committed 86fcf4ff): ~60  1.35x   — Codex diagnosis, Claude verified +16.5% back-to-back A/B, τ-invariant
+ K=3 (cheap rollback unlocks deeper chain): ~68 1.52x
+ p_min=0.50 tune (refined recipe):  69.57       1.56x   — 5-run fresh-process median, τ≈3.08
```

Refined canonical MTP-solo recipe: **K=3, p_min=0.50, compressed-serial, --no-chatml, --kv-mode q8**.
The goal's thesis — "the wins are in runtime/dispatch structure, not weights" — was correct: a ~40% margin gain (1.12x→1.56x) with no retraining, after a session of head-retraining (v1-v7) had produced flat-to-negative runtime gains.

## Why ≥100 is structurally unreachable on the native head (the heavy-lever analysis)

The bottleneck is the bandwidth-bound trunk verify forward (~14 ms hard floor: 13.5 GB MQ4 weights / 960 GB/s). The verify runs at batch M=K+1; at K=3 that's M=4, filling only 4/16 of the 16×16 WMMA tile while paying the full weight-read bandwidth bill. Prior rocprof established the verify wall is ~constant from M=4 to M=16 — so *filling* the tile with more candidate tokens is near-free on the trunk side. That tile slack is exactly what DFlash exploits (M=16 block draft → τ≈9). The Goal-A question was whether native MTP could fill the tile.

The only native-MTP mechanism to fill the verify tile is a sparse token-tree (Medusa-style). The full tree scaffold already exists in-repo for DFlash's DDTree path (`build_ddtree_tree_with_cutoff`, tree-attention-mask linearization, `follow_verified_tree`, a tree-aware GDN innovation-tape replay) — so the tree would have been a *wire-up*, not net-new infrastructure. But it cannot reach the floor for two compounding structural reasons, confirmed independently:

**1. Serial node-generation tax (the term the optimistic estimate omitted).** A tree's nodes must be GENERATED before the trunk can verify them. The MTP head generates them autoregressively — one serial `mtp_head_forward_block_only` per node at ~1.3 ms (cycle-anatomy K-sweep delta, on-hardware). A 16-node tree costs ~21 ms of serial generation vs the linear K=3 chain's ~4 ms — a new ~17 ms/cycle that lands the tree near ~75-85 tok/s, not 100. DFlash escapes this only because its drafter emits all M=16 candidates in one batched non-autoregressive forward (~5 ms total) — the architectural capability a single autoregressive MTP block lacks by construction.

   - **Source confirmation (Codex):** the one exposed batched MTP path, `mtp_head_forward_block_batched` (mtp_head.rs:1395-1450), is *self-only attention*; same-depth tree siblings cannot share a logical position while using distinct physical KV slots because `positions[i]` drives both RoPE phase and KV-slot indexing simultaneously. A correct tree-sibling batched pass needs parent-conditioned attention the generator does not implement — and that coupling is the same tree-mode RoPE-phase-slot skew that produced the project's documented DDTree token-attractor incidents.
   - **Bench confirmation (Claude):** a linear high-K sweep (the tree's serial-cost proxy) at p_min=0.50 — K=3 → ~69.6 tok/s τ≈3.1; K=6 → ~69.9 tok/s τ≈3.4. τ rises with depth but tok/s is dead flat: the deeper-acceptance gain is exactly cancelled by the extra serial node-generation forwards. The cancellation mechanism, observed on hardware.

**2. Greedy verify caps tree-width payoff.** The Goal-A recipe is temp=0 greedy; `spec_step_ddtree`'s verify is strict argmax-match (`follow_verified_tree` selects the single child equal to the trunk's argmax). Tree *width* therefore recovers at most one extra accepted child per cycle — and only on cycles where the MTP top-1 diverged from the trunk — while paying the full serial generation cost. Expected width payoff does not cover its cost. (Tree depth is separately exhausted: τ saturates ~3.4.)

A correct tree-aware *batched* MTP-head generator (parent-conditioned attention + decoupled logical-position/KV-slot indexing) is the only thing that could break the serial tax — a multi-week kernel project that re-enters the tree-mode-skew attractor minefield with a payoff bounded by the temp=0 width-weakness. Outside the scope and spirit of Goal A's cheap-runtime-lever brief.

## The ruling

Native MTP-solo greedy on this trunk/hardware is **asymptotic in the ~75-85 tok/s band** (we sit at ~70 with cheap levers banked; the residual to ~85 would come from a hipGraph launch-overhead trim — a partial lever both sides judged not worth the plumbing for a goal whose floor is unreachable regardless). The ≥100 floor requires a batched non-autoregressive drafter to fill the verify tile cheaply — that is DFlash, excluded from Goal A by definition.

**Mutual asymptote agreement (Codex source analysis + Claude empirical bench, independently concurring) → the Goal-A→B gate opens.**

## Hard-won lessons logged (so the failure narrows the search)

- MTP-head weight retraining is a dead end for *runtime* speedup (v1-v7 this session: offline argmax-agreement +15-18pp, runtime flat-to-negative). Documented separately in the FastMTP v1-v7 docs.
- The asymptote's root cause — verify-tile under-fill that only a batched non-AR drafter resolves — is precisely the architectural gap DFlash was built to fill. This *reframes Goal B's hypothesis*: DFlash already supplies the cheap M=16 tile fill; the composition question is whether MTP candidates add anything *orthogonal* on top, and the same serial-node-gen tax that sank MTP-solo will weigh on the MTP contribution within composition (memory's "previously a wash"). The new variable Goal B tests is whether the GDN-tape-reduced MTP cycle latency tips that balance.

## Goal B kickoff (gate now open)

1. Re-establish the DFlash-solo baseline on *this* hardware/prompt as the bar to beat (the goal cites a 255 tok/s 27B perfmax — verify the current k9lin number under the fixed protocol).
2. Audit the existing composed path (`mtp_compose.rs::spec_step_dflash_mtp`) — what state it's in and why it was previously net-zero.
3. Deliver the empirical orthogonality verdict + a composed path that beats DFlash-solo, coherence-clean.
