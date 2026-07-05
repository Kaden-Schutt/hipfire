# 2-Stage Decode-Optimization Autoresearch — Design

**Date:** 2026-07-05
**Status:** Approved (design), pending implementation
**Author:** Kaden Schutt (with Claude)

## Goal

A **model-agnostic** kernel-decode autoresearch harness that finds parity-clean
per-arch decode wins in two stages: (1) codex open-explores, self-targeting from
the *measured bottleneck*; (2) when the self-targetable space is exhausted, Claude
brainstorms the levers codex missed *from the exploration logs*. One launcher,
two modes, one shared certify.

## Non-negotiable constraints

- **Model-agnostic.** Nothing hard-codes a3b / mq4 / a specific kernel. The
  launcher is parameterized by `{arch, model, quant, kernel-set, dev, card}`;
  the a3b MoE-decode kernels are merely the *current input*, not baked in. The
  same flow must run for ds4, gemma4, cohere2moe, dense models, any quant.
- **Parity-gated.** Both stages certify through `ab_certify_v2p` — token-id
  parity vs baseline (byte-exact / value-preserving) is mandatory; a
  coherent-but-numerically-wrong candidate (e.g. a work-dropping change) is
  rejected as `PARITY_FAIL`. This is the guard the false `gateup_halfgrid` win
  exposed.
- **`.hip`-only surface** (kernel bodies). No Rust / launch-config editing (that
  surface — `EXPANDED_SURFACE.md` — is parked; free codex editing of the Rust
  dispatch produced only build-fails).
- **Mechanical control, not self-report.** The dry-streak counter and the ledger
  live in the **loop driver** wrapping codex, so "no new WIN in N" and "what was
  tried" are *measured*, not reported by codex (which breaks discipline).

## Architecture

```
fire_decode_ar.sh --mode explore|queue --arch <gfx> --model <path> --quant <q>
                  --kernels <comma-list> --dev N --card N [--dry-streak N]
   │
   ├─ --mode explore ─► Stage 1 driver ─► codex (hip_task_bod.txt, kernel-set injected)
   │                       │                 ▲  reads target_var/profile from each verdict
   │                       │                 └─ certifies each candidate
   │                       ├─ ab_certify_v2p (parity + adaptive + PREBUILT_BASE + BOD emit)
   │                       ├─ appends stage1_ledger.jsonl  {lever,kernel,verdict,delta_pct,target_var_delta}
   │                       └─ dry-streak counter → on N-no-win → write stage1_exhausted + final BOD
   │
   └─ --mode queue ───► Stage 2 driver ─► codex (loop_round_prompt_queue.txt + certify_queue.json)
                           └─ ab_certify_v2p  (same certify)

Stage 1 exhausted ──► Claude workflow reads stage1_ledger.jsonl (+ final BOD)
                      ──► brainstorm + red-team MISSED levers ──► emits certify_queue.json
                      ──► operator runs --mode queue
```

## Components

### 1. `fire_decode_ar.sh` (launcher, model-agnostic)
- Parses `--mode --arch --model --quant --kernels --dev --card --dry-streak`.
- `--mode explore`: renders `hip_task_bod.txt` with `{arch, arch_pred, model,
  quant, kernels, dev, card, baseline_ref}`, launches the Stage-1 driver.
- `--mode queue`: renders `loop_round_prompt_queue.txt`, launches the Stage-2
  driver (expects `certify_queue.json` present).
- Replaces the ad-hoc `fire_moe.sh` (which hard-coded moe kernels).

### 2. Stage-1 driver (wraps codex, owns the mechanical state)
- Spawns codex with the BOD prompt; codex proposes candidate `.hip` + calls
  `ab_certify_v2p`.
- After each certify: parse the verdict JSON, append a `stage1_ledger.jsonl` row
  (`lever`, `kernel`, `verdict`, `delta_pct`, `target_var_delta` from the emitted
  `profile_feedback`).
- Maintain `dry_streak`: reset to 0 on a parity-clean WIN, else +1. When
  `dry_streak >= N` (default 10): write `stage1_exhausted` marker + a final BOD
  snapshot, stop the loop.

### 3. `hip_task_bod.txt` (Stage-1 prompt — the immediate deliverable)
- Same as `hip_task.txt` (`.hip`-only, parity-gated, kernel-set injected) PLUS:
  > After EACH certify, READ the returned `profile_feedback` / `target_var`
  > (occ / L2-hit / VGPR / mem_busy vs baseline). Your NEXT candidate MUST attack
  > the measured limiter it names — state which var you're targeting and why.
  > Do NOT pick levers from generic priors; drive from the BOD.
- Kernel-set is a `{kernels}` placeholder (injected), not a fixed list.

### 4. Stage-2 Claude workflow (`decode_ar_stage2`)
- Input: `stage1_ledger.jsonl` (tried levers + verdicts + profile deltas) + final
  BOD.
- Reasons: *what codex tried, what died and WHY (the profile delta), where the
  bottleneck ended up* → brainstorms levers codex missed (structural / cross-kernel
  / ones needing math it wouldn't invent), red-teams them, emits
  `certify_queue.json` (each item: `order, name, kernel, arch, change, precheck,
  coherence_check, certify_protocol`).
- Model-agnostic: no assumptions about which model/kernels — reads them from the
  ledger.

## Data flow

`certify verdict (w/ profile) → codex reads target_var → self-targets next lever →
certify → driver appends ledger + updates dry-streak → …(N no-win)… →
stage1_exhausted + final BOD → Claude brainstorm → certify_queue.json →
--mode queue → Stage-2 loop`

## Error handling / edge cases

- **Codex ignores the BOD** (picks a blind lever): allowed — the certify still
  grades it; the ledger records the miss. The prompt *requires* BOD justification
  but the harness doesn't hard-block (soft discipline; the ledger surfaces
  non-compliance for review).
- **Parity fail / build fail**: normal ledger rows, count toward dry-streak (a
  build-fail is a non-win).
- **False win class**: impossible to bank — parity gate rejects value-changing
  candidates before the A/B.
- **Empty queue at Stage 2**: `--mode queue` refuses to start without a
  `certify_queue.json` (fail fast, not a silent no-op).

## Testing

- **No-GPU**: launcher arg-parsing + prompt rendering (placeholders all
  substituted, no leftover `CN`-style tokens — the substitution bug that caused
  `BASELINE_BUILD_FAIL`); ledger append + dry-streak state machine as a unit test
  with synthetic verdict JSON.
- **GPU (fleet)**: one explore loop on a known kernel-set reaches a real
  parity-clean WIN and a real DEAD; dry-streak trips after N synthetic no-wins and
  writes the marker; Stage-2 workflow emits a schema-valid `certify_queue.json`
  from a captured ledger.

## Out of scope (YAGNI)

- Auto-triggering Stage 2 from the marker (operator runs `--mode queue` — keeps a
  human gate on GPU spend and lets the ledger be inspected first).
- Launch-config / Rust surface (parked in `EXPANDED_SURFACE.md`).
- Cross-arch fold/promote (separate existing flow).
