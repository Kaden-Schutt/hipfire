# Adaptive DFlash — proposal

**Status:** proposal / not implemented. Parked sidequest (surfaced while investigating
DFlash behavior through the serve path; the active branch work is the TUI + serve UX).
**Hardware of record:** k9lin / gfx1100 (RX 7900 XTX). **Target:** qwen3.6-27b.mq4 +
`qwen36-27b-dflash-mq4` drafter, kv `q8`, through `serve` (`/v1/chat/completions`).

## Thesis

DFlash's decode speedup is **entirely a function of τ** (accepted draft tokens per
verify cycle), and **τ is observable at runtime after a handful of cycles.** So the
static dispatch gate (`greedy ∧ thinking-off ∧ arch`) — which decides once, blind to
τ — is the wrong instrument: it both leaves wins on the table (it forbids
thinking-agentic, which is a measured win) and cannot avoid the prose-loss case.
Replace it with a **probe-then-commit τ gate**.

## Evidence (measured 2026-06-19, current build, through serve)

AR baseline (qwen3.6-27b, thinking-on default): **≈41 decode tok/s**.

| workload | mode | τ | decode tok/s | vs AR |
| --- | --- | --- | --- | --- |
| chat / prose ("octopus fact") | greedy dflash | **0.97** | 36.9 | **−10% (loss)** |
| code-gen (fib memo) | greedy dflash | 8 | 169 | ~4.1× |
| code (lru_cache, thinking off) | greedy dflash | 11 | 217 | ~5.3× |
| tool-call (`get_weather`) | greedy dflash | 9 | 182.8 | ~4.5× (valid `<tool_call>`, `finish=tool_calls`) |
| reasoning (17×24, thinking **on**) | greedy dflash | 6.3 | 133.8 | ~3.3× |
| code (fib), **sampled** temp 0.7 | sampled dflash | 7.4–8.2 | 94–99 | ~2.3× |

Takeaways that drive the design:

1. **τ tracks output predictability.** Structured/code/tool-call → τ≈8–11 → 4–5×.
   Free-form prose → τ≈1 → *negative* value (pure draft overhead).
2. **Thinking + dflash works now** (τ≈6, 133 tok/s) — the `budgeted_thinking_needs_ar`
   fallback is stale (its code comment's "DFlash doesn't continue after the forced
   think-close" no longer holds; observed a clean `<think>…</think>` + correct answer,
   `finish=stop`).
3. **Sampling is correct but costlier.** The acceptance loop is proper Leviathan
   rejection sampling (`speculative.rs:~3515`, `u·p_d ≤ p_t`, residual resample on
   reject); forced at temp 0.7 it produced coherent, *varied* (genuinely sampled),
   correct output. τ held (~8) but decode tok/s ~halved (169→~95) — per-cycle softmax
   over the 248k vocab + rejection/residual sampling + loss of the greedy argmax/n-gram
   fast path. Still ~2.3× over AR.

## Current gating (what we'd change)

`daemon.rs:~9521`:

```rust
if m.dflash.is_some()
    && temp <= 1e-6                      // greedy only (dispatch gate)
    && (m.arch_id == 5 || m.arch_id == 6)
    && !budgeted_thinking_needs_ar       // thinking budget -> AR
    && !force_ar_chat                    // HIPFIRE_DFLASH_CHAT=0
{ generate_dflash(...) }
```

Plus the spec-step call site hardcodes `temp = 0.0` (`daemon.rs:~7945`), so sampling
never reaches the (capable) rejection-sampling path.

## Design — probe-then-commit

1. **Start broad.** For any eligible request (`dflash.is_some() && arch∈{5,6}`),
   *start* in dflash. **Drop `budgeted_thinking_needs_ar`** — thinking is a measured win.
2. **Probe.** Track a windowed τ̄ over the first **K** committed tokens (K≈8–16 — one to
   two verify cycles of signal).
3. **Commit.** After the probe: if `τ̄ ≥ τ_breakeven`, stay dflash; else **fall back to
   AR** for the remainder.
4. **(V2) Re-probe.** Every N tokens after a fallback, try a few dflash cycles; if τ̄
   recovers (a code block appears mid-prose), switch back. V1 is one-way — it covers the
   dominant case (a prose answer stays prose; a code answer stays code).

### Why the dflash→AR switch is ~free

DFlash already maintains the **full target state** — target KV + DeltaNet prefix + the
LCP prompt-cache machinery (`daemon.rs:~7078-7120`). AR needs a *subset*. So "fall back"
= stop calling `spec_step_dflash`, do plain per-token target forwards on the **same warm
state**, discard the draft scratch. No re-prefill, no rebuild — it reuses exactly the
state the LCP-reuse path already manages.

### Calibrating τ_breakeven

The crossover (dflash decode tok/s = AR) sits around **τ ≈ 1.2–1.5** from the data above
(where a dflash cycle's extra cost — draft forward + verify — stops paying for itself).
Calibrate per-arch / per-GPU (the dflash-cycle-cost / AR-step-cost ratio differs on
gfx1100 vs gfx1201) and set the *threshold* conservatively (bail below ~τ=2) so dflash
only persists on a real win. Optional fast-path prior: a request carrying `tools` or a
code-heavy system prompt almost certainly clears the bar → commit immediately, skip the
probe; a bare chat turn could even start in AR.

## Code touch-points

- **Dispatch gate** (`daemon.rs:~9521`): loosen to `dflash.is_some() && arch∈{5,6}`;
  remove the thinking sub-gate. Keep greedy as the default speed path (sampling is a
  separate, lower-priority axis — correct but ~2× slower).
- **Verify loop** in `generate_dflash`: add a τ-window accumulator + a one-line branch to
  the existing AR step on the shared state.
- **Knobs:** `HIPFIRE_DFLASH_TAU_MIN` (threshold), probe window K, re-probe toggle —
  defaulting to adaptive-on so the common user gets it for free. Surface a one-line note
  in the TUI Settings *info* view (fixes the "thinking silently kills my speedup"
  surprise).

## Risks / must-validate-before-trusting

1. **Byte-identical handoff** — the one thing to verify first: an AR continuation *after*
   a mid-stream fallback must match a pure-AR run from that switch point (no KV/DeltaNet
   drift across the boundary). Expected to hold (AR state ⊂ dflash state) but **untested**.
2. **Probe overhead** — a prose answer pays ~K low-τ cycles before bailing (~8–16 tokens
   slightly slower than AR). Tunable; small.
3. **Coherence battery** — non-negotiable: `scripts/coherence-gate-dflash.sh` across
   chat / code / tool-call / thinking, especially the long-agentic path where dflash +
   thinking attractors have historically appeared. The measurements above are a dozen
   ad-hoc prompts — promising, **not** validated.

## Validation plan (before shipping)

1. Calibrate τ_breakeven (the crossover) on gfx1100 + gfx1201.
2. Mixed-workload A/B vs static: confirm chat falls back (no loss), code / tool-call /
   thinking stay dflash (full win).
3. Byte-identical handoff check (#1 above).
4. Full `coherence-gate-dflash.sh` across the matrix.

## Effort

~1 day to a working draft behind `HIPFIRE_DFLASH_TAU_MIN`; the calibration + handoff
parity + coherence battery are the bulk of the real work, not the probe logic itself.
