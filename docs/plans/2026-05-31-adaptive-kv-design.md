# Adaptive KV — Design

> Runtime VRAM-fit downshift of **both** K and V precision as context grows,
> recomputing the existing KV cache in place. Builds on the composable
> FWHT-K × Lloyd-V quant matrix (branch `feat/kv-vquant-fwht-lloyd-v`).
> Status: design approved 2026-05-31. Next: implementation plan.

## 1. Goal

As the live context grows toward the VRAM ceiling, the daemon **downshifts KV
precision** (V: q8→lloyd4→lloyd3→lloyd2; K: fwht4→fwht2) along a user-selected
**pattern** so it can keep fitting more tokens — turning what is today an OOM /
hard `max_seq` ceiling into **graceful degradation**. Short contexts stay at the
fastest, highest-precision modes; precision drops only as needed.

Motivated by the measured **capacity-not-speed** result: low-bit V is a
VRAM/capacity lever, not a speed lever (lloyd-V is *slower* than q8-V at every
context). So: use the fast high-precision modes where they fit; spend precision
only to reach contexts they can't.

## 2. Non-goals (v1)

- **Multi-GPU.** `set_v_mode_realloc` is single-GPU today; the transcode path is
  single-GPU in v1. Multi-GPU is a follow-up.
- **DFlash / spec-decode decode path.** v1 wires the standard linear `generate`
  decode loop. DFlash already has the sibling `maybe_evict` hook site, so it is
  an immediate fast-follow within this feature, not a rewrite.
- **K below fwht2 / V below lloyd2.** The floor is the 2-bit tier on both sides.
- **Auto-deriving the byte budget from free VRAM.** The budget is tied to the
  existing `max_seq` knob (§3). VRAM autodetect → `max_seq` is the setup
  wizard's job (see `project_hipfire_kv_vquant_wizard`), kept separate.

## 3. Capacity model — the core contract

Allocate each layer's K and V buffers **once at load**, sized for the **floor**
tier across the full advertised context. Buffers are **never reallocated**:

```
budget(layer) = max_seq × n_kv_heads × ( k_bytes_per_head(fwht2) + v_bytes_per_head(lloyd2) )
              = max_seq × n_kv_heads × ( 68 + 68 )            # head_dim = 256
```

Both buffers are **position-major, byte-strided by the current tier's
bytes-per-head** (confirmed in `kv_cache_write_fwht256_*.hip`:
`out = base + pos*bytes_per_pos + h*bytes_per_head`, with
`bytes_per_head = 4 + head_dim*bits/8`; `physical_cap` does **not** appear in V
indexing). K and V are **separate fixed-size buffers**, each allocated at its
own floor tier (`k_buf = max_seq·n_kv_heads·k_bph(K_floor)`,
`v_buf = max_seq·n_kv_heads·v_bph(V_floor)`). The usable token capacity at any
tier is the **min over the two buffers** of how many positions each holds at its
current stride (NOT a shared pool — a shared-pool formula over-estimates capacity
in lopsided states and would overflow the binding buffer). `n_kv_heads` cancels
in each ratio:

```
cap(K_tier, V_tier) = min( max_seq·k_bph(K_floor)/k_bph(K_tier),
                           max_seq·v_bph(V_floor)/v_bph(V_tier) )
```

Bytes-per-head at head_dim=256:

| cache | tier | B/head | | cache | tier | B/head |
|---|---|---|---|---|---|---|
| K | fwht4 | 132 | | V | q8 | 272 |
| K | fwht3 | 100 | | V | lloyd4 | 132 |
| K | fwht2 | 68  | | V | lloyd3 | 100 |
|   |       |     | | V | lloyd2 | 68  |

**Mental model for users:** `max_seq` is the context you are *guaranteed* at the
floor; short contexts run at the fast high-precision tiers, and precision drops
as you approach the ceiling. K + RoPE allocation is unchanged in shape — K is
just allocated at its fwht4 footprint up front. Total KV VRAM is fixed and
predictable: `max_seq × n_kv_heads × (k_bph(fwht2) + v_bph(lloyd2))`.

### Why this works without reallocation

In a fixed (floor-sized) buffer, a higher-precision tier simply uses *more bytes
per token*, so *fewer tokens* fit — exactly the smaller `cap`. Downshifting
rewrites the live tokens into a *smaller* per-token stride, compacting them to
the front, which raises `cap` in the same buffer. Forward in-place transcode is
safe because the write stride is strictly smaller than the read stride (the
write pointer trails the read pointer for every `pos > 0`).

## 4. The downshift pattern (both caches, one ordered schedule)

A **pattern** is an ordered list of steps; each step downshifts **one** cache by
one tier. This is the unifying abstraction that makes it *adaptive KV* (not just
adaptive V) and makes "selectable shift pattern" meaningful.

- K starts at **fwht4** (128-wide), V starts at **q8**.
- V chain is **q8 → lloyd4 → lloyd3 → lloyd2**.
- K tiers are **fwht4 (128) / fwht3 (256) / fwht2 (128)**, all selectable as
  floors. The **default/preset chains use `fwht4 → fwht2`** (both 128-wide ⇒ a
  cheap pure index-remap in rotated space; they reach balance without needing
  fwht3). **fwht3 enters the chain only when explicitly selected as a K tier
  under the advanced selector** (§4.1), which engages the re-rotation transcode
  (§5.3) because it crosses the 128↔256 width boundary.

**Default pattern (`balanced`, the initial proposal — see §9, finalized by the
KLD/coherence sweep):** keep the K/V bit-depth gap ≤ 1 tier at each stage, per
the validated "balanced K/V beats lopsided at equal bytes" finding, and
front-load the biggest byte win:

Capacities below are **min-of-two-buffers** at the balanced floors (K_floor=fwht2,
V_floor=lloyd2 ⇒ both buffers `max_seq·n_kv_heads·68 B`). A step fires when
`seq_pos` reaches the cap of the state *before* it (minus a margin):

| step | action | state (K/V) | binding | fire-at cap (× max_seq) |
|---|---|---|---|---|
| start | — | fwht4 / q8 | V (272 B) | 0.250 |
| 1 | V q8→lloyd4 (FWHT) | fwht4 / lloyd4 | K=V (132) | 0.515 |
| 2 | V lloyd4→lloyd3 (remap) | fwht4 / lloyd3 | K (132) | 0.515 |
| 3 | K fwht4→fwht2 (remap) | fwht2 / lloyd3 | V (100) | 0.680 |
| 4 | V lloyd3→lloyd2 (remap) | fwht2 / lloyd2 | — | 1.000 (floor) |

(Steps 2 and 3 share a fire-at point of ~0.515·max_seq because once V reaches
lloyd3 the K buffer becomes binding; `maybe_downshift` applies all crossed steps
in one call. The shared-pool idealization in earlier drafts gave a smoother
0.337/0.515/0.586/0.810 curve, but min-of-two is the physically correct,
overflow-safe model.)

Thresholds fall out as `cap(state) − margin`. **Presets** select floor +
interleave:
- `conservative` — V→lloyd4 only, K fixed at fwht4. (smallest gain, safest)
- `balanced` (default) — the table above.
- `aggressive` — same floor, K stepped earlier for capacity sooner.

The controller executes **any** ordered pattern; presets are just named
patterns. The `balanced` step order above is the starting default and is
finalized empirically (§9).

### 4.1 Advanced selector — independently configurable floors

Adaptive is **configurable, not assumed.** Alongside the three presets, an
**advanced** mode exposes two independent floor pickers:

- **K floor ∈ {fwht4, fwht3, fwht2}** — the lowest K tier the descent reaches.
- **V floor ∈ {lloyd4, lloyd3, lloyd2}** — the lowest V tier the descent reaches.

The controller **auto-generates the descending interleave** (the balanced
rule: keep the K/V bit-gap ≤ 1 tier, front-load the biggest byte win) from the
fixed start tiers (K=fwht4, V=q8) down to the chosen floors. Choosing a floor
equal to the start tier means that cache simply does not adapt (e.g. K floor =
fwht4 ⇒ K stays at fwht4 for the whole run). Selecting **K floor = fwht3** (or
fwht3 as an intermediate) is the only thing that engages the re-rotation K
transcode (§5.3); every other floor combination stays on the cheap same-width
remaps.

Note: pinning a cache to a *single static tier with no adaptation at all* is
already available via the existing non-adaptive `kv_mode` / `kv_v` load options
(shipped with the composable-KV matrix). Advanced-adaptive governs the *descent
floor*; static-pin is the orthogonal existing knob.

## 5. Components

### 5.1 `KMode` enum + `set_k_mode_realloc` (new, mirrors `VMode`)
K mode is currently encoded as `quant_asym{2,3,4}` booleans read across many
dispatch sites. Introduce a `KMode { Fwht4, Fwht3, Fwht2 }` accessor that
derives those booleans (and the rotation width: fwht4/fwht2 = 128, fwht3 =
256), so the controller can flip K tier coherently. The booleans remain the
source the forward pass reads each call (no graph baking of K mode — see §7).

### 5.2 `KvAdaptive` controller (new; sibling of `EvictionCtx`)
Holds: the resolved `pattern` (Vec of steps), current step index, current
(K_tier, V_tier), per-step thresholds, `margin`, and a **one-layer transcode
scratch** (precedent: `EvictionCtx.v_compact`). One per inference session.

```
maybe_downshift(gpu, kv, seq_pos) -> HipResult<Option<Step>>
```
Called after every committed token write, at the **same site as
`maybe_evict`**. 99% of tokens: a single integer compare → `None`. When
`seq_pos ≥ threshold(current_step)`, it runs the transcode pass for that step,
invalidates the replay graph (§7), advances the step, and returns `Some(step)`.

### 5.3 Transcode kernels
All operate per FA layer, positions `0..=seq_pos`, layer-by-layer through the
1-layer scratch (read whole layer → scratch → write back compacted) for
crash-safety.

- **V `q8 → lloyd4`** — read q8 (normal space) → dequant → **FWHT-rotate**
  (256-wide, signs seeds 42/1042) → quantize to `TURBO_C4_256` + per-(pos,head)
  cnorm. *The only transcode that does an FWHT.*
- **V `lloyd_hi → lloyd_lo`** (lloyd4→3, lloyd3→2) — read rotated indices →
  remap each to the nearest lower-LUT centroid → repack. **No FWHT.**
- **K `fwht4 → fwht2`** — same shape as the V lloyd remap, 128-wide, on the K
  buffer. **No FWHT** (same rotation width). This is the default/preset K step.
- **K re-rotation transcode** (`fwht4 → fwht3`, `fwht3 → fwht2`) — used only
  when fwht3 is selected as a K tier (advanced, §4.1). Crosses the 128↔256 width
  boundary, so it cannot be a same-width remap: reconstruct normal-space K
  (dequant + inverse rotation at the source width) → re-rotate at the target
  width → quantize to the target LUT. This **reuses the existing
  `kv_cache_write_fwht{2,3,4}` write kernels** fed reconstructed normal-space K;
  the only genuinely new piece is the dequant+inverse-rotation read. Costlier
  than the remap (a normal-space round-trip), but off the hot/default path.

**Plan spike (cheap, do first):** confirm `TURBO_C{2,3,4}_256` (V) and the K
LUTs are normalized to a shared scale. If yes, every lloyd→lloyd / fwht4→fwht2
remap collapses to a **fixed host-built `idx_hi→idx_lo` table** (16→8→4 / 16→4
entries), cnorm unchanged — a pure gather. If not, the remap recomputes cnorm
per (pos,head). Either way it is a single rotated-space pass.

### 5.4 Decode-loop hook (`crates/hipfire-runtime/examples/daemon.rs`)
In `generate` (linear path), after the token append + `seq_pos += 1`, call
`kv_adaptive.maybe_downshift(gpu, kv, seq_pos)` (guarded by `Option`). Mirrors
the existing `ev.maybe_evict` placement. DFlash (`generate_dflash`) gets the
same call at its committed-position site as the fast-follow.

### 5.5 Config / TUI surfacing
Mirror the existing `kv_mode` / `HIPFIRE_KV_V` wiring:
- **daemon**: `HIPFIRE_KV_ADAPTIVE=off|conservative|balanced|aggressive` env +
  per-load `params.kv_adaptive`.
- **TUI** (`cli/index.ts` settings menu, alongside `kv_cache`, `physical_cap`,
  CASK descriptions): a `kv_adaptive` entry — off + the three presets + an
  **advanced** option (§4.1) exposing independent K-floor / V-floor pickers +
  help text explaining the max_seq-as-floor-context contract. The env / per-load
  form accepts the preset names *and* an explicit floor pair (e.g.
  `kv_adaptive=advanced:k=fwht3,v=lloyd2`).
- **Constraint**: requires an FWHT K mode. When adaptive is ON, the loader
  forces K=fwht4 (satisfies the constraint by construction). If a non-FWHT
  K mode is otherwise selected, adaptive is ignored with a warning (reuse the
  existing `set_v_mode_realloc` guard).

## 6. Data flow

```
per committed token:
  forward (replay graph at current K kernel + V kernarg)
  → sample → append: write K@cur_K_tier, V@cur_V_tier
  → seq_pos += 1
  → maybe_downshift(seq_pos):
       if seq_pos < threshold(step):  return None        # the common case
       else:
         transcode the step's cache (one rotated-space pass, via 1-layer scratch)
         invalidate AR replay graph cache                 # §7
         flip kv tier (VMode kernarg  OR  KMode booleans)
         advance step; recompute next threshold
  ...
  floor reached → maybe_downshift is a no-op → fall through to existing behavior
                  (CASK eviction if enabled, else the normal max_seq ceiling)
```

## 7. Critical integration risk: the AR replay graph

AR decode uses a captured/replayed HIP graph (`Gpu.replay_graph_cache`,
`ar_forward_replay_enabled`, `captured_graph`, `graph_exec`, keyed by batch `n`).
A captured graph bakes the dispatched **K kernel** and may bake the **V-mode
kernarg** by value. Therefore **any** downshift — K *or* V — MUST invalidate the
replay cache so the next forward re-captures at the new mode. This is the #1
plumbing item and is validated by the very **first** spike:

> **Spike 0 (blocking):** with adaptive forced to shift at a fixed early
> position, confirm that *without* replay-cache invalidation the post-shift
> tokens are corrupt/stale, and *with* invalidation (`replay_graph_cache.clear()`
> for the affected `n`, drop `replay_warmed_up`) they are correct. This gates the
> whole design; resolve it before building the pattern controller.

## 8. Error handling / edge cases

- **Transcode HIP error** → poison the cache + return a clean generation error;
  the 1-layer scratch guarantees no half-rewritten *live* buffer.
- **Budget too small** for even one token at the start tier (K4/q8) → clamp /
  assert at load with an actionable message.
- **`head_dim == 256` and FWHT-K** already asserted by `set_v_mode_realloc`.
- **Re-prompt / cache reset** mid-session must reset the controller to step 0 and
  the buffers to the start tiers (re-quantize forward, or simply restart at q8/
  fwht4 on the next prefill — prefill rewrites the cache anyway).

## 9. Default on/off + pattern finalization

- **Default OFF (opt-in)** for v1, consistent with "capacity-not-speed → keep
  Q8-V default." Because adaptive runs the fast high-precision tiers until the
  cap, enabling it has **zero perf cost at short context** and only helps at long
  context — so **default-on is a strong post-validation follow-up**, flagged but
  not v1.
- The **exact `balanced` step order is finalized empirically** by the existing
  KLD + coherence sweep (we do not hand-guess the optimal interleave). The §4
  table is the starting proposal; the sweep confirms or reorders it.

## 10. Testing / validation

- **Spike 0** (§7) — replay-graph invalidation. Blocking; first.
- **Unit** — `cap(tier)` + threshold arithmetic; transcode correctness:
  `q8→lloyd4` transcode of a known V ≈ a direct lloyd4 write of the same V
  (within quant error); each `lloyd_hi→lloyd_lo` and `fwht4→fwht2` remap ≈ a
  direct write at the target tier.
- **Coherence (the critical gate)** — a long generation crossing **all four**
  steps must pass `scripts/coherence-gate.sh` with no attractor at any
  transition (K transitions especially — softmax-exponent sensitivity). A
  corrupting transcode surfaces precisely as a transition-point attractor.
- **KLD continuity** — a sequence that transcodes at position P matches the
  static-mode KLD for the tier it lands in (reuse the 12-cell matrix harness).
- **Perf** — short-ctx (start tier = K=fwht4 / V=q8) perf == the equivalent
  static config, with zero adaptive overhead until the first threshold;
  transcode-pass cost measured and confirmed amortized (one O(ctx) pass per
  step, 4 steps total over a full context).
- All on gfx1100 / `qwen3.6-27b.mq4`, established harness.

## 11. Build sequence

1. **Spike 0** — replay-graph invalidation (blocking gate).
2. **Shared infra** — `KvAdaptive` controller, capacity/threshold math, decode
   hook, config/TUI surfacing, transcode orchestration + 1-layer scratch.
3. **V transcodes** — `q8→lloyd4` (FWHT) + lloyd→lloyd remaps; validate KLD +
   coherence across the V-only sub-pattern.
4. **K transcodes** — `KMode` accessor + `fwht4→fwht2` remap + flag-flip;
   validate K transitions *hard* (coherence).
5. **Pattern tuning** — finalize the `balanced` default via the sweep.
6. **Wire-up** — presets, env, per-load param, TUI entry; opt-in default.
7. **(fast-follow within feature)** DFlash decode-path hook.

## 12. Status — shipped 2026-05-31

Implemented end-to-end and validated on gfx1100 (qwen3.6-27b.mq4), fleet-hardened
on gfx1201. Commits on `feat/kv-vquant-fwht-lloyd-v` (Spike 0 → wire-up).

**What landed:**
- `Gpu::invalidate_for_kv_mode_switch` (defensive graph invalidation; Spike 0).
- `kv_adaptive` module: `KMode{Fwht4,Fwht3,Fwht2}`, capacity = **min-of-two
  separate buffers** (corrected from the shared-pool draft — see §3), `KvAdaptive`
  controller (presets + advanced floors), `maybe_downshift` (applies all crossed
  steps). CPU-unit-tested.
- Transcode kernels: `kv_transcode_v_q8_to_lloyd4` (FWHT), `kv_transcode_v_lloyd_down`,
  `kv_transcode_k_fwht4_to_fwht2` (same-width remap), `kv_transcode_k_fwht4_to_fwht3`
  (re-rotation). `KvCache::transcode_v_step` / `transcode_k_step` orchestrate
  in place via a 1-layer scratch; `set_adaptive_floor_alloc` floor-sizes K and V
  + upgrades signs to 256.
- Daemon: `LoadedModel.kv_adaptive`, hooks after the prefill-chunk and decode
  eviction sites, `HIPFIRE_KV_ADAPTIVE` env + per-load `params.kv_adaptive`.
- CLI: `kv_adaptive` settings-menu entry (off | conservative | balanced |
  aggressive | advanced:k=,v=).

**Validated:** synthetic transcode≈direct (all 4 kernels, max diff = one
quant-boundary step) on gfx1100 AND gfx1201; preset + advanced coherence
end-to-end (downshifts fire at predicted positions; fluent through every
transition incl. the attractor-prone K steps). Default OFF (opt-in).

**Deferred (see `NEXT-STEPS.md`):** DFlash hook, default-on decision, multi-GPU,
pattern-tuning KLD sweep, `Aggressive` differentiation, recency-tiered precision.
