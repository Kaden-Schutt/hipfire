# Design: PR #549 VMM — all static KV modes + adaptive KV

**Date:** 2026-07-28  
**Status:** design approved (interactive user approval 2026-07-28); merge-complete target for FireMap / PR #549  
**Claim class:** `planned` (this record) + scoped `measured` MQ4R evidence below (gfx1201 only)  
**Related:** PR #549 (`feat/firemap-vmm-kv`), `docs/plans/2026-05-31-adaptive-kv-design.md`, `docs/plans/2026-05-31-kv-vquant-fwht-lloyd-v-design.md`, `docs/VALIDATION.md`, `docs/REDLINE.md`

This document is the user-approved merge-complete design for extending PR #549
VMM beyond Q8/Asym3 to the full single-GPU Qwen3.5/3.6 masked KV matrix,
including legal Lloyd-V overrides and adaptive KV. It freezes architecture,
lifecycle, failure, and evidence gates. It does not change code by itself.

**Approval:** design was approved interactively by the user on 2026-07-28.

---

## 1. Status and context

### 1.1 What PR #549 already is

FireMap (PR #549) adds an opt-in HIP VMM backend for single-GPU Qwen3.5 KV:

- Reserve a stable virtual address range up front.
- Map physical pages in coarse chunks as context grows.
- Keep dense KV kernels and captured-graph base pointers unchanged.
- Default remains `contiguous`; enable with `--kv-backend vmm`.
- Reject unsupported models / PP / TP / CASK / TriAttention / unsupported
  encodings instead of silent fallback.

Draft PR scope was Q8 + Asym3 only. Merge-complete scope (this design) widens
**KV encoding** coverage, not weight formats.

### 1.2 Weight quantization vs KV encoding

These are independent axes. Do not conflate them in CLI, logs, or claims.

| Axis | What it selects | Examples |
|---|---|---|
| **Weight quantization** | How model weights are stored/decoded | MQ4R, Q8 weights, Paro, HFQ families, Lloyd **weight** dtypes |
| **KV encoding** | How the attention K/V cache is packed | static `q8`, `asym2/3/4`, `fwht2/3/4`; Lloyd-V overrides; adaptive tier ladder |

VMM is a **KV storage backend**. It does not change weight codecs. MQ4R in the
evidence section is a **weight** format used on the measured route; the KV
modes under test are separate.

### 1.3 Why this design exists

Contiguous KV pays full `max_seq × current_stride` physical VRAM at load.
Adaptive KV already floor-sizes contiguous buffers and downshifts in place, but
still commits the full floor-sized physical allocation immediately. VMM keeps
the same logical contract while deferring physical commit to mapped prefixes —
and, under adaptive, while allowing tier stride changes without replacing the
stable virtual owners.

---

## 2. Goals

1. **Single-GPU Qwen3.5 / Qwen3.6 masked KV** on VMM for:
   - all **seven static** KV modes: `q8`, `asym2`, `asym3`, `asym4`, `fwht2`,
     `fwht3`, `fwht4`;
   - **legal Lloyd-V overrides** on FWHT-K paths (`lloyd2` / `lloyd3` /
     `lloyd4` V with the existing FWHT-K + `head_dim=256` guards);
   - **adaptive KV** (floor-reserved, start FWHT4/Q8, pattern-driven downshift).
2. **One stable K VMM arena and one stable V VMM arena per real attention
   layer** for the life of the load. Kernels and retained/graph captures keep
   base pointers.
3. **Resolve final layout before allocation.** Load never creates VMM owners
   then replaces them through contiguous reallocators (`set_v_mode_realloc`,
   floor realloc that swaps buffers, etc.).
4. **Tier-aware capacity:** current K and V tier strides independently govern
   mapping growth and token capacity (min-of-two-buffers, same as adaptive).
5. **Fail closed** on map/transcode/teardown errors (no log-and-skip; no clean
   unload/reload with pending arenas; poison on partial transcode).
6. **Evidence-gated merge:** CPU layout/failure tests, gfx1201 all-seven static
   hardware, adaptive transition/reset tests, existing Q8/FWHT3 DFlash path
   proof, MQ4R matched retained-PM4 + VRAM proof (claims scoped gfx1201 only).

---

## 3. Explicit non-goals

Out of scope for this merge-complete design:

| Non-goal | Note |
|---|---|
| **Pipeline parallel (PP)** | VMM path remains single-GPU; PP stays contiguous / rejected on VMM |
| **Multi-GPU / TP** | No multi-device VMM arenas, no cross-GPU pooling |
| **CASK / TriAttention eviction and compaction** | Reject on VMM; ordinary masked-cache `physical_cap` remains part of VMM reserve sizing |
| **Flat-Llama carriers** | Masked Qwen3.5/3.6 attention layers only; flat Llama KV constructors stay contiguous |
| **Adaptive DFlash** | Existing static Q8/FWHT3 DFlash-on-VMM proof stays; adaptive×DFlash is follow-up |
| **PagedAttention / RadixAttention / prefix sharing** | Still not this PR |
| **Token-level mappings** | Coarse chunk map growth only |
| **Default-on VMM or default-on adaptive** | Both remain opt-in |
| **Kernel ISA changes for VMM** | Dense kernels unchanged; backend/ownership/lifecycle only |
| **Unscoped perf speedup claims** | No “VMM is faster” product claim; see measured MQ4R table only |

---

## 4. Current defects (must fix in the same merge)

These are known correctness/lifecycle holes the approved design closes. They are
requirements, not optional polish.

### 4.1 Adaptive controller drop

If adaptive is enabled but the controller is dropped, never installed, or not
hooked on a path that still writes KV, the cache stays at the start tier while
`seq_pos` grows past start-tier capacity → overflow / corruption.  
**Requirement:** adaptive engagement is all-or-nothing at load: controller
present, floors applied to reserve math, hooks on every committed-write path in
scope (prefill chunk boundary + decode). Missing controller ⇒ hard load error,
not silent contiguous-style growth.

### 4.2 Balanced floor mismatch

`Preset::Balanced` must be the genuine middle rung **K=`fwht3`, V=`lloyd3`**
(not identical to Aggressive `fwht2`/`lloyd2`). Reserve, thresholds, and user
docs must use the same floors. A preset that advertises balanced but reserves
aggressive floors (or the reverse) is a merge blocker.

### 4.3 Cache / controller reset desync

Any cold KV reset (context rollover, explicit reset, or failed-generation
cleanup that clears the cache) **must atomically restore both sides**:

- cache encoding state: K=`fwht4`, V=`q8`;
- adaptive controller: K=`fwht4`, V=`q8`, step index 0.

Resetting only the controller while leaving `KvCache` flags at a downshifted
tier is a desynchronization bug, as is resetting cache flags without the
controller thresholds.

### 4.4 Log-and-skip errors

Map growth failure, transcode failure, and teardown/unreserve failure must
**propagate**. Logging and continuing is forbidden: it leaves kernels pointing
at unmapped VA, half-transcoded layers, or process-global pending VMM state.

---

## 5. Considered approaches

### 5.1 Contiguous only (status quo for most modes)

Full physical commit at load. Simple, but pays peak KV VRAM immediately and
forces adaptive to either over-commit high tiers or reallocate (pointer churn,
graph invalidation storms, realloc hazard with captured pointers).

### 5.2 Single shared VA arena for K+V

One reserve per layer packing K and V. Fewer HIP VMM objects, but couples K/V
growth, complicates independent tier strides, and makes min-of-two capacity
accounting error-prone. Rejected.

### 5.3 Reallocate-on-downshift (replace owners)

On each adaptive step, free and allocate new buffers at the new stride. Breaks
stable graph/retained base pointers, fights FireMap’s purpose, and reintroduces
the contiguous realloc path the design forbids for VMM owners. Rejected.

### 5.4 Grow-only VMM at max(current) stride without floor reserve

Reserve only current-tier bytes × max_seq, then try to extend the VA when
downshift needs more tokens. HIP VA reserve extension is not a portable
primitive here; would require remap/copy and pointer changes. Rejected.

### 5.5 Selected: floor-reserved, tier-aware dual arenas (K and V)

**Winner.** Per real attention layer:

- One **K** VMM arena + one **V** VMM arena.
- **Static modes:** reserve at the **current** K/V stride × configured token
  horizon (`physical_cap` / `max_seq` as today’s constructors define).
- **Adaptive:** reserve at the **floor** K stride and **floor** V stride ×
  `max_seq` (guaranteed floor context); start tiers are FWHT4 / Q8; live token
  capacity is min-of-two at **current** strides; downshift **compacts in place**
  inside the existing VA, increases token capacity, does **not** unmap or
  replace owners, and invalidates HipGraph + retained replay.

Matches shipped adaptive capacity math, preserves FireMap’s stable-pointer
invariant, and composes with independent K/V tier changes.

---

## 6. Storage and capacity invariants

### 6.1 Owners

For each real FA/masked attention layer index `i`:

- `K_arena[i]`: stable VA, VMM-owned, never replaced for the load lifetime.
- `V_arena[i]`: stable VA, VMM-owned, never replaced for the load lifetime.
- Non-KV layers keep today’s cheap placeholders (not VMM-backed).

Placeholder layers must not register VMM arenas.

### 6.2 Reserve sizing

Let `bph_k(tier)` / `bph_v(tier)` be bytes-per-head at `head_dim` (existing
tables: K fwht4=132, fwht3=100, fwht2=68; asym packed layouts as today; V
q8=272, lloyd4=132, lloyd3=100, lloyd2=68 at hd=256).

```
static:
  K_reserve_bytes = physical_cap × n_kv_heads × bph_k(current_k)
  V_reserve_bytes = physical_cap × n_kv_heads × bph_v(current_v)

adaptive:
  K_reserve_bytes = max_seq × n_kv_heads × bph_k(k_floor)
  V_reserve_bytes = max_seq × n_kv_heads × bph_v(v_floor)
  start: cur_k = fwht4, cur_v = q8
```

Reserve is virtual. Physical pages commit via mapped-prefix growth only.

### 6.3 Token capacity (min-of-two)

```
cap(cur_k, cur_v) = min(
  reserve_tokens_k × bph_k(k_reserve_tier) / bph_k(cur_k),
  reserve_tokens_v × bph_v(v_reserve_tier) / bph_v(cur_v),
)
```

For static, reserve tier = current tier ⇒ `cap = physical_cap` (modulo existing
constructor rules). For adaptive, reserve tier = floor ⇒ capacity grows as
strides shrink. **Never** use a shared-pool sum of K+V bytes for capacity;
lopsided states would over-admit tokens on the binding side.

### 6.4 Mapping vs logical length

- `mapped_tokens_k` / `mapped_tokens_v` track how far physical mapping covers
  at the **current** stride (chunk-rounded).
- Before any KV write or attention read that touches position `p`, both arenas
  must be mapped through `p` at current strides.
- **Mapping failure aborts before write.** No kernel launch into unmapped VA.

### 6.5 Downshift compaction

On adaptive step:

1. Copy each live source-tier layer prefix `0..seq_pos` to the existing
   source-sized scratch buffer.
2. Transcode from scratch into the compacted prefix at the front of the
   **same** stable K or V VA. Do not rely on overlapping read/write safety.
3. Update current tier flags / kernarg-facing mode bits only after every real
   layer succeeds.
4. Recompute `cap` and mapping high-water in **new** stride units (the same
   mapped bytes can cover more tokens).
5. **Do not unmap** solely because stride shrank; do not replace arenas.
6. Invalidate HipGraph + retained replay (§8).

### 6.6 Seven static modes + Lloyd + adaptive coverage

| Mode | K encoding | Default V | VMM static | Lloyd-V override | Adaptive role |
|---|---|---|---|---|---|
| `q8` | Q8_0 K | Q8_0 | yes | no (V already Q8) | V start tier only |
| `asym2` | rotated 2-bit K | Q8_0 | yes | no (asym≠fwht path) | none |
| `asym3` | rotated 3-bit K | Q8_0 | yes | no | none |
| `asym4` | rotated 4-bit K | Q8_0 | yes | no | none |
| `fwht2` | FWHT 2-bit K | Q8_0 | yes | legal lloyd2/3/4 if guards pass | optional K floor |
| `fwht3` | FWHT 3-bit K | Q8_0 | yes | legal lloyd2/3/4 | optional K floor |
| `fwht4` | FWHT 4-bit K | Q8_0 | yes | legal lloyd2/3/4 | mandatory K start; optional K floor |

Lloyd-V remains subject to existing invariants: FWHT-K, `head_dim==256`, and
any width/pairing asserts already enforced on contiguous paths. Illegal pairs
fail at load resolution, not mid-session. Adaptive always starts FWHT4/Q8;
`fwht2` and `fwht3` in this table describe possible floors, not alternate
start modes.

Adaptive presets (floors):

| Preset | K floor | V floor |
|---|---|---|
| conservative | fwht4 | lloyd4 |
| **balanced** | **fwht3** | **lloyd3** |
| aggressive | fwht2 | lloyd2 |
| advanced | user K ∈ {fwht4,fwht3,fwht2}, V ∈ {lloyd4,lloyd3,lloyd2} | |

---

## 7. Load and runtime transition flow

### 7.1 Load (single resolution pass)

```
parse CLI/config
  → resolve kv_mode, kv_v, kv_adaptive, kv_backend
  → reject PP/multi-GPU/CASK/flat-llama/unsupported combos on vmm
  → if adaptive:
       require/resolve the start encoding exactly as K=fwht4, V=q8
       compute floor reserve strides from preset/advanced
       install controller (mandatory)
  → else:
       compute static current-tier strides (incl. Lloyd-V if set)
  → allocate dual VMM arenas per real layer at resolved reserve
  → map initial prefix (policy: at least one chunk / prefill needs)
  → never call contiguous reallocators on those owners afterward
```

**Invariant:** final K/V layout, backend, and adaptive floors are known
**before** the first `hipMem*` reserve. No “allocate Q8 contiguous, then
convert to VMM / lloyd / floor.”

### 7.2 Prefill / decode

```
ensure_mapped(k, v, through=need_pos)   # fail → abort before write
write KV at current tier
attend
on committed boundary:
  maybe_downshift(...)                  # adaptive only
    → compact in place
    → invalidate graph + retained replay
    → continue
```

### 7.3 Static tier changes after load

User mid-load “switch kv mode” that would change reserve stride is out of
scope. Contiguous `set_v_mode_realloc`-style **owner replacement** is forbidden
for VMM. Adaptive downshifts are the only approved mid-session tier changes, and
they keep owners.

---

## 8. Graph and retained-replay invalidation

Any successful adaptive downshift (K or V) **must**:

1. Invalidate HipGraph capture/replay state for affected batch keys
   (`invalidate_for_kv_mode_switch` or equivalent full clear used today).
2. Invalidate **retained replay** / Redline tape state that baked KV base
   pointers, tier flags, or kernargs dependent on K/V mode.

No “graph stays live, only V kernarg changes” optimization in this design:
K kernel selection and V-mode bits both change across the ladder; retained
paths must not replay stale packets into a new logical layout.

Static VMM without tier change does **not** require per-chunk invalidation:
mapped-prefix growth keeps the same base VA (FireMap property). Growth still
preflights before touch so replay never faults on unmapped tails.

---

## 9. Reset, failure, and unload semantics

### 9.1 Reset

On KV cold reset:

- Clear logical seq/len/watermarks.
- Reset adaptive controller to start (fwht4/q8, step 0) **with** the cache.
- Do not free VMM arenas; do not require unmap of the whole reserve.
- Next prefill rewrites the live prefix at start tiers.

### 9.2 Mapping failure

- Detected in ensure-mapped / grow.
- **Abort before write or attend.**
- Error propagates to the generation/load caller.
- No partial token commit that assumed the map succeeded.

### 9.3 Partial transcode failure

- If any layer fails mid-downshift, **poison the model** until full reload.
- Do not leave a mixed-tier cache runnable.
- Scratch discipline must not present half-applied steps as success.

### 9.4 Teardown / unload

- Teardown errors **propagate** (not log-and-skip).
- Unmap + unreserve every **registered** arena; on failure, **retry** remaining
  registered arenas; still report error.
- **No clean unload / no new load** while pending arenas remain registered in
  process-global VMM ownership tables.
- Process must not advertise a successful idle unload if arenas leak.

### 9.5 Poison and engagement

- Poisoned model rejects further generate until unload+load.
- Serve/harness path proofs must show actual VMM engagement markers when
  `kv_backend=vmm` was requested (fail closed if absent).

---

## 10. CLI / harness review corrections (fold into merge)

Approved review fixes that ship with this design:

1. **Final DFlash draft projection runs after the CLI selector**  
   Speculator/draft resolution order: CLI speculation selector wins, then draft
   projection/finalization. Do not project a draft path that the selector has
   already disabled, and do not let an earlier projection override the selector.

2. **Request-level DFlash proof**  
   A “draft loaded” marker proves only load-time availability. After requests
   complete, every harness run that requires DFlash must show a request-level
   discriminator such as DFlash `tau`/cycle accounting or an exact
   per-request drafter-route marker. An all-AR run fails even when a draft was
   loaded successfully.

3. **Current-attempt log slicing**  
   Capture the serve-log offset before each spawn attempt and evaluate VMM,
   draft-load, graph, and request-route markers only after that offset. Run
   request-level assertions after requests return; prior retry attempts must
   not satisfy the current attempt.

These are merge requirements alongside the KV lifecycle rules above.

---

## 11. Validation and merge gates

No gate is optional for the claim it covers. Vocabulary follows
`docs/VALIDATION.md` (automatic vs manual; fail closed).

| Gate | Purpose |
|---|---|
| **CPU layout / capacity tests** | Every static K mode × legal V tier; reserve math; min-of-two capacity; mapped-token capacity increase after downshift; balanced floor distinct from aggressive; controller/cache reset pairing |
| **CPU failure tests** | Deterministic `hipMemMap`, `hipMemSetAccess`, `hipMemUnmap`, and `hipMemRelease` failures; map-fail-before-write; constructor rollback; teardown retry and pending-arena refusal; poison on partial transcode |
| **gfx1201 all-seven static hardware** | Each of `q8`, `asym2`, `asym3`, `asym4`, `fwht2`, `fwht3`, and `fwht4` through load, prefill, AR decode, graph capture/replay, mapping-boundary growth, unload, and reload |
| **Adaptive hardware** | Conservative/balanced/aggressive transitions during multi-chunk prefill and AR; stable K/V addresses; mapped-token capacity growth after downshift; transcode parity against direct target-tier writes; reset, unload, and reload |
| **Existing Q8 / FWHT3 DFlash** | Static VMM+DFlash request-level route proof on both already-supported encodings; adaptive DFlash remains rejected |
| **MQ4R retained-PM4 + VRAM** | Canonical Q8 TG128 route proof; matched contiguous/VMM throughput; max-context load VRAM; exact post-unload idle baseline |
| **CLI and serve regressions** | Config-off + `run --spec dflash` retains inherited draft; DFlash requires per-request evidence; all serve assertions use the current-attempt log slice |
| **Final repository gates** | Changed-file rustfmt, workspace clippy, workspace unit suite, GitNexus change-scope report, and independent adversarial rereview |

Claims from hardware evidence are **gfx1201-scoped** unless a future record
adds another arch with its own table.

---

## 12. Observed MQ4R evidence (gfx1201 only)

Recorded measurements for the matched MQ4R route. Do not generalize to other
arches, models, or KV modes without new evidence. Not an admission row.

### 12.1 Decode throughput (tok/s) and retained route proof

| Arm | tok/s | Retained route proof |
|---|---:|---|
| Golden contiguous | **200.870** | valid route |
| Matched contiguous | **200.975** | valid at positions 128/255 |
| VMM arm A | **202.544** | valid at positions 128/255 |
| VMM arm B | **201.121** | valid at positions 128/255 |

All listed retained route proofs are **valid** at positions **128** and **255**.

These figures are observational matched runs, not a product claim that “VMM is
faster.” Deltas are within normal stationary noise for this setup; the merge
bar is **no material regression + valid route proof**, not a speedup admission.

### 12.2 VRAM (Q8 KV, max_seq=32768)

| Backend | Bytes |
|---|---:|
| Contiguous | **24,591,974,400** |
| VMM | **24,256,430,080** |
| Reduction | **320 MiB** |

### 12.3 Unload

| Check | Bytes |
|---|---:|
| Exact idle after unload | **59,912,192** |

Unload must reach this exact idle footprint in the measured configuration;
pending-arena leaks fail the gate.

---

## 13. Open implementation notes (unavoidable only)

1. **HIP VMM FFI surface** stays behind the existing optional load path; no
   hard link requirement when `kv_backend=contiguous`.
2. **Chunk size policy** remains coarse page-aligned growth (FireMap planner);
   not part of the tier math except through mapping round-up.
3. **Sign-table width** for fwht2/4 vs fwht3 and lloyd-V follows existing
   adaptive/contiguous rules (256-wide signs when lloyd or fwht3 requires them).
4. **Adaptive × DFlash** deliberately open: static Q8/FWHT3 DFlash-on-VMM is in
   gate; wiring `maybe_downshift` on DFlash committed positions is a follow-up
   (non-goal: adaptive DFlash).
5. **Implementation may land in stacked commits** but must not merge with any
   §4 defect still open or any §11 gate skipped for its claim class.

---

## 14. Summary

Merge-complete PR #549 VMM is **floor-reserved, tier-aware dual arenas** for
masked single-GPU Qwen3.5/3.6 KV: seven static modes, legal Lloyd-V, and
adaptive KV with stable K/V VA owners, min-of-two capacity, in-place downshift,
strict map/transcode/teardown failure semantics, and harness proof order fixes.
PP, multi-GPU, CASK, flat-Llama carriers, and adaptive DFlash stay out.
Evidence is explicit; MQ4R numbers above are gfx1201-only and complete as
recorded.

**User-approved design: 2026-07-28.**
