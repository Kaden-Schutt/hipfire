# PP + hetero MTP combinability audit (2026-05-28)

Goal: assess whether the existing pipeline-parallel (PP=2) trunk path
can compose with the hetero MTP path shipped today (commit 0d1826fa) on
the gfx906/gfx1031 box. v1 ROI question: can we get useful ctx growth
AND keep the MTP head's spec-decode acceleration?

## TL;DR

**Yes in principle, no as-shipped today.** There is no current code path
that combines PP-trunk with MTP-anywhere — the daemon's `load_model_pp`
returns `LoadedModel { mtp: None }` unconditionally. Building the combo
needs (1) `_multi_filtered` KV-cache constructors (to stop allocating
useless full-size KV for LinearAttention layers), (2) MTP-head load on
a chosen PP device with mirror plumbing, and (3) chain dispatch that
knows the MTP head lives elsewhere. Estimated ~600 LOC for v1. ROI
depends on whether you want the freed gfx906 VRAM (~800 MB MTP) for
longer context, or the layer offload (a few GB) for larger ctx, or both.

## Q1+Q3: VRAM accounting and FA/LA distribution

**qwen3.6-27b layer structure (verified by demo log):**
- 64 layers total, 16 FullAttention + 48 LinearAttention
- FA indices: 3, 7, 11, 15, 19, ..., 63 (every 4th, offset +3)
- Per-FA-layer Q8 KV cost: 2176 bytes/token
- Per-LA-layer KV cost: 0 bytes (semantically) — the layer doesn't use KV
- **Per-LA-layer KV cost in TODAY'S code: SAME as FA** (`alloc_kv_per_layer_multi` allocs full size for all 64) — see "Critical sub-finding" below

**gfx1031 fixed cost (one-time):**
- MTP head (weights + scratch + KV) ~800 MB
- Mirrored trunk.token_embd (~1.29 GB measured)
- Headroom (JIT cache, allocator, command buffers) ~1 GB
- Verify scratch (only if FA layers land on gfx1031) ~500 MB
- **Subtotal: ~3.6 GB**

**Layer budget (gfx1031, 12 GB total, Q8 KV at 128k ctx):**

| layers_on_1031 | FA_on_1031 | weights | KV (filtered_multi) | total (filt) | total (UNFILT today) |
| ---:           | ---:       | ---:    | ---:                | ---:         | ---:                 |
| 4              | 1          | 0.88    | 0.27                | 4.73         | 5.53                 |
| 8              | 2          | 1.75    | 0.53                | 5.87         | 7.46                 |
| 12             | 3          | 2.62    | 0.80                | 7.01         | 9.40                 |
| 16             | 4          | 3.50    | 1.06                | 8.15         | 11.34                |
| 20             | 5          | 4.38    | 1.33                | 9.29         | 13.28 ❌             |
| 24             | 6          | 5.25    | 1.59                | 10.43        | 15.21 ❌             |
| 32             | 8          | 7.00    | 2.12                | 12.71 ❌     | 19.09 ❌             |

Today's `_multi` allocates ~272 MB/layer of KV (Q8 @ 128k) for ALL 64
layers, including the 48 LA layers that never use it. With the
unfiltered allocator, the practical limit on gfx1031 is **~16 layers
(25% of trunk) at 128k ctx**, and only **~20 layers at 64k ctx**. With
a filtered_multi (needs to be built), **24 layers (~37.5% of trunk) at
128k ctx fits**, and **32 layers (50% of trunk) at 32k-64k ctx fits**.

### Critical sub-finding

**`KvCache::new_gpu_*_multi` has NO `_filtered` variant.** The
single-gpu constructors (`new_gpu_q8_filtered`, `new_gpu_asym3_capped_filtered`,
etc.) save multi-GB on hybrid models by allocating ~4-byte placeholders
for LA layers. The 20 `_multi` constructors at lines 4148-4481 of
`llama.rs` all use `alloc_kv_per_layer_multi`, which allocates
full-size for every layer. For qwen3.6-27b with 48 LA + 16 FA layers,
this is **3× more KV VRAM than needed**.

Adding `_multi_filtered` is straightforward (mirror the single-gpu
filter pattern into a new `alloc_kv_per_layer_filtered_multi` helper).
Estimated ~150 LOC for the full constructor set. Independent value to
PP itself (not just the PP+MTP combo) — any PP user of a hybrid-attn
model is wasting ~3× KV VRAM today.

## Q2: Daemon path-existence check

`daemon.rs` has these orthogonal load paths:

- `load_model` (pp=1): supports MTP via `--mtp-head` → `LoadedModel.mtp = Some(...)`. **Trunk on dev 0 only.**
- `load_model_pp` (pp>1): supports `HIPFIRE_PP_LAYERS=a,b,...` for asymmetric splits via `Gpus::init_layers`. **Returns `LoadedModel { mtp: None }`** at line 2471 unconditionally — there is no codepath that loads MTP under PP.

The `generate_mtp` dispatch path that handles MTP spec-decode requires
`LoadedModel.mtp = Some(...)`. So today, even if you start the daemon
with `pp=2 mtp-head=...`, MTP is silently dropped. (Probably no
diagnostic at the moment either — that should be a one-line fix
regardless.)

**ROCm 6.4.3 peer-access ordering is respected:** `load_model_pp`
calls `gpus.enable_peer_all()` AFTER all major allocations (line 2444),
matching the gotcha note from our hetero work. Combining MTP load on a
specific device into this path needs to slot in BEFORE the
`enable_peer_all` call.

## Q4: Cycle-time interaction

PP boundary copies per cycle (qwen3.6-27b, 5 verify tokens):
- residual stream (5 × dim × f32 = 5 × 5120 × 4 = 100 KB) per band crossing
- one band crossing if pp=2

So PP verify adds ~one 100 KB peer copy per cycle. MTP per-cycle peer
copy is 20 KB. Both fit on the same PCIe link sequentially with
negligible additional latency vs. PP alone.

**No bad-interaction stacking.** PP boundary copy at end-of-verify (or
mid-verify per pp layout) is on the trunk path. Hetero-MTP peer copy is
at end-of-cycle. Order: trunk verify (with PP boundaries) → MTP
end-of-cycle peer copy → next cycle's chain on drafter. Streams are
already device-affine; events keep ordering correct.

If MTP head is co-located with a PP band on gfx1031 (likely, since
that's the smaller card), the MTP chain runs serially after the trunk
finishes its part of the verify on that band. No race.

## Layout options for the combined system

### Option A: MTP on gfx1031, NO trunk layers on gfx1031 (pp=1 trunk)

= what we ship today. Gives us back ~800 MB of gfx906 (the MTP head)
at -10% tok/s cost. Doesn't grow ctx.

### Option B: MTP on gfx1031 + trunk PP=2 (gfx906 wide, gfx1031 narrow)

Layers split e.g. 56/8 or 48/16. gfx906 hosts most of the trunk + verify
hot path; gfx1031 hosts a thin band + MTP head + token_embd mirror.

Math (16 layers on gfx1031 at 128k ctx, today's unfiltered _multi):
gfx906 has 48 layers × ~219 MB = ~10.5 GB trunk weights + 12 FA × 272 MB
KV @ 128k = ~3.3 GB. Total gfx906: ~14 GB out of 32 GB → 18 GB headroom
for prefill scratch / hipgraph cache / etc. Plenty.

gfx1031: 3.5 GB weights + 1.06 GB KV + 3.6 GB fixed (mtp + mirror +
headroom) = 8.15 GB out of 12 GB → 3.8 GB headroom. **Fits at 128k ctx
with filtered_multi, fits at 64k with today's unfiltered_multi.**

Wins:
- Frees up ~14.5 GB on gfx906 (10.5 GB shifted + 800 MB MTP + scratch)
  for longer context support, larger DFlash drafters, etc.
- Keeps MTP spec-decode active.
- ~80% of the trunk verify runs on the fast card.

Cost:
- 1 extra PP boundary copy per cycle (100 KB)
- 1 MTP peer copy per cycle (20 KB)
- Some cross-arch dispatch overhead at the band boundary

Tok/s estimate: PP=2 alone on this hardware would add ~5-10% wall (one
band crossing at 100 KB cost is small but real); MTP-on-drafter we
measured at -10%. Combined: probably -15-20% vs single-gpu-MTP. The
question is whether the freed VRAM is worth that.

### Option C: pp=2 with MTP on the WIDE card (gfx906)

This is "PP for ctx growth, MTP stays on the fast card." Doesn't gain
us anything beyond plain PP — we lose the MTP-frees-gfx906-VRAM
property. Only interesting if there's some reason the MTP head MUST
share locality with most trunk layers, which there isn't.

## User-direction (2026-05-28 follow-up)

User confirmed: **longer ctx is the goal**, alongside any real perf
gain from PP itself. Asked whether co-locating MTP head with the band
that holds the early layers (or the output band) can eliminate any
syncs.

### Where co-location actually helps

Walking the data path for PP=2 (Variant 2 layout, output_norm + lm_head
on `gpus.output_device` = the LAST device, per `qwen35.rs:9871-9873`):

```
[dev 0] embed_lookup(token_embd lives here)
  ↓
[dev 0] layers 0..k
  ↓ boundary_copy: residual stream 5×dim×f32 = 100 KB
[dev 1] layers k..N
  ↓ same device
[dev 1] output_norm + lm_head   ← output_device by convention
  ↓ verify_hidden, verify_logits both allocated on dev 1
[host] argmax → candidates
[? gpu] MTP chain reads prev_hidden, embeds, runs head, produces next draft
```

The hetero-MTP impl shipped today (`spec_step_mtp_compressed_serial_hetero`)
peer-copies `state.verify_hidden[advance-1]` from the trunk gpu to the
drafter gpu's `prev_hidden` at cycle exit (line 2417 of mtp_spec.rs).
In PP, that trunk gpu IS `gpus.output_device`.

**If MTP head is co-located on `gpus.output_device` (the last device),
the per-cycle prev_hidden peer copy collapses to a same-device
`memcpy_dtod_at`** — exactly what the original single-gpu
`capture_prev_hidden_from_verify_row` does. Saves the 112 µs/cycle the
microbench measured for the cross-device handoff.

### Where co-location does NOT help

- **Boundary copy for trunk verify is unaffected.** Residual stream
  flows dev_0→dev_last regardless of which gpu hosts MTP; MTP isn't on
  the trunk's hot path.
- **First-band locality has no advantage.** PP work per band is
  ~symmetric; there's no first-vs-last asymmetry that benefits from
  putting MTP near the embedding side. Putting MTP near `dev 0` would
  *re-introduce* the per-cycle peer copy for prev_hidden (now flowing
  output_device→dev_0) and cost more than it saves.
- **token_embd mirror is still needed.** `token_embd` lives on
  `gpus.devices[0]` by convention; MTP chain reads it every step. The
  cheap fix is the same one-shot ~1.29 GB peer-clone we already do, NOT
  a per-step copy. Total cost: 1.29 GB drafter VRAM, paid once at session
  init (~200 ms peer-DMA, validated by mirror smoke).
- **Async overlap potential.** With MTP on a SEPARATE card from
  output_device, the MTP chain could in principle overlap with the
  trunk's next verify. With MTP co-located on output_device they
  serialize. We don't exploit async overlap today (the cycle is all
  sync), so this is a "future loss" not a "current loss" — and the
  microbench has already shown the sync handoff is cheap. Net: pick
  co-location, revisit async-overlap only if perf demands it.

### Verdict: MTP head on `output_device` (= last PP device = gfx1031)

The cleanest design is the one that already falls out of existing PP
conventions:

- gfx906 (32 GB) = dev 0: bands 0..k of trunk + token_embd. Most of the
  trunk-weight bulk, most of the KV cache.
- gfx1031 (12 GB) = dev 1 = `output_device`: bands k..N of trunk +
  output_norm + lm_head + **MTP head + token_embd peer-mirror**.

Saves the 38-112 µs/cycle prev_hidden handoff. Slots into the existing
PP code with NO layout changes (MTP gets loaded onto
`gpus.devices[gpus.output_device]`, naturally co-located with verify
output).

## Recommendation (UPDATED)

**Build in two stages:**

**Stage 1 (cheap, ~150 LOC, independent value): add `_multi_filtered`
KV constructors.** This is a pure PP improvement — any current PP user
of qwen3.5/3.6 hybrid models is wasting 3× KV. No new architecture
required, just mirror the existing single-gpu `_filtered` pattern into
the multi path. Ships on its own. ROI **today** = unblocks longer-ctx
PP without the MTP combo. Validated by the table above: 24 layers on
gfx1031 at 128k jumps from impossible (15.2 GB ❌) to comfortable
(10.4 GB ✓) with the filter.

**Stage 2 (~450 LOC, needs Stage 1): PP-trunk + MTP-on-output_device.**

- Extend `load_model_pp` to accept an `mtp_head_path` and load it
  onto `gpus.devices[gpus.output_device]`. NO env override needed in
  v1 — the output_device convention is the right placement.
- Mirror `trunk.weights.token_embd` from dev 0 (where the trunk loader
  put it) to output_device using `mtp_mirror::peer_clone_tensor`,
  same primitive shipped today.
- Reuse the `MtpHeteroDrafterState` struct verbatim — it already takes
  drafter_gpu as a `&mut Gpu`, doesn't care it's a PP band.
- Trunk-side `MtpSpecState` (verify_hidden, verify_logits, trunk_snap,
  etc.) gets allocated on `output_device` since that's where the
  trunk verify writes them — natural fit, no logic change.
- Per-cycle handoff: when drafter_gpu == output_device, use the existing
  single-gpu `state.capture_prev_hidden_from_verify_row` (same-device
  D2D memcpy). When drafter_gpu != output_device, use the
  `memcpy_peer_offset` path we shipped today. **One unified
  spec_step function gets both behaviors via a same-device branch.**
- Daemon serve wiring: extend `generate_mtp` dispatch to handle the
  `pp_gpus.is_some() && mtp.is_some()` case. Trunk verify routes
  through the existing pp forward; MTP chain routes through the hetero
  spec function with drafter_gpu pointed at output_device.

**Don't build Stage 2 without Stage 1** — without filtered_multi, the
gfx1031 budget caps at 25% of trunk layers at 128k ctx, which doesn't
free enough on gfx906 to materially raise the ctx cap.

## Projected outcomes

With Stage 1 + Stage 2 deployed, layout: 40 layers on gfx906 + 24 on
gfx1031 (37.5% offload), filtered_multi for KV, MTP head on
output_device = gfx1031:

| metric                | today (pp=1 + hetero-MTP) | Stage 1+2 (pp=2 + MTP-on-out_dev) |
| ---                   | ---                       | ---                                |
| gfx906 VRAM used      | ~14 GB (full trunk - 0.8 GB freed) | ~6 GB (40/64 layers + their KV) |
| gfx906 VRAM free      | ~18 GB                    | ~26 GB                             |
| gfx1031 VRAM used     | ~2.1 GB (head + mirror)   | ~10.4 GB (24/64 layers + KV + head + mirror) |
| gfx1031 VRAM free     | ~10 GB (mostly idle)      | ~1.6 GB                            |
| ctx ceiling (current) | ~120k (per existing serve)| ~256k+ (gfx906 has 26 GB free)     |
| per-cycle MTP overhead| 112 µs cross-device       | ~0 µs (same-device on out_dev)     |
| trunk verify overhead | 0 (single-gpu)            | 1 PP boundary copy ~100 KB/cycle   |
| projected tok/s vs today | baseline               | maybe -5% from PP boundary; +0% from MTP handoff being free; net **~-5%** |

Notes:
- Stage 1 alone (PP without MTP combo) already unblocks ctx growth
  for the existing pp=2 serve path. Land it first regardless.
- The "ctx ceiling" estimate above is the GPU-VRAM ceiling; actual
  serving ctx may still be capped by the existing `physical_cap` /
  rope_theta / etc. constraints. Worth validating once Stage 1 lands.
- Stage 2's "~-5%" estimate assumes PP boundary copy cost dominates;
  microbenching the actual PP=2 cost on this hardware pair should be
  the FIRST validation step after Stage 1 lands (don't build Stage 2
  if PP boundary cost on this hardware turns out to be much worse than
  predicted).

## Tasks summary

- [x] Audit (this doc, updated)
- [ ] Stage 1: `_multi_filtered` KV constructors (~150 LOC, ships on its own)
- [ ] Microbench PP=2 boundary cost on gfx906↔gfx1031 (gate before Stage 2)
- [ ] Stage 2: PP+MTP combo with MTP-on-output_device (~450 LOC)
- [ ] Validation: 27B at 256k ctx serving via combined path
