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

## Recommendation

**Build Option B in two stages:**

**Stage 1 (cheap, ~150 LOC, independent value): add `_multi_filtered`
KV constructors.** This is a pure PP improvement — any current PP user
of qwen3.5/3.6 hybrid models is wasting 3× KV. No new architecture
required, just mirror the existing single-gpu `_filtered` pattern into
the multi path. Likely worth landing on its own regardless of the MTP
combo decision.

**Stage 2 (~450 LOC, needs Stage 1): combine PP-trunk with MTP-on-drafter.**

- Extend `load_model_pp` to accept an `mtp_head_path` and load it
  onto `gpus.devices[mtp_device]` (an env-selectable index, default
  the last device).
- Reuse the `MtpHeteroDrafterState` already shipped. The trunk-side
  half of MtpSpecState gets allocated on the chosen "verify primary"
  device (the device that owns the lm_head output norm).
- Per-cycle handoff: same row-correct peer copy we already do for
  hetero MTP, but with the source device being whichever PP device
  owns the output norm (not necessarily dev 0).
- Daemon serve wiring: rebuild the `generate_mtp` dispatch path to
  handle the `pp_gpus.is_some() && mtp.is_some()` case.

**Don't build Stage 2 without Stage 1** — without filtered_multi, the
gfx1031 budget is tight enough that the configuration won't fit useful
contexts.

## Open question for the user

Whether to build this depends on what we'd USE the freed VRAM for. The
hetero-MTP ship today freed ~800 MB on gfx906. Combined with Option B,
we'd shift another ~10 GB to gfx1031. **What's the gfx906 VRAM
target?** Specifically:

1. Run 27B at much longer ctx (we currently cap at ~120k → would 256k
   be useful, or do we already saturate the model's positional
   encoding)?
2. Co-resident DFlash drafter for hybrid spec-decode at low-temp
   sampling cases?
3. PFlash drafter co-resident with full-ctx 27B?
4. Just headroom (no specific destination yet)?

If the answer is (4), Option B is speculative-ROI work. If any of
(1)-(3) is a real near-term need, Option B + Stage 1 is well-justified.

## Tasks summary

- [x] Audit (this doc)
- [ ] Decision on Option B build vs. shelve
- [ ] If build: Stage 1 — `_multi_filtered` KV constructors
- [ ] If build: Stage 2 — PP+MTP combo in daemon + spec function
- [ ] If shelve: keep hetero-MTP as the stable v1 deliverable
