# Goal: medgemma vision pipeline — cache it, serve it, compress the KV

Status: **active** — started 2026-06-20 (chaingun). Follows the vision-encode
perf work (138s → ~1.5s/image, ~92×, committed: `vit_attention_opt` fix → bf16
WMMA tower → f16-KV WMMA flash). Builds on `2026-06-19-medgemma-vision-bringup.md`.

## Context

The SigLIP vision encode is **done**: bf16 WMMA GEMM + f16-KV WMMA flash attention
took it from 138s to ~1.5s/image on gfx1151. Profiling + the AMD matrix calculator
+ the FSR4 RDNA3 study all agree the pipeline is **bandwidth-bound on unified
memory**, and the empirical finding is that further GPU shaving of the encode
(MQ8 weights, INT8 vision KV, GPU avg-pool) is ≤~0.3s of diminishing returns.

So the leverage has moved **out of the encode kernels**. The three goals below are
the real remaining wins, in priority order. They share one strategic spine:
**INT8 + variance-normalization is the precision representation** — it halves
bandwidth on the bandwidth-bound GPU, holds quality at 8-bit, and is the NPU's
native idiom, so the GPU win and the eventual NPU port are the same work.

**Out of scope here (deferred): the NPU vision backend.** It's the architectural
endgame, but it's large and gated on the INT8 spine these goals establish. Tracked
separately; do not start it under this goal.

---

## Goal 1 — Vision embedding cache (xxh64-keyed, on-disk LRU)

**Why:** the encode is the dominant per-request cost, and video makes it K× (K
frames = K encodes). The same image/frame recurs constantly (re-runs, repeated
frames across a clip, multi-turn). Caching the *projected* rows skips the tower
entirely on a hit — beats any further encode-kernel shaving.

**What:**
- **Key:** `xxh64` of the **submitted image bytes** (pre-decode), namespaced by
  vision-config / arch identity so embeddings never alias across models/towers.
- **Value:** the post-projector rows (`mm_tokens × text_hidden` f32, e.g.
  256×2560) — a hit bypasses SigLIP + projector + the host download.
- **Store:** one on-disk file/dir, **configurable max size + LRU eviction**,
  persistent across daemon restarts; per-entry mmap/pread (loader already prefers
  pread on UMA APUs).
- **Path:** daemon hashes on submission → probe → splice cached rows on hit;
  encode + insert (respecting the cap) on miss.

**Open:** cache pre- vs post-projector features; `HIPFIRE_VISION_CACHE_*`
env/CLI surface (path + max size). See `TODO.md` "Vision embedding cache".

**Done when:** a repeat image/frame submission skips the encode (verified by
timing + a cache-hit counter); LRU eviction holds the file under its cap across
restarts; coherence unchanged on hit vs miss.

**Progress (2026-06-20):** standalone lib crate `crates/hipfire-vision-cache`
landed — GPU-free, one dep (`twox-hash`/xxhash64). 128-bit `(ns_hash, img_hash)`
key (namespace = vision-config/arch identity, so towers never alias); value =
`CachedEmbedding{n_rows,n_cols,data:Vec<f32>}`. On-disk: one `.vrow` payload file
per entry (header + checksummed f32-LE payload, atomic tmp+rename, per-entry read)
+ a binary `manifest` (atomic replace; rebuilt by scanning `.vrow` headers if
lost). Approximate-LRU (recency bumped in memory on `get`, flushed on
insert/evict); evict never drops the just-inserted entry or below one entry.
10 no-GPU unit tests: key determinism/namespace-scoping, byte-exact round-trip,
**hit==miss equality** (Goal-4 evidence for Goal 1), eviction-holds-budget,
LRU-recency, persist-across-reopen, manifest-rebuild, corruption→miss, replace.
**Wired into the daemon under Goal 2b** (`3e2d6e06`); hit==miss verified across a
real encode on gfx1151 (2 daemon sessions, byte-identical output). The
deterministic hit==miss unit test remains as the no-GPU stand-in.

## Goal 2 — Daemon arch-13 serving + video protocol (the gate) *(= original Phase 3)*

**Why:** none of the vision work is reachable in production until medgemma serves
through the daemon. This is **greenfield** — neither gemma3 (arch 12) nor
gemma3-vl (arch 13) is wired into the daemon's generate dispatch today; the
`ServingBackend::serve` trait impl exists but is never invoked, and the model
loader has no arch-13 branch.

**What:**
- **Protocol** (`crates/hipfire-daemon/src/main.rs` ~3054/3302): accept a
  `"video"` path and/or `"max_frames"` alongside `image`/`image_base64`; when a
  video (or an `image` path that `hipfire_media::is_video`) is given, decode to
  owned frames before dispatch. Update the protocol doc-comment at `main.rs:20`.
- **Model load:** add an `arch_id == 13` branch to `crates/hipfire-serving-core/
  src/load.rs` that calls `hipfire_arch_gemma3_vl::load_vl(...)`, stores the
  bundle on `LoadedModel` (new field, e.g. `Option<Gemma3VlBackend>`; ~10
  construction sites get a `None`), and sets `vision_config` so the `has_vl` gate
  (`main.rs:3304`) is true for arch 13.
- **Serve route:** add `generate_vl_gemma3()` in
  `crates/hipfire-serving-core/src/generate_vl.rs` mirroring `generate_vl` /
  `generate_vl_dots_ocr`, taking decoded frames; route `arch_id == 13` from the
  daemon VL branch (`main.rs:3353`). `serve()`→`decode_loop` already emits the
  daemon's exact `{"type":"token"/"done"}` schema, so it can be reused internally.
  Extend `GenerateVLParams`/`ImageSource` (`crates/hipfire-generate/src/lib.rs`)
  to carry multiple frames.

**Done when:** `hipfire serve` loads medgemma (arch 13) and answers a `generate`
request with `image`/`images[]`/`video` over the JSONL protocol; multi-image and
video both stream coherent output; integrates the Goal-1 cache on the hash path.

**DONE (2026-06-20, validated on gfx1151 Strix Halo + medgemma-1.5-4b).**
- **2a** (`4ff4945f`): `LoadedModel.gemma3_vl` + arch-13 load branch + daemon
  dispatch (`decode_vl_frames` for image/video/base64) + `video`/`max_frames`
  protocol. Routes to `generate_vl_gemma3` → `Gemma3VlBackend::serve`.
- **coherence** (`41fe57eb`): `decode_loop` gained a repeat-penalty token pick
  (greedy unchanged at penalty ≤ 1.0); daemon defaults arch-13 to 1.3; `images[]`
  multi-image input. Bare greedy attractored on near-duplicate video slices
  (`ình` wall) — penalty fixes it.
- **2b** (`3e2d6e06`): Goal-1 cache wired on the encode path — `encode_image`
  (pub) + `serve_with_embeds` split; per-frame xxh64 probe namespaced by
  `model_path|gemma3vl|img|patch|mm|th`; `HIPFIRE_VISION_CACHE_*` env surface.
- **reporting**: `loaded` event now reports `arch=gemma3_vl, dim=2560,
  layers=34, vocab=262208, vl=true` for arch 13 (was `qwen3/0/0/0/false`).

**E2E evidence (all greedy+1.3 penalty, daemon JSONL):**
- single image → coherent, anatomy-aware (cerebrum/cerebellum/brainstem,
  gray/white matter, ventricles).
- video (3 MRI slices) → decode→encode→splice→coherent medical summary.
- `images[]` (2 distinct MRI) → coherent comparison ("same structure,
  different contrast … left side shows ventricles …").
- cache: 2 fresh daemon sessions, shared dir → session 1 `misses=1`, session 2
  `hits=1`, **byte-identical output (hit == miss)**; `.vrow` = 48B header +
  256×2560×4 f32, persists across restart.

**Goal 4 — DONE (2026-06-20, `943ddfed`).** The `hipfire-eval` `Vision` battery
(previously a stub) now drives the gemma3-vl path through the daemon executor and
PASSES on gfx1151 with medgemma-1.5-4b:
- **`describe_image`** — loads the model, sends a committed MRI fixture
  (`benchmarks/vision/images/mri_human_brain.jpg`) via `image_base64` + a
  byte-identical prompt (`benchmarks/prompts/vision_describe_image.txt`), asserts
  finite + non-degenerate (unique_word_ratio ≥ 0.30, max_word_freq ≤ 0.50;
  measured 0.89 / 0.054).
- **`cache_hit_determinism`** — reset + re-run the same image (vision-cache hit on
  the 2nd pass); asserts byte-identical output — the in-harness hit==miss guard
  (measured byte_identical=true).
- Gated on `arch == "gemma3_vl"` (skip otherwise); image sent through the existing
  `GenerateTextRequest.image_base64` (no wire-protocol extension); 3 no-GPU unit
  tests for the coherence stat. Run:
  `hipfire-eval --battery vision --no-cache --executor daemon --model <medgemma>.hfq`.
- **Gotcha:** `find_daemon_bin` prefers a stale `~/.hipfire/bin` install over
  `target/release` — set `HIPFIRE_DAEMON_BIN` or `hipfire install` for a fresh
  daemon (affects all daemon-executor batteries, not just Vision).

All four goals (1, 2, 4) are complete and GPU-validated; **Goal 3 (KVarN-8) is
being taken independently by the user.**

## Goal 3 — 8-bit variance-normalized KV ("KVarN-8")

**Why:** decode-side bandwidth. A multi-image/video request builds a *long*
prefill (K×256 image tokens + text); the long-context decode KV cache is the
bandwidth bottleneck there. 8-bit KV halves the cache bandwidth and is
**near-lossless** (256 levels for bounded post-norm KV), so the variance-norm is a
quality top-up rather than the survival requirement it is at 4-bit. And **INT8 is
NPU-native** — this is the format that ports.

**What (8-bit adaptation of NEXT-STEPS.md Phase D / `KVARN_MLA_BACKEND_SPEC`):**
- Start from the existing `q8` KV mode as the floor; add the variance-normalization
  step (per-channel scale/zp + per-token row scale; Sinkhorn balance over a tile).
  At 8-bit a lighter normalization than the 4-bit Sinkhorn may suffice — measure.
- **Gate first (cheap, before kernels):** a CPU reconstruction gate on REAL
  Qwen3.5/gemma3 KV activations (capture via a hook) comparing varnorm-INT8 vs
  plain `q8` — confirm the quality margin justifies the work at 8-bit (it may be
  small; that's a valid finding). Reuse `kvarn::variance_normalize` /
  `kvarn::quantize_tile` references.
- If the gate passes: GPU staged-write + normalize + INT8-pack + dequant-on-read
  (→ fp16 scratch → stock flash), as a `"kvarn8"` KvCache mode. GQA-shaped (Qwen3.5
  FullAttention), not the MLA-latent reference layout.

**Done when:** the CPU gate quantifies varnorm-INT8 vs naïve-q8 on real KV; if it
clears the bar, the `kvarn8` mode halves decode KV bandwidth with coherence intact
(`tests/coherence-gate-dflash.sh`) and a measured long-context decode tok/s win.

## Goal 4 — Evidence: evals, gates, docs *(= original Phase 4; cross-cutting)*

**Why:** per AGENTS.md, **model/runtime evidence belongs in `hipfire-eval`**, and
every kernel/forward/KV change must pass the coherence gates. The encode + multi-
image work landed with unit tests + parity guards but **no `hipfire-eval`
battery** and the bring-up doc's follow-up list isn't reconciled. This goal makes
the work verifiable and keeps it that way as Goals 1–3 land.

**What:**
- **`hipfire-eval` battery for the medgemma vision path** — multi-image + video
  (decode → encode → splice → describe), asserting coherence/non-degeneracy on a
  fixed prompt + the `MRI_BRAIN` fixtures (byte-identical prompts via
  `benchmarks/prompts/*.txt`). This is the canonical evidence home; shell gates
  stay as enforcement wrappers only.
- **Already landed (keep green):** `hipfire-media` unit tests, gemma3-vl
  multi-image splice tests, the `vit_attn_parity` / `flash_bf16_parity` /
  `bench_siglip_attn` kernel guards. Add a parity guard for the daemon serve path
  (Goal 2) and the cache hit==miss equality (Goal 1).
- **Gates:** `tests/no-gpu-ci.sh` for workflow-only changes;
  `tests/coherence-gate-dflash.sh` for any KV/forward change (Goal 3 especially);
  the pre-commit speed+coherence gate already runs per commit.
- **Docs:** reconcile `docs/plans/2026-06-19-medgemma-vision-bringup.md` follow-ups
  (mark daemon wiring + video/multi-image done; note the encode perf trajectory);
  update `TODO.md` (embedding cache → in-progress when Goal 1 starts); regen
  `docs/CLI.md` if the daemon protocol/CLI surface changes (Goal 2).
- **Per-arch re-validation:** the ~92× encode is **gfx1151-measured**; record a
  cross-arch perf check on gfx1100 (k9lin) / gfx1201 (hiptrx) as evidence (FSR4:
  RDNA3 optimizations don't transfer 1:1).

**Done when:** the `hipfire-eval` battery covers the medgemma vision path and runs
in CI; the no-gpu subset is green; coherence gates pass for every landed goal; the
bring-up doc + `TODO.md` reflect reality; a cross-arch perf number is on record.

---

## Sequencing & cross-cutting

1. **Goal 1 (cache)** — highest workflow leverage, self-contained, no daemon
   dependency for the core lib (wire into the daemon under Goal 2).
2. **Goal 2 (daemon, = Phase 3)** — the gate; unblocks production use of everything.
3. **Goal 3 (KV)** — decode-side; independent of 1/2, can proceed in parallel,
   but its value shows most once long multi-image contexts are served (Goal 2).
4. **Goal 4 (evals/gates/docs, = Phase 4)** — cross-cutting; each of Goals 1–3
   lands its evidence here (eval battery row + gate run + doc reconcile), not as
   an afterthought.

**Precision spine (applies across all three):** prefer INT8 + variance-norm
wherever bytes move (cache values can stay f32 — small; KV → INT8). Keep f32
accumulation (measured free on RDNA3). This keeps the GPU work and the future NPU
port one and the same.

**Verification:** per-goal "done when" above; `tests/no-gpu-ci.sh` for
workflow-only changes; `tests/coherence-gate-dflash.sh` for any KV/forward change;
re-validate perf claims per-arch (the ~92× is gfx1151-measured — FSR4 shows RDNA3
optimizations don't transfer 1:1 to gfx1100/gfx1201). Commit per milestone (Rule 2).
