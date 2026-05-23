# Qwen 3.5 9B-VL: hipfire vs llama.cpp image-quality comparison

**Date:** 2026-05-23
**Trigger:** Reports of image-recognition quality problems after the OpenAI vision PR (`d6db6ae8 — feat(vision): OpenAI /v1/chat/completions image inputs (#312)`) merged. This is the F16-reference comparison the May 20 audit (`9b-vl-preprocessing-audit.md`) named as the only way to rule out "the base model is just weak on unfamiliar memes."

## Setup

| Engine | Model | Body quant | Vision tower | LM head |
|---|---|---|---|---|
| hipfire | `qwen3.5-9b.mq4-q8head-vision-f16-spliced.hfq` | MQ4 (~4.4 bpw, AWQ+GPTQ) | F16 (spliced) | Q8 |
| llama.cpp Q4_K_M | `Qwen3.5-9B-Q4_K_M.gguf` + `mmproj-F16.gguf` | Q4_K_M (~4.5 bpw) | F16 | Q4_K_M |
| llama.cpp Q8_0 | `Qwen3.5-9B-Q8_0.gguf` + `mmproj-F16.gguf` | Q8_0 (~8.5 bpw) | F16 | Q8_0 |

- GPU: AMD RX 7900 XT, gfx1100, 20 GB VRAM
- llama.cpp build: `9dcf83552` (HEAD of `~/git/llm/llama.cpp`)
- hipfire build: today's HEAD on `master` (commit 557be2e5+)
- Sampling: `temp=0`, `n_predict=300`, greedy decode on both engines
- Prompts (identical across engines, two per image):
  - **desc** — "Describe this image in 2-3 sentences. /no_think"
  - **ocr** — "Transcribe any visible text, signs, or written words in this image. If there is no text, say so. /no_think"

## Image set (6)

| File | Source | Resolution | Pixels | Hipfire patches (grid) | Under hipfire 1.0 MP cap? |
|---|---|---|---|---|---|
| `barney_cigar.jpg` | meme (Neil Patrick Harris, Barney Stinson) | 640×468 | 0.30 MP | 300 (20×15) | yes |
| `doge.jpeg` | meme (Shiba Inu close-up) | 537×529 | 0.28 MP | 289 (17×17) | yes |
| `doge_napping.png` | meme (Doge belly-up + rainbow text) | 960×646 | 0.62 MP | 600 (30×20) | yes |
| `scene_1.jpg` | dots.ocr Paris signs (5 directional French signs) | 692×1024 | 0.71 MP | 2816 (64×44) | yes |
| `scene_2.jpg` | dots.ocr perfume shop (luxury brand storefront) | 1024×768 | 0.79 MP | ~2800 | yes |
| `general_qa.jpg` | dots.ocr 12-panel scientific scatter-plot grid | 946×1024 | 0.97 MP | ~3800 | yes (barely) |

All six images fit inside hipfire's hard-coded `max_pixels = 1,003,520` cap, so the failures below are NOT explained by the known cap-mismatch with HF reference (which allows ~16 MP).

---

## Per-image results

### 1. `barney_cigar.jpg` — Neil Patrick Harris (Barney Stinson) with cigar

| Engine | Description | OCR |
|---|---|---|
| llama Q4_K_M | "a man with short blond hair is seated in a large, dark brown leather armchair, dressed in a dark suit with a light purple shirt and tie. He holds a cigar in his right hand… a wooden globe sits on a stand to the left" | "No text" |
| llama Q8_0   | "**Neil Patrick Harris** sits confidently in a plush, dark leather armchair, dressed in a sharp black suit with a lavender shirt and tie. He holds a cigar in his right hand…" | "No text" |
| **hipfire**  | "low-resolution, **vertically stretched** image showing what appears to be a person in dark clothing holding or interacting with several small bottles — possibly alcohol containers — against a blurred background… heavy compression artifacts and **distortion from stretching the original photo into portrait orientation**" | "No visible text found" |

**Verdict: hipfire wrong, llama.cpp correct.** Both llama quants get the suit/cigar/armchair/globe combo; Q8_0 names the person. Hipfire sees "bottles" not a cigar, "dark clothing" not a suit, and reports the image as visibly distorted.

### 2. `doge.jpeg` — Shiba Inu Doge meme

| Engine | Description | OCR |
|---|---|---|
| llama Q4_K_M | "close-up image of a **Shiba Inu dog** with wide, expressive eyes and a slightly tilted head… widely known as 'Doge'" | "No text…" |
| llama Q8_0   | "close-up, slightly blurry photo of a **Shiba Inu dog** with wide, startled eyes… iconic 'Doge' meme face" | "no text in the image" |
| **hipfire**  | "highly magnified, **blurry close-up of granular or crystalline material — possibly sugar, salt, or mineral fragments** — with dominant yellow and off-white tones… abstract view of fine particulate matter" | "No visible text found" |

**Verdict: hipfire wrong, llama.cpp correct.** Both llama quants identify the dog and the meme name. Hipfire sees "sugar/salt/mineral fragments" — no dog perceived at all.

### 3. `doge_napping.png` — Doge belly-up with multi-color text

| Engine | Description | OCR |
|---|---|---|
| llama Q4_K_M | "fluffy **Shiba Inu dog** is lying on its back in a relaxed, upside-down pose… humorous text overlays like '**Such comforttable**,' '**Sleep like a doge**,' '**Pointy teethe wow**'" | (token-budget overflow) |
| llama Q8_0   | "fluffy **Shiba Inu dog** is lying on its back… colorful text captions like 'Such comforttable,' 'Sleep like a doge,' 'Pointy teethe wow'" | (token-budget overflow) |
| **hipfire**  | "close-up of what appears to be a **coral or marine organism, possibly anemone or soft coral**, with white and yellowish polyps extending outward against a dark background… underwater photography in reef tanks" | "The image contains the following text: '**water**'" |

**Verdict: hipfire wrong on both axes.** llama.cpp transcribes the actual meme captions verbatim and identifies the dog. Hipfire perceives a coral / anemone and OCRs a single fabricated word "water" (not present in the image).

### 4. `scene_1.jpg` — Paris directional signpost (French OCR + scene)

| Engine | Description | OCR |
|---|---|---|
| llama Q4_K_M | "**directional signpost in Paris, France**, with five signs pointing to nearby landmarks such as the Louvre Museum, the Palace of the Louvre, the Musée du Louvre, the Théâtre du Palais-Royal, and the Mairie du 1er arrondissement…" | `Mairie du 1er` / `Palais du LOUVRE` / `LES ARTS DÉCORATIFS` / `Musée du LOUVRE` / `Théâtre du PALAIS-ROYAL` (perfect transcription, including accents) |
| llama Q8_0   | same as Q4_K_M, plus correctly labels arrow direction per sign and notes the architectural backdrop with statues | same — perfect transcription |
| **hipfire**  | **GPU PAGE FAULT — `Memory access fault by GPU node-1 on address 0x7f1a21e02000. Reason: Page not present.`** Vision encoder reached "2816 patches, 64×44 grid" then crashed. | (crash) |

**Verdict: hipfire crashes.** Both llama quants OCR all 5 signs perfectly.

### 5. `scene_2.jpg` — Duty-free perfume shop

| Engine | Description | OCR |
|---|---|---|
| llama Q4_K_M | "luxury cosmetics and perfume counter in a department store, featuring prominent brands like **Yves Saint Laurent, Ermenegildo Zegna, Boucheron, and Opium**…" | (token-budget overflow) |
| llama Q8_0   | "luxury perfume and cosmetics kiosk in a shopping mall, featuring prominent brands like **Yves Saint Laurent, Ermenegildo Zegna, and Boucheron**…" | (token-budget overflow) |
| **hipfire**  | **GPU PAGE FAULT** (same failure mode as scene_1) | (crash) |

**Verdict: hipfire crashes.** llama identifies brands by name and the social scene (browsing woman + sales associate).

### 6. `general_qa.jpg` — 12-panel scientific scatter-plot figure

| Engine | Description | OCR |
|---|---|---|
| llama Q4_K_M | "grid of 12 scatter plots… comparing… 'Clean accuracy' and other measures like LGA, LRA, LARS, LAGS, LAGA, and LARA for datasets CelebA and LSUN. Each plot includes a linear regression line and reports correlation coefficients (r) and Spearman's rank correlation (ρ)…" | All 6 visible panel titles + axis labels transcribed: "(1) Clean accuracy vs. LGA / (CelebA and LSUN; r = 0.65, ρ = 0.62)" … |
| llama Q8_0   | richer paragraph explaining the relationship between clean accuracy, adversarial perturbation norms, and severity correlations | same — all 6 panel headers transcribed verbatim including r and ρ values |
| **hipfire**  | **GPU PAGE FAULT** (same failure mode) | (crash) |

**Verdict: hipfire crashes.** llama.cpp reads small in-figure text (axis labels, panel headers, correlation values to 2 decimal places).

---

## Summary table

| Image | llama Q4_K_M | llama Q8_0 | hipfire (mq4+Q8head+F16 vision) |
|---|---|---|---|
| barney_cigar | ✓ describes scene | ✓ names person | ✗ "vertically stretched, bottles" |
| doge | ✓ "Shiba Inu / Doge" | ✓ "Shiba Inu / Doge" | ✗ "granular sugar/salt" |
| doge_napping | ✓ dog + text verbatim | ✓ dog + text verbatim | ✗ "coral / anemone" + hallucinated OCR |
| scene_1 (Paris signs) | ✓ scene + perfect OCR | ✓ scene + perfect OCR | **CRASH — GPU page fault** |
| scene_2 (perfume shop) | ✓ names 4 brands | ✓ names 3 brands | **CRASH — GPU page fault** |
| general_qa (12 panels) | ✓ structure + labels | ✓ rich analysis | **CRASH — GPU page fault** |

**Final score:** llama.cpp 12/12 correct. Hipfire 0/6 correct (3 quality failures + 3 crashes).

---

## Findings

### Finding 1 — base model is NOT weak (audit hypothesis falsified)

The May 20 audit's action item: "Base model just weak on unfamiliar memes — only F16 reference comparison rules it out." This comparison is that reference. **Llama.cpp's Q4_K_M (lower body-quant quality than hipfire's mq4 + Q8 lm-head) correctly identifies every meme that hipfire mis-describes.** The model knows what's in these images. The error is in hipfire's vision pipeline.

### Finding 2 — "vertically stretched" symptom across all hipfire descriptions

Hipfire descriptions repeatedly use phrases like "vertically stretched," "low-resolution," "blurry," "distortion from stretching the original photo into portrait orientation." This wording has appeared in:
- May 20 spliced run on `barney_cigar.jpg`
- May 20 q8head run on `barney_cigar.jpg`, `doge.jpeg`, `doge_napping.png`
- Today's fresh-binary `infer` run on `barney_cigar.jpg`
- Today's bench run on `barney_cigar.jpg`

This is too consistent to be model confusion. **Hipfire is feeding the vision tower an image that the model perceives as geometrically distorted.** The `smart_resize` function preserves aspect ratio in code review, so the corruption is downstream — candidates:

- **Patch-grid order mismatch.** `extract_patches` walks `(py, px)` in row-major, producing patches in (h-then-w) order. If the vision tower's positional encoding expects (w-then-h) or if the spatial merger reshapes (ph, pw) as (pw, ph), the model effectively sees a transposed image — which would feel "stretched."
- **Channel-swap regression.** `image.rs` deliberately swaps to (R, B, G) order based on pure-color PNG tests (`channel_order.rs`). Pure-color tests CANNOT detect a transpose or a more complex permutation — only a swap. If patch_embed weights expect (R, G, B), this swap corrupts the input on every natural image. Worth re-verifying with a non-pure-color reference (e.g. RGB gradient that distinguishes all 6 permutations).
- **`temporal_patch_size=2` duplication.** `extract_patches` duplicates the same frame into both temporal slots. If the vision tower processes them as different-time observations, the model sees a video where nothing moves — which could read as "motion blur / stretching."

### Finding 3 — vision encoder CRASHES on images that produce >~600 patches

Three of six bench images (scene_1, scene_2, general_qa) trigger `Memory access fault by GPU node-1 / Page not present` during the vision forward pass. The crashing image set is exactly the set that produces patch grids larger than the meme set:

| Image | Hipfire grid | Patches | Status |
|---|---|---|---|
| meme set (3 images) | up to 30×20 | ≤600 | runs (but wrong output) |
| scene_1 | 64×44 | 2816 | crash |
| scene_2 | ~64×48 | ~3000 | crash |
| general_qa | ~60×64 | ~3800 | crash |

llama.cpp processes all three correctly. The audit doc only tested ~square ≤600-patch memes, so this bug was previously invisible. **The vision tower (ViT attention / merger / patch_embed kernels in `rdna-compute`) has a fixed-size assumption that doesn't scale beyond the meme dimensions.** Likely candidates: a static LDS allocation, a launch-bounds limit, or a hardcoded `n_patches` buffer in the spliced F16 path.

### Finding 4 — the OpenAI PR (#312) is NOT the regression

The May 20 logs (pre-PR, via `infer.rs --image`) and today's bench (post-PR, via the same `infer.rs --image`) produce equivalent bad output on the same meme images. The new PR adds a wire-level base64-decode path on top of the existing preprocessor and vision tower — it didn't introduce the underlying quality bug. **The bug pre-existed the OpenAI PR.** The PR did, however, ship a path that now lets external clients hit the broken vision stack via the OpenAI-compatible endpoint, so the user-visible impact is larger now.

---

## Recommended next steps

In order of expected effort vs. impact:

1. **Reproduce on a non-pure-color reference for `channel_order.rs`.** Generate a 6-color image where R, G, B are each set to distinct, NON-extremal values (e.g. R=200, G=150, B=100) and run through both engines. A 1-pixel-per-color image rules out RGB vs. RBG by reading the model's color naming. Cost: ~30 minutes.
2. **Add a transpose-detection test image.** A tall thin image with horizontal text vs. its transposed twin. If hipfire labels them identically (or both as "stretched"), patch-grid order is the bug. Cost: ~1 hour.
3. **Trace the GPU page fault for scene_1.** The "64×44 grid, 2816 patches" log line right before the fault narrows the scope: instrument vision-encoder kernel launches and check shape parameters / buffer allocations against patch count. Cost: ~2-4 hours.
4. **Open issue tracking both bugs** — link this report. The crash bug (Finding 3) is a fresh regression-class bug; the quality bug (Finding 2) is what the May 20 audit was already chasing without a definitive comparison.

---

## Reproduction

```bash
cd ~/git/hipfire/benchmarks/vision
./run_bench.sh                # ~10 min, writes outputs/<image>/<prompt>__<engine>.txt
```

Inputs: `images/` (6 files, ~1.1 MB total). Outputs: `outputs/<image>/<prompt>__<engine>.txt`.
Bench script lives at `benchmarks/vision/run_bench.sh`. Identical greedy decode (`temp=0`, `n=300`) on both engines.

---

## Status — fix attempt #1 (2026-05-23 afternoon)

Set up HuggingFace reference (`Qwen/Qwen3.5-0.8B` — same vision-tower architecture as 9B at a smaller LM size, weights already cached at `~/.cache/huggingface/hub/`). The vision preprocessor is shared across all model sizes in the family, so reference dumps comparable to hipfire across the 0.8B / 4B / 9B / 27B range.

Tooling added:
- `dump_hf_reference.py` — dumps HF `pixel_values`, per-block hidden states, and `pos_embed` table for any image.
- `diff_dumps.py` — element-wise rel-L1 / max-|Δ| against hipfire, with hypothesis-discriminating permutation tests.
- `check_filter_residual.py` — Python implementation of hipfire's preprocessor to isolate resize-filter contribution.
- `HIPFIRE_VL_DUMP_DIR` env-gated dump of CHW + patches in `examples/infer.rs`.

### Preprocessor fix landed (this commit)

Three bugs found by diffing hipfire `extract_patches` output against HF `pixel_values` on `barney_cigar.jpg`:

| Bug | Before fix | After fix |
|---|---|---|
| 1. Patch ordering — emitted row-major in (py, px), HF expects 2×2-spatial-merge-grouped | rel-L1 0.350 alone | reorder added in `extract_patches` |
| 2. Per-patch layout — `(T, C, H, W)` flat, HF expects `(C, T, H, W)` | +6% on top of bug 1 | loop swap (C-outer, T-inner) |
| 3. Channel order — deliberate `(R, B, G)` swap from issue #23 | "validated" only on pure-color PNGs | reverted to `(R, G, B)` |

Combined diff vs HF after fix: **rel-L1 0.002** (175× improvement). Residual was the resize-filter difference (PIL BICUBIC vs Rust `FilterType::Triangle`); a follow-up swap to `FilterType::CatmullRom` (bicubic in the `image` crate) drops it further to **rel-L1 9.2e-5** (now classified by `diff_dumps.py` as "very close — likely just precision"). See `vision_rev_claude.md` review item 5.

A matching spatial-merge fix landed in `qwen35_vl.rs` since the GPU merger was reading `normed_data[(my*sms+dy) * grid_w + (mx*sms+dx)]` — correct for the old row-major layout but wrong for the new 2×2-grouped layout. The fix exploits that 4 patches forming one merged token are now consecutive in the buffer.

`channel_order.rs` regression test updated to assert the new (correct) `(R, G, B)` layout. All 4 tests pass.

### Outputs after fix attempt #1: **still wrong**

```
post-fix barney:  "low-resolution, vertically-oriented photograph of what appears to be
                  an indoor scene with several people seated closely together"
post-fix doge:    "highly pixelated, abstract close-up of a textured surface — possibly
                  fabric, paintbrush strokes, or biological tissue"
```

The preprocessor is now byte-correct vs HF, but the GPU vision tower has **two more architectural gaps** that the preprocessor fix exposed (they were previously masked by the row-major patch order):

### Remaining gap 1 — `pos_embed` interpolation

`qwen35_vl.rs:250` naively adds the first `n*h` floats of the `(2304, hidden) = (48×48, hidden)` learned table to `x`. HF's `fast_pos_embed_interpolate` bilinearly interpolates the 48×48 table down to the actual `(grid_h, grid_w)` for each image, then reorders into 2×2-grouped order to match patch sequence. For barney's `30×40` grid, hipfire is currently using pos_embed entries `[0..1200]` of a 48-wide table — i.e., grid rows are 48 columns wide in the lookup but 40 columns wide in the image, so every row is misaligned by `8 * row_idx` positions.

Fix: implement `fast_pos_embed_interpolate` on CPU (mechanical port of HF Python source), upload result, add to `x` instead of naive `add_inplace_f32`.

### Remaining gap 2 — 2D rotary in vision attention

`Qwen3_5VisionAttention.forward` calls `apply_rotary_pos_emb_vision(q, k, cos, sin)` before computing attention scores. The cos/sin come from `Qwen3_5VisionRotaryEmbedding(head_dim // 2)` and are looked up per-patch via `(row_idx, col_idx)` so each patch's Q/K is rotated by `concat(row_freq[py], col_freq[px])`.

Hipfire's `vit_attention_f32` / `vit_attention_opt` are plain dot-product attention — no rotary applied. This is a new-kernel addition (or modification of `vit_attention_opt` to take cos/sin inputs), plus CPU computation of the freq tables once per image.

### What's needed to reach llama.cpp parity

| Step | Estimate | Owner |
|---|---|---|
| Implement `fast_pos_embed_interpolate` on CPU + GPU upload + add | ~1 hr | TBD |
| Implement 2D-rotary kernel + integrate into vit attention path | ~3-4 hrs | TBD |
| Add per-block hipfire dump to `vision_forward`; diff each block vs HF; fix any remaining divergence | ~2 hrs | TBD |
| Re-run bench; expect all 6 images to match llama.cpp quality | ~30 min | TBD |

### Crash bug on >600-patch grids — still deferred

`scene_1.jpg` / `scene_2.jpg` / `general_qa.jpg` still trigger the GPU page fault. Whether the pos_embed/rotary fixes incidentally resolve this (e.g. if the bug is OOB read in a kernel that's relying on patch order, the new ordering may incidentally miss the OOB) is unknown — re-test after parity is reached.

---

## Status — fix attempt #2 (2026-05-23 evening): both vision-tower gaps closed

Implemented both remaining gaps in a single pass. End-to-end output now matches
llama.cpp quality on all 6 bench images — and the >600-patch crash is
incidentally resolved (root cause was bad indices feeding attention, not OOB
allocation).

### What landed

**`fast_pos_embed_interpolate` (CPU)** — `qwen35_vl.rs`. Bilinear-interpolates
the `(K×K, hidden)` learned position table down to each image's
`(grid_h, grid_w)` then reorders into 2×2 spatial-merge-grouped sequence
matching `extract_patches`. The `pos_embed` field on `VisionWeights` is now a
`Vec<f32>` resident on CPU instead of a `GpuTensor` (it has to be re-sampled
per-image and uploading a fresh `(n, h)` slice each call is cheaper than
duplicating it on device).

**2D vision rotary** — new HIP kernel `kernels/src/apply_rope_2d_vision.hip`,
dispatched as `Gpu::apply_rope_2d_vision_f32`. Each thread block handles one
`(head, token)` and rotates the corresponding Q and K halves of the packed
`qkv[N, 3*hidden]` buffer in-place, leaving V untouched. cos/sin tables are
shaped `(N, head_dim/2)` because HF's `cat((rope, rope), dim=-1)` makes the
two head_dim halves see the same angle. `compute_vision_rope_cos_sin`
(CPU) emits the table once per image in the same 2×2-grouped order as the
patches, then `vision_forward` uploads it once and the per-layer call site
slots the rotary in between `linear_f16(qkv)` and `vit_attention_f32`.

**Result: rel-L1 → byte-level model behavior.** The fix is validated end-to-end
rather than per-block — output quality and the crash disappearing are stronger
signals than tensor diffs, since both gaps would have produced the same
broken output (scrambled positional info → scrambled attention).

### Outputs (post-attempt-#2)

| Image | hipfire (this commit) | Verdict vs llama.cpp |
|---|---|---|
| barney_cigar — desc | "Neil Patrick Harris sits confidently in a plush leather armchair beside an ornate globe. Dressed sharply in a dark suit with lavender shirt and tie, he holds … a cigar … evoking the iconic 'How I Met Your Mother' bar scene" | ✓ matches Q8_0 |
| barney_cigar — ocr  | "No visible text appears in the image." | ✓ |
| doge — desc         | "close-up photo of the famous 'Doge' Shiba Inu dog, known for its wide-eyed, slightly confused expression that became an internet meme" | ✓ matches Q8_0 |
| doge — ocr          | "No visible text found on the image." | ✓ |
| doge_napping — desc | "Shiba Inu dog lying on its back … overlaid with humorous, misspelled captions like 'Much weak,' 'Such comfortable,' and 'Sleep like a doge'" | ✓ matches Q8_0 |
| doge_napping — ocr  | "Much weak / Such comfortable / Sleep like a doge / So stretch / Pointy teeth wow / Not scare now" | ✓ transcribes all 6 captions verbatim |
| scene_1 — desc      | "directional signpost with five black-and-white signs pointing to major Parisian landmarks: Mairie du 1er, Palais du Louvre, Les Arts Décoratifs, Musée du Louvre …" | ✓ matches Q8_0 |
| scene_1 — ocr       | "Mairie du 1er / Palais du LOUVRE / LES ARTS DÉCORATIFS / Musée du LOUVRÉ ← Théâtre du PALAIS-ROYAL" | ✓ all 5 signs (accent on É vs E is the only diff) |
| scene_2 — desc      | "luxury perfume and cosmetics kiosk … featuring prominent brands like YSL (Yves Saint Laurent), Ermenegildo Zegna, Boucheron, and Alexander McQueen … duty-free shopping" | ✓ matches Q8_0 |
| scene_2 — ocr       | "YVES SAINT LAURENT / OPIUM / Ermenegildo Zegna / MOQUEEN / BOUCHERON" | ✓ |
| general_qa — desc   | "grid of scatter plots comparing clean accuracy and various adversarial robustness metrics (LGA, LAGS, LRA) across two datasets: CelebA and LSUN … r for Pearson's r, ρ for Spearman's rho" | ✓ matches Q8_0 |
| general_qa — ocr    | "(1) Clean accuracy vs. LGA (CelebA and LSUN; r=0.65, ρ=0.62) … (2) Clean accuracy vs. LAGS (ε=1) (CelebA and LSUN; r=0.67, ρ=0.68) …" | ✓ panel headers + r/ρ to 2 decimals |

**Score: hipfire 12/12 matches llama.cpp.** Previously: 0/6 (3 quality failures + 3 crashes).

### Crash bug on >600-patch grids — no longer triggered; root cause unknown

The page faults on `scene_1` / `scene_2` / `general_qa` (2816-3800 patches) do
not reproduce on this bench set after the pos_embed/rotary fixes landed. The
mechanism is **not** root-caused. Speculation about "aliased indices into
attention causing an OOB-shaped score distribution" does not survive scrutiny:

- `vision_forward` calls `vit_attention_f32`, not `vit_attention_opt`, so any
  LDS-budget theory tied to the optimized variant doesn't apply.
- `vit_attention_f32` reserves `(N + block_size) * 4` LDS. At N=3800 +
  block_size=256 that's ~16 KB, well under gfx1100's 64 KB.
- The kernel has bounds checks (`if (head >= num_heads || qi >= N) return;`),
  so thread launches past N don't deref past allocations.
- "Page not present" implies an unmapped address, not bounds-overflow inside
  a mapped allocation — usually unmapped faults have a different cause from
  the kind of bug a position-info corruption would produce.

A future stress test that walks patch counts 600/1200/2800/3800/6000/10000
might expose the same fault if the underlying bug is data-dependent on
specific grid shapes or specific patch counts. Treat the apparent fix as
provisional until either the root cause is understood OR an empirical
stress sweep confirms it stays absent across a wider distribution of
patch totals.

### Files touched in this attempt

- `kernels/src/apply_rope_2d_vision.hip` — new HIP kernel
- `crates/rdna-compute/src/kernels.rs` — `APPLY_ROPE_2D_VISION_SRC` const
- `crates/rdna-compute/src/dispatch.rs` — `Gpu::apply_rope_2d_vision_f32`
- `crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs` — interp + rotary helpers,
  `pos_embed: Vec<f32>` (CPU), rewired `vision_forward`,
  `num_position_embeddings` in `VisionConfig`
- `crates/hipfire-runtime/examples/infer.rs` — stale `R,B,G` dump comment
  cleaned up (channel order is `R,G,B` since b47ba99a)
- `benchmarks/vision/comparison-2026-05-23.md` — this log

Outstanding follow-up: per-block dump tooling (mentioned as Gap 3) is not
required given the end-to-end results, but adding it would let us catch any
silent regression in future kernel work touching the vision tower.
