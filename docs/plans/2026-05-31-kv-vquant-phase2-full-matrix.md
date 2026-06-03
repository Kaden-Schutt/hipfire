# V-cache quant Phase 2 — full K/V matrix via UNIFORM 256-wide V

**Date:** 2026-05-31 · **Branch:** feat/kv-vquant-fwht-lloyd-v (continues Phase 1).
**Priority (user-set), in order; docs/defaults come AFTER:**
1. lloyd2/3/4-V under **fwht2-K and fwht4-K** too → full 3 K × 4 V matrix.
2. Fleet-portable kernels (load+run on gfx942 wave64, gfx1201, etc.).
3. Full **KLD + perf matrix**.
4. THEN: KV-system docs + optimized auto-defaults (adaptive-KV, see memory `project_hipfire_adaptive_kv`).

## Key realization (why this is easy — supersedes the earlier 128-wide plan)

The fwht2/4 kernels are 128-wide and fwht3 is 256-wide **only by lineage + bit-packing**, not by any FWHT requirement:
- Each fwht mode was cloned from the matching asym (Givens) mode and inherited its layout (kernel headers confirm: fwht2 *"structurally identical to asym_k_givens2"* = 4 dims/thread per half; fwht3 = 8 dims/thread).
- The driver is **byte-alignment of packed indices**: 3-bit only aligns at 8 dims/thread (24 bits = 3 bytes) → forced 256-wide; 2-bit (1 byte) and 4-bit (2 bytes) align at 4 dims/thread → the inherited 128-wide. 2/4-bit would work 256-wide too (8×2=2 bytes, 8×4=4 bytes — both aligned).

**The V phase is decoupled from the K-scoring layout** (verified in `attention_flash_fwht2_tile.hip`): scores live in LDS; the V phase does `out_vec[d] += scores[t]·V[t][d]` and writes `partials[2+d]` per *absolute* dim; the reduce kernel sums per-dim, layout-agnostic. So **V's rotation width need not match K's.** → Use a **single uniform 256-wide V layout** (the `fwht256_2bit`/`fwht3`/`fwht256_4bit` writers already built in Phase 1; all byte-aligned at 256-wide) and read it identically in every K kernel.

**Signs:** `gen_fwht_signs(seed,n)` is a pure LCG of `seed` (verified) ⇒ `gen(42,256)[0..128] == gen(42,128)`. So allocating a fwht2/4 cache's sign tables at **256 elements** lets the 128-wide K rotation read the first 128 (identical values → unchanged) AND the 256-wide V read all 256 — **no separate V-sign tensor, no new kernel param.**

Net: full matrix = **uniform 256-wide V everywhere**, no 128-wide V kernels, no awkward 128-wide-3-bit.

---

## Task 2A — enable lloyd-V on fwht2/4 caches (signs + guard); write side
**File:** `crates/hipfire-runtime/src/llama.rs`.
- `set_v_mode_realloc`: when `v_mode != Q8`, ensure the cache's FWHT sign tables (`givens_cos`/`givens_sin`) are **256-element**. If already 256 (fwht3) leave them; if 128 (fwht2/4) reallocate to `gen_fwht_signs(42,256)` / `(1042,256)` and re-upload. (head_dim must be 256.) The 128-wide K rotation reading the first 128 is unchanged (LCG prefix).
- Relax the guard from fwht3-only to **fwht{2,3,4}-K**: `assert!((quant_asym2||quant_asym3||quant_asym4) && quant_fwht || v_mode==Q8, ...)`.
- **Write side needs no other change**: the selector `kv_write_v_by_mode` already routes v_mode → the 256-wide writers (`fwht256_2bit`/`fwht3_vec`/`fwht256_4bit`, K-mode-independent), and the fwht2/4 fused/batched wrappers already call it (6a) — they just pass the now-256 signs.
- (Single-GPU only here; multi-GPU `_multi_filtered` V-mode is a later follow-up.)

## Task 2B — fwht2/4 attention: 256-wide lloyd V-read (the read side)
**Files:** `kernels/src/attention_flash_fwht{2,4}_tile.hip` + `_batched.hip`; `crates/rdna-compute/src/dispatch.rs` (`attention_flash_fwht2`, `attention_flash_fwht4`, `attention_flash_fwht{2,4}_batched_masked`); `crates/hipfire-arch-qwen35/src/qwen35.rs`.
- **Single tiles** (`attention_flash_fwht{2,4}_tile.hip`): add `int v_mode` as the final kernel param (3a did this for fwht3). Wrap the existing Q8 V read in `if (v_mode==8){…} else {<256-wide lloyd branch>}`. The lloyd branch is the **verbatim 256-wide branch from `attention_flash_fwht3_tile.hip`** (8 dims/thread `tid*8`, `TURBO_C{2,3,4}_256` by v_mode, `4 + head_dim*v_mode/8` layout, `fwht_shfl_inverse_256(out_vec[0..7], signs1, signs2, tid)`), gated `head_dim==256`. It reuses the kernel's existing `signs1/signs2` (now 256-element for the cache) — the Q rotation still reads the first 128, the V inverse reads all 256.
- **Batched tiles** (`_tile_batched.hip`): already have `int v_mode` (3b). Add the same 256-wide lloyd branch.
- **Single launchers** (`attention_flash_fwht2`, `attention_flash_fwht4`): add `v_mode_bits: i32` param + append as the last tile kernarg (mirror `attention_flash_fwht3`).
- **Batched_masked wrappers** (fwht2/4): currently pass literal `V_MODE_Q8` to `launch_asym_flash_batched` — change to take a `v_mode_bits` param and pass it (mirror fwht3_batched_masked).
- **qwen35.rs**: every `attention_flash_fwht2(`/`fwht4(` (single) and `attention_flash_fwht2_batched_masked(`/`fwht4_…(` call passes `kv_cache.v_mode_bits()`.

## Task 2C — validate (the gate: coherence + KLD + perf)
- **Smoke KLD** (4-chunk): `--kv-mode fwht2 --kv-v {lloyd2,lloyd3,lloyd4}` and `fwht4 × {…}` on qwen3.6-27b.mq4 — finite, monotonic vs their q8-V baselines (no OOB/garbage from the signs/layout).
- **Full KLD matrix** (24-chunk, qwen3.6-27b.mq4): fwht{2,3,4} × {q8,lloyd2,lloyd3,lloyd4} = 12 cells. Fill the design-doc grid. Surface the **equal-byte K/V-split** comparisons (e.g. 200 B/head = fwht2/lloyd4 vs fwht3/lloyd3 vs fwht4/lloyd2 — which split is most accurate?).
- **Coherence gate** on the leading new cells (≥ fwht2/lloyd4, fwht4/lloyd4) — exit 0 + fluent on 3.6-27b.mq4.
- **Perf**: warmed decode A/B for a couple new cells; + (optional, orthogonal) the **per-tile→reduce-kernel inverse** optimization to recover the ~4.4% short-ctx tax, re-measured.
- Ship/keep only cells that pass all three. (Default stays Q8-V; lloyd-V opt-in.)

## Fleet portability (Task 2D, folds into 2C)
The new V kernels are copies of the fleet-validated K kernels (32-thread blocks, `ds_swizzle`), so this is verification not rewrite: confirm load+correct KLD on **MI300X gfx942 (wave64)** and an **R9700 gfx1201**. The 256-wide path is the same machinery the fwht3 K KLD already ran on MI300X, so low risk.

## Then (Phase 3, after 2A–2C): docs + defaults
- KV-system reference (every K×V mode, byte formulas, width/packing rules, the KLD+perf matrix, when to pick what).
- Optimized auto-defaults / adaptive-KV (VRAM-fit autoselect) so non-technical users don't choose manually.
