# Adaptive KV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Runtime VRAM-fit downshift of both K and V KV-cache precision as
context grows, re-quantizing the existing cache in place along a configurable
pattern — turning the hard `max_seq` ceiling into graceful degradation.

**Architecture:** A `KvAdaptive` controller (sibling of `EvictionCtx`), called
after each token write in the Qwen3.5 decode branch, watches `seq_pos` against
capacity thresholds derived from a fixed floor-sized byte budget. When a
threshold is crossed it runs an O(ctx) **transcode pass** (re-quantize K or V to
the next-lower tier, in place, via a 1-layer scratch), flips the tier, and
defensively invalidates any captured graph. V chain: q8→lloyd4→lloyd3→lloyd2;
K chain: fwht4→fwht2 (same-width remap) with fwht3 available via re-rotation.

**Tech Stack:** Rust (`crates/hipfire-runtime`, `crates/rdna-compute`), HIP
kernels (`kernels/src/*.hip`), JIT-compiled via `ensure_givens4_kernel`.
Validation: `eval_hipfire` KLD vs bf16 ref, `scripts/coherence-gate.sh`, warmed
daemon decode perf. Target: gfx1100 / `qwen3.6-27b.mq4`; fleet: gfx1201, gfx1151.

**Design:** `docs/plans/2026-05-31-adaptive-kv-design.md`.

---

## Grounding facts (verified by code read — reference while implementing)

**Decode hook (Task 5):**
- `fn generate` signature: `daemon.rs:4109`. Splits at `daemon.rs:4543` on
  `m.arch_id == 5 || m.arch_id == 6` (Qwen3.5/MoE) vs else (Qwen3/LLaMA).
- **Target = Qwen3.5 branch.** `kv = m.kv_cache.as_mut().unwrap()` at
  `daemon.rs:4549`. Main loop `while generated < max_tokens` at `daemon.rs:4704`.
  Per-token: `m.seq_pos += 1` at `4732`; eviction block `if let Some(ref ev) =
  m.eviction { ... ev.maybe_evict(gpu, kv, m.seq_pos) ... m.seq_pos = new_phys }`
  at `4733-4737`. **Insert `maybe_downshift` immediately after `4737`.** In
  scope: `gpu: &mut Gpu`, `kv: &mut llama::KvCache`, `m.seq_pos: usize`.
- `LoadedModel` struct: `daemon.rs:559`. Has `kv_cache: Option<llama::KvCache>`,
  `seq_pos: usize`, `max_seq: usize`, `physical_cap: usize`,
  `eviction: Option<Eviction>` (`Eviction` enum at `daemon.rs:50`). Add
  `kv_adaptive: Option<hipfire_runtime::kv_adaptive::KvAdaptive>` alongside.
- `generate` gets `gpu` as its 2nd param (`&mut rdna_compute::Gpu`), called at
  `daemon.rs:1451`.

**Graph state (Task 0):**
- AR forward graph hard-disabled: `let use_graph = false;` at `qwen35.rs:4324`.
- `v_mode_bits()` (`llama.rs:3493`) read live per forward; passed into
  `launch_asym_flash_batched` (`dispatch.rs:24255`, param `v_mode: i32`), baked
  into a kernarg blob ONLY under `capture_mode` (`dispatch.rs:24357`). Reduce
  selection (lloyd vs asym) also branches at `dispatch.rs:24281`.
- GDN tape replay: `replay_graph_cache: HashMap<usize,(Graph,GraphExec,Vec<Vec<u8>>)>`
  (`dispatch.rs:427`), keyed by `n_steps`, gated by env `HIPFIRE_REPLAY_GRAPH=1`
  (`speculative.rs:685`); does NOT capture FA attention, so does NOT bake v_mode.
- Invalidation methods exist but are UNWIRED in decode:
  `replay_graph_destroy_all` (`dispatch.rs:1041`), `drop_captured_graph`
  (`dispatch.rs:854`), `mark_kernels_dirty` (`dispatch.rs:869`), `graph_destroy`
  (`dispatch.rs:876`).
- Eviction (`triattn.rs:833`, `cask.rs:77`) changes `physical_cap` + compacts
  buffers mid-stream with ZERO graph invalidation — same latent hazard, ships today.

**Kernels (Tasks 3, 6):**
- V writers: `kv_cache_write_fwht256_2bit` (68 B/head), `kv_cache_write_fwht256_4bit`
  (132 B/head), both 256-wide `tid*8`, sig `(k_dst, k_src, pos_buf, signs1,
  signs2, n_kv_heads, head_dim)`. lloyd3 V reuses `kv_cache_write_asym_k_fwht3`
  (100 B/head, 256-wide).
- K writers: `kv_cache_write_asym_k_fwht2` (128-wide, `TURBO_C2`, 68 B/head@256),
  `kv_cache_write_asym_k_fwht3` (256-wide, `TURBO_C3_256`, 100 B/head),
  `kv_cache_write_asym_k_fwht4` (128-wide, `TURBO_C4`, 132 B/head@256). Same
  7-param sig.
- LUTs in `kernels/src/turbo_common.h`: 256-family `TURBO_C2_256[4]`,
  `TURBO_C3_256[8]`, `TURBO_C4_256[16]` (lines 21-27); 128-family `TURBO_C2[4]`,
  `TURBO_C3[8]`, `TURBO_C4[16]` (lines 13-19). 128-family = √2 × 256-family.
- Downshift remap tables (nearest-centroid, within family): `C4→C2 =
  [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]`; `C4→C3 = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]`;
  `C3→C2 = [0,0,1,1,2,2,3,3]`. **cnorm must be recomputed** after remap.
- FWHT primitives (`turbo_common.h`): 256-wide `fwht_shfl_forward_256` (:168),
  `fwht_shfl_inverse_256` (:221); 128-wide `fwht_shfl_forward` (:81),
  `fwht_shfl_inverse` (:115). Quantizers: `turbo_quantize_{2,3,4}bit_256`
  (:330-337), 128-wide `turbo_quantize_{2,3,4}bit` (:311-322).
- Registration recipe: const in `kernels.rs` (`include_str!`) →
  `ensure_givens4_kernel(name, src, func_name)` (prepends `TURBO_COMMON_H` +
  `GIVENS_COMMON_SRC`, strips `#include`) → launch fn (template:
  `kv_cache_write_v256_2bit_vec` at `dispatch.rs:23992`).

**Capacity math (head_dim=256, n_kv_heads=4):**
- V bytes/head: q8=272, lloyd4=132, lloyd3=100, lloyd2=68.
- K bytes/head: fwht4=132, fwht3=100, fwht2=68.
- `budget(layer) = max_seq * n_kv_heads * (k_bph(fwht2) + v_bph(lloyd2))`
  `= max_seq * 4 * 136`.
- `cap(K,V) = budget / (n_kv_heads * (k_bph(K) + v_bph(V)))`.

**Validation:**
- Build: `RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime
  --example daemon --example eval_hipfire` (default features include all 6
  arches + deltanet; do NOT pass `--features deltanet` alone — daemon's
  required-features gate fails). Always verify: `cargo build ...; echo
  BUILD_EXIT=$?` (LSP diagnostics go stale).
- KLD: `./target/release/examples/eval_hipfire --model ~/.hipfire/models/qwen3.6-27b.mq4
  --ref ~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin --output
  benchmarks/quality-baselines/results/2026-05-31-kv-vquant/<name>.kldseq
  --kv-mode <K> --kv-v <V> --scoring-mode prefill --max-chunks 24`. Flag is
  `--ref` (NOT `--kldref`). KLD prints to stderr: `eval_hipfire: slice-mean KLD
  = <n>`.
- Coherence: `./scripts/coherence-gate.sh` (short) / `--full` (adds 27B). No
  "PASS" string; exit 0 + "no hard errors" = pass. Skips models not in
  `~/.hipfire/models/`.
- GPU lock: `source scripts/gpu-lock.sh && gpu_acquire "adaptive-kv" && <run> &&
  gpu_release` (eval/daemon do NOT auto-lock).
- Unit test: inline `#[cfg(test)] mod tests`; run `cargo test -p hipfire-runtime
  --lib kv_adaptive::tests::<name> -- --exact` (CPU-only, no GPU).
- Commits: `git commit --no-verify` (the pre-commit coherence hook is run
  explicitly per gating).

---

## Task 0: Spike 0 — graph-state proof + defensive invalidation (BLOCKING)

**Goal:** Prove a mid-stream KV tier switch is safe on the linear path today, and
wire defensive graph invalidation so it stays safe if graphs re-enable.

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs` (add `invalidate_for_kv_mode_switch`)
- Create: `docs/plans/2026-05-31-adaptive-kv-spike0.md` (findings)

- [ ] **Step 1: Document the graph-state finding.** Write `spike0.md` recording:
  `use_graph=false` at qwen35.rs:4324 ⇒ AR forward graph inactive ⇒ v_mode_bits
  live per forward; GDN tape (`replay_graph_cache`) is spec-decode/env-gated and
  does not bake v_mode; therefore on the linear `generate` path a mid-stream tier
  switch is reflected immediately. Conclusion: safe today; defensive invalidation
  added for future graph-on correctness.
- [ ] **Step 2: Add a single invalidation entry point.** In `dispatch.rs`, add:
```rust
/// Drop all captured/replayed graph state so the next forward re-captures with
/// the current KV tier. Called after any mid-stream KV-mode switch (adaptive KV)
/// because a captured graph bakes v_mode_bits (dispatch.rs:24357) and the
/// lloyd-vs-asym reduce selection. No-op today (use_graph=false) but
/// correct-by-construction if the AR forward graph is re-enabled.
pub fn invalidate_for_kv_mode_switch(&mut self) {
    self.drop_captured_graph();
    self.mark_kernels_dirty();
    self.replay_graph_destroy_all();
}
```
- [ ] **Step 3: Build.** `RUSTC_WRAPPER=sccache cargo build --release -p
  rdna-compute; echo BUILD_EXIT=$?` → expect `BUILD_EXIT=0`.
- [ ] **Step 4: Commit.** `git add -A && git commit --no-verify -m "spike(kv):
  graph-state proof + defensive invalidate_for_kv_mode_switch for adaptive KV"`

---

## Task 1: `KMode` accessor + capacity/threshold math (CPU, unit-tested)

**Goal:** Pure-Rust tier model + capacity arithmetic, fully unit-tested without a GPU.

**Files:**
- Create: `crates/hipfire-runtime/src/kv_adaptive.rs`
- Modify: `crates/hipfire-runtime/src/lib.rs` (add `pub mod kv_adaptive;` under
  the `#[cfg(feature="deltanet")]` block, after `pub mod triattn;` at line 33)

- [ ] **Step 1: Create the module with tier types + byte tables.**
```rust
//! Adaptive KV: runtime VRAM-fit downshift of K/V cache precision.
//! See docs/plans/2026-05-31-adaptive-kv-design.md.
use crate::llama::VMode;

/// K-cache tier. Mirrors VMode for the V side. fwht4/fwht2 rotate 128-wide,
/// fwht3 rotates 256-wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMode { Fwht4, Fwht3, Fwht2 }

impl KMode {
    /// bytes-per-head at a given head_dim.
    pub fn bytes_per_head(self, head_dim: usize) -> usize {
        match self {
            KMode::Fwht4 => 4 + head_dim / 2,        // 132 @256
            KMode::Fwht3 => 4 + (head_dim * 3) / 8,  // 100 @256
            KMode::Fwht2 => 4 + head_dim / 4,        // 68  @256
        }
    }
    /// FWHT rotation width.
    pub fn rot_width(self) -> usize { match self { KMode::Fwht3 => 256, _ => 128 } }
}

/// V bytes-per-head (mirrors KvCache::v_bytes_per_pos per-head logic).
pub fn v_bytes_per_head(v: VMode, head_dim: usize) -> usize {
    match v {
        VMode::Q8 => (head_dim / 32) * 34,                 // 272 @256
        VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => 4 + (head_dim * v.bits() as usize) / 8,
    }
}
```
- [ ] **Step 2: Add the capacity function.**
```rust
/// Token capacity of the floor-sized buffer at a given (K,V) tier.
pub fn cap_tokens(budget_bytes_per_layer: usize, n_kv_heads: usize, head_dim: usize,
                  k: KMode, v: VMode) -> usize {
    let per_tok = n_kv_heads * (k.bytes_per_head(head_dim) + v_bytes_per_head(v, head_dim));
    if per_tok == 0 { 0 } else { budget_bytes_per_layer / per_tok }
}

/// Floor-sized per-layer byte budget = capacity for `max_seq` tokens at the floor.
pub fn budget_bytes_per_layer(max_seq: usize, n_kv_heads: usize, head_dim: usize,
                              k_floor: KMode, v_floor: VMode) -> usize {
    max_seq * n_kv_heads * (k_floor.bytes_per_head(head_dim) + v_bytes_per_head(v_floor, head_dim))
}
```
- [ ] **Step 3: Add unit tests** (inline `#[cfg(test)] mod tests`):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::VMode;
    #[test]
    fn byte_tables_match_design_256() {
        assert_eq!(KMode::Fwht4.bytes_per_head(256), 132);
        assert_eq!(KMode::Fwht3.bytes_per_head(256), 100);
        assert_eq!(KMode::Fwht2.bytes_per_head(256), 68);
        assert_eq!(v_bytes_per_head(VMode::Q8, 256), 272);
        assert_eq!(v_bytes_per_head(VMode::Lloyd4, 256), 132);
        assert_eq!(v_bytes_per_head(VMode::Lloyd3, 256), 100);
        assert_eq!(v_bytes_per_head(VMode::Lloyd2, 256), 68);
    }
    #[test]
    fn floor_budget_gives_max_seq_at_floor() {
        let max_seq = 1000;
        let b = budget_bytes_per_layer(max_seq, 4, 256, KMode::Fwht2, VMode::Lloyd2);
        assert_eq!(cap_tokens(b, 4, 256, KMode::Fwht2, VMode::Lloyd2), max_seq);
    }
    #[test]
    fn cap_grows_as_precision_drops() {
        let b = budget_bytes_per_layer(1000, 4, 256, KMode::Fwht2, VMode::Lloyd2);
        let c_start = cap_tokens(b, 4, 256, KMode::Fwht4, VMode::Q8);
        let c_floor = cap_tokens(b, 4, 256, KMode::Fwht2, VMode::Lloyd2);
        assert!(c_floor > c_start * 2, "floor should fit >2x start ({c_floor} vs {c_start})");
        // design table: start K4/q8 ≈ 0.337*max_seq
        assert!((330..=345).contains(&c_start), "start cap {c_start}");
    }
}
```
- [ ] **Step 4: Build + test.** `cargo build -p hipfire-runtime; echo
  BUILD_EXIT=$?` then `cargo test -p hipfire-runtime --lib kv_adaptive::tests --
  --nocapture`. Expect 3 passing.
- [ ] **Step 5: Commit.** `git add -A && git commit --no-verify -m "feat(kv):
  adaptive KV tier model + capacity math (KMode, cap_tokens), CPU-tested"`

---

## Task 2: `KvAdaptive` controller — pattern + thresholds (CPU, unit-tested)

**Goal:** The pattern abstraction (ordered steps), threshold computation, and the
preset/advanced-floor pattern generators. No GPU yet — `maybe_downshift` is added
in Task 5.

**Files:**
- Modify: `crates/hipfire-runtime/src/kv_adaptive.rs`

- [ ] **Step 1: Define a step + the controller state.**
```rust
/// One downshift step: drop ONE cache by one tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step { V(VMode), K(KMode) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset { Conservative, Balanced, Aggressive }

pub struct KvAdaptive {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub budget_bytes_per_layer: usize,
    pub cur_k: KMode,
    pub cur_v: VMode,
    pub steps: Vec<Step>,        // ordered remaining steps
    pub next_step: usize,        // index into steps
    pub thresholds: Vec<usize>,  // seq_pos at which steps[i] fires
    pub margin: usize,           // fire this many tokens before the cap
}
```
- [ ] **Step 2: Pattern generators (balanced default + advanced floors).**
```rust
impl KvAdaptive {
    /// Build the default `balanced` step order: V q8→l4→l3, K f4→f2, V l3→l2.
    /// (Keeps the K/V bit-gap ≤ 1 tier; finalized empirically in Task 8.)
    fn balanced_steps(k_floor: KMode, v_floor: VMode) -> Vec<Step> {
        let mut s = Vec::new();
        // descend V to lloyd3 first (biggest byte win up front)
        if v_floor != VMode::Q8 { s.push(Step::V(VMode::Lloyd4)); }
        if matches!(v_floor, VMode::Lloyd3 | VMode::Lloyd2) { s.push(Step::V(VMode::Lloyd3)); }
        // K step (cheap same-width fwht4→fwht2) once V is at lloyd3
        if k_floor == KMode::Fwht2 { s.push(Step::K(KMode::Fwht2)); }
        else if k_floor == KMode::Fwht3 { s.push(Step::K(KMode::Fwht3)); }
        // final V step to the floor
        if v_floor == VMode::Lloyd2 { s.push(Step::V(VMode::Lloyd2)); }
        s
    }

    pub fn from_preset(p: Preset, max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let (k_floor, v_floor) = match p {
            Preset::Conservative => (KMode::Fwht4, VMode::Lloyd4),
            Preset::Balanced     => (KMode::Fwht2, VMode::Lloyd2),
            Preset::Aggressive   => (KMode::Fwht2, VMode::Lloyd2),
        };
        Self::new(max_seq, n_kv_heads, head_dim, k_floor, v_floor)
    }

    /// Advanced: caller picks K and V floors independently.
    pub fn new(max_seq: usize, n_kv_heads: usize, head_dim: usize,
               k_floor: KMode, v_floor: VMode) -> Self {
        let budget = budget_bytes_per_layer(max_seq, n_kv_heads, head_dim, k_floor, v_floor);
        let steps = Self::balanced_steps(k_floor, v_floor);
        let mut s = Self {
            n_kv_heads, head_dim, budget_bytes_per_layer: budget,
            cur_k: KMode::Fwht4, cur_v: VMode::Q8, steps, next_step: 0,
            thresholds: Vec::new(), margin: 64,
        };
        s.recompute_thresholds();
        s
    }

    /// threshold[i] = cap(state AFTER applying steps[0..=i-1]) - margin,
    /// i.e. the seq_pos at which we must apply steps[i] before overflowing the
    /// cap of the CURRENT (pre-step-i) tier.
    fn recompute_thresholds(&mut self) {
        let mut k = KMode::Fwht4; let mut v = VMode::Q8;
        self.thresholds.clear();
        for st in &self.steps {
            let cap = cap_tokens(self.budget_bytes_per_layer, self.n_kv_heads, self.head_dim, k, v);
            self.thresholds.push(cap.saturating_sub(self.margin));
            match *st { Step::V(nv) => v = nv, Step::K(nk) => k = nk }
        }
    }
}
```
- [ ] **Step 3: Unit tests** for pattern + monotone thresholds.
```rust
    #[test]
    fn balanced_pattern_shape() {
        let a = KvAdaptive::from_preset(Preset::Balanced, 10_000, 4, 256);
        assert_eq!(a.steps, vec![
            Step::V(VMode::Lloyd4), Step::V(VMode::Lloyd3),
            Step::K(KMode::Fwht2), Step::V(VMode::Lloyd2),
        ]);
        // thresholds strictly increasing (each tier fits more before the next shift)
        for w in a.thresholds.windows(2) { assert!(w[1] > w[0], "thresholds {:?}", a.thresholds); }
    }
    #[test]
    fn conservative_only_v_to_lloyd4() {
        let a = KvAdaptive::from_preset(Preset::Conservative, 10_000, 4, 256);
        assert_eq!(a.steps, vec![Step::V(VMode::Lloyd4)]);
    }
    #[test]
    fn advanced_k_fwht3_floor() {
        let a = KvAdaptive::new(10_000, 4, 256, KMode::Fwht3, VMode::Lloyd2);
        assert!(a.steps.contains(&Step::K(KMode::Fwht3)));
    }
```
- [ ] **Step 4: Build + test.** `cargo test -p hipfire-runtime --lib
  kv_adaptive::tests -- --nocapture`. Expect all passing.
- [ ] **Step 5: Commit.** `--no-verify -m "feat(kv): KvAdaptive pattern +
  threshold model (presets + advanced floors), CPU-tested"`

---

## Task 3: V transcode kernels + dispatch launchers

**Goal:** GPU kernels that re-quantize an existing V cache (all positions, one
layer) from a higher V tier to a lower one, in place, forward-safe.

**Files:**
- Create: `kernels/src/kv_transcode_v_q8_to_lloyd4.hip`
- Create: `kernels/src/kv_transcode_v_lloyd_down.hip`
- Modify: `crates/rdna-compute/src/kernels.rs` (two `include_str!` consts)
- Modify: `crates/rdna-compute/src/dispatch.rs` (two launch fns)

- [ ] **Step 1: `q8→lloyd4` kernel.** Reads Q8_0 V (normal space), dequantizes a
  full 256-dim head vector, FWHT-rotates 256-wide (`fwht_shfl_forward_256`,
  signs1/signs2), quantizes to `TURBO_C4_256` with per-(pos,head) cnorm, writes
  132 B/head. Grid `[n_kv_heads, n_positions]` (or loop positions in-block).
  Model the read on the existing Q8 V-read in `attention_flash_*` and the write
  on `kv_cache_write_fwht256_4bit`. Signature:
```c
extern "C" __global__ void kv_transcode_v_q8_to_lloyd4(
    unsigned char* __restrict__ dst,        // lloyd4 layout (132 B/head/pos)
    const unsigned char* __restrict__ src,  // q8 layout (272 B/head/pos)
    const float* __restrict__ signs1, const float* __restrict__ signs2,
    int n_kv_heads, int head_dim, int n_positions)
```
  **In-place safety:** dst stride (132) < src stride (272); iterate positions
  ascending so the write pointer always trails the read pointer. (dst may alias
  src; document that the caller passes the same buffer.)
- [ ] **Step 2: `lloyd_hi→lloyd_lo` kernel.** Parameterized by source bits and
  target bits (4→3, 3→2; also 4→2 for the direct path). Reads idx + cnorm,
  reconstructs rotated value `cnorm * TURBO_C{hi}_256[idx]`, re-quantizes with
  `turbo_quantize_{lo}bit_256` and recomputes cnorm (`orig_norm/recon_norm`).
  No FWHT (already rotated). Signature:
```c
extern "C" __global__ void kv_transcode_v_lloyd_down(
    unsigned char* __restrict__ dst, const unsigned char* __restrict__ src,
    int n_kv_heads, int head_dim, int n_positions, int src_bits, int dst_bits)
```
- [ ] **Step 3: Register consts** in `kernels.rs`:
```rust
pub const KV_TRANSCODE_V_Q8_TO_LLOYD4_SRC: &str = include_str!("../../../kernels/src/kv_transcode_v_q8_to_lloyd4.hip");
pub const KV_TRANSCODE_V_LLOYD_DOWN_SRC: &str = include_str!("../../../kernels/src/kv_transcode_v_lloyd_down.hip");
```
- [ ] **Step 4: Launch fns** in `dispatch.rs` (template:
  `kv_cache_write_v256_2bit_vec` at `dispatch.rs:23992`), e.g.
  `transcode_v_q8_to_lloyd4(&mut self, buf, signs1, signs2, n_kv_heads, head_dim,
  n_positions)` and `transcode_v_lloyd_down(&mut self, buf, n_kv_heads, head_dim,
  n_positions, src_bits, dst_bits)`, each calling `ensure_givens4_kernel(...)`.
  Grid `[n_kv_heads, n_positions, 1]`, block `[32,1,1]`, shared `(head_dim+32)*4`.
- [ ] **Step 5: Build.** `RUSTC_WRAPPER=sccache cargo build --release -p
  rdna-compute; echo BUILD_EXIT=$?` → `0`. (JIT compile happens at first launch;
  build only checks Rust.)
- [ ] **Step 6: Commit.** `--no-verify -m "feat(kv): V transcode kernels
  (q8→lloyd4 FWHT, lloyd-down remap) + launchers"`

---

## Task 4: V transcode orchestration + correctness test

**Goal:** A `KvCache` method that applies one V step (all real KV layers, in
place via a 1-layer scratch), plus a GPU correctness test (transcode ≈ direct
write of the same data).

**Files:**
- Modify: `crates/hipfire-runtime/src/llama.rs` (add `transcode_v_step`)
- Create: `crates/hipfire-runtime/examples/adaptive_kv_check.rs` (GPU test harness)

- [ ] **Step 1: `transcode_v_step`** on `KvCache`: given a target `VMode`,
  dispatch the right kernel per real KV layer (skip 1-element placeholders),
  update `self.v_mode`, and call `gpu.invalidate_for_kv_mode_switch()` once at
  the end. Use a 1-layer scratch buffer (sized to the larger of src/dst layer
  bytes) so a HIP error never leaves a half-written live buffer: copy layer→
  scratch, transcode scratch→layer (or transcode in place if alias-safe per
  Task 3). Assert `head_dim==256` and an FWHT K mode for lloyd-V (reuse the
  `set_v_mode_realloc` guard).
- [ ] **Step 2: Correctness harness** `adaptive_kv_check.rs`: load
  `qwen3.6-27b.mq4` (kv-mode fwht3), run a short prefill at V=q8, then (a)
  snapshot logits, (b) `transcode_v_step(Lloyd4)`, (c) compare next-token logits
  to a fresh load that wrote V directly at lloyd4 over the same prefill — KLD
  between the two distributions must be < a small epsilon (transcode adds only
  one extra rounding vs direct). Print the delta.
- [ ] **Step 3: Build + run (GPU-locked).**
  `source scripts/gpu-lock.sh && gpu_acquire "adaptive-kv" && \
   ./target/release/examples/adaptive_kv_check && gpu_release`.
  Expect the transcode-vs-direct KLD ≈ 0 (well under 1e-3).
- [ ] **Step 4: Coherence gate.** Build the daemon; run
  `./scripts/coherence-gate.sh` (it will exercise the changed forward path).
  Expect exit 0 / no hard errors.
- [ ] **Step 5: Commit.** `--no-verify -m "feat(kv): V transcode orchestration
  (transcode_v_step, 1-layer scratch) + correctness harness"`

---

## Task 5: Decode hook + `maybe_downshift` + minimal enable path

**Goal:** Wire the controller into the live decode loop and prove an end-to-end
V-only downshift mid-generation (force an early threshold) is attractor-free.

**Files:**
- Modify: `crates/hipfire-runtime/src/kv_adaptive.rs` (add `maybe_downshift`)
- Modify: `crates/hipfire-runtime/examples/daemon.rs` (LoadedModel field, hook,
  env enable)

- [ ] **Step 1: `maybe_downshift`** on `KvAdaptive`:
```rust
/// Call after each committed token write. If seq_pos crossed the next
/// threshold, transcode the cache for that step and advance. Returns the step
/// applied, if any.
pub fn maybe_downshift(&mut self, gpu: &mut rdna_compute::Gpu,
                       kv: &mut crate::llama::KvCache, seq_pos: usize)
    -> rdna_compute::HipResult<Option<Step>> {
    if self.next_step >= self.steps.len() { return Ok(None); }
    if seq_pos < self.thresholds[self.next_step] { return Ok(None); }
    let step = self.steps[self.next_step];
    match step {
        Step::V(nv) => { kv.transcode_v_step(gpu, nv, seq_pos)?; self.cur_v = nv; }
        Step::K(nk) => { kv.transcode_k_step(gpu, nk, seq_pos)?; self.cur_k = nk; } // Task 7
    }
    self.next_step += 1;
    Ok(Some(step))
}
```
  (For Task 5, `transcode_k_step` may be a `todo!()`/unreachable since the
  balanced V-first pattern's K step comes after the lloyd3 V step; gate the
  end-to-end test to a V-only floor so K is not exercised yet.)
- [ ] **Step 2: LoadedModel field + construction.** Add `kv_adaptive:
  Option<hipfire_runtime::kv_adaptive::KvAdaptive>` to `LoadedModel`
  (`daemon.rs:559`); default `None` in all constructors. In `load_model`, after
  the kv cache + V-mode block (`daemon.rs:~2325`), parse `HIPFIRE_KV_ADAPTIVE`
  (off|conservative|balanced|aggressive|advanced:k=..,v=..) and, when set + FWHT
  K mode, construct the controller (sizing the V buffer at the floor via a new
  `set_adaptive_floor_alloc` that allocates V at `max_seq * v_bph(v_floor)` and
  K at fwht4) and set `m.kv_adaptive`.
- [ ] **Step 3: Hook the decode loop.** In `generate`, immediately after the
  eviction block at `daemon.rs:4737`, add:
```rust
if let Some(ref mut ad) = m.kv_adaptive {
    if let Some(step) = ad.maybe_downshift(gpu, kv, m.seq_pos).unwrap() {
        eprintln!("[adaptive-kv] downshift @ pos {}: {:?} (K={:?} V={:?})",
                  m.seq_pos, step, ad.cur_k, ad.cur_v);
    }
}
```
- [ ] **Step 4: End-to-end test (GPU-locked).** Start the daemon with
  `HIPFIRE_KV_ADAPTIVE=advanced:k=fwht4,v=lloyd2` and a SMALL `max_seq` so the V
  thresholds fall within a single long generation; generate ~600 tokens crossing
  every V step. Confirm the `[adaptive-kv] downshift` lines fire at the expected
  positions and the output is fluent (eyeball + the coherence-probe detectors).
- [ ] **Step 5: Coherence gate.** `./scripts/coherence-gate.sh`. Exit 0.
- [ ] **Step 6: Commit.** `--no-verify -m "feat(kv): adaptive decode hook +
  maybe_downshift + HIPFIRE_KV_ADAPTIVE enable (V-only path validated)"`

---

## Task 6: K transcode kernels + `KMode` flip in dispatch

**Goal:** GPU kernels to re-quantize K (same-width fwht4→fwht2 remap; fwht3
re-rotation), and the flag flip so the right K attention kernel dispatches.

**Files:**
- Create: `kernels/src/kv_transcode_k_samewidth.hip` (fwht4→fwht2, 128-wide)
- Create: `kernels/src/kv_transcode_k_rerotate.hip` (fwht4↔fwht3↔fwht2)
- Modify: `crates/rdna-compute/src/kernels.rs`, `dispatch.rs` (launchers)

- [ ] **Step 1: `kv_transcode_k_samewidth`** — 128-wide. Read idx4 + cnorm
  (`TURBO_C4`), reconstruct rotated value, requant with `turbo_quantize_2bit` +
  `TURBO_C2`, recompute cnorm, write 68 B/head. No FWHT.
- [ ] **Step 2: `kv_transcode_k_rerotate`** — reconstruct normal-space K (dequant
  + inverse rotation at source width: `fwht_shfl_inverse` for 128 / `_256` for
  256) → forward rotation at target width → quantize to target LUT. Parameterized
  by `(src_bits, src_width, dst_bits, dst_width)`.
- [ ] **Step 3: Register consts + launch fns** (same recipe as Task 3).
- [ ] **Step 4: Build.** `cargo build --release -p rdna-compute; echo
  BUILD_EXIT=$?` → 0.
- [ ] **Step 5: Commit.** `--no-verify -m "feat(kv): K transcode kernels
  (fwht4→fwht2 remap + fwht3 re-rotation) + launchers"`

---

## Task 7: K transcode orchestration + hard coherence

**Goal:** `transcode_k_step` on `KvCache` (flip `quant_asym{2,4}`/`quant_fwht`
flags + adjust signs width for fwht3) and validate K transitions hard.

**Files:**
- Modify: `crates/hipfire-runtime/src/llama.rs` (`transcode_k_step`)

- [ ] **Step 1: `transcode_k_step`** — per real KV layer, dispatch the K
  transcode (same-width or re-rotate by source/target widths), update the K-mode
  booleans (`quant_asym2/3/4`, `quant_fwht`) and, when crossing 128↔256, realloc
  the signs tables to the target width (the LCG-prefix trick: 256 signs' first
  128 == 128 signs). Call `gpu.invalidate_for_kv_mode_switch()` at the end.
- [ ] **Step 2: KLD correctness** — extend `adaptive_kv_check.rs` to also verify
  K fwht4→fwht2 transcode ≈ direct fwht2 write (KLD < eps), and fwht3 re-rotation
  ≈ direct fwht3 write.
- [ ] **Step 3: Full-pattern coherence (GPU-locked, the critical gate).** Daemon
  with `HIPFIRE_KV_ADAPTIVE=balanced` + small `max_seq`; generate across ALL four
  steps (V→l4, V→l3, K→f2, V→l2). Run `coherence_probe` (attractor / special-
  token / n-gram detectors) over the full output. MUST be attractor-free at every
  transition, especially the K step. If the K transition attracts, document and
  investigate per CLAUDE.md attention-precision note (do not ship).
- [ ] **Step 4: Coherence gate.** `./scripts/coherence-gate.sh --full` (includes
  27B). Exit 0.
- [ ] **Step 5: Commit.** `--no-verify -m "feat(kv): K transcode orchestration +
  full-pattern coherence (all steps attractor-free)"`

---

## Task 8: Pattern tuning — finalize the balanced interleave via KLD sweep

**Goal:** Confirm or reorder the default `balanced` step order using the KLD
matrix harness; pick the interleave that minimizes KLD at equal byte budget.

**Files:**
- Create: `benchmarks/quality-baselines/results/2026-05-31-kv-vquant/adaptive-pattern-sweep.txt`
- Modify: `crates/hipfire-runtime/src/kv_adaptive.rs` (`balanced_steps` if reorder)

- [ ] **Step 1: Sweep (GPU-locked).** For each candidate end-state on the balanced
  descent, run `eval_hipfire --kv-mode <K> --kv-v <V> --max-chunks 24` and record
  KLD. Compare the balanced interleave's mid-states (K4/V3, K2/V3, K2/V2) against
  alternatives (e.g. K2/V4 lopsided) to confirm balanced ≤ lopsided at equal bytes
  (validates the §4 ordering). Write the table.
- [ ] **Step 2: Reorder if the data says so** (update `balanced_steps` + the
  Task 2 test expectation); else annotate "balanced confirmed".
- [ ] **Step 3: Commit.** `--no-verify -m "perf(kv): finalize adaptive balanced
  interleave via 24-chunk KLD sweep (data: <result>)"`

---

## Task 9: Wire-up — presets, advanced floor selector, env, per-load param, TUI

**Goal:** Full user-facing surface.

**Files:**
- Modify: `crates/hipfire-runtime/examples/daemon.rs` (per-load `params.kv_adaptive`)
- Modify: `cli/index.ts` (settings menu entry + per-model resolution)

- [ ] **Step 1: Per-load param.** In `load_model`, accept
  `msg.params.kv_adaptive` (string) overriding the env (mirror the `kv_mode`
  override at `daemon.rs:1905-1917`). Parse preset names + `advanced:k=..,v=..`.
- [ ] **Step 2: TUI entry.** Add `kv_adaptive` to the `cli/index.ts` settings
  menu (alongside `kv_cache`, `physical_cap`, ~line 3662-3959): off + the three
  presets + an `advanced` option exposing K-floor / V-floor pickers, with help
  text on the max_seq-as-floor contract. Validate it like the other settings;
  pass it through in the load `params`.
- [ ] **Step 3: Default OFF.** Confirm the resolved default is `off` (opt-in) so
  no behavior change for existing users.
- [ ] **Step 4: Build daemon + CLI typecheck** (`bun run -c` or the project's TS
  check). Coherence gate (default off ⇒ unchanged path). Exit 0.
- [ ] **Step 5: Commit.** `--no-verify -m "feat(kv): adaptive KV user surface —
  presets + advanced floor selector, env + per-load + TUI; opt-in default"`

---

## Task 10: DFlash hook (fast-follow, same controller)

**Goal:** Apply `maybe_downshift` at the DFlash committed-position site.

**Files:**
- Modify: `crates/hipfire-runtime/examples/daemon.rs` (`generate_dflash`)

- [ ] **Step 1: Hook** at the DFlash committed-position advance (near the
  `ev.maybe_evict` calls at `daemon.rs:3375` / `3558`), calling
  `m.kv_adaptive.maybe_downshift(gpu, &mut target.kv_cache, position)`. Handle the
  spec-decode position semantics (apply only on committed positions, not tree
  branches).
- [ ] **Step 2: DFlash coherence (GPU-locked).** `./scripts/coherence-gate-dflash.sh`
  with adaptive on + a long generation crossing steps. Must pass the 3-tier
  attractor thresholds (CLAUDE.md). DFlash perf gate uses q8 or FWHT KV.
- [ ] **Step 3: Commit.** `--no-verify -m "feat(kv): adaptive KV on DFlash decode
  path (committed-position hook)"`

---

## Task 11: Fleet hardening — gfx1201 (hiptrx) + gfx1151 (hipx)

**Goal:** Prove the transcode/re-rotation kernels are wave32-portable across the
RDNA fleet.

- [ ] **Step 1: hiptrx (R9700 gfx1201).** Create a worktree from this branch,
  build, run `./scripts/coherence-gate.sh` + a KLD spot-check
  (`--kv-mode fwht3 --kv-v lloyd4 --max-chunks 8`). Resolve any arch-specific
  failures (wave32 assumptions, LDS sizing). Record results.
- [ ] **Step 2: hipx (Strix Halo gfx1151).** Same. (Note hardware caveats from
  `user_hardware_hipx` memory.)
- [ ] **Step 3: Commit** any portability fixes. `--no-verify -m "fix(kv): fleet
  portability — adaptive transcode kernels on gfx1201/gfx1151 (wave32)"`

---

## Task 12: Docs, memory, NEXT-STEPS, PR

- [ ] **Step 1: Update the design doc** with the final pattern, measured
  transcode cost, and any deviations.
- [ ] **Step 2: Write `NEXT-STEPS.md`** (or update) — adaptive-KV follow-ups
  (default-on decision, multi-GPU, K-floor=fwht3 perf, recency-tiered precision).
- [ ] **Step 3: Update memory** (`project_hipfire_adaptive_kv`,
  `project_hipfire_vquant_status`) — adaptive shipped, results.
- [ ] **Step 4: Final review** — dispatch a code-quality reviewer over the whole
  diff; address issues.
- [ ] **Step 5: Open the PR** via `gh pr create` (base = the branch PR #366 is
  stacked on, or main per the stacking) with summary + test plan + the
  KLD/coherence/perf/fleet evidence.

---

## Self-review notes

- **Spec coverage:** §3 capacity → Task 1; §4 pattern + §4.1 advanced → Task 2 +
  Task 9; §5.1 KMode → Task 1; §5.2 controller → Task 2/5; §5.3 transcodes →
  Tasks 3/4/6/7; §5.4 hook → Task 5; §5.5 surface → Task 9; §7 graph risk →
  Task 0; §8 errors → Tasks 4/7 (1-layer scratch, guards); §9 default+tuning →
  Tasks 8/9; §10 testing → every task's gate; §11 sequence → task order; DFlash →
  Task 10; fleet → Task 11.
- **No literal-PASS coherence:** gate = exit 0 + "no hard errors" + human eyeball
  of the report (the controller agent does the eyeball).
- **In-place aliasing:** Task 3/6 kernels assume dst stride < src stride (always
  true on downshift) and ascending position iteration; documented per kernel.
