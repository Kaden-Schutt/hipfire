# C8 GPU Sampler Scope — Chain temp>0 Full-Probs D2H Elimination

**Date:** 2026-06-29  
**Branch:** feature/speculator-ddtree  
**Status:** READ-ONLY SCOPE — no implementation  
**Roadmap entry:** Stage 2 / C8 in `gpu-resident-spec-roadmap.md`

---

## 1. The Current temp>0 Chain Accept Path

### 1a. What happens under FAST_SAMPLE (default ON, production path)

Flag: `gpu.flags.dflash_fast_sample` = `true` by default  
(`crates/rdna-compute/src/feature_flags.rs:310`, `HIPFIRE_DFLASH_FAST_SAMPLE != "0"`)

Active whenever: `use_temp_sampling && gpu.flags.dflash_fast_sample`  
(`speculative.rs:3053`)

**Draft side (step 4b, one call per cycle):**

```rust
// speculative.rs:3316–3371
if use_temp_sampling && fast_sample_active {
    let probs_gpu = gpu.alloc_tensor(&[batch * vocab], DType::F32)?;
    if topp_active {
        // GPU softmax + nucleus (tau_cut, Z per row)
        gpu.softmax_temp_topp_batched_into_f32(&logits_batch, &probs_gpu,
            &tau_gpu, &z_gpu, vocab, batch, temp, top_p, top_k, 0.0)?;
        let tau = gpu.download_f32(&tau_gpu)?;   // b×4 bytes — tiny
        let z   = gpu.download_f32(&z_gpu)?;     // b×4 bytes — tiny
    } else {
        gpu.softmax_temp_batched_into_f32(&logits_batch, &probs_gpu, vocab, batch, temp)?;
    }
    let host_probs = gpu.download_f32(&probs_gpu)?;  // ← C8a: batch×vocab×4 ≈ 9 MB
    for i in 0..batch {
        let mut probs = host_probs[i*vocab..(i+1)*vocab].to_vec();
        if topp_active { apply_topp_trunc(&mut probs, tau[i], z[i]); }
        let u = xorshift_next_unit(rng_state);
        let t = sample_categorical(&probs, u);
        draft_probs_at_drafted.push(probs[t as usize]);
        drafted.push(t);
        draft_softmaxes.push(probs);   // ← stores FULL vocab-sized vec
    }
}
```

**Verify side (step 7, one call per cycle):**

```rust
// speculative.rs:3596–3631
let fast_tgt_probs: Option<Vec<f32>> = if fast_sample_active {
    let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
    let probs_gpu = gpu.alloc_tensor(&[b * vocab], DType::F32)?;
    if topp_active {
        gpu.softmax_temp_topp_batched_into_f32(&logits_batch, &probs_gpu,
            &tau_gpu, &z_gpu, vocab, b, temp, top_p, top_k, 0.0)?;
        fast_tgt_tau = Some(gpu.download_f32(&tau_gpu)?);  // b×4 bytes — tiny
        fast_tgt_z   = Some(gpu.download_f32(&z_gpu)?);    // b×4 bytes — tiny
    } else {
        gpu.softmax_temp_batched_into_f32(&logits_batch, &probs_gpu, vocab, b, temp)?;
    }
    let host = gpu.download_f32(&probs_gpu)?;  // ← C8b: b×vocab×4 ≈ 9 MB
    Some(host)
};
```

**Accept loop (step 7, sequential over b rows):**

```rust
// speculative.rs:3640–3722
for i in 0..b - 1 {
    target_probs = fast_tgt_probs[i*vocab..(i+1)*vocab].to_vec();
    if topp_active { apply_topp_trunc(&mut target_probs, tau[i], z[i]); }
    let t = block[i + 1] as usize;
    let p_d = draft_probs_at_drafted[i];        // scalar — from draft side
    let p_t = target_probs[t];                  // scalar — target prob at drafted token
    // CACTUS bump (optional):
    let accept_prob = p_t + sqrt(2*delta*p_t*(1-p_t));  // delta=0 → plain p_t
    let u = xorshift_next_unit(rng_state);      // host xorshift64*
    if u * p_d <= accept_prob {
        accept_len += 1;
    } else {
        // CACTUS: revise target_probs in-place → h distribution
        let u2 = xorshift_next_unit(rng_state);
        // sample_residual needs FULL target_probs AND FULL draft_softmaxes[i]
        rejected_bonus = Some(sample_residual(&target_probs, &draft_softmaxes[i], u2));
        break;
    }
}
// All-accepted: bonus from target_softmax[b-1], full prob vector needed
```

### 1b. What the accept step actually needs from the full prob vectors

The accept loop needs per-position:

| Datum | Where used | Size |
|-------|-----------|------|
| `target_probs[t]` = p_t at the drafted token | Acceptance ratio numerator | 1 scalar |
| `draft_probs_at_drafted[i]` = p_d at the drafted token | Acceptance ratio denominator | 1 scalar (already on host, tiny) |
| **Full `target_probs[0..vocab]`** | `sample_residual` on REJECTION (renorm `(p_t - p_draft)+`) | vocab f32s |
| **Full `draft_softmaxes[i][0..vocab]`** | `sample_residual` second argument | vocab f32s |
| **Full `target_probs[b-1][0..vocab]`** | All-accepted: bonus token sample | vocab f32s |

**The blocker for a full GPU-resident accept:** `sample_residual` at  
`speculative.rs:3696` requires BOTH the full target and full draft probability  
vectors. The draft full-probs are already stored on-host in `draft_softmaxes`  
(a `Vec<Vec<f32>>` of length `batch`). Without also keeping draft probs on-device,  
the bonus computation on rejection requires a round-trip for both sides.

There are two distinct strategies:

**Strategy A — Compact D2H only (simpler, immediate win):**  
Keep the acceptance decision on-device, but only move the minimum to the host:  
per-position `{target_p_at_drafted_token, draft_p_at_drafted_token, accept_bool,  
bonus_token_if_rejected}`. The CACTUS residual sampling and bonus draws happen  
on-GPU. D2H becomes `b × (2 f32 + 1 i32) ≈ 48 bytes` for the compact decision  
struct, plus `1 × u32` for the bonus token. Full prob vectors never cross the bus.

**Strategy B — GPU-only accept kernel (full GPU-resident):**  
A fused kernel that takes `probs_tgt[b×vocab]` + `probs_draft[b×vocab]` on-device,  
runs the entire accept loop (including CACTUS, `sample_residual`, bonus), and writes  
`{accept_len, bonus_token, rejected_at}` to a small output buffer. Eliminates BOTH  
the 9 MB downloads. Requires keeping `draft_softmaxes` on-device too.

Currently `draft_softmaxes` is a `Vec<Vec<f32>>` built on-host from the  
`host_probs` D2H (line 3370). Strategy B needs `draft_softmaxes` to stay in a  
device buffer `[batch × vocab]` (never downloaded). This is the structural change.

### 1c. What the DDTree SWOR path downloads (for contrast)

The SWOR walk (`swor_walk_gpu`, `speculative.rs:4272–4338`) downloads only  
`(2 + num_pos) × 4 ≈ 68 bytes` via:

```rust
let raw = gpu.download_f32(&t_out)?;  // speculative.rs:4325
// out[0]=accept_len, out[1]=bonus_token, out[2+i]=accepted child node index
```

The full target logits (`verify_scratch.logits`, `n_slots × vocab × 4 ≈ 37 MB`)  
and full draft logits stay device-resident. The SWOR walk kernel computes the  
entire accept decision, residual sampling, and bonus selection on-GPU, yielding  
only the 68-byte committed-path struct. This is the gold standard the chain  
temp>0 path should match.

---

## 2. Existing GPU Sampling Kernels

### 2a. Single-row sampler: `sample_top_p_pf` / `sample_top_p_parallel_impl`

**Location:** `crates/rdna-compute/src/sampling.rs:185–459`

**Signature (Rust dispatch side):**
```rust
pub fn sample_top_p_pf(
    &mut self,
    logits: &GpuTensor,      // [vocab] f32, in-place modified for penalty
    result_buf: &GpuTensor,  // [2] u32 output: [token_id, new_rng_state]
    repeat_buf: &GpuTensor,  // [repeat_window] u32 recent tokens
    vocab_size: usize,
    temperature: f32,
    top_p: f32,
    rng_state: u32,          // HOST-SIDE xorshift32 (LCG in AR kernel)
    repeat_window: usize,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
) -> HipResult<(u32, u32)>   // (token_id, new_rng_state) — 8-byte D2H
```

**RNG convention in AR kernel:** LCG `s = s * 1664525 + 1013904223`; extract  
`(s >> 8) / 2^24`. This is the same LCG used in `ddtree_swor_walk.hip` lines  
83–84/124–125/167–168. The AR path receives `rng_state: u32` from the HOST and  
returns the new u32 state.

**Spec-decode host RNG:** `xorshift_next_unit` (`speculative.rs:2066`) is a  
**different algorithm** — xorshift64* (`s ^= s<<13; s ^= s>>7; s ^= s<<17`),  
extracting `(s >> 40) / 2^24` from a `u64` state. It produces a `u32`-width  
uniform but from a different PRNG family than the AR kernel.

**What it outputs:** result `[0]` = token id (u32), result `[1]` = new rng state  
(u32). D2H: 8 bytes synchronously.

**3-phase parallel variant** (`sample_top_p_parallel_impl`, lines 292–458):  
1. Penalty prepass (optional, in-place on logits)  
2. `sample_topk_partial`: 128 blocks × 256 threads, computes partial top-K  
3. `sample_topk_finalize`: 1 block × 256 threads, merges → final top-K, softmax, top-p, RNG sample

**The gap to batched:** `sample_top_p_pf` handles **one row**. It takes a single  
`[vocab]` logits tensor and one rng_state. For spec-decode we need it over `b`  
rows of `logits_batch[b × vocab]`, producing per-row `(token_id, prob_at_token)`.

Critical gap: the AR kernel outputs `new_rng_state`, but the spec-decode  
accept loop needs the **probability at the drawn token** (not just the token id)  
for `p_d` in the acceptance ratio, and for `sample_residual` the **full  
re-normalized prob vector** on rejection. The existing AR kernel does NOT output  
either. New outputs are required.

### 2b. Batched softmax kernels (already exist)

- `softmax_temp_batched_f32`: `kernels/src/softmax_temp_batched.hip:27`, grid  
  `[rows, 1, 1]`, block `[min(256, vocab), 1, 1]`, one block per row. Outputs  
  full `[rows × vocab]` probs.
- `softmax_temp_topp_batched_f32`: same grid, also emits `tau_cut[rows]` and  
  `Z[rows]` for nucleus truncation.

These are the current building blocks for C8a/C8b. They produce full probs on-device  
but then the D2H of those probs is what we want to eliminate.

### 2c. SWOR walk kernel (already exists, shows the target pattern)

`ddtree_swor_walk_f32` (`kernels/src/ddtree_swor_walk.hip`): one block of 256  
threads, sequential depth loop, block-parallel vocab sweeps. RNG: per-thread-0  
LCG state `s_rng`, seeded from `seed` (a `u64` caller-side, passed as `u32`  
after `seed | 1`). Outputs `[2 + num_pos]` i32 = 68 bytes max.

---

## 3. Batched Categorical GPU Sampler Kernel Design

### 3a. What the chain accept loop actually requires

The accept loop is NOT a simple "sample one token per row" problem. It has  
two distinct sub-cases:

**Case 1 — Accept (probability p_t/p_d ≥ u):** Only `target_p_at_token_t` is needed.  
No full vectors needed for the accept decision itself.

**Case 2 — Reject (probability 1 - p_t/p_d):** Need to sample from  
`normalize(relu(target_probs - draft_probs))`. This requires BOTH full vectors.  
After rejection, the loop breaks; all subsequent positions are not examined.

**Case 3 — All accepted:** Need to sample from `target_probs[b-1]`. One full  
target prob vector needed.

The fact that rejection breaks the loop makes this inherently sequential: position  
`i+1` is only examined if position `i` accepted. This is NOT naively parallelizable  
across positions — it is a dependent chain.

### 3b. Two-kernel approach (recommended)

**Kernel 1: `chain_accept_spec_f32`**

One block of 256 threads. Receives on-device:
- `target_probs_dev[b × vocab]` — from `softmax_temp_batched_f32`
- `draft_probs_dev[b × vocab]` — draft side probs (NEW: kept on-device)
- `draft_p_at_token[b]` — scalar per position, from the draft sample (tiny)
- `draft_tokens[b]` — the drafted token ids (already on-device as `logits_batch`-adjacent argmax; for temp>0 these come from the GPU sample)
- `target_tau[b], target_z[b]` — from `softmax_temp_topp_batched_f32` (optional)
- `draft_tau[b], draft_z[b]` — from draft-side softmax (optional)
- RNG seed (u32 or u64)
- CACTUS delta (f32, 0.0 = disabled)

Outputs (device buffer `out[4]`):
```
out[0]: accept_len (i32)
out[1]: bonus_token (u32)
out[2]: rejected_at (i32, -1 if all accepted)
out[3]: new_rng_state (u32)
```

D2H: 16 bytes synchronously.

**Inner loop (sequential — thread 0 drives, block-parallel where possible):**

```
for i in 0..b-1:
    // apply topp truncation in-place on target_probs row i (parallel sweep)
    // read p_t = target_probs[i * vocab + draft_tokens[i]]  (single read, thread 0)
    // read p_d = draft_p_at_token[i]  (thread 0)
    // compute accept_prob = p_t  (or CACTUS: p_t + sqrt(2*delta*p_t*(1-p_t)))
    // rng step; u = LCG(seed)
    // if u * p_d <= accept_prob: accept_len++; continue
    // else: REJECTION
    //   apply CACTUS h-distribution in-place on target row i (parallel sweep)
    //   draw bonus from residual: parallel compute sum(max(tgt[j]-dft[j],0))
    //                             parallel cumsum to find sample point
    //   write accept_len, bonus, rejected_at, new_rng; break
// if no rejection: bonus = sample from target_probs[b-1] row
//   apply topp on row b-1, draw categorical, write bonus
```

**Block/grid mapping:**
- Grid: `[1, 1, 1]` — one block for the whole `b` positions (b ≤ 31 for  
  spec budget=30). The sequential accept chain is the critical path; width  
  comes from block-parallel vocab sweeps per position.
- Block: 256 threads. Per-row vocab sweep: each thread handles `vocab/256` elements.  
- LDS: one `float red[256]` for block reductions, one `float q_row_lds[256]` staging  
  buffer for the draft row during the residual compute (to avoid repeated global reads).  
  Total LDS ≈ 2 KB — well within 64 KB.

**Top-p in-kernel:** After reading the full probs row for residual computation,  
apply `apply_topp_trunc` equivalent in-kernel using the pre-computed `tau[i], z[i]`  
from the existing `softmax_temp_topp_batched_f32` output. This is a conditional  
scale that can be fused into the residual sweep: `p_eff[j] = (p[j] >= tau) ? p[j]/z : 0`.

**Precision:** The residual `sum(max(p_tgt[j]-p_dft[j], 0))` is a block reduction  
over 152K elements. Use float32 throughout; accumulate in registers not LDS to  
avoid catastrophic cancellation. The SWOR walk uses the same approach (`p_res`  
accumulation in `red[]`). The attractor risk is in the RNG matching, not arithmetic  
(see §6).

### 3c. Draft prob vector: keep on-device

Currently `draft_softmaxes` is built host-side from the 9 MB D2H:
```rust
// speculative.rs:3362–3370
let host_probs = gpu.download_f32(&probs_gpu)?;   // ← C8a: eliminate this
for i in 0..batch {
    let mut probs = host_probs[i*vocab..(i+1)*vocab].to_vec();
    ...
    draft_softmaxes.push(probs);  // full vocab per row
}
```

To eliminate C8a, the draft probs must remain in a persistent device buffer  
`draft_probs_dev[batch × vocab]`. The draft sample is then a GPU kernel that:  
1. Takes `probs_gpu[batch × vocab]` (already produced by `softmax_temp_batched`)  
2. Draws one sample per row (one block per row, using the GPU categorical sampling  
   pattern from `sample_top_p_pf`)  
3. Writes `draft_tokens[batch]` (i32) and `draft_p_at_token[batch]` (f32) — tiny  
4. Leaves `probs_gpu` intact for `chain_accept_spec_f32` to use as `draft_probs_dev`

This is **Kernel 0: `batched_categorical_sample_f32`**:

```
grid: [batch, 1, 1] — one block per draft position
block: [256, 1, 1]
inputs: probs[batch×vocab] (already softmaxed), tau[batch], z[batch], seed(u32)
outputs: tokens[batch] (i32), prob_at_token[batch] (f32)
D2H: batch×8 bytes ≈ 120 bytes for batch=15
```

This kernel is simpler than `chain_accept_spec_f32` — each block is independent.  
RNG convention: per-block LCG seeded with `seed ^ blockIdx.x`. Sequential draw  
within the block using `p_eff[j] = (p[j] >= tau) ? p[j]/z : 0` and an LDS-backed  
CDF scan.

---

## 4. Accept-Path Refactor: Chain temp>0 Fully GPU-Resident

### 4a. Current call sequence (FAST_SAMPLE default)

```
Step 4a: DFlash draft forward → logits_batch[b × vocab] on device
Step 4b: softmax_temp_batched_into_f32(logits_batch → probs_gpu)
Step 4b: download_f32(probs_gpu) → host_probs  ← C8a: ~9 MB D2H
Step 4b: CPU: sample_categorical × batch → drafted[], draft_softmaxes[], draft_probs_at_drafted[]
Step 5: verify_dflash_block(want_full_logits=false, ...) → verify_scratch.logits on device
Step 7: softmax_temp_batched_into_f32(verify_scratch.logits → probs_gpu)
Step 7: download_f32(probs_gpu) → fast_tgt_probs  ← C8b: ~9 MB D2H
Step 7: CPU accept loop (sequential over b positions)
```

### 4b. Proposed call sequence (fully GPU-resident)

```
Step 4a: DFlash draft forward → logits_batch[b × vocab] on device
Step 4b: softmax_temp_batched_into_f32(logits_batch → draft_probs_dev)  [unchanged]
         softmax_temp_topp_batched_into_f32(...) only if topp_active     [unchanged for tau/Z]
Step 4b: [NEW] batched_categorical_sample_f32(
             draft_probs_dev, tau_dev, z_dev → draft_tokens_dev, draft_p_at_token_dev)
         download 2 × batch × 4 bytes ← replaces 9 MB C8a
         host sets block[] from draft_tokens (as before)
Step 5: verify_dflash_block(..., want_full_logits=false) → verify_scratch.logits on device
Step 7: softmax_temp_batched_into_f32(verify_scratch.logits → tgt_probs_dev)  [unchanged]
         softmax_temp_topp_batched_into_f32(...) only if topp_active     [unchanged for tau/Z]
Step 7: [NEW] chain_accept_spec_f32(
             tgt_probs_dev, draft_probs_dev, draft_tokens_dev, draft_p_at_token_dev,
             tau_dev, z_dev, rng_seed, cactus_delta →
             out[accept_len, bonus_token, rejected_at, new_rng])
         download 16 bytes ← replaces 9 MB C8b
         host: reads accept_len, bonus_token, updates rng_state
```

**Total D2H for temp>0 chain: ~136 bytes** instead of ~18 MB (C8a + C8b).

### 4c. Files to change

1. **`crates/hipfire-arch-qwen35/src/speculative.rs`**:
   - Lines 3316–3370: Replace `download_f32(&probs_gpu)` + CPU sampling loop  
     with `batched_categorical_sample_f32` launch + tiny D2H of tokens/probs.  
     Keep `probs_gpu` alive (rename `draft_probs_dev`) — do NOT free it here.
   - Lines 3596–3631: Replace `download_f32(&probs_gpu)` with `tgt_probs_dev`  
     tensor (keep on-device).
   - Lines 3583–3722: Replace the entire `if use_temp_sampling { ... }` accept  
     block with `chain_accept_spec_f32` launch + 16-byte D2H.
   - `draft_softmaxes: Vec<Vec<f32>>` type → no longer needed; remove.
   - `fast_tgt_probs: Option<Vec<f32>>` → remove.

2. **`crates/rdna-compute/src/sampling.rs`**:
   - Add `batched_categorical_sample_f32` dispatch method.
   - Add `chain_accept_spec_f32` dispatch method.

3. **`kernels/src/` (two new files)**:
   - `chain_accept_spec.hip` — the fused accept kernel.
   - Optionally extend `softmax_temp_batched.hip` with a `batched_categorical_sample`  
     entry point fused after softmax (saves a second kernel launch).

4. **`crates/rdna-compute/src/kernels.rs`**:
   - Add `CHAIN_ACCEPT_SPEC_SRC: &str = include_str!(...)` and  
     `BATCHED_CATEGORICAL_SAMPLE_SRC: &str = include_str!(...)`.

---

## 5. Distribution-Preserving Validation Plan

### 5a. RNG-match feasibility assessment

**Host path RNG:** `xorshift_next_unit` (speculative.rs:2066) — xorshift64\*  
with `u64` state, extracting `(s >> 40) / 2^24`.  
**SWOR walk kernel RNG:** LCG `s = s * 1664525 + 1013904223`, extracting  
`(s >> 8) / 2^24`.

These are incompatible algorithms. **Byte-exact RNG match is infeasible** for  
the chain GPU sampler — it would require either:
- Rewriting the host path to use the kernel's LCG, OR
- Implementing xorshift64\* in the GPU kernel and passing the u64 host state in

The xorshift64\* on GPU is practical (it is a 3-instruction sequence), but it requires  
the kernel to carry a full u64 seed and return the updated u64 state, which does  
not fit the `u32 rng_state` convention used throughout the existing AR kernel.

**Recommended approach:** Accept distribution-parity (not byte-parity), exactly  
as `HIPFIRE_DFLASH_FAST_SAMPLE` does today for the softmax computation  
(speculative.rs:3046–3051: "distribution-only, NOT byte-identical ... validated  
coherent across genres, so default-on"). The GPU sampler will use the LCG (matching  
the SWOR walk convention), seeded from the host `rng_state` cast to u32. The  
host `rng_state` is advanced by `b` draws after the kernel (consuming the same  
number of draws as the CPU loop would have).

**Consequence:** At the session/token level, output token sequences will differ  
from the host path at the same seed. This is already true of FAST_SAMPLE (the  
comment at line 3051 explicitly acknowledges "can differ at the last ULP and  
rarely flip a borderline accept"). The distribution is provably identical (both  
draw from `Categorical(probs_row_i)` with the same probs), so τ is unchanged.

### 5b. Validation steps

1. **Temp=0 regression:** At temp=0, `use_temp_sampling` is false and this  
   entire path is inactive. No change. Greedy path: byte-identical by  
   construction. Gate: existing `coherence-gate.sh`.

2. **MC total-variation test (temp=0.7):**  
   Method from `ddtree-naive-sampling-verify` memory entry: draw N=10000 samples  
   from the host path and N=10000 from the GPU kernel over the same fixed  
   `probs_gpu` vector, compute empirical TV distance. Target: TV < 0.01.  
   Implementation: a test binary that generates a fixed random prob vector,  
   calls both paths, histograms the outputs. No GPU required for the host side.

3. **Coherence gate (attractor check):**  
   `./scripts/coherence-gate-dflash.sh` with temp=0.7. The three-tier attractor  
   detector (unique-token-ratio on first/last 128, 3gram density) will catch any  
   distribution collapse introduced by the GPU sampler.

4. **Serve-multiturn gate:**  
   `./scripts/serve-multiturn-gate.sh` — cross-request state-bleed check. The  
   GPU sampler adds no new state to the model; this is a regression guard for  
   the daemon's RNG advance bookkeeping (ensure `rng_state` is advanced  
   consistently after each GPU-sampled cycle).

5. **Distribution-preserving check for CACTUS:**  
   When `cactus_delta > 0`, the h-distribution construction must be verified.  
   The CACTUS modification of `target_probs` in the GPU kernel is a deterministic  
   transformation (no sampling); only the final `sample_residual` draw introduces  
   randomness. Validate: run CACTUS acceptance and compare bonus distribution  
   host vs GPU (MC TV < 0.01 at delta=1.0).

6. **Attractor check at temp=0.7 after 200 tokens:**  
   spec-decode bench on the canonical 27B-3.5 LRU code prompt with  
   `--temperature 0.7 --max 256`. Record unique-token-ratio and max-single-token-freq  
   on the first and last 128 tokens. Compare τ to baseline within ±5%.

### 5c. CACTUS in-kernel feasibility

The CACTUS h-distribution rewrite (speculative.rs:3679–3693) is a vocab-sized  
in-place transformation: `probs[j] = if j==t { gamma_star } else { scale * probs[j] }`.  
This is trivially parallelizable in a block-parallel kernel sweep — each thread  
handles its `vocab/256` elements. The degenerate case (`qn >= 1 - 1e-6`) is  
handled by thread 0 writing the one-hot before the sweep. No new algorithmic risk.

---

## 6. Effort Estimate and Risks

### 6a. Effort

| Task | Effort | Notes |
|------|--------|-------|
| `chain_accept_spec.hip` kernel | M | ~200 lines; sequential accept loop + block-parallel vocab sweeps |
| `batched_categorical_sample_f32` | S | ~80 lines; one block per row, fuse into existing softmax hip or new file |
| Rust dispatch wrappers (sampling.rs) | S | Pattern-copy of existing dispatch methods |
| speculative.rs accept-path refactor | M | Remove `draft_softmaxes`, `fast_tgt_probs`, replace with device tensors |
| MC TV test + attractor validation | S | Script using existing gate infrastructure |
| coherence-gate-dflash + serve-multiturn | S | Standard gates; run as-is |

Total: **M** (roughly 2–4 engineer-days including validation).

### 6b. Risks

**Risk 1 — Precision/attractor risk (CRITICAL).**  
Per `feedback_attention_precision.md` (referenced in CLAUDE.md): 5% attention  
error cascades to an attractor within ~10 tokens under greedy decode. For sampling,  
the risk is subtler: `sample_residual` on rejection computes  
`sum(max(p_tgt[j] - p_draft[j], 0))` as a block-parallel reduction. If the  
reduction accumulates in a different order than the host sequential sum, borderline  
elements (`p_tgt[j] ≈ p_draft[j]`) may flip the sign, shifting probability mass  
to different tokens. This is the same ULP-flip risk as fast_sample, and it is  
ALREADY accepted for the softmax step. The MC TV test (§5b step 2) at N=10000  
will catch any systematic shift.

**Risk 2 — RNG advance bookkeeping.**  
The host `rng_state` (u64 xorshift) must be advanced by `b` steps after each  
GPU-sampled cycle to maintain a consistent stream across cycles. If the advance  
count is wrong (off-by-one in the number of draft samples vs accepted positions),  
the sequence will drift relative to temperature-sensitive prompts. Validate by  
comparing rng_state after N cycles between host-sampled and GPU-sampled code paths.

**Risk 3 — top_k / min_p parity.**  
The existing `softmax_temp_topp_batched_f32` already handles top_k and min_p via  
the histogram nucleus. The `chain_accept_spec_f32` kernel must apply the SAME  
`apply_topp_trunc` logic (threshold on `tau[i]`, renorm by `z[i]`) consistently  
for both the draft probs (C8a side) and target probs (C8b side). Mismatched  
nucleus application between draft and target sides shifts the acceptance ratio  
away from the theoretical distribution. This is a correctness requirement, not  
just a precision concern.

**Risk 4 — `draft_probs_dev` lifetime.**  
The draft probs tensor (`probs_gpu`, renamed `draft_probs_dev`) must live from  
step 4b through step 7 (verify + accept). Currently it is freed immediately after  
the D2H at line 3358. The refactor extends its lifetime across `verify_dflash_block`,  
which may allocate large temporaries in the same pool. On a 96 GB UMA box (gfx1151)  
this is not an issue; on a 24 GB dGPU (gfx1100) with a 27B model, VRAM headroom  
is approximately 2–3 GB during inference. `draft_probs_dev` at `batch × vocab × 4 =  
15 × 152064 × 4 ≈ 9 MB` is well within headroom. No VRAM risk.

**Risk 5 — SWOR walk vs chain accept consistency.**  
The DDTree SWOR path and the chain temp>0 path now both use GPU sampling, but  
with different kernels and RNG states. Ensure the two paths are not accidentally  
cross-seeded or share a RNG buffer. They are separate code paths with separate  
device tensor allocations; no structural risk, but worth an explicit check.

---

## 7. Summary: What Must Leave the Device (Irreducible Minimum)

After the full C8 elimination:

| Data | Bytes | Why |
|------|-------|-----|
| `draft_tokens[batch]` | batch×4 ≈ 60 B | CPU needs drafted[] for block[] and n-gram override |
| `draft_p_at_token[batch]` | batch×4 ≈ 60 B | Needed for D2H only if not merged into accept kernel |
| Accept result `{accept_len, bonus, rejected_at, new_rng}` | 16 B | CPU needs for streaming, EOS, RNG advance |
| Draft argmax D2H (temp=0 path, unchanged) | batch×4 ≈ 60 B | Greedy path, not affected |

**Total irreducible D2H per cycle: ~136 bytes** (vs ~18 MB today for the  
FAST_SAMPLE chain temp>0 path). This matches the DDTree SWOR pattern (68 bytes)  
in order of magnitude.

The draft tokens D2H (60 B) can optionally be folded into the  
`batched_categorical_sample_f32` output and downloaded together with  
`draft_p_at_token`, giving a single 120-byte D2H for the draft side. The accept  
kernel result is always a separate 16-byte D2H.
