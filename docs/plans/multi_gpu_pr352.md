# Port PR #352 Device-Token-Chain + GPU-Accept to Multi-GPU MTP Path

**Status:** Plan (review-amended — see Revision Log below)
**Author:** Agent (reviewed by maintainer)
**Date:** 2026-06-01
**Branch:** `merge/master-pr352` (has #352 merged)
**Target function:** `spec_step_mtp_compressed_serial_multi` (`mtp_spec.rs:3607`)
**Companion benchmark:** `scripts/bench-ppmtp-split.sh`

### Revision Log

- 2026-06-01 initial draft
- 2026-06-01 consolidated review
  (self + Claude + Gemini; see `multi_gpu_pr352_plan_rev_glm5.md`). Applied:
  - §1: objective now matches §10 recommendation (delete hetero, not
    unify).
  - §2/§9: clarified that §2 validates kernels on-arch, not the port
    itself; port validated by §8 coherence + A/B.
  - §5: updated hetero scope to match §10.
  - §7 Step 2: added lockstep note for eligibility gate reuse.
  - §7 Step 5: added `verify_argmax` aliasing assumption comment note.
  - §8.2: replaced single-GPU regression with ported-`_multi` token-
    equality test.
  - §8.4: added exit criterion for legacy env-gated branch.
  - §9: added risk entry for hetero deletion losing demo + DIFF
    instrumentation.

---

## 1. Objective

Port the three PR #352 single-GPU MTP optimizations into the multi-GPU
(PpMtp) spec-step function, **and delete the dead `_hetero` spec
stepper** to eliminate code duplication. The goal is a data-driven
go/no-go on whether async PP+MTP decode can close the gap between
PpMtp (~14.2 tok/s) and pp2-AR (17.6 tok/s), or whether PpMtp remains
a long-ctx capacity play only.

### Why this matters

The pre-#352 single-GPU MTP uplift was 1.15× over AR (devlog 2026-05-29).
After #352's device-chain landed on the single-GPU path, that moved to
~1.18× — modest, confirming the devlog's hypothesis that gfx906 compute
throughput is the binding constraint. However, the PP+MTP path
(`spec_step_mtp_compressed_serial_multi`) did NOT receive any of #352's
optimizations. Its K-step draft chain still does per-step host
roundtrips (memcpy_dtoh per argmax, host-side vocab_map lookup, host-side
greedy accept). This plan ports those optimizations to measure their
combined effect on the PP cycle.

---

## 2. gfx1031 validation — already run

The critical question was whether #352's GPU kernels
(`argmax_token_chain_f32`, `greedy_accept_from_argmax_i32`,
`embed_device_token_into`) work on gfx1031 (RDNA2), since that's where
the MTP head lives in the PP+MTP hetero setup.

### Test setup

Extracted a 9B MTP head from the Qwen3.5-9B BF16 safetensors:

```bash
./target/release/mtp_extract \
  --hf-dir /local/hipfire/Qwen3.5-9B-BF16-st/ \
  --output /local/hipfire/qwen3.5-9b.mtp \
  --quant mq4
```

The 9B trunk (5 GB MQ4) + MTP head (123 MB) fits entirely on gfx1031
(12 GB), enabling a single-GPU MTP test purely on the RDNA2 card.

### Results (3 runs each, max_tokens=64)

| Config | decode tok/s | τ | cycles |
|---|---|---|---|
| Legacy (chain=0) gfx1031 | 55.8 | 2.62 | 24 |
| Device chain (chain=1) gfx1031 | 55.9 | 2.62 | 24 |
| Device chain + GPU accept gfx1031 | 55.9 | 2.62 | 24 |
| Legacy (chain=0) gfx906, 27B | 21.5 | 2.82 | 11 |
| Device chain (chain=1) gfx906, 27B | 21.5 | 2.82 | 11 |

**All three optimizations are performance-neutral on both cards.** The
host-sync overhead that #352 eliminates (K × ~30 µs D2H roundtrips per
cycle) is negligible when GPU compute is the binding constraint. At K=3
on gfx1031, the 3 × 30 µs = 90 µs saved is <0.5% of the ~18 ms cycle.

**Correctness is confirmed:** identical τ, identical cycle counts, no
errors. The kernels compile and run correctly on gfx1031 (RDNA2).

> **Review note:** §2 validates the #352 kernels on the gfx1031 arch
> via the single-GPU `_serial` path. It does **not** validate the port
> into `_multi` — the new helper (`embed_device_token_into_drafter`),
> the new state fields, and the new chain wiring are untested at this
> point. The port itself is validated by §8 coherence + A/B benchmark.

### Implication for the plan

The port is still worth doing for:

1. **Architectural consistency** — having three near-identical spec-step
   functions (`_serial`, `_hetero`, `_multi`) with different chain
   implementations is a maintenance burden. Unifying on the device-chain
   pattern makes future changes (sampling, p_min, higher K) apply once.
2. **The unknown PP-specific interaction** — the PP cycle has additional
   sync points (boundary copies, rollback replay) that may interact with
   the host-sync pattern differently. Worth measuring the actual PpMtp
   delta even though single-GPU is neutral.
3. **Forward-compat** — at higher K (K=5+) or on faster GPUs, the host-
   sync fraction grows and the device chain starts to matter. Unifying
   now avoids a later port under time pressure.

But the expected decode-speed improvement from the port alone is **near
zero** on current hardware. The plan should be evaluated as a cleanup +
measurement exercise, not a performance play.

---

## 3. What #352 added (single-GPU only)

PR #352 (`ed7b9656`) introduced three optimizations to
`spec_step_mtp_compressed_serial` (the single-GPU spec stepper):

### A. Device-resident token chain

**Before:** Each of K chain steps does argmax → D2H 4 B → host vocab_map
lookup → push to host Vec. K D2H syncs per cycle.

**After:** A GPU-side `[max_n + 1]` i32 tensor (`mtp_token_chain`) holds
the chain in-place. `argmax_token_chain_f32` performs argmax + vocab_map
remap + writes the result into the chain's next slot — entirely on-device.
Only one bulk D2H (max_n × 4 B) at cycle end to harvest candidates.

### B. GPU-side greedy accept

**Before:** Full verify argmax D2H (n_verify × 4 B) → host-side
elementwise comparison against candidates → count accepted, pick bonus.

**After:** `greedy_accept_from_argmax_i32` runs a single-thread GPU scan
over device-resident argmax + candidate arrays. Returns exactly 2 ints
(accept_count + bonus token) → 8 B D2H.

### C. Device-token embedding

**Before:** Host knows the token id from argmax → passes it as a u32
parameter to the next step's `drafter_embed_lookup` (H2D embedding row
read per step).

**After:** `embed_device_token_into` reads the token id directly from
`mtp_token_chain[k]` on GPU. Eliminates the per-step H2D scalar transfer
and lets the chain run with zero host involvement between steps.

### D. Last-token-logits skip (verify phase)

`forward_prefill_batch_with_pbs_opts` gained a `needs_last_token_logits`
flag. When the caller passes `per_token_hidden_out` (which MTP verify
always does — it needs hidden states for all verify positions), the
forward can optionally skip computing `s.logits` for the last token.

**What this actually avoids:** In the batched prefill, after the layer
loop runs all N positions, the final step is:

```
output_norm(x_batch[last_row]) → s.tmp
weight_gemv(weights.output, s.tmp) → s.logits   // ← this GEMV
```

That's one rmsnorm + one GEMV (lm_head projection of a single row at
`[vocab × dim]`). For the 27B at vocab=151936, dim=5120, that's a
~780 M-op GEMV. At ~1–2 ms per call, this is ~2–3% of the MTP verify
cycle. Skipping it is a small but clean win.

**Why MTP can skip it:** MTP verify already computes all N verify logits
from `verify_hidden` via its own batched lm_head GEMM immediately after
the forward returns. The forward's `s.logits` write would be immediately
overwritten. The skip avoids writing a tensor that no one reads.

**For multi-GPU:** The multi-GPU forward
(`forward_prefill_batch_multi_with_caps`) currently hardcodes
`needs_last_token_logits: true` (qwen35.rs:11189). Extending it with
the `_opts` variant is the same pattern as the single-GPU extension —
pass the flag through to `forward_prefill_chunk`. The savings would be
the same ~2–3% of the verify phase, which is itself a fraction of the
PP+MTP cycle. Worth doing for consistency but not a major lever.

---

## 4. Current state of `_multi`

`spec_step_mtp_compressed_serial_multi` (`mtp_spec.rs:3607`) is the PP+MTP
spec stepper. It was built in Stage 2b (commits `8989aaf4`..`e6a25615`)
before #352 landed, and the #352 merge did NOT touch it.

### K-step chain (lines 3660–3700)

Still uses the old pattern:

```rust
for k in 0..max_n {
    let next_tok = if k == 0 { last_committed } else { candidates[k - 1] };
    drafter_embed_lookup(drafter_gpu, drafter_state, next_tok, dim)?;
    mtp_head_forward_compressed_with_embed(...)?;
    drafter_gpu.argmax_f32_batched(logits_c, &argmax_view, cvs, 1)?;
    let mut argmax_host: [i32; 1] = [0];
    // per-step D2H:
    drafter_gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
    let token_id = vocab_map[draft_idx]; // host-side remap
    candidates.push(token_id);
}
```

**K D2H roundtrips per cycle.** On gfx1031 with K=3, that's 3 × ~30 µs
host-sync latencies per cycle — validated as negligible (§2).

### Verify + accept (lines 3760–3860)

Still uses host-side accept:

```rust
gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
drafter_gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?; // full n_verify D2H
// host-side greedy loop
for k in 0..drafts_generated {
    if argmax_per_pos[k] == candidates[k] { ... }
}
```

### What already matches #352

- **Skip-replay-on-full-accept** (lines 3900–3920): Already implemented.
  `replay_skipped` fires when `advance == drafts_generated + 1 && !hit_eos`.
- **Same-device prev_hidden handoff** (lines 3880–3890): Pure D2D memcpy
  on output_device. Already optimal.
- **Compressed-sidecar path** (`lm_head_draft`, `vocab_map`): Already
  wired.

---

## 5. Scope of the port

Port optimizations A, B, and C into `_multi`. D (last-token-logits skip)
requires extending `forward_prefill_batch_multi_with_caps` with the same
opts wrapper — included in scope since it's a small, well-understood
change.

### What changes

| Component | Single-GPU (#352) | _multi (today) | _multi (after port) |
|---|---|---|---|
| K-step argmax | `argmax_token_chain_f32` (on-device) | `argmax_f32_batched` + D2H per step | `argmax_token_chain_f32` on output_device |
| K-step embedding | `embed_device_token_into` (reads chain on-GPU) | `drafter_embed_lookup` (host-known token) | `embed_device_token_into` on output_device |
| K-step token remap | GPU-side via `lm_head_draft_vocab_map_gpu` | Host `vocab_map[draft_idx]` | GPU-side (map already on output_device) |
| K-step candidate harvest | Bulk D2H max_n × 4 B once | Per-step D2H 4 B × K | Bulk D2H max_n × 4 B once |
| Verify accept | `greedy_accept_from_argmax_i32` → 8 B D2H | Full argmax D2H + host loop | `greedy_accept_from_argmax_i32` → 8 B D2H |
| Verify lm_head | Already in-place | Already in-place | No change |
| Last-token-logits | `needs_last_token_logits=false` | `true` (hardcoded) | `false` |
| Replay skip | Already in-place | Already in-place | No change |

### What does NOT change

- Function signature of `spec_step_mtp_compressed_serial_multi` (same
  params, same return type).
- `generate_pp_mtp` in daemon.rs (calls the spec stepper unchanged).
- The hetero path (`spec_step_mtp_compressed_serial_hetero`) will be
  **deleted** (§10 — it is dead code in the daemon, only used by
  `mtp_only_demo`).

---

## 6. New state required

The device-token chain needs two new GPU tensors on the drafter device
(output_device):

| Tensor | Shape | DType | Owner |
|---|---|---|---|
| `mtp_token_chain` | `[max_n + 1]` | F32 (stores i32) | New field on `MtpHeteroDrafterState` |
| `mtp_token_embed` | `[n_embd]` | F32 | New field on `MtpHeteroDrafterState` |

**Why on `MtpHeteroDrafterState` and not `MtpSpecState`:** The chain and
embed scratch live on the drafter device (output_device) alongside
`mirrored_token_embd`, `step_embd`, `prev_hidden`, and the rest of the
drafter state. `MtpSpecState` contains trunk-resident verify state
(`verify_hidden`, `verify_logits`, etc.). Adding them to the drafter
state keeps the device-affinity invariant clean.

**VRAM cost:** `[max_n + 1] × 4 + dim × 4 = (5 × 4) + (5120 × 4) = 20,540 B`
at K=4 on 27B. Under 21 KB — negligible.

**Allocation site:** `MtpHeteroDrafterState::new_for_slot` (mtp_spec.rs:3073).
Add two `alloc_tensor` calls after the existing `mtp_lm_argmax` allocation.

**Free site:** `MtpHeteroDrafterState::free_gpu` (mtp_spec.rs:3123). Add
corresponding `free_tensor` calls.

### What about `lm_head_draft_vocab_map_gpu`?

Already exists on `Qwen35MtpHeadWeights` and is loaded onto output_device
when the head is loaded (`load_model_pp` line 2967). No new allocation
needed — the port just reads it from `head.weights.lm_head_draft_vocab_map_gpu`
the same way the single-GPU path does.

---

## 7. Implementation steps

### Step 1: Extend `MtpHeteroDrafterState` (~20 LOC)

Add `mtp_token_chain: GpuTensor` and `mtp_token_embed: GpuTensor` fields
to the struct. Allocate in `new_for_slot` on `drafter_gpu`. Free in
`free_gpu`.

```rust
pub mtp_token_chain: GpuTensor,  // [max_n + 1] F32 (i32 slots)
pub mtp_token_embed: GpuTensor,  // [n_embd] F32
```

No new constructor parameter needed — `max_n` and `n_embd`
(`head.config.n_embd`) are already available.

### Step 2: Add device-chain eligibility helper (~10 LOC)

The single-GPU `mtp_device_token_chain_eligible_for` gates on:
- `!use_sampling && !use_p_min`
- `embd_format` is HFQ4G256 or Q8_0

The `_multi` path already asserts greedy-only and no-p_min, so the check
reduces to the format gate. **Reuse the existing function** — call
`mtp_device_token_eligible_for(drafter_state.embd_format, true, false)`.
Do NOT duplicate it; duplicated eligibility checks drift over time
(Gemini review finding 2.3).

### Step 3: Add `embed_device_token_into_drafter` helper (~15 LOC)

The single-GPU `embed_device_token_into` takes `&Qwen35Weights` to read
`embd_format` and `token_embd`. The multi-GPU drafter doesn't have a
`Qwen35Weights` — it has `MtpHeteroDrafterState.mirrored_token_embd`
and `.embd_format`. Add a variant that takes `(Gpu, &GpuTensor, EmbeddingFormat, &GpuTensor, &GpuTensor, usize)`:

```rust
fn embed_device_token_into_drafter(
    gpu: &mut Gpu,
    mirrored_embd: &GpuTensor,
    embd_format: EmbeddingFormat,
    out: &GpuTensor,
    token_id: &GpuTensor,
    dim: usize,
) -> HipResult<()> {
    match embd_format {
        EmbeddingFormat::HFQ4G256 =>
            gpu.embedding_lookup_hfq4g256_batched(mirrored_embd, out, token_id, 1, dim),
        EmbeddingFormat::Q8_0 =>
            gpu.embedding_lookup_q8_batched(mirrored_embd, out, token_id, 1, dim),
        other => panic!("device-token chain: unsupported embd format {other:?}"),
    }
}
```

**Lockstep note:** This helper only handles the formats admitted by
`mtp_device_token_chain_eligible_for` (HFQ4G256, Q8_0). The eligibility
gate ensures the `panic!` branch is unreachable. If the eligibility
gate is extended to new formats, this helper must be extended in
lockstep. The existing `drafter_embed_lookup` supports 5 formats
(HFQ4G256, HFQG128, Q8_0, Q4K, F32); adding them here requires the
`_batched` embedding lookup variant to exist for that format.

### Step 4: Rewrite K-step chain body (~60 LOC)

Replace the current per-step-host-sync loop (lines 3660–3700) with the
device-token-chain pattern from the single-GPU path. The chain phase
operates entirely on `gpus.devices[output_device]` (same as today).

See the original plan draft (§6 Step 3 in git history) for the full
pseudocode. The pattern is identical to the single-GPU path, except:
- `gpu` → `drafter_gpu` (already bound to output_device)
- `weights.token_embd` → `drafter_state.mirrored_token_embd`
- `state.mtp_token_chain` → `drafter_state.mtp_token_chain`
- `state.mtp_token_embed` → `drafter_state.mtp_token_embed`

### Step 5: Wire GPU-side greedy accept (~30 LOC)

After the verify lm_head GEMM + argmax, replace the host-side accept
loop with `greedy_accept_from_argmax_i32`:

```rust
let use_gpu_accept = use_device_token_chain
    && mtp_gpu_greedy_accept_enabled_from_env();

let accepted = if use_gpu_accept {
    let candidate_device = drafter_state.mtp_token_chain.sub_offset(1, drafts_generated);
    let accept_result = state.verify_argmax.sub_offset(0, 2);
    target_gpu.greedy_accept_from_argmax_i32(
        &argmax_v, &candidate_device, &accept_result,
        drafts_generated, eos_token_id,
    )?;
    // 8 B D2H...
    assemble_greedy_accept_from_gpu_result(...)
} else {
    // Legacy host-side accept (unchanged)
};
```

**`verify_argmax` aliasing note:** `accept_result` is a sub-offset of
`argmax_v` (both are views into `state.verify_argmax`). The kernel
`greedy_accept_from_argmax_i32` is single-thread: it reads all inputs
into registers before writing its 2-int output. This in-place aliasing
is safe but depends on the kernel remaining single-thread. The same
pattern already ships in the single-GPU `_serial` path. Add a comment
at the call site: `// ALIASING: accept_result overlaps argmax_v[0..2];
// safe because greedy_accept is single-thread (reads all inputs before
// writing output).`

**Cross-device note:** `candidate_device` lives on output_device (via
`drafter_state.mtp_token_chain`). `argmax_v` also lives on output_device
(verify runs there). `greedy_accept_from_argmax_i32` runs on the GPU
it's called on (`target_gpu` = `gpus.devices[output_device]`). So both
input tensors and the kernel launch are on the same device — no peer
access needed for the accept kernel.

### Step 6: Last-token-logits skip for multi-GPU (~40 LOC)

Add `forward_prefill_batch_multi_with_caps_opts` mirroring the single-GPU
`forward_prefill_batch_with_pbs_opts`. The change:
- New parameter `needs_last_token_logits: bool`
- Pass it through to `forward_prefill_chunk` (line 11189, currently
  hardcoded `true`)
- Update `spec_step_mtp_compressed_serial_multi` to call the `_opts`
  variant with `needs_last_token_logits: false`

### Step 7: Leave legacy fallback path intact

Both the chain and accept paths should have `use_device_token_chain` /
`use_gpu_accept` gates with the original host-sync code as the `else`
branch. This ensures:
- The port is bisectable (toggling the env var shows before/after).
- If the device chain hits an issue, the fallback is one env var away.
- Forward-compat with future embedding formats not yet in the eligibility
  set.

---

## 8. Validation plan

### 8.1 Build + coherence

```bash
cargo build --release --example daemon -p hipfire-runtime --features deltanet
./scripts/coherence-gate-pp.sh   # PP-AR + PP-MTP coherence
```

Hard fail on any output divergence (PpMtp must produce identical tokens
to the pre-port baseline at temp=0 with the same seed/prompt).

### 8.2 Ported-`_multi` token-equality test

Run the ported `_multi` (PP+MTP path) at pp=2 with the device chain
ON vs OFF, same prompt, temp=0. The two runs must emit byte-identical
token sequences. This is the actual validation of the port — §2 only
validated the kernels on-arch, not the new wiring.

```bash
# Legacy mode
HIPFIRE_MTP_DEVICE_TOKEN_CHAIN=0 \
  HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=22 \
  ./target/release/examples/daemon < /tmp/pp_mtp_test.jsonl > /tmp/port_off.jsonl

# Ported mode
HIPFIRE_MTP_DEVICE_TOKEN_CHAIN=1 HIPFIRE_MTP_GPU_GREEDY_ACCEPT=1 \
  HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=22 \
  ./target/release/examples/daemon < /tmp/pp_mtp_test.jsonl > /tmp/port_on.jsonl

# Compare decoded text (must be identical)
diff <(jq -r '.content' /tmp/port_off.jsonl) <(jq -r '.content' /tmp/port_on.jsonl)
```

### 8.3 PP+MTP A/B benchmark

```bash
# A: device chain OFF (legacy _multi behavior)
HIPFIRE_MTP_DEVICE_TOKEN_CHAIN=0 HIPFIRE_MTP_GPU_GREEDY_ACCEPT=0 \
  ./scripts/bench-ppmtp-split.sh

# B: device chain ON (ported optimizations)
HIPFIRE_MTP_DEVICE_TOKEN_CHAIN=1 HIPFIRE_MTP_GPU_GREEDY_ACCEPT=1 \
  ./scripts/bench-ppmtp-split.sh
```

Each cell runs 3× in fresh processes. Record prompt md5, binary md5, and
decode tok/s per cell.

### 8.4 Expected outcomes

Based on the §2 data showing the optimizations are neutral on both
gfx906 and gfx1031 for single-GPU MTP:

**Expected: PpMtp stays at ~14.2 tok/s ± noise.** The host-sync savings
(~90 µs/cycle on gfx1031, ~90 µs/cycle on gfx906) are too small relative
to the PP cycle wall (~70 ms) to move the needle.

**The benchmark's value is confirmatory, not exploratory.** If it
defies expectations and shows a real delta, that reveals something
unexpected about the PP cycle's interaction with host sync (e.g., D2H
contention with the boundary copy). If it shows no delta, it closes the
host-sync investigation definitively.

**Exit criterion for the legacy branch:** If the A/B benchmark shows
<2% decode tok/s delta between chain ON and chain OFF, remove the
`HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` and `HIPFIRE_MTP_GPU_GREEDY_ACCEPT`
env gates in a follow-up commit, keeping only the device-chain path.
There is no value in maintaining a permanently dual-pathed spec
stepper for a zero-delta toggle.

### 8.5 Decision gate after benchmark

| Result | Implication |
|---|---|
| PpMtp ≥ pp2-AR (≥17.6 tok/s) | Unlikely given §2 data, but if so: pursue async overlap. |
| PpMtp 15–17 tok/s (partial close) | Unexpected — profile the remaining gap. Likely PP boundary serialization. Pursue no-replay rollback (Opt 3) next. |
| PpMtp <15 tok/s (<5% improvement) | **Most likely.** PP+MTP decode speed is structurally limited on this hardware. PpMtp remains a long-ctx play. Close the decode-speed investigation. |

---

## 9. Risk assessment

### Low risk

- **Kernel arch-compat:** `argmax_token_chain_f32` and
  `greedy_accept_from_argmax_i32` are pure HIP compute (no WMMA/MFMA).
  Compile and run on any arch. Validated on gfx1031 (RDNA2) and gfx906
  (CDNA1).
- **Regression:** Legacy fallback preserved behind env gates.
- **Port correctness:** The kernels are validated on-arch (§2). The
  port itself (new helper, new state, new chain wiring in `_multi`) is
  validated by §8.2 token-equality test + §8.1 coherence gate.

### Low-medium risk

- **`embed_device_token_into_drafter` adaptation:** The single-GPU version
  reads from `weights.token_embd` (a `WeightTensor` on the trunk GPU).
  The multi-GPU version reads from `drafter_state.mirrored_token_embd`
  (a plain `GpuTensor` on output_device). The underlying kernels take
  `&GpuTensor` (the `WeightTensor.buf`), so dispatch is identical. But
  `embd_format` must be read from `drafter_state.embd_format`. Small
  surface.

- **Last-token-logits skip in multi-GPU prefill:** Requires threading the
  flag through `forward_prefill_batch_multi_with_caps` →
  `forward_prefill_chunk`. The single-GPU version is already shipped and
  tested. Risk is in the multi-GPU plumbing only. Verify no other caller
  of the multi forward relies on `s.logits[last]` being populated — the
  existing comment says "preserve multi-GPU post-condition" so this
  needs explicit sign-off.

- **`verify_argmax` buffer aliasing in GPU accept:** The GPU accept
  kernel reads from `verify_argmax[0..n_verify]` and writes to
  `verify_argmax[0..2]`, aliasing the buffer. Safe because the kernel is
  single-thread (reads all inputs into registers before writing). The
  same pattern ships in single-GPU `_serial`. Add a comment at the
  call site documenting the assumption. After GPU accept, positions
  0–1 are clobbered — no downstream consumer should read them.

- **Deleting `_serial_hetero` removes the `--mtp-device` hetero demo
  mode** and the `HIPFIRE_HETERO_DIFF` cross-device bit-comparison
  instrumentation from `mtp_only_demo`. If peer-copy regression
  testing is still needed for new GPU pair bring-up, consider keeping
  `_hetero` as a test-only function behind `#[cfg(test)]` or in a
  separate test file. The production daemon does not use it.

---

## 10. Unification: merge hetero + multi spec steppers

### 10.1 Structural analysis

The three spec steppers in `mtp_spec.rs`:

| Function | Lines | Chain | Verify | Accept | Handoff | Rollback |
|---|---|---|---|---|---|---|
| `_serial` (single-GPU) | 1149 | 350 | 269 | 75 | inline | 276 |
| `_serial_hetero` | 410 | 94 | 95 | 34 | 39 | 84 |
| `_serial_multi` | 456 | 68 | 104 | 27 | 16 | 191 |

Total: **2015 LOC, ~50% of `mtp_spec.rs`.**

**Important: `_hetero` is dead code in the daemon.** It is only called from
`mtp_only_demo` (line 506). The production daemon's `generate_mtp` (pp=1)
always uses the single-GPU `_serial`. The `MtpState.drafter_state` field
that would trigger hetero routing is only consumed by `generate_pp_mtp`
(PpMtp path). So the hetero stepper exists purely as a demo/research tool.

**Hetero is not "multi without arch enforcement" — it's a different
configuration entirely:**

| | Hetero | Multi (PP+MTP) |
|---|---|---|
| Trunk placement | All layers on one GPU (GPU A) | Layers split across GPU A + B (PP) |
| MTP head placement | Separate GPU (GPU B) | output_device (GPU B, same as last PP band) |
| Trunk verify call | `forward_prefill_batch_with_pbs(target_gpu, ...)` | `forward_prefill_batch_multi_with_caps(gpus, ...)` |
| prev_hidden handoff | Cross-device peer copy | Same-device D2D memcpy |
| GDN tape replay | Available | Disabled (per-band dn_state ownership issue) |
| Used in production | **No** — only `mtp_only_demo` | **Yes** — `generate_pp_mtp` |

Hetero was the original proof-of-concept for "MTP head on a sibling GPU"
before the PP+MTP combo was built. It served its purpose (validated the
cross-device handoff, measured the 112 µs peer-copy overhead, caught the
row-0 peer-copy bug documented in devlog 2026-05-28). Now that PP+MTP
exists, hetero is the PP=1 special case of multi — same layout but with
all trunk layers on one device.

### 10.2 Where hetero and multi are identical

The `_hetero` and `_multi` functions share **~80% identical code**:

1. **K-step draft chain** — same `drafter_embed_lookup` →
   `mtp_head_forward_compressed_with_embed` → `argmax_f32_batched` →
   D2H → host vocab_map remap → `t_mtp_out` save loop. Only the GPU handle
   differs (`drafter_gpu: &mut Gpu` vs `gpus.devices[output_device]`).

2. **lm_head GEMM + argmax** — same dtype match, same GEMM calls, same
   argmax. Only the GPU handle differs.

3. **Host-side greedy accept** — byte-identical loop.

4. **Skip-replay-on-full-accept** — identical pattern.

### 10.3 Where they differ

| Aspect | Hetero | Multi |
|---|---|---|
| Trunk verify forward | `forward_prefill_batch_with_pbs(target_gpu, ...)` | `forward_prefill_batch_multi_with_caps(gpus, ...)` |
| Rollback forward | `forward_prefill_batch(target_gpu, ...)` / `forward_scratch(target_gpu, ...)` | `forward_prefill_batch_multi(gpus, ...)` / `forward_scratch_multi(gpus, ...)` |
| GDN tape replay | May take `replay_gdn(target_gpu, ...)` | Disabled (`tape_captured = false`), `unreachable!` branch |
| prev_hidden handoff | Cross-device peer copy target_gpu → drafter_gpu | Same-device D2D memcpy on output_device |
| Trunk state access | `target.dn_state`, `target.kv_cache`, `target.scratch` | Separate params: `target_dn`, `target_kv`, `pp_scratch_set` |
| Config access | `target.config` | `target_config` param |
| HIPFIRE_HETERO_DIFF | Capture points in chain (~20 LOC) | None |
| Stream init | Assumes active_stream already set | Guard: `if active_stream.is_none { bind + create }` |

### 10.4 Unification strategy

Since `_hetero` is dead code in production (only `mtp_only_demo` uses
it), the cleanest path is:

**Option A (recommended): port `_multi` + delete `_hetero`.** ~1 day.

1. Port device chain + GPU accept into `_multi` (steps 1–6 from §7).
2. Delete `_serial_hetero` (410 LOC).
3. Update `mtp_only_demo` to use single-GPU path for the hetero case
   (or remove hetero mode from the demo entirely).
4. Net: ~-280 LOC.

This gives the maintenance reduction (one fewer spec stepper) without
the complexity of a new abstraction. The only production caller of the
multi-GPU MTP spec stepper is `generate_pp_mtp` in daemon.rs, and it
only calls `_multi` — so there's no benefit to abstracting over
dispatch variants.

**Option B: full unification with `TrunkForward` enum.** ~2 days.

Introduce a `TrunkForward` enum that abstracts over single-GPU and
multi-GPU trunk dispatch:

```rust
enum TrunkForward<'a> {
    SingleGpu {
        gpu: &'a mut Gpu,
        config: &'a Qwen35Config,
        weights: &'a Qwen35Weights,
        kv: &'a mut KvCache,
        dn: &'a mut DeltaNetState,
        scratch: &'a Qwen35Scratch,
    },
    MultiGpu {
        gpus: &'a mut Gpus,
        output_device: usize,
        config: &'a Qwen35Config,
        weights: &'a Qwen35Weights,
        kv: &'a mut KvCache,
        dn: &'a mut DeltaNetState,
        scratch_set: &'a Qwen35ScratchSet,
    },
}
```

With methods for `verify_forward`, `scratch_forward`, `batch_replay`,
`output_gpu`, and `supports_gdn_tape`. Then a unified function
`spec_step_mtp_compressed_serial_distributed` replaces both `_hetero`
and `_multi`. `mtp_only_demo` would construct `TrunkForward::SingleGpu`
for its hetero case.

This adds ~120 LOC for the enum impl and saves ~-475 LOC net, but
introduces an abstraction that currently has only one production arm
(MultiGpu). The SingleGpu arm exists solely for `mtp_only_demo`.

**Borrow-checker note (Option B):** In the PP+MTP case, `drafter_gpu`
and `trunk.output_gpu()` are the same physical device. The unified
function must not hold concurrent `&mut Gpu` refs to the same device.
Solution: re-borrow `gpus.single_mut(output_device)` at each phase
boundary (chain → verify → rollback), releasing before the next phase.
This matches the existing `_multi` pattern where borrows are scoped in
`{ ... }` blocks.

**Recommendation: Option A.** The `TrunkForward` abstraction only pays
for itself if a third caller emerges (e.g., hetero pp=1 in the daemon).
If that never happens, the enum is overhead without benefit. Delete the
dead code, port the optimization, ship.

### 10.5 What about single-GPU?

The single-GPU `_serial` is 1149 LOC because it handles modes that
hetero/multi don't (full-vocab, sampling, p_min, proposal graph). These
modes make its chain and accept phases fundamentally different in
structure, not just dispatch target. Unifying all three into one function
would require gating every branch on mode enums — making the already-
complex single-GPU path harder to read without reducing LOC.

**Recommendation: don't unify single-GPU.** Keep `_serial` as the
full-featured path. It already has the #352 device chain.

### 10.6 Scope and LOC estimate (Option A)

| Change | LOC |
|---|---|
| Port device chain + GPU accept into `_multi` | ~130 |
| Add `embed_device_token_into_drafter` helper | ~15 |
| Extend `MtpHeteroDrafterState` with chain + embed tensors | ~20 |
| Delete `_serial_hetero` (410 LOC) | -410 |
| Update `mtp_only_demo` (remove hetero mode) | ~-10 |
| **Net** | **~-255 LOC** |

### 10.7 Recommended commit order (Option A)

1. Extend `MtpHeteroDrafterState` with `mtp_token_chain` + `mtp_token_embed`.
2. Add `embed_device_token_into_drafter` + eligibility helper.
3. Port device chain + GPU accept into `_multi`.
4. Add `forward_prefill_batch_multi_with_caps_opts` for last-token-logits skip.
5. Delete `_serial_hetero`. Update `mtp_only_demo`.
6. Coherence-gate + benchmark.

---

## 11. Deferred (explicitly out of scope)

| Item | Reason | Where it would go |
|---|---|---|
| **Proposal graph capture for PP** | Evaluated as net-negative on single-GPU post-PR5/PR6 (#352). No evidence it helps on PP. | Would need PP-aware graph capture with per-device streams. |
| **No-replay rollback (Opt 3)** | The biggest architectural lever for PP+MTP decode speed (eliminates the 2nd PP boundary crossing on ~65% of cycles). But requires changes to DeltaNet state management under PP. Independent. | Would modify the rollback section of `_multi` to skip full replay when verify already advanced correctly. |
| **Option B: `TrunkForward` unification** | Only worthwhile if a third caller emerges (hetero pp=1 in daemon). Otherwise the enum adds complexity for one production arm. | Would replace `_multi` with `spec_step_mtp_compressed_serial_distributed` + `TrunkForward`. |

---

## 12. File change summary

| File | Change | LOC (est.) |
|---|---|---|
| `crates/hipfire-arch-qwen35/src/mtp_spec.rs` | Add `mtp_token_chain` + `mtp_token_embed` to `MtpHeteroDrafterState`. Add `embed_device_token_into_drafter` helper. Port device chain + GPU accept into `_multi`. Delete `_serial_hetero`. | +165 / -410 |
| `crates/hipfire-arch-qwen35/src/qwen35.rs` | Add `forward_prefill_batch_multi_with_caps_opts` wrapper. Thread `needs_last_token_logits` through to `forward_prefill_chunk`. | ~40 |
| `crates/hipfire-runtime/examples/mtp_only_demo.rs` | Remove hetero mode (or downgrade to single-GPU path). | ~-10 |
| `scripts/bench-ppmtp-split.sh` | No changes needed. | 0 |
| `docs/plans/multi_gpu_pr352.md` | This file. | — |

**Net: ~-205 LOC**. Estimated implementation: 1 day including
coherence-gate validation + benchmarking.

---

## 13. Timeline

| Step | Description | Time |
|---|---|---|
| 1 | Extend `MtpHeteroDrafterState` with chain + embed tensors | 30 min |
| 2 | Add `embed_device_token_into_drafter` + eligibility helper | 45 min |
| 3 | Port device chain + GPU accept into `_multi` | 2 hr |
| 4 | Last-token-logits skip for multi-GPU | 1 hr |
| 5 | Delete `_serial_hetero`, update `mtp_only_demo` | 1 hr |
| 6 | Build + coherence-gate | 30 min |
| 7 | Single-GPU regression on gfx1031 | 15 min |
| 8 | PP+MTP A/B benchmark (3 runs × 6 cells = 18 runs) | 2 hr |
| 9 | Devlog with results + decision gate | 30 min |
| **Total** | | **~8 hr (~1 day)** |
