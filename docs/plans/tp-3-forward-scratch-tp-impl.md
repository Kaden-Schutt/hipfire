# Stage 3 — `forward_scratch_tp` implementation guide

**Status:** ✅ COMPLETE (2026-05-28, uncommitted). DeltaNet extraction +
`forward_scratch_tp` orchestrator + full-model parity harness all landed and
green. TP=2 ↔ TP=1 fp32 parity = **2.6e-6, 32/32 argmax identical, PASS**.
**Target (met):** TP=2 ↔ TP=1 logits within 1e-4 on `qwen3.5-0.8b.mq4` greedy.
**Prereqs done:** RCCL all-reduce (`1cde4e80`), `ShardConfig`+`init_tp`
(`87b6c90a`), 3a wo+all-reduce mechanism smoke green (`44735ed0`),
FA-body split (`e5afa07b`), FA-layer parity (`5f728d98`).

## Acceptance result + two bugs found (2026-05-28)

`crates/hipfire-arch-qwen35/examples/tp_attn_parity.rs` drives the reference
greedy path and force-feeds it to TP (identical input path → apples-to-apples
per-position logit delta). Final: **fp32 KV + fp32 DeltaNet state → worst rel
Δ 2.6e-6, 32/32 argmax match, PASS.**

Getting there surfaced two real bugs (both fixed):

1. **`run_fa_layer_body` q8 attention ignored `flash_mode`.** It always called
   non-flash `attention_q8_0_kv`, but the inline decode FA arm uses a
   `use_flash` dispatch and on gfx11/gfx12 `flash_mode` defaults to **2 (always
   flash)**. So ref used `attention_flash_q8_0` and TP used non-flash → they
   agree at pos 0 (single key → V regardless of scores) but diverge at pos ≥ 1.
   Fix: `run_fa_layer_body`'s q8 branch now mirrors the inline arm's flash
   dispatch (fulfilling its documented "byte-exact with the FA branch"
   contract). The FA-layer parity (`5f728d98`) missed this because *both* sides
   used `run_fa_layer_body` and tested pos 0 only.

2. **`pos_buf` written on the null stream while kernels run on `active_stream`.**
   `forward_scratch_tp` sets `active_stream` (required by the all-reduce); every
   kernel then launches on it (`dispatch.rs:1150`), but `pos_buf` was written
   with raw `memcpy_htod` (null stream) → ordering hazard. Fixed to
   `memcpy_htod_async` on the active stream. (Turned out not to be the
   divergence cause here — null/active gave identical results — but it is a
   latent correctness hazard, so the fix stays.)

## The q8-sensitivity finding — it's the DeltaNet STATE, not the KV (NOT a TP bug)

Root cause: the cross-rank all-reduce computes `wo@cols[0:1024] +
wo@cols[1024:2048]` in a different summation order than the single-GPU
`wo@cols[0:2048]` → an unavoidable ~1e-7 reassociation in `s.x`. Quantized
state amplifies that 1e-7 across quantization buckets and recurrence compounds
it. Single-GPU is immune only because it is perturbation-free (ref-vs-ref is
bit-exact 0.0). This is the `feedback_attention_precision` "tiny attention error
→ cascade" sensitivity at the quantization-boundary level — a model-precision
property, not a TP-implementation bug.

**Isolation matrix (qwen3.5-0.8b, full ChatML prompt, 32 decode steps,
TP=2↔TP=1 along the identical forced path):**

| KV   | DeltaNet state | worst rel logit Δ | argmax agreement |
|------|----------------|-------------------|------------------|
| fp32 | fp32           | **2.6e-6**        | 32/32            |
| q8   | fp32           | **4.1e-3**        | **32/32**        |
| fwht | fp32           | **3.5e-3**        | **32/32**        |
| fp32 | q8             | **3.7e-1**        | 32/32 (on edge)  |
| q8   | q8             | 3.7e-1            | 30/32            |
| fwht | q8             | 3.2e-1            | 31/32            |

**The DeltaNet recurrent *state* quant is the amplifier; the KV cache is not —
for q8 AND FWHT KV alike.** Both q8 and fwht KV add only ~3–4e-3 with zero token
flips: the KV cache is *read* (a fresh weighted sum per position), so its
quantization noise doesn't compound. The DeltaNet state is *recurrently updated*
and carries forward, so q8 state turns the 1e-7 reassociation into ~3.2–3.7e-1
regardless of KV mode. Physically sensible.

(FWHT KV under TP required fixing `run_fa_layer_body`'s asym4 branch to add the
`fwht4` sub-branch the inline decode arm already had — without it TP ran plain
asym4 while single-GPU ran fwht4. The 3.5e-3 fwht/fp32-state result validates
that fix; a broken fwht4 dispatch would have shown a large flash-bug-style
mismatch even at fp32 state.)

**Production recommendation for TP:** keep the KV cache quantized — **q8 or
FWHT both work** (the big VRAM consumer) — and run the DeltaNet recurrent state
in **fp32** (small per-layer matrix — cheap). Either KV mode holds
token-identical to single-GPU in this test (32/32) at ~3–4e-3 logit Δ. Do NOT
run q8 DeltaNet state under TP (token flips by ~step 8–12).

**Gate:** the default fp32/fp32 acceptance isolates and validates the TP forward
math (2.6e-6 < 1e-4, PASS). The harness exposes `HIPFIRE_PARITY_KV={fp32,q8}` and
`HIPFIRE_PARITY_STATE={fp32,q8}` independently to reproduce any cell above.
Implication for A3B: validate TP math at fp32; ship TP with q8 KV + fp32
DeltaNet state.

Diagnostics built into the harness (env-gated): `HIPFIRE_PARITY_NTOK`,
`HIPFIRE_PARITY_REFREF`, `HIPFIRE_PARITY_REFSTREAM`, `HIPFIRE_PARITY_RANKDIFF`,
`HIPFIRE_PARITY_Q8`. RANKDIFF confirmed the two ranks stay **bit-identical**
(0.0) — replication is exact; the only divergence vs single-GPU is the
all-reduce reassociation.

This is the execution recipe grounded in the real post-#352 code. All
line numbers are current as of this commit; re-grep if the file moved.

## 0. Model reality (verified)

`qwen3.5-0.8b.mq4` is a **hybrid**, qwen3.5-vl text wrapper: `dim=1024,
n_heads=8, n_kv_heads=2, head_dim=256, num_experts=0`, tied embeddings,
Q8 KV. **FullAttention at layers 3,7,11,15,19,23; DeltaNet
(LinearAttention) on the rest** (layer 0 is DeltaNet). So the parity gate
runs *mostly DeltaNet layers* — `forward_scratch_tp` must handle both.

**TP decomposition decision:** shard only FullAttn layers (masked-head
attention + partial `wo` + all-reduce). Run **DeltaNet layers replicated**
(full + identical on every rank — the residual `s.x` stays in sync because
all ranks start each layer with the same `s.x` and run identical
deterministic DeltaNet weights). DeltaNet TP sharding (16 value heads +
recurrent state) is a later stage.

## Progress (2026-05-28)

- **FA-body split** (`FaPhase`) — DONE, committed `e5afa07b`, coherence
  gate green.
- **FA TP machinery validated end-to-end** — `tp_fa_layer_parity.rs`
  (uncommitted) runs the FULL `TpAttn → all_reduce(s.o) → add_f32(s.x) →
  TpFfn` flow on a real FullAttn layer (layer 3 of 0.8B) vs single-GPU
  `FaPhase::Full`: **max rel 1.18e-7**. The novel/risky TP piece is proven.
- **Remaining:** DeltaNet body extraction (§2) + the `forward_scratch_tp`
  orchestrator (§3, now just "loop the validated FA path + replicated
  DeltaNet over all layers") + full-model parity harness (§4).

## 1. FA-body split (DONE — committed e5afa07b)

`run_fa_layer_body` (qwen35.rs ~8408) now takes a `phase: FaPhase`:

```rust
pub enum FaPhase<'a> { Full, TpAttn { mask: &'a GpuTensor }, TpFfn }
```

- `Full`: attention → `weight_gemv_residual(wo → s.x)` → FFN (via the new
  `run_fa_ffn_body`). Byte-identical to pre-TP.
- `TpAttn { mask }`: attention → `sigmoid_mul` → `mul_f32(fa_attn_out,
  mask, fa_attn_out)` (zero non-local Q-heads) → `weight_gemv(wo →
  s.o)` (PARTIAL, non-residual) → **return before FFN**.
- `TpFfn`: skip attention, run only `run_fa_ffn_body` on the
  already-all-reduced `s.x`.

`run_fa_ffn_body(gpu, weights, config, layer_idx, s)` is the extracted FFN
(was inline 8648–8736). Only caller of `run_fa_layer_body` besides the new
TP path is the prefill fallback (~7588), updated to `FaPhase::Full`.

**Verify before commit:** `./scripts/coherence-gate.sh` (qwen35.rs is a
hotspot). The split is byte-preserving by construction (FFN relocated, not
retyped; `Full` runs the identical kernel sequence).

## 2. DeltaNet body extraction (TODO — main-loop surgery)

The DeltaNet branch is **inline** in `forward_scratch_layers`
(~8895–9296, arm `(LayerWeights::DeltaNet(layer), LayerType::Linear
Attention)`), ending with `delta_layer_idx += 1` (9296). There is no
existing extracted helper (unlike FA's `run_fa_layer_body`).

Extract `run_dn_layer_body(gpu, weights, config, layer_idx,
delta_layer_idx, kv_layer_idx, pos, kv_cache, dn_state, s, hidden_rb?)`
covering 8895–~9295 (NOT the `+= 1`). Replace the inline arm with a call +
keep the `delta_layer_idx += 1`. `delta_layer_idx` indexes
`dn_state.{conv_states,s_matrices,s_scales}` (9101/9148/9153/9159).
This touches the hot decode path → **coherence gate mandatory**; do it as a
pure cut-paste-call (no logic change) so output is byte-identical.

Alternative if extraction feels too risky: copy the branch into a new
`tp_dn_layer_body` (duplication, zero risk to the production path; the TP
parity gate validates correctness). Lower blast radius, higher transcription
risk — prefer extraction + coherence gate.

## 3. `forward_scratch_tp` orchestrator (TODO)

New `pub fn` in qwen35.rs. Signature (per-rank state, rank-major):

```rust
pub fn forward_scratch_tp(
    gpus: &mut Gpus,
    shard: &ShardConfig,
    weights: &[Qwen35Weights],         // len tp, replicated
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_caches: &mut [llama::KvCache],  // len tp
    dn_states: &mut [DeltaNetState],   // len tp
    scratches: &[Qwen35Scratch],       // len tp
    fa_masks: &[GpuTensor],            // len tp; mask_r = 1.0 on rank r's
                                       // local Q-heads (wo_col_range), else 0
) -> HipResult<()>
```

`shard.is_single()` → delegate to `forward_scratch` (degenerate TP=1).

Body (TP>1):
1. **Embedding** per rank: reproduce the embedding dispatch from
   `forward_scratch` (~4328) into each `scratches[r].x` (identical).
   Consider `forward_scratch_embed` (8774) if it does exactly this.
2. **Layer loop** `for layer_idx in 0..n_layers`, tracking per-rank
   `delta_layer_idx`/`kv_layer_idx` (same on all ranks):
   - **DeltaNet:** `for r in 0..tp { gpus.devices[r].bind_thread();
     run_dn_layer_body(rank r's gpu/weights/kv/dn_state/scratch) }`.
     Replicated; no all-reduce. `s.x` identical across ranks after.
   - **FullAttn:**
     a. `for r: run_fa_layer_body(.., FaPhase::TpAttn{ mask: &fa_masks[r] })`
        → each rank's partial `wo` lands in `scratches[r].o`; `s.x`
        untouched.
     b. `gpus.devices[r].hip.device_synchronize()` per rank (FA kernels
        run on default stream; all-reduce runs on `active_stream`).
     c. `gpus.all_reduce_sum_f32(&[&scratches[r].o.buf ...], dim)` →
        each `scratches[r].o` now holds the full attention contribution.
     d. `for r: gpus.devices[r].add_f32(&s.x, &s.o, &s.x)` → residual
        update; `s.x` now identical across ranks.
     e. `for r: run_fa_layer_body(.., FaPhase::TpFfn)` → FFN on the
        synced `s.x`. (Or `run_fa_ffn_body` directly.)
3. **Final norm + lm_head** on rank 0 only (replicated lm_head, sampling
   reads rank 0 per §3.5): `rmsnorm_f32(output_norm) → weight_gemv(output)
   → scratches[0].logits` (reproduce from forward_scratch_layers tail
   ~8396 or forward_scratch).

**Streams:** set `gpus.devices[r].active_stream = Some(stream)` once (the
all-reduce requires it — see `tp_allreduce_smoke`). Per-rank
`bind_thread()` before every `Gpu::*` call (single-thread-per-HIP-work
invariant).

**Buffer reuse:** `scratches[r].o` ([dim]) is free in the FA mq4 path →
used as the partial-wo buffer. `mul_f32`/`add_f32` are `Gpu` methods
(dispatch.rs 22786/22695).

## 4. Parity harness (TODO)

`crates/hipfire-arch-qwen35/examples/tp_attn_parity.rs`:

1. Single-GPU reference: load 0.8B on 1 GPU, run greedy decode N steps via
   `forward_scratch`, capture per-step `download_f32(&scratch.logits)`.
2. TP=2: `Gpus::init_tp(2, n_layers)`, set streams, load replicated on each
   rank, alloc per-rank scratch/kv/dn_state, build `fa_masks` (rank r:
   1.0 on `wo_col_range(r, n_heads, head_dim)`, else 0 — n_heads*head_dim
   = 2048 wide), run `forward_scratch_tp` N steps, capture rank-0 logits.
3. Assert per-step `max|Δ| / max|ref|  < 1e-4` AND identical argmax token
   stream. Print decoded text for eyeball (per CLAUDE.md coherence rules).

Loading: `load_weights(&mut hfq, &config, &mut gpus.devices[r])` per rank
(open `HfqFile` once; `enable_peer_all` AFTER all allocs per the ROCm
gotcha in multi_gpu.rs). Tokenizer via `Tokenizer::from_hfq_metadata`.

## 5. Gate before any claim

- `./scripts/coherence-gate.sh` (FA split + DeltaNet extraction touch the
  decode path).
- TP=2↔TP=1 parity (this harness) — the Stage 3 acceptance.
- 1e-4, NOT bit-exact: the cross-rank sum reorders fp adds (the 3a smoke
  measured ~1e-7 for the wo step alone; full-forward drift accumulates but
  should stay << 1e-4 at greedy temp=0; if argmax diverges, investigate
  per `feedback_attention_precision` — 5% attn error → attractor).

## 6. Risks / watch-items

- **DeltaNet extraction regressing single-GPU** — pure cut-paste-call,
  coherence-gate it. Highest-blast-radius step.
- **`device_synchronize` vs stream ordering** — FA kernels likely use the
  default stream; the all-reduce uses `active_stream`. Sync between them
  (the 3a smoke does a full `device_synchronize` before all-reduce).
- **fa_mask layout** — mask is over the *attention output* (n_heads*head_dim
  = 2048), NOT `dim` (1024). `wo_col_range` is in attention-output units.
- **Embedding/lmhead reproduction drift** — prefer calling existing
  `forward_scratch_embed` / the tail of `forward_scratch_layers` over
  retyping, to avoid convention drift.
- This validates **wo sharding + DeltaNet-replicated**; it does NOT yet
  shard the attention *compute* (each rank still runs full attention then
  masks). True compute/memory savings = 3b (per-rank wq row-slice load +
  the wo column-slice quant loader) + running attention on the local head
  subset.
