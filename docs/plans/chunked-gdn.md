# Chunked (parallel) GDN for hipfire

> **Status: F32 chunked == sequential VALIDATED at machine precision
> (max out-diff 1.332e-15); q8_ef boundary error-feedback VALIDATED
> (all four claims pass).** The math below is no longer a sketch — it is
> the parity-proven recurrence from the two numpy references in
> `/home/kaden/.claude/jobs/3b7dee40/tmp/` (`gdn_chunk_parity_3.py`,
> `gdn_chunk_q8ef.py`). What remains is the HIP kernel.

## Why
hipfire's gated-delta-net runs **sequentially per token** (`gated_delta_net_*_batch_seq`,
`qwen35.rs:6147` "still sequential per token") for *every* multi-token pass — prefill AND the
spec-decode verify. Lucebox (`Luce-Org/llama.cpp-dflash-ggml`, `build_delta_net_chunking`) flips to a
**chunked parallel** scan the instant `n_seq_tokens > 1`, sequential only for `n==1` decode.
Same qwen, same accurate-but-divergent math; Lucebox parallelizes the recurrence across the
batch and we don't. That gap is why their DDTree wins and ours loses, and why our prefill walks
the prompt token-by-token.

**Payoff (honest):** lifts **prefill/TTFT** (clean, spec-independent — the prompt is `n>1`) and the
**spec-decode verify** (DFlash chain *and* DDTree — both `n>1`, makes the tree's extra candidates
cheap). Does **NOT** lift plain AR decode (`n==1`, one token, nothing to chunk — Lucebox is
sequential there too).

## The two recurrences

**hipfire — sequential, per token** (`kernels/src/gated_delta_net_f32_batch_seq.hip:58-92`, the parity ref):
```
alpha = exp(gate[t]); beta = beta[t]
kv    = S · k_t                       # S_old times key  (HD vector)
delta = beta * (v_t - alpha * kv)     # *** alpha is INSIDE the delta ***
S     = alpha * S + outer(delta, k_t) # decay-first, then rank-1 write
out_t = S · q_t                       # post-update read
```
The three hipfire-specific quirks that the chunked form must reproduce exactly:
1. **decay-first**: `S <- alpha*S` happens *before* the write, so `delta` is measured against
   the **decayed** state, not the pre-decay one.
2. **alpha inside delta**: `delta = beta*(v - alpha*kv)` — the gate multiplies the recurrent read
   `kv` inside the delta. This is the "divergent formulation" vs standard linear-attn factorings.
3. **post-update read**: `out_t = S_new · q_t`, the output sees the just-written token.

## The validated F32 chunked recurrence (parity ground truth)

Ref: `gdn_chunk_parity_3.py` (`gdn_chunked_f32`). Within a chunk of length `C` (local idx
`i = 0..C-1`, local gates `g_i`), define the **inclusive** cumsum and pairwise decay:
```
G_i   = sum_{m<=i} g_m                 # inclusive log-gate cumsum  (C,)
D[i,j]= exp(G_i - G_j)                 # pairwise decay            (C,C)
```
Unrolling the decay-first recurrence inside the chunk against `S_in` (the state entering the
chunk) gives `S_after_i = exp(G_i)·S_in + Σ_{j<=i} exp(G_i-G_j)·outer(delta_j, k_j)`. Every
`alpha_j` that sits *inside* `delta_j` is absorbed by the decay exponentials, because
`alpha_j·exp(G_{j-1}-G_l) = exp(G_j-G_l)` — that algebraic fold is *why* the alpha-inside-delta
convention does not need a special term; it becomes a convention-agnostic decay weight. The
delta-rule's within-chunk dependency (token `j`'s delta depends on earlier deltas `l<j`) collapses
to a **unit-lower-triangular solve**:

```
# Inter-chunk pull (state read, measured against the post-decay state):
w_j   = exp(G_j) · (S_in @ k_j)                      # (C, HD)
# Strict-lower intra coupling:
A[j,l]= exp(G_j - G_l) · (k_j · k_l)   for l<j       # = tril(D * (K@Kᵀ), -1)
# Delta-rule correction system  (I + diag(beta)·A) delta = beta ⊙ (v - w):
L     = diag(beta) · A                               # strict-lower (C,C)
delta = forward_subst_unit_lower(L, beta⊙(v - w))    # unit-lower solve, NOT np.linalg.solve
# Output = inter (decayed S_in via q) + intra (lower-tri kq @ corrected delta):
out_i = exp(G_i)·(S_in @ q_i) + tril(D * (Q@Kᵀ), 0) @ delta
# State carry to next chunk:
S_out = exp(G_C)·S_in + (delta * exp(G_C - G)[:,None])ᵀ @ K     # G_C = G[-1]
```

The solve is done by **explicit forward-substitution** (`forward_subst_unit_lower`,
`x_i = b_i − Σ_{l<i} L[i,l]·x_l`), *not* `np.linalg.solve` — this mirrors the `solve_tri` HIP
kernel the port will write, where the unit diagonal means no division. **Tail handling**
(`T` not a multiple of `CS`): the last chunk is shortened by `C = min(c0+CS, T) − c0`; every
cumsum/decay/solve size follows `C`, so the short chunk is exact — verified on `T=33, CS∈{4,8}`.

### Parity result
`gdn_chunk_parity_3.py` sweeps HD∈{4,8} × T∈{8,16,33} × CS∈{4,8} × 2 heads × random `S0`
(24 configs, including the `T=33` non-multiple-of-CS tail) and asserts chunked == sequential:

```
worst out_max_abs_diff   = 1.332e-15
worst state_max_abs_diff = 3.886e-16
PARITY OK   (CHUNKED == SEQUENTIAL across all configs including T=33 tail)
```
This is float64 round-off — the chunked decomposition is **algebraically exact**, not approximate.

### Divergence from Lucebox (why this is a derivation, not a port)
Lucebox's `build_delta_net_chunking` (`delta-net-base.cpp:16-289`) is GGML/f32 and divergently
factored. Three concrete differences the parity test forced out:
- **alpha placement.** Lucebox factors the gate out front (standard linear-attn style); hipfire
  folds it inside the delta and decays-first. Our `A[j,l]=exp(G_j-G_l)(k_j·k_l)` and
  `w_j=exp(G_j)(S_in@k_j)` are both measured against the **post-decay** state — matching the
  kernel — whereas a literal Lucebox transcription measures against the pre-decay state and fails
  parity by an `alpha` factor per row.
- **solve form.** Lucebox builds `attn = (I+L)^{-1}` via `solve_tri(I+L, -L) + I` (a matrix
  inverse). We solve the system `(I+diag(beta)A) delta = rhs` directly by forward-substitution —
  same answer, one fewer matmul, and a cleaner map to a per-chunk HIP solve_tri.
- **cumsum convention.** Inclusive `G_i` (not exclusive) is what makes the alpha-fold telescope;
  an exclusive cumsum shifts every decay exponent by one gate and breaks parity.

## q8_ef state-carry across chunk boundaries (the load-bearing extension)

hipfire's DeltaNet state can be **Q8 with sigma-delta error-feedback** (`DeltaNetState.s_ef_residual`,
`qwen35.rs:909-960` — per-head int8 codes + per-row absmax scale + f16 per-element residual,
DEFAULT-ON since 2026-06-08). The decode kernel requants **per token**
(`gated_delta_net_q8.hip:124-159`): fold the carried f16 residual into the state *before* absmax,
deterministic round-to-nearest, re-store the fresh residual `efr = val − qf·scale` (captures round
*and* clamp-saturation). This yields ~FP32 coherence at Q8's byte size.

**The chunked claim:** S is requantized only at each **chunk boundary** (not per token), so the EF
residual must carry the boundary quant error forward across chunks. Ref `gdn_chunk_q8ef.py`
(`gdn_chunked_q8`): each boundary dequants Q8→float (= `S_in`), runs the parity-proven f32 chunk →
`S_out` float, then requants `S_out`→Q8 with EF **exactly per `gated_delta_net_q8.hip:130-159`**
(fold residual before absmax, deterministic round, per-row scale = absmax/127, store fresh f16
residual). The state representation is byte-identical to the existing per-token path — only the
requant *cadence* changes (per-chunk instead of per-token).

### One real bug found en route (state divergence, not an EF bug)
Initial NaN / f16-overflow was the **state diverging** (S absmax 38→2972→2.7e5→1.6e7 across chunks):
random `k` was not L2-normalized, so `outer(delta, k)` accumulated unboundedly. DeltaNet
L2-normalizes q/k per token (the delta-rule is a bounded memory *overwrite*). After normalizing,
the recurrence is contractive and stable. This also surfaced the regime dependence below.

### Accuracy result (rel err = max|truth − out| / max|truth|, HD=128 real head dim)
EF only earns its keep in the **near-1 (long-retention) decay regime** — under strong contraction
the decay self-heals quant error and EF ≈ NOEF, so both regimes are tested.

**Near-1 decay (α≈0.985–0.9995, the load-bearing regime), growth vs chunk count (CS=16):**

| #chunks | q8_ef | q8_NOEF | seq_q8_ef |
|--:|--:|--:|--:|
| 1  | 6.87e-3 | 6.87e-3 | 6.29e-3 |
| 8  | 6.14e-3 | 8.66e-3 | 7.63e-3 |
| 16 | 6.03e-3 | 9.45e-3 | 6.39e-3 |
| 48 | 7.48e-3 | **1.16e-2** | 7.58e-3 |

- slope/chunk: **q8_ef = +1.9e-5 (flat)** vs **q8_NOEF = +1.07e-4 (5.6× steeper, drifting up)**;
  over 1→48 chunks q8_ef grows 1.09× while q8_NOEF grows 1.69×.
- 12-head aggregate (256 tok = 16 chunks): q8_ef mean **7.3e-3**, seq_q8_ef mean **7.6e-3**,
  q8_NOEF mean **1.15e-2**. Boundary-EF tracks per-token-EF (ef/seq = 0.96) and beats the
  no-feedback control by 1.58×.

**Mid decay (α≈0.5–0.95, strongly contractive):** all three variants ~5e-3, flat — decay damps
quant error so EF and NOEF are indistinguishable. Confirms the chunked-Q8 math is correct/bounded,
but this is not where EF matters.

### Verdict (all four claims pass)
1. **Boundary-EF stays bounded** — no growth with chunk count (16/1 chunk = 1.09×). ✅
2. **Boundary-EF ≈ per-token-EF coherence band** — ef/seq mean ratio 0.96. ✅
3. **EF beats the NOEF control** — noef/ef mean ratio 1.58. ✅
4. **NOEF drifts up with chunks while EF is flat** — slope_NOEF +1.07e-4 vs slope_EF +1.9e-5. ✅

The load-bearing idea holds: requantizing S only at chunk boundaries and carrying the EF residual
across them keeps the chunked state ~FP32 across arbitrarily many chunks, at Q8's byte container.

## Kernel plan

### What hipfire already has (reuse)
- The **GDN sequential kernels** for the `n==1` arm and as the parity oracle:
  `gated_delta_net_f32_batch_seq.hip` / `gated_delta_net_q8.hip`, launched from
  `crates/rdna-compute/src/norm.rs:2479` (f32) and `:2111` (q8). Dispatch sites in qwen35.rs
  (`:8804` f32 / `:8816` q8 prefill; `:10628`/`:10640` the second path; tree variants at
  `:8752`/`:8772`).
- The **Q8-EF state representation** — `DeltaNetState` (`qwen35.rs:902-960`): per-head int8 codes,
  per-row absmax scales (`s_scales`), f16 `s_ef_residual`, `HIPFIRE_DN_STATE_EF` toggle. The
  chunked path requants into the *same* tensors; only the cadence changes (per-chunk vs per-token).
- **GEMM primitives** for the chunk body (`Q@Kᵀ`, `K@Kᵀ`, `Mkq@delta`, the `Sᵀ@K` carry, the
  `S_in@q`/`S_in@k` reads): `crates/rdna-compute/src/gemm.rs`,
  `crates/hipfire-dispatch/src/families/gemm.rs`, `tables/gemm_table.rs`. These are all small
  dense matmuls (`C×C`, `C×HD`, `HD×HD` with C∈{16,32,64}) — no new GEMM needed.

### What is net-new (the three primitives)
A grep of `kernels/src/` confirms **none of these exist today**:
1. **`solve_tri` / forward-substitution** (the crux). A per-chunk unit-lower-triangular solve
   `(I + diag(beta)·A) delta = rhs`, `A` strict-lower `C×C`, `rhs` `C×HD`. Implemented as the
   validated `forward_subst_unit_lower` (`x_i = b_i − Σ_{l<i} L[i,l]·x_l`, unit diagonal ⇒ no
   division). C is small (16–64), so a single-workgroup sequential-in-`i`, parallel-over-`HD`-and-
   RHS-columns kernel suffices for the PoC. This is the one piece that makes chunked DeltaNet ≠
   chunked linear attention.
2. **inclusive cumsum** of the log-gate within a chunk → `G_i` (`C` elements, a warp scan).
3. **decay-mask / tri build** — `D[i,j]=exp(G_i−G_j)`, then `tril(D*K@Kᵀ,-1)` and `tril(D*Q@Kᵀ,0)`.
   A `C×C` elementwise exp + masked multiply fused onto the GEMM epilogues.

### Dispatch (mirrors Lucebox's split)
```
n_seq_tokens == 1  ->  sequential gated_delta_net_{f32,q8}_batch_seq   (unchanged; nothing to chunk)
n_seq_tokens  > 1  ->  chunked GDN  (prefill AND spec-decode verify)
```
Wire at the existing prefill/verify dispatch sites (`qwen35.rs:8804/8816`, `:10628/10640`); the
`n==1` AR decode arm stays on the sequential kernel exactly as today.

### Staging (parity gate at every step)
1. **F32 prefill PoC, small CS (16 or 32).** Implement as a sequence of launches (cumsum +
   decay-mask + the GEMMs + a naive single-workgroup `solve_tri` + state carry) — correctness
   before fusion. **GATE:** per-LA-layer max-abs-diff vs `gated_delta_net_f32_batch_seq` < 1e-4
   (the numpy ref hits 1.3e-15; the kernel will be looser in f32 but must stay ≪ 1e-4). A divergent
   chunk form is silent memory corruption, so this gate is non-negotiable.
2. **q8_ef state.** Carry S as Q8 across chunk boundaries, requant per `gated_delta_net_q8.hip:130-159`
   with the EF residual folded in. **GATE:** rel-err vs F32 truth in the q8_ef band (~7e-3 near-1,
   ~5e-3 mid) and flat across chunk count, matching `gdn_chunk_q8ef.py`. Then the full
   `coherence-gate.sh` + `serve-multiturn-gate.sh` (state-bleed across requests).
3. **Spec-decode verify.** Route the DFlash/DDTree verify (`n>1`) through chunked; re-run
   `coherence-gate-dflash.sh` (τ + token-attractor tiers). The verify is where the tree's extra
   candidates get cheap — the spec-decode payoff.

## Effort
The **triangular solve** is the only real new primitive; cumsum and the decay-mask are warp-scan /
fused-epilogue work, and every matmul reuses existing GEMM. The parity gate (now anchored at
1.3e-15 in the reference) keeps each stage honest, and the prefill payoff is spec-independent so it
can't fully whiff. This is the first spec-decode lever grounded in a concrete, validated math
difference rather than a microbench.

## Refs
- Parity ref (ground truth): `kernels/src/gated_delta_net_f32_batch_seq.hip:58-92` (sequential).
- Q8-EF requant convention: `kernels/src/gated_delta_net_q8.hip:124-159`;
  state struct `crates/hipfire-arch-qwen35/src/qwen35.rs:902-960`.
- GDN launch sigs: `crates/rdna-compute/src/norm.rs:2479` (f32) / `:2111` (q8).
- Validated numpy references (tooling, no GPU / not in the hot path):
  - `/home/kaden/.claude/jobs/3b7dee40/tmp/gdn_chunk_parity_3.py` — F32 chunked == sequential,
    max out-diff 1.332e-15.
  - `/home/kaden/.claude/jobs/3b7dee40/tmp/gdn_chunk_q8ef.py` — q8_ef boundary error-feedback,
    all four claims pass.
- Algorithm ref (NOT code — divergent GGML/f32 formulation):
  `Luce-Org/llama.cpp-dflash-ggml` `src/models/delta-net-base.cpp:16-289`
  (`build_delta_net_chunking`), dispatch `:426-447`.
