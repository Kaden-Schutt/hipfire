# Adversarial Review: Gemma 4 Forward-as-Pipeline Migration Plan

This document provides a detailed adversarial review of the proposed migration plan in [gemma4_forward_as_pipeline.md](file:///home/kread/git/hipfire/docs/plans/gemma4_forward_as_pipeline.md). It outlines critical design flaws, correctness issues, conceptual errors, and optimization feedback to ensure the implementation is robust, correct, and matches the rest of the hipfire codebase.

---

## Summary of Key Findings

| Finding ID | Category | Severity | Description |
|---|---|---|---|
| **REV-01** | Correctness / Design | **Critical** | Full attention layer lacks `v_proj` weights and uses separate Q/K projections; the proposed `Proj(QKV)` + `Proj(V)` sequence will crash or fail to compile. |
| **REV-02** | Correctness | **High** | Potential duplication of post-attention RMSNorm between `Norm(POST_ATTN)` and `ResidualGemv(POST_ATTN)` will lead to model divergence. |
| **REV-03** | Design / Redundancy | **Medium** | Proposed `ResidualGemv(POST_FFN)` adjustment for MoE is unnecessary as combination is already encapsulated inside `apply_moe_branch`. |
| **REV-04** | Architecture | **Medium** | Misunderstanding of Qwen 3.5's `hidden_rb` gate leads to gating off the lowered path under hipGraph capture, losing performance benefits. |
| **REV-05** | Conceptual | **Low** | Prefill is described as latency-dominated instead of throughput-dominated. |
| **REV-06** | Usability / Naming | **Low** | Reference to `forward_scratch_layers` does not match the actual `forward_scratch_inner` function name in the codebase. |

---

### REV-01: Full Layer Projections vs. QKV Projection

> [!CAUTION]
> The proposed program layouts for `FullDense` and `FullMoe` assume a `Proj(QKV)` followed by `Proj(V)` step. This is a critical design bug that will cause compile-time or runtime failures.

#### Analysis
In Gemma 4's full attention layers (global layers):
1. **No `v_proj` exists:** The architecture uses `k_eq_v = true`, meaning `V` is a copy of `K`'s pre-norm projection. There is no `v_proj` weight tensor in `FullLayerWeights`. Running `Proj(V)` will fail because there is no weight tensor to bind.
2. **Q and K projections are separate:** Unlike the sliding layers where `q_proj` and `k_proj` are executed in a single `execute_steps` call (which the plan refers to as `Proj(QKV)`), the full layers execute separate `weight_gemv` calls for `q_proj` and `k_proj` independently.

#### Recommendation
- Replace the `Proj(QKV)` + `Proj(V)` sequence in the `FullDense` and `FullMoe` program layouts with two separate projection steps: `Proj(Q)` (opcode `PROJ_Q`) and `Proj(K)` (opcode `PROJ_K`).
- Perform the `k_eq_v` prelude (copying `scratch.k` to `scratch.v` before `k_norm` is applied to `scratch.k`) as part of the `Attend(FULL)` super-op execution.

```diff
 FullDense:
   Norm(INPUT)
-  Proj(QKV)
-  Proj(V)
+  Proj(Q)            — q_proj via weight_gemv
+  Proj(K)            — k_proj via weight_gemv
   Attend(FULL)       — q/k/v_norm + partial rope + k_eq_v prelude + kv_write + flash_attn
```

---

### REV-02: Post-Attention RMSNorm Duplication

> [!WARNING]
> Implementing the plan's Step 4 as written will run the post-attention RMSNorm twice, producing corrupted hidden states and causing immediate validation failure.

#### Analysis
In the plan, the `SlidingDense` program defines:
```
  Norm(POST_ATTN)    — rmsnorm
  ResidualGemv(POST_ATTN) — memcpy residual + add
```
However, the text under **Step 4 — Wire up ResidualGemv** states:
* `- RESID_POST_ATTN: memcpy residual + add_inplace + post_attn_norm + memcpy residual again`

If `post_attn_norm` is executed inside the `RESID_POST_ATTN` handler *and* as a standalone `Norm(POST_ATTN)` super-op, it will be executed twice.

#### Recommendation
Clarify that `Norm(POST_ATTN)` handles the RMSNorm (writing the normalized output to `scratch.tmp`), and `ResidualGemv(POST_ATTN)` only handles the residual logic:
1. Copy `scratch.residual` to `scratch.x`.
2. Add `scratch.tmp` to `scratch.x` in-place.
3. Copy `scratch.x` to `scratch.residual` (to store the post-attention hidden state for the FFN/MoE residual stream).

---

### REV-03: Redundant MoE Adjustment in `ResidualGemv(POST_FFN)`

> [!NOTE]
> The proposed adjustment in Step 5 for `RESID_POST_FFN` under the MoE variant is redundant and should be simplified.

#### Analysis
Step 5 states:
* `Adjust RESID_POST_FFN to handle the MoE variant (which has a different residual combination: x = cur_mlp + moe_out + residual)`

However, in the hand path implementation (`apply_moe_branch` in `gemma4.rs`), the MoE branch combines the dense MLP output and MoE output internally:
```rust
    // combined = cur_mlp + cur_moe → scratch.tmp
    gpu.add_f32(&scratch.moe_cur_mlp, &scratch.moe_cur_moe, &scratch.tmp)?;
    // tmp = post_feedforward_layernorm(combined)
    gpu.rmsnorm_f32(&scratch.tmp, post_ffn_norm, &scratch.tmp, config.norm_eps)?;
```
As a result, `scratch.tmp` is already populated with the normalized combined output on exit from `Moe(MOE_BRANCH)`. 
At the layer level, the residual connection is simply `x = residual + tmp`. This is identical to the dense FFN path.

#### Recommendation
Keep `ResidualGemv(POST_FFN)` identical for both dense and MoE layers. It only needs to load the residual, add `scratch.tmp`, and apply the layer scalar. No MoE-specific branch is needed inside this handler.

---

### REV-04: Graph Capture Gating Misunderstanding

> [!IMPORTANT]
> Gating off the lowered path whenever a hipGraph is captured or replayed will degrade performance.

#### Analysis
The plan states:
* `Gate off for graph-capture mode (same as qwen35: hidden_rb.is_none() equivalent)`

This is a misunderstanding of Qwen 3.5's implementation. In `qwen35.rs`, the `hidden_rb` check gates off the lowered path during speculative decoding (DFlash) verification, NOT during standard `hipGraph` capture/replay. Standard autoregressive `hipGraph` capture in Qwen 3.5 runs with `hidden_rb = None` and successfully records and replays the lowered path.

If Gemma 4 gates off the lowered path during any graph capture, it will lose all CPU overhead reduction benefits in graph replay mode (which is default-on in many production environments).

#### Recommendation
Ensure the lowered path is fully compatible with hipGraph capture. Do not gate it off under `use_graph`. Since Gemma 4 does not use a `HiddenStateRingBuffer` for speculative validation in the same way, this bypass is not needed for general graph execution.

---

### REV-05: Prefill Latency/Throughput Conceptual Error

#### Analysis
Under "Non-goals (deferred)", the plan states:
* `Prefill is latency-dominated, not throughput-dominated, so the per-resolve overhead matters less.`

This statement is conceptually backwards:
* **Prefill** processes a large number of prompt tokens at once, making it compute-bound and **throughput-dominated**.
* **Decode** processes one token at a time, making it memory-bandwidth bound and **latency-dominated**.

The decision to defer prefill migration is correct because the resolution overhead is amortized over a large number of tokens (making it negligible per token). However, the explanation should be corrected to reflect accurate terminology.

---

### REV-06: Codebase Naming Discrepancies

#### Analysis
The plan repeatedly refers to `forward_scratch_layers` as the target for gating and code removal.
However, in the Gemma 4 codebase, the layers loop function is called `forward_scratch_inner`, not `forward_scratch_layers` (which is the function name in Qwen 3.5).

#### Recommendation
Update references in the plan from `forward_scratch_layers` to `forward_scratch_inner` to avoid developer confusion when searching the codebase.
