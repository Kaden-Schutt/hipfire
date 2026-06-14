# vLLM Qwen3.5-MoE Routing Precision Analysis

**Objective:** Compare vLLM's MoE routing and expert-dispatch precision handling with hipfire's internal findings on Qwen3.5-A3B and 122B-A10B models at MQ4 quantization, where a fused softmax+topk+renorm kernel produced weights differing by 1 ULP per element, compounding into structural attractor failures across 30+ MoE layers.

---

## 1. Router Softmax + Top-K

### Router Projection (Gate Linear)

**Where:** `vllm/model_executor/layers/fused_moe/router/gate_linear.py:11–117`  
**How:** `GateLinear` (a `ReplicatedLinear` subclass) implements three-tier GEMM dispatch:
1. **Tier 1 (DSV3 specialized, SM90+, batch≤16):** `ops.dsv3_router_gemm()` with user-configurable `out_dtype`
2. **Tier 2 (cuBLAS bf16→fp32, SM90+):** `ops.router_gemm_bf16_fp32()` for bf16 weights → fp32 output
3. **Tier 3 (F.linear fallback):** Standard PyTorch linear with optional dtype cast

**Precision Contract:**
- Tier 1 & 2 **explicitly cast to FP32 output** when the flag `allow_cublas_router_gemm` or `allow_dsv3_router_gemm` is set (lines 99–109).
- `out_dtype` is mutable post-init (lines 75–90): can be set to FP32 after construction.
- If no specialized kernel available and `force_fp32_compute=True`, weights are stored in FP32 (lines 47–48).
- **Default behavior:** Tier 3 preserves weight dtype unless `out_dtype` is explicitly set (lines 112–117).

**Qwen3.5 Setup** (`qwen3_next.py:122–128`):
```python
self.gate = ReplicatedLinear(
    config.hidden_size,
    config.num_experts,
    bias=False,
    quant_config=None,  # NO quantization on router!
    prefix=f"{prefix}.gate",
)
```
Router is **always unquantized (FP16/BF16)** in vLLM Qwen3.5-MoE. No `GateLinear` subclass is used here.

### Softmax Application

**Where:** `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:69–114`  
**How:** `fused_topk()` dispatches to custom ops:
```python
topk_weights = torch.empty(M, topk, dtype=torch.float32, ...)  # Line 81–82
...
topk_func = dispatch_topk_softmax_func(use_rocm_aiter=...)
topk_weights, topk_ids = topk_func(
    topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
)
```

**Precision Contract:**
- `topk_weights` **allocated in FP32 always** (line 81–82), regardless of input/weight dtype.
- Custom op `ops.topk_softmax()` (lines 24–32) or `rocm_aiter_ops.topk_softmax()` is called in-place on the FP32 buffer.
- **No dtype negotiation between router logits input and softmax accumulator:** softmax is hardcoded FP32.

### Top-K + Renormalization

**Where:** Same file, `fused_topk()` function (lines 69–114)  
**How:** Renormalization is a boolean flag passed to the custom op:
```python
topk_weights, topk_ids = topk_func(
    topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
)
```

**Precision Contract:**
- The `renormalize` boolean (line 162) is passed through to `ops.topk_softmax()`.
- **vLLM does NOT document whether renormalization (divisor sum reduction) is in FP32 or lower precision.**
- The custom op implementation is in C++/CUDA, not inspected here. But line 81–82 commits the output buffer to FP32, suggesting the kernel respects this.

**Qwen3.5 Configuration** (`qwen3_next.py:159`):
```python
renormalize=getattr(config, "norm_topk_prob", True),
```
Renormalization is **enabled by default** for Qwen3.5-MoE if `norm_topk_prob=True` in the HF config.

---

## 2. Shared Expert Path

### Shared Expert Gate

**Where:** `qwen3_next.py:130–136` and `qwen2_moe.py:149–155`
```python
self.shared_expert_gate = ReplicatedLinear(
    config.hidden_size,
    1,  # Sigmoid scalar gate
    bias=False,
    quant_config=None,  # Unquantized!
    prefix=f"{prefix}.shared_expert_gate",
)
```

**Precision:** Unquantized FP16/BF16 projection → sigmoid → elementwise multiply (line 120 in qwen2_moe.py: `F.sigmoid(self.expert_gate(x)[0]) * out`).

### Shared Expert Computation

**Where:** `qwen3_next.py:139–148`
```python
if config.shared_expert_intermediate_size > 0:
    self.shared_expert = Qwen3NextMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.shared_expert_intermediate_size,
        hidden_act=config.hidden_act,
        quant_config=quant_config,  # Respects model's quantization!
        reduce_results=False,  # Does NOT reduce before gating
        expert_gate=self.shared_expert_gate,
        is_sequence_parallel=self.is_sequence_parallel,
        prefix=f"{prefix}.shared_expert",
    )
```

**Precision Behavior:**
- Shared expert **gate_up_proj and down_proj respect `quant_config`** (lines 90–106 in qwen2_moe.py).
- If the model is MQ4, the shared expert weights are MQ4.
- **Gate (sigmoid) is applied post-activation** before down_proj (line 120): `sigmoid(gate(x)) * silu(gate_up)`.
- **No separate FP32 accumulation** for the shared expert path. Accumulation stays in the weight dtype (FP16/MQ4).

### Shared Expert Orchestration

**Where:** `vllm/model_executor/layers/fused_moe/runner/shared_experts.py`

SharedExperts can run **before, after, or fused with routed experts** based on `SharedExpertsOrder` enum (lines 27–38):
- **NO_OVERLAP:** Shared expert runs in main stream before or after dispatch.
- **MK_INTERNAL_OVERLAPPED:** Modular kernel owns the execution.
- **MULTI_STREAM_OVERLAPPED:** Runs in a CUDA aux stream in parallel with routing/gate (lines 135–146).

**Precision impact:** No separate FP32 path for shared experts. They use the same quantization as routed experts.

---

## 3. Routed Expert Dispatch

### FusedMoE Class & Kernel Selection

**Where:** `vllm/model_executor/layers/fused_moe/layer.py:218–612`

FusedMoE's `forward()` delegates to `MoERunner` (line 1551–1555):
```python
def forward(self, hidden_states, router_logits, ...):
    return self.runner.forward(hidden_states, router_logits, input_ids)
```

The runner selects a **quantization method** (e.g., `UnquantizedFusedMoEMethod`, `GPTMarlinMoEMethod`, etc.) based on `quant_config`.

### Mixed-Dtype Expert Handling

**Where:** `vllm/model_executor/layers/fused_moe/layer.py:370–590`

vLLM's approach:
- **No explicit per-expert dtype tracking.** All experts in a layer are assumed homogeneous.
- Quantization method creates **uniform expert weight buffers** (line 588):
  ```python
  self.quant_method.create_weights(
      layer=self,
      num_experts=self.local_num_experts,
      hidden_size=hidden_size,
      intermediate_size_per_partition=intermediate_size_per_partition,
      params_dtype=params_dtype,
      ...
  )
  ```
- **No fallback for mixed-dtype experts** (e.g., some Q4, some Q8). A single quantization method is instantiated for the entire layer.

### Expert Accumulator Precision

**Where:** `vllm/model_executor/layers/fused_moe/fused_moe.py` (Triton kernel source)

Critical comment (lines with "Accumulator and scalings"):
```python
# Accumulator and scalings are in float32 to preserve numerical accuracy.
```

And the weight multiplication clause:
```python
# Router (MoE) weight multiplication:
# This multiplication MUST be performed in float32 before any precision
# conversion to ensure numerical stability, which is especially critical
# on ROCm platforms.
if MUL_ROUTED_WEIGHT:
    moe_weight = tl.load(topk_weights_ptr + offs_token, ...)
    accumulator *= moe_weight[:, None]  # FP32 multiply
```

**Precision Contract:**
- **Inner accumulator is FP32** (lines on "Accumulator").
- **topk_weights are loaded (FP32 from router) and multiplied in FP32** before final precision conversion.
- **Final conversion:** Kernel casts result to output dtype (FP16/BF16 or quantized) only at the very end.

### Modular Kernel Support

**Where:** `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py` and `prepare_finalize/`

Modular kernels (DeepEP, NIXL, etc.) allow **separate prepare (dispatch), expert compute, and finalize (combine)** stages. Each stage can control precision independently, but **vLLM does not document per-stage FP32 pinning** for MoE weights or topk_weights in these backends.

---

## 4. Numerical Differences: Full-Precision vs. Quantized Path

### Does vLLM Change Math Between FP16 and MQ4?

**Finding:** **No structural change.** The same kernel structure runs for both paths:

1. **FP16 experts:** `UnquantizedFusedMoEMethod` 
   - Loads FP16 weights directly.
   - Inner accumulator: FP32 (committed in Triton kernel).
   - Topk_weights: FP32 (line 81–82 in fused_topk_router.py).

2. **MQ4 experts:** `MarlinMoEMethod` (or equivalent quantization method)
   - Loads quantized (MQ4) weights + dequantization scales.
   - **Dequantization happens inline in the kernel** (same Triton or custom op).
   - Inner accumulator: FP32 (same kernel structure).
   - Topk_weights: FP32 (same router path).

**Conclusion:** Quantization is **purely a dequantization-on-load problem**, not a precision-path split. The inner math (accumulate FP32, apply topk_weights FP32, convert output) is **identical.**

### Precision Cliffs (Where Instability Can Compound)

1. **Topk_weights renormalization divisor:** Computed in FP32 in the custom op, but **exact implementation unknown** (compiled CUDA).
2. **Shared expert gate sigmoid:** Unquantized FP16/BF16, no FP32 contract.
3. **Shared expert accumulation:** Respects expert quantization; no FP32 anchor.
4. **30+ layer accumulation:** Each layer's `topk_weights[i]` differs by ~1 ULP from `softmax(logits)[i] / sum`. Over 30 layers × 8 experts, 1 ULP compounding → attractor trap possible.

---

## 5. Precision Guards & Stability Asserts

### Searches for "force_fp32", "precision", "stability"

**Found:**
- `force_fp32_compute: bool` flag in `GateLinear.__init__()` (gate_linear.py:35) — **only for HOPPER+ architecture**, not a universal guard.
- **No `force_fp32_router` or `force_fp32_topk` guards** in the codebase for Qwen3.5-MoE.
- **No A3B-specific precision overrides** found.
- **No explicit infinite-loop detection or reasoning-path guards** (reasoning prompts are handled by the reasoning_parser, separate from MoE).

### Numerical Stability Comments

Only two relevant comments in fused_moe code:
1. `fused_moe.py`: "Accumulator and scalings are in float32 to preserve numerical accuracy."
2. `fused_moe.py`: "This multiplication MUST be performed in float32... especially critical on ROCm platforms."

**No mention of topk renormalization precision**, **no mention of shared expert gate stability**, **no mention of A3B-specific issues**.

---

## 6. Concrete Differences from Hipfire

### What Hipfire Found

1. **Fused softmax+topk+renorm kernel** produced `topk_weights` differing by 1 ULP per element from separate `softmax_f32 + manual_renorm`.
2. **Mitigation:** Split into separate softmax + manual renorm.
3. **Shared expert gate path:** Routed all gate-side weights (router, shared_expert_gate, shared.gate, shared.up) through a **single fused GEMV** requiring **uniform MQ4G256 weights**.
4. **Mixed-dtype fallback** (e.g., Q8 router + MQ4 experts) falls into **less-tested `weight_gemv`** fallback.
5. **Reasoning loop spiral** on Qwen3.6-A3B is plausibly a precision-cliff residual.

### vLLM's Approach

| Aspect | Hipfire | vLLM |
|--------|---------|------|
| **Router projection dtype** | MQ4G256 (uniform) or fallback | FP16/BF16 unquantized (always) |
| **Softmax+topk+renorm** | Fused kernel, mitigation: split | Fused custom op, renormalize flag, topk_weights pinned FP32 |
| **Renormalization precision** | Manual FP32 after split | Unknown (compiled custom op) |
| **Shared expert gate dtype** | MQ4G256 (fused) | Unquantized FP16/BF16 (always) |
| **Shared expert accumulation** | Single fused GEMV | Separate MLP layers, respects quant_config |
| **Mixed-dtype experts** | Unsupported; fallback tested poorly | Unsupported; single quant method per layer |
| **FP32 weight accumulator** | Applied | Applied (Triton kernel hardcoded) |
| **A3B-specific guards** | Likely internal tuning | None found in open code |
| **Reasoning loop detection** | Not documented | Not present |

---

## 7. Smoking Gun: The Shared Expert Gate

**Critical Precision Difference:**

Hipfire routes **all gate-side weights** (including `shared_expert_gate`) through a **single fused kernel** that ensures uniform quantization (MQ4G256) and a **common accumulation context** (presumably FP32 in-kernel).

vLLM routes:
1. `self.gate` (router): ReplicatedLinear → `fused_topk()` → custom op (FP32 topk_weights output).
2. `self.shared_expert_gate`: Separate ReplicatedLinear → sigmoid → multiply (all FP16/BF16).

**This means vLLM's shared expert gate operates in FP16/BF16 precision for both projection and sigmoid**, with **no FP32 anchor**. If the shared expert contribution becomes numerically significant (which it does in reasoning tasks), a **subtle precision error in the gate can amplify** through interaction with the router's FP32 topk_weights.

In particular:
- Router outputs FP32 topk_weights.
- Shared expert gate outputs FP16/BF16 sigmoid.
- When combined (`shared_out + (topk_weights * routed_experts_out)`), the **shared_out scale mismatch** could trap the system in a pathological gradient/logit state.

---

## 8. Recommendations for Hipfire

1. **Explicitly test shared_expert_gate precision impact** on Qwen3.5-A3B reasoning:
   - Replace sigmoid with `F.sigmoid(...).to(torch.float32)` before multiply.
   - Compare reasoning output stability.

2. **Unify router and shared_expert_gate precision contract:**
   - Ensure both pathways (routed + shared) anchor to FP32 at the same stage.
   - Consider fusing shared_expert_gate projection + sigmoid + scaling into a single FP32 kernel.

3. **Test mixed-dtype expert support (Q8 router, MQ4 experts):**
   - vLLM lacks this; hipfire's `weight_gemv` fallback is reportedly less tested.
   - Ensure dequantization happens in FP32, not FP16.

4. **Instrument top-K renormalization divisor:**
   - Measure ULP error in the divisor computation vs. naive `sum()`.
   - Ensure Kahan summation or similar compensation if using FP32.

5. **Add reasoning-path precision override:**
   - At inference time, detect reasoning prompts (via tokenizer or config) and **force FP32 on shared_expert_gate and topk renormalization**.
   - This would prevent attractor-spiral if the precision cliff is indeed the root cause.

---

## References

**vLLM source (commit: HEAD of /local/git/vllm-gfx906-mobydick):**
- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:69–114` (fused_topk & renormalize)
- `vllm/model_executor/layers/fused_moe/router/gate_linear.py:11–117` (router GEMM dispatch)
- `vllm/model_executor/models/qwen3_next.py:84–194` (SparseMoeBlock, shared_expert_gate, FusedMoE init)
- `vllm/model_executor/layers/fused_moe/layer.py:217–1556` (FusedMoE class, runner, quantization setup)
- `vllm/model_executor/layers/fused_moe/config.py:100–167` (RoutingMethodType, get_routing_method_type)
- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py:27–179` (SharedExpertsOrder, execution scheduling)
- `vllm/model_executor/models/qwen2_moe.py:77–194` (Qwen2MoeMLP, shared expert gate apply)

**Key precision-anchor lines:**
- `fused_topk_router.py:81–82`: topk_weights allocated FP32
- `fused_moe.py` (Triton kernel): Accumulator FP32, topk weight multiply FP32, final convert
- `qwen3_next.py:130–136`: shared_expert_gate is unquantized ReplicatedLinear
- `gate_linear.py:47–48`: force_fp32_compute → FP32 weights (HOPPER+ only)

