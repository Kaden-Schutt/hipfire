# Adversarial Review: `mtp_multi_refactor.md`

**Reviewer:** Gemini CLI (Adversarial Mode)
**Date:** May 28, 2026
**Subject:** Refactoring toward unified `generate_qwen35` (Stage 2b)

---

## 1. Architectural Risks: The "God Function" Trap

The plan proposes collapsing three (soon to be four) distinct generation paths into a single `generate_qwen35` function. While this reduces code duplication, it risks creating a "God Function" that is difficult to audit, maintain, and debug.

*   **State Explosion:** The function signature already includes `m`, `gpu`, `drafter_gpu`, `pflash_state`, `pflash_cfg`, `tools`, `messages_history`, etc. Adding `PrefillCtx` and `SpecPath` increases the cognitive load. 
*   **Borrow Checker Fragility:** Threading `&mut Gpu` vs. `&mut Gpus` through a unified function is a recipe for `borrowck` nightmares. If `m.pp_gpus` contains `gpu` (the first device), Rust will likely forbid simultaneous mutable access to both. The plan mentions `gpu` is "dev 0 for both pp=1 and pp>1", but `generate_multi` typically needs the whole `Gpus` set. If the unified function tries to hold both a specific `&mut Gpu` and the `&mut Gpus` container, it will fail to compile.
*   **Implicit vs. Explicit Refusal:** Collapsing refusal logic into a single match statement (#3) is clean, but it hides *why* certain combinations are refused. If the load handler and the generate function drift, we might end up with "Refused" branches that are unreachable or, worse, reachable but undefined.

## 2. Implementation Hazards: The "Surgery" Commit

Step 5 ("The unification") is described as a surgery commit. In complex Rust codebases, "surgery" usually means "everything is broken for 12 hours."

*   **The `PrefillContext` Leakage:** If `PrefillContext` only handles the prefill batch, but the decode loop still needs to distinguish between `Gpu` and `Gpus`, the abstraction is "leaky." Every line in the decode loop that interacts with the GPU will need a match statement or a similar abstraction, potentially negating the LOC savings.
*   **MTP Speculative State Divergence:** `spec_step_mtp_compressed_serial` (single) and `spec_step_mtp_compressed_serial_hetero` (multi) have different expectations for where the drafter state lives. Merging them requires extreme care to ensure that `drafter_gpu` is correctly assigned and that `peer_clone_tensor` shortcuts don't accidentally trigger on non-peer devices.
*   **PFlash composition:** The plan suggests enabling PFlash + MTP "silently." This is dangerous. PFlash changes the KV cache layout/indexing for the prompt. MTP relies on specific KV offsets for speculative verification. If they don't agree on the "ground truth" KV state, we'll get silent coherence failures that are extremely hard to debug.

## 3. Testing Gaps: The Permutation Problem

The plan relies on `coherence-gate.sh`, but the number of permutations is growing:
1. AR (Single)
2. AR (Multi)
3. MTP (Single)
4. MTP (Multi) - *New*
5. DFlash (Single)
6. PFlash + AR (Multi)
7. PFlash + MTP (Multi) - *Proposed*

*   **Combinatorial Explosion:** Each additional feature (PFlash, DFlash, MTP, PP) doubles the test surface. A single `coherence-gate.sh` run might not cover the specific edge case where, for example, PFlash + PP + MTP interacts with a specific context length.
*   **Performance Regressions:** The plan notes a 10% tolerance for `tok/s`. This is quite high. A unified function often introduces "branchiness" that can thrash the instruction cache or lead to suboptimal register allocation in the hot loop. We should aim for 0% regression on the AR baseline.

## 4. Performance & Resource Concerns

*   **VRAM Management:** `load_model_pp` already handles MTP head load. Does the unified `generate_qwen35` ensure that we aren't re-allocating `Scratch` sets or speculative buffers?
*   **Latency of the Match:** While negligible compared to a GPU kernel, placing a large match statement inside the token-by-token decode loop adds CPU latency. In high-throughput scenarios (small models, fast GPUs), this can become a bottleneck.

## 5. Unaddressed Edge Cases

*   **Device Failure:** If one GPU in a PP chain fails, `generate_multi` usually errors out. Does the unified function handle the partial cleanup of `drafter_gpu` vs. `pp_gpus` correctly?
*   **Cancellation:** If a user cancels a request mid-generation, we need to ensure the KV cache state is consistent for the *next* request. A unified function might make it harder to track which "cleanup" path to take if a SpecPath was interrupted.
*   **Eviction:** The plan suggests "disable at load" for PP+MTP. This is safe but limits the utility of the refactor. If the goal is a "unified" path, it should ideally handle eviction for all paths or explicitly delegate it.

## Recommendations

1.  **Decompose before Unifying:** Instead of one giant function, create a `Generator` trait or a series of strategy-specific structs that share a `BaseGenerator` for the prelude/postlude. This avoids the "God Function" and the borrow checker issues.
2.  **Explicit PFlash+MTP Guard:** Do NOT enable PFlash + MTP "silently." Explicitly refuse it until a dedicated coherence test suite is added.
3.  **Benchmark Early:** Run a micro-benchmark on the unified match loop *before* the surgery commit to ensure no CPU-side regressions.
4.  **Split the Surgery:** Instead of one "surgery commit," migrate one path at a time (e.g., move `generate` to the new structure, then `generate_multi`, etc.).
5.  **Audit `drafter_gpu` Ownership:** Ensure that the `drafter_gpu: Option<&mut Gpu>` doesn't conflict with `m.pp_gpus`. If the drafter *is* one of the PP GPUs (specifically the output device), we need a safe way to borrow it twice or handle the alias.
