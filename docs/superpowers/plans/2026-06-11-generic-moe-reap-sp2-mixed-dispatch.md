# Generic MoE REAP — SP2: Mixed-Tier Bucketed Dispatch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a single MoE layer's routed experts run at DIFFERENT quant tiers (per-expert), by bucketing the top-k selected experts by tier and calling the existing single-tier indexed GEMV kernel once per tier — reusing today's kernels, no new kernel families.

**Architecture:** A pure `bucket_topk_by_tier` partitions the k selected experts by their tier (CPU, unit-tested). `MoeDtypes` gains `per_expert_gate_up/down: Option<Vec<DType>>` (`None` ⇒ today's uniform path, untouched) and `MoeResolution` a `mixed` flag. When mixed, `run_moe_decode` loops the buckets, calling the existing `gemv_*_moe_*_indexed` kernel per tier over that tier's subset of `topk_indices` (the kernels already address experts by per-expert pointer, so a subset of indices into the same `expert_*_ptrs` table works), accumulating into the same `down_expanded` output.

**Tech Stack:** Rust, `hipfire-dispatch` (`families/moe.rs`, `pipeline/mod.rs`), HIP kernels (existing, reused). The partition + resolve logic is CPU; the kernel-dispatch loop is GPU.

**Spec:** `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md` §2.

> ⛔ **GPU EMBARGO — verification split.** **CPU-verifiable now:** the `bucket_topk_by_tier` partition (Tasks 1) and the `MoeDtypes`/`MoeResolution` mixed-detection (Task 2) — unit-tested. **GPU-DEFERRED (implemented + code-reviewed, numerical gate owed):** the bucketed kernel-dispatch loop (Task 3) and arch population (Task 4). The **bucketing-equivalence gate** — a layer whose experts are all one tier, routed through the MIXED path, must produce bit-identical output to the uniform path — is the key correctness check and must run once the GPU frees, on BOTH `run_moe_decode` and the deepseek4 bias-aware/hash paths (the two-dispatch-site gotcha from the reap memory).

**Scope (this plan):** buffer arches (qwen35, lfm2moe) where per-expert dtypes are naturally representable (each expert is its own `WeightTensor`+pointer). **DEFERRED to a follow-on:** blob-packing arches (deepseek4, minimax) need per-tier SUB-BLOBS (their experts share one stride-addressed blob, which a single tier assumes) — a larger loader change; and the deepseek4 bias-aware (`run_moe_decode_bias_aware`) + hash (`ffn_hash_routed`) dispatch sites need the same bucket loop. This plan lands the partition primitive, the `MoeDtypes` plumbing, the `run_moe_decode` (generic) bucketed path, and qwen35/lfm2moe population — enough for a buffer-arch per-expert-mixed model to serve once GPU-verified.

---

### Task 1: `bucket_topk_by_tier` — pure partition (CPU)

**Files:**
- Create: `crates/hipfire-dispatch/src/families/moe_buckets.rs`
- Modify: `crates/hipfire-dispatch/src/families/moe.rs` (add `mod moe_buckets;` or inline) / `lib.rs`
- Test: inline

- [ ] **Step 1: Write the failing tests**

`moe_buckets.rs`:
```rust
use rdna_compute::DType;

/// One tier's slice of the selected experts: the tier's DType plus the
/// (rank, expert_index) pairs whose expert is at that tier. `rank` is the
/// position in the original top-k (0..k); `expert_index` is the routed-expert
/// id the kernel uses to index `expert_*_ptrs`.
#[derive(Debug, Clone, PartialEq)]
pub struct TierBucket {
    pub tier: DType,
    pub ranks: Vec<usize>,        // positions in the top-k
    pub experts: Vec<u32>,        // expert ids (parallel to ranks)
}

/// Partition the `k` selected experts (`topk` = expert ids, len k) by their
/// per-expert tier (`tier_of[expert_id]`). Returns one bucket per distinct
/// tier PRESENT among the selected experts, in first-seen order. A layer whose
/// selected experts are all one tier yields exactly ONE bucket (== uniform).
pub fn bucket_topk_by_tier(topk: &[u32], tier_of: &[DType]) -> Vec<TierBucket> {
    let mut buckets: Vec<TierBucket> = Vec::new();
    for (rank, &e) in topk.iter().enumerate() {
        let tier = tier_of[e as usize];
        match buckets.iter_mut().find(|b| b.tier == tier) {
            Some(b) => { b.ranks.push(rank); b.experts.push(e); }
            None => buckets.push(TierBucket { tier, ranks: vec![rank], experts: vec![e] }),
        }
    }
    buckets
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType::*;
    #[test]
    fn uniform_layer_is_single_bucket() {
        let topk = [3u32, 7, 1, 5];
        let tier_of = vec![MQ4G256; 8];
        let b = bucket_topk_by_tier(&topk, &tier_of);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].tier, MQ4G256);
        assert_eq!(b[0].ranks, vec![0,1,2,3]);
        assert_eq!(b[0].experts, vec![3,7,1,5]);
    }
    #[test]
    fn mixed_layer_partitions_by_tier_preserving_rank() {
        // experts: 0,2->MQ4 ; 1,3->MQ6
        let tier_of = vec![MQ4G256, MQ6G256, MQ4G256, MQ6G256];
        let topk = [1u32, 0, 3, 2]; // ranks 0..3
        let b = bucket_topk_by_tier(&topk, &tier_of);
        assert_eq!(b.len(), 2);
        // first-seen tier is MQ6 (expert 1 at rank 0)
        assert_eq!(b[0].tier, MQ6G256);
        assert_eq!(b[0].ranks, vec![0, 2]); assert_eq!(b[0].experts, vec![1, 3]);
        assert_eq!(b[1].tier, MQ4G256);
        assert_eq!(b[1].ranks, vec![1, 3]); assert_eq!(b[1].experts, vec![0, 2]);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p hipfire-dispatch moe_buckets` → 2 pass.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-dispatch/src/families/moe_buckets.rs crates/hipfire-dispatch/src/families/moe.rs
git commit -m "feat(reap): bucket_topk_by_tier partition primitive (CPU)"
```

---

### Task 2: `MoeDtypes` per-expert tiers + `MoeResolution.mixed`

**Files:**
- Modify: `crates/hipfire-dispatch/src/families/moe.rs` (`MoeDtypes` `:41`, `MoeResolution` `:56`, `resolve` `:66`)
- Test: inline

- [ ] **Step 1: Write the failing tests**

Add to `moe.rs` `#[cfg(test)]`:
```rust
#[test]
fn resolve_none_per_expert_is_not_mixed() {
    let d = MoeDtypes { /* all fields uniform MQ4, per_expert_*: None */ };
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.mixed);
}
#[test]
fn resolve_some_per_expert_with_varied_tiers_is_mixed() {
    let mut d = /* uniform MQ4 base */;
    d.per_expert_gate_up = Some(vec![DType::MQ4G256, DType::MQ6G256]); // varies
    d.per_expert_down    = Some(vec![DType::MQ4G256, DType::MQ6G256]);
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.mixed);
}
#[test]
fn resolve_some_per_expert_all_same_is_not_mixed() {
    // a per-expert table that is uniform should NOT trigger the mixed path
    let mut d = /* uniform MQ4 base */;
    d.per_expert_gate_up = Some(vec![DType::MQ4G256, DType::MQ4G256]);
    d.per_expert_down    = Some(vec![DType::MQ4G256, DType::MQ4G256]);
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.mixed, "a uniform per-expert table must take the fast uniform path");
}
```
(Fill the `MoeDtypes` literal from the real struct — all existing fields + the two new `Option<Vec<DType>>` defaulting to `None`.)

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p hipfire-dispatch resolve_` → FAIL (fields/`mixed` missing).

- [ ] **Step 3: Implement**

- Add `pub per_expert_gate_up: Option<Vec<DType>>` and `pub per_expert_down: Option<Vec<DType>>` to `MoeDtypes` (`:41`). Every existing constructor of `MoeDtypes` (grep — the arch `MoeDtypes` builders) must set them to `None` (uniform) — this keeps all current call sites on the unchanged path; the compiler will flag each.
- Add `pub mixed: bool` to `MoeResolution` (`:56`).
- In `resolve` (`:66`): compute `mixed = per_expert tables are Some AND contain >1 distinct DType` (a `Some` table that is all-equal is NOT mixed — it collapses to uniform). Set it on the returned struct. The existing uniform fields (`routed_indexable_*`, `use_gpu_topk`, etc.) stay computed from the representative `routed_gate_up`/`routed_down` as today; when `mixed`, the dispatch (Task 3) overrides the routed path.

- [ ] **Step 4: Run tests + whole crate**

Run: `cargo test -p hipfire-dispatch` → green (the 3 new + no regressions; every `MoeDtypes` builder now sets `None`).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-dispatch/src/families/moe.rs
git commit -m "feat(reap): MoeDtypes per-expert tier tables + MoeResolution.mixed (uniform=unchanged)"
```

---

### Task 3 [GPU — DEFERRED]: bucketed dispatch in `run_moe_decode`

**Files:**
- Modify: `crates/hipfire-dispatch/src/pipeline/mod.rs` (the indexed routed-expert dispatch `:377-435`)

- [ ] **Step 1: Implement the mixed branch**

In `run_moe_decode`, when `res.mixed`: instead of the single `gemv_*_moe_*_indexed` calls, build the per-tier buckets with `bucket_topk_by_tier(topk_host, &dtypes.per_expert_gate_up.as_ref().unwrap())` (the topk indices must be read host-side or a device-side partition prepared — for decode, the topk is small; reuse the existing CPU-topk fallback's host indices if available, else a single D2H of `topk_indices`). For each bucket, upload its compact `ranks`/`experts` index list to a scratch `GpuTensor` and call that tier's existing kernel (`gemv_hfq4g256_moe_gate_up_k8_indexed` / hfq6 / paro) over the bucket — gate_up into the bucket's rank slots of `gate_batch`/`up_batch`, then the fused activation, then the tier's `*_down_*_indexed_batched_expanded` into the bucket's rank slots of `down_expanded`. Because every bucket writes DISJOINT rank slots, accumulation into `down_expanded` is conflict-free.
- Per-tier strides: a mixed layer needs each tier's `routed_gate_up_k`/`routed_down_m`/`routed_down_k`. Add per-tier stride lookup to `MoeParams` (or derive from the tier's GpuTensor shapes). Thread it through.

- [ ] **Step 2: [GPU — DEFERRED] bucketing-equivalence gate**

Do NOT run under the embargo. Once GPU is free: a layer whose experts are ALL one tier, forced through the mixed path (`per_expert_* = Some(uniform table)`), must produce **bit-identical** `down_expanded` to the uniform path. This proves bucketing is an exact decomposition before any real tier mixing. Leave unchecked; note "deferred: GPU in use".

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-dispatch/src/pipeline/mod.rs
git commit -m "feat(reap): bucketed mixed-tier dispatch in run_moe_decode [GPU-gate deferred]"
```

---

### Task 4 [GPU — DEFERRED]: populate per-expert tiers in qwen35/lfm2moe

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs` (MoeDtypes builder ~`:4686`), `crates/hipfire-arch-lfm2moe/src/lfm2moe.rs`

- [ ] **Step 1: Implement**

Where each buffer arch builds `MoeDtypes` per layer, when the layer's experts do NOT all share one dtype (read each `ffn.experts[e].gate_up.gpu_dtype`/`.down.gpu_dtype`), populate `per_expert_gate_up`/`per_expert_down` with the per-expert dtype vec; otherwise leave `None` (uniform fast path). This is where an overlay (SP3) that re-quantized some experts surfaces as a mixed layer.

- [ ] **Step 2: [GPU — DEFERRED] serve check**

Build a per-expert-mixed qwen35 overlay (SP4a) + load (SP3) + serve; confirm output is sane. Deferred under embargo.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-arch-qwen35 crates/hipfire-arch-lfm2moe
git commit -m "feat(reap): qwen35/lfm2moe populate per-expert tier tables when a layer is mixed [GPU-deferred]"
```

---

### Task 5: Docs

- [ ] **Step 1:** In `scripts/reap/README.md`, add an "Intra-layer mixed tiers — SP2" note: per-expert tiers within a layer now bucket the top-k and reuse single-tier kernels; buffer arches (qwen35/lfm2moe) supported; blob arches (ds4/minimax) need per-tier sub-blobs (follow-on); the bucketing-equivalence + serve gates are GPU-owed.

- [ ] **Step 2: Commit** `docs(reap): SP2 mixed-tier dispatch notes`.

---

## Self-Review

**Spec coverage (§2):**
- `MoeDtypes.per_expert_*` + bucketed dispatch reusing single-tier kernels → Tasks 2,3. ✓
- Partition by tier, disjoint rank-slot accumulation → Task 1,3. ✓
- `None` = uniform unchanged → Task 2 (every builder sets None; mixed only when tables vary). ✓
- Both dispatch sites → **generic `run_moe_decode` done; ds4 bias-aware/hash DEFERRED** (noted). ✓ (documented gap)
- Blob-arch sub-blobs → **DEFERRED** (noted). ✓

**Placeholder scan:** Task 2/3 use `/* uniform MQ4 base */` for the `MoeDtypes` literal — the implementer fills from the real struct (grep). Task 3's per-tier stride threading is described, not coded verbatim, because it depends on the real `MoeParams` shape — acceptable as a GPU-deferred task with a clear mechanism, but the implementer must read `MoeParams` + the kernel signatures and wire concretely.

**Type consistency:** `bucket_topk_by_tier(&[u32], &[DType]) -> Vec<TierBucket>`, `TierBucket{tier,ranks,experts}`, `MoeDtypes.per_expert_gate_up/down: Option<Vec<DType>>`, `MoeResolution.mixed: bool`. Consistent.

**Known follow-ups (all GPU/large):** ds4+minimax per-tier sub-blob packing; bias-aware + hash dispatch-site bucketing; the bucketing-equivalence + serve numerical gates; per-tier stride source in MoeParams.
