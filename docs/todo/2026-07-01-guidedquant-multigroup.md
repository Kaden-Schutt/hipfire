# TODO — GuidedQuant multi-group Hessians (per-output-channel-group `H̄_k`)

Status: **not started** (design). Prereq mechanism is DONE + verified correct
(see the `calib_guided` driver, `CalibCollector::capture_weighted`,
`model_calib_down_backward`, and the `hipfire-quantize --ldlq-probe`
crossover/perturbation diagnostic).

## Why

The **g=1** first move (a single per-token Fisher weight `w[n] = mean_c
(∂ℓ/∂Z[n,c])²` averaged over *all* output channels) is implemented correctly and
optimizes its objective **in-sample** (textbook crossover: guided wins the
Fisher-eval, plain wins the plain-eval), but it **loses on held-out** — a pure
generalization gap. The per-token weights from a small calib set are dominated by
a few extreme-gradient tokens (measured: `‖Hg-Hp‖/‖Hp‖ = 0.63`, diag ratio
CV 0.27, range [0.19, 8.4]) and overfit the split.

Two levers close the gap. Lever #1 (more calib tokens) is the cheap test tracked
separately. **This doc is lever #2: the structured, generalizable signal g=1
throws away — per-output-channel-group Hessians.**

## The math

GuidedQuant's scalable objective partitions the `d_out` output channels of a
linear into `g` groups `J_1..J_g` (`g ≪ d_out`) and gives each group its own
Hessian:

```
H̄_k = Xᵀ · Diag( ḡ_k² ) · X          ḡ_k²[n] = (1/|J_k|) Σ_{j∈J_k} (∂ℓ/∂Z[n,j])²
```

then LDLQ-quantizes each output channel `j ∈ J_k` against **its group's** `H̄_k`:

```
min_{ŵ_j} (w_j − ŵ_j)ᵀ H̄_k (w_j − ŵ_j)     for j ∈ J_k
```

g=1 is the special case (one group = all channels). The paper groups
`d_out/g` consecutive channels; simple and works.

## What changes

Four pieces, on top of the g=1 mechanism that already exists:

1. **Grouped gradient reduction (kernel).** Replace the single
   `calib_row_meansq_f32(d[n,d_out]) → w[n]` with a per-group reduction:
   `calib_row_group_meansq_f32(d[n,d_out], w[n,g], d_out, g)` where
   `w[n,k] = (1/|J_k|) Σ_{j∈J_k} d[n,j]²`. (Consecutive-channel grouping ⇒ each
   group is `d_out/g` contiguous columns.)

2. **`g` weighted Hessians per tensor (collector).** Generalize
   `capture_weighted` to accumulate `g` Hessians `H̄_1..H̄_g` for one tensor
   (loop the existing `calib_hessian_outer_weighted_f32` over the `g` weight
   columns). Cost: **g× the Hessian memory + compute** — the main constraint
   (down_proj `[8192,8192]` f32 is 1 GB; `g=4` ⇒ 4 GB/tensor). Keep `g` small
   (4–16) and note the paper uses `g ≪ d_out`.

3. **On-disk `.calib.hfq` + reader.** Emit `<tensor>.hessian.g{k}` entries (or a
   `[g,k,k]` block) + record `g` and the channel→group map in metadata.
   `hessian_io::HessianSidecar` gains a `get_group(name, k) -> HessianRef`.

4. **Group-aware LDLQ (quantizer).** `qtip_ldlq_dequant_bits` currently does all
   `m` output rows with one `H`. Either call it per group (slice the `m` rows of
   `W` belonging to `J_k`, run with `H̄_k`, stitch) or generalize it to take a
   `row→group` map + `[g,k,k]` Hessians. Per-group slicing is the low-risk path.

## Validation

- Extend `--ldlq-probe` to multi-group: per-group LDLQ + the same held-out
  crossover. Expect guided (multi-group) to now **beat** plain on the held-out
  Fisher-eval (the whole point).
- Then the real end-to-end: quantize + KLD once a group-aware format lands
  (still blocked on the qtip3-sim/bf16-load path — see the GuidedQuant memory).

## Constraints / notes

- **LLaMA-dense only** (`hipfire-train`'s model). MoE (minimax/qwen35-moe) needs
  trainable models — a separate lift.
- Memory is the gating cost (`g×` Hessians). For big models, stream/group-tile.
- Do **lever #1 (more calib tokens) first** — if it alone closes the g=1 gap,
  multi-group may be unnecessary for the target bit-rates. This doc is the
  fallback / the "real GuidedQuant" if #1 is insufficient.
