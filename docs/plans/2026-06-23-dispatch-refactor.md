# Refactor `rdna-compute::dispatch` — split the 52k-line monolith

Status: **planned** — 2026-06-23. Owner: chaingun.
Motivation thread: the kernel test matrix (reference floor vs arch overlays) can only
be file-scoped if the *selection* is file-separated. The kernel **sources** already
are (`kernels/src/<arch>/*.<arch>.hip` overlays vs `kernels/src/*.hip` reference
floor), but the **dispatch** that selects between them is one file, so any change to
it conservatively forces the whole gate. See the
[reference-kernel-layer plan](2026-06-22-reference-kernel-layer.md) and
`generic_warn.rs` (the `warn_generic_once` / `generic_fallback_count` coverage map).

## The target

`crates/rdna-compute/src/dispatch.rs` is **52,251 lines**: one `pub struct Gpu`
(`dispatch.rs:429`) plus essentially one `impl Gpu` block with **848 methods (795
`pub`)** — the entire GPU dispatch surface (kernel JIT, launch, graph capture, memcpy,
and every op's kernel selection) in a single file. The 79 `is_gfx*/is_rdna*/is_cdna*`
arch branches that pick overlay-vs-reference are scattered through the op methods.

Problems this causes:
- **Test scoping is impossible at the dispatch layer.** A one-line change anywhere in
  `dispatch.rs` is indistinguishable from a core-floor change, so it must run the full
  gate (and, once the reference cell lands, *both* cells). The kernel sources are
  already path-separated; the dispatch isn't, so it's the bottleneck.
- **Navigability / review.** 52k lines, 848 methods; finding the gemm path or the
  gfx1151 overlay for an op means scrolling a monolith.
- **Incremental compile.** Any edit recompiles the whole 52k-line module.

## The enabling fact (why this is mechanical, not a rewrite)

`Gpu`'s fields are **module-private** to `dispatch` (`compiler`, `modules`,
`functions`, … are private; only `hip`/`arch`/`flags`/`arch_caps`/`device_id`/
`integrated` are `pub`). Rust lets a **descendant** module access its ancestors'
private items, so methods moved into **child** modules of `dispatch`
(`dispatch::gemm`, `dispatch::attention`, …) can still touch `self.compiler` etc.
**with no visibility changes.** Combined with Rust allowing multiple `impl Gpu`
blocks across files in one crate, the split is a **pure, behavior-preserving move** of
method bodies — no logic edits, no API changes, no `pub(crate)` churn.

## Op-family clustering (method counts, the split boundaries)

From `grep ^pub fn` prefixes in `dispatch.rs`:

| Family | ~methods | Notes |
|---|---|---|
| gemm | 248 | by far the largest; sub-split likely (by dtype / reference-vs-overlay) |
| gemv | 113 | decode-shape GEMMs |
| attention | 57 | + flash decode / batched prefill / swa(3) / softmax(4) |
| fused | 44 | fused rmsnorm+rope+rotate etc. |
| kv | 30 | + kvarn(5) |
| deepseek4 | 25 | + hc(14) + indexer(8) + compressor(6) — arch-9 cluster |
| rope / rotate | 13 / 6 | + sigmoid(4) |
| moe | 12 | |
| gated / conv1d | 12 / 7 | the SSM/DeltaNet path |
| embedding | 8 | |
| rmsnorm | 5 | |
| triattn / pflash | 5 / 6 | |
| sample / quantize / dequantize | 3 / 3 | |
| **infra** | rest | init, ensure_kernel(8), launch_blob, graph capture/replay(6+), memcpy_*, kernel cache — arch-agnostic plumbing |

## Target module layout

```
crates/rdna-compute/src/dispatch/
├── mod.rs            # `pub struct Gpu` + private fields; init; the kernel-cache /
│                     #   ensure_kernel JIT; launch_kernel_blob / launch_maybe_blob;
│                     #   graph capture+replay; memcpy_* ; Drop. The arch-agnostic core.
├── gemm.rs           # impl Gpu — dense GEMM dispatch (may sub-split: gemm_mq.rs, gemm_f16.rs)
├── gemv.rs           # impl Gpu — decode GEMV dispatch
├── attention.rs      # flash decode / batched prefill / swa / softmax
├── fused.rs          # fused rmsnorm/rope/rotate/gate kernels
├── norm_rope.rs      # rmsnorm, rope, rotate, sigmoid (small, can merge)
├── moe.rs            # MoE grouped GEMM + combine
├── ssm.rs            # gated_delta_net + conv1d (the DeltaNet/Mamba path)
├── kv.rs             # KV cache ops + kvarn
├── deepseek4.rs      # deepseek4 + hyper-connections + indexer + compressor
├── embedding.rs      # embedding lookup + quantize/dequantize + sample
└── pflash_triattn.rs # pflash + triattn eviction kernels
```

Each op file is an `impl Gpu { ... }` block. `mod.rs` declares the children
(`mod gemm; mod attention; …`) so they're descendants of `dispatch` and inherit field
access.

## Two phases

### Phase 1 — split by op family (mechanical, behavior-preserving)

Pure relocation of method bodies into the child modules above. No logic change. This
**alone** delivers: a navigable tree, faster incremental compile, and op-family
test-scoping granularity (a change in `dispatch/gemm.rs` ⇒ gemm tests, not the whole
gate). The arch branches still live inside each op method — reference-vs-overlay is not
yet file-separated, but the *family* is.

### Phase 2 — separate reference selection from arch overlays (the actual goal)

Within each family, extract the overlay selection out of the reference path. Today:

```rust
fn gemm_X(&self, …) -> Kernel {
    if self.arch_caps.is_gfx1151() { /* gfx1151 overlay */ }
    else if self.arch_caps.is_gfx1100() { /* gfx1100 overlay */ }
    else { /* reference floor */ }
}
```

→ family file keeps the reference floor + the entry; overlays move to per-arch files:

```
dispatch/overlays/gfx1151/gemm.rs   // impl Gpu { fn gemm_X_overlay_gfx1151(…) -> Option<Kernel> }
dispatch/overlays/gfx1100/gemm.rs
…
// entry, in dispatch/gemm.rs:
fn gemm_X(&self, …) -> Kernel {
    self.gemm_X_overlay(…).unwrap_or_else(|| self.gemm_X_reference(…))
}
```

Now path↔file is exact, enabling the scoped gate matrix:

| Changed files | Cells |
|---|---|
| `dispatch/overlays/<arch>/**` | that arch's overlay cell (+ its parity diff vs reference) |
| `dispatch/<family>.rs` (reference floor) | **both** — reference + dependent overlays |
| `kernels/src/<arch>/**` | overlay cell for that arch |
| `kernels/src/*.hip` (floor) | **both** |

(Phase 2 pairs with the `HIPFIRE_FORCE_GENERIC` reference-cell + differential-parity
gate proposed in the reference-kernel-layer follow-up; that gate is what the
`overlays/<arch>` vs `<family>.rs` scoping keys.)

## Sequencing (incremental; compile each step, gate at milestones)

0. **Prep + de-risk.** `dispatch.rs` → `dispatch/mod.rs` verbatim (no content change);
   confirm it builds. Then move ONE tiny family (e.g. the 3 `quantize`/`dequantize`
   methods) into `dispatch/embedding.rs` to *prove* the descendant-field-access
   invariant on a real compile before committing to the big moves.
1. **Leaf/low-coupling families first:** embedding, sample, quantize, norm_rope
   (rmsnorm/rope/rotate/sigmoid), moe, ssm (gated/conv1d). Compile + commit each.
2. **Mid families:** attention/flash/swa/softmax, kv/kvarn, fused.
3. **Big/coupled:** gemm (248 — likely its own sub-split commit), gemv (113),
   deepseek4 cluster (deepseek4+hc+indexer+compressor).
4. Infra (init, ensure_kernel, graph, memcpy, launch) stays in `mod.rs`.
5. **Phase 2** per family: extract `dispatch/overlays/<arch>/<family>.rs`, starting
   with the families that actually have gfx1151 overlays (the ~10 `is_gfx1151()`
   selection sites + the 18 `kernels/src/gfx1151/*` files name them).

Gate (gfx1151 coherence + MQ4 speed) after each milestone group. Because every move is
behavior-preserving, the gates should pass unchanged — a *failing* gate after a pure
move is a real bug (e.g. a missed method, a macro-expansion site) and stops the train.

## Risks / constraints

- **Hot path.** Every kernel launch goes through these methods. Mitigate: move-only (no
  logic edits); gate per milestone; the `cargo check` + gfx1151 gate are the guards.
- **The `moe_scalar_indexed_wrappers!` macro** (`dispatch.rs:102`) generates methods —
  move the macro + its invocation together into `dispatch/moe.rs` (or keep macro in
  `mod.rs`, invoke in the child).
- **`mod tests`** (`dispatch.rs:52226`) — relocate per-family unit tests next to their
  family, or keep a `dispatch/tests.rs`.
- **Privacy edge cases:** any field/helper that turns out *not* to be reachable from a
  child becomes a compile error at that step (caught immediately) — fix by moving the
  helper to `mod.rs` or marking `pub(crate)`. Expect a handful.
- **Not in scope here (optional follow-ons):** splitting `kernels.rs` (4,600 lines of
  `*_SRC` consts) per family, and `arch_caps.rs` (644). Lower priority; can mirror the
  same family boundaries once dispatch is split.

## Success criteria

- No single dispatch file > ~4k lines; `mod.rs` is just struct + infra.
- Zero behavior change: gfx1151 coherence + MQ4 speed gates pass at every milestone.
- Phase 2: an overlay-only change (`dispatch/overlays/gfx1151/*` or
  `kernels/src/gfx1151/*`) is provably isolatable from the reference floor, so the
  scoped gate matrix runs only the affected cell(s).
- The pre-commit hook's forward/kernel globs (TODO: "tighten relevance globs") key the
  reference/overlay cells off these now-separated paths.
