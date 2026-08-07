# 2026-08-07 — Flash-prefill kernel cache keyed on name, not (br, bc)

A pre-existing defect in `attention_q8_0_flash_prefill_slots`'s compile cache,
found while building the Task 7 multi-slot correctness harness. It is **not**
introduced by this branch (`feat/batched-attn-impl`) — the same cache-key
pattern exists on `origin/beta` today, in code this branch did not touch. It
is filed here because Task 8 (multi-slot bench + `TILE_SIZE`/`br`/`bc` sweep)
will hit it directly and needs to know before it starts measuring anything.

## The bug

`Gpu::attention_q8_0_flash_prefill_slots`
(`crates/rdna-compute/src/attention.rs:1953-2130`) compiles a distinct
kernel binary per `(br, bc)` pair — the module name and `#define`s are
correctly parameterized:

```rust
// crates/rdna-compute/src/attention.rs:2032
let module = format!("attention_q8_0_flash_prefill_br{br}_bc{bc}");
...
// crates/rdna-compute/src/attention.rs:2037-2045
if !self.functions.contains_key("attention_q8_0_flash_prefill") {
    let stripped = kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC
        .replace("#include \"kv_slot_desc.h\"", "");
    let src = format!(
        "#define BR {br}\n#define BC {bc}\n#define NTHREADS {NTHREADS}\n{}\n{}",
        kernels::KV_SLOT_DESC_H,
        stripped
    );
    self.ensure_kernel(&module, &src, "attention_q8_0_flash_prefill")?;
}
```

But the **recompile guard** on line 2037 checks
`self.functions.contains_key("attention_q8_0_flash_prefill")` — a constant
string that does not vary with `br`/`bc`. `ensure_kernel`'s `func_name`
argument is this same constant (`"attention_q8_0_flash_prefill"`), and the
underlying cache insert in `compile_and_load_kernel`
(`crates/rdna-compute/src/scratch.rs:77-96`) is also keyed by `func_name`,
not `module_name`:

```rust
// crates/rdna-compute/src/scratch.rs:77-79
pub(crate) fn compile_and_load_kernel(..., func_name: &str) -> HipResult<()> {
    if functions.contains_key(func_name) {
        return Ok(());
    }
    ...
```

Net effect: the **first** `(br, bc)` pair a given `Gpu` instance launches
with "wins." Every subsequent call on that same `Gpu` with a **different**
`(br, bc)` finds `"attention_q8_0_flash_prefill"` already present in
`self.functions`, skips compilation entirely, and dispatches through
`launch_maybe_blob("attention_q8_0_flash_prefill", ...)`
(`attention.rs:2104-2130`) — which resolves that name back to the **stale**
kernel binary compiled for the *first* `(br, bc)`, not the one just
requested.

## How it manifests

**Silently wrong results, not a crash or a compile error.** The host side
(grid dimensions, LDS allocation size, `dpt` register-file sizing — all
computed from the `br`/`bc` arguments passed to this call) is fully correct
for the *new*, intended `(br, bc)`. Only the *running kernel binary* is
stale, still built for the *first* `(br, bc)` the `Gpu` instance ever used.
Concretely, with `BR` stale-small and the caller's real `br` larger, the
kernel's own row partitioning (`row0 + threadIdx-derived offset`, bounded by
the compiled-in `BR`) does not match the grid the host launched: rows beyond
`grid_x_requested * BR_stale` are silently left unwritten (`out` starts as
`gpu.zeros`, so those rows read back as exact zero, not garbage — a
plausible-looking but wrong result, not an obvious crash). Depending on how
close the stale and requested `(br, bc)` are, this can also show up as
partially-correct rows or a wrong `LDS`/`dpt` interaction that a downstream
softmax silently absorbs rather than trapping on.

This was found empirically, not by code inspection first: an early run of
`test_batched_attn_slots.rs`'s `shapes_prefill()` sweep — which deliberately
varies `(br, bc)` per shape (realistic; production chunk-prefill sizing
varies `br`/`bc` too) — reported "golden mismatch" failures starting at the
*second* distinct `(br, bc)` pair used against one shared `Gpu` instance.
The mismatches disappeared entirely once each `(br, bc)` pair got its own
fresh `Gpu`, isolating the cause to the compile cache rather than to the
multi-slot descriptor addressing the harness is actually chartered to test.

## The harness's workaround

`test_batched_attn_slots.rs`'s Q8 flash-prefill sweep
(`crates/rdna-compute/examples/test_batched_attn_slots.rs`, `main()`, around
the `"### Q8 flash-prefill kernel sweep"` section) allocates a **fresh
`Gpu::init()` per `(shape, br, bc)`** rather than reusing the shared `gpu`
used by the general-tile and LDS-decode sweeps earlier in the same run, and
calls `pgpu.drain_pool()` after each one to release real device memory
before the next `Gpu::init()`. This sidesteps the defect entirely (a fresh
`Gpu` has an empty `functions` cache, so its first — and in the harness's
case, only — `(br, bc)` pair always compiles correctly) but does nothing to
fix the underlying cache-key bug for any caller that reuses one `Gpu`
instance across multiple `(br, bc)` pairs, which is the normal, intended
usage pattern for a persistent inference server.

## Warning for Task 8

**Task 8 explicitly sweeps `br`, `bc`, and `TILE_SIZE`.** If it reuses one
`Gpu` instance across that sweep — the natural way to write a benchmark loop,
and the *opposite* of what this harness does for correctness reasons — every
measurement after the first `(br, bc)` pair will silently execute the first
pair's stale kernel binary while reporting timings and (if it checks
correctness at all) shapes computed for the *requested* configuration. The
resulting numbers will not correspond to the configurations they claim to
measure, and depending on how close the stale/requested `(br, bc)` pairs are,
wrongness may not be visually obvious (see "How it manifests" above — this
is a zeros-in-tail-rows bug, not a crash).

Task 8 must do one of:

1. **Allocate a fresh `Gpu` per `(br, bc)` (or per `TILE_SIZE`) configuration**,
   mirroring this harness's workaround, and drain each instance's pool
   before creating the next; or
2. **Fix the cache key** in `attention.rs`/`scratch.rs` so it is derived
   from `(br, bc)` (e.g. reuse the already-correct `module` string, or thread
   a `(br, bc)`-qualified name through as `func_name`) so one `Gpu` instance
   can safely serve multiple `(br, bc)` pairs.

**This report does not attempt option 2.** Fixing the cache key is out of
Task 7's scope — the shared `compile_and_load_kernel`/`ensure_kernel` code
path in `scratch.rs` is used by many other kernels beyond flash-prefill, and
changing its keying discipline needs its own review to confirm it does not
regress caching behavior for callers that legitimately want name-based
reuse across compatible configurations elsewhere in the codebase.
