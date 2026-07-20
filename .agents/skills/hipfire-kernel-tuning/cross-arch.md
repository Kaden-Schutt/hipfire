# Cross-arch portability (kernel tuning)

How a tuning change stays correct on arches you are not sitting in front
of. This skill covers **making an already-supported kernel faster**
without breaking the fallback matrix. Brand-new ISA ports belong in
`.agents/skills/hipfire-arch-port/`.

Mutable inventories (exact baseline numbers, env tables, admission rows)
live in their owners — link, do not copy:

| Concern | Owner |
|---|---|
| Validation route per claim class | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |
| Capability vs perf-variant selection | [`docs/methodology/perf-arch-discipline.md`](../../../docs/methodology/perf-arch-discipline.md) |
| Bench protocol / noise / prompt md5 | [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) |
| Arch capability atoms/molecules | `crates/rdna-compute/src/arch_caps.rs` |
| Low-level Gpu / kernel load | `crates/rdna-compute/src/{dispatch,gemm,gemv}.rs` |
| Table-driven KernelKey predicates | `crates/hipfire-dispatch/src/tables/` |
| Speed floors (when a file exists) | `tests/speed-baselines/<arch>.txt` |
| Kernel compile chip→family→base | `scripts/compile-kernels.sh` |

---

## 1. Capability layers (source of truth)

`ArchCaps` is a three-layer descriptor built once at `Gpu::init`:

1. **Atoms** — exact chip codes (`is_gfx1100`, `is_gfx1201`, `is_gfx942`, …).
2. **Molecules** — families (`is_rdna2`, `is_rdna3`, `is_rdna3_dgpu`,
   `is_rdna3p5`, `is_rdna4`, `is_cdna3`, …).
3. **Capabilities** — ISA features (`has_wmma`, `has_wmma_w32`,
   `has_wmma_w32_gfx12`, `has_dot2_f32_f16`, `is_wave64_native`, …).

**Use capabilities for correctness** (“does this chip have this
intrinsic family?”). **Do not** use a broad capability molecule to pick
a *perf sub-variant* of the same kernel (e.g. `plain` vs `ldscoop`).
That failure class is documented in
`docs/methodology/perf-arch-discipline.md` and in
`case-studies.md` CS-9.

Illustrative capability *names* (not a live membership table — re-read
`crates/rdna-compute/src/arch_caps.rs` for current atoms/molecules):

- `has_wmma` / `has_wmma_w32` / `has_wmma_w32_gfx12` — WMMA family split
  (gfx11 vs gfx12 builtins are distinct; do not lower gfx11 on gfx12)
- `has_dot2_f32_f16` — chip allowlist is **not** “all RDNA”; some gfx10
  atoms are excluded. Read source before claiming membership.
- `is_wave64_native` / `is_wave32` — wave-size molecules

RDNA3 is intentionally broad as a molecule (dGPU + Strix-class APUs).
Memory hierarchy still differs (dGPU GDDR6+IC vs Strix LPDDR5X vs small
APU caches) — hence perf allowlists must name atoms or narrow molecules
(`is_rdna3_dgpu`, `is_rdna3p5`), not bare `is_rdna3()`, when selecting
sub-variants. Confirm which atoms each molecule covers in `arch_caps.rs`.

---

## 2. Where selection actually lives

There is no single `dispatch.rs` mega-tree that owns every GEMM anymore.
For tuning work, expect to touch one or more of:

| Layer | Role |
|---|---|
| `rdna-compute` `Gpu` methods (`gemm.rs`, `gemv.rs`, …) | Concrete kernel name, grid, arch-specific sibling call |
| `rdna-compute` `ArchCaps` / feature flags | ISA gates and env overrides |
| `hipfire-dispatch` tables + family arms | `KernelKey` → `ArchPredicate` (correctness availability) |
| Arch crate forward (e.g. qwen35, lfm2moe) | Model-shaped batching; may further gate by atom/env/fixture — read that crate’s forward/daemon gate; truth state via `docs/INDEX.md` / empty `docs/admissions.yml` |

When you add a fast path:

1. Prefer an existing capability helper over stringly `starts_with`.
2. Put the new branch **above** slower fallbacks; baseline/scalar last.
3. **No unreachable branches** — if a more-specific check absorbs an
   arch a broader check used to handle, narrow the broader check in the
   **same** diff (arch-port skill enforces this for new chips; tuning
   inherits the rule).
4. Perf sub-variant defaults stay **conservative portable**, with
   measured atom/class allowlists for wins — never “inherit best tuned
   variant by capability.”

---

## 3. File-level arch tags

Per-arch HIP sources use the compile script’s variant tags:

```
kernels/src/<base>.hip                 # default all archs
kernels/src/<base>.gfx1100.hip         # chip override
kernels/src/<base>.gfx12.hip           # family (gfx1200 + gfx1201)
kernels/src/<base>.gfx1201.hip         # chip beats family
kernels/src/<base>.wave64.hip          # wave-size variant (where used)
```

Resolution in `scripts/compile-kernels.sh`:

1. `${name}.${arch}.hip` — chip
2. `${name}.${arch_family}.hip` — family (`${arch:0:5}`, e.g. `gfx12`, `gfx94`)
3. `${name}.hip` — default

Default compile arch list: read the header of `scripts/compile-kernels.sh`
(mutable; do not freeze a second copy here).

Use **family** tags when one source is correct for every chip in the
family; use **chip** tags for occupancy/prefetch/tuning that is not
portable even inside the family.

---

## 4. Adding a tuned fast path (minimum workflow)

1. **Author** the kernel with the right tag (chip vs family vs wave64).
2. **Register** source (`include_str!` / ensure-kernel path used by that
   crate — follow neighbors, not a single hard-coded `kernels.rs` myth).
3. **Wire** a `Gpu` method or extend an existing one (grid, kernarg, name).
4. **Gate** with the narrowest correct capability or atom allowlist.
5. **Validate** with the **narrowest** route in `docs/VALIDATION.md`
   for your claim class — typically:
   - new/changed `.hip` numeric → `test_kernels` on the target arch, **then**
     the applicable model/path-level manual route from
     [`docs/VALIDATION.md`](../../../docs/VALIDATION.md). If that route or a
     required oracle does not exist, mark the claim **BLOCKED** — do not
     proceed to perf or public dispatch on channel alone
   - dispatch bind surface → `scripts/verify-bind-thread.sh`
   - perf claim → `docs/methodology/perf-benchmarking.md` +
     `scripts/speed-gate.sh` / fresh-process probe when applicable
6. **Retired batteries** are not current evidence — use [`docs/VALIDATION.md`](../../../docs/VALIDATION.md).

If you cannot measure the target arch: land the kernel + channel-test
**without** flipping the public default dispatch (methods available,
path opt-in or unrouted). Flip only after a dated measurement on real
hardware. That is how gfx12 landed initially and how unbenched perf
variants stay fail-closed.

---

## 5. Speed baselines vs support

`tests/speed-baselines/<arch>.txt` is the speed-gate floor **when the
file exists**. Presence of a baseline is not the same as “supported,”
and absence is not “unsupported” — it means the gate has nothing to
compare unless `HIPFIRE_BASELINE_ARCH` / an authored file says otherwise.

**Lookup, do not copy.** List baseline files and their header hardware notes
from disk before any completeness claim:

```bash
ls tests/speed-baselines/
# read each file header for capture notes (chip / SKU / date / commit)
```

Absence of a `<arch>.txt` means the speed-gate has nothing to compare for
that atom unless `HIPFIRE_BASELINE_ARCH` / an authored file says otherwise —
**not** permission to inherit a parent arch’s floors or expected shape.
Contributing a new baseline is welcome via the tester skill / bench flow;
it is not required to land a gated fast path you measured and left
unrouted on other chips.

**KV defaults:** the old CLI `archDefaults` table is **removed**. KV mode
defaults are product/registry concerns (`default_kv_mode` / runtime), not
a per-arch table inside this skill. Do not resurrect arch→KV maps here.

---

## 6. What “won’t break other arches” means

For each arch that can execute the touched path:

- If your fast path **matches**, speed-gate (or an equivalent fresh
  protocol measurement) must stay within the baseline tolerance policy
  for that arch’s baseline file when one applies.
- If your fast path **misses**, the code visible to that arch should be
  a no-op delta: same kernel name, same fallback, no accidental predicate
  widen that steals the path onto unmeasured chips.

Local pre-commit speed-gate only sees **this machine’s** arch. Multi-branch
dispatch edits need either contributor hardware, remote rental, or an
explicit “unrouted until measured” landing.

Compile-check the matrix you claim:

```bash
./scripts/compile-kernels.sh gfx1010 gfx1030 gfx1100 gfx1200 gfx1201
# add gfx906 / gfx942 / gfx1151 when those paths changed
```

---

## 7. Model-arch gates are not GPU-arch gates

GPU `ArchCaps` ≠ model `arch_id`. Model-local gates (batch path, fixture
shape, env flag, product state) are **crate source + admissions**, not this
skill. Example discipline for LFM batched prefill — **re-read source**, do
not freeze predicates here:

1. Read the gate in `crates/hipfire-arch-lfm2moe/src/forward.rs` (and any
   daemon/bench call site) for the current GPU atom, env flag, and fixture
   shape checks.
2. Label truth via [`docs/INDEX.md`](../../../docs/INDEX.md): branch work is
   **branch-implemented** until admitted; [`docs/admissions.yml`](../../../docs/admissions.yml)
   is empty / fail-closed — runtime presence ≠ product admission.
3. Never promote on capability molecules (`is_rdna4`, `has_wmma`) when the
   crate names a single chip atom.

When tuning LFM or any crate-local path, read that crate’s gate before
assuming family-wide or capability-wide promotion.

Model id table owner: [`docs/architecture-ids.md`](../../../docs/architecture-ids.md).

---

## 8. Anti-patterns

| Anti-pattern | Instead |
|---|---|
| `if is_rdna3() { best_strix_variant }` | Atom/class allowlist + portable default |
| Copying `tests/speed-baselines` numbers into PRs as “current SOTA” | Cite baseline file + commit; re-measure for claims |
| Enabling public dispatch for an arch you never ran | Opt-in / unrouted until dated evidence |
| Retired batteries as current evidence | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) claim → route map |
| Assuming gfx1200 ≡ gfx1201 for chip-tagged kernels | Family tag only when source is truly shared |
| Treating multi-GPU PP baseline as single-GPU floor | Separate files / separate claims |

---

## 9. Related skills

- **hipfire-arch-port** — new chip ISA, WMMA builtin matrix, first bring-up
- **hipfire-tester** — bench submission / hardware matrix runs
- **hipfire-kernel-atlas** — phase-aware measurement corpus (not policy)
- **hipfire-autoheal** — runtime/JIT bring-up failures, not lever selection
