# Multi-GPU bring-up lessons (dev log)

Captures non-obvious gotchas and root-causes from bringing up hetero
PP=2 across a cross-arch AMD pair (gfx906 MI50 CDNA1 + gfx1031 RX 6700 XT
RDNA2). Companion to the operational `docs/multi-gpu.md` — that one
covers what PP gives you and how to deploy it; this one covers what
broke and why during bring-up.

Audience: anyone wiring up a new multi-GPU configuration in hipfire,
especially one that mixes architectures or includes a previously
untested device.

## 1. Per-arch JIT kernel cache

**Lesson:** the runtime JIT cache MUST be per-arch keyed at the cache
directory level, not just per kernel-source-hash.

**What we found:** before this branch, `compiler.rs` keyed
pre-compiled blobs under `.hipfire_kernels/{arch}/` (correct) but
wrote runtime-JIT'd blobs to the **flat** `.hipfire_kernels/*.hsaco`
root. Single-arch users were unaffected — the source+arch hash check
on read would force a recompile when the blob was from a different
arch. But:

- **Cross-arch in one process** (hetero spec-decode prototyping):
  drafter on arch_A and target on arch_B race for the same flat path.
  Each `ensure_kernel` for shared kernel names overwrites the other's
  binary. The hash invalidates the freshly-compiled one on the next
  read, triggering ping-pong recompiles.
- **Parallel cross-arch workflows on the same machine** (CI matrix
  with multiple `--offload-arch` targets, or developer running tests
  on dev box with two cards): same pathological collisions in the
  flat layout.

**Fix shape:** `cache_dir = base.join(arch)` at the `KernelCompiler`
constructor. Reads + writes both land under `.hipfire_kernels/{arch}/`,
matching the pre-compiled install-blob layout. The hot-path seeding
helper already knew about per-arch dirs; only the runtime-compile
writeback was wrong.

**Side effect for existing single-arch users:** stale blobs at the
flat root get ignored. On first run after the fix they're re-JIT'd
under `{arch}/` (~10s of warmup). Can be cleaned up with
`find .hipfire_kernels -maxdepth 1 -type f -delete` if desired.

## 2. `Gpus::init_layers` is the asymmetric-VRAM escape hatch — let it act like one

**Lesson:** when a function's docstring says "explicit escape hatch
for asymmetric VRAM / hand-tuned splits," the function should not
re-apply a uniform-VRAM tolerance check.

**What we found:** `init_layers` shares its pre-flight with
`init_uniform`. That pre-flight enforces a 2 GiB free-VRAM delta
across devices (`HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB`). Reasonable for
`init_uniform` (where the caller is implicitly saying "treat both
cards equally"). Wrong for `init_layers` where the caller has just
hand-typed `HIPFIRE_PP_LAYERS=24,8` to declare asymmetric intent.

A 32 GB MI50 + 12 GB 6700 XT pair has a 20 GB free-VRAM delta after
target load — far over the 2 GiB tolerance. The caller already split
the layers in a way that respects the asymmetry; the gate fires
anyway and refuses the load.

**Fix shape:** split `preflight_vram` into a `preflight_vram_with_opts(check_vram_delta: bool)`
so `init_uniform` can request the delta check and `init_layers` can
skip it. The arch-mismatch and per-device bind/free probe still run
on both paths.

## 3. `HIPFIRE_ALLOW_MIXED_ARCH=1` was documented but never wired

**Lesson:** when a PRD claims an env var is shipped, audit the actual
code path before relying on it.

**What we found:** `docs/env-vars.md:230` references `HIPFIRE_ALLOW_MIXED_ARCH`
as a tunable. `docs/plans/hetero-pflash-dflash.prd:138-148` claims
the env was wired into the arch-mismatch path. But `preflight_vram`
hard-errored unconditionally on any arch mismatch — the env-var
parsing didn't exist.

**Fix shape:** read the env in `preflight_vram_with_opts`, downgrade
from hard-error to `eprintln!` warning when set. The default
behavior is preserved (still rejects mismatched arches), but the
opt-in path actually works now.

## 4. AMD-AMD peer access works across mixed arches (don't assume otherwise)

**Lesson:** test peer access empirically on your specific pair; folklore
"peer access only works between same-arch cards" is wrong on AMD ROCm
6.4+ for pairs that share a PCIe root.

**What we found:** gfx906 (CDNA1) + gfx1031 (RDNA2), both on the same
AMD Starship root complex, support full bidirectional peer access:

- `hipDeviceCanAccessPeer(0, 1) = 1` and reverse.
- `hipDeviceEnablePeerAccess` succeeds in both directions (returns
  `hipSuccess`).
- `hipMemcpyPeer` between them correctness-passes on a 4 MiB
  round-trip; cross-card peer reads inside a kernel also work
  (PR #204 leveraged this to have drafter-bound kernels read target's
  `token_embd` directly).

The constraint is **same PCIe root + AMD-AMD**, not same-arch. eGPU
enclosures (Thunderbolt / OCuLink) are the configuration where peer
access typically fails; direct slot-mounted AMD pairs typically don't.

**Practical impact:** the hetero spec-decode work in PR #204 used
peer-access cross-card memory reads inside drafter-bound kernels
(target's `token_embd` is target-resident, but the drafter-side
embedding lookup kernel reads it directly via peer mapping). This
avoids host-bounce D2H/H2D entirely. Plan for the optimistic path.

## 5. `Gpu::bind_thread()` keeps a thread-local cache; `hip.set_device()` bypasses it

**Lesson:** never call `gpu.hip.set_device(id)` directly when working
with multiple `Gpu` instances. Always go through `Gpu::bind_thread()`.

**What we found:** `bind_thread()` (dispatch.rs:846) keeps a
`LAST_BOUND_DEVICE` thread-local and skips a redundant HIP `set_device`
call when the device is already current. `hip.set_device(id)` calls
the FFI directly **without updating** that thread-local.

If a code path does `gpu_a.hip.set_device(a.device_id)` followed
later by `gpu_b.bind_thread()`, the thread-local says "current is a",
so the bind_thread skips its set_device call — but the HIP runtime
state is whatever the last raw `set_device` left it as. Kernel
launches on whichever Gpu thinks it's bound silently target the
wrong device.

Symptoms we saw: `hipModuleLaunchKernel: invalid device ordinal`
(error 101) from the first kernel after a multi-Gpu init sequence.
Investigation initially blamed stream-device binding (we worked
around it with stream destroy+recreate), but the actual root cause
was the bind-cache desync. Replacing raw `set_device` calls with
`bind_thread` removed the workaround entirely.

**Practical guidance:**
- In application code: always use `gpu.bind_thread()?` to switch
  device. Treat `hip.set_device` as a low-level primitive that
  should only be called from inside the `bind_thread` implementation.
- For multi-Gpu setup sequences (peer-access enable, model load,
  etc.) explicitly `bind_thread` each Gpu before calling functions
  that operate on it.

## 6. HIP streams are bound to the device that was current when they were created

**Lesson:** `hipStreamCreate` reads the currently-bound device and
attaches the new stream to it. Creating a stream while bound to the
wrong device produces a "stream" that launches kernels onto a
different device than your application expects.

**What we saw:** during a multi-Gpu spec-decode bring-up, both Gpus
needed their own `active_stream`. The naive sequence
`gpu_t.active_stream = Some(gpu_t.hip.stream_create()?); gpu_d.active_stream = Some(gpu_d.hip.stream_create()?);`
created one stream bound to whatever device happened to be current
at the moment. Later kernel launches on the "wrong" stream failed
with the same "invalid device ordinal" error.

**Fix shape:** `bind_thread()` immediately before every
`stream_create()` so the new stream is attached to its owning Gpu's
device. The `Gpu::ensure_draft_stream()` helper on this branch does
this explicitly; copy the pattern for any future secondary streams.

## 7. `hipDeviceEnablePeerAccess` direction matters — bind the source device first

**Lesson:** the peer-access API takes a peer device id but operates
on the **currently-bound** device. Forgetting to bind first registers
self→peer instead of source→peer (or vice versa).

**What we saw:** PR #204's PR3 step 2 commit message documents this:
the prior code called `gpu_target.hip.enable_peer_access(drafter_dev_id)`
from a thread that was actually bound to the drafter (post-`init_with_device`
left the drafter bound). This silently registered drafter→drafter peer
access (a no-op) instead of target→drafter. Errors weren't surfaced —
just nothing happened, and downstream peer-copy attempts went through
host-staging fallback.

**Fix shape:** explicit `bind_thread` of the SOURCE device before each
`enable_peer_access(peer_device_id)` call. Validate the return
code (don't `let _ = ...` errors). Recipe for enabling bidirectional
peer access between gpu_a and gpu_b:
```rust
gpu_a.bind_thread()?;
gpu_a.hip.enable_peer_access(gpu_b.device_id)?;
gpu_b.bind_thread()?;
gpu_b.hip.enable_peer_access(gpu_a.device_id)?;
gpu_a.bind_thread()?;  // restore application's "current" device convention
```

## 8. Long-context spec-decode is verify-bound; AR wins past a cliff

**Lesson:** spec-decode's per-cycle batched verify cost grows with KV
size; τ (mean acceptance length) tends to collapse at long context;
combined effect is that pure autoregressive decode beats spec-decode
past some context-length threshold. The crossover sits at very
different context lengths for different architectures.

**What we observed:** on 27B-3.6 dflash drafter + 27B-3.6 target at
short context (~240 tokens), spec-decode delivers τ=8.14 and a
~10× tok/s gain over pure AR. At 13K context: τ collapses to 1.6,
verify cycle wall balloons, and AR (3.76 tok/s) beats spec-decode
(1.85 tok/s solo or 1.65 hetero). PR #204 saw the same proportional
pattern on faster gfx1100 hardware (45.4 tok/s AR vs 33 tok/s
hetero dflash at 16K).

**Practical impact:** don't assume spec-decode is strictly faster than
AR. Bench both at the context lengths your workload actually uses.
Multi-GPU spec-decode pipelining cannot fix the long-context cliff —
the overlap saves `min(draft, verify)` which is dominated by the
small drafter, not the huge verify wall.

## 9. The `Command::terminate()` stack overflow at long contexts

**Lesson:** AMD HIP 6.4 accumulates an internal command-chain when
many small async memcpys queue without intervening syncs. The chain
releases recursively, which blows the host thread stack at >~50K
queued commands.

**Source:** PR #204 commit message for `c0983236` (May 9 2026).
50K commands ≈ 10K rows × 5 extract layers worth of
`scatter_hidden_block_to_interleaved` memcpys. The recursive release
chain (`Command::terminate → ReferenceCountedObject::release →
Command::releaseResources → terminate`) blows the 8 MB main-thread
stack on the FIRST kernel launch following the unbounded queueing.

**Fix shape (per PR #204):** periodic `device_synchronize()` inside
the scatter loop (every 1024 rows) to drain the command chain.
Trailing drain conditional on `n_rows % CHAIN_DRAIN_INTERVAL != 0`
so short-prompt cycles don't pay an extra sync.

Not currently a problem on master because `scatter_hidden_block_to_interleaved`
is only invoked in the spec-decode path where B ≤ 16; long-prompt
contexts that hit this issue specifically require the cross-card
hetero scatter pattern PR #204 was prototyping. **If you build
out any new code path that does N×M cross-card or same-card memcpys
in a loop without a sync, budget a periodic device_synchronize from
day one.**

## See also

- `docs/multi-gpu.md` — operational guide (memory budgets, deployment recipes, throughput).
- `docs/plans/hetero-pflash-dflash.prd` — the hetero PFlash+DFlash architectural plan.
- `docs/plans/path_d.md` — same-card spec-decode pipelining design (closed without merge after empirical dominated by chain-mode; preserved for future revival).
- PR #204 (closed) — full hetero spec-decode prototype with the bind_thread pattern, long-context findings, and the pivot to native MTP as the actual perf lever.
