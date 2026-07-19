# Hipfire graft provenance and enablement boundary

This crate is copied from the standalone Redline repository at commit
`50f59ca` after its historical local gfx1201 and remote R9700 acceptance
campaign. The low-level public ROCr ABI provenance is retained in
`../redline-rocr/PROVENANCE.md`.

This file preserves graft provenance and dated evidence. The current normative
workflow is the canonical [Redline contributor guide](../../docs/REDLINE.md).

The graft is default-off except for the productized automatic admission of the
single-GPU Qwen A3B `.mq4r` route on gfx12, where the product default requests
`auto` with the retained PM4 transport. An explicit `HIPFIRE_REPLAY_BACKEND`
bypasses that automatic model
default; `HIPFIRE_REPLAY_TRANSPORT` changes only the transport and does not
enable replay by itself.

On the automatic product path, the first eligible ordinary-HIP forward records
the tape and successful preparation moves the controller from `Captured` to
`Ready`; it does not traverse `ShadowValidated`. Manual shadow observations and
their observation threshold are pre-promotion evidence, not a durable automatic
admission threshold. Preparation failure occurs after the warmup HIP forward
has completed, poisons retained routing, and allows later eligible calls to use
the HIP-side policy. A retained replay execution failure instead errors the
current call, poisons the controller, and permits fallback only on later calls;
there is no same-call HIP retry. A successful model swap resets the
process-local controller so a prior tape cannot bleed into a new model.

The source repository's results accepted under that historical campaign were:

- real token DAG: 1.076x local and 1.059-1.060x R9700 on GPU timestamps;
- expanded independent set: 1.378-1.381x local and 1.265-1.292x R9700 by
  uninstrumented host-total timing, with separately profiled GPU lower bounds
  still above the historical RADV 1.14x result.

No raw KMD submission path from Hipfire's older `redline` crate is used by the
Qwen replay route.

The Qwen3.5 adapter now records the exact padded HIP kernarg bytes and owning
code object for one ordinary single-token AR forward. `auto` lowers that fixed
sequence to one public-HSA queue after the first eligible forward. Speculative,
MTP re-seed, and verify forwards share HipGraph's one-shot
`ar_graph_eligible` contract and can neither populate nor consume the plain-AR
replay. Dynamic position stays in `pos_buf`; the stochastic GDN frame scalar is
patched between replays.

For adapter development, pair `HIPFIRE_REPLAY_MANUAL_CAPTURE=1` with an
explicit non-HIP backend such as `shadow`. This bypasses the automatic model
default and lifecycle; the daemon's `bench_prefill` and `bench_decode` probes
may then delimit one phase and return
its launch-sequence fingerprint. That fingerprint is discovery evidence only:
these diagnostics neither install nor route a plan, and product route proof is
separately required by the [canonical guide](../../docs/REDLINE.md).

## Local gfx1201 dated evidence (2026-07-11, automatic clocks)

- Qwen3.5 0.8B: 356 dispatches / 21 kernels, sequence hash
  `55f99a58cb4b9363`.
- Qwen3.5 9B: 475 dispatches / 21 kernels, sequence hash
  `ac6495c537cd3e2a`.
- Fifteen consecutive positions are bit-exact for logits, KV, and recurrent
  state against both ordinary HIP execution and an exact HIP-kernarg-blob
  oracle on both models.
- The terminal packet uses agent release: the host consumes only the completion
  signal, while payload buffers are consumed by the next queue on the same GPU.
  This remained exact across the 15-position gate and reduced direct shadow
  time to 42.999 ms vs 45.886 ms on 0.8B and 156.569 ms vs 158.078 ms on 9B.
- Against the already-tuned HipGraph product path, per-dispatch AQL publication
  is not yet a product win: resident medians were 363.762 vs 364.129 tok/s on
  0.8B and 97.820 vs 97.806 tok/s on 9B. Treat both as neutral.

The retained gfx12 PM4 indirect-buffer transport is now implemented behind
`HIPFIRE_REPLAY_TRANSPORT=pm4`. It keeps public ROCr queue/resource ownership,
but replaces the 356/475 per-dispatch AQL packet publications with one AMD
vendor packet pointing at executable HSA command memory. Kernel descriptor
resources and the relocated code entry are parsed from the exact HSA-loaded
code object; unsupported scratch or implicit-SGPR contracts fail closed.

The daemon coherence seam established the cache policy instruction by
instruction. `CS_PARTIAL_FLUSH` orders dependent dispatches, while gfx12
`ACQUIRE_MEM` is retained around the full-scope repeat-interleave, RoPE,
MQ-rotation, and fused-SiLU boundaries. Four proven-independent sibling pairs
omit their intermediate compute-idle wait and fan in at the next dependent
boundary. Fifteen consecutive positions remain bit-exact for logits, KV, and
recurrent state on both models.

Matched resident product reports at automatic clocks (10 runs, 100 decode
positions per run, median) compared the requested HipGraph arm with the
requested `auto`/PM4 arm. Those reports lack the controller `Ready`, fallback,
observed-replay, packet/queue/dword, and anti-HIP/HipGraph fields required by
the current guide. These rows are therefore nominal/requested-PM4 dated
evidence, not proof that the timed arm executed retained replay:

- Qwen3.5 0.8B: 363.682 -> 392.248 tok/s, **1.07855x**.
- Qwen3.5 9B: 97.727 -> 98.775 tok/s, **1.01073x**.

## Qwen3.6 A3B kernel-oracle integration (2026-07-11, automatic clocks)

The same transport was grafted onto `origin/feat/rdna-kernel-oracle` at
`35502d550`; that branch contains the `loop/gfx1201` winning kernels through
`53aab4775`. Redline does not replace those kernels. It retains and replays the
833-launch ordinary-AR tape emitted by that already-tuned branch. Two replay
artifact aliases bind runtime-specialized launch names to their actual loaded
code objects: the shared residual-scale GEMV and the indexed K=8 MoE gate/up
GEMV.

The A3B tape contains 26 kernels and has sequence hash `8d5620ca2ca8a536`.
The initial conservative requested-PM4 arm measured 174.087 tok/s against
164.220 tok/s for the requested HipGraph arm (**1.06009x** nominally). The
shared-expert down and routed-expert gate/up launches are independent: they
read distinct activation buffers, write distinct result buffers, and join only
at the later MoE combine. Removing their intermediate compute-idle wait at all
40 layer boundaries raised the matched requested-route 10-by-100 median to
165.839 -> 178.320 tok/s (**1.07526x** nominally). These timed rows have the
same missing controller and anti-fallback ledger described above.

The expanded policy remained bit-exact for logits, KV, recurrent state, and the
captured HIP kernarg blobs across 15 consecutive positions. It uses automatic
clocks throughout. The raw reports are
`.redline-work/a3b-r3/product-pm4-mq4r-overlap.json` and
`.redline-work/a3b-r3/shadow15-overlap.json` in the isolated hiptrx checkout;
they are measurement artifacts rather than source inputs.

The product lifecycle must arm capture at the first eligible plain-AR forward;
recording from model load accidentally mixes prefill setup into the decode tape
and correctly triggers fail-closed artifact validation. Reproduce with
`scripts/redline_product_bench.py --transport pm4`; `.redline-work/` holds the
local raw JSON and daemon logs and is not a source artifact.

The user-facing `serve_harness.py` was also run with PM4 requested and produced
dated transport/performance evidence at automatic clocks without reported
replay faults. It does not add the missing timed-arm controller or anti-fallback
ledger. A five-prompt greedy battery reported no runaways and averaged 384.7
decode tok/s for Qwen3.5 0.8B and 99.3 tok/s for Qwen3.5 9B; prompt prefill
ranged from 4.7-7.4 ms and 18.3-33.3 ms respectively. The response-coherence
gate did not pass: the current Qwen thinking/content split yielded empty visible
`content` after valid `stop` finishes. That failure reproduces outside the
requested retained transport and remains a separate response-framing issue.

## MTP boundary

The retained transport was also tested on the bundled full-vocabulary
`qwen3.5-9b.mq4-mtp` proposal head. A single retained region reproduces the
first proposal step, but K>1 diverges when a later proposal consumes mutable
token/KV state from the prior proposal. Splitting the region at token-chain
boundaries and switching between PM4 and direct AQL does not restore parity;
K=1 is slower than HIP (124.75 vs 134.78 tok/s). That route is therefore not
shipped or selectable.

The existing HipGraph proposal executor is now permitted to capture the same
full-vocabulary Q8 head. It is token-identical at K=3 (tau 3.4737 and matching
output), but remains neutral/slightly negative at 186.06 vs 187.95 tok/s, so it
stays behind the existing explicit `HIPFIRE_MTP_PROPOSAL_GRAPH=on` opt-in. The
product default consequently requests `auto`/PM4 for eligible ordinary
decode and uses the established HIP MTP path for speculative decode; there is
no unsafe automatic crossover.
