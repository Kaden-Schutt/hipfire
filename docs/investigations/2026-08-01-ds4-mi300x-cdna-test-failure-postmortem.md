# DeepSeek V4 Flash MQ2R on MI300X: failed CDNA port postmortem

Date: 2026-08-01  
Disposition: archived on `ds4-cdna-test-fail`; do not merge into gfx1151  
Exact model SHA-256: `392325b5a8cd284c8f305f23f74f178007a14b88173babeb3f4784ec4fc0e511`

## Executive verdict

The campaign failed its objective. It did not reach 50 tok/s ordinary
autoregressive decode on the exact DeepSeek V4 Flash MQ2R artifact. The best
repo-native `hipfire bench` discovery result was 32.1932 tok/s on a 2048-token
prompt with 32 generated tokens. That result is not publication or promotion
evidence: it is one sample, it is shorter than the 2048/510 acceptance fixture,
and it did not pass a byte-identical two-arm output gate.

The campaign spent too much of its budget repairing a path that was not a
working CDNA port and then optimizing individual inherited kernels. MI300X's
HBM and compute capacity were not the limiting resources. The decode graph
remained a roughly 2,500-2,800-launch serial pipeline per token, leaving the
hardware under-occupied. Narrow kernel wins could not remove the approximately
12 ms/token needed to move from about 31 tok/s to 50 tok/s.

## Measured endpoint

| State | Fixture | Samples | Result | Verdict |
|---|---:|---:|---:|---|
| Corrected repo-native baseline | 2048/32, batch 1, top-k 6, AR | 1 | 31.0982 tok/s | Discovery reference |
| Shared/routed FFN two-stream overlap | Same | 1 | 31.1284 tok/s | Null, +0.097%; rejected |
| Compressor sentinel gate + fused HC finalize | Same | 1 | 32.1932 tok/s | Best discovery, +3.521%; not promoted |
| Target | Ordinary AR | — | 50.0000 tok/s | Missed; another +55.3% was required |

An attempted graph-plus-HC screen was interrupted before it produced a result.
It is not evidence and has no inferred value.

## What went wrong

### 1. The starting path was not a valid CDNA product path

The initial route inherited assumptions from RDNA implementations. Source and
hardware checks found a wave64 attention softmax defect and a grouped-MoE
prefill route that attempted to compile a wave32 WMMA kernel on gfx942. Before
performance work could begin, the campaign had to make the model coherent and
repair routing.

### 2. A correctness bug masqueraded as a context-performance problem

At 2048 prompt tokens, the original indexer top-k implementation fell from a
parallel path into a thread-0 serial selection sort. Decode collapsed to about
2 tok/s. Parallel top-k removed the cliff and restored the low-30 tok/s range,
but this was a portability/correctness repair rather than exploitation of
MI300X's compute or HBM bandwidth.

The route also showed pre-existing greedy-output nondeterminism associated with
non-finite indexer scores. This made the original byte-identical candidate
contract unsatisfiable until the baseline itself was repaired or the authority
contract was changed.

### 3. The model remained launch- and dependency-bound

The campaign's own census placed the token graph near 2,795 launches before
the retained fixes and still around 2,500 after them. The hardware has 304 CUs
and multi-terabyte-per-second HBM, but most batch-1 kernels expose little work
and remain serially dependent. More bandwidth or a faster isolated GEMV cannot
repay thousands of dispatch and synchronization boundaries.

The required move from 31.0982 to 50 tok/s is 32.15 ms/token to 20 ms/token:
about 12.15 ms/token had to disappear. The accepted launch bundle saved only
about 1.10 ms/token.

### 4. The attempted CDNA-native kernels were too narrow

- A wave64x8 MQ2 gate-up candidate was bit-identical but only 0.774x as fast as
  the incumbent.
- A native E8 wave64x4 scheduler was 3.16% slower than its correctness oracle.
- An eight-wave E8 family improved one shared projection pair 1.377x, but its
  occurrence-weighted product ceiling was only about 0.57%; larger shapes
  regressed.
- Group-local O-LoRA LDS8 was 0.828x.
- A two-stream shared/routed FFN overlap product screen was a +0.097% null.

These results do not show that CDNA lacks useful kernels. They show that
optimizing these isolated shapes was the wrong level of attack for this graph.

### 5. Measurement discipline was corrected too late

Several early model-level screens used a custom feeder. Those numbers were
later retracted because they were not produced by the repo-native product
harness. Only the subsequent `hipfire bench` measurements are retained as
product-path discovery data. Time spent interpreting the invalid screens
reduced the budget available for graph-level work.

### 6. The campaign mixed three jobs

It simultaneously attempted architecture bring-up, correctness repair, and
performance optimization. Each is individually substantial. Treating them as
one short performance sprint caused local fixes to be mistaken for progress
toward the 50 tok/s objective.

## What is worth preserving

This failed branch still contains useful forensic work:

- an exact-gfx942, model-owned backend/capability shape intended to prevent
  Qwen, MiniMax, gfx1151, and gfx1100 architecture bleed;
- the ROCm 7.14 rocBLAS solution-enumeration ABI correction;
- wave64 attention and prefill-routing failure evidence;
- the parallel top-k cliff diagnosis and fixture;
- raw-bit primitive tests for MQ2 rotation, HC finalization, E8 grouping, and
  rejected native schedulers;
- exact-byte prompt-file support in the developer harness;
- the complete chronological ledger and 11 MB raw evidence bundle.

Preservation does not imply promotion. Some sources are channel-only rejected
experiments, some product routes were only compile-verified, and the final
bundle was never certified on the 2048/510 fixture.

## Why the campaign is archived instead of repaired

The user ended the CDNA product-port effort and chose to use MI300X without
porting hipfire to it. Continuing to modify the runtime would therefore be both
out of scope and likely to repeat the same failure mode. The correct operational
state is the pre-MI300X accepted hipfire CLI/daemon pair, copied to the host as
an immutable tool binary, with no gfx942 product claim.

## If this work is ever resumed

Do not begin from individual GEMV variants. First establish a coherent,
deterministic stock route and profile the complete token graph with ROCm's CDNA
tools. A new attempt should be admitted only with a graph-level design capable
of removing or coarsening hundreds of serial launches—persistent execution,
large producer/consumer fusion, or another architecture-native scheduling
mechanism. Require a projected end-to-end saving of at least 12 ms/token before
spending on a 50 tok/s campaign.

The raw evidence and ledger are in
[`evidence/ds4-mi300x-cdna-test-fail/`](evidence/ds4-mi300x-cdna-test-fail/).
