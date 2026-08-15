<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 heterogeneous G1: exact-target cooperative layer chain

## Verdict

G1 passes at `a1c13add9`. The complete 43-layer synthetic ownership chain is
raw-bit exact, keeps all reusable state persistent, has no host wait in the
layer loop, and proves device-timeline overlap on every layer. The warmed
50-sample result is:

| Arm | p50 wall | Relative result | Correctness |
|---|---:|---:|---|
| One-byte sync-only, 86 ROCr copies | 0.462065 ms | fixed dependency control | complete |
| Host-synchronized exact kernels | 134.130327 ms | control | raw-bit pass |
| Device-ordered serial AQL | 102.471360 ms | 1.3090x vs host | raw-bit pass |
| Device-overlapped AQL | **67.172991 ms** | **1.5255x vs serial; 1.9968x vs host** | raw-bit pass |

This is a scheduling and ownership proof, not a model tok/s claim. G2 may now
implement transactional asymmetric loading; production DS4 lowering remains
gated on that loader.

## Fixture

The committed fixture is
`crates/rdna-compute/examples/hetero_gfx11_cooperative.rs`.

- logical GPU 0: `gfx1100`, PCI `0000:66:00.0`;
- logical GPU 1: `gfx1151`, PCI `0000:bf:00.0`;
- 43 layers, hidden 4096, top-k-shaped route metadata for six experts;
- approximately 84 MiB/layer of dense/shared reads on gfx1100;
- approximately 42 MiB/layer of routed/expert reads on gfx1151;
- one contiguous 16,448-byte activation/routing packet to gfx1151;
- one contiguous 16,384-byte routed result to gfx1100;
- two state, activation, branch, and result slots reused by layer parity;
- exact unsigned arithmetic with one CPU oracle for host, serial AQL, and
  overlapped AQL arms.

The byte volumes model the measured per-token dense and routed tiers closely
enough to make overlap material. They deliberately do not claim to model the
production quant kernels' absolute throughput.

## Exact-target compilation

The fixture compiles the same source twice, once per exact target. It does not
reuse the earlier `gfx11-generic` portability image:

| Target | Bundle target ID | SHA-256 |
|---|---|---|
| gfx1100 | `hipv4-amdgcn-amd-amdhsa--gfx1100` | `61402a44fd7a6a23b25ded45042f2e0897235af5054420b9efa1c2207cd4b831` |
| gfx1151 | `hipv4-amdgcn-amd-amdhsa--gfx1151` | `eb56497e6648b4c1fb15aad6a89697fcfe90f5762a24137ecab2a0f969e03283` |

Compiler/runtime identity was ROCm/HIP `7.14.60850-0000000`, AMD clang 23.
The benchmark binary SHA-256 is
`8f9c0f8e247451d01a5cc2cf3aa93ac36964dae1c74064aede0c5a6a75b04abd`.

## Dependency and memory-scope contract

Each layer submits this device-resident DAG:

```text
gfx1100 producer (System release)
  |-- ROCr SDMA engine 0x2 --> gfx1151 System-acquire barrier --> expert
  `-- gfx1100 shared branch --------------------------------------.
                                                                  |
gfx1151 expert (System release) --> ROCr SDMA engine 0x1 ----------+
                                                                  v
                                               gfx1100 System-acquire join
```

All kernel, copy, and barrier completion signals are allocated once and reset
with release semantics only after terminal completion. Both AQL queues are
filled before either doorbell is rung. The serial arm changes only the first
copy dependency from producer completion to shared-branch completion.

HSA dispatch profiling uses the runtime system timestamp domain on both
agents. In the warmed stress pass:

- serial: shared/expert overlap was 0 us on 0/43 layers;
- overlapped: the last reported sample overlapped 34,546.660 us on 43/43
  layers;
- total shared dispatch time was 66,048.225 us and total expert dispatch time
  was 34,546.660 us, so the complete expert branch was hidden by the shared
  branch rather than merely submitted concurrently from the host.

## Correctness and stress

The canonical stress run used three warmups followed by 50 measured samples
per AQL arm. Every measured and warmup output was compared as 4096 raw u32
words against the CPU oracle. The same oracle passed three host-sync samples.
There were zero mismatches, queue faults, negative async-copy completions, or
timeouts across persistent signal, queue, kernarg, and double-buffer reuse.

The first attempt is intentionally retained. Host-sync passed, but AQL output
first differed at element 256 because Clang's hidden block-count/group-size
kernargs were zero. Commit `a1c13add9` populates that Code Object V5/V6 launch
suffix. The failure is evidence that the oracle catches a plausible partial
dispatch rather than only transfer corruption.

## Evidence

Durable evidence is on hipx under:

`/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-hetero-g1/`

| File | SHA-256 |
|---|---|
| `first-run.log` | `20e13293c0d1d2d3f0d07e35066d7907f8eb84a9724a1b922612bae4ec609ebd` |
| `second-run.log` | `f05a96f1efe7611f43d323e85965795abf347b8b009ab1a7c57c527a5cd9fee3` |
| `stress-run.log` | `f3eeeb303cdf529623564ae5359f9d513f659590de1007154b71d08a49e9aa71` |
| `artifact-identity.txt` | `841146a918b34f9ba38b62dc08f2ccc12bd27a09e7fa5827615405e325498fc0` |

## What was skipped

- No model artifact was opened or lowered; that is prohibited before G2.
- No DS4 throughput, quality, or product-performance claim was made.
- No generic `gfx11` code object was used for the hot fixture kernels.
- No RCCL fallback was retried after its concrete mixed-agent blocker in G0.
- No PM4 work was started; dual-device retained replay is G6.

## Next gate

G2 must implement a transactional asymmetric load with typed `gfx1100` dense
and `gfx1151` expert roles, per-device budgets and provenance, rollback after
injected mid-load failure, owner-correct unload/reload, and unchanged symmetric
routes. It must fail closed rather than inheriting a process-global mixed-arch
escape hatch.
