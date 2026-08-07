# DS4 harmonic H3 ring data plane

Date: 2026-08-07

## Verdict

The product-shaped harmonic ring is admitted to H3 composition. It moves one
full routed-layer activation/result chain between independent persistent
processes without a per-layer control message. The ten-run release median is
4.626 us per 32,832-byte chain, projecting to 0.199 ms for all 43 routed layers.
This is a CPU transport micro, not a model-throughput claim.

## Change

- Route identity advanced from `DS4HARM1` to `DS4HARM2`; wire version is 2.
- Product mode uses one-writer bulk payload copies followed by release state
  publication and acquire observation.
- Byte-wise FNV remains available in the correctness-oracle integrity mode but
  is absent from the product hot path.
- The persistent worker polls epochs directly from the mmap ring. The Unix
  socket remains cold lifecycle control only.
- An older terminal occupant of a reused physical slot is treated as
  backpressure. A genuinely newer occupant remains a fail-closed stale-worker
  error.

## Measurement

Command:

```text
cargo build -q --release -p hipfire-arch-deepseek4 --example harmonic_ring_probe
target/release/examples/harmonic_ring_probe
```

Fixture per process:

- 10,000 sequential chains
- 16,416-byte activation plus 16,416-byte result per chain
- independent persistent worker process
- zero per-chain socket, JSON, signal, launch, or control messages
- no HIP or ROCr runtime opened

Release measurements in microseconds per chain:

```text
2.856  2.820  4.571  4.680  3.187
5.585  3.826  5.841  5.494  6.719
```

Median: 4.626 us. Range: 2.820-6.719 us. Ten of ten rebuilt release runs
passed 10,000 chains. At 43 routed layers, the median host-ring projection is
0.199 ms/token.

The prior correctness-oracle implementation measured 74.593 us per chain
after buffer preallocation, or approximately 3.21 ms across 43 layers. The
product path is approximately 16.1 times leaner by the median comparison.

## Correctness observation

An initial repetition of the pre-fix release binary failed when the worker
advanced to epoch N+2 while the corresponding two-slot physical slot still
held completed epoch N. The protocol incorrectly classified the older
occupant as a stale worker. The corrected rule distinguishes older-slot
backpressure from a newer-slot skipped-epoch fault. The focused IPC suite is
9/9 after the fix, and the rebuilt release probe is 10/10.

## Next gate

Wire the data plane into the real DS4 gfx1100 dense/router source and the
already-implemented gfx1151 resident expert service. Measure actual HIP
composition before lowering either owner-local route to retained PM4.
