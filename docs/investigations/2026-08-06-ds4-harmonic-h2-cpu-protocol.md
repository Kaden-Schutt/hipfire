<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic H2 CPU protocol checkpoint

Status: partial H2 checkpoint; no GPU execution performed.

## Scope

This checkpoint implements the pure CPU state machine which constrains the
future gfx1100/gfx1151 transport. It does not create a HIP context, queue,
stream, allocation, or device signal, and it does not remove the H0 product
quarantine.

The protocol freezes:

- the DeepSeek V4 Flash 0731 MQ2R artifact identity;
- the route identity, 43-layer extent, top-k 6 routing, and two physical slots;
- dense/source ownership on gfx1100 and expert/destination ownership on
  gfx1151;
- independent source and destination allocation generations;
- monotonically increasing epochs, exact layer/slot mapping, payload extents,
  monotonic deadlines, expert IDs, and raw-bit route weights.

## Reclamation invariant

A terminal slot is reusable only after both sides are safe:

1. the dense/source side has observed or abandoned the terminal result; and
2. the expert/destination side has completed, acknowledged the terminal state,
   or been isolated.

This is deliberately stronger than a single terminal bit. Isolating a failed
producer cannot by itself authorize reuse while the surviving peer might still
have an in-flight read or write. Worker restart is rejected unless the worker
first exited, and its replacement must use a strictly newer allocation
generation.

## CPU evidence

Focused protocol battery:

```text
cargo test -p hipfire-arch-deepseek4 --lib harmonic::tests:: -- --nocapture
10 passed; 0 failed
```

Coverage includes:

- 10,000 exact synthetic double-buffer chains;
- malformed route, owner, generation, extent, and vacant epoch rejection;
- timeout and late-completion rejection;
- mid-copy cancellation and stale-completion rejection;
- source and destination worker loss;
- terminal observation and destination-quiescence requirements;
- strictly newer allocation generation on restart; and
- stale epoch rejection after physical-slot reuse.

The changed-file formatter passed. Repository-wide DS4 library and dependent
crate checks are recorded with the commit that lands this checkpoint.

## H2 work still required

This checkpoint does **not** complete H2. The following exits remain:

- isolated long-lived CPU worker processes with an independent lifecycle per
  future GPU owner;
- persistent shared packet slots and a bounded supervisor teardown path;
- process-level kill, timeout, malformed-packet, producer-loss, and mid-copy
  fault injection in both directions;
- controller wakeup and round-trip measurement; and
- after explicit user authorization, proof that each unaffected GPU can run
  its existing single-device oracle after peer-worker faults.

Until those exits pass, the H0 quarantine remains unconditional and no
harmonic GPU product or performance harness may execute.
