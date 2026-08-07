<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic H2 CPU protocol checkpoint

Status: partial H2 checkpoint; all CPU state-machine and isolated-process exits
pass; no GPU execution performed.

## Scope

This checkpoint implements the pure CPU state machine, persistent shared ring,
and isolated worker-process oracle which constrain the future gfx1100/gfx1151
transport. It does not create a HIP context, queue, stream, allocation, or
device signal, and it does not remove the H0 product quarantine.

The protocol freezes:

- the DeepSeek V4 Flash 0731 MQ2R artifact identity;
- the route identity, 43-layer extent, top-k 6 routing, and two physical slots;
- dense/source ownership on gfx1100 and expert/destination ownership on
  gfx1151;
- independent source and destination allocation generations;
- monotonically increasing epochs, exact layer/slot mapping, payload extents,
  monotonic deadlines, expert IDs, and raw-bit route weights.

The second increment adds a file-backed, double-buffered shared-memory ring.
Every metadata and payload word is atomic. Each slot carries the actual
16,448-byte activation and 16,384-byte result, not just an integrity tag.
Publication uses release/acquire transitions through internal `Publishing` and
`Completing` states so a peer cannot consume partial bytes.

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
- mid-service cancellation and stale-completion rejection;
- source and destination worker loss;
- terminal observation and destination-quiescence requirements;
- strictly newer allocation generation on restart; and
- stale epoch rejection after physical-slot reuse.

Process-isolation probe:

```text
./target/debug/examples/harmonic_protocol_probe
status=pass
gpu_touched=false
chains=10000
payload_bytes_per_chain=32832
rpc_count=30000
elapsed_ms=25338
mean_rpc_us=844.624
dense_generation=3
expert_generation=2
```

Two long-lived worker processes mapped the same persistent ring and
round-tripped 328.32 MB of exact payload data. The bounded fault battery
rejected a malformed owner, cancelled an acquired epoch, and expired a running
epoch. It then reserved a slot, wrote half of its activation payload, left the
slot in `Publishing`, killed and reaped the dense producer, and proved that the
surviving expert could not acquire the partial packet. The battery also killed
and replaced the expert during service, then killed and replaced the dense
worker after expert acquisition. The surviving peer remained responsive in
every case, and the final replacement generations completed another exact
chain. No child process remained afterward.

The measurement also rejects per-layer synchronous host RPC as a product
transport: three controller wakeups cost about 2.534 ms per packet chain in
this debug build. At 43 layers that would dominate a token. This process path
is the containment oracle; product notification must be coalesced or moved off
the per-layer critical path after H2 safety exits.

The changed-file formatter passed. Repository-wide DS4 library and dependent
crate checks are recorded with the commit that lands this checkpoint.

## H2 work still required

This checkpoint completes the CPU portion of H2. H2 remains open only for the
hardware exit below:

- after explicit user authorization, prove that each unaffected GPU can run
  its existing single-device oracle after peer-worker faults.

Until those exits pass, the H0 quarantine remains unconditional and no
harmonic GPU product or performance harness may execute.
