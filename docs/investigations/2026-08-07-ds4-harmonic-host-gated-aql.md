# DS4 harmonic finite host-gated AQL screen

Date: 2026-08-07  
Branch: `ds4-beta-staging`  
Implementation commit: `26cf5143fb0e6f7ac075a95188b7999bbafff089`

## Verdict

Accept the finite host-gated AQL primitive as the replacement for the rejected
per-layer queue transaction shape. A single exact-gfx1100 queue carried 129
dispatches and 43 ordered route checkpoints with one batch publication. All 43
host-written continuation inputs and the terminal output were exact.

The measured gate tax is `6.181 us/gate`, or `0.2658 ms/token` for 43 DS4
layers. The rejected integrated route left approximately `51.66 ms/token` in
checkpoint transaction overhead. This primitive therefore clears the mechanism
screen by a wide margin; it is not yet a model throughput claim.

## Fixture and result

- Device selector: `name:7900`
- Resolved device: AMD Radeon RX 7900 XTX, gfx1100, `0000:66:00.0`
- Dispatches: 129
- Gates: 43
- Samples: 40
- Gated packets: 173, queue ID 2
- Ungated packets: 130, queue ID 3
- Gated median: `566.510 us`
- Ungated median: `300.723 us`
- Added wall time: `265.787 us`, or `6.181 us/gate`
- Cancellation latency: `1684.434 us`
- Correctness: every checkpoint and terminal output exact
- Post-cancel direct-HIP proof: pass
- Post-run KFD owners: none
- GPU lock: released
- Binary SHA-256:
  `1e5e74bafe2a4e0a2a49b7ad5318b277ff0d7713948a205e473b016d630a09d7`

## Contract

The full packet batch is published once to one exact-BDF owner queue. Each gate
has two owner-local HSA signals:

1. the route-stage dispatch publishes a System-release completion signal;
2. the host observes that signal and publishes the typed route externally;
3. owner work after route staging continues independently;
4. an owner-local barrier immediately before combine waits on a host-completed
   resume signal;
5. timeout inactivates the queue, and cancellation was proven recoverable.

No GPU packet waits on a peer GPU or an unbounded memory predicate.

## Next gate

Compose the entire gfx1100 token as one host-gated AQL program, first validate a
two-token exact replay, then run one TG128 screening sample. Only a passing,
useful TG128 result advances to the 2,048/512 promotion fixture.

## Evidence

`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-harmonic-host-gate-v1/`

## Skipped

- No model execution.
- No 2,048/512 promotion run.
- No gfx1151 PM4 or AQL change.
- No Qwen path, weight, quant, sampling, KV, or format change.
