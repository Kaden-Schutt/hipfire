# DeepSeek V4 gfx1201 TP3 + dedicated DSpark device screen

Date: 2026-08-07

Verdict: **rejected before serving integration**. A dedicated fourth gfx1201
makes DSpark drafting and target-hidden transport cheap, but the resulting
single-stream speculative window is slower than the promoted TP3 AR route.

## Topology and gate

- target: DeepSeek V4 Flash 0731 MQ2R, TP3 on devices 0, 1, and 2
- drafter: certified 0731 DSpark sidecar on device 3
- target position: 2,048
- target batch: 3 tokens
- shipping expert count: top-k 6
- historical code-prompt acceptance statistic: tau 3.02
- one warmup and three timed samples per component
- probe commits: `4ea93d189`, `3b3b3f0ad`, `685fd6854`

The probe explicitly disables sidecar loading on the three target ranks. Device
3 loads only the three DSpark stages, then receives the target's three captured
hidden vectors over peer DMA. Rank-zero embedding and head weights are cloned
once to device 3, so the steady-state draft never reads them remotely.

## Result

| Component | Median |
|---|---:|
| target-hidden peer copy, 49,152 bytes | 2.093 us |
| device-3 DSpark draft | 14.490 ms |
| TP3 batched trunk plus last head, B=3 | 147.656 ms |
| all-row verify heads | 2.064 ms |
| TP3 batched verify total | 149.717 ms |
| TP3 retained sequential verify, 3 tokens | 56.754 ms |

At tau 3.02:

- batched topology projection: 164.208 ms/window, **18.391 tok/s**
- retained-sequential topology projection: 71.246 ms/window, **42.389 tok/s**
- promoted target-only TP3 AR: **54.903757 tok/s**

The TP3 retained control captured three rank graphs with 86 barriers and 7,349
kernarg blobs. Its 56.754 ms for three tokens is consistent with the promoted
18.21 ms/token AR line, so the target route did not regress in this probe.

## Mechanism and reopen condition

Peer transport is not the problem, and the DSpark body is not the problem. The
missing route is a fast small-B TP3 target verify: the current batched prefill
lowering takes 149.717 ms while three retained single-token calls take 56.754
ms. LM-head work is only 2.064 ms; the loss is in the uncaptured B=3 trunk and
its inter-rank synchronization.

To equal 54.903757 tok/s at tau 3.02, the whole speculative window must be at
most 55.005 ms. After the measured 14.490 ms draft and 0.002 ms peer copy, TP3
verify must fall below **40.513 ms**. That is 28.6% below the retained
three-token control and 72.9% below the current batched path.

Do not wire this topology into user-facing serving until an exact, captured
TP3 B=3 verify route independently clears 40.5 ms. The likely prerequisite is
graph-capturing the multi-rank small-B trunk while preserving dynamic inputs
and the existing deterministic peer reductions. A dedicated drafter alone
cannot overcome the current verify cost.

## Evidence and skipped work

Raw logs:

`hiptrx:/home/kaden/ds4-gfx1201-evidence/2026-08-07-tp3-dspark-device3/`

The sidecar SHA-256 was verified as
`bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e`.

This was an admission micro, not a product claim. No serving integration,
decoded-output comparison, sampled generation, long-context run, weight or
format change, top-k change, PM4/Redline work, or adjacent-architecture run was
performed after the screen missed the target decisively.
