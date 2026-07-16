# Redline PM4 resource-derived waits

Branch: `redline`
Host: `hiptrx`, gfx1201, automatic clocks
Model: `qwen3.6-35b-a3b.mq4r`, FWHT3 KV

## Result

The retained PM4 path now derives compute-idle waits from allocation-wide
read/write effects instead of a kernel-name allowlist. Capture resolves each
typed kernarg pointer with `hipMemGetAddressRange`; unknown kernel signatures,
ABI-size mismatches, unknown pointers, and allocation-query failures all retain
the wait.

The hazard check covers the entire outstanding resource frontier since the
last compute idle, not just the adjacent launch pair. This prevents an
independent `A -> B` and `B -> C` pair from incorrectly overlapping when `A`
and `C` conflict.

## Audit and correctness gate

The zero-change audit kept the certified allowlist's PM4 stream while comparing
both decisions at every boundary in the 833-dispatch A3B decode tape:

- 832/832 boundaries had complete typed allocation coverage;
- all 80 existing allowlist elisions were reproduced;
- no allowlist elision conflicted with the resource analysis;
- 50 additional boundaries were independent: 10 K/V cache-writer pairs and 40
  MoE top-k/shared-expert pairs.

Enabling the resource policy removed those 50 additional waits, shrinking the
tape from 32,975 to 32,875 dwords. The 15-consecutive-position gate remained
bit-exact against both the ordinary HIP oracle and captured kernarg-blob oracle,
including logits, KV cache, and recurrent state.

## 8K decode A/B

Matched retained-PM4-only arms used context 8192, max sequence 32768, 100 decode
positions per run, three warmups, ten measured runs, fresh resident daemon
processes, and automatic clocks.

| Arm | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| kernel-name allowlist | 168.833 tok/s | 168.415 tok/s | 168.624 tok/s |
| allocation resources | 172.887 tok/s | 172.246 tok/s | 172.567 tok/s |

The resource policy is **1.02338x (+2.338%)** over the allowlist control. Both
candidate processes exceeded both control processes; the conservative crossed
pair improvements are +2.022% and +2.656%.

## Product gate

The final no-environment default was also checked with the sampled eight-turn
serve harness at max sequence 32768, medium thinking effort, registry sampling,
and 4096 total tokens.

- 8/8 turns completed with visible coherent answers;
- 0 runaway, 0 empty, and 0 attractor cases;
- recall 2/3 on both recall turns;
- average decode 162.7 tok/s;
- turn-context slope: 178.8 tok/s at 49 tokens to 148.2 tok/s at 19,470
  tokens.

## Reproduction artifacts

Raw reports remain on `hiptrx` under:

```text
/home/kaden/.redline-work/hipfire-fence-a/.redline-work/fence-a/
```

Notable files are `frontier-audit-shadow2.json`,
`frontier-resource-shadow15.json`, `wait-allowlist-a.json`,
`wait-allowlist-a2.json`, `wait-resource-b2.json`, and
`wait-resource-default-b3.json`. The sampled product report is
`serve-resource-default.json`.
