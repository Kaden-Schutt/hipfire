# Redline PM4 stateful encoding

Branch: `redline`
Base: `fd472a1b2`
Host: `hiptrx`, gfx1201, automatic clocks
Model: `qwen3.6-35b-a3b.mq4r`

## Result

The retained PM4 encoder can now omit unchanged SH-register runs within one
indirect buffer. Three policies were isolated:

- `legacy`: re-emit every dispatch register;
- `static`: retain only queue-global invariants (`COMPUTE_TMPRING_SIZE`,
  `COMPUTE_RESOURCE_LIMITS`, and the four shader-engine masks);
- `stateful`: retain all unchanged program, resource, workgroup, user-data, and
  invariant registers.

The certified default is the conservative `static` policy. Full state retention
remains available with `HIPFIRE_REPLAY_PM4_STATEFUL=1`, but is not the default
because its additional command reduction did not improve throughput.

| Policy | FWHT3 tape | Reduction |
|---|---:|---:|
| legacy | 32,875 dwords | — |
| static | 22,891 dwords | 30.37% |
| full stateful | 18,090 dwords | 44.97% |

## Correctness

The default static policy passed 15 consecutive positions bit-exact against
both ordinary HIP and exact captured-kernarg-blob execution. Logits, KV cache,
and recurrent state all matched. Full state retention passed the same
15-position gate, so the decision to leave it opt-in is performance-based, not
a known correctness failure.

## Five-run `tg128`

Ordinary AR used Q8 KV, DFlash/speculation off, the canonical 128-token prompt,
fresh resident daemon processes, and automatic clocks.

| Policy | Process medians | Mean | Change vs legacy |
|---|---|---:|---:|
| legacy | 185.60, 184.04 | 184.82 tok/s | — |
| static | 184.90 | 184.90 tok/s | +0.04% |
| full stateful | 184.00, 184.54 | 184.27 tok/s | -0.30% |

Static retention is neutral at short context. Suppressing dynamic
program/workgroup rewrites did not help and produced a small aggregate
regression, despite the smaller command stream.

## 8K decode

Matched retained-PM4-only FWHT3 arms used context 8192, max sequence 32768,
100 positions per row, three warmups, ten measured rows, fresh processes, and
automatic clocks.

| Policy | Process medians | Mean | Change vs legacy |
|---|---|---:|---:|
| legacy | 172.160, 172.039, 171.881 | 172.027 tok/s | — |
| static | 172.687, 173.472 | 173.079 tok/s | **+0.612%** |
| full stateful | 172.467 | 172.467 tok/s | +0.256% |

The static arm won both interleaved comparisons and the final reverse legacy
control stayed below both static runs. The improvement is small but repeatable;
the primary structural result is that command parsing is not a major remaining
decode bottleneck even after removing 30-45% of the tape.

## Sampled product gate

The final no-environment static default used FWHT3 KV, registry sampling,
medium thinking effort, 4096 total tokens, max sequence 32768, and the committed
eight-turn coding/recall session:

- 8/8 turns completed with visible coherent answers;
- 0 runaway, 0 empty, and 0 attractor cases;
- recall 2/3 on both recall turns;
- average decode 163.1 tok/s;
- turn-context slope: 180.2 tok/s at 49 tokens to 147.4 tok/s at 20,390
  tokens.

## Reproduction artifacts

Raw reports remain on `hiptrx` under:

```text
/home/kaden/.redline-work/hipfire-fence-a/.redline-work/fence-a/
```

Notable files are `static-fwht3-shadow15.json`,
`stateful-fwht3-shadow15.json`, `stateful-8k-legacy-a.json`,
`stateful-8k-legacy-a2.json`, `stateful-8k-legacy-a3.json`,
`stateful-8k-static-c.json`, and `stateful-8k-static-c2.json`. The sampled
product report is `serve-static-default.json`.
