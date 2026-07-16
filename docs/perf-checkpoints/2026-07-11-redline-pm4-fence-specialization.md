# Redline PM4 fence specialization

Branch: `redline`
Base: `fe044f551`
Host: `hiptrx`, gfx1201, automatic clocks
Model: `qwen3.6-35b-a3b.mq4r`, FWHT3 KV

## Result

The retained PM4 tape no longer emits full-system `ACQUIRE_MEM` packets around
`fused_silu_mul_mq_rotate` or after `mq_rotate_x`. It still preserves:

- the HIP-to-PM4 ownership-boundary acquire;
- acquires adjacent to `repeat_interleave_qk_f32`;
- acquires adjacent to `rope_partial_halfsplit_f32`;
- all existing compute dependency waits and the terminal compute idle.

This removes 201 mid-tape acquires from the 833-dispatch A3B decode tape and
shrinks it from 34,583 to 32,975 dwords.

## Correctness isolation

Each family was removed independently before combining the safe arms:

| Policy | Consecutive positions | Exact logits/KV/recurrent state | Verdict |
|---|---:|---:|---|
| entry acquire only | 15 | no | invalid |
| omit repeat-interleave | 2 | no | acquire required |
| omit RoPE | 2 | no | acquire required |
| omit fused-SiLU rotation | 2 | yes | redundant |
| omit MQ rotation | 2 | yes | redundant |
| required-only combined | 15 | yes | candidate passes |

The invalid arms disagreed with both the ordinary HIP oracle and the captured
HIP-kernarg-blob oracle. They were not timed as performance candidates.

## 8K decode A/B

Matched retained-PM4-only arms used context 8192, max sequence 16384, 100
decode positions per run, three warmups, ten measured runs, and fresh resident
daemon processes. Alternating control/candidate medians were:

| Arm | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| conservative acquires | 166.974 tok/s | 166.827 tok/s | 166.901 tok/s |
| required-only acquires | 168.907 tok/s | 168.749 tok/s | 168.828 tok/s |

The required-only policy is **1.01155x (+1.155%)** over the conservative PM4
control. A separate HipGraph control drifted between the first two processes,
so the claim uses the alternating PM4-only comparison rather than attributing
that unrelated drift to the fence change.

## Sampled eight-turn product gate

The final candidate used registry sampling, medium thinking effort, 4096 total
tokens, max sequence 32768, and the committed eight-turn recall session:

- 8/8 turns completed with visible coherent answers;
- 0 runaway, 0 empty, 0 attractor;
- recall 2/3 on both recall turns;
- average decode 159.7 tok/s;
- turn-context slope: 175.9 tok/s at 49 tokens to 145.6 tok/s at 19,854 tokens.

An initial harness invocation with `max_tokens=2048` exactly equalled the
medium thinking cap and produced an empty length-stop. It was discarded and is
not a fence result; the corrected gate reserves answer budget beyond thinking.

## Reproduction artifacts

Raw reports remain on `hiptrx` under:

```text
/home/kaden/.redline-work/hipfire-fence-a/.redline-work/fence-a/
```

Notable files are `required-only-shadow15.json`,
`conservative-product.json`, `required-only-product.json`,
`conservative-auto-r2.json`, `required-only-auto-r2.json`, and
`serve-required-32k.json`.
