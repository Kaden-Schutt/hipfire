# Native-MTP spec-decode: losslessness violation (greedy output derails)

**Status:** open bug, root cause not yet pinpointed. Trunk/quant/corpus all
exonerated. The defect is in the native-MTP accept/verify path.

**Branch:** `mtp-kevin` · **Host:** gfx1151 (Strix Halo) · **Date:** 2026-05-26

## TL;DR

`mtp_only_demo` (native-MTP spec-decode) produces **incoherent greedy output**
that derails into a token attractor after ~30 tokens, on `qwen3.6-27b.mq4`
(a known-good AWQ+GPTQ trunk). Pure AR greedy and DFlash spec-decode on the
**same trunk, same post-merge code, same greedy settings** are **fully
coherent**. Since MTP greedy is supposed to be **lossless** (commit only the
trunk's full-vocab argmax — identical to pure AR), the native-MTP **accept/commit
path is violating losslessness**: it commits tokens that are not the trunk's
argmax. This is a **standing bug**, not a merge regression — the prior session's
transcript shows the identical garbage.

## Symptom

`mtp_only_demo --compressed-serial --mtp-p-min 0.65` (and every other MTP
variant) on the canonical LRU prompt:

```
         if key in  self.cache:
             self._add_to_front(self.cache[key])
             return self.cache[key].value
         return -1
     def  put(self,  key:  0) -> None__:int,value:int__:int__:int__:int__->__:int__None__->__:int__None...   ← attractor
```

First ~30 tokens coherent, then collapses. Other prompts collapse into CJK-token
loops (`的的的的`, `是is` substituting for `is`). τ ≈ 1.6–1.7, tok/s ≈ 7 (below
the AR baseline ~14.3). The `是`-for-`is` substitution is a near-tied-logit flip
signature.

## What was ruled out (with evidence)

All runs greedy (temp=0) on `/local/hipfire/qwen3.6-27b.mq4`, LRU + simple prompts.

| hypothesis | test | result |
|---|---|---|
| cvs / compressed head | full Q8 head vs compressed cvs head | **identical** garbage → not the compression |
| trunk file corruption | md5 edge-hash `/local` vs `/data` | **byte-identical** (intact registry default) |
| chat framing | `--no-chatml` vs chatml | both garbage (chatml → `的的的`) |
| stale kernel cache | force-clean rdna-compute rlib + wipe `.hipfire_kernels/` + move stale cold seed aside → 100% fresh compile | **identical** garbage |
| flash attention mode | `HIPFIRE_ATTN_FLASH=0` (non-flash) | **identical** garbage |
| gfx11 lm_head WMMA | `HIPFIRE_LM_HEAD_WMMA=0` | **identical** garbage |
| quant quality | user confirms AWQ+GPTQ (high quality); on-disk `paro` string is the `WeightTensor` field name, not paro weights | n/a |
| **trunk forward pass** | **pure AR via daemon** (current code, temp=0, no MTP) | **✅ COHERENT** — "Paris", clean LRU |
| **shared batched verify + kv=q8** | **daemon DFlash spec** (batched verify, kv=q8, temp=0) | **✅ COHERENT** |
| **native MTP** | `mtp_only_demo` | **❌ garbage** |

**Key insight:** the garbage is *byte-identical across every kernel toggle*, so
it is upstream of all swappable kernels. Pure AR and DFlash (which share the
batched spec-verify forward and kv=q8) are clean. The defect is therefore
**specific to the native-MTP code path**.

## Not a merge regression

Initial hypothesis was that the master merge (`f1fa68d3`, 2026-05-25 12:43)
regressed a forward primitive the MTP path calls. Falsified:

- **Master never touched the MTP files** (`mtp_compose.rs`, `mtp_spec.rs`,
  `mtp_head.rs`): `git log f1fa68d3^1..f1fa68d3^2 -- <those files>` is empty.
- **Every `.mtp` head we tested is dated after the merge** (`/data` 15:42,
  `/local` 13:17–22:02 vs merge 12:43). "Our new MTP quant" *never existed
  pre-merge* — there is no pre-merge baseline it could have regressed from.
- **The prior session's transcript shows the identical garbage** (itself
  post-merge): `的的的` ×124, `__:int` ×156, `__value__` ×34, `是is` ×9, and the
  same τ values (1.6761 / 1.7000 / 1.6081). The "MTP characterization" we
  remembered as working was these same derailed runs.

Conclusion: **we have no evidence the native-MTP path ever produced coherent
output.** Treat it as a standing implementation bug, not a regression. A
pre-merge bisect is low-value.

## The precise bug

MTP greedy spec-decode is **lossless by construction**: the trunk verifies the
draft chain and only commits tokens equal to its own full-vocab argmax; the
"bonus" token at the first mismatch is the trunk's argmax there. Committed output
**must** therefore equal pure-AR greedy — which we proved is coherent.

It doesn't. So the **accept/verify/commit logic commits tokens that are not the
trunk's argmax.** Either:

1. the **accept comparison** treats a draft token as accepted when it does not
   actually match the trunk's argmax at that position (e.g. compares against the
   wrong logits row / position / a stale argmax / compressed-vocab index vs
   full-vocab id mismatch), or
2. the **bonus token** (trunk argmax at first miss) is read from the wrong
   position / logits row, or
3. the **committed token → next-step embedding/KV** wiring advances state with a
   token that differs from what was scored, desyncing the trunk.

Evidence pointer: LRU run reported `committed_total: 123, accepted_mtp_total: 65,
bonus_total: 57`. If the accept check is wrong, those 65 "accepted" tokens
include non-argmax tokens and the output derails while τ stays low.

## Suspect code

Native-MTP exclusive (DFlash does **not** use these — and DFlash is clean):

- `crates/hipfire-arch-qwen35/src/mtp_spec.rs` — `spec_step_mtp`, the
  accept-prefix / bonus logic, `forward_compressed` path.
- `crates/hipfire-arch-qwen35/src/mtp_compose.rs` — the composite forward over
  `[seed, draft₁..draftₖ]` and argmax extraction (`mtp_lm_argmax`,
  `argmax_f32_batched`).

DFlash uses `qwen35::forward_prefill_batch` / `forward_scratch` and is coherent,
so the shared trunk-verify forward is fine — the divergence is in MTP's own
accept/commit wiring.

## Reproduction

```
# pure AR (coherent baseline) — current daemon, greedy:
cargo build --release --example daemon --features deltanet
printf '%s\n' \
  '{"type":"load","model":"/local/hipfire/qwen3.6-27b.mq4","params":{"max_seq":4096}}' \
  '{"type":"generate","id":"r1","prompt":"What is the capital of France? Answer in one sentence.","temperature":0.0,"max_tokens":40,"repeat_penalty":1.0}' \
  '{"type":"unload"}' | ./target/release/examples/daemon

# native MTP (derails):
./target/release/examples/mtp_only_demo \
  --target /local/hipfire/qwen3.6-27b.mq4 \
  --mtp-head /local/hipfire/qwen3.6-27b-full.mtp \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt --max 120 --no-chatml
```

## Pinpoint progress (probes in `spec_step_mtp`, env-gated)

Three debug probes added to `spec_step_mtp` (uncommitted, on `mtp-kevin`):

| env | what it does | result |
|---|---|---|
| `HIPFIRE_MTP_AR_ONLY=1` | commit only batched-verify `argmax_per_pos[0]`, ignore all drafts (τ=1.0) | **garbage** (`是is`, derail) |
| `+ HIPFIRE_MTP_SKIP_CHAIN=1` | also skip the MTP candidate chain (`mtp_head_forward_block_only`) | **garbage** → chain is NOT the cause |
| `+ HIPFIRE_MTP_AR_FWDSCRATCH=1` | commit `forward_scratch`'s OWN logits on the restored DN state | **✅ COHERENT** |

Conclusions, in order:

1. **Not the accept/commit/multi-token/replay logic** — `AR_ONLY` (commit only
   trunk argmax, no draft acceptance) still derails.
2. **Not the candidate chain** — `SKIP_CHAIN` still derails; the chain doesn't
   corrupt shared GPU scratch.
3. **Not the DN snapshot/restore** — `AR_FWDSCRATCH` commits a single-token
   `forward_scratch` result computed on the *restored* DN state and is coherent,
   so `trunk_snap.save_from`/`restore_to` are intact.
4. **The bug is the batched verify forward.**
   `qwen35::forward_prefill_batch_with_pbs` over `[last_committed, c1..cK]` →
   `verify_hidden[slot 0]` → lm_head → `argmax_per_pos[0]` yields the **wrong
   token**, while a single-token `forward_scratch` for the identical token,
   position, and DN state yields the **correct** token.

## Leading hypothesis

Slot 0 (position `cur_pos`, token `last_committed`) is causally independent of
the future candidate slots, yet its batched-verify prediction is wrong — so the
verify is **attending to / capturing hidden from the future candidate positions**
(a non-causal mask, position-offset, or `per_token_hidden_out` slot-alignment
bug in the batched path as MTP invokes it: `mask_override: None`, lines 764-780).
This also explains the *attractor*: the trunk's verification logits get
contaminated by the very draft tokens it is meant to check → self-reinforcing
loop → `的的的` / `__:int`.

DFlash calls the same `forward_prefill_batch_with_pbs` and is coherent, so the
divergence is in **MTP's specific invocation / hidden capture**, not the function
itself. Note the batched prefill is causal-correct for normal prompt prefill
(the post-prefill seed token is correct) — the fault is specific to the short,
mid-sequence verify batch.

## Verify-call comparison: MTP vs DFlash

Both call `qwen35::forward_prefill_batch_with_pbs`. Arg diff:

| arg | DFlash (clean, speculative.rs:2089) | MTP (garbage, mtp_spec.rs:764) |
|---|---|---|
| `hidden_rb` | `Some` | `None` |
| `gdn_tape` | `Some` | `None` |
| `tree_verify` | `tree_verify` | `None` |
| `mask_override` | `None` | `None` (**same — not the mask**) |

`gdn_tape` is a write-only capture sink during the forward (for later rollback
replay) — capturing vs not should NOT change the forward output math — so it is
probably not the direct cause. Mask is identical. So the divergence is most
likely in **which slot's argmax each caller commits**, not the forward call args.

## Refined conclusion

The batched verify's **first-position (slot-0) output is wrong** vs a single-token
`forward_scratch` for the identical token / position / DN-state.
`AR_ONLY+SKIP_CHAIN` shows it is independent of the candidate *values* in slots
1..K, so it is a structural first-position issue in the batched GDN/attention
path over a short mid-sequence window — not contamination by candidate values
and not the mask. DFlash uses the same batched function but is immune, most
likely because its accept indexing never commits slot-0's argmax the way MTP's
`argmax_per_pos[0]` (accept-check for `candidates[0]` and the `accept_count==0`
bonus) does.

## Open questions / next experiments

1. **Confirm the slot-0 batched bug directly:** in `spec_step_mtp`, after the
   batched verify, also run a single-token `forward_scratch` on the snapshot and
   compare its argmax to `argmax_per_pos[0]`. Log cycles where they differ. (The
   `AR_FWDSCRATCH` probe already shows they differ enough to flip coherence.)
2. **Test the `gdn_tape` hypothesis** (cheap falsification): make DFlash pass
   `gdn_tape: None` on its verify and see if DFlash *also* derails. If it does,
   the no-tape batched path is genuinely buggy (not just a capture sink).
3. **Inspect the batched GDN kernel's first-position handling** for a mid-sequence
   window with KV/DN history (the chunked path in the rewritten
   `forward_prefill_batch` body, qwen35.rs ~5289-5900).

## Workaround (proven, not for ship)

`HIPFIRE_MTP_AR_FWDSCRATCH` commits single-token `forward_scratch` logits and is
coherent — but it forfeits the batched-verify speedup (τ→1, AR-speed). Fix the
batched path rather than ship this.

## Debug scaffolding to remove before commit

`spec_step_mtp` carries three env-gated probes added during this investigation:
`HIPFIRE_MTP_AR_ONLY`, `HIPFIRE_MTP_SKIP_CHAIN`, `HIPFIRE_MTP_AR_FWDSCRATCH`
(and the `skip_chain` local + the chain-skip branch). Remove once fixed.
