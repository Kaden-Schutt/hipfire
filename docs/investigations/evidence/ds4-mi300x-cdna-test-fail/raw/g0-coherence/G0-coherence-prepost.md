# G0 — Coherence PRE/POST attention wave64 fix (gfx942 / MI300X)

Agent: `G0CoherencePrePost`  
Host: `7` (AMD Instinct MI300X VF, `gfx942:sramecc+:xnack-`)  
Worktree: `/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx`  
Evidence root: `/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/g0-coherence/`  
Date (UTC): 2026-08-01

---

## VERDICT

**PASS — wave64 softmax fix restores ordinary-AR decode coherence on gfx942.**

On byte-identical inputs (same MQ2R artifact, same prompt bytes, temp 0, max_tokens 64, batch 1, `HIPFIRE_SPECULATION=off`, fresh daemon binary pinned by sha256):

| Arm | Judgement | finish_reason | tokens | drafter |
|---|---|---|---|---|
| **PRE-FIX** | **garbage** (multilingual/symbol salad after ~1 coherent clause) | `length` (64) | 64 | `ar` |
| **POST-FIX** | **coherent English** (clean two-sentence definition; natural stop) | `stop` | 51 | `ar` |

The defect analysis predicted visibly wrong PRE-FIX output; that is exactly what was observed. After applying the 5-kernel `__shfl(..., 32)` / additive `__shfl_down(..., 32)` fix, clearing the JIT caches, and rebuilding, the same prompt yields fluent, on-topic English that stops at EOS.

No timing/benchmark claim is made. This gate is decode-text coherence only.

---

## EVIDENCE

### Identity (pre-run)

- Hostname: `7`
- GPU: AMD Instinct MI300X VF, `gfx942`, DID `0x74b5`, only KFD peer `gpuagent` (pid 1823) outside our single job
- Worktree HEAD: `d4ab7434a9dad15d0bf6456c8f3c12779ac0edb5` (branch `codex/ds4-mi300x-agentmaxx`)
- PRE dirty-diff sha256: `403be49e54400be70d433225879bf3f5c0eb5985ddb68a27428beee8a675ae4c`
- Model: `/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r`  
  size `82191362222` bytes (byte-frozen; not modified)
- Prompt (exact bytes, no trailing LF):  
  `Explain what a GPU kernel is in two sentences.`  
  sha256 `9a151396bc7ae69dcc2c8a5607ec464391419ae94772e1a63cf096c4885342b1`

### Binary identities (must differ)

| Binary | PRE-FIX sha256 | POST-FIX sha256 |
|---|---|---|
| `target/release/hipfire` | `c1a99f7bd359ac435a1169ba25d85096761a0c757c3c96f6b3381a948acc88d6` | `fa7aa7fa257c6e249bc0e25f06908ff4b41d6a9576a6c7f581aea5461e8bdc88` |
| `target/release/examples/daemon` | `6207b9a3fe2be4a7719acb34b992d8e7ca84b965fa83166cff0aeca9ed10bc67` | `838034a0efa6acdd172311f5aabe65e61370f59882b2840aa0f158120a800ca8` |

PRE md5: hipfire `663c60e60301006227a4e468cab07b5f`, daemon `33b09a7c67566c35d1dd8ff3ecd3675d`  
POST md5: hipfire `6a03d862665e7b8237f6eb107af6f151`, daemon `ce713501043c4f5775ec87c49779d51c`

Running daemon was verified by sha256 against the just-built path via `HIPFIRE_DAEMON_BIN` before each arm (no installed/stale `~/.hipfire/bin/daemon`).

### JIT cache clear

Default hot cache root (`crates/rdna-compute/src/compiler.rs`): **`$CWD/.hipfire_kernels/{arch}`**, overridable by `HIPFIRE_KERNEL_CACHE`.

Moved aside (not deleted) before POST-FIX:

- `/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx/.hipfire_kernels` →  
  `.../g0-coherence/20-jit-cache-cwd-20260801T135117Z.aside` (210 files)
- `/root/.hipfire_kernels` →  
  `.../g0-coherence/20-jit-cache-root-20260801T135117Z.aside` (282 files)

POST-FIX run recreated a fresh `$W/.hipfire_kernels/gfx942` with **180** files, including the product attention objects:

- `deepseek4_attn_swa_batched.{hip,hsaco,hash}`
- `deepseek4_attn_swa_buf.{hip,hsaco,hash}`
- `deepseek4_attn_swa_topk_batched.{hip,hsaco,hash}`
- `deepseek4_attn_swa_topk_f32_buf.{hip,hsaco,hash}`

No fallback / missing-kernel / JIT-failure lines on either arm (`10-prefix-fallback-jit.txt` and `30-postfix-fallback-jit.txt` are empty). PRE-FIX daemon noted `pre-compiled kernels: .hipfire_kernels/gfx942`; POST-FIX had no precompiled dir and JIT-built into the fresh cache.

### Fix application

Transfer verified on arrival:

- `campaign-tracked.patch` sha256 `829b9c1637b4c462d2c792ff9f37f1c14a028b4151030b25ffa47bc774d33f12`
- `campaign-untracked.tgz` sha256 `1073c55755e14625dc624e1f964ba63633d145989e5a9b9e0e9bcea9addf7c1c`

Procedure (no `git checkout -- .`):

1. `git diff HEAD > $E/20-prefix-tree.patch` (snapshot)
2. `git checkout HEAD -- <11 tracked paths in patch>`
3. `git apply` campaign-tracked.patch (clean)
4. `tar -xzf` campaign-untracked.tgz over worktree

Confirm:

- `__shfl(thread_sum, src, 32)` present in all 5 product attn kernels (`batched`, `buf`, `topk_batched`, `topk_buf`, `topk_direct_batched`); additional `source, 32` sites + 2× `__shfl_down(..., 32)` in `topk_buf`
- `MfmaGfx942` at `crates/hipfire-dispatch/src/pipeline/mod.rs:1401,1429,1484`
- Release rebuild: cli ~42 s, daemon ~22 s; binaries differ from PRE (table above)

### Method

`hipfire run` is broken on this tree (omits `attempt_id`). Used a small JSONL stdio driver (`g0-coherence-driver.py`) against `HIPFIRE_DAEMON_BIN=<worktree>/target/release/examples/daemon` with:

```json
{"type":"load","model":"...mq2r","params":{"max_seq":4096,"kv_mode":"q8","kv_backend":"contiguous","speculation":"off","dflash_mode":"off","mtp_mode":"off","ngram_draft":false,"dspark_mode":"off"}}
{"type":"generate","id":"g0-…","attempt_id":1,"prompt":"Explain what a GPU kernel is in two sentences.","max_tokens":64,"temperature":0.0}
```

Env: `HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off`. One GPU job at a time; stuck prior g0-build daemon (pid 53356) terminated before PRE-FIX (gpuagent untouched).

### Verbatim decoded text

#### PRE-FIX (garbage)

```
A GPU kernel is a function or small program that runs on the GPU's cores in parallel,鼎antsay *}---|---|--- económ中海 licensierad(migrations dátummal hostility变成一个acleTreeLabel弯腰allas Presence atividistoitu peيارىгӀ الساعيهually.”# الشعاعيه marketplace- 

）”）” opportunityies a function the next:0b0008 8
```

- sha256: `fdaeae308323eaaaed5a7bc64d54db5dcf85c8c22a5d805ca0dd9f3b421f391c`
- bytes: 354
- **Judgement: garbage** — one partial English clause, then mixed CJK/Arabic/Cyrillic/symbol attractor through the full 64-token length cap.

#### POST-FIX (coherent)

```
A GPU kernel is a function or small program written by a user that is executed in parallel across many threads on the GPU's cores. It defines the computation to be performed by each individual thread, typically operating on a single element of a large data array.
```

- sha256: `cfe868fcf5b68a1e6e3211e91c28fb569e78912b7a11b2bdbac94f20f4404f91`
- bytes: 263
- **Judgement: coherent English** — two clean sentences, on-topic, natural `stop` at 51 tokens (under the 64 cap).

### Side-by-side

| | PRE-FIX | POST-FIX |
|---|---|---|
| Opening | “A GPU kernel is a function or small program that runs on the GPU's cores in parallel,” | “A GPU kernel is a function or small program written by a user that is executed in parallel across many threads on the GPU's cores.” |
| Continuation | multilingual/symbol garbage | “It defines the computation to be performed by each individual thread, typically operating on a single element of a large data array.” |
| Stop | forced `length` @ 64 | natural `stop` @ 51 |
| Text sha256 | `fdaeae30…f391c` | `cfe868fc…04f91` |

### Ordinary-AR proof

**PRE-FIX** (`commit_ready` event; daemon did not echo the `[req …] drafter=ar` line on this arm, but the wire event carries it):

```text
{"type":"commit_ready","id":"g0-prefix",...,"finish_reason":"length","drafter":"ar","attempt_id":1}
```

Load params forced `speculation/dflash/mtp/dspark=off`, `ngram_draft=false`.

**POST-FIX** (event + daemon stderr line):

```text
{"type":"commit_ready","id":"g0-postfix",...,"finish_reason":"stop","drafter":"ar","attempt_id":1}
[req g0-postfix] drafter=ar tau=1.00 tok/s=3.0 decode (51 tok, autoregressive)
```

`grep -F 'drafter=ar'` hits on POST daemon stderr and both event streams. No speculative route observed.

### Peak VRAM

| Arm | peak_used_B | peak_used_GiB |
|---|---|---|
| PRE-FIX | `87435280384` | **81.430** |
| POST-FIX | `87435198464` | **81.430** |

(~84 GB class load as previously reported; residual after unload ~74–79 GB briefly before full free, final KFD only `gpuagent`.)

### Fallback / missing-kernel / JIT-failure lines

**None** on either arm.  
Daemon notes of interest (not failures):

- PRE & POST: `deepseek4: MQ2R P3 tensor recipe verified; gfx1151 route v2 is ineligible on gfx942, using portable dispatch`
- PRE only: `pre-compiled kernels: .hipfire_kernels/gfx942`
- Both: `[rocblas] loaded for gfx942`

No WMMA-fallback or missing-kernel lines that would bear on H2 prefill WMMA were observed on this short AR decode path.

### Step 4 (optional)

`crates/rdna-compute/examples/test_deepseek4_cdna_channels.rs` is **ABSENT** in this checkout — skipped. No synthetic zero-Q/K kill harness was constructed.

### Evidence file inventory

Root: `/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/g0-coherence/`

Identity / cleanup:

- `00-identity.txt`, `00-cleanup.txt`

PRE-FIX (`10-prefix-*`):

- `10-prefix-git-HEAD.txt`, `10-prefix-git-diff.sha256`
- `10-prefix-binaries.sha256`, `10-prefix-binaries.md5`, `10-prefix-prerun-verify.txt`
- `10-prefix-prompt.txt`, `10-prefix-prompt.sha256`, `10-prefix-prompt.od`
- `10-prefix-decoded.txt`, `10-prefix-result.json`, `10-prefix-events.jsonl`
- `10-prefix-daemon.stderr`, `10-prefix-driver.stdout`, `10-prefix-driver.stderr`
- `10-prefix-ar-proof.txt`, `10-prefix-fallback-jit.txt` (empty)
- `10-prefix-vram-{before,after,timeseries}.txt`, `10-prefix-peak-vram.txt`
- `10-prefix-kfd-{before,after}.txt`, `10-prefix-run-meta.txt`, `10-prefix-time.txt`

Fix / rebuild (`20-*`):

- `20-prefix-tree.patch`, `20-prefix-tree.patch.sha256`, `20-prefix-porcelain-before.txt`
- `campaign-tracked.patch`, `campaign-untracked.tgz`, `untracked.list`, `20-transfer-sha256.txt`
- `20-patch-tracked-paths.txt`, `20-git-apply.log`, `20-fix-confirm.txt`, `20-untracked-members.txt`
- `20-build-{cli,daemon}.{log,time}`, `20-build-summary.txt`
- `20-jit-cache-before.txt`, `20-jit-cache-cleared.txt`
- `20-jit-cache-cwd-20260801T135117Z.aside/`, `20-jit-cache-root-20260801T135117Z.aside/`
- `20-kfd-before-fix.txt`

POST-FIX (`30-postfix-*`):

- `30-postfix-binaries.sha256`, `30-postfix-binaries.md5`, `30-postfix-prerun-verify.txt`
- `30-postfix-prompt.txt`, `30-postfix-prompt.sha256`
- `30-postfix-decoded.txt`, `30-postfix-result.json`, `30-postfix-events.jsonl`
- `30-postfix-daemon.stderr`, `30-postfix-driver.stdout`, `30-postfix-driver.stderr`
- `30-postfix-ar-proof.txt`, `30-postfix-fallback-jit.txt` (empty)
- `30-postfix-vram-{before,after,timeseries}.txt`, `30-postfix-peak-vram.txt`
- `30-postfix-kfd-{before,after}.txt`, `30-postfix-run-meta.txt`, `30-postfix-time.txt`
- `30-postfix-jit-created-count.txt`, `30-text-compare.sha256`

Summary / driver:

- `40-summary-raw.txt`, `40-decoded.sha256`, `40-decoded.wc`, `40-channel-test-note.txt`
- `g0-coherence-driver.py`

Local report path (this file):  
`/home/kaden/ClaudeCode/autorocm/hipfire/.claude/worktrees/ds4-mi300x-agentmaxx/.codeinsight+research/ds4-mi300x-agentmaxx/reports/G0-coherence-prepost.md`

---

## INVARIANTS

- Model bytes untouched; only campaign worktree + evidence root written.
- `/root/hipfire-work/ds4-gfx942-port` WIP left alone; source tree `/home/kaden/ClaudeCode/autorocm/hipfire` not modified.
- No `git checkout -- .`; only the 11 tracked paths in the new patch were reset to HEAD before apply.
- Single GPU job; `gpuagent` never killed; no `/tmp` canonical evidence.
- Ordinary AR only: `HIPFIRE_SPECULATION=off` + load params off for dflash/mtp/ngram/dspark; both arms prove `drafter=ar`.
- `HIPFIRE_DAEMON_BIN` pinned to worktree-built daemon; sha256 verified each arm.
- JIT caches moved aside (not deleted) so POST-FIX could not silently reuse PRE-FIX code objects.
- No push, upload, package install, or model delete.
- No timing/ABBA/benchmark measurement in this job.

---

## UNKNOWNS

1. **Which exact attention kernel name served this short prompt** (SWA buf vs topk vs batched) was not pinned in the daemon log beyond the JIT directory listing that includes all four product families. [INFERENCE] topk path is likely for DS4 product decode, but not wire-proven here.
2. **PRE-FIX arm did not emit the `[req …] drafter=ar tau=…` stderr line** (only the `commit_ready.drafter=ar` field). POST-FIX did emit it. Cause not investigated; AR is still proven from the event field on both arms.
3. **Terminal-control `commit` → `aborted/client_cancelled` quirk on PRE** (text still fully delivered via token events before commit_ready). POST completed with clean `done`. Driver issue / race, not model output — decoded text was taken from token stream / result JSON.
4. **H2 WMMA prefill question** is untouched; this gate is short AR decode only. No missing-kernel/WMMA lines appeared, but that does not certify prefill WMMA on gfx942.
5. **Synthetic kill test** (zero-Q/K, all-ones-V → 0.5 vs 1.0) not run — no ready harness in tree.
6. **Long-context / 2048p/510g product fixture** not exercised here (out of G0-coherence scope).

---

## NEXT ACTION

1. Treat G0 coherence as **unblocked for A**: the wave64 softmax denominator fix is necessary and, on this fixture, sufficient to turn garbage decode into coherent English.
2. Conductor may promote the campaign delta (attn width-32 shfl + `MfmaGfx942` import) as the new baseline on `codex/ds4-mi300x-agentmaxx` after any remaining G0 census/firewall agents land.
3. Proceed to ordinary-AR product measurements only after the validation-owned 2048/510 fixture + bench `--max-tokens` gap are closed — do not claim tok/s from this job.
4. Optionally land a tiny CDNA attention kill test (0.5→1.0 sumexp fixture) under `rdna-compute` examples so future arch ports cannot regress this class of shuffle bug silently.

---

## KILL CRITERION

Abandon or re-open this gate immediately if any of the following is later shown:

- POST-FIX coherent text was produced by a **stale/pre-fix code object** (JIT cache not cleared, or daemon sha mismatched the post-fix binary).
- Either arm ran a **non-AR** drafter (`drafter!=ar`) or any dflash/mtp/dspark/ngram path.
- Model artifact bytes, prompt bytes, temperature, or max_tokens differed across arms.
- POST-FIX decode is re-run and returns garbage/degenerate text under the same identities.
- A first-divergence oracle later shows the attention denominator is still wrong on gfx942 product shapes despite coherent short English (coherence is necessary, not a full numerical certificate).
- Competing GPU process or non-`gfx942` device served either arm.
