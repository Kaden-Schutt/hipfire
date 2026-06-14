# Tokenizer interned symbols + loud OOV (#202 follow-ups + #203)

**Branch:** `refactor/tokenizer-interned-symbols-202-203`
**Base:** `upstream/master @ 5716dcf`
**Predecessor:** PR #229 (`perf/tokenizer-hotpath-202`, closes #202)
**Issues addressed:** #202 follow-ups F3/F4/F6, plus #203
**Date:** 2026-05-10

---

## Why bundle these

The four items share a single architectural change: switch the GPT-2 BPE
encoder from string-keyed merge state to **token-id-keyed** merge state.

| Item | Source | Concern | Resolved by interned symbols |
|------|--------|---------|------------------------------|
| F3   | review | `push_pair` clones `(syms[l], syms[r])` per heap push (~64K clones / encode) | `Vec<u32>` symbols, `(u32,u32)` HashMap keys → no clones |
| F4   | review | `merges` Vec duplicates string data also stored in `merge_pair_rank` | Single ranked Vec of `(left_id, right_id, result_id)` becomes the canonical store |
| F6   | review | `build_merge_pair_rank` clones 300K strings at construction | u32 keys are `Copy` → no construction-time clones |
| #203 | upstream issue | `encode_gpt2_bpe` final walk silently maps OOV symbols to id 0; SP fallback silently drops missing chars | Constructor must resolve every merge symbol to an id → consistency check is forced; encoder operates on validated ids → final-walk `unwrap_or(0)` becomes unreachable |

Doing #203 alone first means adding a consistency-check pass that the
F3/F4/F6 refactor will then absorb. Doing F3/F4/F6 alone first
perpetuates the silent-OOV bug at a deeper level (every `Vec<u32>`
slot needs *some* id, so a missing merge result silently falls back to
id 0 or panics — neither is acceptable). The refactor and the
correctness fix are two halves of the same change.

---

## Data model

### Before (post-#202)

```rust
pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: Vec<(String, String)>,                            // (left_str, right_str), rank-ordered
    merge_pair_rank: HashMap<(String, String), usize>,        // (left_str, right_str) -> rank
    special_tokens: Vec<(String, u32)>,
    bos_id: u32, eos_id: u32, eot_id: Option<u32>,
    is_gpt2_bpe: bool,
}
```

### After

```rust
/// One BPE merge rule, fully resolved to token ids at construction time.
#[derive(Debug, Clone, Copy)]
struct MergeRule {
    left:   u32,  // token id of left symbol
    right:  u32,  // token id of right symbol
    result: u32,  // token id of merged symbol (= left.string + right.string)
}

pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// Rank-ordered result ids: `merges[i]` is the merged token id at rank `i`.
    /// Empty for SP tokenizers. Replaces both the prior `Vec<(String,String)>`
    /// and the `Vec<MergeRule { left, right, result }>` first draft (the
    /// left/right fields turned out to be unread on every code path; pair →
    /// rank lookups go through `merge_pair_rank` directly).
    merges: Vec<u32>,
    /// (left_id, right_id) -> index into `merges` (= rank).
    /// Built once at construction. `(u32,u32)` is Copy — zero clone cost.
    merge_pair_rank: HashMap<(u32, u32), u32>,
    /// For GPT-2 BPE only: byte b -> token id of `byte_to_gpt2_char(b).to_string()`.
    /// Construction guarantees every byte 0..=255 has a valid id, so encode_gpt2_bpe's
    /// initial seed is infallible. `[u32; 256]` keeps the whole table in one cache line × 4.
    /// `None` for SentencePiece tokenizers.
    byte_to_id: Option<[u32; 256]>,
    special_tokens: Vec<(String, u32)>,
    bos_id: u32, eos_id: u32, eot_id: Option<u32>,
    is_gpt2_bpe: bool,
}
```

**Notes on the choice:**

- `merge_pair_rank` value is `u32` not `usize` — merge ranks fit in u32 for any
  realistic vocab (~500K merges max), and 4-byte values keep the HashMap
  smaller than 8-byte usize values.
- `merges: Vec<u32>` is 4 bytes per entry, `Copy`, no heap. Iterating in
  rank order is a Vec scan. Diagnostics (`build_merge_rank_table`) become
  trivial: `for (i, &id) in merges.iter().enumerate() { out.entry(id).or_insert(i); }`.
- `byte_to_id: Option<[u32; 256]>` — stored as a fixed array, not a HashMap.
  At 1KB per Tokenizer it's negligible; lookups become a single load.
- `vocab` and `token_to_id` are unchanged. They're the source of truth for
  string ↔ id mapping; everything else is derived.

### Why not eliminate the `merges` Vec entirely

The diagnostic methods (`build_merge_rank_table`, `merge_rank(id)`, plus
`dump_prompt_heat` per the existing code) need *rank-ordered* iteration.
HashMap doesn't preserve insertion order. Keeping `merges: Vec<MergeRule>`
as the canonical ranked storage and `merge_pair_rank` as the derived
hash index is cleaner than maintaining order in some other structure.

---

## Constructor changes

### Signature

```rust
// before
pub fn from_gguf(gguf: &GgufFile) -> Option<Self>;
pub fn from_hf_json(json_str: &str) -> Option<Self>;
pub fn from_hfq_metadata(metadata_json: &str) -> Option<Self>;
pub fn from_gguf_meta_json(meta: &serde_json::Value) -> Option<Self>;

// after
pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError>;
pub fn from_hf_json(json_str: &str) -> Result<Self, TokenizerError>;
pub fn from_hfq_metadata(metadata_json: &str) -> Result<Self, TokenizerError>;
pub fn from_gguf_meta_json(meta: &serde_json::Value) -> Result<Self, TokenizerError>;
```

### `TokenizerError`

```rust
#[derive(Debug)]
pub enum TokenizerError {
    /// A required metadata field was missing or had the wrong type.
    /// Replaces the silent `None` returns from `?` in the prior `Option` constructors.
    MetadataMissing { field: &'static str },
    /// Raw JSON did not parse.
    MalformedJson(serde_json::Error),
    /// `byte_to_gpt2_char(b)` produced a char with no entry in `token_to_id`.
    /// GPT-2 BPE tokenizers MUST cover every byte 0..=255 — without this,
    /// `encode_gpt2_bpe`'s initial seed silently maps to id 0.
    MissingByteSymbol { byte: u8, char: char },
    /// A merge rule referenced a left/right symbol with no entry in `token_to_id`.
    MissingMergeOperand { rank: usize, left: String, right: String, missing_side: Side },
    /// A merge rule's resolved result (`left + right`) has no entry in `token_to_id`.
    /// The encoder cannot represent the post-merge state without this.
    MissingMergeResult { rank: usize, expected: String },
}

#[derive(Debug)]
pub enum Side { Left, Right, Both }

impl std::fmt::Display for TokenizerError { /* ... */ }
impl std::error::Error for TokenizerError { /* ... */ }
impl From<serde_json::Error> for TokenizerError { /* ... */ }
```

Public type — callers may want to discriminate (e.g. log + degrade vs. abort).

### Construction sequence (all three `from_*` constructors)

After `vocab`, `token_to_id`, raw `merges_strings`, and `is_gpt2_bpe` are
loaded from the source format, run the resolution + check pass:

```rust
fn resolve_merges(
    merges_strings: &[(String, String)],
    token_to_id: &HashMap<String, u32>,
) -> Result<(Vec<MergeRule>, HashMap<(u32, u32), u32>), TokenizerError> {
    let mut merges = Vec::with_capacity(merges_strings.len());
    let mut merge_pair_rank = HashMap::with_capacity(merges_strings.len());
    for (rank, (l_str, r_str)) in merges_strings.iter().enumerate() {
        let left_id  = token_to_id.get(l_str.as_str())
            .copied().ok_or_else(|| TokenizerError::MissingMergeOperand {
                rank, left: l_str.clone(), right: r_str.clone(), missing_side: Side::Left,
            })?;
        let right_id = token_to_id.get(r_str.as_str())
            .copied().ok_or_else(|| TokenizerError::MissingMergeOperand {
                rank, left: l_str.clone(), right: r_str.clone(), missing_side: Side::Right,
            })?;
        // The merged result string is left + right (BPE merge invariant).
        let mut result_str = String::with_capacity(l_str.len() + r_str.len());
        result_str.push_str(l_str);
        result_str.push_str(r_str);
        let result_id = token_to_id.get(result_str.as_str())
            .copied().ok_or(TokenizerError::MissingMergeResult { rank, expected: result_str })?;
        merges.push(MergeRule { left: left_id, right: right_id, result: result_id });
        merge_pair_rank.insert((left_id, right_id), rank as u32);
    }
    Ok((merges, merge_pair_rank))
}

fn build_byte_to_id(token_to_id: &HashMap<String, u32>)
    -> Result<[u32; 256], TokenizerError> {
    let mut out = [0u32; 256];
    for b in 0u32..=255 {
        let ch = byte_to_gpt2_char(b as u8);
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        let id = token_to_id.get(s).copied()
            .ok_or(TokenizerError::MissingByteSymbol { byte: b as u8, char: ch })?;
        out[b as usize] = id;
    }
    Ok(out)
}
```

For SP tokenizers `merges` is empty so `resolve_merges` is a no-op;
`byte_to_id` is `None`. (SP tokenizers don't have the byte-coverage
guarantee — that's a property of GPT-2's byte-level BPE.)

---

## Encoder changes

### `encode_gpt2_bpe`

```rust
// before:  syms: Vec<String>, push_pair clones, final walk does unwrap_or(0)
// after:   syms: Vec<u32>,    push_pair is allocation-free, final walk pushes raw

fn encode_gpt2_bpe(&self, text: &str) -> Vec<u32> {
    let byte_to_id = self.byte_to_id
        .as_ref()
        .expect("encode_gpt2_bpe called on non-GPT2 tokenizer");
    let mut syms: Vec<u32> = text.bytes().map(|b| byte_to_id[b as usize]).collect();
    let n = syms.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return syms; }                  // direct, no map lookup

    let merge_pair_rank = &self.merge_pair_rank;
    let merges = &self.merges;

    // ... (linked list state + heap unchanged in shape)

    let push_pair = |heap: &mut BinaryHeap<_>, syms: &[u32], gen: &[u32], l: usize, r: usize| {
        if let Some(&rank) = merge_pair_rank.get(&(syms[l], syms[r])) {
            heap.push(Reverse((rank, l, gen[l])));          // l stays usize per Q3
        }
    };

    while let Some(Reverse((rank, l, gen_at_push))) = heap.pop() {
        if dead[l] || gen[l] != gen_at_push { continue; }
        let r = next[l]; if r < 0 { continue; } let r = r as usize;

        // No re-hash on pop — fetch result directly via index.
        // Gen-tag invariant guarantees rank still describes (syms[l], syms[r]).
        debug_assert_eq!(merge_pair_rank.get(&(syms[l], syms[r])), Some(&rank),
            "BPE pq invariant: popped rank must match live pair rank");
        syms[l] = merges[rank as usize];                    // O(1) Vec<u32> index
        dead[r] = true;
        gen[l] += 1;

        // ... splice + two re-pushes unchanged
    }

    // Final walk: emit raw u32. No HashMap lookup, no `unwrap_or(0)`.
    let mut result = Vec::with_capacity(n);
    let head = (0..n).find(|&i| prev[i] == -1 && !dead[i]);
    let mut p: i32 = head.map(|i| i as i32).unwrap_or(-1);
    while p >= 0 {
        let pi = p as usize;
        result.push(syms[pi]);
        p = next[pi];
    }
    result
}
```

**Key wins:**
- No `String::clone()` in the hot loop.
- No `String` allocation/concat for `merged`.
- No re-hash on heap pop — `merges[rank]` is a `Vec<u32>` index.
- Final walk emits validated ids — `unwrap_or(0)` is gone, #203 is fixed
  for the BPE path.

**Heap entry size:** `Reverse<(u32, usize, u32)>` = 24 bytes on 64-bit (per Q3
decision: keep `usize` for `l` to avoid cast noise). Same size as the prior
`Reverse<(usize, usize, u32)>` — no shrink, but the syms-Vec shrinks 6× from
`Vec<String>` (24-byte header per entry + heap) to `Vec<u32>` (4 bytes per
entry inline), which is the structural cache-friendliness win.

### `encode_sentencepiece`

**No structural change.** SP doesn't use `merges` or `merge_pair_rank`.

The remaining #203 concern for SP — the `if let Some(&id) = ...` fallback
silently *drops* missing chars — is **out of scope for this PR**. SP input is
unbounded Unicode; we can't pre-validate every input char against vocab at
construction. Possible future fixes (a `encode_strict` variant returning
`Result<Vec<u32>, EncodeError>`, or a counter exposed via a method) are
left for a separate, smaller PR.

The doc comment will be updated to make the silent-skip behavior explicit
so callers aren't surprised.

---

## Diagnostic methods

```rust
// before
pub fn build_merge_rank_table(&self) -> HashMap<u32, usize> {
    let mut out = HashMap::with_capacity(self.merges.len());
    let mut buf = String::new();
    for (i, (l, r)) in self.merges.iter().enumerate() {
        buf.clear();
        buf.push_str(l); buf.push_str(r);
        if let Some(&id) = self.token_to_id.get(&buf) {
            out.entry(id).or_insert(i);
        }
    }
    out
}

// after — no string concat, no fallible lookup, just enumerate
pub fn build_merge_rank_table(&self) -> HashMap<u32, usize> {
    let mut out = HashMap::with_capacity(self.merges.len());
    for (i, m) in self.merges.iter().enumerate() {
        out.entry(m.result).or_insert(i);
    }
    out
}

// before
pub fn merge_rank(&self, id: u32) -> Option<usize> {
    let s = self.vocab.get(id as usize)?;
    if s.len() <= 1 { return Some(0); }
    let mut buf = String::new();
    for (i, (l, r)) in self.merges.iter().enumerate() {
        buf.clear(); buf.push_str(l); buf.push_str(r);
        if buf == *s { return Some(i); }
    }
    None
}

// after
pub fn merge_rank(&self, id: u32) -> Option<usize> {
    if (id as usize) < self.vocab.len() {
        if self.vocab[id as usize].len() <= 1 { return Some(0); }   // base byte
    }
    self.merges.iter().position(|m| m.result == id)
}
```

Both get simpler, both stop allocating, both stop calling `token_to_id` on
the cold path. Naming hazard from F1 (field `merge_pair_rank` vs method
`merge_rank(id)`) remains acceptable — they remain visually distinct
(call syntax vs field access), and the docstring on the field flags
the distinction.

---

## Caller migration (`Option` → `Result`)

Survey needed before final implementation. Known callers from the existing
codebase (verify with grep at implementation time):

- `crates/hipfire-runtime/src/prompt_frame.rs:311` — test code, `expect("test tokenizer")`.
- Engine, daemon, CLI — every model-load path. Likely all currently do
  `Tokenizer::from_*(...).ok_or_else(|| anyhow!("..."))` or similar.

**Migration pattern:** `from_*().ok_or(...)?` → `from_*()?` with `From<TokenizerError>`
implemented for the caller's error type (or `.map_err(|e| ...)`).

The migration is mechanical but touches every call site. Will be one
sweep at the end of the PR after the tokenizer changes compile.

---

## Test plan

### New tests

**Constructor consistency checks (all three `from_*`):**
- `from_gguf_rejects_missing_byte_symbol` — vocab missing one of the
  256 byte-mapped chars; expect `Err(MissingByteSymbol { .. })`.
- `from_gguf_rejects_missing_merge_operand_left` — merges contain a
  left symbol absent from vocab; expect `Err(MissingMergeOperand)`.
- `from_gguf_rejects_missing_merge_operand_right` — same for right.
- `from_gguf_rejects_missing_merge_result` — merges contain a pair
  whose concat isn't in vocab; expect `Err(MissingMergeResult)`.
- `from_gguf_succeeds_on_consistent_vocab` — known-good fixture round-trips.
- (Repeat the structural cases for `from_hf_json` and `from_gguf_meta_json`.)

**Encoder correctness:**
- Existing 6 BPE tests + 4 SP tests must keep passing — they're the byte-
  identicality contract.
- Add `encode_gpt2_bpe_returns_validated_ids`: assert no token in the
  output is id 0 unless the vocab actually maps id 0 to a real symbol that
  appears in the input. Lock the #203 fix.

**Synthesis helpers:**
- Update `synth` and `synth_sp` to always produce a consistent vocab/merges
  pair. They become `Result<Tokenizer, TokenizerError>` returning helpers
  in tests, or panic on inconsistent input (test-internal contract).

### Existing tests must continue passing

- 139 existing lib tests (BPE + SP + prompt_norm + prompt_frame + etc.)
- The `prompt_frame::tests::make_tokenizer()` builder uses `from_hf_json`
  with a 256-byte vocab covering every `byte_to_gpt2_char` output. Per
  the test code that's already the case, but this needs verification —
  the constructor's new byte-coverage check will reject anything less.

### Bench

The user already noted in F8 that #134 closure needs a multi-segment
bench. This refactor should preserve or improve the perf characteristic
of #229; running the same bench before/after lets us confirm.

---

## Risks and unknowns

1. **API break — decided: clean break, bump to `0.2.0`.** `Option<Self>` →
   `Result<Self, TokenizerError>` is visible to every caller of
   `Tokenizer::from_*`. Workspace is currently `0.1.20` (pre-1.0;
   SemVer permits breaking changes on minor bumps); all internal
   consumers use `path = ...` deps so they update atomically.
   Per CLAUDE.md *"Don't use feature flags or backwards-compatibility
   shims when you can just change the code"* — a deprecated
   `Option`-returning shim that calls `try_*().ok()` would re-introduce
   #203's silent-OOV failure for legacy callers, actively defeating the
   PR's purpose. So no shim. Bump workspace version to `0.2.0` in the
   same PR to signal the break.

2. **Existing vocabs that legitimately have inconsistent merges.** If any
   real-world model file has a merge rule whose result isn't in its vocab
   (e.g. due to a quantizer truncating the vocab post-merge-list-export),
   the new constructor will reject it — and that model will stop loading.
   This is *correct behavior* per #203, but it's a regression for any
   user relying on the silent fallback. **Mitigation:** before merging,
   load every model-format fixture in `assets/test/` (or wherever the
   GGUF/MQ4 test fixtures live) through the new constructor and confirm
   they all succeed. If any fail, decide per-case whether the model is
   broken or whether the check is too strict.

3. **Heap entry width — decided: keep `usize` for `l`.** Halving the
   heap with `u32` would save ~256 KB on a 32K-token prompt (1 MB →
   768 KB) at the cost of casts at every push/pop and a defensive
   `assert!(text.len() <= u32::MAX as usize)`. The structural win is
   already coming from `Vec<String>` → `Vec<u32>` for `syms` (24 →
   4 bytes per entry, 6× shrink); squeezing another 25% from the heap
   is diminishing returns and adds cast noise. Smaller diff is the
   right call here.

4. **`debug_assert_eq!` on heap pop** — the existing assertion compares
   the popped `rank` to a re-lookup. With u32 keys it's a free check.
   Keep it for defensive correctness.

5. **#203's SP fallback path stays silent — decided: defer.** Adding
   `encode_strict()` for SP would close #203 fully but expands scope
   onto a separate code path that doesn't share plumbing with the
   constructor consistency check. The constructor check is the high-
   value half (catches inconsistent vocab/merges *at model load*,
   before any inference); the encode-time strict variant is debug
   ergonomics. Defer to a separate, smaller PR if real usage surfaces
   demand for it.

---

## Sequencing

1. **Setup** — add `MergeRule` struct, `TokenizerError` enum, `Side` enum.
   Public exports, Display/Debug/Error impls, `From<serde_json::Error>`.
2. **`resolve_merges` and `build_byte_to_id` helpers** — module-private free
   functions (no `&self`). Test in isolation with a synthetic vocab.
3. **`from_gguf` rewrite** — first constructor; build the new struct
   layout end-to-end. Convert `Option` → `Result`, replace `?` on
   `Option::None` with explicit `MetadataMissing` errors.
4. **`from_hf_json` and `from_gguf_meta_json` rewrite** — mirror the gguf
   pattern. Each gets its own consistency check pass. `from_hfq_metadata`
   delegates to one of the two and propagates `Result`.
5. **`encode_gpt2_bpe` rewrite** — `Vec<u32>` syms, new heap payload
   (keep `l: usize` per Q3 decision), index-based result lookup, raw
   final walk (no `unwrap_or(0)`).
6. **Diagnostic methods (`build_merge_rank_table`, `merge_rank`)** — port
   to the new structure.
7. **Test helpers (`synth`, `synth_sp`)** — update to produce consistent
   vocabs; helper stays panicking (test-internal contract — `expect`
   the construction succeeds since fixtures are hand-built).
8. **New consistency-check tests** — write them last, against the now-
   stable constructor signatures.
9. **Caller migration** — sweep `Tokenizer::from_*` callers across the
   workspace, update each to handle `Result`. Compile-driven: each
   broken site is a clear fix.
10. **Workspace version bump** — `Cargo.toml` workspace `version`:
    `0.1.20` → `0.2.0`. Signals the breaking change to anyone pinning
    `0.1.x`. Per Q4 decision.
11. **Verification** — run `cargo test -p hipfire-runtime --lib`;
    `cargo build` workspace-wide; if available, run the multi-segment
    bench against #229's commit to verify no perf regression.

Estimated diff: ~400–600 lines on `tokenizer.rs`, plus ~5–20 lines per
caller migration site (probably <10 sites). 3–5 hours focused work.

---

## Decisions (locked, 2026-05-10)

| Q | Decision | Rationale |
|---|----------|-----------|
| Q1 — byte-coverage strictness | Reject loud (option a) | Silent fallback is the bug we're fixing |
| Q2 — `encode_strict` for SP | Defer to separate PR | Different code path, debug-ergonomics value, scope hygiene |
| Q3 — heap entry width | Keep `usize` for `l` | Already getting 6× win from `syms: Vec<String>` → `Vec<u32>`; cast noise not worth 25% heap shrink |
| Q4 — API stability | Clean break, bump workspace `0.1.20` → `0.2.0` | Pre-1.0 SemVer permits; all internal consumers atomic; CLAUDE.md forbids compat shims |
