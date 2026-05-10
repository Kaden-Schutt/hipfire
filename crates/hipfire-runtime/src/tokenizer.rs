//! BPE tokenizer loaded from GGUF metadata.
//! Supports encode (text → token IDs) and decode (token IDs → text).

use crate::gguf::{GgufFile, MetaValue};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub struct Tokenizer {
    /// Token ID → string
    vocab: Vec<String>,
    /// String → token ID (for encoding)
    token_to_id: HashMap<String, u32>,
    /// BPE merge rules: (left, right) → merged token
    merges: Vec<(String, String)>,
    /// Pre-built BPE merge-rank lookup: (left, right) → rank index. Built
    /// ONCE per tokenizer construction so `encode_gpt2_bpe` doesn't pay the
    /// O(M) String-clone-into-HashMap cost on every encode call (M is the
    /// merges count, ~150K for Qwen3+ — without this, ~50ms per encode call,
    /// which compounds across the 9+ encodes-per-request the daemon does for
    /// chat-template scaffolding and adds ~450ms to TTFT for short prompts).
    merge_rank: HashMap<(String, String), usize>,
    /// Special tokens: strings like "<|im_start|>" → their token ID
    /// Sorted longest-first for greedy matching
    special_tokens: Vec<(String, u32)>,
    /// Special tokens
    pub bos_id: u32,
    pub eos_id: u32,
    /// Auxiliary end-of-generation id (e.g. `<|endoftext|>` when `eos_id` is
    /// `<|im_end|>`). When a raw-text draft without ChatML finishes naturally
    /// it emits this, not `eos_id` — stop-loops must check both via
    /// `is_terminator()`. None if the vocab only has one terminator.
    pub eot_id: Option<u32>,
    /// True for GPT-2 BPE (Qwen), false for SentencePiece (LLaMA)
    is_gpt2_bpe: bool,
}

/// Pre-build the BPE merge-rank lookup. Called once at tokenizer construction
/// time to amortize the O(M) String-clone-into-HashMap across the lifetime of
/// the tokenizer rather than paying it per encode call.
fn build_merge_rank(merges: &[(String, String)]) -> HashMap<(String, String), usize> {
    merges
        .iter()
        .enumerate()
        .map(|(i, (l, r))| ((l.clone(), r.clone()), i))
        .collect()
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        // Read vocabulary
        let tokens_meta = gguf.meta("tokenizer.ggml.tokens")?;
        let vocab: Vec<String> = match tokens_meta {
            MetaValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    MetaValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => return None,
        };

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Read merge rules
        let merges = if let Some(MetaValue::Array(arr)) = gguf.meta("tokenizer.ggml.merges") {
            arr.iter()
                .filter_map(|v| {
                    if let MetaValue::String(s) = v {
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            Some((parts[0].to_string(), parts[1].to_string()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let bos_id = gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);
        let endoftext = token_to_id.get("<|endoftext|>").copied();
        let im_end    = token_to_id.get("<|im_end|>").copied();
        let eot_id = match (endoftext, im_end) {
            (Some(et), Some(ie)) if et != eos_id && ie == eos_id => Some(et),
            (Some(et), _) if et != eos_id => Some(et),
            _ => None,
        };

        // Detect tokenizer type
        let model_type = gguf.meta_str("tokenizer.ggml.model").unwrap_or("llama");
        let is_gpt2_bpe = model_type == "gpt2";

        // Build special tokens list: vocab entries matching <|...|> or </...> patterns
        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        for (i, tok) in vocab.iter().enumerate() {
            if (tok.starts_with("<|") && tok.ends_with("|>"))
                || (tok.starts_with("<") && tok.ends_with(">") && tok.len() > 3 && !tok.contains(' '))
            {
                special_tokens.push((tok.clone(), i as u32));
            }
        }
        // Sort longest-first for greedy matching
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let merge_rank = build_merge_rank(&merges);

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_rank,
            special_tokens,
            bos_id,
            eos_id,
            eot_id,
            is_gpt2_bpe,
        })
    }

    /// Load tokenizer from HuggingFace tokenizer.json (embedded in HFQ metadata).
    pub fn from_hf_json(json_str: &str) -> Option<Self> {
        let tok: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let model = tok.get("model")?;

        let vocab_map = model.get("vocab")?.as_object()?;
        let vocab_size = vocab_map.len();

        let mut vocab = vec![String::new(); vocab_size + 100];
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        for (token, id_val) in vocab_map {
            let id = id_val.as_u64()? as u32;
            if (id as usize) >= vocab.len() {
                vocab.resize(id as usize + 1, String::new());
            }
            vocab[id as usize] = token.clone();
            token_to_id.insert(token.clone(), id);
        }

        let merges = if let Some(merges_arr) = model.get("merges").and_then(|v| v.as_array()) {
            merges_arr.iter()
                .filter_map(|v| {
                    // HF tokenizer.json stores merges as either "a b" strings or ["a", "b"] arrays
                    if let Some(s) = v.as_str() {
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            return Some((parts[0].to_string(), parts[1].to_string()));
                        }
                    }
                    if let Some(arr) = v.as_array() {
                        if arr.len() == 2 {
                            if let (Some(a), Some(b)) = (arr[0].as_str(), arr[1].as_str()) {
                                return Some((a.to_string(), b.to_string()));
                            }
                        }
                    }
                    None
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        if let Some(added) = tok.get("added_tokens").and_then(|v| v.as_array()) {
            for at in added {
                if let (Some(content), Some(id)) = (
                    at.get("content").and_then(|v| v.as_str()),
                    at.get("id").and_then(|v| v.as_u64()),
                ) {
                    let id = id as u32;
                    if (id as usize) >= vocab.len() {
                        vocab.resize(id as usize + 1, String::new());
                    }
                    vocab[id as usize] = content.to_string();
                    token_to_id.insert(content.to_string(), id);
                    let is_special = at.get("special").and_then(|v| v.as_bool()).unwrap_or(false)
                        || (content.starts_with("<") && content.ends_with(">") && content.len() > 3 && !content.contains(' '));
                    if is_special {
                        special_tokens.push((content.to_string(), id));
                    }
                }
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let bos_id = token_to_id.get("<|endoftext|>").copied()
            .or_else(|| token_to_id.get("<s>").copied())
            .unwrap_or(1);
        let eos_id = token_to_id.get("<|im_end|>").copied()
            .or_else(|| token_to_id.get("<|endoftext|>").copied())
            .or_else(|| token_to_id.get("</s>").copied())
            .unwrap_or(2);
        let endoftext = token_to_id.get("<|endoftext|>").copied();
        let eot_id = match endoftext {
            Some(et) if et != eos_id => Some(et),
            _ => None,
        };

        let is_gpt2_bpe = token_to_id.contains_key("Ġthe") || token_to_id.contains_key("Ġ");

        let merge_rank = build_merge_rank(&merges);

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_rank,
            special_tokens,
            bos_id,
            eos_id,
            eot_id,
            is_gpt2_bpe,
        })
    }

    /// Load tokenizer from HFQ metadata. Tries (in order):
    ///
    ///   1. `meta.tokenizer` as a HuggingFace `tokenizer.json` blob —
    ///      the format the safetensors-side quantizer writes.
    ///   2. `meta.gguf_meta.tokenizer.ggml.*` array fields — the format the
    ///      GGUF-side quantizer writes (preserves the original GGUF
    ///      tokenizer verbatim, no HF-format translation).
    ///
    /// Returns None if neither is present.
    pub fn from_hfq_metadata(metadata_json: &str) -> Option<Self> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
        if let Some(tok_str) = meta.get("tokenizer").and_then(|v| v.as_str()) {
            return Self::from_hf_json(tok_str);
        }
        if let Some(gguf_meta) = meta.get("gguf_meta") {
            return Self::from_gguf_meta_json(gguf_meta);
        }
        None
    }

    /// Load tokenizer from a JSON-serialized GGUF metadata tree. Mirrors
    /// `from_gguf` field-for-field but reads `serde_json::Value` instead of
    /// the live `GgufFile`. Used by the GGUF→MQ4 quantize path so a
    /// converted `.mq4` is fully self-sufficient (no GGUF-on-disk fallback).
    pub fn from_gguf_meta_json(meta: &serde_json::Value) -> Option<Self> {
        let tokens_arr = meta.get("tokenizer.ggml.tokens")?.as_array()?;
        let vocab: Vec<String> = tokens_arr
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        let merges: Vec<(String, String)> = meta
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| {
                        let s = s.as_str()?;
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            Some((parts[0].to_string(), parts[1].to_string()))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let bos_id = meta
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let eos_id = meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as u32;

        let endoftext = token_to_id.get("<|endoftext|>").copied();
        let im_end = token_to_id.get("<|im_end|>").copied();
        let eot_id = match (endoftext, im_end) {
            (Some(et), Some(ie)) if et != eos_id && ie == eos_id => Some(et),
            (Some(et), _) if et != eos_id => Some(et),
            _ => None,
        };

        let model_type = meta
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");
        let is_gpt2_bpe = model_type == "gpt2";

        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        for (i, tok) in vocab.iter().enumerate() {
            if (tok.starts_with("<|") && tok.ends_with("|>"))
                || (tok.starts_with("<") && tok.ends_with(">") && tok.len() > 3 && !tok.contains(' '))
            {
                special_tokens.push((tok.clone(), i as u32));
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let merge_rank = build_merge_rank(&merges);

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_rank,
            special_tokens,
            bos_id,
            eos_id,
            eot_id,
            is_gpt2_bpe,
        })
    }

    /// True if `id` is any end-of-generation terminator (`eos_id` or the
    /// auxiliary `eot_id` — e.g. `<|endoftext|>` when `eos_id` is `<|im_end|>`).
    /// Decode loops MUST check this instead of `== eos_id` — a raw-text draft
    /// without ChatML naturally emits `<|endoftext|>`, not `<|im_end|>`, and a
    /// bare `eos_id` compare silently falls through, causing the post-EOT
    /// attractor loop (bench findings 2026-04-24 §3.5).
    #[inline]
    pub fn is_terminator(&self, id: u32) -> bool {
        id == self.eos_id || self.eot_id == Some(id)
    }

    /// Look up a special token's ID by literal content. Returns `None`
    /// when the token is not registered as a special token in this
    /// tokenizer (e.g. an older Qwen vocab without `<tool_call>`).
    pub fn special_token_id(&self, content: &str) -> Option<u32> {
        self.special_tokens.iter()
            .find(|(s, _)| s == content)
            .map(|(_, id)| *id)
    }

    /// Decode a sequence of token IDs to text.
    /// Handles both GPT-2 BPE (Ġ=space, Ċ=newline) and SentencePiece (▁=space).
    /// For GPT-2 BPE: collects all bytes first, then does UTF-8 conversion once
    /// (individual tokens can be incomplete UTF-8 sequences in byte-level BPE).
    pub fn decode(&self, tokens: &[u32]) -> String {
        if self.is_gpt2_bpe {
            String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
        } else {
            let mut result = String::new();
            for &id in tokens {
                if let Some(tok) = self.vocab.get(id as usize) {
                    let decoded = tok.replace('▁', " ");
                    let decoded = decode_hex_escapes(&decoded);
                    result.push_str(&decoded);
                }
            }
            result
        }
    }

    /// Decode tokens to raw bytes (for incremental UTF-8 streaming).
    /// Use with `std::str::from_utf8()` + `valid_up_to()` to emit only
    /// complete UTF-8 sequences, buffering partial multi-byte chars.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in tokens {
            if let Some(tok) = self.vocab.get(id as usize) {
                if self.is_gpt2_bpe {
                    for ch in tok.chars() {
                        match ch {
                            'Ġ' => bytes.push(b' '),
                            'Ċ' => bytes.push(b'\n'),
                            'ĉ' => bytes.push(b'\t'),
                            c if c.is_ascii() => bytes.push(c as u8),
                            c => {
                                if let Some(b) = gpt2_char_to_byte(c) {
                                    bytes.push(b);
                                } else {
                                    let mut buf = [0u8; 4];
                                    let s = c.encode_utf8(&mut buf);
                                    bytes.extend_from_slice(s.as_bytes());
                                }
                            }
                        }
                    }
                } else {
                    let decoded = tok.replace('▁', " ");
                    let decoded = decode_hex_escapes(&decoded);
                    bytes.extend_from_slice(decoded.as_bytes());
                }
            }
        }
        bytes
    }

    /// Encode text to token IDs.
    /// Special tokens (e.g. <|im_start|>) are matched first, then remaining
    /// segments are encoded via BPE or SentencePiece.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.special_tokens.is_empty() {
            return self.encode_raw(text);
        }

        // Split text at special token boundaries (greedy longest match)
        let mut result = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            // Try to match a special token at current position
            let mut matched = false;
            for (st, id) in &self.special_tokens {
                if remaining.starts_with(st.as_str()) {
                    result.push(*id);
                    remaining = &remaining[st.len()..];
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            // Find the next special token occurrence
            let mut next_special = remaining.len();
            for (st, _) in &self.special_tokens {
                if let Some(pos) = remaining.find(st.as_str()) {
                    if pos < next_special {
                        next_special = pos;
                    }
                }
            }
            // Encode the segment before the next special token
            let segment = &remaining[..next_special];
            if !segment.is_empty() {
                result.extend(self.encode_raw(segment));
            }
            remaining = &remaining[next_special..];
        }
        result
    }

    /// Encode without special token handling.
    fn encode_raw(&self, text: &str) -> Vec<u32> {
        if !self.is_gpt2_bpe {
            return self.encode_sentencepiece(text);
        }
        self.encode_gpt2_bpe(text)
    }

    /// SentencePiece greedy encoding: prepend ▁ for spaces, longest-match lookup.
    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        // SentencePiece convention: spaces become ▁, start of text gets ▁
        let sp_text = text.replace(' ', "\u{2581}");
        let sp_text = format!("\u{2581}{}", sp_text);

        let chars: Vec<char> = sp_text.chars().collect();
        let mut pos = 0;

        while pos < chars.len() {
            // Greedy longest match from vocabulary
            let mut best_len = 0;
            let mut best_id = 0u32;

            for end in (pos + 1..=chars.len()).rev() {
                let candidate: String = chars[pos..end].iter().collect();
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    best_len = end - pos;
                    best_id = id;
                    break;
                }
            }

            if best_len == 0 {
                // Single character fallback — look up the byte
                let ch = chars[pos];
                if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                    tokens.push(id);
                }
                pos += 1;
            } else {
                tokens.push(best_id);
                pos += best_len;
            }
        }
        tokens
    }

    /// GPT-2 BPE encoding (for Qwen3, etc.)
    ///
    /// Implementation: O(N log N) priority-queue BPE. The earlier naive
    /// `loop { full-scan + Vec::remove }` was O(N²) with heavy String
    /// allocation per pair lookup — on 32K-token prompts it spent
    /// 5-10 minutes purely in tokenizer hot path (perf showed >90% of
    /// daemon CPU in encode_raw / Hasher / String::clone / malloc).
    ///
    /// Algorithm:
    /// 1. Symbols held as a doubly-linked list of indices (`prev[i]`,
    ///    `next[i]`). `syms[i]` holds the live string at slot `i`; merged
    ///    slots are tombstoned via `dead[i] = true`.
    /// 2. `gen[i]` increments every time slot `i` absorbs its right
    ///    neighbor — this lets us discard stale heap entries in O(1).
    /// 3. Min-heap keyed on `(rank, left_idx, gen_at_push)`. Every active
    ///    pair `(l, next[l])` is pushed at most once per generation.
    /// 4. Pop best pair; verify `!dead[l]` and `gen[l] == gen_at_push`;
    ///    splice out the right neighbor; push the two newly-formed pairs
    ///    `(prev[l], l)` and `(l, next[l])`.
    ///
    /// Byte-identicality with the naive scan — load-bearing invariant:
    /// the heap key is `(rank, l, gen_at_push)`, ordered lexicographically.
    /// On equal `rank`, the smaller `l` (leftmost surviving symbol) wins
    /// the pop. This matches the naive scan's `if rank < best_rank`
    /// strict-`<` tiebreak (first-occurrence-wins from the left), which
    /// is the BPE encoding contract HuggingFace `tokenizers` and OpenAI
    /// `tiktoken` follow. Future refactors that change the heap key
    /// ordering MUST preserve this leftmost-on-tie property or the
    /// encoded token IDs will silently diverge from every reference
    /// implementation.
    fn encode_gpt2_bpe(&self, text: &str) -> Vec<u32> {
        // 1. Convert text to GPT-2 byte-encoded symbols.
        let mut syms: Vec<String> = text
            .bytes()
            .map(|b| byte_to_gpt2_char(b).to_string())
            .collect();
        let n = syms.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.token_to_id.get(&syms[0]).copied().unwrap_or(0)];
        }

        // 2. Use the pre-built merge rank map cached on the Tokenizer.
        // Earlier this was rebuilt every call (~50ms per encode for Qwen3+
        // with ~150K merges); the daemon makes 9+ encode calls per request
        // for chat-template scaffolding, so the per-call rebuild added
        // ~450ms to TTFT for short prompts. Now O(1) per encode.
        let merge_rank = &self.merge_rank;

        // 3. Doubly-linked-list state. `prev/next` are i32 with -1 sentinel.
        let mut prev: Vec<i32> = (0..n as i32).map(|i| i - 1).collect();
        let mut next: Vec<i32> = (0..n as i32)
            .map(|i| if i + 1 < n as i32 { i + 1 } else { -1 })
            .collect();
        let mut dead: Vec<bool> = vec![false; n];
        let mut gen: Vec<u32> = vec![0; n];

        // 4. Min-heap of (rank, left_idx, gen_at_push). Reverse for min-heap.
        let mut heap: BinaryHeap<Reverse<(usize, usize, u32)>> = BinaryHeap::with_capacity(n);
        let push_pair = |heap: &mut BinaryHeap<Reverse<(usize, usize, u32)>>,
                         syms: &[String],
                         gen: &[u32],
                         l: usize,
                         r: usize| {
            // HashMap key requires owned Strings — clone is the cost. The
            // naive impl did this on every (i, i+1) pair × every merge
            // step (O(N²) clones); we do it O(N log N) times.
            if let Some(&rank) = merge_rank.get(&(syms[l].clone(), syms[r].clone())) {
                heap.push(Reverse((rank, l, gen[l])));
            }
        };

        // 5. Seed heap with initial adjacent pairs.
        for i in 0..n - 1 {
            push_pair(&mut heap, &syms, &gen, i, i + 1);
        }

        // 6. Main merge loop. Each pop is O(log N); validation is O(1);
        // splice is O(1); two pushes are O(log N). Total O(N log N).
        while let Some(Reverse((rank, l, gen_at_push))) = heap.pop() {
            // Validate: slot still alive, generation matches, right neighbor
            // exists. Stale heap entries are dropped here cheaply.
            if dead[l] || gen[l] != gen_at_push {
                continue;
            }
            let r = next[l];
            if r < 0 {
                continue;
            }
            let r = r as usize;

            // The gen-tag invariant guarantees the popped rank still describes
            // the live `(syms[l], syms[r])` pair: any merge that could change
            // either side bumps `gen[l]` (or kills `r`) and would have failed
            // the check above. Verified in debug builds; release trusts it.
            debug_assert_eq!(
                merge_rank
                    .get(&(syms[l].clone(), syms[r].clone()))
                    .copied(),
                Some(rank),
                "BPE pq invariant: popped rank must match live pair rank",
            );

            // Apply the merge: l absorbs r.
            let merged = {
                let mut s = String::with_capacity(syms[l].len() + syms[r].len());
                s.push_str(&syms[l]);
                s.push_str(&syms[r]);
                s
            };
            syms[l] = merged;
            dead[r] = true;
            // Plain `+= 1`: bumps are bounded by N − 1 per slot, N < 2³², so
            // overflow is unreachable. `wrapping_add` would silently un-stale
            // heap entries on overflow — wrong semantics. Debug panics, release
            // aborts: both communicate that wraparound is a bug, not a feature.
            gen[l] += 1;

            // Splice r out of the linked list.
            let nr = next[r];
            next[l] = nr;
            if nr >= 0 {
                prev[nr as usize] = l as i32;
            }

            // Push the two newly-adjacent pairs.
            let pl = prev[l];
            if pl >= 0 {
                // `l`'s left neighbor's right pair is now (pl, l) with new
                // syms[l]; bump pl's gen so its old heap entries die. (Not
                // strictly required since we revalidate on pop, but tightens
                // the invariant.)
                gen[pl as usize] += 1;
                push_pair(&mut heap, &syms, &gen, pl as usize, l);
            }
            if next[l] >= 0 {
                push_pair(&mut heap, &syms, &gen, l, next[l] as usize);
            }
        }

        // 7. Walk the linked list collecting live symbols → token ids.
        // Slot 0 is always the head under our merge-left-into-right invariant,
        // but we scan explicitly for `prev == -1 && !dead`. The scan is O(N)
        // once and is robust against any future invariant breakage (a
        // `debug_assert!` here would be a release-mode no-op and walk
        // tombstoned data silently; the explicit scan can't).
        let mut result = Vec::with_capacity(n);
        let head = (0..n).find(|&i| prev[i] == -1 && !dead[i]);
        let mut p: i32 = head.map(|i| i as i32).unwrap_or(-1);
        while p >= 0 {
            let pi = p as usize;
            result.push(self.token_to_id.get(&syms[pi]).copied().unwrap_or(0));
            p = next[pi];
        }
        result
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Read-only view of the full vocabulary by token id. Index `i` is the
    /// canonical byte string for token id `i`. Used for cross-tokenizer
    /// equivalence checks (PFlash drafter / target compatibility gate).
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// Read-only view of `(string, id)` for every special token (chat / EOT
    /// / image markers, etc.). Sorted longest-first by `Tokenizer::from_*`
    /// constructors. Used for cross-tokenizer equivalence checks.
    pub fn special_tokens(&self) -> &[(String, u32)] {
        &self.special_tokens
    }

    /// Stable 64-bit signature derived from the full vocab + every special
    /// token + bos/eos/eot ids. Two tokenizers with equal signatures are
    /// guaranteed to produce identical encodings for any input drawn from
    /// the shared vocab. Uses fxhash-style mixing — collision resistance
    /// is enough for the equivalence-check use case.
    ///
    /// Cost: O(N) over the vocab, called once per drafter load.
    pub fn signature(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= 0xff;
            h = h.wrapping_mul(0x100000001b3);
        };
        // Vocab in id order (canonical).
        for tok in &self.vocab {
            mix(tok.as_bytes());
        }
        // Specials in their stored order (longest-first; deterministic per
        // constructor).
        for (s, id) in &self.special_tokens {
            mix(s.as_bytes());
            mix(&id.to_le_bytes());
        }
        // Sentinel ids.
        mix(&self.bos_id.to_le_bytes());
        mix(&self.eos_id.to_le_bytes());
        mix(&self.eot_id.unwrap_or(u32::MAX).to_le_bytes());
        h
    }
}

/// GPT-2 byte-to-char mapping (matches OpenAI's bytes_to_unicode() exactly).
/// Printable bytes map to themselves as Unicode chars. Non-printable bytes get
/// sequential codepoints starting from U+0100, in order of byte value.
fn byte_to_gpt2_char(b: u8) -> char {
    let b32 = b as u32;
    match b32 {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => char::from_u32(b32).unwrap_or('?'),
        _ => {
            let offset = GPT2_BYTE_TO_OFFSET[b as usize];
            char::from_u32(256 + offset as u32).unwrap_or('?')
        }
    }
}

/// Reverse of byte_to_gpt2_char.
fn gpt2_char_to_byte(c: char) -> Option<u8> {
    let c = c as u32;
    if (0x21..=0x7E).contains(&c)
        || (0xA1..=0xAC).contains(&c)
        || (0xAE..=0xFF).contains(&c)
    {
        Some(c as u8)
    } else if c >= 256 && c < 256 + 68 {
        GPT2_OFFSET_TO_BYTE.get((c - 256) as usize).copied()
    } else {
        None
    }
}

/// Lookup table: for each non-printable byte, its sequential offset from U+0100.
static GPT2_BYTE_TO_OFFSET: [u8; 256] = {
    let mut table = [0xFFu8; 256];
    let mut n = 0u8;
    let mut b = 0u16;
    while b < 256 {
        let is_printable = (b >= 0x21 && b <= 0x7E)
            || (b >= 0xA1 && b <= 0xAC)
            || (b >= 0xAE && b <= 0xFF);
        if !is_printable {
            table[b as usize] = n;
            n += 1;
        }
        b += 1;
    }
    table
};

/// Reverse lookup: for each sequential offset, the original byte value.
static GPT2_OFFSET_TO_BYTE: [u8; 68] = {
    let mut table = [0u8; 68];
    let mut n = 0usize;
    let mut b = 0u16;
    while b < 256 {
        let is_printable = (b >= 0x21 && b <= 0x7E)
            || (b >= 0xA1 && b <= 0xAC)
            || (b >= 0xAE && b <= 0xFF);
        if !is_printable {
            table[n] = b as u8;
            n += 1;
        }
        b += 1;
    }
    table
};

/// Decode SentencePiece hex escapes like <0x0A> to actual bytes.
fn decode_hex_escapes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            // Try to match <0xHH> pattern
            let mut hex = String::new();
            let mut matched = false;
            let mut temp: Vec<char> = Vec::new();
            temp.push(c);
            if chars.peek() == Some(&'0') {
                temp.push(chars.next().unwrap());
                if chars.peek() == Some(&'x') || chars.peek() == Some(&'X') {
                    temp.push(chars.next().unwrap());
                    // Read hex digits
                    while let Some(&ch) = chars.peek() {
                        if ch.is_ascii_hexdigit() {
                            hex.push(chars.next().unwrap());
                            temp.push(*hex.as_bytes().last().unwrap() as char);
                        } else {
                            break;
                        }
                    }
                    if chars.peek() == Some(&'>') && !hex.is_empty() {
                        chars.next(); // consume '>'
                        if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                            result.push(byte as char);
                            matched = true;
                        }
                    }
                }
            }
            if !matched {
                for ch in temp {
                    result.push(ch);
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Heat-class buckets keyed off BPE merge rank. Lower rank = earlier merge =
/// more common building block during BPE training. Empirical proxy for
/// training-data frequency.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HeatClass {
    /// Base byte / no merge (rank 0). The most universal building blocks.
    Base,
    /// Merge rank < 1000. Top-1k merges — extremely common multi-byte tokens.
    Hot,
    /// Merge rank 1000-9999. Common but not top-tier.
    Warm,
    /// Merge rank 10000-99999. Uncommon — likely a τ depressor when adjacent
    /// to model-defining tokens.
    Cold,
    /// Merge rank ≥ 100000. Exotic / out-of-distribution.
    Frozen,
    /// Token id has no merge entry (special tokens, isolated vocab).
    Unknown,
}

impl HeatClass {
    pub fn from_rank(rank: Option<usize>) -> Self {
        match rank {
            None => Self::Unknown,
            Some(0) => Self::Base,
            Some(r) if r < 1000 => Self::Hot,
            Some(r) if r < 10000 => Self::Warm,
            Some(r) if r < 100000 => Self::Cold,
            Some(_) => Self::Frozen,
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            Self::Base => "BASE   ",
            Self::Hot => "HOT    ",
            Self::Warm => "WARM   ",
            Self::Cold => "COLD   ",
            Self::Frozen => "FROZEN ",
            Self::Unknown => "SPECIAL",
        }
    }
}

impl Tokenizer {
    /// Build a token-id → merge-rank table by scanning the BPE merges list.
    /// O(n_merges) one-time. Used only by diagnostics; not on the hot path.
    pub fn build_merge_rank_table(&self) -> HashMap<u32, usize> {
        let mut out = HashMap::with_capacity(self.merges.len());
        let mut buf = String::new();
        for (i, (l, r)) in self.merges.iter().enumerate() {
            buf.clear();
            buf.push_str(l);
            buf.push_str(r);
            if let Some(&id) = self.token_to_id.get(&buf) {
                out.entry(id).or_insert(i);
            }
        }
        out
    }

    /// Look up a single token's merge rank. For repeated lookups, cache
    /// `build_merge_rank_table` once instead — this method is O(merges).
    pub fn merge_rank(&self, id: u32) -> Option<usize> {
        let s = self.vocab.get(id as usize)?;
        if s.len() <= 1 {
            return Some(0); // base byte
        }
        let mut buf = String::new();
        for (i, (l, r)) in self.merges.iter().enumerate() {
            buf.clear();
            buf.push_str(l);
            buf.push_str(r);
            if buf == *s {
                return Some(i);
            }
        }
        None
    }

    fn rank_of(&self, id: u32, table: &HashMap<u32, usize>) -> Option<usize> {
        table.get(&id).copied().or_else(|| {
            let s = self.vocab.get(id as usize)?;
            if s.len() <= 1 { Some(0) } else { None }
        })
    }

    /// Dump a per-position heat map for `text`, plus a summary line.
    /// Identifies cold-zone tokens that depress draft/target acceptance in DFlash.
    /// Env knobs:
    /// - `HIPFIRE_PROMPT_HEAT_LIMIT=N` — max rows (default 64)
    /// - `HIPFIRE_PROMPT_HEAT_JSON=1` — emit JSON to stdout instead of pretty stderr
    pub fn dump_prompt_heat(&self, text: &str) {
        let ids = self.encode(text);
        let table = self.build_merge_rank_table();
        let total = ids.len().max(1);
        let mut counts = [0usize; 6];
        for &id in &ids {
            counts[HeatClass::from_rank(self.rank_of(id, &table)) as usize] += 1;
        }
        if std::env::var("HIPFIRE_PROMPT_HEAT_JSON").ok().as_deref() == Some("1") {
            let mut s = String::with_capacity(2048);
            s.push_str("{\"bytes\":");
            s.push_str(&text.len().to_string());
            s.push_str(",\"tokens\":");
            s.push_str(&ids.len().to_string());
            s.push_str(",\"summary\":{");
            s.push_str(&format!("\"base\":{},\"hot\":{},\"warm\":{},\"cold\":{},\"frozen\":{},\"special\":{}",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]));
            s.push_str("},\"positions\":[");
            for (pos, &id) in ids.iter().enumerate() {
                if pos > 0 { s.push(','); }
                let rank = self.rank_of(id, &table);
                let decoded = self.decode(&[id]).replace('\\', "\\\\").replace('"', "\\\"")
                    .replace('\n', "\\n").replace('\t', "\\t").replace('\r', "\\r");
                s.push_str(&format!("{{\"pos\":{pos},\"id\":{id},\"rank\":{},\"text\":\"{decoded}\"}}",
                    rank.map(|r| r.to_string()).unwrap_or_else(|| "null".to_string())));
            }
            s.push_str("]}");
            println!("{s}");
            return;
        }
        let limit: usize = std::env::var("HIPFIRE_PROMPT_HEAT_LIMIT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        eprintln!("[token-heat] prompt={} bytes  tokens={}", text.len(), ids.len());
        eprintln!("[token-heat] {:>4}  {:>6}  {:>7}  {:7}  {}", "pos", "id", "rank", "class", "decoded");
        for (pos, &id) in ids.iter().take(limit).enumerate() {
            let rank = self.rank_of(id, &table);
            let class = HeatClass::from_rank(rank);
            let display = self.decode(&[id]).replace('\n', "\\n").replace('\t', "\\t");
            let rank_str = rank.map(|r| r.to_string()).unwrap_or_else(|| "-".to_string());
            eprintln!("[token-heat] {pos:>4}  {id:>6}  {rank_str:>7}  {}  {display:?}", class.label());
        }
        if ids.len() > limit {
            eprintln!("[token-heat] ... ({} more tokens omitted)", ids.len() - limit);
        }
        eprintln!("[token-heat] summary: BASE={} ({:.0}%)  HOT={} ({:.0}%)  WARM={} ({:.0}%)  COLD={} ({:.0}%)  FROZEN={} ({:.0}%)  SPECIAL={} ({:.0}%)",
            counts[0], 100.0*counts[0] as f32/total as f32,
            counts[1], 100.0*counts[1] as f32/total as f32,
            counts[2], 100.0*counts[2] as f32/total as f32,
            counts[3], 100.0*counts[3] as f32/total as f32,
            counts[4], 100.0*counts[4] as f32/total as f32,
            counts[5], 100.0*counts[5] as f32/total as f32);
        let cold_frac = (counts[3] + counts[4]) as f32 / total as f32;
        if cold_frac > 0.05 {
            eprintln!("[token-heat] WARNING: {:.1}% cold tokens — likely τ depressor", 100.0 * cold_frac);
        }
    }
}

/// Collapse runs of 3+ '\n' chars to exactly two.
///
/// Cold zone in BPE merges: `\n\n\n` → token 1358 (RARE) on Qwen3.5/3.6 vocab,
/// while `\n\n` → token 271 (HOT). Rare tokens drop draft/target acceptance
/// (DFlash τ) by ~17% in the worst case observed (PEP-8 PEP-8 strict on 27B-3.5
/// LRU max=120: 161 tok/s τ=8.07 vs single-blank 184 tok/s τ=9.42).
///
/// Single newlines and double newlines pass through unchanged.
pub fn collapse_newline_runs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut nl_run: usize = 0;
    for ch in s.chars() {
        if ch == '\n' {
            nl_run += 1;
            if nl_run <= 2 {
                out.push('\n');
            }
        } else {
            nl_run = 0;
            out.push(ch);
        }
    }
    out
}

/// Replace `\r\n` and bare `\r` with `\n`.
///
/// Cold zone in BPE merges: Qwen3.x training corpora are LF-normalized.
/// `\r` (byte 0x0D) is a non-printable byte that maps to GPT-2 escape
/// `Č` (U+010C); merges containing it have very high rank. Windows-pasted
/// or git-line-ending-mishandled prompts therefore tokenize through cold
/// `\r`/`\r\n` paths instead of the hot `\n` (id 198) / `\n\n` (id 271).
///
/// Order matters: `\r\n` is collapsed first to a single `\n`. A bare `\r`
/// (rare, Mac-classic line endings) is then mapped to `\n` so it doesn't
/// silently survive as a cold byte. `\n\r` (extremely rare, looks like a
/// blank line under reverse-CR convention) becomes `\n\n`.
pub fn normalize_line_endings(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            out.push('\n');
            // Skip an immediately-following \n to avoid CRLF → \n\n.
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Replace U+00A0 NO-BREAK SPACE with regular ASCII space.
///
/// Cold zone in BPE merges: NBSP is rare in code/instruction corpora but
/// shows up via copy-paste from word processors, PDFs, and some Markdown
/// renderers. UTF-8 encodes NBSP as two bytes (0xC2 0xA0) which BPE
/// tokenizes as a high-rank merge (often `Â ` artifacts) rather than the
/// hot ` ` (id 220). Visually identical, semantically equivalent.
pub fn replace_nbsp_with_space(s: &str) -> String {
    s.replace('\u{00A0}', " ")
}

/// Strip trailing whitespace runs (` ` and `\t`) that immediately precede a `\n`.
///
/// Style-only: does not strip trailing whitespace at end-of-string, since
/// completion-mode prompts (e.g. `"def foo():\n    return "`) intentionally
/// end with whitespace and the model is meant to continue from there.
///
/// Cold zone in BPE merges: trailing space/tab before `\n` produces tokens
/// like ` \n` (often cold) or `\t\n` (very cold) instead of clean `\n`.
/// Codebases run through formatters strip these — but copy-pasted snippets
/// or hand-edited prompts often retain them.
pub fn strip_trailing_line_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    // Buffer each pending whitespace character verbatim. A naive `count`
    // approach corrupts tab indentation: `"a\tb"` would flush as `"a b"`,
    // silently downgrading non-trailing tabs to spaces and breaking
    // tab-significant content (Makefiles, TSV, mixed-indent Python).
    let mut pending: Vec<char> = Vec::new();
    for ch in s.chars() {
        match ch {
            ' ' | '\t' => pending.push(ch),
            '\n' => {
                // Drop pending whitespace before the newline.
                pending.clear();
                out.push('\n');
            }
            _ => {
                // Flush pending whitespace verbatim — it's mid-line, keep it
                // exactly as it appeared (tabs stay tabs, spaces stay spaces).
                for &p in &pending {
                    out.push(p);
                }
                pending.clear();
                out.push(ch);
            }
        }
    }
    // End-of-string trailing whitespace: PRESERVE for completion-style prompts.
    for &p in &pending {
        out.push(p);
    }
    out
}

/// Prompt normalization for higher DFlash τ.
///
/// **Default ON since 2026-04-26.** Phase 1 (commit 8a4a211) shipped only
/// `\n{3,}` → `\n\n`, which lifted 27B-3.5 LRU DFlash from 159 → 196 tok/s
/// (+24%) on PEP-8-strict prompts. Phase 3 (issue #40) extends the rule
/// table with more rare-token rewrites. Set `HIPFIRE_NORMALIZE_PROMPT=0`
/// to opt out (rare cases where the raw whitespace/encoding is semantically
/// load-bearing).
///
/// Pipeline (in order):
///   1. `normalize_line_endings` — `\r\n` / `\r` → `\n`
///   2. `replace_nbsp_with_space` — U+00A0 → ` `
///   3. `strip_trailing_line_ws` — drop ` `/`\t` runs before `\n`
///   4. `collapse_newline_runs` — `\n{3,}` → `\n\n`
///
/// Order matters: line-ending normalization first so trailing `\r` doesn't
/// survive as bytes that downstream rules don't recognize. Newline collapse
/// runs last so trailing-ws-stripping can expose adjacent newlines that
/// then collapse (e.g. `"a   \n\n   \n   b"` → `"a\nb"`).
///
/// Returns `Cow::Borrowed` when input is already clean or when explicitly
/// disabled; `Cow::Owned` only on actual rewrite. Each step in the pipeline
/// is itself a no-op fast-path when its trigger pattern is absent.
pub fn maybe_normalize_prompt(s: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    // Default ON. Explicit "0" / "false" / "off" / "no" opts out.
    if let Ok(v) = std::env::var("HIPFIRE_NORMALIZE_PROMPT") {
        let v = v.to_ascii_lowercase();
        if v == "0" || v == "false" || v == "off" || v == "no" {
            return Cow::Borrowed(s);
        }
    }

    let mut cur: Cow<'_, str> = Cow::Borrowed(s);
    if needs_line_ending_normalize(&cur) {
        cur = Cow::Owned(normalize_line_endings(&cur));
    }
    if needs_nbsp_replace(&cur) {
        cur = Cow::Owned(replace_nbsp_with_space(&cur));
    }
    if needs_trailing_ws_strip(&cur) {
        cur = Cow::Owned(strip_trailing_line_ws(&cur));
    }
    if needs_newline_collapse(&cur) {
        cur = Cow::Owned(collapse_newline_runs(&cur));
    }
    cur
}

fn needs_newline_collapse(s: &str) -> bool {
    let mut nl_run: usize = 0;
    for b in s.bytes() {
        if b == b'\n' {
            nl_run += 1;
            if nl_run >= 3 {
                return true;
            }
        } else {
            nl_run = 0;
        }
    }
    false
}

fn needs_line_ending_normalize(s: &str) -> bool {
    s.as_bytes().contains(&b'\r')
}

fn needs_nbsp_replace(s: &str) -> bool {
    // UTF-8 of U+00A0 is 0xC2 0xA0. Cheap two-byte scan beats Unicode-aware
    // contains() for the common no-NBSP case.
    let b = s.as_bytes();
    for i in 0..b.len().saturating_sub(1) {
        if b[i] == 0xC2 && b[i + 1] == 0xA0 {
            return true;
        }
    }
    false
}

fn needs_trailing_ws_strip(s: &str) -> bool {
    // True iff any line in `s` has ` ` or `\t` immediately before `\n`.
    // (End-of-string trailing whitespace is preserved by `strip_trailing_line_ws`,
    // so it doesn't count as "needs strip".)
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'\n' && i > 0 {
            let prev = bytes[i - 1];
            if prev == b' ' || prev == b'\t' {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod bpe_tests {
    use super::*;

    /// Build a synthetic GPT-2 BPE Tokenizer from a vocab list and an ordered
    /// merge list. Used to assert exact `Vec<u32>` output of the priority-queue
    /// encoder against hand-computed expected token sequences. Locks the
    /// byte-identicality contract in CI.
    fn synth(vocab: &[&str], merges: &[(&str, &str)]) -> Tokenizer {
        let vocab: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        let token_to_id: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let merges: Vec<(String, String)> = merges
            .iter()
            .map(|(l, r)| (l.to_string(), r.to_string()))
            .collect();
        let merge_rank = build_merge_rank(&merges);
        Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_rank,
            special_tokens: Vec::new(),
            bos_id: 0,
            eos_id: 0,
            eot_id: None,
            is_gpt2_bpe: true,
        }
    }

    #[test]
    fn encode_full_cascade() {
        // "hello" with merges chained from highest priority to lowest:
        //   ("h","e") rank 0 → "he"
        //   ("l","l") rank 1 → "ll"
        //   ("he","ll") rank 2 → "hell"
        //   ("hell","o") rank 3 → "hello"
        // Final symbol list: ["hello"] → [id 7]
        let tok = synth(
            &["h", "e", "l", "o", "he", "ll", "hell", "hello", "lo"],
            &[("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("l", "o")],
        );
        assert_eq!(tok.encode_gpt2_bpe("hello"), vec![7]);
    }

    #[test]
    fn encode_partial_merge() {
        // "lol" — only ("l","o") is reachable; "ol" is not in merges.
        // Init ["l","o","l"] → after merge ["lo","l"] → [id 8, id 2].
        let tok = synth(
            &["h", "e", "l", "o", "he", "ll", "hell", "hello", "lo"],
            &[("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("l", "o")],
        );
        assert_eq!(tok.encode_gpt2_bpe("lol"), vec![8, 2]);
    }

    #[test]
    fn encode_no_merges() {
        // "ho" — no ("h","o") merge in the table; output is the two byte tokens.
        let tok = synth(
            &["h", "e", "l", "o", "he", "ll", "hell", "hello", "lo"],
            &[("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("l", "o")],
        );
        assert_eq!(tok.encode_gpt2_bpe("ho"), vec![0, 3]);
    }

    #[test]
    fn encode_leftmost_on_tie_priority() {
        // Equal-rank tiebreak invariant: when the same merge could fire at
        // multiple positions, the leftmost wins (matches naive `<` scan and
        // every reference BPE implementation). Heap key `(rank, l, gen)`
        // encodes this — equal rank → smaller `l` pops first.
        //
        // Setup: merges ("a","b") rank 0, ("ab","a") rank 1.
        // Input "ababa" → init ["a","b","a","b","a"].
        // Both (0,1) and (2,3) are ("a","b") at rank 0. Leftmost (0,1) merges
        // first → ["ab","a","b","a"] → ("a","b") at (1,2) → ["ab","ab","a"] →
        // now ("ab","a") at (1,2) rank 1 → ["ab","aba"]. No more merges.
        // Expected: [id of "ab", id of "aba"].
        let tok = synth(
            &["a", "b", "ab", "aba"],
            &[("a", "b"), ("ab", "a")],
        );
        assert_eq!(tok.encode_gpt2_bpe("ababa"), vec![2, 3]);
    }

    #[test]
    fn encode_empty_and_single() {
        let tok = synth(&["a", "b"], &[]);
        assert_eq!(tok.encode_gpt2_bpe(""), Vec::<u32>::new());
        assert_eq!(tok.encode_gpt2_bpe("a"), vec![0]);
    }

    #[test]
    fn encode_long_input_pq_stress() {
        // 1024-byte input exercises the priority-queue path with many merges
        // and many stale heap entries (each merge invalidates ≤ 2 prior
        // entries via the gen tag). Verifies we don't panic, deadlock, or
        // produce a non-decreasing-length output for a known shape.
        let tok = synth(
            &["a", "aa", "aaaa"],
            &[("a", "a"), ("aa", "aa")],
        );
        let input = "a".repeat(1024);
        let out = tok.encode_gpt2_bpe(&input);
        // 1024 bytes → 512 "aa" pairs (rank 0) → 256 "aaaa" (rank 1). No
        // further merges, so the linked list collapses to 256 tokens, all
        // pointing at vocab id 2 ("aaaa").
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&id| id == 2));
    }
}

#[cfg(test)]
mod prompt_norm_tests {
    use super::*;

    #[test]
    fn collapse_three_to_two() {
        assert_eq!(collapse_newline_runs("a\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn collapse_six_to_two() {
        assert_eq!(collapse_newline_runs("a\n\n\n\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn pass_two_unchanged() {
        assert_eq!(collapse_newline_runs("a\n\nb"), "a\n\nb");
    }

    #[test]
    fn pass_one_unchanged() {
        assert_eq!(collapse_newline_runs("a\nb"), "a\nb");
    }

    #[test]
    fn no_newlines_unchanged() {
        assert_eq!(collapse_newline_runs("hello world"), "hello world");
    }

    #[test]
    fn multiple_independent_runs() {
        assert_eq!(
            collapse_newline_runs("a\n\n\nb\n\n\n\nc"),
            "a\n\nb\n\nc"
        );
    }

    #[test]
    fn detector_finds_three() {
        assert!(needs_newline_collapse("a\n\n\nb"));
    }

    #[test]
    fn detector_skips_two() {
        assert!(!needs_newline_collapse("a\n\nb"));
    }

    #[test]
    fn pep8_lrucache_collapses_to_single_blank() {
        // PEP-8 strict snippet: top-level class boundary uses \n\n\n.
        let pep8 = "from typing import Optional\n\n\nclass ListNode:\n    def __init__(self):\n        pass\n\n\nclass LRUCache:\n    pass\n";
        let collapsed = collapse_newline_runs(pep8);
        assert!(!collapsed.contains("\n\n\n"));
        assert!(collapsed.contains("Optional\n\nclass ListNode"));
        assert!(collapsed.contains("pass\n\nclass LRUCache"));
    }

    #[test]
    fn default_on_collapses_when_env_unset() {
        // Default flipped to ON 2026-04-26 — env unset → still collapses.
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
        let s = "a\n\n\nb";
        let out = maybe_normalize_prompt(s);
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        assert_eq!(out.as_ref(), "a\n\nb");
    }

    #[test]
    fn explicit_zero_opts_out() {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        let s = "a\n\n\nb";
        let out = maybe_normalize_prompt(s);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), "a\n\n\nb");
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
    }

    #[test]
    fn cow_borrowed_when_no_runs() {
        // Even with default-ON, no `\n{3,}` runs means no rewrite needed.
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
        let s = "a\n\nb"; // already single-blank
        let out = maybe_normalize_prompt(s);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), "a\n\nb");
    }

    // ---- normalize_line_endings ----

    #[test]
    fn line_endings_crlf_to_lf() {
        assert_eq!(normalize_line_endings("a\r\nb"), "a\nb");
    }

    #[test]
    fn line_endings_bare_cr_to_lf() {
        // Mac-classic line endings: bare \r between content.
        assert_eq!(normalize_line_endings("a\rb\rc"), "a\nb\nc");
    }

    #[test]
    fn line_endings_no_op_pure_lf() {
        assert_eq!(normalize_line_endings("a\nb\n\nc"), "a\nb\n\nc");
    }

    #[test]
    fn line_endings_mixed_crlf_and_lf() {
        // git-attributes mishap: some lines CRLF, some LF.
        assert_eq!(
            normalize_line_endings("a\r\nb\nc\r\nd"),
            "a\nb\nc\nd"
        );
    }

    #[test]
    fn line_endings_lonely_cr_at_end() {
        assert_eq!(normalize_line_endings("a\r"), "a\n");
    }

    #[test]
    fn line_endings_preserves_multibyte_utf8() {
        // Regression: an earlier impl used `out.push(byte as char)`, which
        // re-encoded each UTF-8 byte as a separate codepoint and mangled
        // multi-byte sequences (e.g. NBSP 0xC2 0xA0 → orphan 0xC2 byte
        // surviving as `Â`). Make sure CRLF rewriting and arbitrary multi-byte
        // chars compose correctly.
        let nbsp_with_crlf = "Use\u{00A0}foo\r\nbar";
        let out = normalize_line_endings(nbsp_with_crlf);
        assert_eq!(out, "Use\u{00A0}foo\nbar");
        // NBSP must be exactly one char (2 UTF-8 bytes), not 2 chars.
        assert_eq!(out.chars().filter(|&c| c == '\u{00A0}').count(), 1);
        // Composed pipeline must end up with NBSP gone, CRLF gone.
        let composed = replace_nbsp_with_space(&out);
        assert_eq!(composed, "Use foo\nbar");
        assert!(!composed.as_bytes().contains(&0xC2));
    }

    #[test]
    fn line_endings_preserves_emoji() {
        // 4-byte UTF-8 sequence (🦀 = 0xF0 0x9F 0xA6 0x80) must survive.
        let s = "rust 🦀 \r\nis fast";
        assert_eq!(normalize_line_endings(s), "rust 🦀 \nis fast");
    }

    #[test]
    fn line_endings_detector() {
        assert!(needs_line_ending_normalize("a\r\nb"));
        assert!(needs_line_ending_normalize("a\rb"));
        assert!(!needs_line_ending_normalize("a\nb"));
        assert!(!needs_line_ending_normalize("plain text"));
    }

    // ---- replace_nbsp_with_space ----

    #[test]
    fn nbsp_replaced_with_space() {
        // PDF/word-processor copy-paste artifact: NBSP between words.
        let s = "hello\u{00A0}world";
        assert_eq!(replace_nbsp_with_space(s), "hello world");
    }

    #[test]
    fn nbsp_no_op_on_plain_ascii() {
        assert_eq!(replace_nbsp_with_space("hello world"), "hello world");
    }

    #[test]
    fn nbsp_detector() {
        assert!(needs_nbsp_replace("a\u{00A0}b"));
        assert!(!needs_nbsp_replace("a b"));
        // Other Latin-1 chars starting with 0xC2 must not false-positive.
        assert!(!needs_nbsp_replace("caf\u{00E9}")); // é = 0xC3 0xA9
        assert!(!needs_nbsp_replace("\u{00A2}"));    // ¢ = 0xC2 0xA2
    }

    // ---- strip_trailing_line_ws ----

    #[test]
    fn strip_trailing_spaces_before_newline() {
        assert_eq!(strip_trailing_line_ws("a   \nb"), "a\nb");
    }

    #[test]
    fn strip_trailing_tabs_before_newline() {
        assert_eq!(strip_trailing_line_ws("a\t\t\nb"), "a\nb");
    }

    #[test]
    fn strip_trailing_mixed_ws_before_newline() {
        assert_eq!(strip_trailing_line_ws("a \t \t\nb"), "a\nb");
    }

    #[test]
    fn strip_preserves_eos_whitespace() {
        // Completion-style prompt — model is meant to continue from the trailing space.
        // We MUST NOT strip it.
        assert_eq!(
            strip_trailing_line_ws("def foo():\n    return "),
            "def foo():\n    return "
        );
    }

    #[test]
    fn strip_preserves_midline_whitespace() {
        // Spaces between words / leading indent stay.
        assert_eq!(
            strip_trailing_line_ws("    def foo():\n        return 1\n"),
            "    def foo():\n        return 1\n"
        );
    }

    #[test]
    fn strip_handles_multiple_lines() {
        let dirty = "line one   \nline two\t\nline three  \n";
        assert_eq!(
            strip_trailing_line_ws(dirty),
            "line one\nline two\nline three\n"
        );
    }

    #[test]
    fn strip_preserves_non_trailing_tabs() {
        // Non-trailing tabs MUST round-trip verbatim. Tab-indented Python,
        // Makefile recipes (tabs are syntactically required), and TSV data
        // would silently break if we downgraded tabs to spaces in
        // mid-line position.
        assert_eq!(strip_trailing_line_ws("a\tb"), "a\tb");
        assert_eq!(
            strip_trailing_line_ws("\tdef foo():\n\t\treturn 1\n"),
            "\tdef foo():\n\t\treturn 1\n"
        );
        // Mixed tab + space indentation also round-trips.
        assert_eq!(
            strip_trailing_line_ws("\t \tx = 1\n"),
            "\t \tx = 1\n"
        );
        // Trailing tabs at end of line still get stripped.
        assert_eq!(strip_trailing_line_ws("a\tb\t\nc"), "a\tb\nc");
    }

    #[test]
    fn strip_blank_line_with_indent() {
        // Whitespace-only "blank" line between code blocks — strip the indent.
        assert_eq!(
            strip_trailing_line_ws("a\n    \nb"),
            "a\n\nb"
        );
    }

    #[test]
    fn strip_detector() {
        assert!(needs_trailing_ws_strip("a \nb"));
        assert!(needs_trailing_ws_strip("a\t\nb"));
        assert!(!needs_trailing_ws_strip("a\nb"));
        // EOS whitespace alone does not trigger.
        assert!(!needs_trailing_ws_strip("a "));
        assert!(!needs_trailing_ws_strip("a\nb "));
    }

    // ---- composition through maybe_normalize_prompt ----

    #[test]
    fn pipeline_crlf_and_trailing_ws() {
        // Windows-pasted snippet with trailing whitespace.
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
        let s = "def foo():   \r\n    return 1   \r\n";
        let out = maybe_normalize_prompt(s);
        assert_eq!(out.as_ref(), "def foo():\n    return 1\n");
    }

    #[test]
    fn pipeline_blank_line_indent_then_collapse() {
        // Indented blank line between top-level defs:
        //   "a\n    \n\nb" — line 2 is whitespace-only, lines 2-3 form a `\n\n\n`
        //   run after stripping. Collapse should reduce to `\n\n`.
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
        let s = "a\n    \n\nb";
        let out = maybe_normalize_prompt(s);
        assert_eq!(out.as_ref(), "a\n\nb");
    }

    #[test]
    fn pipeline_nbsp_in_prose() {
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
        let s = "Use\u{00A0}foo()\u{00A0}for\u{00A0}this.";
        let out = maybe_normalize_prompt(s);
        assert_eq!(out.as_ref(), "Use foo() for this.");
    }

    #[test]
    fn pipeline_clean_input_is_borrowed() {
        // No CRLF, no NBSP, no trailing ws, no \n{3,} — must stay Borrowed.
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
        let s = "Plain prompt.\nSecond line.\n\nThird paragraph.\n";
        let out = maybe_normalize_prompt(s);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), s);
    }

    #[test]
    fn pipeline_explicit_opt_out_skips_all_rules() {
        // Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse.
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        let s = "a\r\nb\u{00A0}c   \nd\n\n\ne";
        let out = maybe_normalize_prompt(s);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), s);
        std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT");
    }
}
