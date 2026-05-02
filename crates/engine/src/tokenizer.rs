//! BPE tokenizer loaded from GGUF metadata.
//! Supports encode (text → token IDs) and decode (token IDs → text).

use crate::gguf::{GgufFile, MetaValue};
use std::collections::HashMap;

pub struct Tokenizer {
    /// Token ID → string
    vocab: Vec<String>,
    /// String → token ID (for encoding)
    token_to_id: HashMap<String, u32>,
    /// BPE merge rules: (left, right) → merged token
    merges: Vec<(String, String)>,
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

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
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

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
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

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
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
    fn encode_gpt2_bpe(&self, text: &str) -> Vec<u32> {
        // Convert text to GPT-2 byte-encoded tokens
        let byte_tokens: Vec<String> = text
            .bytes()
            .map(|b| {
                let ch = byte_to_gpt2_char(b);
                ch.to_string()
            })
            .collect();

        // Apply BPE merges greedily
        let mut symbols = byte_tokens;

        // Build merge priority map
        let merge_rank: HashMap<(String, String), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, (l, r))| ((l.clone(), r.clone()), i))
            .collect();

        loop {
            if symbols.len() < 2 {
                break;
            }

            // Find the highest-priority (lowest rank) merge
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break; // no more merges possible
            }

            // Apply the merge
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        // Convert symbols to token IDs
        symbols
            .iter()
            .map(|s| self.token_to_id.get(s).copied().unwrap_or(0))
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
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

/// Prompt normalization for higher DFlash τ.
///
/// **Default ON since 2026-04-26.** Empirical: collapsing `\n{3,}` → `\n\n`
/// at engine entry lifts 27B-3.5 LRU DFlash from 159 → 196 tok/s (+24%) by
/// putting the BPE tokenizer on a tighter τ trajectory. Tested across
/// PEP-8-strict (3-newline) prompts that are dominant in real-world code.
/// Set `HIPFIRE_NORMALIZE_PROMPT=0` to opt out (rare cases where the raw
/// `\n{3,}` whitespace is semantically load-bearing).
///
/// Returns Cow::Borrowed when input has no `\n{3,}` runs or when explicitly
/// disabled; Cow::Owned only on actual rewrite.
pub fn maybe_normalize_prompt(s: &str) -> std::borrow::Cow<'_, str> {
    // Default ON. Explicit "0" / "false" / "off" opts out.
    if let Ok(v) = std::env::var("HIPFIRE_NORMALIZE_PROMPT") {
        let v = v.to_ascii_lowercase();
        if v == "0" || v == "false" || v == "off" || v == "no" {
            return std::borrow::Cow::Borrowed(s);
        }
    }
    if !needs_newline_collapse(s) {
        return std::borrow::Cow::Borrowed(s);
    }
    std::borrow::Cow::Owned(collapse_newline_runs(s))
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
}
