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
    /// True for GPT-2 BPE (Qwen3/3.5): spaces as `Ġ`, bytes via GPT-2 byte-char
    /// encoding, merges run on byte-encoded symbols.
    is_gpt2_bpe: bool,
    /// True for SPM-style BPE (Gemma 4): spaces normalized to `▁` (U+2581),
    /// merges run on raw unicode chars (no byte encoding). Distinct from
    /// plain SentencePiece (`is_gpt2_bpe=false && is_spm_bpe=false`) which
    /// uses greedy longest-match on the full vocab. The tokenizer.json
    /// metadata flags this as `model.type="BPE"` with a `Replace " " → ▁`
    /// normalizer; we detect it by presence of ▁-prefixed common words.
    is_spm_bpe: bool,
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

        // Detect tokenizer type
        let model_type = gguf.meta_str("tokenizer.ggml.model").unwrap_or("llama");
        // Prefer SPM-BPE detection (Gemma 4) when both flavours look plausible —
        // Gemma 4 has "Ġ" in vocab as a byte-fallback slot, which would wrongly
        // trigger GPT-2 detection.
        let is_spm_bpe = token_to_id.contains_key("▁the") || token_to_id.contains_key("▁a");
        let is_gpt2_bpe = !is_spm_bpe && model_type == "gpt2";

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
            is_gpt2_bpe,
            is_spm_bpe,
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

        // SPM-BPE (Gemma 4) uses ▁ for spaces. GPT-2 (Qwen) uses Ġ. Check SPM
        // first — Gemma 4's vocab includes "Ġ" as a byte-fallback slot, which
        // would otherwise mis-trigger GPT-2 detection and break encoding.
        let is_spm_bpe = token_to_id.contains_key("▁the") || token_to_id.contains_key("▁a");
        let is_gpt2_bpe = !is_spm_bpe
            && (token_to_id.contains_key("Ġthe") || token_to_id.contains_key("Ġ"));

        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
            special_tokens,
            bos_id,
            eos_id,
            is_gpt2_bpe,
            is_spm_bpe,
        })
    }

    /// Load tokenizer from HFQ metadata (extracts embedded tokenizer.json).
    pub fn from_hfq_metadata(metadata_json: &str) -> Option<Self> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
        let tok_str = meta.get("tokenizer")?.as_str()?;
        Self::from_hf_json(tok_str)
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
        if self.is_spm_bpe {
            return self.encode_spm_bpe(text);
        }
        if !self.is_gpt2_bpe {
            return self.encode_sentencepiece(text);
        }
        self.encode_gpt2_bpe(text)
    }

    /// SPM-style BPE (Gemma 4).
    ///
    /// Pipeline (matches `tokenizer.json`'s `normalizer` + `model.type="BPE"`
    /// + the HF tokenizers library, and llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GEMMA4`
    /// post-PR #21343):
    ///   1. Normalize: replace every ASCII space with ▁ (U+2581).
    ///   2. Treat the normalized string as a sequence of raw unicode chars
    ///      (no GPT-2 byte-to-char mapping — `byte_encode = false`).
    ///   3. Apply BPE merges iteratively, picking the lowest-rank pair each
    ///      round (same loop as `encode_gpt2_bpe` but on unicode chars).
    ///   4. Map merged symbols to token IDs.
    ///
    /// Why this matters: Gemma 4's vocab includes the GPT-2 byte-fallback
    /// slots (Ġ, Ĉ, etc. as `<0xNN>` stand-ins). The old detector treated
    /// that as GPT-2 BPE and inserted a stray `Ġ` (token 245237) between
    /// every word, which caused the Phase 3 smoke test to produce gibberish
    /// like `"tech hack d facing"`.
    fn encode_spm_bpe(&self, text: &str) -> Vec<u32> {
        let normalized = text.replace(' ', "\u{2581}");
        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();

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
                break;
            }
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        symbols
            .iter()
            .filter_map(|s| self.token_to_id.get(s).copied())
            .collect()
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
