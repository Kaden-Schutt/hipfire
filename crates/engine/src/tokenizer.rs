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

        // Detect tokenizer type
        let model_type = gguf.meta_str("tokenizer.ggml.model").unwrap_or("llama");
        let is_gpt2_bpe = model_type == "gpt2";

        // Build special tokens list: vocab entries matching <|...|> or </...> patterns
        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        for (i, tok) in vocab.iter().enumerate() {
            if (tok.starts_with("<|") && tok.ends_with("|>"))
                || (tok.starts_with("</") && tok.ends_with(">") && tok.len() > 3)
                || (tok.starts_with("<") && tok.ends_with(">") && tok.contains("0x"))
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
        })
    }

    /// Decode a sequence of token IDs to text.
    /// Handles both GPT-2 BPE (Ġ=space, Ċ=newline) and SentencePiece (▁=space).
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        for &id in tokens {
            if let Some(tok) = self.vocab.get(id as usize) {
                if self.is_gpt2_bpe {
                    // GPT-2 BPE byte-level decoding
                    let mut bytes = Vec::new();
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
                    result.push_str(&String::from_utf8_lossy(&bytes));
                } else {
                    // SentencePiece decoding: ▁ = space, tokens are raw text
                    let decoded = tok.replace('▁', " ");
                    // Handle hex escapes like <0x0A> = newline
                    let decoded = decode_hex_escapes(&decoded);
                    result.push_str(&decoded);
                }
            }
        }
        result
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

/// GPT-2 byte-to-char mapping.
/// Bytes 33-126, 161-172, 174-255 map to themselves as Unicode chars.
/// Bytes 0-32, 127-160, 173 map to chars 256-288, 289-322, 323.
fn byte_to_gpt2_char(b: u8) -> char {
    let b = b as u32;
    let c = match b {
        0x21..=0x7E => b,        // '!' to '~'
        0xA1..=0xAC => b,        // '¡' to '¬'
        0xAE..=0xFF => b,        // '®' to 'ÿ'
        _ => b + 256,            // control chars and special
    };
    char::from_u32(c).unwrap_or('?')
}

/// Reverse of byte_to_gpt2_char.
fn gpt2_char_to_byte(c: char) -> Option<u8> {
    let c = c as u32;
    if (0x21..=0x7E).contains(&c)
        || (0xA1..=0xAC).contains(&c)
        || (0xAE..=0xFF).contains(&c)
    {
        Some(c as u8)
    } else if c >= 256 && c < 256 + 256 {
        Some((c - 256) as u8)
    } else {
        None
    }
}

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
