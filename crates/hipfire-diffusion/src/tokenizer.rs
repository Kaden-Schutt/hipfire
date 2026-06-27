// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! CLIP byte-level BPE tokenizer for diffusion text conditioning: the
//! bytes-to-unicode table, merge-rank BPE, and HFQ tokenizer loading.

use super::*;

#[derive(Debug, Clone)]
pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    byte_encoder: Vec<String>,
    start_token: u32,
    end_token: u32,
    pad_token: u32,
    max_length: usize,
    pattern: Regex,
}

impl ClipTokenizer {
    pub fn from_hfq_file(hfq: &HfqFile) -> DiffusionResult<Self> {
        Self::from_hfq_file_with_prefix(hfq, "tokenizer")
    }

    pub fn from_hfq_file_with_prefix(hfq: &HfqFile, prefix: &str) -> DiffusionResult<Self> {
        let vocab_entry = format!("{prefix}/vocab.json");
        let merges_entry = format!("{prefix}/merges.txt");
        let (_, vocab_bytes) = hfq
            .tensor_data_vec(&vocab_entry)
            .ok_or_else(|| DiffusionError::InvalidMetadata(format!("{vocab_entry} is missing")))?;
        let (_, merges_bytes) = hfq
            .tensor_data_vec(&merges_entry)
            .ok_or_else(|| DiffusionError::InvalidMetadata(format!("{merges_entry} is missing")))?;
        Self::from_bytes(&vocab_bytes, &merges_bytes, 77)
    }

    pub fn from_bytes(
        vocab_json: &[u8],
        merges_txt: &[u8],
        max_length: usize,
    ) -> DiffusionResult<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_slice(vocab_json)
            .map_err(|err| DiffusionError::InvalidMetadata(format!("invalid CLIP vocab: {err}")))?;
        let mut merges = HashMap::new();
        let merges_text = String::from_utf8_lossy(merges_txt);
        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(left) = parts.next() else {
                continue;
            };
            let Some(right) = parts.next() else {
                continue;
            };
            let rank = merges.len();
            merges.insert((left.to_string(), right.to_string()), rank);
        }
        let start_token = *vocab.get("<|startoftext|>").ok_or_else(|| {
            DiffusionError::InvalidMetadata("CLIP vocab missing start token".to_string())
        })?;
        let end_token = *vocab.get("<|endoftext|>").ok_or_else(|| {
            DiffusionError::InvalidMetadata("CLIP vocab missing end token".to_string())
        })?;
        Ok(Self {
            vocab,
            merges,
            byte_encoder: clip_byte_encoder(),
            start_token,
            end_token,
            pad_token: end_token,
            max_length,
            pattern: Regex::new(
                r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+",
            )
            .map_err(|err| DiffusionError::InvalidMetadata(format!("invalid CLIP regex: {err}")))?,
        })
    }

    pub fn encode_padded(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(self.max_length);
        tokens.push(self.start_token);
        for piece in self.tokenize(text) {
            if tokens.len() + 1 >= self.max_length {
                break;
            }
            tokens.push(piece);
        }
        tokens.push(self.end_token);
        tokens.resize(self.max_length, self.pad_token);
        tokens
    }

    pub fn end_token_id(&self) -> u32 {
        self.end_token
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let cleaned = whitespace_clean(text).to_lowercase();
        for mat in self.pattern.find_iter(&cleaned) {
            let token = mat.as_str();
            if let Some(&id) = self.vocab.get(token) {
                out.push(id);
                continue;
            }
            let mut encoded = String::new();
            for byte in token.as_bytes() {
                encoded.push_str(&self.byte_encoder[*byte as usize]);
            }
            for bpe_token in self.bpe(&encoded) {
                if let Some(&id) = self.vocab.get(&bpe_token) {
                    out.push(id);
                }
            }
        }
        out
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word = token.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        if word.len() == 1 {
            return word;
        }
        loop {
            let Some((best_idx, _)) = word
                .windows(2)
                .enumerate()
                .filter_map(|(idx, pair)| {
                    self.merges
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (idx, *rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let merged = format!("{}{}", word[best_idx], word[best_idx + 1]);
            word.splice(best_idx..=best_idx + 1, [merged]);
            if word.len() == 1 {
                break;
            }
        }
        word
    }
}

fn whitespace_clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clip_byte_encoder() -> Vec<String> {
    let mut bs = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs = bs.iter().map(|&b| b as u32).collect::<Vec<_>>();
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec![String::new(); 256];
    for (byte, codepoint) in bs.into_iter().zip(cs.into_iter()) {
        out[byte as usize] = char::from_u32(codepoint).unwrap().to_string();
    }
    out
}
