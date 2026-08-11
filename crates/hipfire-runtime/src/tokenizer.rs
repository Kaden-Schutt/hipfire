// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! BPE tokenizer loaded from GGUF metadata.
//! Supports encode (text → token IDs) and decode (token IDs → text).

use crate::gguf::{GgufFile, MetaValue};
use regex::Regex;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::sync::OnceLock;

/// GPT-2 / cl100k-style pre-tokenization regex. Same family of pattern
/// every reference byte-level BPE encoder uses (tiktoken, HF tokenizers,
/// GPT-2): chunk input into contractions, letter words, digit runs ≤3,
/// punctuation runs, and whitespace runs. BPE then runs per-chunk
/// instead of on the whole prompt, restoring the canonical O(N) shape
/// with bounded per-chunk constants.
///
/// Lookahead `\s+(?!\S)` is omitted from the canonical pattern; the
/// `regex` crate doesn't support lookaround and the surviving `\s+`
/// branch matches the same byte spans. Order of alternation preserves
/// the priority the reference encoders use, so chunking boundaries
/// match HF tokenizers' Split-then-ByteLevel pipeline byte-for-byte
/// (verified against locked niah_4k token md5).
const GPT2_PRETOK_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

fn gpt2_pretok_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(GPT2_PRETOK_PATTERN)
            .expect("GPT2_PRETOK_PATTERN must compile — pattern is a const")
    })
}

/// Which side of a merge rule failed validation. Used by `MissingMergeOperand`.
#[derive(Debug, Clone, Copy)]
pub enum Side {
    Left,
    Right,
}

/// Errors returned by `Tokenizer::from_*` constructors.
///
/// Replaces the prior silent `Option::None` returns. Variants distinguish
/// malformed-input failures (`MetadataMissing`, `MalformedJson`) from
/// vocab/merges inconsistencies that would cause the encoder to silently
/// corrupt output downstream (`MissingByteSymbol`, `MissingMergeOperand`,
/// `MissingMergeResult`) — see #203.
#[derive(Debug)]
pub enum TokenizerError {
    /// A required metadata field was missing or had the wrong type in the source format.
    MetadataMissing { field: &'static str },
    /// Raw JSON did not parse.
    MalformedJson(serde_json::Error),
    /// `byte_to_gpt2_char(b)` produced a char with no entry in `token_to_id`.
    /// GPT-2 BPE tokenizers MUST cover every byte 0..=255; without this,
    /// `encode_gpt2_bpe`'s initial seed would silently map to id 0.
    MissingByteSymbol { byte: u8, char: char },
    /// A merge rule referenced a left or right symbol with no entry in `token_to_id`.
    MissingMergeOperand {
        rank: usize,
        left: String,
        right: String,
        missing_side: Side,
    },
    /// A merge rule's resolved result (`left + right`) has no entry in `token_to_id`.
    /// The encoder cannot represent the post-merge state without this.
    MissingMergeResult { rank: usize, expected: String },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetadataMissing { field } => {
                write!(f, "tokenizer metadata field missing or wrong type: {field}")
            }
            Self::MalformedJson(e) => write!(f, "tokenizer JSON parse error: {e}"),
            Self::MissingByteSymbol { byte, char } => write!(
                f,
                "GPT-2 BPE vocab missing byte symbol: byte 0x{byte:02x} maps to char {char:?} which is not in token_to_id"
            ),
            Self::MissingMergeOperand { rank, left, right, missing_side } => write!(
                f,
                "merge rule rank {rank} ({left:?}, {right:?}): {missing_side:?} symbol not in vocab"
            ),
            Self::MissingMergeResult { rank, expected } => write!(
                f,
                "merge rule rank {rank}: merged result {expected:?} not in vocab"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MalformedJson(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        Self::MalformedJson(e)
    }
}

pub struct Tokenizer {
    /// Token ID → string
    vocab: Vec<String>,
    /// String → token ID (for encoding)
    token_to_id: HashMap<String, u32>,
    /// BPE merge results in rank order: `merges[i]` is the merged token id
    /// for the merge rule at rank `i`. The pair → rank lookup is in
    /// `merge_pair_rank` (keyed on `(left_id, right_id)`); when a rule
    /// fires, the encoder writes `merges[rank]` directly. Empty for SP.
    ///
    /// Replaces the prior `Vec<(String, String)>` (and the intermediate
    /// `Vec<MergeRule { left, right, result }>` in earlier drafts of this
    /// refactor — `left` and `right` were never read on any path because
    /// pair lookups already go through `merge_pair_rank`, so storing only
    /// the result is sufficient and ~3× smaller per entry).
    merges: Vec<u32>,
    /// `(left_id, right_id) → rank` (= index into `merges`). Built once at
    /// construction. `(u32, u32)` is `Copy` so heap-pair lookups never clone.
    ///
    /// Named `merge_pair_rank` (not `merge_rank`) to disambiguate from the
    /// public `merge_rank(&self, id: u32) -> Option<usize>` diagnostic
    /// method (token-id → rank, O(merges) linear scan — different lookup,
    /// different shape). They compiled fine sharing the name but invited
    /// subtle field-vs-method confusion at call sites.
    merge_pair_rank: HashMap<(u32, u32), u32>,
    /// For GPT-2 BPE only: byte `b` → token id of `byte_to_gpt2_char(b).to_string()`.
    /// Construction guarantees every byte 0..=255 has a valid id (else
    /// `from_*` returns `MissingByteSymbol`), so `encode_gpt2_bpe`'s initial
    /// seed is infallible. `None` for SentencePiece tokenizers (no
    /// byte-level encoding).
    byte_to_id: Option<[u32; 256]>,
    /// Special tokens: strings like "<|im_start|>" → their token ID.
    /// Sorted longest-first for greedy matching.
    special_tokens: Vec<(String, u32)>,
    pub bos_id: u32,
    /// Prepend `bos_id` in `encode()` when the model's HF tokenizer_config has
    /// `add_bos_token: true` (e.g. Cohere2: true, BOS=`<BOS_TOKEN>`=2). Without
    /// the leading BOS, BOS-trained models (Cohere, Llama) emit degenerate
    /// output. Default false (Qwen/ChatML models add no BOS).
    pub add_bos: bool,
    pub eos_id: u32,
    /// Auxiliary end-of-generation id (e.g. `<|endoftext|>` when `eos_id` is
    /// `<|im_end|>`). When a raw-text draft without ChatML finishes naturally
    /// it emits this, not `eos_id` — stop-loops must check both via
    /// `is_terminator()`. None if the vocab only has one terminator.
    pub eot_id: Option<u32>,
    /// True for GPT-2 BPE (Qwen), false for SentencePiece (LLaMA).
    is_gpt2_bpe: bool,
    /// SentencePiece `add_dummy_prefix`: when true, `encode_sentencepiece`
    /// prepends a `▁` to each raw text segment (the LLaMA convention —
    /// `normalizer: [Prepend("▁"), Replace(" "→"▁")]`). CONFIG-DRIVEN, not
    /// hardcoded: Gemma 4's tokenizer.json normalizer is a bare
    /// `Replace(" "→"▁")` with NO Prepend, so "The capital of France is"
    /// must encode to [818 ("The"), …] — the unconditional prefix shifted
    /// the first word to 669 ("▁The"), a one-token divergence from the HF
    /// ground truth that made short raw prompts OOD (gemma4 12B garbage
    /// replies, 2026-06-09). Resolved per-source:
    ///   - HF tokenizer.json → `sp_dummy_prefix_from_hf_json` (normalizer
    ///     Prepend / Metaspace pre_tokenizer signals);
    ///   - GGUF → `tokenizer.ggml.add_space_prefix` (default true, the
    ///     llama.cpp SPM convention).
    /// Unused on the GPT-2 BPE path.
    sp_dummy_prefix: bool,
}

/// Resolve a list of `(left_string, right_string)` merge pairs into a
/// rank-ordered `Vec<u32>` of result ids + `(left_id, right_id) → rank`
/// lookup, verifying that every operand and every merged result is
/// present in `token_to_id`. Returns `Err` on the first missing entry —
/// see #203.
fn resolve_merges(
    merges_strings: &[(String, String)],
    token_to_id: &HashMap<String, u32>,
) -> Result<(Vec<u32>, HashMap<(u32, u32), u32>), TokenizerError> {
    let mut merges = Vec::with_capacity(merges_strings.len());
    let mut merge_pair_rank: HashMap<(u32, u32), u32> =
        HashMap::with_capacity(merges_strings.len());
    let mut result_buf = String::new();
    for (rank, (l_str, r_str)) in merges_strings.iter().enumerate() {
        let left = token_to_id.get(l_str.as_str()).copied().ok_or_else(|| {
            TokenizerError::MissingMergeOperand {
                rank,
                left: l_str.clone(),
                right: r_str.clone(),
                missing_side: Side::Left,
            }
        })?;
        let right = token_to_id.get(r_str.as_str()).copied().ok_or_else(|| {
            TokenizerError::MissingMergeOperand {
                rank,
                left: l_str.clone(),
                right: r_str.clone(),
                missing_side: Side::Right,
            }
        })?;
        // Merged result string is `left + right` by BPE construction. Reuse
        // one scratch buffer across all rules.
        result_buf.clear();
        result_buf.push_str(l_str);
        result_buf.push_str(r_str);
        let result = token_to_id
            .get(result_buf.as_str())
            .copied()
            .ok_or_else(|| TokenizerError::MissingMergeResult {
                rank,
                expected: result_buf.clone(),
            })?;
        merges.push(result);
        // First-rank-wins on duplicate `(left_id, right_id)` pairs:
        // `entry().or_insert()` keeps the earlier rank. This is correct for
        // BPE semantics because the encoder's min-heap pops lowest rank
        // first — a later duplicate would never fire even if we inserted
        // it. Pathological vocabs with duplicate merge rules thus behave
        // deterministically (drop the later rule) instead of being
        // overwritten silently.
        merge_pair_rank.entry((left, right)).or_insert(rank as u32);
    }
    Ok((merges, merge_pair_rank))
}

/// For GPT-2 BPE tokenizers: build the byte → token-id lookup table by
/// running every byte 0..=255 through `byte_to_gpt2_char` and resolving
/// the resulting char string against `token_to_id`. Returns `Err` on the
/// first byte whose char isn't in the vocab — that would silently corrupt
/// `encode_gpt2_bpe`'s initial seed (#203).
fn build_byte_to_id(token_to_id: &HashMap<String, u32>) -> Result<[u32; 256], TokenizerError> {
    let mut out = [0u32; 256];
    let mut buf = [0u8; 4];
    for b in 0u32..=255 {
        let ch = byte_to_gpt2_char(b as u8);
        let s = ch.encode_utf8(&mut buf);
        let id = token_to_id
            .get(s)
            .copied()
            .ok_or(TokenizerError::MissingByteSymbol {
                byte: b as u8,
                char: ch,
            })?;
        out[b as usize] = id;
    }
    Ok(out)
}

/// Decide whether the SentencePiece encode path should prepend a dummy `▁`
/// by reading the HF tokenizer.json normalizer / pre_tokenizer config
/// instead of assuming the LLaMA convention.
///
/// Signals that mean "yes, prepend":
///   - a `Prepend` normalizer, top-level or inside a `Sequence` — the
///     LLaMA-2 style `[Prepend("▁"), Replace(" "→"▁")]` chain;
///   - a `Metaspace` pre_tokenizer (top-level or inside a `Sequence`)
///     whose `prepend_scheme` is `"always"`/`"first"`, or — legacy
///     serialization — whose `add_prefix_space` is true (the T5-style
///     `Precompiled` charsmap + `Metaspace` pairing; the charsmap itself
///     never carries the dummy-prefix flag, Metaspace does).
///
/// No signal (e.g. Gemma 4's bare `Replace(" "→"▁")` normalizer, null
/// pre_tokenizer) means the model's tokenizer does NOT want a dummy
/// prefix — see the `sp_dummy_prefix` field doc for the gemma4 failure
/// mode that motivated this.
fn sp_dummy_prefix_from_hf_json(tok: &serde_json::Value) -> bool {
    fn normalizer_prepends(n: &serde_json::Value) -> bool {
        match n.get("type").and_then(|v| v.as_str()) {
            Some("Prepend") => true,
            Some("Sequence") => n
                .get("normalizers")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().any(normalizer_prepends))
                .unwrap_or(false),
            _ => false,
        }
    }
    fn pretokenizer_prepends(p: &serde_json::Value) -> bool {
        match p.get("type").and_then(|v| v.as_str()) {
            Some("Metaspace") => match p.get("prepend_scheme").and_then(|v| v.as_str()) {
                Some("always") | Some("first") => true,
                Some(_) => false, // "never"
                // Legacy Metaspace serialization (tokenizers < 0.14):
                // `add_prefix_space: bool`, default true.
                None => p
                    .get("add_prefix_space")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
            },
            Some("Sequence") => p
                .get("pretokenizers")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().any(pretokenizer_prepends))
                .unwrap_or(false),
            _ => false,
        }
    }
    tok.get("normalizer").map(normalizer_prepends).unwrap_or(false)
        || tok
            .get("pre_tokenizer")
            .map(pretokenizer_prepends)
            .unwrap_or(false)
}


impl Tokenizer {
    /// Load tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        // Read vocabulary
        let tokens_meta =
            gguf.meta("tokenizer.ggml.tokens")
                .ok_or(TokenizerError::MetadataMissing {
                    field: "tokenizer.ggml.tokens",
                })?;
        let vocab: Vec<String> = match tokens_meta {
            MetaValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    MetaValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => {
                return Err(TokenizerError::MetadataMissing {
                    field: "tokenizer.ggml.tokens",
                })
            }
        };

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Read merge rules (raw string pairs; resolved to ids below).
        let merges_strings: Vec<(String, String)> =
            if let Some(MetaValue::Array(arr)) = gguf.meta("tokenizer.ggml.merges") {
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
        let im_end = token_to_id.get("<|im_end|>").copied();
        let eot_id = match (endoftext, im_end) {
            (Some(et), Some(ie)) if et != eos_id && ie == eos_id => Some(et),
            (Some(et), _) if et != eos_id => Some(et),
            _ => None,
        };

        // Detect tokenizer type
        let model_type = gguf.meta_str("tokenizer.ggml.model").unwrap_or("llama");
        let is_gpt2_bpe = model_type == "gpt2";
        // SP dummy prefix: GGUF carries it as `tokenizer.ggml.add_space_prefix`
        // (llama.cpp SPM convention, default true when absent).
        let sp_dummy_prefix = match gguf.meta("tokenizer.ggml.add_space_prefix") {
            Some(MetaValue::Bool(b)) => *b,
            _ => true,
        };

        // Build special tokens list: vocab entries matching <|...|> or </...> patterns
        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        for (i, tok) in vocab.iter().enumerate() {
            if (tok.starts_with("<|") && tok.ends_with("|>"))
                || (tok.starts_with("<")
                    && tok.ends_with(">")
                    && tok.len() > 3
                    && !tok.contains(' '))
            {
                special_tokens.push((tok.clone(), i as u32));
            }
        }
        // Sort longest-first for greedy matching
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Resolve merges to token ids; reject inconsistent vocab/merges (#203).
        let (merges, merge_pair_rank) = resolve_merges(&merges_strings, &token_to_id)?;
        // For GPT-2 BPE, every byte 0..=255 must have a vocab entry. SP doesn't need this.
        let byte_to_id = if is_gpt2_bpe {
            Some(build_byte_to_id(&token_to_id)?)
        } else {
            None
        };

        Ok(Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_pair_rank,
            byte_to_id,
            special_tokens,
            bos_id,
            eos_id,
            add_bos: false,
            eot_id,
            is_gpt2_bpe,
            sp_dummy_prefix,
        })
    }

    /// Load tokenizer from HuggingFace tokenizer.json (embedded in HFQ metadata).
    pub fn from_hf_json(json_str: &str) -> Result<Self, TokenizerError> {
        let tok: serde_json::Value = serde_json::from_str(json_str)?;
        let model = tok
            .get("model")
            .ok_or(TokenizerError::MetadataMissing { field: "model" })?;

        let vocab_map = model.get("vocab").and_then(|v| v.as_object()).ok_or(
            TokenizerError::MetadataMissing {
                field: "model.vocab",
            },
        )?;
        let vocab_size = vocab_map.len();

        let mut vocab = vec![String::new(); vocab_size + 100];
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        for (token, id_val) in vocab_map {
            let id = id_val.as_u64().ok_or(TokenizerError::MetadataMissing {
                field: "model.vocab[*]: non-integer id",
            })? as u32;
            if (id as usize) >= vocab.len() {
                vocab.resize(id as usize + 1, String::new());
            }
            vocab[id as usize] = token.clone();
            token_to_id.insert(token.clone(), id);
        }

        let merges_strings: Vec<(String, String)> =
            if let Some(merges_arr) = model.get("merges").and_then(|v| v.as_array()) {
                merges_arr
                    .iter()
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
                    // Every `added_tokens` entry is an atomic token in the
                    // tokenizer's intended encoding — the `special` field
                    // marks SEMANTIC specials (BOS/EOS that need engine-
                    // side handling) but does NOT mean "merge me back into
                    // BPE." The DeepSeek V4 tokenizer marks all of
                    // `<｜User｜>`, `<｜Assistant｜>`, `｜DSML｜`, the
                    // `<｜tool▁*｜>` family, etc. as `special=false` even
                    // though they MUST encode as single tokens for the
                    // model to recognize them. Without this fix
                    // `<｜DSML｜tool_calls>` BPE-fragments into 9 pieces
                    // and tool calls round-trip into garbled output.
                    let _ = at.get("special"); // we no longer gate on this
                    special_tokens.push((content.to_string(), id));
                }
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let bos_id = token_to_id
            .get("<|endoftext|>")
            .copied()
            .or_else(|| token_to_id.get("<s>").copied())
            .unwrap_or(1);
        let eos_id = token_to_id
            .get("<|im_end|>")
            .copied()
            .or_else(|| token_to_id.get("<|endoftext|>").copied())
            .or_else(|| token_to_id.get("</s>").copied())
            .unwrap_or(2);
        let endoftext = token_to_id.get("<|endoftext|>").copied();
        let eot_id = match endoftext {
            Some(et) if et != eos_id => Some(et),
            _ => None,
        };

        // GPT-2 byte-level BPE uses Ġ-prefixed space tokens (Qwen); SentencePiece
        // BPE uses ▁ (Gemma, LLaMA). A single stray "Ġ" token is NOT enough to
        // call it GPT-2 — decide by which space-marker DOMINATES (Gemma has 1 Ġ
        // token but ~137k ▁ tokens, and must take the SP path or byte-coverage fails).
        let n_gpt2 = token_to_id.keys().filter(|t| t.starts_with('Ġ')).count();
        let n_sp = token_to_id.keys().filter(|t| t.starts_with('▁')).count();
        let is_gpt2_bpe = token_to_id.contains_key("Ġthe") || n_gpt2 > n_sp;
        // SP dummy prefix is a property of the model's normalizer /
        // pre_tokenizer config, NOT of SentencePiece itself — read it from
        // the tokenizer.json instead of assuming the LLaMA convention.
        let sp_dummy_prefix = sp_dummy_prefix_from_hf_json(&tok);

        let (merges, merge_pair_rank) = resolve_merges(&merges_strings, &token_to_id)?;
        let byte_to_id = if is_gpt2_bpe {
            Some(build_byte_to_id(&token_to_id)?)
        } else {
            None
        };

        Ok(Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_pair_rank,
            byte_to_id,
            special_tokens,
            bos_id,
            eos_id,
            add_bos: false,
            eot_id,
            is_gpt2_bpe,
            sp_dummy_prefix,
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
    /// Returns `Err(MetadataMissing)` if neither is present.
    ///
    /// Load tokenizer from a tokenizer.json file on disk.
    /// Returns `Ok(None)` if the file is missing or unreadable; `Err` if the
    /// file exists but fails to parse as a valid tokenizer.
    pub fn from_tokenizer_json(path: &std::path::Path) -> Result<Option<Self>, TokenizerError> {
        let json_str = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        Self::from_hf_json(&json_str).map(Some)
    }

    pub fn from_hfq_metadata(metadata_json: &str) -> Result<Self, TokenizerError> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json)?;
        if let Some(tok_str) = meta.get("tokenizer").and_then(|v| v.as_str()) {
            let mut t = Self::from_hf_json(tok_str)?;
            // `from_hf_json` infers bos/eos from vocab heuristics (it defaults bos
            // to `<|endoftext|>`), which is WRONG for models whose BOS differs —
            // e.g. LFM2.5 bos = `<|startoftext|>` (id 1), not `<|endoftext|>`
            // (id 2). `generation_config.{bos,eos}_token_id` is the authoritative
            // HF source; override from it when present + scalar (an array eos =
            // multi-eos: keep the heuristic). Embedded by the quantizer at convert
            // time. (LFM2.5-350M daemon bring-up, 2026-06-07.)
            if let Some(gc) = meta.get("generation_config") {
                if let Some(b) = gc.get("bos_token_id").and_then(|v| v.as_u64()) {
                    if (b as usize) < t.vocab.len() {
                        t.bos_id = b as u32;
                    }
                }
                if let Some(e) = gc.get("eos_token_id").and_then(|v| v.as_u64()) {
                    if (e as usize) < t.vocab.len() {
                        t.eos_id = e as u32;
                    }
                }
            }
            // Honor HF `add_bos_token` from the embedded tokenizer_config (e.g.
            // Cohere2 = true). Drives the BOS prepend in `encode()`. An explicit
            // true or false is authoritative; only fall through to arch defaults
            // when the key is absent.
            let add_bos_explicit = meta
                .get("tokenizer_config")
                .and_then(|tc| tc.get("add_bos_token"))
                .and_then(|v| v.as_bool());
            if let Some(ab) = add_bos_explicit {
                t.add_bos = ab;
            } else if let Some(arch) = meta.get("architecture").and_then(|v| v.as_str()) {
                // LFM2.5 family (arch "lfm2" / "lfm2_moe") is BOS-trained: the
                // chat_template starts with `{{- bos_token -}}` and HF's
                // AutoTokenizer adds BOS for raw prompts (verified: 350M
                // "The capital of France is" -> [1, 1098, ...] via HF, but
                // hipfire's add_bos is false when tokenizer_config lacks
                // `add_bos_token`, so raw prompts miss BOS and the model
                // sees position-shifted inputs -> single-token attractor
                // (230M: 856 " is" x8, 350M: 540 x8). Default add_bos true
                // for LFM only when the flag is absent; encode() still
                // suppresses a double-BOS when text already starts with the
                // BOS literal (Jinja-rendered chat path).
                if arch == "lfm2" || arch == "lfm2_moe" {
                    t.add_bos = true;
                }
            }
            return Ok(t);
        }
        if let Some(gguf_meta) = meta.get("gguf_meta") {
            return Self::from_gguf_meta_json(gguf_meta);
        }
        Err(TokenizerError::MetadataMissing {
            field: "tokenizer | gguf_meta",
        })
    }

    /// Load tokenizer from a JSON-serialized GGUF metadata tree. Mirrors
    /// `from_gguf` field-for-field but reads `serde_json::Value` instead of
    /// the live `GgufFile`. Used by the GGUF→MQ4 quantize path so a
    /// converted `.mq4` is fully self-sufficient (no GGUF-on-disk fallback).
    pub fn from_gguf_meta_json(meta: &serde_json::Value) -> Result<Self, TokenizerError> {
        let tokens_arr = meta
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or(TokenizerError::MetadataMissing {
                field: "tokenizer.ggml.tokens",
            })?;
        let vocab: Vec<String> = tokens_arr
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        let merges_strings: Vec<(String, String)> = meta
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
        // Mirrors `from_gguf`: llama.cpp SPM convention defaults the dummy
        // prefix ON when `tokenizer.ggml.add_space_prefix` is absent.
        let sp_dummy_prefix = meta
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        for (i, tok) in vocab.iter().enumerate() {
            if (tok.starts_with("<|") && tok.ends_with("|>"))
                || (tok.starts_with("<")
                    && tok.ends_with(">")
                    && tok.len() > 3
                    && !tok.contains(' '))
            {
                special_tokens.push((tok.clone(), i as u32));
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let (merges, merge_pair_rank) = resolve_merges(&merges_strings, &token_to_id)?;
        let byte_to_id = if is_gpt2_bpe {
            Some(build_byte_to_id(&token_to_id)?)
        } else {
            None
        };

        Ok(Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_pair_rank,
            byte_to_id,
            special_tokens,
            bos_id,
            eos_id,
            add_bos: false,
            eot_id,
            is_gpt2_bpe,
            sp_dummy_prefix,
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
        self.special_tokens
            .iter()
            .find(|(s, _)| s == content)
            .map(|(_, id)| *id)
    }

    /// Decode a sequence of token IDs to text.
    /// Handles both GPT-2 BPE (Ġ=space, Ċ=newline) and SentencePiece (▁=space).
    /// For GPT-2 BPE: collects all bytes first, then does UTF-8 conversion once
    /// (individual tokens can be incomplete UTF-8 sequences in byte-level BPE).
    pub fn decode(&self, tokens: &[u32]) -> String {
        // hunt3 H-C: delegate to the byte-correct decode_bytes path then
        // from_utf8_lossy once, so multi-byte chars split across <0xHH>
        // fallback tokens (e.g. CJK 中 = E4 B8 AD) reassemble instead of
        // mojibake-ing via `byte as char`. Covers both BPE and SentencePiece.
        String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
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
                    // hunt3 H-C: byte-correct path — <0xHH> fallback tokens must
                    // emit raw bytes, not `byte as char` re-UTF8-encoded codepoints.
                    let decoded = tok.replace('▁', " ");
                    bytes.extend_from_slice(&decode_hex_escapes_bytes(&decoded));
                }
            }
        }
        bytes
    }

    /// Encode text to token IDs.
    /// Special tokens (e.g. <|im_start|>) are matched first, then remaining
    /// segments are encoded via BPE or SentencePiece.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        // Honor HF `add_bos_token` (Cohere2/LFM2): prepend bos_id once for
        // raw prompts. Suppress a duplicate when `text` already begins with
        // the BOS vocabulary literal (Jinja chat templates emit
        // `{{- bos_token -}}` up front).
        if self.add_bos {
            let bos_str = self
                .vocab
                .get(self.bos_id as usize)
                .map(|s| s.as_str())
                .unwrap_or("");
            if bos_str.is_empty() || !text.starts_with(bos_str) {
                result.push(self.bos_id);
            }
        }
        if self.special_tokens.is_empty() {
            result.extend(self.encode_raw(text));
            return result;
        }

        // Split text at special token boundaries (greedy longest match)
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

    /// SentencePiece greedy encoding: spaces become ▁, longest-match lookup.
    /// A leading dummy `▁` is prepended only when the tokenizer config asks
    /// for it (`sp_dummy_prefix` — see the field doc; gemma4 says no,
    /// LLaMA says yes).
    ///
    /// Each suffix trial is a `&str` slice of `sp_text` keyed against
    /// `token_to_id` (HashMap's `Borrow<str>` impl matches `String` keys
    /// against `&str` lookups by hash). The previous impl built a
    /// `Vec<char>` and reallocated a `String` per trial — same iteration
    /// shape, but allocator-heavy on long prompts.
    ///
    /// Algorithmic shape (unchanged from the pre-rewrite code): for each
    /// position `pos` we scan `end` from `n_chars` down to `pos+1`,
    /// stopping at the first vocab match (greedy longest). The loop has
    /// no `max_token_len` cap — typical inputs `break` early once a
    /// match is found, but worst-case unmatchable suffixes are O(N²)
    /// HashMap lookups. Capping by a precomputed `max_token_chars`
    /// would fix this and is tracked as a follow-up; this PR is scoped
    /// to allocator pressure only.
    ///
    /// Note: the single-char fallback silently *drops* missing chars
    /// (cf. #203) — different failure mode from `encode_gpt2_bpe`'s
    /// `unwrap_or(0)`. Fix is a separate concern from this PR.
    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        // Spaces become ▁; start of text gets ▁ ONLY if the model's
        // tokenizer config has the dummy prefix (sp_dummy_prefix).
        // Single-pass build: the prior `text.replace(...)` + `format!(...)`
        // allocated twice; iterating chars and pushing into a pre-sized
        // String allocates once. ▁ is 3 bytes UTF-8, so the worst case
        // (all-space input) needs `text.len() * 3 + 3` bytes; typical
        // inputs fit in the lower-bound hint and the String grows only
        // if needed.
        let mut sp_text = String::with_capacity(text.len() + 3);
        if self.sp_dummy_prefix {
            sp_text.push('\u{2581}');
        }
        for ch in text.chars() {
            sp_text.push(if ch == ' ' { '\u{2581}' } else { ch });
        }

        // Char-boundary byte offsets. `boundaries[i]` is the byte index of
        // char `i`; the trailing entry is `sp_text.len()` so `boundaries[end]`
        // is always a valid slice endpoint, including `end == n_chars`.
        let mut boundaries: Vec<usize> = Vec::with_capacity(sp_text.len() + 1);
        for (i, _) in sp_text.char_indices() {
            boundaries.push(i);
        }
        boundaries.push(sp_text.len());
        let n_chars = boundaries.len() - 1;

        let mut pos = 0usize;
        // Cap the greedy scan at the longest vocab token (in chars). The
        // uncapped (pos+1..=n_chars).rev() scan is O(text^2) hash work over
        // megabyte-long candidate slices on corpus-sized inputs (the wt2
        // KLD slice never finished tokenizing). No vocab entry longer than
        // max_tok_chars can ever match, so capping is byte-identical.
        let max_tok_chars = self
            .token_to_id
            .keys()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1);
        while pos < n_chars {
            // Greedy longest match from vocabulary (high-`end` first).
            let mut best_len = 0;
            let mut best_id = 0u32;
            let scan_end = (pos + max_tok_chars).min(n_chars);
            for end in (pos + 1..=scan_end).rev() {
                let candidate = &sp_text[boundaries[pos]..boundaries[end]];
                if let Some(&id) = self.token_to_id.get(candidate) {
                    best_len = end - pos;
                    best_id = id;
                    break;
                }
            }

            if best_len == 0 {
                // Single-character fallback — look up the byte slice for
                // the one char at `pos`. Silently skips unknown chars.
                let ch_slice = &sp_text[boundaries[pos]..boundaries[pos + 1]];
                if let Some(&id) = self.token_to_id.get(ch_slice) {
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
        // Pre-tokenize via GPT-2 / cl100k regex, then BPE each chunk
        // independently. Reference behaviour for byte-level BPE; without
        // this step the merge PQ degenerates to a single O(N log N)
        // problem over the whole prompt (~150K cross-word merges with
        // stale-entry churn on a 14K-byte prompt). Per-chunk problems
        // are typically ≤10 bytes with <8 merges each, so total work
        // becomes O(N) with tiny constants — matches HF tokenizers /
        // tiktoken / GPT-2 reference encoders byte-for-byte.
        //
        // Capacity hint: typical BPE compression on prose is ~3.5× bytes
        // → tokens, so `len/4` is a sane lower bound that avoids early
        // reallocs without wasting memory on short inputs.
        let mut result: Vec<u32> = Vec::with_capacity(text.len() / 4 + 1);
        for m in gpt2_pretok_re().find_iter(text) {
            self.encode_gpt2_chunk(m.as_str().as_bytes(), &mut result);
        }
        result
    }

    /// BPE-encode a single pre-tokenized chunk (byte slice from
    /// `gpt2_pretok_re().find_iter`) and append the resulting token ids
    /// to `out`. Splitting this out of `encode_gpt2_bpe` lets the regex
    /// driver feed many small chunks through the same PQ machinery
    /// without re-allocating the heap/linked-list state across the full
    /// prompt — each chunk is typically ≤10 bytes so the per-chunk state
    /// is tiny.
    fn encode_gpt2_chunk(&self, chunk_bytes: &[u8], out: &mut Vec<u32>) {
        // 1. Convert chunk bytes to GPT-2 byte-encoded symbol IDs.
        // Construction guarantees `byte_to_id` covers every byte 0..=255
        // for GPT-2 BPE tokenizers (else `from_*` returned
        // `MissingByteSymbol`), so the table lookup is infallible.
        let byte_to_id = self
            .byte_to_id
            .as_ref()
            .expect("encode_gpt2_chunk called on non-GPT2 tokenizer");
        let mut syms: Vec<u32> = chunk_bytes
            .iter()
            .map(|&b| byte_to_id[b as usize])
            .collect();
        let n = syms.len();
        if n == 0 {
            return;
        }
        if n == 1 {
            out.push(syms[0]);
            return;
        }

        // 2. Pre-built merge rank map keyed on (left_id, right_id). u32
        // pairs are `Copy` so heap-pair lookups never clone. Cached on the
        // Tokenizer so we don't pay the O(M) build cost per encode call
        // (was ~50ms per encode for Qwen3+ with ~150K merges; the daemon
        // makes 9+ encode calls per request for chat-template scaffolding,
        // so the per-call rebuild added ~450ms to TTFT for short prompts).
        let merge_pair_rank = &self.merge_pair_rank;
        let merges = &self.merges;

        // 3. Doubly-linked-list state. `prev/next` are i32 with -1 sentinel.
        let mut prev: Vec<i32> = (0..n as i32).map(|i| i - 1).collect();
        let mut next: Vec<i32> = (0..n as i32)
            .map(|i| if i + 1 < n as i32 { i + 1 } else { -1 })
            .collect();
        let mut dead: Vec<bool> = vec![false; n];
        let mut gen: Vec<u32> = vec![0; n];

        // 4. Min-heap of (rank, left_idx, gen_at_push). Reverse for min-heap.
        let mut heap: BinaryHeap<Reverse<(u32, usize, u32)>> = BinaryHeap::with_capacity(n);
        let push_pair = |heap: &mut BinaryHeap<Reverse<(u32, usize, u32)>>,
                         syms: &[u32],
                         gen: &[u32],
                         l: usize,
                         r: usize| {
            // O(1) HashMap lookup on a Copy `(u32, u32)` key — no clones.
            if let Some(&rank) = merge_pair_rank.get(&(syms[l], syms[r])) {
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
                merge_pair_rank.get(&(syms[l], syms[r])).copied(),
                Some(rank),
                "BPE pq invariant: popped rank must match live pair rank",
            );

            // Apply the merge: l absorbs r. The merged token id is a direct
            // index into `merges` (rank-ordered) — no string concat, no
            // HashMap lookup, infallible by construction (#203).
            syms[l] = merges[rank as usize];
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

        // 7. Walk the linked list, appending live symbol ids to `out`.
        // Every slot holds a token id that was validated at construction
        // time: `byte_to_id` rejected any byte whose char wasn't in vocab,
        // `resolve_merges` rejected any merge whose result wasn't in
        // vocab, and the merge loop only writes `merges[rank]` (which is
        // itself a validated id). The encoder therefore CANNOT emit id 0
        // for an OOV symbol — the previous `unwrap_or(0)` fallback (#203)
        // is structurally unreachable now and removed. The explicit head
        // scan (`prev == -1 && !dead`) survives any future invariant
        // breakage that a `debug_assert!` would silently miss in release.
        let head = (0..n).find(|&i| prev[i] == -1 && !dead[i]);
        let mut p: i32 = head.map(|i| i as i32).unwrap_or(-1);
        while p >= 0 {
            let pi = p as usize;
            out.push(syms[pi]);
            p = next[pi];
        }
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
    if (0x21..=0x7E).contains(&c) || (0xA1..=0xAC).contains(&c) || (0xAE..=0xFF).contains(&c) {
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
        let is_printable =
            (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
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
        let is_printable =
            (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if !is_printable {
            table[n] = b as u8;
            n += 1;
        }
        b += 1;
    }
    table
};

/// Decode SentencePiece hex escapes like `<0x0A>` to actual bytes.
///
/// Byte-fallback tokens like `<0xE4>` are emitted as the RAW byte 0xE4, NOT as
/// the Unicode codepoint U+00E4 (which would re-UTF8-encode to two bytes and
/// mangle CJK / any non-ASCII byte-fallback sequence). Ordinary chars are
/// encoded as UTF-8. Returns bytes so a multi-byte char split across several
/// `<0xHH>` tokens (e.g. 中 = E4 B8 AD) reassembles via a single
/// `from_utf8`/`from_utf8_lossy` at the call site.
// hunt3 H-C: prior `decode_hex_escapes` did `result.push(byte as char)` which
// corrupted byte-fallback tokens (0xHH → U+00HH → 2-byte UTF-8); this variant
// pushes the parsed value as a raw byte instead.
fn decode_hex_escapes_bytes(s: &str) -> Vec<u8> {
    let mut result: Vec<u8> = Vec::with_capacity(s.len());
    let mut buf = [0u8; 4];
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
                            // hunt3 H-C: raw byte, NOT `byte as char`
                            result.push(byte);
                            matched = true;
                        }
                    }
                }
            }
            if !matched {
                for ch in temp {
                    result.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        } else {
            result.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
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
        // `merges: Vec<u32>` is rank-ordered result ids, so the merged-id
        // is just `*m` — no string concatenation, no `token_to_id` lookup,
        // infallible by construction.
        let mut out = HashMap::with_capacity(self.merges.len());
        for (i, &result_id) in self.merges.iter().enumerate() {
            out.entry(result_id).or_insert(i);
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
        // Linear scan of the rank-ordered `merges` Vec for the result id.
        self.merges.iter().position(|&m| m == id)
    }

    fn rank_of(&self, id: u32, table: &HashMap<u32, usize>) -> Option<usize> {
        table.get(&id).copied().or_else(|| {
            let s = self.vocab.get(id as usize)?;
            if s.len() <= 1 {
                Some(0)
            } else {
                None
            }
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
        if crate::config::get().prompt_heat_json {
            let mut s = String::with_capacity(2048);
            s.push_str("{\"bytes\":");
            s.push_str(&text.len().to_string());
            s.push_str(",\"tokens\":");
            s.push_str(&ids.len().to_string());
            s.push_str(",\"summary\":{");
            s.push_str(&format!(
                "\"base\":{},\"hot\":{},\"warm\":{},\"cold\":{},\"frozen\":{},\"special\":{}",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
            ));
            s.push_str("},\"positions\":[");
            for (pos, &id) in ids.iter().enumerate() {
                if pos > 0 {
                    s.push(',');
                }
                let rank = self.rank_of(id, &table);
                let decoded = self
                    .decode(&[id])
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\t', "\\t")
                    .replace('\r', "\\r");
                s.push_str(&format!(
                    "{{\"pos\":{pos},\"id\":{id},\"rank\":{},\"text\":\"{decoded}\"}}",
                    rank.map(|r| r.to_string())
                        .unwrap_or_else(|| "null".to_string())
                ));
            }
            s.push_str("]}");
            println!("{s}");
            return;
        }
        let limit: usize = crate::config::get().prompt_heat_limit;
        eprintln!(
            "[token-heat] prompt={} bytes  tokens={}",
            text.len(),
            ids.len()
        );
        eprintln!(
            "[token-heat] {:>4}  {:>6}  {:>7}  {:7}  {}",
            "pos", "id", "rank", "class", "decoded"
        );
        for (pos, &id) in ids.iter().take(limit).enumerate() {
            let rank = self.rank_of(id, &table);
            let class = HeatClass::from_rank(rank);
            let display = self.decode(&[id]).replace('\n', "\\n").replace('\t', "\\t");
            let rank_str = rank
                .map(|r| r.to_string())
                .unwrap_or_else(|| "-".to_string());
            eprintln!(
                "[token-heat] {pos:>4}  {id:>6}  {rank_str:>7}  {}  {display:?}",
                class.label()
            );
        }
        if ids.len() > limit {
            eprintln!(
                "[token-heat] ... ({} more tokens omitted)",
                ids.len() - limit
            );
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
            eprintln!(
                "[token-heat] WARNING: {:.1}% cold tokens — likely τ depressor",
                100.0 * cold_frac
            );
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
    // Default ON. Explicit "0" / "false" / "off" / "no" opts out (parsed once in
    // RuntimeConfig::from_process_config). Delegates to the flag-parameterized core so the
    // pipeline is unit-testable without the memoized `config::get()` singleton.
    normalize_prompt_with(s, crate::config::get().normalize_prompt)
}

/// Core prompt-normalization pipeline, parameterized on the enable flag.
/// `maybe_normalize_prompt` is the production entry point; tests call this
/// directly with an explicit flag. The global `config::get()` is an immutable
/// process snapshot, so per-call mutation is intentionally unsupported.
pub fn normalize_prompt_with(s: &str, enabled: bool) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    if !enabled {
        return Cow::Borrowed(s);
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
        let merges_strings: Vec<(String, String)> = merges
            .iter()
            .map(|(l, r)| (l.to_string(), r.to_string()))
            .collect();
        let (merges, merge_pair_rank) = resolve_merges(&merges_strings, &token_to_id)
            .expect("synth: merges must be consistent with vocab");
        // Test-only `byte_to_id`: best-effort fill, defaults to 0 for any
        // byte whose char isn't in vocab. Production constructors require
        // full 256-byte coverage (else MissingByteSymbol). The BPE tests
        // deliberately use minimal vocabs and only feed ASCII inputs whose
        // bytes ARE present.
        let mut byte_to_id_arr = [0u32; 256];
        for b in 0u32..=255 {
            let ch = byte_to_gpt2_char(b as u8);
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = token_to_id.get(s) {
                byte_to_id_arr[b as usize] = id;
            }
        }
        Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_pair_rank,
            byte_to_id: Some(byte_to_id_arr),
            special_tokens: Vec::new(),
            bos_id: 0,
            eos_id: 0,
            add_bos: false,
            eot_id: None,
            is_gpt2_bpe: true,
            sp_dummy_prefix: true,
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
            &[
                ("h", "e"),
                ("l", "l"),
                ("he", "ll"),
                ("hell", "o"),
                ("l", "o"),
            ],
        );
        assert_eq!(tok.encode_gpt2_bpe("hello"), vec![7]);
    }

    #[test]
    fn encode_partial_merge() {
        // "lol" — only ("l","o") is reachable; "ol" is not in merges.
        // Init ["l","o","l"] → after merge ["lo","l"] → [id 8, id 2].
        let tok = synth(
            &["h", "e", "l", "o", "he", "ll", "hell", "hello", "lo"],
            &[
                ("h", "e"),
                ("l", "l"),
                ("he", "ll"),
                ("hell", "o"),
                ("l", "o"),
            ],
        );
        assert_eq!(tok.encode_gpt2_bpe("lol"), vec![8, 2]);
    }

    #[test]
    fn encode_no_merges() {
        // "ho" — no ("h","o") merge in the table; output is the two byte tokens.
        let tok = synth(
            &["h", "e", "l", "o", "he", "ll", "hell", "hello", "lo"],
            &[
                ("h", "e"),
                ("l", "l"),
                ("he", "ll"),
                ("hell", "o"),
                ("l", "o"),
            ],
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
        let tok = synth(&["a", "b", "ab", "aba"], &[("a", "b"), ("ab", "a")]);
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
        let tok = synth(&["a", "aa", "aaaa"], &[("a", "a"), ("aa", "aa")]);
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
mod consistency_tests {
    //! Tests for the constructor-time vocab/merges consistency checks
    //! introduced for #203. The pre-refactor `Option`-returning constructors
    //! silently dropped inconsistent vocabs (or worse, let the encoder emit
    //! id 0 for missing symbols); the new `Result`-returning constructors
    //! reject loud at load time.
    //!
    //! All tests use `from_gguf_meta_json` since it takes a
    //! `serde_json::Value` directly — the cheapest way to construct
    //! deliberately-broken inputs without going through GGUF or HF formats.

    use super::*;
    use serde_json::json;

    /// Build a SentencePiece-mode meta JSON. Skips byte-coverage check.
    fn sp_meta(tokens: &[&str], merges: &[&str]) -> serde_json::Value {
        json!({
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.merges": merges,
            "tokenizer.ggml.model": "llama",
        })
    }

    /// Build a GPT-2-mode meta JSON. Triggers byte-coverage check.
    /// `tokens` becomes `[byte0_char, byte1_char, ..., byte255_char, ...extras]`,
    /// optionally replacing one byte's char with a placeholder to trigger
    /// `MissingByteSymbol`.
    fn gpt2_meta_full_bytes(
        skip_byte: Option<u8>,
        extras: &[&str],
        merges: &[&str],
    ) -> serde_json::Value {
        let mut tokens: Vec<String> = (0u32..=255)
            .map(|b| {
                if Some(b as u8) == skip_byte {
                    "__placeholder__".to_string()
                } else {
                    byte_to_gpt2_char(b as u8).to_string()
                }
            })
            .collect();
        for e in extras {
            tokens.push((*e).to_string());
        }
        json!({
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.merges": merges,
            "tokenizer.ggml.model": "gpt2",
        })
    }

    #[test]
    fn rejects_missing_merge_operand_left() {
        // Vocab has "b" and "ab" but not "a"; merge "a b" → left absent.
        let meta = sp_meta(&["b", "ab"], &["a b"]);
        let err = match Tokenizer::from_gguf_meta_json(&meta) {
            Err(e) => e,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        match err {
            TokenizerError::MissingMergeOperand {
                rank: 0,
                missing_side: Side::Left,
                ref left,
                ref right,
            } => {
                assert_eq!(left, "a");
                assert_eq!(right, "b");
            }
            other => panic!("expected MissingMergeOperand{{Left}}, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_merge_operand_right() {
        // Vocab has "a" and "ab" but not "b"; merge "a b" → right absent.
        let meta = sp_meta(&["a", "ab"], &["a b"]);
        let err = match Tokenizer::from_gguf_meta_json(&meta) {
            Err(e) => e,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        match err {
            TokenizerError::MissingMergeOperand {
                rank: 0,
                missing_side: Side::Right,
                ..
            } => {}
            other => panic!("expected MissingMergeOperand{{Right}}, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_merge_result() {
        // Vocab has "a" and "b" but NOT "ab"; merge "a b" produces an "ab"
        // that isn't a token id.
        let meta = sp_meta(&["a", "b"], &["a b"]);
        let err = match Tokenizer::from_gguf_meta_json(&meta) {
            Err(e) => e,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        match err {
            TokenizerError::MissingMergeResult {
                rank: 0,
                ref expected,
            } => {
                assert_eq!(expected, "ab");
            }
            other => panic!("expected MissingMergeResult, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_byte_symbol_for_gpt2() {
        // GPT-2 vocab missing byte 0x41 ('A')'s char triggers
        // MissingByteSymbol — encode_gpt2_bpe's initial seed would have
        // silently mapped to id 0 otherwise.
        let meta = gpt2_meta_full_bytes(Some(b'A'), &[], &[]);
        let err = match Tokenizer::from_gguf_meta_json(&meta) {
            Err(e) => e,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        match err {
            TokenizerError::MissingByteSymbol {
                byte: 0x41,
                char: 'A',
            } => {}
            other => panic!("expected MissingByteSymbol{{0x41,'A'}}, got {other:?}"),
        }
    }

    #[test]
    fn skips_byte_check_for_sp() {
        // SP tokenizers (model != "gpt2") don't have byte-coverage
        // requirement — minimal vocab + no merges should succeed.
        let meta = sp_meta(&["a", "b"], &[]);
        let tok = Tokenizer::from_gguf_meta_json(&meta)
            .expect("minimal SP vocab with no merges should succeed");
        assert_eq!(tok.vocab.len(), 2);
        assert!(tok.byte_to_id.is_none());
        assert!(tok.merges.is_empty());
    }

    #[test]
    fn succeeds_on_consistent_gpt2_with_one_merge() {
        // GPT-2 with full 256-byte coverage + one consistent merge.
        // Need 'A' (0x41) and 'B' (0x42) in vocab plus an "AB" extra.
        // byte_to_gpt2_char maps printable ASCII to itself, so the
        // byte chars *are* "A" and "B".
        let meta = gpt2_meta_full_bytes(None, &["AB"], &["A B"]);
        let tok =
            Tokenizer::from_gguf_meta_json(&meta).expect("consistent GPT-2 vocab should succeed");
        assert!(tok.byte_to_id.is_some());
        assert_eq!(tok.merges.len(), 1);
        // Merge resolved to ids: A → 0x41, B → 0x42, AB → 256 (first extra).
        // `merges[0]` is the result id; `merge_pair_rank` carries the pair → rank.
        assert_eq!(tok.merges[0], 256);
        assert_eq!(tok.merge_pair_rank.get(&(0x41, 0x42)).copied(), Some(0));
    }

    #[test]
    fn rejects_metadata_missing_tokens() {
        let meta = json!({});
        let err = match Tokenizer::from_gguf_meta_json(&meta) {
            Err(e) => e,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        match err {
            TokenizerError::MetadataMissing { field } => {
                assert_eq!(field, "tokenizer.ggml.tokens");
            }
            other => panic!("expected MetadataMissing, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod sp_tests {
    //! Tests for `encode_sentencepiece`. Added alongside the
    //! `Vec<char>` + per-trial `String::collect()` → boundary-offset
    //! + `&str` slice rewrite from #229. The four cases below cover
    //! the boundary-vec correctness contract (off-by-one between
    //! char index and byte offset, multi-byte UTF-8, and the
    //! `best_len == 0` fallback branch).

    use super::*;

    /// Build a synthetic SentencePiece Tokenizer from a vocab list.
    /// `is_gpt2_bpe = false` selects the SentencePiece encode path.
    /// No merges (SP doesn't use them), no byte_to_id (SP doesn't
    /// require byte-level coverage).
    fn synth_sp(vocab: &[&str]) -> Tokenizer {
        let vocab: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        let token_to_id: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Tokenizer {
            vocab,
            token_to_id,
            merges: Vec::new(),
            merge_pair_rank: HashMap::new(),
            byte_to_id: None,
            special_tokens: Vec::new(),
            bos_id: 0,
            eos_id: 0,
            add_bos: false,
            eot_id: None,
            is_gpt2_bpe: false,
            // The pre-existing SP unit tests below were written against the
            // unconditional-prefix behavior; keep them on the LLaMA
            // convention. Config-driven coverage lives in
            // `sp_dummy_prefix_tests`.
            sp_dummy_prefix: true,
        }
    }

    #[test]
    fn sp_decode_byte_fallback_cjk() {
        // hunt3 H-C regression: SentencePiece byte-fallback emits a CJK char as
        // a run of <0xHH> tokens. "中" = U+4E2D = bytes E4 B8 AD. The old
        // `byte as char` path mapped each 0xHH to U+00HH and re-UTF8-encoded it
        // to 2 bytes → mojibake. decode_bytes must reassemble the raw 3 bytes,
        // and decode() must from_utf8 them back into "中".
        let tok = synth_sp(&["<0xE4>", "<0xB8>", "<0xAD>"]);
        let toks = [0u32, 1, 2];
        assert_eq!(tok.decode_bytes(&toks), "中".as_bytes());
        assert_eq!(tok.decode(&toks), "中");
    }

    #[test]
    fn sp_decode_hex_escapes_bytes_ascii_and_literal() {
        // ASCII byte-fallback (<0x0A> = newline) and a non-matching literal
        // "<0xZZ>" must pass through untouched, encoded as UTF-8.
        assert_eq!(decode_hex_escapes_bytes("<0x0A>"), vec![b'\n']);
        assert_eq!(decode_hex_escapes_bytes("a<0xZZ>b"), b"a<0xZZ>b".to_vec());
    }

    #[test]
    fn sp_encode_longest_match_ascii() {
        // Vocab has the full "▁hello" plus shorter prefixes. sp_text is
        // "▁hello"; greedy starts at end=6 (full string) and matches
        // immediately → single token.
        let tok = synth_sp(&["\u{2581}hello", "\u{2581}", "h", "e", "l", "o"]);
        assert_eq!(tok.encode_sentencepiece("hello"), vec![0]);
    }

    #[test]
    fn sp_encode_multibyte_char() {
        // 4-byte UTF-8 emoji 🦀 (U+1F980). sp_text = "▁🦀" (3 + 4 = 7
        // bytes, 2 chars). boundaries = [0, 3, 7]. Vocab has "▁" and
        // "🦀" individually but not the pair. Expected: [0, 1].
        // Guards `boundaries[pos]..boundaries[end]` against off-by-one
        // when char span ≠ byte span.
        let tok = synth_sp(&["\u{2581}", "\u{1F980}"]);
        assert_eq!(tok.encode_sentencepiece("\u{1F980}"), vec![0, 1]);
    }

    #[test]
    fn sp_encode_unmatched_fallback() {
        // sp_text = "▁ab"; vocab has "▁" and "a" but not "b". The "b"
        // position drives `best_len == 0`; the fallback slice
        // `boundaries[pos]..boundaries[pos+1]` looks up "b", which is
        // also absent, and the position is silently skipped.
        let tok = synth_sp(&["\u{2581}", "a"]);
        assert_eq!(tok.encode_sentencepiece("ab"), vec![0, 1]);
    }

    #[test]
    fn sp_encode_space_substitution() {
        // Spaces become ▁. "hello world" → "▁hello▁world" → matches
        // "▁hello" (id 0) then "▁world" (id 1). Verifies the single-pass
        // `sp_text` build correctly substitutes spaces.
        let tok = synth_sp(&["\u{2581}hello", "\u{2581}world"]);
        assert_eq!(tok.encode_sentencepiece("hello world"), vec![0, 1]);
    }
}

#[cfg(test)]
mod prompt_norm_tests {
    use super::*;

    // The flag-routing tests below call `normalize_prompt_with(s, enabled)`
    // directly instead of toggling `HIPFIRE_NORMALIZE_PROMPT` and going through
    // `maybe_normalize_prompt`. The env var is consumed once by the memoized
    // `config::get()` singleton, so per-call toggling can't drive it in a
    // shared test process — testing the parameterized core keeps these
    // deterministic and parallel-safe.

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
        assert_eq!(collapse_newline_runs("a\n\n\nb\n\n\n\nc"), "a\n\nb\n\nc");
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
    fn enabled_collapses_newline_runs() {
        // Enabled (the production default) → `\n{3,}` collapses to `\n\n`.
        let s = "a\n\n\nb";
        let out = normalize_prompt_with(s, true);
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        assert_eq!(out.as_ref(), "a\n\nb");
    }

    #[test]
    fn disabled_passes_through_borrowed() {
        // Opt-out (HIPFIRE_NORMALIZE_PROMPT=0 → enabled=false) → input untouched.
        let s = "a\n\n\nb";
        let out = normalize_prompt_with(s, false);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), "a\n\n\nb");
    }

    #[test]
    fn cow_borrowed_when_no_runs() {
        // Even enabled, no `\n{3,}` runs means no rewrite needed → Borrowed.
        let s = "a\n\nb"; // already single-blank
        let out = normalize_prompt_with(s, true);
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
        assert_eq!(normalize_line_endings("a\r\nb\nc\r\nd"), "a\nb\nc\nd");
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
        assert!(!needs_nbsp_replace("\u{00A2}")); // ¢ = 0xC2 0xA2
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
        assert_eq!(strip_trailing_line_ws("\t \tx = 1\n"), "\t \tx = 1\n");
        // Trailing tabs at end of line still get stripped.
        assert_eq!(strip_trailing_line_ws("a\tb\t\nc"), "a\tb\nc");
    }

    #[test]
    fn strip_blank_line_with_indent() {
        // Whitespace-only "blank" line between code blocks — strip the indent.
        assert_eq!(strip_trailing_line_ws("a\n    \nb"), "a\n\nb");
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
        let s = "def foo():   \r\n    return 1   \r\n";
        let out = normalize_prompt_with(s, true);
        assert_eq!(out.as_ref(), "def foo():\n    return 1\n");
    }

    #[test]
    fn pipeline_blank_line_indent_then_collapse() {
        // Indented blank line between top-level defs:
        //   "a\n    \n\nb" — line 2 is whitespace-only, lines 2-3 form a `\n\n\n`
        //   run after stripping. Collapse should reduce to `\n\n`.
        let s = "a\n    \n\nb";
        let out = normalize_prompt_with(s, true);
        assert_eq!(out.as_ref(), "a\n\nb");
    }

    #[test]
    fn pipeline_nbsp_in_prose() {
        let s = "Use\u{00A0}foo()\u{00A0}for\u{00A0}this.";
        let out = normalize_prompt_with(s, true);
        assert_eq!(out.as_ref(), "Use foo() for this.");
    }

    #[test]
    fn pipeline_clean_input_is_borrowed() {
        // No CRLF, no NBSP, no trailing ws, no \n{3,} — must stay Borrowed.
        let s = "Plain prompt.\nSecond line.\n\nThird paragraph.\n";
        let out = normalize_prompt_with(s, true);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), s);
    }

    #[test]
    fn pipeline_explicit_opt_out_skips_all_rules() {
        // Opt-out (enabled=false) must skip CRLF/NBSP/trailing-ws too, not just
        // newline collapse.
        let s = "a\r\nb\u{00A0}c   \nd\n\n\ne";
        let out = normalize_prompt_with(s, false);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), s);
    }
}


#[cfg(test)]
mod sp_dummy_prefix_tests {
    //! Config-driven SP dummy-prefix coverage (gemma4 first-word bug,
    //! 2026-06-09). The unconditional `▁` prefix in `encode_sentencepiece`
    //! shifted the FIRST word of every raw prompt on models whose
    //! tokenizer.json has a no-dummy-prefix normalizer: gemma4 12B encoded
    //! "The capital of France is" as [2, 669 ("▁The"), …] vs the HF ground
    //! truth [2, 818 ("The"), …] — a one-token divergence that made short
    //! raw prompts OOD (garbage replies on hiptrx). All expected ids below
    //! are REAL google/gemma-4-12B-it ids, snapshotted from the hiptrx
    //! model's tokenizer.json and cross-checked against HF `tokenizers`
    //! 0.22.2 (`Tokenizer.encode`, 2026-06-09).

    use super::*;

    /// Minimal gemma4-12B HFQ-metadata fixture: real token ids, real
    /// normalizer shape (bare `Replace(" "→"▁")`, NO Prepend), real
    /// `Split(MergedWithPrevious)` pre_tokenizer, real special-token ids.
    /// The ▁-variants (669 "▁The", 3305 "▁thought", 2430 "▁user") are
    /// deliberately included so the pre-fix encoder's output is
    /// representable — the regression would resurface as 669/3305/2430
    /// instead of silently degrading to char fallback.
    fn gemma4_fixture_metadata() -> String {
        let tok_json = serde_json::json!({
            "normalizer": {
                "type": "Replace",
                "pattern": {"String": " "},
                "content": "▁"
            },
            "pre_tokenizer": {
                "type": "Split",
                "pattern": {"String": " "},
                "behavior": "MergedWithPrevious",
                "invert": false
            },
            "model": {
                "type": "BPE",
                "vocab": {
                    "The": 818,
                    "▁The": 669,
                    "▁capital": 5279,
                    "▁of": 529,
                    "▁France": 7001,
                    "▁is": 563,
                    "thought": 45518,
                    "▁thought": 3305,
                    "user": 2364,
                    "▁user": 2430,
                    "model": 4368,
                    "\n": 107
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": true},
                {"id": 1, "content": "<eos>", "special": true},
                {"id": 2, "content": "<bos>", "special": true},
                {"id": 3, "content": "<unk>", "special": true},
                {"id": 100, "content": "<|channel>", "special": true},
                {"id": 101, "content": "<channel|>", "special": true},
                {"id": 105, "content": "<|turn>", "special": true},
                {"id": 106, "content": "<turn|>", "special": true}
            ]
        });
        serde_json::json!({
            "tokenizer": tok_json.to_string(),
            "generation_config": {"bos_token_id": 2, "eos_token_id": 1}
        })
        .to_string()
    }

    #[test]
    fn gemma4_no_dummy_prefix_first_word_matches_hf() {
        let t = Tokenizer::from_hfq_metadata(&gemma4_fixture_metadata())
            .expect("fixture parses");
        assert_eq!(t.bos_id, 2, "generation_config bos override");
        let mut ids = vec![t.bos_id];
        ids.extend(t.encode("The capital of France is"));
        assert_eq!(ids, vec![2, 818, 5279, 529, 7001, 563]);
    }

    #[test]
    fn gemma4_chat_tail_thought_channel_matches_hf() {
        let t = Tokenizer::from_hfq_metadata(&gemma4_fixture_metadata())
            .expect("fixture parses");
        assert_eq!(
            t.encode("<|channel>thought\n<channel|>The capital of France is"),
            vec![100, 45518, 107, 101, 818, 5279, 529, 7001, 563],
        );
    }

    #[test]
    fn llama_style_prepend_normalizer_keeps_dummy_prefix() {
        let tok_json = serde_json::json!({
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "Prepend", "prepend": "▁"},
                    {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
                ]
            },
            "model": {
                "type": "BPE",
                "vocab": {"▁The": 10, "The": 11, "▁capital": 12},
                "merges": []
            },
            "added_tokens": []
        });
        let t = Tokenizer::from_hf_json(&tok_json.to_string()).expect("parses");
        assert_eq!(t.encode("The capital"), vec![10, 12]);
    }

    #[test]
    fn detection_signals() {
        use serde_json::json;
        assert!(!sp_dummy_prefix_from_hf_json(&json!({
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
            "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious"}
        })));
        assert!(!sp_dummy_prefix_from_hf_json(&json!({})));
        assert!(!sp_dummy_prefix_from_hf_json(&json!({"normalizer": null, "pre_tokenizer": null})));
        assert!(sp_dummy_prefix_from_hf_json(&json!({
            "normalizer": {"type": "Prepend", "prepend": "▁"}
        })));
        for (scheme, want) in [("always", true), ("first", true), ("never", false)] {
            assert_eq!(
                sp_dummy_prefix_from_hf_json(&json!({
                    "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme}
                })),
                want,
                "Metaspace prepend_scheme={scheme}",
            );
        }
        assert!(sp_dummy_prefix_from_hf_json(&json!({
            "normalizer": {"type": "Precompiled", "precompiled_charsmap": ""},
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}
        })));
        assert!(sp_dummy_prefix_from_hf_json(&json!({
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}
        })));
        assert!(!sp_dummy_prefix_from_hf_json(&json!({
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": false}
        })));
        assert!(sp_dummy_prefix_from_hf_json(&json!({
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"String": " "}},
                {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"}
            ]}
        })));
    }

    #[test]
    fn gguf_add_space_prefix_default_on_and_explicit_off() {
        use serde_json::json;
        let meta = json!({
            "tokenizer.ggml.tokens": ["▁hello", "▁world", "hello"],
            "tokenizer.ggml.model": "llama"
        });
        let t = Tokenizer::from_gguf_meta_json(&meta).expect("parses");
        assert_eq!(t.encode("hello"), vec![0], "default: dummy prefix ON");
        let meta = json!({
            "tokenizer.ggml.tokens": ["▁hello", "▁world", "hello"],
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.add_space_prefix": false
        });
        let t = Tokenizer::from_gguf_meta_json(&meta).expect("parses");
        assert_eq!(t.encode("hello"), vec![2], "explicit off: no prefix");
    }
}

#[cfg(test)]
mod lfm2_bos_tests {
    //! Behavioral coverage for LFM2/lfm2_moe `add_bos` resolution in
    //! `from_hfq_metadata` and the encode-time single-BOS guard.
    use super::*;
    use serde_json::json;

    /// Smallest HF tokenizer.json that `from_hf_json` accepts and that can
    /// encode ASCII via the SentencePiece path (no `Ġ` → non-GPT2).
    fn minimal_tok_json() -> String {
        json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<unk>": 0,
                    "h": 3,
                    "i": 4,
                    "\u{2581}": 5
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 1, "content": "<|startoftext|>", "special": true},
                {"id": 2, "content": "<|endoftext|>", "special": true}
            ]
        })
        .to_string()
    }

    fn lfm_meta(add_bos_token: Option<bool>) -> String {
        let mut root = json!({
            "architecture": "lfm2",
            "tokenizer": minimal_tok_json(),
            "generation_config": {
                "bos_token_id": 1,
                "eos_token_id": 2
            },
            "tokenizer_config": {}
        });
        if let Some(ab) = add_bos_token {
            root["tokenizer_config"]["add_bos_token"] = json!(ab);
        }
        root.to_string()
    }

    #[test]
    fn lfm2_absent_add_bos_defaults_true_and_prepends_once() {
        let tok = Tokenizer::from_hfq_metadata(&lfm_meta(None)).expect("meta");
        assert!(
            tok.add_bos,
            "absent add_bos_token must default true for lfm2"
        );
        assert_eq!(tok.bos_id, 1);
        let ids = tok.encode("hi");
        assert_eq!(
            ids.first().copied(),
            Some(1),
            "raw text must begin with bos_id"
        );
        assert_eq!(
            ids.iter().filter(|&&id| id == 1).count(),
            1,
            "exactly one leading bos_id"
        );
    }

    #[test]
    fn lfm2_explicit_false_is_authoritative() {
        let tok = Tokenizer::from_hfq_metadata(&lfm_meta(Some(false))).expect("meta");
        assert!(
            !tok.add_bos,
            "explicit add_bos_token=false must not be overwritten by LFM default"
        );
        let ids = tok.encode("hi");
        assert_ne!(
            ids.first().copied(),
            Some(tok.bos_id),
            "must not prepend bos when add_bos is false"
        );
    }

    #[test]
    fn lfm2_jinja_bos_literal_not_double_prepended() {
        let tok = Tokenizer::from_hfq_metadata(&lfm_meta(None)).expect("meta");
        assert!(tok.add_bos);
        // Jinja chat path already emits the BOS vocabulary string.
        let ids = tok.encode("<|startoftext|>hi");
        assert_eq!(ids.first().copied(), Some(tok.bos_id));
        assert_eq!(
            ids.iter().take_while(|&&id| id == tok.bos_id).count(),
            1,
            "must not double-prepend when text already starts with BOS literal"
        );
    }
}
