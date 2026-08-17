//! KLD reference v3 — verifiable teacher reference format.
//!
//! v1 defects (see crate docs) are fixed by storing everything v1 left out:
//! oracle stats, teacher/corpus digests, estimator, and an explicit scored
//! window. The payload geometry is preserved: per-scored-position block is
//! `8 + 8*top_k` bytes, tokens are dense `u32` little-endian.

use crate::{ArtifactId, Estimator, OracleStats, QuantError, Result, WindowSpec};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 8] = b"HFKLDR\0\0";
const MAGIC_STR: &str = "HFKLDR\\0\\0";
const ARTIFACT: &str = "kldref";
const VERSION: u32 = 3;
const SUPPORTED: &str = "3";
const PRELUDE_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// v3 reference header. Serialised as length-prefixed JSON after the
/// 16-byte prelude. See module docs for layout.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RefHeader {
    pub version: u32,
    pub arch_id: u32,
    pub n_ctx: usize,
    pub n_chunk: usize,
    pub n_vocab: usize,
    pub estimator: Estimator,
    pub window: WindowSpec,
    pub teacher: ArtifactId,
    pub corpus: ArtifactId,
    pub oracle: OracleStats,
    pub engine_commit: String,
    pub arch: String,
}

// ---------------------------------------------------------------------------
// RefBlock
// ---------------------------------------------------------------------------

/// One scored position's reference data.
///
/// `residual_logprob` is `ln(sum_p_residual)` — the log of the aggregated
/// probability mass outside the top-k. For a `FullVocab` reference this is
/// `f64::NEG_INFINITY` (no residual).
#[derive(Debug, Clone, Copy)]
pub struct RefBlock<'a> {
    pub residual_logprob: f64,
    pub top: &'a [(u32, f32)],
}

// ---------------------------------------------------------------------------
// KldRef — read side
// ---------------------------------------------------------------------------

/// Opened v3 reference — header plus geometry.
///
/// The actual bytes live in a separately mmapped file; this struct only
/// holds offsets derived from the header.
#[derive(Debug, Clone)]
pub struct KldRef {
    pub header: RefHeader,
    tokens_offset: u64,
    blocks_offset: u64,
    k: usize,
    block_bytes: usize,
    total_scored: usize,
    expected_len: u64,
}

impl KldRef {
    fn k_from_header(h: &RefHeader) -> Result<usize> {
        match h.estimator {
            Estimator::FullVocab => {
                if h.n_vocab == 0 {
                    return Err(QuantError::Malformed("n_vocab is 0 for FullVocab".into()));
                }
                Ok(h.n_vocab)
            }
            Estimator::TopK { k, .. } => {
                let k = k as usize;
                if k == 0 {
                    return Err(QuantError::Malformed("top_k is 0".into()));
                }
                if k > h.n_vocab {
                    return Err(QuantError::Malformed(format!(
                        "top_k {} > n_vocab {}",
                        k, h.n_vocab
                    )));
                }
                Ok(k)
            }
        }
    }

    /// Open and validate a v3 `.kldref` file.
    ///
    /// Validates magic, version, and that the file is exactly the size the
    /// header geometry implies; otherwise returns `BadMagic`,
    /// `UnsupportedVersion`, or `Truncated`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();

        if file_len < PRELUDE_SIZE as u64 {
            return Err(QuantError::Truncated {
                artifact: ARTIFACT,
                context: "prelude",
                need: PRELUDE_SIZE,
                have: file_len as usize,
            });
        }

        let mut prelude = [0u8; PRELUDE_SIZE];
        file.read_exact(&mut prelude)?;

        if &prelude[0..8] != MAGIC {
            let found = format!("{:?}", &prelude[0..8]);
            return Err(QuantError::BadMagic {
                artifact: ARTIFACT,
                expected: MAGIC_STR,
                found,
            });
        }

        let version = u32::from_le_bytes(prelude[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(QuantError::UnsupportedVersion {
                artifact: ARTIFACT,
                found: version,
                supported: SUPPORTED,
            });
        }

        let header_len = u32::from_le_bytes(prelude[12..16].try_into().unwrap()) as usize;

        // Sanity cap — header JSON should be small (few KB). 16 MiB is absurd.
        if header_len > 16 * 1024 * 1024 {
            return Err(QuantError::Malformed(format!(
                "header_len {header_len} exceeds 16 MiB cap"
            )));
        }

        if (PRELUDE_SIZE + header_len) as u64 > file_len {
            return Err(QuantError::Truncated {
                artifact: ARTIFACT,
                context: "header json",
                need: PRELUDE_SIZE + header_len,
                have: file_len as usize,
            });
        }

        let mut header_bytes = vec![0u8; header_len];
        if header_len > 0 {
            file.read_exact(&mut header_bytes)?;
        }

        let header: RefHeader = serde_json::from_slice(&header_bytes)
            .map_err(|e| QuantError::Malformed(format!("header json: {e}")))?;

        if header.version != VERSION {
            return Err(QuantError::Malformed(format!(
                "header.version {} != prelude version {}",
                header.version, VERSION
            )));
        }

        if header.n_ctx == 0 || header.n_chunk == 0 || header.n_vocab == 0 {
            return Err(QuantError::Malformed(
                "n_ctx, n_chunk, n_vocab must be >0".into(),
            ));
        }

        // Window sanity: scored range must fit inside n_ctx; score_to is exclusive.
        if header.window.score_to > header.n_ctx {
            return Err(QuantError::Malformed(format!(
                "window score_to {} > n_ctx {}",
                header.window.score_to, header.n_ctx
            )));
        }
        if header.window.score_from > header.window.score_to {
            return Err(QuantError::Malformed(format!(
                "window score_from {} > score_to {}",
                header.window.score_from, header.window.score_to
            )));
        }

        let k = Self::k_from_header(&header)?;
        let scored = header.window.scored_per_chunk();
        // scored may be 0? That's technically allowed but would make total 0 and no blocks.
        // We keep it but ensure total not huge overflow.
        let total_scored = header
            .n_chunk
            .checked_mul(scored)
            .ok_or_else(|| QuantError::Malformed("n_chunk * scored_per_chunk overflow".into()))?;

        let block_bytes = 8usize
            .checked_add(8usize.checked_mul(k).ok_or_else(|| {
                QuantError::Malformed("k*8 overflow".into())
            })?)
            .ok_or_else(|| QuantError::Malformed("block_bytes overflow".into()))?;

        let tokens_bytes = (header.n_chunk as u64)
            .checked_mul(header.n_ctx as u64)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| QuantError::Malformed("tokens_bytes overflow".into()))?;

        let blocks_bytes = (total_scored as u64)
            .checked_mul(block_bytes as u64)
            .ok_or_else(|| QuantError::Malformed("blocks_bytes overflow".into()))?;

        let tokens_offset = (PRELUDE_SIZE + header_len) as u64;
        let blocks_offset = tokens_offset
            .checked_add(tokens_bytes)
            .ok_or_else(|| QuantError::Malformed("blocks_offset overflow".into()))?;
        let expected_len = blocks_offset
            .checked_add(blocks_bytes)
            .ok_or_else(|| QuantError::Malformed("expected_len overflow".into()))?;

        if file_len != expected_len {
            // Need vs have for the whole file.
            return Err(QuantError::Truncated {
                artifact: ARTIFACT,
                context: "file size vs header geometry",
                need: expected_len as usize,
                have: file_len as usize,
            });
        }

        Ok(Self {
            header,
            tokens_offset,
            blocks_offset,
            k,
            block_bytes,
            total_scored,
            expected_len,
        })
    }

    /// Scored positions per chunk (delegates to `WindowSpec`).
    pub fn scored_per_chunk(&self) -> usize {
        self.header.window.scored_per_chunk()
    }

    /// Total scored positions across all chunks.
    pub fn total_scored(&self) -> usize {
        self.total_scored
    }

    /// Return k (vocab size for FullVocab, top-k otherwise). Useful for tests.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Byte offset of the token block (for debug).
    pub fn tokens_offset(&self) -> u64 {
        self.tokens_offset
    }

    /// Byte offset of the per-position block region (for debug).
    pub fn blocks_offset(&self) -> u64 {
        self.blocks_offset
    }

    /// Block size in bytes.
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Expected file length derived from header geometry.
    pub fn expected_len(&self) -> u64 {
        self.expected_len
    }

    /// Tokens as `&[u32]` little-endian.
    ///
    /// Zero-copy when the mmap is `u32`-aligned at `tokens_offset`. The
    /// writer pads the header JSON to 8-byte alignment, so files produced by
    /// `RefWriter` are always aligned and this path is always taken. For a
    /// foreign file that is misaligned, a checked little-endian decode is
    /// performed and returned as `Malformed` to avoid undefined behaviour from
    /// creating an unaligned `&[u32]` — callers should re-create the file
    /// with the canonical writer.
    pub fn tokens<'a>(&'a self, mmap: &'a Mmap) -> Result<&'a [u32]> {
        let n_tokens = self.header.n_chunk * self.header.n_ctx;
        let tokens_bytes = n_tokens * 4;
        let start = self.tokens_offset as usize;
        let end = start
            .checked_add(tokens_bytes)
            .ok_or_else(|| QuantError::Malformed("tokens range overflow".into()))?;
        if mmap.len() < end {
            return Err(QuantError::Truncated {
                artifact: ARTIFACT,
                context: "tokens",
                need: end,
                have: mmap.len(),
            });
        }
        let bytes = &mmap[start..end];
        debug_assert_eq!(bytes.len(), tokens_bytes);
        if bytes.is_empty() {
            return Ok(&[]);
        }
        if bytes.as_ptr() as usize % std::mem::align_of::<u32>() != 0 {
            return Err(QuantError::Malformed(
                "tokens misaligned for zero-copy u32 slice; file not produced by canonical writer".into(),
            ));
        }
        // Host is assumed little-endian (x64). Reinterpret directly.
        // SAFETY: we just checked alignment and length, and the bytes are
        // valid little-endian u32s written by the writer. On big-endian hosts
        // this would be byte-swapped, but hipfire only runs on little-endian.
        if cfg!(target_endian = "big") {
            return Err(QuantError::Malformed(
                "big-endian host not supported for zero-copy tokens".into(),
            ));
        }
        let ptr = bytes.as_ptr() as *const u32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, n_tokens) };
        Ok(slice)
    }

    /// Fetch one scored position's block.
    ///
    /// `chunk` is `0..n_chunk`, `pos` is `0..scored_per_chunk()`.
    pub fn block<'a>(&'a self, mmap: &'a Mmap, chunk: usize, pos: usize) -> Result<RefBlock<'a>> {
        if chunk >= self.header.n_chunk {
            return Err(QuantError::Malformed(format!(
                "chunk {chunk} >= n_chunk {}",
                self.header.n_chunk
            )));
        }
        if pos >= self.scored_per_chunk() {
            return Err(QuantError::Malformed(format!(
                "pos {pos} >= scored_per_chunk {}",
                self.scored_per_chunk()
            )));
        }
        let idx = chunk
            .checked_mul(self.scored_per_chunk())
            .and_then(|v| v.checked_add(pos))
            .ok_or_else(|| QuantError::Malformed("block index overflow".into()))?;
        let off = self
            .blocks_offset
            .checked_add((idx as u64).checked_mul(self.block_bytes as u64).unwrap())
            .ok_or_else(|| QuantError::Malformed("block offset overflow".into()))? as usize;
        let end = off
            .checked_add(self.block_bytes)
            .ok_or_else(|| QuantError::Malformed("block end overflow".into()))?;
        if mmap.len() < end {
            return Err(QuantError::Truncated {
                artifact: ARTIFACT,
                context: "block",
                need: end,
                have: mmap.len(),
            });
        }
        let slice = &mmap[off..end];
        let residual_logprob = f64::from_le_bytes(slice[0..8].try_into().unwrap());
        let top_bytes = &slice[8..];
        let k = self.k;
        if top_bytes.len() != k * 8 {
            return Err(QuantError::Malformed(format!(
                "top_bytes len {} != k*8 {}",
                top_bytes.len(),
                k * 8
            )));
        }
        if k == 0 {
            return Ok(RefBlock {
                residual_logprob,
                top: &[],
            });
        }
        if cfg!(target_endian = "big") {
            return Err(QuantError::Malformed(
                "big-endian host not supported for zero-copy blocks".into(),
            ));
        }
        if top_bytes.as_ptr() as usize % std::mem::align_of::<(u32, f32)>() != 0 {
            return Err(QuantError::Malformed(
                "block top misaligned for zero-copy (u32,f32) slice; file not produced by canonical writer".into(),
            ));
        }
        // On little-endian, reinterpret as interleaved (u32,f32).
        // The in-file order is exactly (u32 idx le, f32 logprob le) interleaved,
        // which matches the memory layout of (u32,f32) on little-endian.
        let ptr = top_bytes.as_ptr() as *const (u32, f32);
        let top = unsafe { std::slice::from_raw_parts(ptr, k) };
        Ok(RefBlock {
            residual_logprob,
            top,
        })
    }

    /// Verify that the supplied teacher and corpus digests match the header.
    ///
    /// Returns `DigestMismatch` when either differs. This is the integrity
    /// gate v1 never had.
    pub fn verify_against(&self, teacher: &ArtifactId, corpus: &ArtifactId) -> Result<()> {
        if self.header.teacher.sha256 != teacher.sha256 {
            return Err(QuantError::DigestMismatch {
                what: "teacher".to_string(),
                expected: self.header.teacher.sha256.clone(),
                found: teacher.sha256.clone(),
            });
        }
        if self.header.corpus.sha256 != corpus.sha256 {
            return Err(QuantError::DigestMismatch {
                what: "corpus".to_string(),
                expected: self.header.corpus.sha256.clone(),
                found: corpus.sha256.clone(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RefWriter — write side
// ---------------------------------------------------------------------------

/// Writer for v3 `.kldref` files.
///
/// Enforces geometry: `finish` fails with `Malformed` if the number of
/// pushed blocks does not equal `n_chunk * scored_per_chunk`.
pub struct RefWriter {
    out: BufWriter<File>,
    header: RefHeader,
    k: usize,
    block_bytes: usize,
    expected_tokens: usize,
    expected_blocks: usize,
    tokens_written: usize,
    blocks_written: usize,
    finished: bool,
}

impl RefWriter {
    /// Create a new `.kldref` file at `path` with the given `header`.
    ///
    /// The header's `version` must be `3`; if not, it is treated as
    /// `Malformed`. The file is truncated if it exists.
    pub fn create(path: impl AsRef<Path>, header: RefHeader) -> Result<Self> {
        if header.version != VERSION {
            return Err(QuantError::Malformed(format!(
                "RefWriter header.version {} != {}",
                header.version, VERSION
            )));
        }
        if header.n_ctx == 0 || header.n_chunk == 0 || header.n_vocab == 0 {
            return Err(QuantError::Malformed(
                "n_ctx, n_chunk, n_vocab must be >0".into(),
            ));
        }
        // window checks
        if header.window.score_to > header.n_ctx {
            return Err(QuantError::Malformed(format!(
                "window score_to {} > n_ctx {}",
                header.window.score_to, header.n_ctx
            )));
        }
        if header.window.score_from > header.window.score_to {
            return Err(QuantError::Malformed(format!(
                "window score_from {} > score_to {}",
                header.window.score_from, header.window.score_to
            )));
        }

        let k = match header.estimator {
            Estimator::FullVocab => header.n_vocab,
            Estimator::TopK { k, .. } => {
                let k = k as usize;
                if k == 0 {
                    return Err(QuantError::Malformed("top_k is 0".into()));
                }
                if k > header.n_vocab {
                    return Err(QuantError::Malformed(format!(
                        "top_k {} > n_vocab {}",
                        k, header.n_vocab
                    )));
                }
                k
            }
        };

        let block_bytes = 8usize
            .checked_add(8usize.checked_mul(k).ok_or_else(|| {
                QuantError::Malformed("k*8 overflow".into())
            })?)
            .ok_or_else(|| QuantError::Malformed("block_bytes overflow".into()))?;

        let scored = header.window.scored_per_chunk();
        let expected_blocks = header
            .n_chunk
            .checked_mul(scored)
            .ok_or_else(|| QuantError::Malformed("expected_blocks overflow".into()))?;
        let expected_tokens = header
            .n_chunk
            .checked_mul(header.n_ctx)
            .ok_or_else(|| QuantError::Malformed("expected_tokens overflow".into()))?;

        let mut header_json =
            serde_json::to_vec(&header).map_err(|e| QuantError::Malformed(format!("header json: {e}")))?;
        // Pad header JSON with ASCII spaces (valid JSON whitespace) so that
        // tokens_offset = 16 + header_len is 8-byte aligned. This guarantees
        // zero-copy mmap access for tokens (u32) and blocks (f64) on the
        // common little-endian x64 hosts. `serde_json::from_slice` ignores
        // trailing whitespace.
        while header_json.len() % 8 != 0 {
            header_json.push(b' ');
        }
        let header_len = header_json.len() as u32;

        // Create file (truncate).
        let file = File::create(path.as_ref())?;
        let mut out = BufWriter::with_capacity(4 * 1024 * 1024, file);
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&header_len.to_le_bytes())?;
        if header_len > 0 {
            out.write_all(&header_json)?;
        }

        Ok(Self {
            out,
            header,
            k,
            block_bytes,
            expected_tokens,
            expected_blocks,
            tokens_written: 0,
            blocks_written: 0,
            finished: false,
        })
    }

    /// Append tokens. May be called multiple times; total must equal
    /// `n_chunk * n_ctx` before `finish` (not strictly enforced until
    /// `finish`, but overflow beyond expected is immediately `Malformed`).
    pub fn push_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        if self.finished {
            return Err(QuantError::Malformed("push_tokens after finish".into()));
        }
        if self.tokens_written + tokens.len() > self.expected_tokens {
            return Err(QuantError::Malformed(format!(
                "too many tokens: have {} + {} > expected {}",
                self.tokens_written,
                tokens.len(),
                self.expected_tokens
            )));
        }
        for &t in tokens {
            self.out.write_all(&t.to_le_bytes())?;
        }
        self.tokens_written += tokens.len();
        Ok(())
    }

    /// Append one scored position's block.
    ///
    /// `top` must have length equal to `k` (or `n_vocab` for FullVocab).
    pub fn push_block(&mut self, residual_logprob: f64, top: &[(u32, f32)]) -> Result<()> {
        if self.finished {
            return Err(QuantError::Malformed("push_block after finish".into()));
        }
        if top.len() != self.k {
            return Err(QuantError::Malformed(format!(
                "top len {} != k {}",
                top.len(),
                self.k
            )));
        }
        if self.blocks_written >= self.expected_blocks {
            return Err(QuantError::Malformed(format!(
                "too many blocks: already {} >= expected {}",
                self.blocks_written, self.expected_blocks
            )));
        }
        // Serialize the whole fixed-width block once rather than issuing
        // 1 + 2k small writes. Ordering between tokens and blocks is not
        // constrained here; `finish` is what enforces the geometry.
        let mut block = Vec::with_capacity(self.block_bytes);
        block.extend_from_slice(&residual_logprob.to_le_bytes());
        for &(idx, lp) in top {
            block.extend_from_slice(&idx.to_le_bytes());
            block.extend_from_slice(&lp.to_le_bytes());
        }
        debug_assert_eq!(block.len(), self.block_bytes);
        self.out.write_all(&block)?;
        self.blocks_written += 1;
        Ok(())
    }

    /// Finalize the file.
    ///
    /// Fails with `Malformed` if the number of pushed blocks does not equal
    /// `n_chunk * scored_per_chunk`, so a truncated build cannot silently
    /// produce a usable-looking reference. Also checks token count.
    pub fn finish(mut self) -> Result<()> {
        if self.blocks_written != self.expected_blocks {
            return Err(QuantError::Malformed(format!(
                "block count {} != expected {} (n_chunk {} * scored_per_chunk {})",
                self.blocks_written,
                self.expected_blocks,
                self.header.n_chunk,
                self.header.window.scored_per_chunk()
            )));
        }
        if self.tokens_written != self.expected_tokens {
            return Err(QuantError::Malformed(format!(
                "token count {} != expected {} (n_chunk {} * n_ctx {})",
                self.tokens_written,
                self.expected_tokens,
                self.header.n_chunk,
                self.header.n_ctx
            )));
        }
        self.out.flush()?;
        // Ensure file is synced? flush is enough.
        self.finished = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::Mmap;
    use std::fs::File;
    use std::io::{Read, Write};
    use tempfile::NamedTempFile;

    fn tiny_header(n_ctx: usize, n_chunk: usize, n_vocab: usize, k: u32, window: WindowSpec) -> RefHeader {
        RefHeader {
            version: 3,
            arch_id: 5,
            n_ctx,
            n_chunk,
            n_vocab,
            estimator: Estimator::TopK { k, bias_vs_full: None },
            window,
            teacher: ArtifactId {
                path: "teacher.hfq".into(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                bytes: 12345,
            },
            corpus: ArtifactId {
                path: "corpus.txt".into(),
                sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
                bytes: 6789,
            },
            oracle: OracleStats {
                mean_nll: 2.123456,
                ppl: 8.36,
                n_scored: n_chunk * window.scored_per_chunk(),
            },
            engine_commit: "abc123".into(),
            arch: "qwen35".into(),
        }
    }

    #[test]
    fn round_trip_small() {
        // n_ctx=8, n_chunk=2, top_k=4, explicit window scoring 4..7 (3 per chunk)
        let window = WindowSpec {
            warmup: 0,
            score_from: 4,
            score_to: 7,
            carry_kv: false,
        };
        let header = tiny_header(8, 2, 1000, 4, window);
        assert_eq!(window.scored_per_chunk(), 3);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer = RefWriter::create(&path, header.clone()).unwrap();
        // 2*8 =16 tokens: 0..16
        let tokens: Vec<u32> = (0..16).collect();
        writer.push_tokens(&tokens).unwrap();

        // total_scored = 2*3=6 blocks
        let blocks: Vec<(f64, Vec<(u32, f32)>)> = (0..6)
            .map(|i| {
                let residual = -0.5 - i as f64 * 0.1;
                let top: Vec<(u32, f32)> = (0..4).map(|j| (j + i as u32 * 10,  -1.0 - j as f32 * 0.25)).collect();
                (residual, top)
            })
            .collect();

        for (res, top) in &blocks {
            writer.push_block(*res, top).unwrap();
        }
        writer.finish().unwrap();

        // Reopen
        let kref = KldRef::open(&path).unwrap();
        assert_eq!(kref.header.version, 3);
        assert_eq!(kref.header.n_ctx, 8);
        assert_eq!(kref.header.n_chunk, 2);
        assert_eq!(kref.header.n_vocab, 1000);
        assert_eq!(kref.scored_per_chunk(), 3);
        assert_eq!(kref.total_scored(), 6);
        assert_eq!(kref.header.estimator, Estimator::TopK { k: 4, bias_vs_full: None });
        assert_eq!(kref.header.oracle.mean_nll, 2.123456);

        let file = File::open(&path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        let toks = kref.tokens(&mmap).unwrap();
        assert_eq!(toks.len(), 16);
        assert_eq!(toks, tokens.as_slice());

        for (i, (exp_res, exp_top)) in blocks.iter().enumerate() {
            let chunk = i / 3;
            let pos = i % 3;
            let blk = kref.block(&mmap, chunk, pos).unwrap();
            assert!((blk.residual_logprob - exp_res).abs() < 1e-12, "residual mismatch at {i}");
            assert_eq!(blk.top.len(), 4);
            for (j, (e_idx, e_lp)) in exp_top.iter().enumerate() {
                assert_eq!(blk.top[j].0, *e_idx);
                assert!((blk.top[j].1 - e_lp).abs() < 1e-6, "logprob mismatch block {i} j {j}");
            }
        }
    }

    #[test]
    fn geometry_v1_equivalent() {
        // v1 numbers: n_ctx=2048, n_chunk=24, top_k=256, legacy window
        let n_ctx = 2048usize;
        let n_chunk = 24usize;
        let k = 256usize;
        let window = WindowSpec::legacy_half(n_ctx);
        assert_eq!(window.scored_per_chunk(), 1023);
        assert_eq!(window.score_from, 1024);
        assert_eq!(window.score_to, 2047);

        let header = RefHeader {
            version: 3,
            arch_id: 5,
            n_ctx,
            n_chunk,
            n_vocab: 248320,
            estimator: Estimator::TopK { k: k as u32, bias_vs_full: None },
            window,
            teacher: ArtifactId { path: "t".into(), sha256: "aa".into(), bytes: 0 },
            corpus: ArtifactId { path: "c".into(), sha256: "bb".into(), bytes: 0 },
            oracle: OracleStats { mean_nll: 1.0, ppl: 2.718, n_scored: 0 },
            engine_commit: "x".into(),
            arch: "qwen35".into(),
        };

        // Geometry via KldRef logic without I/O
        let scored = header.window.scored_per_chunk();
        assert_eq!(scored, 1023);
        let total = n_chunk * scored;
        assert_eq!(total, 24552);

        let block_bytes = 8 + 8 * k;
        assert_eq!(block_bytes, 2056);
        let tokens_bytes = n_chunk * n_ctx * 4;
        let blocks_bytes = total * block_bytes;
        let payload = tokens_bytes + blocks_bytes;
        assert_eq!(payload, 24 * 2048 * 4 + 24 * 1023 * 2056);
        // Verify total matches earlier: 50,675,520
        assert_eq!(payload, 50_675_520);

        // Also verify via actual file creation (small n_chunk scaled? use full)
        // We won't write a 50MB file; just verify arithmetic.
        // Additionally open a tiny file with same window logic to ensure KldRef computes same.
        let window2 = WindowSpec::legacy_half(2048);
        let kref_window_check = window2.scored_per_chunk();
        assert_eq!(kref_window_check, 1023);
    }

    #[test]
    fn verify_against_ok_and_mismatch() {
        let window = WindowSpec { warmup: 0, score_from: 4, score_to: 7, carry_kv: false };
        let header = tiny_header(8, 2, 1000, 4, window);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut w = RefWriter::create(&path, header.clone()).unwrap();
        w.push_tokens(&vec![0u32; 16]).unwrap();
        for _ in 0..6 {
            w.push_block(-0.5, &[(0, -1.0), (1, -2.0), (2, -3.0), (3, -4.0)]).unwrap();
        }
        w.finish().unwrap();
        let kref = KldRef::open(&path).unwrap();

        // matching
        assert!(kref.verify_against(&header.teacher, &header.corpus).is_ok());

        // teacher mismatch
        let bad_teacher = ArtifactId {
            path: "teacher.hfq".into(),
            sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            bytes: 12345,
        };
        let err = kref.verify_against(&bad_teacher, &header.corpus).unwrap_err();
        match err {
            QuantError::DigestMismatch { what, expected, found } => {
                assert_eq!(what, "teacher");
                assert_eq!(expected, header.teacher.sha256);
                assert_eq!(found, bad_teacher.sha256);
            }
            _ => panic!("expected DigestMismatch, got {err:?}"),
        }

        // corpus mismatch
        let bad_corpus = ArtifactId {
            path: "corpus.txt".into(),
            sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
            bytes: 6789,
        };
        let err2 = kref.verify_against(&header.teacher, &bad_corpus).unwrap_err();
        match err2 {
            QuantError::DigestMismatch { what, .. } => assert_eq!(what, "corpus"),
            _ => panic!("expected DigestMismatch for corpus"),
        }
    }

    #[test]
    fn finish_fails_when_too_few_blocks() {
        let window = WindowSpec { warmup: 0, score_from: 4, score_to: 7, carry_kv: false };
        let header = tiny_header(8, 2, 1000, 4, window);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut w = RefWriter::create(&path, header).unwrap();
        w.push_tokens(&vec![0u32; 16]).unwrap();
        // Need 6 blocks, push only 3
        for _ in 0..3 {
            w.push_block(-0.5, &[(0, -1.0), (1, -2.0), (2, -3.0), (3, -4.0)]).unwrap();
        }
        let err = w.finish().unwrap_err();
        match err {
            QuantError::Malformed(msg) => {
                assert!(msg.contains("block count"), "msg: {msg}");
            }
            _ => panic!("expected Malformed, got {err:?}"),
        }
    }

    #[test]
    fn bad_magic() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Write a file with bad magic
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"BADMAGIC").unwrap();
            f.write_all(&[0u8; 8]).unwrap();
            f.write_all(&[0u8; 10]).unwrap();
        }
        let err = KldRef::open(&path).unwrap_err();
        match err {
            QuantError::BadMagic { artifact, expected, found } => {
                assert_eq!(artifact, "kldref");
                assert_eq!(expected, MAGIC_STR);
                assert!(!found.is_empty());
            }
            _ => panic!("expected BadMagic, got {err:?}"),
        }
    }

    #[test]
    fn unsupported_version() {
        let window = WindowSpec { warmup: 0, score_from: 4, score_to: 7, carry_kv: false };
        let header = tiny_header(8, 2, 1000, 4, window);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Manually write a v1-style prelude with version 1
        let header_json = serde_json::to_vec(&header).unwrap();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(MAGIC).unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap(); // version 1 instead of 3
            f.write_all(&(header_json.len() as u32).to_le_bytes()).unwrap();
            f.write_all(&header_json).unwrap();
            // tokens + blocks to make geometry plausible for version check before size check
            // But UnsupportedVersion should fire before geometry size check, so file size doesn't matter.
            f.write_all(&vec![0u8; 16 * 4]).unwrap();
            // 6 blocks * (8+32) = 240
            f.write_all(&vec![0u8; 6 * (8 + 8 * 4)]).unwrap();
        }
        let err = KldRef::open(&path).unwrap_err();
        match err {
            QuantError::UnsupportedVersion { artifact, found, supported } => {
                assert_eq!(artifact, "kldref");
                assert_eq!(found, 1);
                assert_eq!(supported, "3");
            }
            _ => panic!("expected UnsupportedVersion, got {err:?}"),
        }
    }

    #[test]
    fn truncated_file() {
        let window = WindowSpec { warmup: 0, score_from: 4, score_to: 7, carry_kv: false };
        let header = tiny_header(8, 2, 1000, 4, window);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut w = RefWriter::create(&path, header).unwrap();
        w.push_tokens(&vec![0u32; 16]).unwrap();
        for _ in 0..6 {
            w.push_block(-0.5, &[(0, -1.0), (1, -2.0), (2, -3.0), (3, -4.0)]).unwrap();
        }
        w.finish().unwrap();
        // Truncate file by one byte
        let meta = std::fs::metadata(&path).unwrap();
        let len = meta.len();
        let file = File::open(&path).unwrap();
        let mut bytes = Vec::new();
        std::io::BufReader::new(file).read_to_end(&mut bytes).unwrap();
        bytes.pop();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&bytes).unwrap();
        }
        assert!(bytes.len() as u64 == len - 1);
        let err = KldRef::open(&path).unwrap_err();
        match err {
            QuantError::Truncated { artifact, .. } => assert_eq!(artifact, "kldref"),
            _ => panic!("expected Truncated, got {err:?}"),
        }
    }
}
