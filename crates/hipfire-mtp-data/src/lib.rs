// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Architecture-neutral MTP feature shards.
//!
//! The producer is architecture-specific (Qwen, DeepSeek, MiniMax), but the
//! expensive product is not: committed token ids plus the exact deployed
//! trunk's post-final-norm hidden rows. Keeping that contract independent of
//! PyTorch and of any one architecture lets R9700 feature generation feed a
//! head-only trainer on R9700 today and an MI300X trainer later.

use half::bf16;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 8] = b"HFMTPF01";
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORD_BYTES: usize = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureHeader {
    pub schema_version: u32,
    pub architecture: String,
    pub model: String,
    pub trunk_path: String,
    pub trunk_sha256: String,
    pub source_manifest_sha256: String,
    pub producer_git_commit: String,
    pub split: String,
    pub hidden_dim: u32,
    pub recursive_steps: u32,
    pub hidden_dtype: String,
    pub record_checksum: String,
    pub kv_mode: String,
    pub state_quant: String,
}

impl FeatureHeader {
    pub fn validate(&self) -> io::Result<()> {
        if self.schema_version != 1 {
            return Err(invalid_data(format!(
                "unsupported feature schema {}",
                self.schema_version
            )));
        }
        if self.hidden_dim == 0 || self.recursive_steps == 0 {
            return Err(invalid_data(
                "hidden_dim and recursive_steps must be non-zero",
            ));
        }
        if self.hidden_dtype != "bf16-le" {
            return Err(invalid_data(format!(
                "unsupported hidden dtype {}",
                self.hidden_dtype
            )));
        }
        if self.record_checksum != "xxh3-64" {
            return Err(invalid_data(format!(
                "unsupported record checksum {}",
                self.record_checksum
            )));
        }
        Ok(())
    }
}

/// One independent MTP attention sequence.
///
/// `tokens` contains `hidden_rows + recursive_steps` entries. Serving-aligned
/// training pairs hidden row `i` (`h[t]`) with `tokens[i + 1]` (`x[t+1]`) and
/// predicts `tokens[i + k + 2]` at recursive depth `k`. Consequently schema 1
/// exposes `hidden_rows - 1` usable rows; the final hidden row is retained for
/// backward-compatible decoding of already-produced feature shards. The
/// head's private attention cache starts empty at `absolute_start + 1`,
/// matching a cropped deployment trajectory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureRecord {
    pub id: String,
    pub source_ordinal: u64,
    pub absolute_start: u32,
    pub hidden_rows: u32,
    pub tokens: Vec<u32>,
    pub hidden_bf16: Vec<u16>,
}

impl FeatureRecord {
    pub fn validate(&self, header: &FeatureHeader) -> io::Result<()> {
        let rows = self.hidden_rows as usize;
        let k = header.recursive_steps as usize;
        let dim = header.hidden_dim as usize;
        if rows == 0 {
            return Err(invalid_data("feature record has zero hidden rows"));
        }
        if self.tokens.len() != rows + k {
            return Err(invalid_data(format!(
                "token count {} != hidden_rows {} + K {}",
                self.tokens.len(),
                rows,
                k
            )));
        }
        let expected = rows
            .checked_mul(dim)
            .ok_or_else(|| invalid_data("hidden element count overflow"))?;
        if self.hidden_bf16.len() != expected {
            return Err(invalid_data(format!(
                "hidden element count {} != rows {} * dim {}",
                self.hidden_bf16.len(),
                rows,
                dim
            )));
        }
        Ok(())
    }
}

pub fn f32_to_bf16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect()
}

/// Evenly cover an assistant trajectory while keeping each MTP attention
/// sequence bounded. The first and last windows are always represented when
/// more than one window is requested; duplicate starts are removed.
pub fn window_starts(
    available_rows: usize,
    window_rows: usize,
    windows_per_record: usize,
) -> Vec<usize> {
    if available_rows == 0 || window_rows == 0 || windows_per_record == 0 {
        return Vec::new();
    }
    if available_rows <= window_rows || windows_per_record == 1 {
        return vec![0];
    }
    let last = available_rows - window_rows;
    let count = windows_per_record.min(last + 1);
    let mut starts = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * last / (count - 1);
        if starts.last().copied() != Some(start) {
            starts.push(start);
        }
    }
    starts
}

pub struct AtomicShardWriter {
    final_path: PathBuf,
    partial_path: PathBuf,
    writer: Option<BufWriter<File>>,
    header: FeatureHeader,
    records: u64,
    hidden_rows: u64,
}

impl AtomicShardWriter {
    pub fn create(path: impl AsRef<Path>, header: FeatureHeader) -> io::Result<Self> {
        header.validate()?;
        let final_path = path.as_ref().to_path_buf();
        let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let file_name = final_path
            .file_name()
            .ok_or_else(|| invalid_data("shard path has no file name"))?
            .to_string_lossy();
        let partial_path = parent.join(format!(".{file_name}.partial"));
        let file = File::create(&partial_path)?;
        let mut writer = BufWriter::new(file);
        let header_json = serde_json::to_vec(&header).map_err(invalid_json)?;
        if header_json.len() > MAX_HEADER_BYTES {
            return Err(invalid_data("feature header is too large"));
        }
        writer.write_all(MAGIC)?;
        write_u32(
            &mut writer,
            checked_u32(header_json.len(), "header length")?,
        )?;
        writer.write_all(&header_json)?;
        Ok(Self {
            final_path,
            partial_path,
            writer: Some(writer),
            header,
            records: 0,
            hidden_rows: 0,
        })
    }

    pub fn write_record(&mut self, record: &FeatureRecord) -> io::Result<()> {
        record.validate(&self.header)?;
        let payload_len = encoded_record_len(record)?;
        if payload_len > MAX_RECORD_BYTES {
            return Err(invalid_data("feature record is too large"));
        }
        let mut payload = Vec::with_capacity(payload_len);
        write_u32(&mut payload, checked_u32(record.id.len(), "id length")?)?;
        write_u64(&mut payload, record.source_ordinal)?;
        write_u32(&mut payload, record.absolute_start)?;
        write_u32(&mut payload, record.hidden_rows)?;
        write_u32(
            &mut payload,
            checked_u32(record.tokens.len(), "token count")?,
        )?;
        payload.extend_from_slice(record.id.as_bytes());
        for token in &record.tokens {
            write_u32(&mut payload, *token)?;
        }
        for value in &record.hidden_bf16 {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        debug_assert_eq!(payload.len(), payload_len);
        let checksum = xxhash_rust::xxh3::xxh3_64(&payload);
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| invalid_data("feature shard is already finished"))?;
        write_u64(writer, payload_len as u64)?;
        write_u64(writer, checksum)?;
        writer.write_all(&payload)?;
        self.records += 1;
        self.hidden_rows += record.hidden_rows as u64;
        Ok(())
    }

    pub fn records(&self) -> u64 {
        self.records
    }

    pub fn hidden_rows(&self) -> u64 {
        self.hidden_rows
    }

    pub fn finish(mut self) -> io::Result<ShardSummary> {
        let mut writer = self
            .writer
            .take()
            .ok_or_else(|| invalid_data("feature shard is already finished"))?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&self.partial_path, &self.final_path)?;
        if let Some(parent) = self.final_path.parent() {
            File::open(parent)?.sync_all()?;
        }
        Ok(ShardSummary {
            path: self.final_path.clone(),
            records: self.records,
            hidden_rows: self.hidden_rows,
        })
    }
}

impl Drop for AtomicShardWriter {
    fn drop(&mut self) {
        if self.writer.is_some() {
            let _ = fs::remove_file(&self.partial_path);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSummary {
    pub path: PathBuf,
    pub records: u64,
    pub hidden_rows: u64,
}

pub struct ShardReader {
    reader: BufReader<File>,
    header: FeatureHeader,
}

impl ShardReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid_data("invalid MTP feature shard magic"));
        }
        let header_len = read_u32(&mut reader)? as usize;
        if header_len > MAX_HEADER_BYTES {
            return Err(invalid_data("feature header is too large"));
        }
        let mut bytes = vec![0u8; header_len];
        reader.read_exact(&mut bytes)?;
        let header: FeatureHeader = serde_json::from_slice(&bytes).map_err(invalid_json)?;
        header.validate()?;
        Ok(Self { reader, header })
    }

    pub fn header(&self) -> &FeatureHeader {
        &self.header
    }

    pub fn read_record(&mut self) -> io::Result<Option<FeatureRecord>> {
        let payload_len = match read_u64_or_eof(&mut self.reader)? {
            Some(value) => usize::try_from(value)
                .map_err(|_| invalid_data("record length does not fit usize"))?,
            None => return Ok(None),
        };
        if payload_len > MAX_RECORD_BYTES {
            return Err(invalid_data("feature record is too large"));
        }
        let expected_checksum = read_u64(&mut self.reader)?;
        let mut payload = vec![0u8; payload_len];
        self.reader.read_exact(&mut payload)?;
        if xxhash_rust::xxh3::xxh3_64(&payload) != expected_checksum {
            return Err(invalid_data("feature record checksum mismatch"));
        }
        let mut cursor = io::Cursor::new(payload.as_slice());
        let id_len = read_u32(&mut cursor)? as usize;
        let source_ordinal = read_u64(&mut cursor)?;
        let absolute_start = read_u32(&mut cursor)?;
        let hidden_rows = read_u32(&mut cursor)?;
        let token_count = read_u32(&mut cursor)? as usize;
        let fixed_bytes = id_len
            .checked_add(
                token_count
                    .checked_mul(4)
                    .ok_or_else(|| invalid_data("feature token byte count overflow"))?,
            )
            .and_then(|value| {
                value.checked_add(
                    (hidden_rows as usize)
                        .checked_mul(self.header.hidden_dim as usize)?
                        .checked_mul(2)?,
                )
            })
            .ok_or_else(|| invalid_data("feature payload size overflow"))?;
        if cursor.position() as usize + fixed_bytes != payload_len {
            return Err(invalid_data("feature payload length does not match fields"));
        }
        let mut id_bytes = vec![0u8; id_len];
        cursor.read_exact(&mut id_bytes)?;
        let id = String::from_utf8(id_bytes)
            .map_err(|error| invalid_data(format!("record id is not UTF-8: {error}")))?;
        let mut tokens = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            tokens.push(read_u32(&mut cursor)?);
        }
        let hidden_count = hidden_rows as usize * self.header.hidden_dim as usize;
        let mut hidden_bf16 = Vec::with_capacity(hidden_count);
        for _ in 0..hidden_count {
            let mut bits = [0u8; 2];
            cursor.read_exact(&mut bits)?;
            hidden_bf16.push(u16::from_le_bytes(bits));
        }
        let record = FeatureRecord {
            id,
            source_ordinal,
            absolute_start,
            hidden_rows,
            tokens,
            hidden_bf16,
        };
        record.validate(&self.header)?;
        Ok(Some(record))
    }
}

fn encoded_record_len(record: &FeatureRecord) -> io::Result<usize> {
    4usize
        .checked_add(8 + 4 + 4 + 4)
        .and_then(|value| value.checked_add(record.id.len()))
        .and_then(|value| value.checked_add(record.tokens.len().checked_mul(4)?))
        .and_then(|value| value.checked_add(record.hidden_bf16.len().checked_mul(2)?))
        .ok_or_else(|| invalid_data("feature record length overflow"))
}

fn checked_u32(value: usize, label: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_data(format!("{label} does not fit u32")))
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u64_or_eof(reader: &mut impl Read) -> io::Result<Option<u64>> {
    let mut bytes = [0u8; 8];
    let mut read = 0;
    while read < bytes.len() {
        match reader.read(&mut bytes[read..])? {
            0 if read == 0 => return Ok(None),
            0 => return Err(io::Error::new(ErrorKind::UnexpectedEof, "truncated u64")),
            count => read += count,
        }
    }
    Ok(Some(u64::from_le_bytes(bytes)))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

fn invalid_json(error: serde_json::Error) -> io::Error {
    invalid_data(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn header() -> FeatureHeader {
        FeatureHeader {
            schema_version: 1,
            architecture: "qwen3.5-a3b".into(),
            model: "Qwen/Qwen3.6-35B-A3B".into(),
            trunk_path: "/models/qwen.mq4r".into(),
            trunk_sha256: "trunk".into(),
            source_manifest_sha256: "source".into(),
            producer_git_commit: "commit".into(),
            split: "train".into(),
            hidden_dim: 2,
            recursive_steps: 3,
            hidden_dtype: "bf16-le".into(),
            record_checksum: "xxh3-64".into(),
            kv_mode: "q8".into(),
            state_quant: "q8".into(),
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("hipfire-mtp-data-{nonce}-{name}"))
    }

    #[test]
    fn round_trip_and_bf16_conversion() {
        let path = temp_path("roundtrip.rwf");
        let record = FeatureRecord {
            id: "sample#w0".into(),
            source_ordinal: 17,
            absolute_start: 9,
            hidden_rows: 2,
            tokens: vec![10, 11, 12, 13, 14],
            hidden_bf16: f32_to_bf16_bits(&[1.0, -2.0, 0.5, 4.0]),
        };
        let mut writer = AtomicShardWriter::create(&path, header()).unwrap();
        writer.write_record(&record).unwrap();
        let summary = writer.finish().unwrap();
        assert_eq!(summary.records, 1);
        assert_eq!(summary.hidden_rows, 2);

        let mut reader = ShardReader::open(&path).unwrap();
        assert_eq!(reader.header(), &header());
        assert_eq!(reader.read_record().unwrap(), Some(record));
        assert_eq!(reader.read_record().unwrap(), None);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_wrong_target_count() {
        let record = FeatureRecord {
            id: "bad".into(),
            source_ordinal: 0,
            absolute_start: 0,
            hidden_rows: 2,
            tokens: vec![1, 2],
            hidden_bf16: vec![0; 4],
        };
        assert!(record.validate(&header()).is_err());
    }

    #[test]
    fn partial_file_is_removed_if_writer_drops() {
        let path = temp_path("drop.rwf");
        let partial = path.parent().unwrap().join(format!(
            ".{}.partial",
            path.file_name().unwrap().to_string_lossy()
        ));
        {
            let _writer = AtomicShardWriter::create(&path, header()).unwrap();
            assert!(partial.exists());
        }
        assert!(!partial.exists());
        assert!(!path.exists());
    }

    #[test]
    fn windows_cover_both_ends_without_duplicates() {
        assert_eq!(window_starts(0, 128, 2), Vec::<usize>::new());
        assert_eq!(window_starts(64, 128, 2), vec![0]);
        assert_eq!(window_starts(384, 128, 2), vec![0, 256]);
        assert_eq!(window_starts(384, 128, 3), vec![0, 128, 256]);
        assert_eq!(window_starts(130, 128, 8), vec![0, 1, 2]);
    }
}
