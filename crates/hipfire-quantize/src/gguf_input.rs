// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal GGUF reader + dequant copied from `crates/engine/src/{gguf.rs,llama.rs}`.
//! Self-contained so hipfire-quantize doesn't pull engine's GPU dependency tree.
//! TODO: factor into a shared `gguf-codec` crate.

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 290,
        }
    }

    pub fn tensor_bytes(self, n: usize) -> usize {
        let bs = self.block_size();
        let nblocks = (n + bs - 1) / bs;
        nblocks * self.block_bytes()
    }
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::U32(v) => Some(*v),
            MetaValue::I32(v) => Some(*v as u32),
            MetaValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn byte_size(&self) -> usize {
        self.dtype.tensor_bytes(self.numel())
    }
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: usize,
    mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap[..]);

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: 0x{magic:08x}"),
            ));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version}"),
            ));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let value = read_meta_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let dtype_raw = cursor.read_u32::<LittleEndian>()?;
            let dtype = GgmlType::from_u32(dtype_raw).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown GGML type: {dtype_raw}"),
                )
            })?;
            let offset = cursor.read_u64::<LittleEndian>()? as usize;
            tensors.push(TensorInfo { name, shape, dtype, offset });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;

        let pos = cursor.position() as usize;
        let tensor_data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            tensor_data_offset,
            mmap,
        })
    }

    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.tensor_data_offset + info.offset;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }

    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta(key).and_then(|v| v.as_u32())
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta(key).and_then(|v| v.as_f32())
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta(key).and_then(|v| v.as_str())
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}")))
}

fn read_meta_value(cursor: &mut Cursor<&[u8]>) -> io::Result<MetaValue> {
    let vtype = cursor.read_u32::<LittleEndian>()?;
    read_typed_value(cursor, vtype)
}

fn read_typed_value(cursor: &mut Cursor<&[u8]>, vtype: u32) -> io::Result<MetaValue> {
    match vtype {
        0 => Ok(MetaValue::U8(cursor.read_u8()?)),
        1 => Ok(MetaValue::I8(cursor.read_i8()?)),
        2 => Ok(MetaValue::U16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(MetaValue::I16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(MetaValue::U32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(MetaValue::I32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(MetaValue::F32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetaValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(MetaValue::String(read_string(cursor)?)),
        9 => {
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_typed_value(cursor, elem_type)?);
            }
            Ok(MetaValue::Array(arr))
        }
        10 => Ok(MetaValue::U64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(MetaValue::I64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(MetaValue::F64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown metadata value type: {vtype}"),
        )),
    }
}

// ─── Dequant (copied from engine/src/llama.rs) ────────────────────────────

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    let exp32 = exp + (127 - 15);
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn dequant_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * 18;
        if off + 18 > data.len() {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for j in 0..16 {
            let byte = data[off + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            let idx = b * block_size + j * 2;
            if idx < n {
                out[idx] = lo as f32 * scale;
            }
            if idx + 1 < n {
                out[idx + 1] = hi as f32 * scale;
            }
        }
    }
    out
}

fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * 34;
        if off + 34 > data.len() {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for j in 0..32 {
            let q = data[off + 2 + j] as i8 as f32;
            let idx = b * block_size + j;
            if idx < n {
                out[idx] = q * scale;
            }
        }
    }
    out
}

fn dequant_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 144;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        let qdata = &data[off + 16..off + 16 + 128];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;
            for l in 0..32 {
                let byte = qdata[group * 32 + l];
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    out[idx_even] = (byte & 0x0F) as f32 * sc_even - m_even;
                }
                if idx_odd < n {
                    out[idx_odd] = ((byte >> 4) & 0x0F) as f32 * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

fn dequant_q5_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 176;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        // 12-byte packed scales/mins — same layout as Q4_K
        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        // 32 bytes of high bits (1 bit per element), then 128 bytes of low nibbles
        let qh = &data[off + 16..off + 48];
        let ql = &data[off + 48..off + 176];

        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;
            for l in 0..32 {
                let byte = ql[group * 32 + l];
                let hbit = ((qh[l] >> group) & 1) as u8;
                let hbit2 = ((qh[l] >> (group + 4)) & 1) as u8;
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    let q = ((byte & 0x0F) | (hbit << 4)) as f32;
                    out[idx_even] = q * sc_even - m_even;
                }
                if idx_odd < n {
                    let q = (((byte >> 4) & 0x0F) | (hbit2 << 4)) as f32;
                    out[idx_odd] = q * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

fn dequant_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 210;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }
        let mut ql = &data[off..off + 128];
        let mut qh = &data[off + 128..off + 192];
        let mut sc = &data[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([data[off + 208], data[off + 209]]));
        let base = b * block_size;
        for group in 0..2 {
            let y_off = base + group * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                let idx0 = y_off + l;
                let idx1 = y_off + l + 32;
                let idx2 = y_off + l + 64;
                let idx3 = y_off + l + 96;
                if idx0 < n { out[idx0] = d * sc[is] as i8 as f32 * q1 as f32; }
                if idx1 < n { out[idx1] = d * sc[is + 2] as i8 as f32 * q2 as f32; }
                if idx2 < n { out[idx2] = d * sc[is + 4] as i8 as f32 * q3 as f32; }
                if idx3 < n { out[idx3] = d * sc[is + 6] as i8 as f32 * q4 as f32; }
            }
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    out
}

/// Dispatcher: dequantize any supported tensor to f32. Panics on unsupported types.
pub fn tensor_to_f32(info: &TensorInfo, data: &[u8]) -> Vec<f32> {
    let n = info.numel();
    match info.dtype {
        GgmlType::F32 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(4).enumerate().take(n) {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        }
        GgmlType::F16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(2).enumerate().take(n) {
                out[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgmlType::BF16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(2).enumerate().take(n) {
                out[i] = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgmlType::Q4_0 => dequant_q4_0(data, n),
        GgmlType::Q8_0 => dequant_q8_0(data, n),
        GgmlType::Q4K => dequant_q4_k(data, n),
        GgmlType::Q5K => dequant_q5_k(data, n),
        GgmlType::Q6K => dequant_q6_k(data, n),
        other => panic!(
            "GGUF tensor type {:?} not implemented (tensor: {})",
            other, info.name
        ),
    }
}
