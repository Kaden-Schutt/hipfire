//! Minimal GGUF file parser. Reads header, metadata, tensor info.
//! Memory-maps the file for zero-copy access to tensor data.

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32 (bytes: 47 47 55 46)

/// GGML tensor types.
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
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        // Only match the types we care about
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

    /// Block size for quantized types (number of elements per block).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            _ => 32,
        }
    }

    /// Bytes per block for quantized types.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,   // 2 (f16 scale) + 16 (32 x 4-bit)
            Self::Q4_1 => 20,   // 2 (f16 scale) + 2 (f16 min) + 16
            Self::Q5_0 => 22,   // 2 + 4 (high bits) + 16
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,   // 2 (f16 scale) + 32 (32 x 8-bit)
            Self::Q8_1 => 40,
            Self::Q2K => 256 / 16 * 2 + 256 / 4 + 2 + 2, // ~84
            Self::Q3K => 256 / 8 * 3 + 256 / 16 * 2 + 12 + 2, // ~110
            Self::Q4K => 2 + 2 + 12 + 128 + 64, // ~144
            Self::Q5K => 2 + 2 + 12 + 128 + 64 + 32, // ~176
            Self::Q6K => 128 + 64 + 16 + 2, // ~210
            Self::Q8K => 256 + 2 + 32, // ~290 (not commonly used)
            _ => 0,
        }
    }

    /// Bytes needed to store `n` elements.
    pub fn tensor_bytes(self, n: usize) -> usize {
        let bs = self.block_size();
        let nblocks = (n + bs - 1) / bs;
        nblocks * self.block_bytes()
    }
}

/// Metadata value types.
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

/// Info about a tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: usize, // offset from start of tensor data section
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.dtype.tensor_bytes(self.numel())
    }
}

/// Parsed GGUF file with memory-mapped tensor data.
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

        // Header
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

        // Metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let value = read_meta_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Tensor info
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
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
            });
        }

        // Tensor data starts at next alignment boundary
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

    /// Get raw bytes for a tensor.
    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.tensor_data_offset + info.offset;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get a metadata value.
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
            // Array
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
