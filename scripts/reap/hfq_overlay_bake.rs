// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Bake one HFQ overlay into a standalone HFQ without re-quantizing anything.
// Every replacement tensor is copied byte-for-byte from the overlay; every
// other tensor is copied byte-for-byte from the base. Metadata defaults to the
// base, but a recipe overlay may explicitly supply the output metadata.
//
// Usage:
//   hfq_overlay_bake <output.hfq> <base.hfq> <overlay.hfq> [expected-overrides]
//     [--metadata-from-overlay]

use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryInto;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct TensorEntry {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_offset: u64,
    data_size: u64,
}

struct ParsedHfq {
    path: PathBuf,
    version: u32,
    arch_id: u32,
    metadata: Vec<u8>,
    tensors: Vec<TensorEntry>,
}

#[derive(Clone, Copy)]
enum Source {
    Base,
    Overlay,
}

struct OutputTensor {
    source: Source,
    tensor: TensorEntry,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <output.hfq> <base.hfq> <overlay.hfq> [expected-overrides] \
             [--metadata-from-overlay]",
            args[0]
        );
        std::process::exit(2);
    }

    let output = Path::new(&args[1]);
    let base = parse_hfq(Path::new(&args[2]))?;
    let overlay = parse_hfq(Path::new(&args[3]))?;
    let mut expected_overrides = None;
    let mut metadata_from_overlay = false;
    let mut index = 4usize;
    if args
        .get(index)
        .is_some_and(|value| !value.starts_with("--"))
    {
        let value = &args[index];
        expected_overrides = Some(
            value
                .parse::<usize>()
                .map_err(|_| invalid(format!("invalid expected-overrides '{value}'")))?,
        );
        index += 1;
    }
    while index < args.len() {
        match args[index].as_str() {
            "--metadata-from-overlay" => metadata_from_overlay = true,
            other => return Err(invalid(format!("unknown option '{other}'"))),
        }
        index += 1;
    }

    if output == base.path || output == overlay.path {
        return Err(invalid("output must differ from base and overlay"));
    }
    if base.version != overlay.version || base.arch_id != overlay.arch_id {
        return Err(invalid(format!(
            "HFQ identity mismatch: base version/arch={}/{} overlay={}/{}",
            base.version, base.arch_id, overlay.version, overlay.arch_id
        )));
    }

    let base_names: BTreeSet<&str> = base.tensors.iter().map(|t| t.name.as_str()).collect();
    let overlay_by_name: BTreeMap<&str, &TensorEntry> = overlay
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    if overlay_by_name.len() != overlay.tensors.len() {
        return Err(invalid("overlay contains duplicate tensor names"));
    }
    for tensor in &overlay.tensors {
        if !base_names.contains(tensor.name.as_str()) {
            return Err(invalid(format!(
                "overlay tensor '{}' is absent from the base",
                tensor.name
            )));
        }
    }
    if let Some(expected) = expected_overrides {
        if overlay.tensors.len() != expected {
            return Err(invalid(format!(
                "overlay tensor count {} != expected {expected}",
                overlay.tensors.len()
            )));
        }
    }

    let mut tensors = Vec::with_capacity(base.tensors.len());
    let mut replacement_bytes = 0_u64;
    for base_tensor in &base.tensors {
        if let Some(overlay_tensor) = overlay_by_name.get(base_tensor.name.as_str()) {
            if overlay_tensor.shape != base_tensor.shape {
                return Err(invalid(format!(
                    "shape mismatch for '{}': base {:?}, overlay {:?}",
                    base_tensor.name, base_tensor.shape, overlay_tensor.shape
                )));
            }
            replacement_bytes += overlay_tensor.data_size;
            tensors.push(OutputTensor {
                source: Source::Overlay,
                tensor: (*overlay_tensor).clone(),
            });
        } else {
            tensors.push(OutputTensor {
                source: Source::Base,
                tensor: base_tensor.clone(),
            });
        }
    }

    let metadata = if metadata_from_overlay {
        overlay.metadata.clone()
    } else {
        base.metadata.clone()
    };

    let index = encode_index(&tensors)?;
    let metadata_offset = 32_u64;
    let unaligned_data_offset = metadata_offset + metadata.len() as u64 + index.len() as u64;
    let data_offset = (unaligned_data_offset + 4095) & !4095;

    let mut output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    output_file.write_all(b"HFQM")?;
    output_file.write_all(&base.version.to_le_bytes())?;
    output_file.write_all(&base.arch_id.to_le_bytes())?;
    output_file.write_all(&(tensors.len() as u32).to_le_bytes())?;
    output_file.write_all(&metadata_offset.to_le_bytes())?;
    output_file.write_all(&data_offset.to_le_bytes())?;
    output_file.write_all(&metadata)?;
    output_file.write_all(&index)?;
    let padding = (data_offset - unaligned_data_offset) as usize;
    output_file.write_all(&vec![0_u8; padding])?;

    let mut base_file = File::open(&base.path)?;
    let mut overlay_file = File::open(&overlay.path)?;
    let mut buffer = vec![0_u8; 16 * 1024 * 1024];
    let mut total = 0_u64;
    for (index, output_tensor) in tensors.iter().enumerate() {
        let source_file = match output_tensor.source {
            Source::Base => &mut base_file,
            Source::Overlay => &mut overlay_file,
        };
        copy_tensor(
            source_file,
            &mut output_file,
            &output_tensor.tensor,
            &mut buffer,
        )?;
        total += output_tensor.tensor.data_size;
        if (index + 1) % 1024 == 0 || index + 1 == tensors.len() {
            eprintln!(
                "baked {}/{} tensors ({:.2} GiB)",
                index + 1,
                tensors.len(),
                total as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
    output_file.sync_all()?;
    eprintln!(
        "wrote {}: {} tensors, {} byte-identical overlay replacements ({:.2} GiB)",
        output.display(),
        tensors.len(),
        overlay.tensors.len(),
        replacement_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn encode_index(tensors: &[OutputTensor]) -> io::Result<Vec<u8>> {
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for output_tensor in tensors {
        let tensor = &output_tensor.tensor;
        let name_len: u16 = tensor
            .name
            .len()
            .try_into()
            .map_err(|_| invalid(format!("tensor name too long: {}", tensor.name)))?;
        let dimensions: u8 = tensor
            .shape
            .len()
            .try_into()
            .map_err(|_| invalid(format!("too many dimensions: {}", tensor.name)))?;
        index.extend_from_slice(&name_len.to_le_bytes());
        index.extend_from_slice(tensor.name.as_bytes());
        index.push(tensor.quant_type);
        index.push(dimensions);
        for dimension in &tensor.shape {
            index.extend_from_slice(&dimension.to_le_bytes());
        }
        index.extend_from_slice(&tensor.group_size.to_le_bytes());
        index.extend_from_slice(&tensor.data_size.to_le_bytes());
    }
    Ok(index)
}

fn copy_tensor(
    source: &mut File,
    output: &mut File,
    tensor: &TensorEntry,
    buffer: &mut [u8],
) -> io::Result<()> {
    source.seek(SeekFrom::Start(tensor.data_offset))?;
    let mut remaining = tensor.data_size;
    while remaining > 0 {
        let chunk = remaining.min(buffer.len() as u64) as usize;
        source.read_exact(&mut buffer[..chunk])?;
        output.write_all(&buffer[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn parse_hfq(path: &Path) -> io::Result<ParsedHfq> {
    let mut file = File::open(path)?;
    let mut header = [0_u8; 32];
    file.read_exact(&mut header)?;
    if &header[..4] != b"HFQM" {
        return Err(invalid(format!("{}: bad HFQ magic", path.display())));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let tensor_count = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap());
    if data_offset < metadata_offset {
        return Err(invalid(format!(
            "{}: data offset precedes metadata",
            path.display()
        )));
    }

    let mut metadata_and_index = vec![0_u8; (data_offset - metadata_offset) as usize];
    file.seek(SeekFrom::Start(metadata_offset))?;
    file.read_exact(&mut metadata_and_index)?;
    let metadata_end = json_object_end(&metadata_and_index)
        .ok_or_else(|| invalid(format!("{}: metadata JSON not found", path.display())))?;
    let metadata = metadata_and_index[..metadata_end].to_vec();
    let index = &metadata_and_index[metadata_end..];
    if index.len() < 4 {
        return Err(invalid(format!("{}: missing HFQ index", path.display())));
    }
    let indexed_count = u32::from_le_bytes(index[..4].try_into().unwrap()) as usize;
    if indexed_count != tensor_count {
        return Err(invalid(format!(
            "{}: header/index tensor count mismatch {tensor_count}/{indexed_count}",
            path.display()
        )));
    }

    let mut position = 4_usize;
    let mut tensor_data_offset = data_offset;
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name_len = read_u16(index, &mut position)? as usize;
        let name_bytes = take(index, &mut position, name_len)?;
        let name = String::from_utf8(name_bytes.to_vec())
            .map_err(|_| invalid(format!("{}: non-UTF8 tensor name", path.display())))?;
        let quant_type = take(index, &mut position, 1)?[0];
        let dimensions = take(index, &mut position, 1)?[0] as usize;
        let mut shape = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            shape.push(read_u32(index, &mut position)?);
        }
        let group_size = read_u32(index, &mut position)?;
        let data_size = read_u64(index, &mut position)?;
        tensors.push(TensorEntry {
            name,
            quant_type,
            shape,
            group_size,
            data_offset: tensor_data_offset,
            data_size,
        });
        tensor_data_offset = tensor_data_offset
            .checked_add(data_size)
            .ok_or_else(|| invalid("tensor data offset overflow"))?;
    }

    Ok(ParsedHfq {
        path: path.to_path_buf(),
        version,
        arch_id,
        metadata,
        tensors,
    })
}

fn json_object_end(bytes: &[u8]) -> Option<usize> {
    let (mut depth, mut in_string, mut escaped) = (0_i32, false, false);
    for (index, &byte) in bytes.iter().enumerate() {
        if escaped {
            escaped = false;
        } else if in_string && byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            in_string = !in_string;
        } else if !in_string && byte == b'{' {
            depth += 1;
        } else if !in_string && byte == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(index + 1);
            }
        }
    }
    None
}

fn take<'a>(bytes: &'a [u8], position: &mut usize, len: usize) -> io::Result<&'a [u8]> {
    let end = position
        .checked_add(len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| invalid("truncated HFQ index"))?;
    let result = &bytes[*position..end];
    *position = end;
    Ok(result)
}

fn read_u16(bytes: &[u8], position: &mut usize) -> io::Result<u16> {
    Ok(u16::from_le_bytes(
        take(bytes, position, 2)?.try_into().unwrap(),
    ))
}

fn read_u32(bytes: &[u8], position: &mut usize) -> io::Result<u32> {
    Ok(u32::from_le_bytes(
        take(bytes, position, 4)?.try_into().unwrap(),
    ))
}

fn read_u64(bytes: &[u8], position: &mut usize) -> io::Result<u64> {
    Ok(u64::from_le_bytes(
        take(bytes, position, 8)?.try_into().unwrap(),
    ))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
