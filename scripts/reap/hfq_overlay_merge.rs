// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Deterministically merge disjoint HFQ overlays without re-quantizing them.
//
// Usage:
//   hfq_overlay_merge <output.hfq> <input-1.hfq> [input-2.hfq ...]

use std::collections::BTreeMap;
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

struct SourcedTensor {
    source: usize,
    tensor: TensorEntry,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <output.hfq> <input-1.hfq> <input-2.hfq> [more...]",
            args[0]
        );
        std::process::exit(2);
    }

    let output = Path::new(&args[1]);
    let inputs: Vec<ParsedHfq> = args[2..]
        .iter()
        .map(|path| parse_hfq(Path::new(path)))
        .collect::<io::Result<_>>()?;
    let first = &inputs[0];

    let mut by_name = BTreeMap::<String, SourcedTensor>::new();
    for (source, input) in inputs.iter().enumerate() {
        if input.version != first.version || input.arch_id != first.arch_id {
            return Err(invalid(format!(
                "{}: HFQ identity {}/{} differs from first input {}/{}",
                input.path.display(),
                input.version,
                input.arch_id,
                first.version,
                first.arch_id
            )));
        }
        for tensor in &input.tensors {
            if by_name
                .insert(
                    tensor.name.clone(),
                    SourcedTensor {
                        source,
                        tensor: tensor.clone(),
                    },
                )
                .is_some()
            {
                return Err(invalid(format!("duplicate tensor {}", tensor.name)));
            }
        }
    }

    let tensors: Vec<SourcedTensor> = by_name.into_values().collect();
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for sourced in &tensors {
        let tensor = &sourced.tensor;
        index.extend_from_slice(&(tensor.name.len() as u16).to_le_bytes());
        index.extend_from_slice(tensor.name.as_bytes());
        index.push(tensor.quant_type);
        index.push(tensor.shape.len() as u8);
        for dimension in &tensor.shape {
            index.extend_from_slice(&dimension.to_le_bytes());
        }
        index.extend_from_slice(&tensor.group_size.to_le_bytes());
        index.extend_from_slice(&tensor.data_size.to_le_bytes());
    }

    let metadata_offset = 32_u64;
    let unaligned_data_offset = metadata_offset + first.metadata.len() as u64 + index.len() as u64;
    let data_offset = (unaligned_data_offset + 4095) & !4095;

    let mut output_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output)?;
    output_file.write_all(b"HFQM")?;
    output_file.write_all(&first.version.to_le_bytes())?;
    output_file.write_all(&first.arch_id.to_le_bytes())?;
    output_file.write_all(&(tensors.len() as u32).to_le_bytes())?;
    output_file.write_all(&metadata_offset.to_le_bytes())?;
    output_file.write_all(&data_offset.to_le_bytes())?;
    output_file.write_all(&first.metadata)?;
    output_file.write_all(&index)?;
    let padding = (data_offset - unaligned_data_offset) as usize;
    output_file.write_all(&vec![0_u8; padding])?;

    let mut source_files: Vec<File> = inputs
        .iter()
        .map(|input| File::open(&input.path))
        .collect::<io::Result<_>>()?;
    let mut buffer = vec![0_u8; 16 * 1024 * 1024];
    let mut total = 0_u64;
    for (index, sourced) in tensors.iter().enumerate() {
        let tensor = &sourced.tensor;
        let source = &mut source_files[sourced.source];
        source.seek(SeekFrom::Start(tensor.data_offset))?;
        let mut remaining = tensor.data_size;
        while remaining > 0 {
            let chunk = remaining.min(buffer.len() as u64) as usize;
            source.read_exact(&mut buffer[..chunk])?;
            output_file.write_all(&buffer[..chunk])?;
            remaining -= chunk as u64;
            total += chunk as u64;
        }
        if (index + 1) % 32 == 0 || index + 1 == tensors.len() {
            eprintln!(
                "merged {}/{} tensors ({:.2} GiB)",
                index + 1,
                tensors.len(),
                total as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
    output_file.sync_all()?;
    eprintln!(
        "wrote {} tensors from {} overlays to {}",
        tensors.len(),
        inputs.len(),
        output.display()
    );
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
        tensor_data_offset += data_size;
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
