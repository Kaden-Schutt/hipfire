// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Copy a name-selected subset of an HFQ overlay without re-quantizing it.
//
// Usage:
//   hfq_overlay_subset <output.hfq> <input.hfq> \
//     [--name <exact-name>]... [--suffix <name-suffix>]... \
//     [--rename <old-name>=<new-name>]...

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

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        usage(&args[0]);
    }
    let output = Path::new(&args[1]);
    let input = parse_hfq(Path::new(&args[2]))?;
    if output == input.path {
        return Err(invalid("output must differ from input"));
    }

    let mut exact = BTreeSet::<String>::new();
    let mut suffixes = Vec::<String>::new();
    let mut renames = BTreeMap::<String, String>::new();
    let mut index = 3;
    while index < args.len() {
        if index + 1 == args.len() {
            usage(&args[0]);
        }
        match args[index].as_str() {
            "--name" => {
                exact.insert(args[index + 1].clone());
            }
            "--suffix" => suffixes.push(args[index + 1].clone()),
            "--rename" => {
                let (old, new) = args[index + 1]
                    .split_once('=')
                    .ok_or_else(|| invalid("--rename requires <old-name>=<new-name>"))?;
                if old.is_empty() || new.is_empty() {
                    return Err(invalid("--rename names must not be empty"));
                }
                if renames.insert(old.to_owned(), new.to_owned()).is_some() {
                    return Err(invalid(format!("duplicate rename source {old:?}")));
                }
                exact.insert(old.to_owned());
            }
            other => return Err(invalid(format!("unknown selector {other:?}"))),
        }
        index += 2;
    }
    if exact.is_empty() && suffixes.is_empty() {
        return Err(invalid("at least one --name or --suffix is required"));
    }

    let mut selected = input
        .tensors
        .iter()
        .filter(|tensor| {
            exact.contains(&tensor.name)
                || suffixes.iter().any(|suffix| tensor.name.ends_with(suffix))
        })
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(invalid("selectors matched zero tensors"));
    }
    let selected_names = selected
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    let missing = exact
        .iter()
        .filter(|name| !selected_names.contains(name.as_str()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(invalid(format!(
            "exact selectors absent from input: {missing:?}"
        )));
    }
    for tensor in &mut selected {
        if let Some(new_name) = renames.get(&tensor.name) {
            tensor.name.clone_from(new_name);
        }
    }
    let output_names = selected
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    if output_names.len() != selected.len() {
        return Err(invalid("selectors/renames produce duplicate output names"));
    }

    let encoded_index = encode_index(&selected)?;
    let metadata_offset = 32_u64;
    let unaligned_data_offset =
        metadata_offset + input.metadata.len() as u64 + encoded_index.len() as u64;
    let data_offset = (unaligned_data_offset + 4095) & !4095;

    let mut output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    output_file.write_all(b"HFQM")?;
    output_file.write_all(&input.version.to_le_bytes())?;
    output_file.write_all(&input.arch_id.to_le_bytes())?;
    output_file.write_all(&(selected.len() as u32).to_le_bytes())?;
    output_file.write_all(&metadata_offset.to_le_bytes())?;
    output_file.write_all(&data_offset.to_le_bytes())?;
    output_file.write_all(&input.metadata)?;
    output_file.write_all(&encoded_index)?;
    output_file.write_all(&vec![0_u8; (data_offset - unaligned_data_offset) as usize])?;

    let mut input_file = File::open(&input.path)?;
    let mut buffer = vec![0_u8; 16 * 1024 * 1024];
    let mut payload_bytes = 0_u64;
    for tensor in &selected {
        input_file.seek(SeekFrom::Start(tensor.data_offset))?;
        let mut remaining = tensor.data_size;
        while remaining > 0 {
            let chunk = remaining.min(buffer.len() as u64) as usize;
            input_file.read_exact(&mut buffer[..chunk])?;
            output_file.write_all(&buffer[..chunk])?;
            remaining -= chunk as u64;
            payload_bytes += chunk as u64;
        }
    }
    output_file.sync_all()?;
    eprintln!(
        "wrote {}: {} byte-identical tensors ({:.2} MiB)",
        output.display(),
        selected.len(),
        payload_bytes as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn usage(program: &str) -> ! {
    eprintln!(
        "usage: {program} <output.hfq> <input.hfq> \
         [--name <exact-name>]... [--suffix <name-suffix>]... \
         [--rename <old-name>=<new-name>]..."
    );
    std::process::exit(2);
}

fn encode_index(tensors: &[TensorEntry]) -> io::Result<Vec<u8>> {
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for tensor in tensors {
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
    let indexed_count = u32::from_le_bytes(
        take(index, &mut 0_usize, 4)?
            .try_into()
            .expect("four bytes"),
    ) as usize;
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
        let name = String::from_utf8(take(index, &mut position, name_len)?.to_vec())
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
