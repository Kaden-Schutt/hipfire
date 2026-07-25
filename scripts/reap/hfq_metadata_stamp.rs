// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Add one top-level JSON metadata field to an HFQ while copying the tensor
// index and every payload byte-for-byte.
//
// Usage:
//   hfq_metadata_stamp <output.hfq> <input.hfq> <field-name> <json-value>

use std::convert::TryInto;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: {} <output.hfq> <input.hfq> <field-name> <json-value>",
            args[0]
        );
        std::process::exit(2);
    }
    let output = Path::new(&args[1]);
    let input = Path::new(&args[2]);
    let field = &args[3];
    let json_value = args[4].as_bytes();
    if output == input {
        return Err(invalid("output must differ from input"));
    }
    if field.is_empty()
        || !field
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(invalid(format!("unsafe metadata field name {field:?}")));
    }
    validate_json_value(json_value)?;

    let mut input_file = File::open(input)?;
    let mut header = [0_u8; 32];
    input_file.read_exact(&mut header)?;
    if &header[..4] != b"HFQM" {
        return Err(invalid(format!("{}: bad HFQ magic", input.display())));
    }
    let tensor_count = u32::from_le_bytes(header[12..16].try_into().unwrap());
    let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let old_data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap());
    if metadata_offset != 32 || old_data_offset < metadata_offset {
        return Err(invalid(format!(
            "{}: unsupported metadata/data offsets {metadata_offset}/{old_data_offset}",
            input.display()
        )));
    }

    let mut metadata_and_index = vec![0_u8; (old_data_offset - metadata_offset) as usize];
    input_file.seek(SeekFrom::Start(metadata_offset))?;
    input_file.read_exact(&mut metadata_and_index)?;
    let metadata_end = json_object_end(&metadata_and_index)
        .ok_or_else(|| invalid(format!("{}: metadata JSON not found", input.display())))?;
    let metadata = &metadata_and_index[..metadata_end];
    let index = &metadata_and_index[metadata_end..];
    if index.len() < 4 {
        return Err(invalid(format!(
            "{}: missing tensor index",
            input.display()
        )));
    }
    let indexed_count = u32::from_le_bytes(index[..4].try_into().unwrap());
    if indexed_count != tensor_count {
        return Err(invalid(format!(
            "{}: header/index tensor count mismatch {tensor_count}/{indexed_count}",
            input.display()
        )));
    }
    let existing_key = format!("\"{field}\"");
    if metadata
        .windows(existing_key.len())
        .any(|window| window == existing_key.as_bytes())
    {
        return Err(invalid(format!(
            "{}: metadata field {field:?} already exists",
            input.display()
        )));
    }

    let mut stamped_metadata =
        Vec::with_capacity(metadata.len() + field.len() + json_value.len() + 5);
    stamped_metadata.extend_from_slice(&metadata[..metadata.len() - 1]);
    stamped_metadata.extend_from_slice(b",\"");
    stamped_metadata.extend_from_slice(field.as_bytes());
    stamped_metadata.extend_from_slice(b"\":");
    stamped_metadata.extend_from_slice(json_value);
    stamped_metadata.push(b'}');

    let unaligned_data_offset =
        metadata_offset + stamped_metadata.len() as u64 + index.len() as u64;
    let new_data_offset = (unaligned_data_offset + 4095) & !4095;
    let mut output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    header[24..32].copy_from_slice(&new_data_offset.to_le_bytes());
    output_file.write_all(&header)?;
    output_file.write_all(&stamped_metadata)?;
    output_file.write_all(index)?;
    output_file.write_all(&vec![
        0_u8;
        (new_data_offset - unaligned_data_offset) as usize
    ])?;

    input_file.seek(SeekFrom::Start(old_data_offset))?;
    let mut buffer = vec![0_u8; 16 * 1024 * 1024];
    let mut payload_bytes = 0_u64;
    loop {
        let read = input_file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output_file.write_all(&buffer[..read])?;
        payload_bytes += read as u64;
    }
    output_file.sync_all()?;
    eprintln!(
        "wrote {}: field={field:?}, tensors={tensor_count}, payload={:.2} GiB copied byte-for-byte",
        output.display(),
        payload_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn validate_json_value(bytes: &[u8]) -> io::Result<()> {
    let first = bytes
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .ok_or_else(|| invalid("JSON value is empty"))?;
    let last = bytes
        .iter()
        .copied()
        .rev()
        .find(|byte| !byte.is_ascii_whitespace())
        .ok_or_else(|| invalid("JSON value is empty"))?;
    let structurally_plausible = matches!(
        (first, last),
        (b'{', b'}') | (b'[', b']') | (b'"', b'"') | (b't', b'e') | (b'f', b'e') | (b'n', b'l')
    ) || first == b'-'
        || first.is_ascii_digit();
    if !structurally_plausible {
        return Err(invalid("JSON value has an unsupported outer shape"));
    }
    Ok(())
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

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
