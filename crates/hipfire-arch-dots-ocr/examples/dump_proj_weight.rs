//! Tiny utility: dump `vision_tower.blocks.{block}.attn.proj.weight` from an
//! HFQ file to an F32 `.npy`. Used by the 2c-5d investigation to compare
//! our proj GEMM to HF's bf16 matmul on identical input.
//!
//! Usage:
//!   cargo run --release --example dump_proj_weight -p hipfire-arch-dots-ocr -- \
//!     --hfq ~/.hipfire/models/dots-ocr.q8.hfq \
//!     --block 1 \
//!     --out /tmp/proj_w_block_01.npy

use std::path::PathBuf;
use std::fs::File;
use std::io::Write;

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    let mut hfq_path: Option<PathBuf> = None;
    let mut block: Option<usize> = None;
    let mut out: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hfq" => { hfq_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--block" => { block = Some(argv[i + 1].parse()?); i += 2; }
            "--out" => { out = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            other => panic!("unknown arg: {other}"),
        }
    }
    let hfq_path = hfq_path.expect("--hfq required");
    let block = block.expect("--block required");
    let out = out.expect("--out required");
    let name = format!("vision_tower.blocks.{block}.attn.proj.weight");

    let hfq = HfqFile::open(&hfq_path)?;
    let (info, data) = hfq.tensor_data_vec(&name)
        .unwrap_or_else(|| panic!("tensor not found: {name}"));
    eprintln!("found: name={name} quant_type={} shape={:?} bytes={}",
        info.quant_type, info.shape, data.len());

    // shape on disk: [embed_dim, embed_dim] for proj. 1536x1536 typically.
    let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
    let n_elements: usize = shape.iter().product();

    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        qt => panic!("unsupported quant_type {qt} for proj_w"),
    };
    assert_eq!(f32_data.len(), n_elements, "element count mismatch");

    write_npy(&out, &f32_data, &shape)?;
    eprintln!("wrote {} bytes to {}", n_elements * 4 + 128, out.display());
    Ok(())
}

fn write_npy(path: &std::path::Path, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let mut shape_str = String::from("(");
    for (i, &s) in shape.iter().enumerate() {
        if i > 0 { shape_str.push_str(", "); }
        shape_str.push_str(&s.to_string());
    }
    if shape.len() == 1 { shape_str.push(','); }
    shape_str.push(')');
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
    let mut padded = header;
    while (10 + padded.len() + 1) % 16 != 0 { padded.push(' '); }
    padded.push('\n');
    let header_len = padded.len() as u16;
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1u8, 0u8])?;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(padded.as_bytes())?;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes)?;
    Ok(())
}
