// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hipfire-native gfx94x Hessian builder for GPTQ-E8.
//!
//! Inputs are the `.acts` files emitted by DeepSeek4's real Hipfire forward
//! path under `HIPFIRE_DS4_DENSE_ACT_DIR`. Each file is `[u32 rows][u32 K]`
//! followed by row-major F32 activations. For every 256-channel slice this
//! tool computes `X^T X` with rocBLAS FP32 GEMM, which routes to MFMA on
//! MI300X, and writes the exact `E8H1` `.hblk` format consumed by
//! `hipfire-quantize --hessian-dir`.
//!
//! This is deliberately not a PyTorch/HuggingFace hook: both the activation
//! producer and Hessian consumer are Hipfire, and the only library operation
//! is the Gram GEMM over activations Hipfire actually executed.

use hip_bridge::{DeviceBuffer, HipRuntime, Rocblas, RocblasDatatype, RocblasOperation, Stream};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

struct ActsFile {
    bytes: Vec<u8>,
    rows: usize,
    k: usize,
}

impl ActsFile {
    fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read activation dump {}: {error}", path.display()))?;
        if bytes.len() < 8 {
            return Err(format!("activation dump is too small: {}", path.display()));
        }
        let rows = read_u32(&bytes, 0) as usize;
        let k = read_u32(&bytes, 4) as usize;
        if k == 0 || k % 256 != 0 {
            return Err(format!(
                "activation K={k} is not a positive multiple of 256: {}",
                path.display()
            ));
        }
        let payload_len = rows
            .checked_mul(k)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| format!("activation dimensions overflow: {}", path.display()))?;
        let expected_len = 8usize
            .checked_add(payload_len)
            .ok_or_else(|| format!("activation length overflow: {}", path.display()))?;
        if bytes.len() != expected_len {
            return Err(format!(
                "activation length mismatch for {}: {} != {expected_len}",
                path.display(),
                bytes.len()
            ));
        }
        Ok(Self { bytes, rows, k })
    }

    fn payload(&self) -> &[u8] {
        &self.bytes[8..]
    }
}

enum InputMode {
    Tensor { acts: Vec<PathBuf>, name: String },
    Directory(PathBuf),
}

struct Args {
    input: InputMode,
    out_dir: PathBuf,
    device: i32,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|index| args.get(index + 1).cloned())
    };
    let acts: Vec<PathBuf> = args
        .windows(2)
        .filter(|pair| pair[0] == "--acts")
        .map(|pair| PathBuf::from(&pair[1]))
        .collect();
    let acts_dir = value("--acts-dir").map(PathBuf::from);
    let name = value("--name");
    let input = match (acts.is_empty(), acts_dir, name) {
        (false, None, Some(name)) => InputMode::Tensor { acts, name },
        (true, Some(directory), None) => InputMode::Directory(directory),
        _ => {
            return Err(
                "use either --acts <path> [--acts <path> ...] with --name, or --acts-dir <dir>"
                    .to_string(),
            )
        }
    };
    let out_dir = value("--out-dir")
        .map(PathBuf::from)
        .ok_or_else(|| "--out-dir <directory> is required".to_string())?;
    let device = value("--device")
        .as_deref()
        .unwrap_or("0")
        .parse::<i32>()
        .map_err(|error| format!("invalid --device: {error}"))?;
    Ok(Args {
        input,
        out_dir,
        device,
    })
}

fn directory_inputs(directory: &Path) -> Result<Vec<(String, Vec<PathBuf>)>, String> {
    let mut inputs = Vec::new();
    for entry in std::fs::read_dir(directory)
        .map_err(|error| format!("read activation directory {}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| {
            format!(
                "read activation directory entry {}: {error}",
                directory.display()
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("acts") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("non-UTF8 activation filename: {}", path.display()))?
            .to_string();
        inputs.push((name, vec![path]));
    }
    inputs.sort_by(|left, right| left.0.cmp(&right.0));
    if inputs.is_empty() {
        return Err(format!("no .acts files found in {}", directory.display()));
    }
    Ok(inputs)
}

fn sanitized_path(out_dir: &Path, tensor_name: &str) -> PathBuf {
    let key = tensor_name.replace(['/', '\\'], "_").replace("..", "_");
    out_dir.join(format!("{key}.hblk"))
}

fn validate_blocks(blocks: &[u8], n_blocks: usize) -> Result<(f32, f32, f32), String> {
    let mut min_diag = f32::INFINITY;
    let mut max_diag = f32::NEG_INFINITY;
    let mut max_asym = 0.0f32;
    for block in 0..n_blocks {
        let base = block * 256 * 256 * 4;
        let value = |i: usize, j: usize| {
            let offset = base + (i * 256 + j) * 4;
            f32::from_le_bytes(blocks[offset..offset + 4].try_into().expect("four bytes"))
        };
        for i in 0..256 {
            let diagonal = value(i, i);
            if !diagonal.is_finite() {
                return Err(format!("non-finite diagonal at block {block}, channel {i}"));
            }
            min_diag = min_diag.min(diagonal);
            max_diag = max_diag.max(diagonal);
            for j in (i + 1)..256 {
                let upper = value(i, j);
                let lower = value(j, i);
                if !upper.is_finite() || !lower.is_finite() {
                    return Err(format!(
                        "non-finite Hessian entry at block {block}, ({i},{j})"
                    ));
                }
                max_asym = max_asym.max((upper - lower).abs());
            }
        }
    }
    if min_diag < -1.0e-5 * max_diag.max(1.0) {
        return Err(format!(
            "materially negative Hessian diagonal: min={min_diag:.6e}, max={max_diag:.6e}"
        ));
    }
    Ok((min_diag, max_diag, max_asym))
}

/// rocBLAS evaluates the two triangles as independent dot products. They are
/// mathematically identical, but long/high-energy rows can differ by a few
/// FP32 ulps because the internal reduction order is not guaranteed to match.
/// Canonicalize once on the host so E8H1 matches Hipfire's f64 CPU collector
/// contract and downstream tools receive an exactly symmetric matrix.
fn symmetrize_blocks(blocks: &mut [u8], n_blocks: usize) -> f32 {
    let mut max_input_asym = 0.0f32;
    for block in 0..n_blocks {
        let base = block * 256 * 256 * 4;
        for i in 0..256 {
            for j in (i + 1)..256 {
                let upper_offset = base + (i * 256 + j) * 4;
                let lower_offset = base + (j * 256 + i) * 4;
                let upper = f32::from_le_bytes(
                    blocks[upper_offset..upper_offset + 4]
                        .try_into()
                        .expect("four bytes"),
                );
                let lower = f32::from_le_bytes(
                    blocks[lower_offset..lower_offset + 4]
                        .try_into()
                        .expect("four bytes"),
                );
                max_input_asym = max_input_asym.max((upper - lower).abs());
                let average = (0.5 * (upper as f64 + lower as f64)) as f32;
                blocks[upper_offset..upper_offset + 4].copy_from_slice(&average.to_le_bytes());
                blocks[lower_offset..lower_offset + 4].copy_from_slice(&average.to_le_bytes());
            }
        }
    }
    max_input_asym
}

fn write_hblk(
    out_dir: &Path,
    tensor_name: &str,
    k: usize,
    rows: usize,
    blocks: &mut [u8],
) -> Result<(), String> {
    let n_blocks = k / 256;
    let expected = n_blocks * 256 * 256 * 4;
    if blocks.len() != expected {
        return Err(format!(
            "internal Hessian length mismatch: {} != {expected}",
            blocks.len()
        ));
    }
    let max_input_asym = symmetrize_blocks(blocks, n_blocks);
    let (min_diag, max_diag, max_asym) = validate_blocks(blocks, n_blocks)?;
    std::fs::create_dir_all(out_dir)
        .map_err(|error| format!("create {}: {error}", out_dir.display()))?;
    let path = sanitized_path(out_dir, tensor_name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    file.write_all(&0x45_38_48_31u32.to_le_bytes())
        .and_then(|_| file.write_all(&(n_blocks as u32).to_le_bytes()))
        .and_then(|_| file.write_all(&(k as u32).to_le_bytes()))
        .and_then(|_| file.write_all(blocks))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    eprintln!(
        "wrote {}: rows={rows} K={k} blocks={n_blocks} diag=[{min_diag:.6e},{max_diag:.6e}] input_asym={max_input_asym:.6e} max_asym={max_asym:.6e}",
        path.display()
    );
    Ok(())
}

fn collect_with_engine(
    hip: &HipRuntime,
    stream: &Stream,
    rocblas: &Rocblas,
    acts_paths: &[PathBuf],
    tensor_name: &str,
    out_dir: &Path,
) -> Result<(), String> {
    let alpha = 1.0f32;
    let beta = 1.0f32;
    let mut output: Option<DeviceBuffer> = None;
    let mut expected_k = None;
    let mut total_rows = 0usize;

    for path in acts_paths {
        let acts = ActsFile::read(path)?;
        if acts.rows == 0 {
            eprintln!("skip zero-row activation dump {}", path.display());
            continue;
        }
        if acts.rows > i32::MAX as usize || acts.k > i32::MAX as usize {
            return Err(format!(
                "activation dimensions exceed rocBLAS limits: {}",
                path.display()
            ));
        }
        if let Some(k) = expected_k {
            if acts.k != k {
                return Err(format!(
                    "K mismatch for {}: {} != {k}",
                    path.display(),
                    acts.k
                ));
            }
        } else {
            expected_k = Some(acts.k);
            let output_bytes = (acts.k / 256) * 256 * 256 * 4;
            let buffer = hip
                .malloc(output_bytes)
                .map_err(|error| format!("allocate Hessian buffer: {error}"))?;
            hip.memset(&buffer, 0, output_bytes)
                .map_err(|error| format!("zero Hessian buffer: {error}"))?;
            output = Some(buffer);
        }

        let input = hip
            .malloc(acts.payload().len())
            .map_err(|error| format!("allocate activation buffer: {error}"))?;
        hip.memcpy_htod(&input, acts.payload())
            .map_err(|error| format!("upload {}: {error}", path.display()))?;
        let output_ref = output.as_ref().expect("allocated with first input");
        for block in 0..(acts.k / 256) {
            // Row-major X[N,K] is a column-major X^T[256,N] view at this
            // pointer with lda=K. A*A^T is the desired block Gram matrix.
            // Since the result is symmetric, column-major storage is bytewise
            // identical to the row-major .hblk contract.
            let x_block = unsafe {
                input
                    .as_ptr()
                    .cast::<u8>()
                    .add(block * 256 * 4)
                    .cast::<c_void>()
            };
            let h_block = unsafe {
                output_ref
                    .as_ptr()
                    .cast::<u8>()
                    .add(block * 256 * 256 * 4)
                    .cast::<c_void>()
            };
            unsafe {
                rocblas.gemm_ex(
                    RocblasOperation::None,
                    RocblasOperation::Transpose,
                    256,
                    256,
                    acts.rows as i32,
                    (&alpha as *const f32).cast::<c_void>(),
                    x_block,
                    RocblasDatatype::F32,
                    acts.k as i32,
                    x_block,
                    RocblasDatatype::F32,
                    acts.k as i32,
                    (&beta as *const f32).cast::<c_void>(),
                    h_block,
                    RocblasDatatype::F32,
                    256,
                    h_block,
                    RocblasDatatype::F32,
                    256,
                    RocblasDatatype::F32,
                )
            }
            .map_err(|error| {
                format!("rocBLAS Gram block {block} for {}: {error}", path.display())
            })?;
        }
        hip.stream_synchronize(stream)
            .map_err(|error| format!("synchronize {}: {error}", path.display()))?;
        hip.free(input)
            .map_err(|error| format!("free activation buffer: {error}"))?;
        total_rows += acts.rows;
        eprintln!(
            "accumulated {} rows from {} on gfx94x rocBLAS",
            acts.rows,
            path.display()
        );
    }

    let k = expected_k.ok_or_else(|| "all activation dumps had zero rows".to_string())?;
    let output = output.expect("allocated with non-empty input");
    let mut blocks = vec![0u8; output.size()];
    hip.memcpy_dtoh(&mut blocks, &output)
        .map_err(|error| format!("download Hessian: {error}"))?;
    write_hblk(out_dir, tensor_name, k, total_rows, &mut blocks)?;
    hip.free(output)
        .map_err(|error| format!("free Hessian buffer: {error}"))?;
    Ok(())
}

#[cfg(test)]
fn collect(
    acts_paths: &[PathBuf],
    tensor_name: &str,
    out_dir: &Path,
    device: i32,
) -> Result<(), String> {
    let hip = HipRuntime::load().map_err(|error| format!("load HIP runtime: {error}"))?;
    hip.set_device(device)
        .map_err(|error| format!("select HIP device {device}: {error}"))?;
    let stream = hip
        .stream_create()
        .map_err(|error| format!("create HIP stream: {error}"))?;
    let rocblas = Rocblas::load().map_err(|error| format!("load rocBLAS: {error}"))?;
    rocblas
        .set_stream(&stream)
        .map_err(|error| format!("bind rocBLAS stream: {error}"))?;
    let result = collect_with_engine(&hip, &stream, &rocblas, acts_paths, tensor_name, out_dir);
    drop(rocblas);
    let destroy_result = hip
        .stream_destroy(stream)
        .map_err(|error| format!("destroy HIP stream: {error}"));
    result.and(destroy_result)
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!(
            "usage: collect_e8_hessian_rocblas \
             (--acts <file> [--acts <file> ...] --name <tensor-name> | \
             --acts-dir <directory>) --out-dir <directory> [--device 0]"
        );
        std::process::exit(2);
    });
    let inputs = match args.input {
        InputMode::Tensor { acts, name } => vec![(name, acts)],
        InputMode::Directory(directory) => directory_inputs(&directory).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(2);
        }),
    };
    let result: Result<(), String> = (|| {
        let hip = HipRuntime::load().map_err(|error| format!("load HIP runtime: {error}"))?;
        hip.set_device(args.device)
            .map_err(|error| format!("select HIP device {}: {error}", args.device))?;
        let stream = hip
            .stream_create()
            .map_err(|error| format!("create HIP stream: {error}"))?;
        let rocblas = Rocblas::load().map_err(|error| format!("load rocBLAS: {error}"))?;
        rocblas
            .set_stream(&stream)
            .map_err(|error| format!("bind rocBLAS stream: {error}"))?;
        let collect_result = inputs.iter().try_for_each(|(name, acts)| {
            collect_with_engine(&hip, &stream, &rocblas, acts, name, &args.out_dir)
        });
        drop(rocblas);
        let destroy_result = hip
            .stream_destroy(stream)
            .map_err(|error| format!("destroy HIP stream: {error}"));
        collect_result.and(destroy_result)?;
        eprintln!("completed {} Hipfire-native Hessian tensors", inputs.len());
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hblk_value(bytes: &[u8], block: usize, i: usize, j: usize) -> f32 {
        let offset = 12 + (block * 256 * 256 + i * 256 + j) * 4;
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn host_canonicalization_produces_exact_symmetry() {
        let mut block = vec![0u8; 256 * 256 * 4];
        let upper = (3.0f32).to_le_bytes();
        let lower = (5.0f32).to_le_bytes();
        let upper_offset = (7 * 256 + 19) * 4;
        let lower_offset = (19 * 256 + 7) * 4;
        block[upper_offset..upper_offset + 4].copy_from_slice(&upper);
        block[lower_offset..lower_offset + 4].copy_from_slice(&lower);

        assert_eq!(symmetrize_blocks(&mut block, 1), 2.0);
        let stats = validate_blocks(&block, 1).unwrap();
        assert_eq!(stats.2, 0.0);
        let canonical_upper =
            f32::from_le_bytes(block[upper_offset..upper_offset + 4].try_into().unwrap());
        let canonical_lower =
            f32::from_le_bytes(block[lower_offset..lower_offset + 4].try_into().unwrap());
        assert_eq!(canonical_upper, 4.0);
        assert_eq!(canonical_lower, canonical_upper);
    }

    #[test]
    #[ignore = "requires a HIP device and rocBLAS"]
    fn gfx94x_gram_matches_host_reference_and_accumulates_inputs() {
        let root =
            std::env::temp_dir().join(format!("hipfire_e8_hessian_rocblas_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let acts_path = root.join("synthetic.acts");
        let rows = 5usize;
        let k = 512usize;
        let values: Vec<f32> = (0..rows * k)
            .map(|index| {
                let row = index / k;
                let column = index % k;
                ((row * 17 + column * 7) as f32 * 0.003).sin()
            })
            .collect();
        let mut acts = Vec::with_capacity(8 + values.len() * 4);
        acts.extend_from_slice(&(rows as u32).to_le_bytes());
        acts.extend_from_slice(&(k as u32).to_le_bytes());
        for value in &values {
            acts.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&acts_path, acts).unwrap();

        // Feed the same corpus twice to exercise beta=1 accumulation.
        collect(
            &[acts_path.clone(), acts_path.clone()],
            "layers.0.test.weight",
            &root,
            0,
        )
        .unwrap();
        let hblk = std::fs::read(root.join("layers.0.test.weight.hblk")).unwrap();
        assert_eq!(read_u32(&hblk, 0), 0x45_38_48_31);
        assert_eq!(read_u32(&hblk, 4), 2);
        assert_eq!(read_u32(&hblk, 8), 512);

        for &(block, i, j) in &[(0, 3, 5), (0, 190, 17), (1, 7, 244)] {
            let column_i = block * 256 + i;
            let column_j = block * 256 + j;
            let expected = 2.0
                * (0..rows)
                    .map(|row| values[row * k + column_i] * values[row * k + column_j])
                    .sum::<f32>();
            let actual = hblk_value(&hblk, block, i, j);
            let tolerance = 2.0e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "block={block} ({i},{j}): {actual} != {expected}"
            );
            assert_eq!(actual, hblk_value(&hblk, block, j, i));
        }

        std::fs::remove_dir_all(root).ok();
    }
}
