// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — collect_e8_hessian: build per-256-block XX^T Hessians for GPTQ-E8.
//
// Captures the per-(tensor,expert) Hessian the LDLQ-on-E8 quantizer consumes.
// The Hessian is BLOCK-DIAGONAL-256 (FWHT decorrelates across 256-blocks, so
// the quantizer only needs the K/256 diagonal 256×256 blocks of XX^T), keyed
// by full safetensors tensor name (+ expert idx baked into the name).
//
// CONTRACT (matches main.rs::load_hessian_blocks):
//   output file `<dir>/<sanitized_name>.hblk`:
//     [u32 magic=0x45384831][u32 n_blocks=K/256][u32 K][f32 ... n_blocks*256*256]
//   block b holds XX^T over input channels [b*256 .. b*256+256), accumulated
//   over the RAW PRE-ROTATION activation rows x_t routed to this tensor/expert.
//
// PRECISION: H is precision-robust — capture from a Q8/bf16 forward (NOT the
// 130G f32 oracle). n_tokens divisor is irrelevant (damping uses mean(diag),
// scale-invariant), so we store the UNNORMALIZED Σ_t x_t x_t^T.
//
// INPUT (this CLI, the simplest disjoint interface for Validate's forward):
//   --acts <file>   raw activation dump: [u32 n_rows][u32 K][f32 rows row-major]
//                   (repeat to combine disjoint calibration corpora).
//   --name <tname>  full safetensors tensor name (the loader key).
//   --out-dir <dir> where to write <name>.hblk.
//
// For an in-process capture (no dump round-trip), Validate can instead call
// `BlockHessian::accumulate_row` per token directly and `write_hblk` at drain.

use std::path::Path;

/// Per-256-block XX^T accumulator for ONE weight tensor (one expert).
pub struct BlockHessian {
    pub k: usize,
    pub n_blocks: usize,
    /// n_blocks × (256*256) f64 accumulators (row-major within each block).
    pub blocks: Vec<Vec<f64>>,
    pub n_tokens: u64,
}

impl BlockHessian {
    pub fn new(k: usize) -> Self {
        assert!(k % 256 == 0, "K={k} must be divisible by 256");
        let n_blocks = k / 256;
        BlockHessian {
            k,
            n_blocks,
            blocks: (0..n_blocks).map(|_| vec![0.0f64; 256 * 256]).collect(),
            n_tokens: 0,
        }
    }

    /// Accumulate one pre-rotation activation row x[0..K] into the per-256-block
    /// diagonal XX^T (block b += x_b x_b^T over its 256 channels).
    pub fn accumulate_row(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len(), self.k);
        for b in 0..self.n_blocks {
            let xb = &x[b * 256..b * 256 + 256];
            let acc = &mut self.blocks[b];
            for i in 0..256 {
                let xi = xb[i] as f64;
                if xi == 0.0 {
                    continue;
                }
                let row = &mut acc[i * 256..i * 256 + 256];
                for j in 0..256 {
                    row[j] += xi * xb[j] as f64;
                }
            }
        }
        self.n_tokens += 1;
    }

    /// Serialize to the `.hblk` format consumed by main.rs::load_hessian_blocks.
    pub fn write_hblk(&self, dir: &Path, tensor_name: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let key = tensor_name.replace(['/', '\\'], "_").replace("..", "_");
        let path = dir.join(format!("{key}.hblk"));
        let mut buf = Vec::with_capacity(12 + self.n_blocks * 256 * 256 * 4);
        buf.extend_from_slice(&0x45_38_48_31u32.to_le_bytes()); // "E8H1"
        buf.extend_from_slice(&(self.n_blocks as u32).to_le_bytes());
        buf.extend_from_slice(&(self.k as u32).to_le_bytes());
        for b in 0..self.n_blocks {
            for &v in &self.blocks[b] {
                buf.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, &buf)?;
        eprintln!(
            "wrote {} ({} blocks, K={}, {} tokens)",
            path.display(),
            self.n_blocks,
            self.k,
            self.n_tokens
        );
        Ok(())
    }
}

fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| -> Option<String> {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let acts: Vec<String> = args
        .windows(2)
        .filter(|pair| pair[0] == "--acts")
        .map(|pair| pair[1].clone())
        .collect();
    if acts.is_empty() {
        eprintln!(
            "usage: collect_e8_hessian --acts <dump> [--acts <dump> ...] \
             --name <tensor> --out-dir <dir>\n\
             dump format: [u32 n_rows][u32 K][f32 rows row-major]"
        );
        std::process::exit(2);
    }
    let name = arg("--name").unwrap_or_else(|| {
        eprintln!("error: --name <full-safetensors-name> required");
        std::process::exit(2);
    });
    let out_dir = arg("--out-dir").unwrap_or_else(|| "/mnt/vol/e8-hessians".to_string());

    let mut h: Option<BlockHessian> = None;
    let mut total_rows = 0usize;
    for acts_path in &acts {
        let bytes = std::fs::read(acts_path).unwrap_or_else(|e| {
            eprintln!("error: cannot read --acts {acts_path}: {e}");
            std::process::exit(1);
        });
        if bytes.len() < 8 {
            eprintln!("error: activation dump too small: {acts_path}");
            std::process::exit(1);
        }
        let n_rows = read_u32(&bytes, 0) as usize;
        let k = read_u32(&bytes, 4) as usize;
        if k % 256 != 0 {
            eprintln!("error: K={k} not divisible by 256");
            std::process::exit(1);
        }
        let want = 8 + n_rows * k * 4;
        if bytes.len() != want {
            eprintln!(
                "error: activation dump size mismatch for {acts_path} ({} != {want})",
                bytes.len()
            );
            std::process::exit(1);
        }
        let acc = h.get_or_insert_with(|| BlockHessian::new(k));
        if acc.k != k {
            eprintln!(
                "error: activation K mismatch for {acts_path}: {} != {}",
                k, acc.k
            );
            std::process::exit(1);
        }
        let mut row = vec![0.0f32; k];
        let mut off = 8usize;
        for _ in 0..n_rows {
            for value in &mut row {
                *value = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                off += 4;
            }
            acc.accumulate_row(&row);
        }
        total_rows += n_rows;
        eprintln!("accumulated {n_rows} rows from {acts_path}");
    }
    let h = h.expect("non-empty --acts set");
    eprintln!(
        "combined {total_rows} rows from {} activation dump(s)",
        acts.len()
    );
    h.write_hblk(Path::new(&out_dir), &name)
        .unwrap_or_else(|e| {
            eprintln!("error: write_hblk failed: {e}");
            std::process::exit(1);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_and_roundtrip() {
        // K = 512 (2 blocks). Accumulate a few rows, write, read back, verify
        // block 0 equals the hand-computed Σ x_i x_j over channels [0,256).
        let k = 512usize;
        let mut h = BlockHessian::new(k);
        let rows = [
            (0..k)
                .map(|c| (c as f32 * 0.001).sin())
                .collect::<Vec<f32>>(),
            (0..k)
                .map(|c| (c as f32 * 0.002).cos())
                .collect::<Vec<f32>>(),
            (0..k).map(|c| ((c % 7) as f32) - 3.0).collect::<Vec<f32>>(),
        ];
        for r in &rows {
            h.accumulate_row(r);
        }
        // hand reference for block 0, entry (3,5)
        let mut ref35 = 0.0f64;
        for r in &rows {
            ref35 += r[3] as f64 * r[5] as f64;
        }
        let got = h.blocks[0][3 * 256 + 5];
        assert!(
            (got - ref35).abs() < 1e-9,
            "XX^T(3,5) mismatch: {got} vs {ref35}"
        );

        // symmetry
        assert!((h.blocks[0][5 * 256 + 3] - h.blocks[0][3 * 256 + 5]).abs() < 1e-9);

        // write + read back via the same magic/layout the loader uses
        let tmp = std::env::temp_dir().join(format!("e8h_test_{}", std::process::id()));
        h.write_hblk(&tmp, "model.test.weight").unwrap();
        let path = tmp.join("model.test.weight.hblk");
        let b = std::fs::read(&path).unwrap();
        assert_eq!(read_u32(&b, 0), 0x45_38_48_31);
        assert_eq!(read_u32(&b, 4) as usize, 2);
        assert_eq!(read_u32(&b, 8) as usize, 512);
        // first f32 = block0[0,0]
        let v00 = f32::from_le_bytes([b[12], b[13], b[14], b[15]]) as f64;
        assert!((v00 - h.blocks[0][0]).abs() < 1e-3 * (h.blocks[0][0].abs() + 1.0));
        std::fs::remove_file(&path).ok();
    }

    /// Python `reference_gptq/formats.py` E8H1 writer → Rust header/layout reader.
    #[test]
    fn python_fixture_e8h1_roundtrip() {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("reference_gptq/fixtures/layers.0.mlp.experts.3.gate_up_proj.weight.hblk");
        assert!(path.is_file(), "missing fixture {path:?}");
        let b = std::fs::read(&path).unwrap();
        assert_eq!(read_u32(&b, 0), 0x45_38_48_31, "magic");
        assert_eq!(read_u32(&b, 4), 2, "n_blocks");
        assert_eq!(read_u32(&b, 8), 512, "K");
        assert_eq!(b.len(), 12 + 2 * 256 * 256 * 4);

        // entry block1[3,5] from make_fixtures.py meta
        let mut meta_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        meta_path.push("reference_gptq/fixtures/e8h1_meta.txt");
        let meta = std::fs::read_to_string(&meta_path).unwrap();
        let mut expect35 = None;
        for line in meta.lines() {
            if let Some(v) = line.strip_prefix("entry_b1_3_5=") {
                expect35 = Some(v.parse::<f64>().unwrap());
            }
        }
        let expect35 = expect35.expect("meta entry");
        let off = 12 + (1 * 256 * 256 + 3 * 256 + 5) * 4;
        let got = f32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as f64;
        // f64→f32→f64 store path in the Python writer
        assert!(
            (got - expect35).abs() < 1e-5 * (expect35.abs() + 1.0),
            "block1[3,5] got={got} expect={expect35}"
        );
    }
}
