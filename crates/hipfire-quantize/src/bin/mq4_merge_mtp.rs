//! Bundle a trunk `.mq4` file with an MTP `.mtp` head into a single
//! `.mq4-mtp` file.
//!
//! Output layout:
//! ```text
//! [trunk.mq4 bytes ─ byte-identical to input]
//! [mtp.mtp bytes  ─ byte-identical to input]
//! [16-byte trailer]:
//!   - magic    : "HFBNDMTP" (8 bytes)
//!   - mtp_off  : u64 LE (8 bytes)  ← byte offset where MTP section starts
//! ```
//!
//! Design notes:
//! - The trunk loader (`HfqFile::open`) reads only as many bytes as the
//!   trunk's metadata + index says it has; the trailing MTP section + trailer
//!   are ignored. Existing `qwen3.5-27b.mq4` loaders open `.mq4-mtp` files
//!   transparently, returning only the trunk.
//! - The MTP loader detects the bundle by checking the trailer magic at
//!   `file_size - 16`. If present, the MTP `.mtp` section is parsed starting
//!   at `mtp_off` (an offset-aware HFQM parser).
//! - The MTP section is byte-identical to a standalone `.mtp` file (15-tensor
//!   no-sidecar variant from `mtp_extract --quant mq4`). No `lm_head_draft`
//!   is included — the runtime uses the trunk's `output` (lm_head) per the
//!   MTP head's `shared_lm_head_with_trunk: true` metadata.
//!
//! Usage:
//! ```text
//! mq4_merge_mtp --trunk qwen3.5-27b.mq4 --mtp qwen3.5-27b.mtp \
//!     --output qwen3.5-27b.mq4-mtp
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Trailer magic. Picked to be highly unlikely to collide with the tail of a
/// valid HFQM tensor payload (HFBNDMTP = "HipFire BuNDle MTP").
pub const BUNDLE_TRAILER_MAGIC: &[u8; 8] = b"HFBNDMTP";
pub const BUNDLE_TRAILER_LEN: u64 = 16;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut trunk_path: Option<PathBuf> = None;
    let mut mtp_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--trunk" => { trunk_path = Some(args[i + 1].clone().into()); i += 2; }
            "--mtp" => { mtp_path = Some(args[i + 1].clone().into()); i += 2; }
            "--output" => { output_path = Some(args[i + 1].clone().into()); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: mq4_merge_mtp --trunk <trunk.mq4> --mtp <head.mtp> --output <bundle.mq4-mtp>");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    let trunk = trunk_path.expect("--trunk required");
    let mtp = mtp_path.expect("--mtp required");
    let out = output_path.expect("--output required");

    eprintln!("mq4_merge_mtp");
    eprintln!("  trunk:  {}", trunk.display());
    eprintln!("  mtp:    {}", mtp.display());
    eprintln!("  output: {}", out.display());

    // Sanity check magics before touching the output file.
    {
        let mut f = File::open(&trunk).expect("open trunk");
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).expect("read trunk magic");
        assert_eq!(&magic, b"HFQM", "trunk file is not an HFQM container");
    }
    {
        let mut f = File::open(&mtp).expect("open mtp");
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).expect("read mtp magic");
        assert_eq!(&magic, b"HFQM", "mtp file is not an HFQM container");
    }

    let trunk_size = std::fs::metadata(&trunk).expect("stat trunk").len();
    let mtp_size = std::fs::metadata(&mtp).expect("stat mtp").len();
    eprintln!("  trunk size: {:.2} GiB", trunk_size as f64 / (1024.0 * 1024.0 * 1024.0));
    eprintln!("  mtp size  : {:.2} MiB", mtp_size as f64 / (1024.0 * 1024.0));

    let mut out_f = File::create(&out).expect("create output");

    // 1. trunk bytes (verbatim)
    let mut trunk_f = File::open(&trunk).expect("open trunk for copy");
    let trunk_written = std::io::copy(&mut trunk_f, &mut out_f).expect("copy trunk");
    assert_eq!(trunk_written, trunk_size, "trunk byte count mismatch");

    // 2. mtp bytes (verbatim) — MTP section starts at trunk_size
    let mtp_offset = trunk_written;
    let mut mtp_f = File::open(&mtp).expect("open mtp for copy");
    let mtp_written = std::io::copy(&mut mtp_f, &mut out_f).expect("copy mtp");
    assert_eq!(mtp_written, mtp_size, "mtp byte count mismatch");

    // 3. 16-byte trailer
    out_f.write_all(BUNDLE_TRAILER_MAGIC).expect("write trailer magic");
    out_f.write_all(&mtp_offset.to_le_bytes()).expect("write mtp_offset");
    out_f.sync_all().expect("fsync");

    let final_size = trunk_size + mtp_size + BUNDLE_TRAILER_LEN;
    let stat_size = std::fs::metadata(&out).expect("stat output").len();
    assert_eq!(stat_size, final_size, "output size mismatch");

    // 4. Verify by re-reading the trailer.
    {
        let mut f = File::open(&out).expect("reopen output");
        f.seek(SeekFrom::End(-(BUNDLE_TRAILER_LEN as i64))).expect("seek trailer");
        let mut trailer = [0u8; 16];
        f.read_exact(&mut trailer).expect("read trailer");
        assert_eq!(&trailer[..8], BUNDLE_TRAILER_MAGIC, "trailer magic mismatch on readback");
        let parsed_offset = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
        assert_eq!(parsed_offset, mtp_offset, "trailer offset mismatch on readback");

        // Verify MTP section starts with HFQM magic at the recorded offset.
        f.seek(SeekFrom::Start(parsed_offset)).expect("seek mtp section");
        let mut mtp_magic = [0u8; 4];
        f.read_exact(&mut mtp_magic).expect("read mtp magic from bundle");
        assert_eq!(&mtp_magic, b"HFQM", "mtp section magic mismatch in bundle");
    }

    eprintln!(
        "wrote {}: {:.2} GiB  (trunk={:.2} GiB, mtp={:.2} MiB, trailer={} B)",
        out.display(),
        final_size as f64 / (1024.0 * 1024.0 * 1024.0),
        trunk_size as f64 / (1024.0 * 1024.0 * 1024.0),
        mtp_size as f64 / (1024.0 * 1024.0),
        BUNDLE_TRAILER_LEN,
    );
    eprintln!("verify: PASS — trailer magic + offset round-trip clean, MTP section at offset {mtp_offset}");
}
