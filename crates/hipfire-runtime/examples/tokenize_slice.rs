//! Tokenize a slice file using hipfire's tokenizer; emit token IDs as binary.
//!
//! Used by `benchmarks/quality-baselines/harness/tokenizer_parity.py` to
//! compare hipfire's tokenization of the eval slice against
//! `llama-tokenize` (built from the pinned llama.cpp commit). If both produce
//! byte-identical token streams, the GGUF anchor track in the eval matrix is
//! viable; if not, fall back to the bridge or drop the anchor entirely
//! (per plan rev-3.2 §"Tokenizer alignment + bridge investigation").
//!
//! Usage:
//!   tokenize_slice --model <path-to-hfq> --slice <path-to-text> --output <path-to-bin>
//!
//! Output format: contiguous u32 little-endian token IDs. No header.
//! Plain so `llama-tokenize --ids` output (whitespace-separated decimal IDs)
//! can be parsed and compared on the Python side.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut slice: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model"  => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--slice"  => { slice = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: tokenize_slice --model <path-to-hfq> --slice <path-to-text> --output <path-to-bin>");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");
    let slice = slice.expect("--slice required");
    let output = output.expect("--output required");

    let hfq = HfqFile::open(&model).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("load tokenizer from hfq metadata");
    let text = std::fs::read_to_string(&slice).expect("read slice");
    eprintln!("tokenize_slice: model={} slice={} ({} bytes)",
              model.display(), slice.display(), text.len());

    let ids: Vec<u32> = tokenizer.encode(&text);
    eprintln!("tokenize_slice: produced {} tokens", ids.len());

    let mut out = std::fs::File::create(&output).expect("create output");
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    out.write_all(&bytes).expect("write tokens");
    eprintln!("tokenize_slice: wrote {} bytes to {}", bytes.len(), output.display());
}
