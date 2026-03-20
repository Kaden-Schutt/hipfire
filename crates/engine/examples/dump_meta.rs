//! Dump all GGUF metadata keys for a model.

use engine::gguf::GgufFile;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kaden/llama.cpp/models/Qwen3-8B-Q4_K_M.gguf".to_string()
    });
    let gguf = GgufFile::open(Path::new(&path)).unwrap();

    let mut keys: Vec<_> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let val = &gguf.metadata[key];
        // Truncate long arrays
        let val_str = match val {
            engine::gguf::MetaValue::Array(arr) if arr.len() > 5 => {
                format!("[Array of {} elements]", arr.len())
            }
            engine::gguf::MetaValue::String(s) if s.len() > 100 => {
                format!("\"{}...\"", &s[..100])
            }
            _ => format!("{val:?}"),
        };
        println!("  {key}: {val_str}");
    }
}
