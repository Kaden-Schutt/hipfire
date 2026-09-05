// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Encode → decode round-trip for a model's embedded tokenizer.
//!
//! Written to chase a specific symptom: the 35B reports the planted checksum
//! `VIOLET-ANVIL-62` as `VIOLETANVIL62` — and volunteers "no hyphens" —
//! even when the source sentence is 500 tokens away, where retrieval is
//! plainly working. Characters going missing at that distance is not memory
//! decay; it points at the text pipeline. If a string does not survive
//! encode→decode, the model never saw what we think we put in the prompt and
//! every downstream "recall" measurement built on it is measuring the wrong
//! thing.

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).map(String::as_str).unwrap_or_else(|| {
        eprintln!("usage: tokenizer_roundtrip <model.hfq> [extra strings...]");
        std::process::exit(1);
    });

    let hfq = HfqFile::open(Path::new(model)).expect("open hfq");
    let tk = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not in hfq metadata");

    let mut cases: Vec<String> = vec![
        "VIOLET-ANVIL-62".into(),
        "the checksum phrase VIOLET-ANVIL-62 before".into(),
        "Kestrel".into(),
        "Reykjavik".into(),
        "Reykjavík".into(),
        "14 March 2019".into(),
        "ratified in Reykjavik on 14 March 2019 by exactly eleven".into(),
        "3.7-second".into(),
        "the Halvorsen Gap".into(),
        "escha_types_never_resolve".into(),
        "a-b-c".into(),
        "well-known".into(),
        "2019".into(),
        "62".into(),
    ];
    cases.extend(args.iter().skip(2).cloned());

    let mut bad = 0;
    for s in &cases {
        let ids = tk.encode(s);
        let back = tk.decode(&ids);
        let ok = &back == s;
        if !ok {
            bad += 1;
        }
        println!(
            "{} {:?}\n    -> {} ids {:?}\n    -> {:?}",
            if ok { "ok  " } else { "FAIL" },
            s,
            ids.len(),
            ids,
            back
        );
    }
    println!(
        "\n{bad} of {} strings did not survive encode->decode",
        cases.len()
    );
    std::process::exit(if bad > 0 { 1 } else { 0 });
}
