// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! GPU admission gate for the exact LFM2.5-350M dense-MQ4 retained route.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
    std::process::exit(1);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::validate_lfm_retained_fixture;
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_arch_lfm2moe::redline_plan::authenticate_retained_artifact;
    use hipfire_arch_lfm2moe::Lfm2MoeBundle;
    use hipfire_runtime::hfq::HfqFile;
    use std::path::PathBuf;

    const EXPECTED_MODEL_MD5: &str = "cb5284b8ad5c6f9e4ca859c0aff0bcd0";

    let model = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: redline_admission_gate <lfm2.5-350m.mq4>");

    let model_md5 = {
        use std::process::Command;
        let output = Command::new("md5sum")
            .arg(&model)
            .output()
            .unwrap_or_else(|e| panic!("md5sum {}: {e}", model.display()));
        assert!(
            output.status.success(),
            "md5sum failed for {}: {}",
            model.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .split_whitespace()
            .next()
            .unwrap_or_else(|| panic!("empty md5sum for {}", model.display()))
            .to_owned()
    };
    assert_eq!(
        model_md5,
        EXPECTED_MODEL_MD5,
        "model md5 mismatch for {}: got {model_md5}, expected {EXPECTED_MODEL_MD5}",
        model.display()
    );

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    assert_eq!(
        gpu.arch.as_str(),
        "gfx1201",
        "admission gate requires exact gpu.arch gfx1201, got {}",
        gpu.arch
    );
    println!("model_path={}", model.display());
    println!("model_md5={model_md5}");
    println!("expected_model_md5={EXPECTED_MODEL_MD5}");
    println!("gpu.arch={}", gpu.arch);
    let mut hfq = HfqFile::open(&model).expect("open model");
    let retained_artifact =
        authenticate_retained_artifact(&mut hfq).expect("authenticate exact retained artifact");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    let mut weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    let mut state = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, 2048).expect("state");

    validate_lfm_retained_fixture(&cfg, &weights, &state, 11)
        .expect("exact loaded fixture must validate");
    assert!(
        validate_lfm_retained_fixture(&cfg, &weights, &state, 6).is_err(),
        "wrong model architecture must reject"
    );

    let layer = weights.layers.pop().expect("fixture layer");
    assert!(
        validate_lfm_retained_fixture(&cfg, &weights, &state, 11).is_err(),
        "missing weight layer must fail structural validation"
    );
    weights.layers.push(layer);

    let conv_state = state.conv_states.pop().expect("fixture conv state");
    assert!(
        validate_lfm_retained_fixture(&cfg, &weights, &state, 11).is_err(),
        "missing conv-state slot must fail structural validation"
    );
    state.conv_states.push(conv_state);

    let mut bundle = Lfm2MoeBundle::new(cfg, weights, state, 0, 11, retained_artifact);
    assert!(
        bundle.retained_fixture_evidence(),
        "bundle construction must cache valid fixture evidence"
    );
    bundle.reset(&mut gpu).expect("request reset");
    assert!(
        bundle.retained_fixture_evidence(),
        "request reset must retain model fixture evidence"
    );

    println!("LFM REDLINE ADMISSION GATE PASS");
}
