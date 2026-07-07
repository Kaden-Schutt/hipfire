//! `hipfire model {compose,decompose}` — reorganize `.hfq` packaging between a
//! single bundled container and separate base + role/feature sidecar files.
//!
//! Native `.hfq`->`.hfq` manipulation (no tensor-byte transform); the heavy
//! lifting lives in `hipfire_runtime::hfq_compose`.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use hipfire_runtime::hfq_compose::{compose_hfq, decompose_hfq_auto, sidecar_tag_from_filename};

use crate::model::find_model;

#[derive(Debug, Args)]
pub struct ModelArgs {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Merge a base `.hfq` and its role/feature sidecars into one bundled
    /// container (records a provenance manifest so `decompose` is lossless).
    Compose(ComposeArgs),
    /// Split a bundled `.hfq` back into its base + sidecar files.
    Decompose(DecomposeArgs),
}

#[derive(Debug, Args)]
pub struct ComposeArgs {
    /// Base container first, then one or more sidecars (file paths or model
    /// aliases).
    #[arg(required = true, num_args = 2..)]
    inputs: Vec<String>,
    /// Output bundle path. Default: the base name with the sidecar feature
    /// dot-groups inserted before the quant token (e.g.
    /// `Model.mq4.hfq` + `Model.mtp.hfq` -> `Model.mtp.mq4.hfq`).
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct DecomposeArgs {
    /// Bundle container to split (file path or model alias).
    bundle: String,
    /// Directory to write the reconstructed component files into.
    output_dir: PathBuf,
    /// Heuristically split a bundle that has no `hipfire_compose` manifest,
    /// using the filename's role dot-groups + tensor-name prefixes. Legacy
    /// bundles with a plain filename fall back to inferring roles from tensor
    /// names alone. Lossy: output files are not byte-identical to any originals.
    /// Bundles that DO carry a manifest still take the exact, lossless path.
    #[arg(long)]
    infer: bool,
}

pub fn run(args: ModelArgs) -> anyhow::Result<()> {
    match args.command {
        ModelCommand::Compose(a) => run_compose(a),
        ModelCommand::Decompose(a) => run_decompose(a),
    }
}

/// Resolve an argument to a concrete path: an existing file path wins, else it
/// is treated as a model alias resolved against the models directory.
fn resolve(arg: &str) -> anyhow::Result<PathBuf> {
    let direct = PathBuf::from(arg);
    if direct.exists() {
        return Ok(direct);
    }
    find_model(arg).ok_or_else(|| anyhow::anyhow!("no such file or model: {arg}"))
}

/// Default bundle path: insert sorted, de-duplicated sidecar feature tags as
/// dot-groups immediately before the base's quant token (the last stem
/// segment), per the artifact naming convention.
fn default_bundle_path(inputs: &[PathBuf]) -> PathBuf {
    let base = &inputs[0];
    let dir = base.parent().map(PathBuf::from);
    let fname = base
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "bundle.hfq".to_string());
    let stem = fname.strip_suffix(".hfq").unwrap_or(&fname);

    let mut tags: Vec<String> = inputs[1..]
        .iter()
        .filter_map(|p| sidecar_tag_from_filename(p))
        .collect();
    tags.sort();
    tags.dedup();

    let mut segs: Vec<String> = stem.split('.').map(|s| s.to_string()).collect();
    // Quant token is the last stem segment by convention; features go before
    // it. If the stem is a single segment there is no quant token to precede,
    // so append instead.
    let insert_at = if segs.len() >= 2 {
        segs.len() - 1
    } else {
        segs.len()
    };
    for (i, tag) in tags.into_iter().enumerate() {
        if !segs.contains(&tag) {
            segs.insert(insert_at + i, tag);
        }
    }
    let out_name = format!("{}.hfq", segs.join("."));
    match dir {
        Some(d) if !d.as_os_str().is_empty() => d.join(out_name),
        _ => PathBuf::from(out_name),
    }
}

fn run_compose(a: ComposeArgs) -> anyhow::Result<()> {
    let inputs: Vec<PathBuf> = a
        .inputs
        .iter()
        .map(|s| resolve(s))
        .collect::<anyhow::Result<_>>()?;
    let out = a.output.unwrap_or_else(|| default_bundle_path(&inputs));
    let written = compose_hfq(&inputs, &out)?;
    println!("composed {} inputs -> {}", inputs.len(), written.display());
    Ok(())
}

fn run_decompose(a: DecomposeArgs) -> anyhow::Result<()> {
    let bundle = resolve(&a.bundle)?;
    let written = decompose_hfq_auto(&bundle, &a.output_dir, a.infer)?;
    println!(
        "decomposed{} {} -> {} file(s) in {}",
        if a.infer { " (heuristic)" } else { "" },
        bundle.display(),
        written.len(),
        a.output_dir.display()
    );
    for p in &written {
        println!("  {}", p.display());
    }
    Ok(())
}
