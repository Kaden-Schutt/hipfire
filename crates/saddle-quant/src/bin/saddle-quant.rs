// saddle-quant CLI — one subcommand per quality-tooling question that
// previously required throwaway Python.
//
// This binary contains no format parsing of its own; every file is opened
// through `saddle_quant::format::*` and every statistic through
// `saddle_quant::stats` or `saddle_quant::format::kldseq`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use saddle_quant::format::imatrix::Imatrix;
use saddle_quant::format::QuantType;
use saddle_quant::format::hfq::HfqFile;
use saddle_quant::{ArtifactId, QuantError};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "saddle-quant", version, about = "saddle-quant quality toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect an HFQ container
    Inspect {
        #[arg(value_name = "PATH")]
        path: PathBuf,
        #[arg(long, help = "emit JSON instead of human-readable table")]
        json: bool,
    },
    /// Diff two HFQ containers by tensor dtype and size
    #[command(name = "diff-hfq")]
    DiffHfq {
        #[arg(value_name = "A")]
        a: PathBuf,
        #[arg(value_name = "B")]
        b: PathBuf,
    },
    /// Inspect a GGUF imatrix
    Imatrix {
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
    /// Diff two imatrices
    #[command(name = "diff-imatrix")]
    DiffImatrix {
        #[arg(value_name = "A")]
        a: PathBuf,
        #[arg(value_name = "B")]
        b: PathBuf,
        #[arg(
            long = "rel-tol",
            default_value = "0.01",
            help = "relative tolerance for per-channel comparison"
        )]
        rel_tol: f64,
        #[arg(long, default_value = "10", help = "number of worst entries to show")]
        worst: usize,
    },
    /// Reduce a directory of *.kldseq files
    Reduce {
        #[arg(value_name = "DIR")]
        dir: PathBuf,
    },
    /// Identify a file by SHA-256
    Identify {
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Pure helpers — testable without I/O
// ---------------------------------------------------------------------------

/// Display name for a wire quant tag.
pub fn dtype_name(tag: u8) -> String {
    match QuantType::from_tag(tag) {
        Some(qt) => format!("{qt:?}"),
        None => format!("unknown({tag})"),
    }
}

/// Format one dtype-histogram row: tag, dtype name, tensor count, total bytes.
///
/// This is the helper covered by tests. It must include all four fields.
pub fn format_histogram_row(tag: u8, count: usize, bytes: u64) -> String {
    let name = dtype_name(tag);
    format!("{tag:>3}  {name:<12}  {count:>5}  {bytes:>14}")
}

/// Extract the slot from a `blk.<N>.<slot>.weight` name.
///
/// Returns `Some(slot)` when the name conforms to that shape, otherwise `None`.
/// The second component must be a decimal block index and the last component
/// must be `weight`.
pub fn extract_slot(name: &str) -> Option<String> {
    let rest = name.strip_prefix("blk.")?;
    let mut parts = rest.split('.');
    let idx = parts.next()?;
    if idx.is_empty() || idx.parse::<u32>().is_err() {
        return None;
    }
    let slot = parts.next()?;
    if slot.is_empty() {
        return None;
    }
    let suffix = parts.next()?;
    if suffix != "weight" {
        return None;
    }
    if parts.next().is_some() {
        return None;
    }
    Some(slot.to_string())
}

/// Histogram of slots over an iterator of tensor names.
pub fn slot_histogram_from_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for name in names {
        if let Some(slot) = extract_slot(name) {
            *out.entry(slot).or_insert(0) += 1;
        }
    }
    out
}

/// Histogram of slots in an opened imatrix.
pub fn imatrix_slot_histogram(imatrix: &Imatrix) -> BTreeMap<String, usize> {
    slot_histogram_from_names(imatrix.names())
}

/// One row of an HFQ diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfqDiffRow {
    pub name: String,
    pub tag_a: Option<u8>,
    pub size_a: Option<u64>,
    pub tag_b: Option<u8>,
    pub size_b: Option<u64>,
}

impl HfqDiffRow {
    fn delta_bytes(&self) -> Option<i64> {
        match (self.size_a, self.size_b) {
            (Some(a), Some(b)) => Some(b as i64 - a as i64),
            _ => None,
        }
    }
}

/// Compare two HFQ files by tensor presence, `quant_tag`, and `data_size`.
///
/// A row is emitted for every name that is present in only one file or whose
/// tag or size differ. Identical tensors produce no row.
pub fn diff_hfq_files(a: &HfqFile, b: &HfqFile) -> Vec<HfqDiffRow> {
    let map_a: BTreeMap<&str, _> = a.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let map_b: BTreeMap<&str, _> = b.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut names = BTreeSet::new();
    for k in map_a.keys() {
        names.insert(*k);
    }
    for k in map_b.keys() {
        names.insert(*k);
    }
    let mut rows = Vec::new();
    for name in names {
        match (map_a.get(name), map_b.get(name)) {
            (Some(ta), Some(tb)) => {
                if ta.quant_tag != tb.quant_tag || ta.data_size != tb.data_size {
                    rows.push(HfqDiffRow {
                        name: name.to_string(),
                        tag_a: Some(ta.quant_tag),
                        size_a: Some(ta.data_size),
                        tag_b: Some(tb.quant_tag),
                        size_b: Some(tb.data_size),
                    });
                }
            }
            (Some(ta), None) => {
                rows.push(HfqDiffRow {
                    name: name.to_string(),
                    tag_a: Some(ta.quant_tag),
                    size_a: Some(ta.data_size),
                    tag_b: None,
                    size_b: None,
                });
            }
            (None, Some(tb)) => {
                rows.push(HfqDiffRow {
                    name: name.to_string(),
                    tag_a: None,
                    size_a: None,
                    tag_b: Some(tb.quant_tag),
                    size_b: Some(tb.data_size),
                });
            }
            (None, None) => {}
        }
    }
    rows
}

/// Format one HFQ diff row for the table.
pub fn format_hfq_diff_row(row: &HfqDiffRow) -> String {
    match (row.tag_a, row.size_a, row.tag_b, row.size_b) {
        (Some(ta), Some(sa), Some(tb), Some(sb)) => {
            let na = dtype_name(ta);
            let nb = dtype_name(tb);
            // Keep the shape the task describes:
            // lm_head.weight  Q8F16(3) 1350860800  ->  Mq4G256(13) 675430400
            let delta = sb as i64 - sa as i64;
            format!(
                "{:<40}  {na}({ta}) {sa:>12}  ->  {nb}({tb}) {sb:>12}  delta {delta:+}",
                row.name
            )
        }
        (Some(ta), Some(sa), None, None) => {
            let na = dtype_name(ta);
            format!("{:<40}  {na}({ta}) {sa:>12}  ->  MISSING", row.name)
        }
        (None, None, Some(tb), Some(sb)) => {
            let nb = dtype_name(tb);
            format!("{:<40}  MISSING  ->  {nb}({tb}) {sb:>12}", row.name)
        }
        _ => format!("{:<40}  incomplete row", row.name),
    }
}

/// Format all HFQ diff rows as a table.
pub fn format_hfq_diff(rows: &[HfqDiffRow]) -> String {
    if rows.is_empty() {
        return "no differences".to_string();
    }
    let mut out = String::new();
    out.push_str(&format!("{:<40}  {}\n", "tensor", "diff"));
    for r in rows {
        out.push_str(&format_hfq_diff_row(r));
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Inspect helpers
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct InspectJson {
    path: String,
    version: u32,
    arch_id: u32,
    tensor_count: usize,
    total_payload_bytes: u64,
    bits_per_weight: f64,
    histogram: Vec<HistogramEntry>,
}

#[derive(serde::Serialize)]
struct HistogramEntry {
    tag: u8,
    dtype: String,
    count: usize,
    bytes: u64,
}

/// Human-readable inspect output from an already-opened HFQ file.
pub fn format_inspect_text(file: &HfqFile, path: &Path) -> String {
    let mut out = String::new();
    out.push_str(&format!("path: {}\n", path.display()));
    out.push_str(&format!("version: {}\n", file.version));
    out.push_str(&format!("arch_id: {}\n", file.arch_id));
    out.push_str(&format!("tensors: {}\n", file.tensors.len()));
    out.push_str(&format!(
        "total_payload_bytes: {}\n",
        file.total_payload_bytes()
    ));
    out.push_str(&format!("bits_per_weight: {:.4}\n", file.bits_per_weight()));
    out.push_str("dtype_histogram:\n");
    out.push_str(&format!(
        "  {:>3}  {:<12}  {:>5}  {:>14}\n",
        "tag", "dtype", "count", "bytes"
    ));
    for (tag, (count, bytes)) in file.dtype_histogram() {
        out.push_str(&format!("  {}\n", format_histogram_row(tag, count, bytes)));
    }
    out
}

/// Machine-readable JSON inspect output.
pub fn format_inspect_json(file: &HfqFile, path: &Path) -> Result<String, QuantError> {
    let hist = file
        .dtype_histogram()
        .into_iter()
        .map(|(tag, (count, bytes))| HistogramEntry {
            tag,
            dtype: dtype_name(tag),
            count,
            bytes,
        })
        .collect();
    let payload = InspectJson {
        path: path.display().to_string(),
        version: file.version,
        arch_id: file.arch_id,
        tensor_count: file.tensors.len(),
        total_payload_bytes: file.total_payload_bytes(),
        bits_per_weight: file.bits_per_weight(),
        histogram: hist,
    };
    serde_json::to_string_pretty(&payload)
        .map_err(|e| QuantError::Malformed(format!("json serialize failed: {e}")))
}

// ---------------------------------------------------------------------------
// Imatrix helpers
// ---------------------------------------------------------------------------

/// Human-readable imatrix report from an already-opened imatrix.
pub fn format_imatrix_report(imatrix: &Imatrix) -> String {
    let mut out = String::new();
    out.push_str(&format!("entries: {}\n", imatrix.len()));
    out.push_str(&format!("skipped_moe: {}\n", imatrix.skipped_moe()));
    out.push_str("slot_histogram:\n");
    let hist = imatrix_slot_histogram(imatrix);
    if hist.is_empty() {
        out.push_str("  (no blk.<N>.<slot>.weight entries)\n");
    } else {
        for (slot, count) in &hist {
            out.push_str(&format!("  {slot:<16} {count:>5}\n"));
        }
    }
    // Per-entry token counts when present.
    let mut any_counts = false;
    for name in imatrix.names() {
        if imatrix.counts(name).is_some() {
            any_counts = true;
            break;
        }
    }
    if any_counts {
        out.push_str("counts:\n");
        // Sort names for deterministic output.
        let mut names: Vec<&str> = imatrix.names().collect();
        names.sort_unstable();
        for name in names {
            if let Some(c) = imatrix.counts(name) {
                out.push_str(&format!("  {name}: {c}\n"));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Diff-imatrix helpers
// ---------------------------------------------------------------------------

/// Format a diff-imatrix report from an already-compared pair.
pub fn format_diff_imatrix(
    diff: &saddle_quant::stats::ImatrixDiff,
    worst: usize,
) -> String {
    let mut out = String::new();
    out.push_str(&diff.summary());
    if !out.ends_with('\n') {
        out.push('\n');
    }
    let worst_entries = diff.worst_by_spearman(worst);
    if worst_entries.is_empty() {
        return out;
    }
    out.push_str(&format!("worst {worst} by spearman (ascending):\n"));
    out.push_str(&format!(
        "  {:<40}  {:>8}  {:>8}  {:>10}  {:>10}\n",
        "tensor", "spearman", "pearson", "cosine", "max_rel_delta"
    ));
    for (name, cmp) in worst_entries {
        out.push_str(&format!(
            "  {name:<40}  {spearman:>8.4}  {pearson:>8.4}  {cosine:>10.4}  {max_rel:>10.4}\n",
            spearman = cmp.spearman,
            pearson = cmp.pearson,
            cosine = cmp.cosine,
            max_rel = cmp.max_rel_delta
        ));
    }
    if !diff.only_in_a.is_empty() {
        out.push_str(&format!("only in A ({}):\n", diff.only_in_a.len()));
        let mut v = diff.only_in_a.clone();
        v.sort_unstable();
        for n in &v {
            out.push_str(&format!("  {n}\n"));
        }
    }
    if !diff.only_in_b.is_empty() {
        out.push_str(&format!("only in B ({}):\n", diff.only_in_b.len()));
        let mut v = diff.only_in_b.clone();
        v.sort_unstable();
        for n in &v {
            out.push_str(&format!("  {n}\n"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Reduce helpers
// ---------------------------------------------------------------------------

/// Sort arms by mean KLD ascending (pure, testable).
pub fn sort_arms_by_kld(mut arms: Vec<saddle_quant::ArmScore>) -> Vec<saddle_quant::ArmScore> {
    arms.sort_by(|a, b| {
        a.mean_kld
            .partial_cmp(&b.mean_kld)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    arms
}

/// Format the reduce table (label / mean KLD / CI / p99 / PPL / n_chunks).
pub fn format_reduce_table(arms: &[saddle_quant::ArmScore]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<24}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}\n",
        "label", "mean_kld", "ci_lo", "ci_hi", "p99_kld", "ppl", "n_chunks"
    ));
    for a in arms {
        out.push_str(&format!(
            "{:<24}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.2}  {:>8}\n",
            a.label, a.mean_kld, a.ci_lo, a.ci_hi, a.p99_kld, a.ppl, a.n_chunks
        ));
    }
    out
}

/// Format the all-pairs separation matrix.
pub fn format_compare_matrix(arms: &[saddle_quant::ArmScore]) -> String {
    let reports = saddle_quant::format::kldseq::compare(arms);
    let mut out = String::new();
    if reports.is_empty() {
        out.push_str("(need at least 2 arms to compare)\n");
        return out;
    }
    let mut not_separated = 0usize;
    for r in &reports {
        let status = if r.separated {
            "separated"
        } else {
            not_separated += 1;
            "NOT SEPARATED"
        };
        out.push_str(&format!(
            "  {} vs {}: {} (delta {:.6})\n",
            r.a, r.b, status, r.delta
        ));
    }
    let total = reports.len();
    out.push_str(&format!(
        "{not_separated} of {total} pairs not separated; ranking among non-separated arms is not supported by the measurement.\n"
    ));
    out
}

// ---------------------------------------------------------------------------
// Identify helpers
// ---------------------------------------------------------------------------

pub fn format_identify(id: &ArtifactId) -> String {
    format!(
        "path: {}\nsha256: {}\nbytes: {}\n",
        id.path, id.sha256, id.bytes
    )
}

// ---------------------------------------------------------------------------
// I/O dispatch — thin wrappers that open files and delegate to pure helpers
// ---------------------------------------------------------------------------

fn run_inspect(path: &Path, json: bool) -> Result<(), QuantError> {
    let file = HfqFile::open(path)?;
    if json {
        let s = format_inspect_json(&file, path)?;
        println!("{s}");
    } else {
        print!("{}", format_inspect_text(&file, path));
    }
    Ok(())
}

fn run_diff_hfq(a: &Path, b: &Path) -> Result<(), QuantError> {
    let fa = HfqFile::open(a)?;
    let fb = HfqFile::open(b)?;
    let rows = diff_hfq_files(&fa, &fb);
    if rows.is_empty() {
        println!("no differences");
    } else {
        // Print table to stdout.
        println!("{}", format_hfq_diff(&rows));
        // Also summarize byte delta when exactly one row with both sizes.
        if rows.len() == 1 {
            if let Some(delta) = rows[0].delta_bytes() {
                println!("delta bytes: {}", delta.abs());
            }
        }
    }
    Ok(())
}

fn run_imatrix(path: &Path) -> Result<(), QuantError> {
    let im = saddle_quant::format::imatrix::open(path)?;
    print!("{}", format_imatrix_report(&im));
    Ok(())
}

fn run_diff_imatrix(a: &Path, b: &Path, rel_tol: f64, worst: usize) -> Result<(), QuantError> {
    let ia = saddle_quant::format::imatrix::open(a)?;
    let ib = saddle_quant::format::imatrix::open(b)?;
    let diff = saddle_quant::stats::diff_imatrix(&ia, &ib, rel_tol);
    print!("{}", format_diff_imatrix(&diff, worst));
    Ok(())
}

fn run_reduce(dir: &Path) -> Result<(), QuantError> {
    let mut arms: Vec<saddle_quant::ArmScore> = Vec::new();
    let read_dir = std::fs::read_dir(dir)?;
    let mut any = false;
    for entry in read_dir {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("kldseq") {
            continue;
        }
        let seq = saddle_quant::format::kldseq::open(&path)?;
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let arm = saddle_quant::format::kldseq::reduce(&label, &seq, 10_000, 42);
        arms.push(arm);
        any = true;
    }
    if !any {
        return Err(QuantError::Malformed(format!(
            "no *.kldseq files in {}",
            dir.display()
        )));
    }
    let arms = sort_arms_by_kld(arms);
    print!("{}", format_reduce_table(&arms));
    println!();
    print!("{}", format_compare_matrix(&arms));
    Ok(())
}

fn run_identify(path: &Path) -> Result<(), QuantError> {
    let id = saddle_quant::format::identify(path)?;
    print!("{}", format_identify(&id));
    Ok(())
}

fn run() -> Result<(), QuantError> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { path, json } => run_inspect(&path, json),
        Commands::DiffHfq { a, b } => run_diff_hfq(&a, &b),
        Commands::Imatrix { path } => run_imatrix(&path),
        Commands::DiffImatrix { a, b, rel_tol, worst } => {
            run_diff_imatrix(&a, &b, rel_tol, worst)
        }
        Commands::Reduce { dir } => run_reduce(&dir),
        Commands::Identify { path } => run_identify(&path),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — pure helpers, no I/O required except via tempfile where needed
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // Only the derive-validation test needs the CommandFactory trait in scope.
    use clap::CommandFactory;
    use saddle_quant::format::{QuantType, TensorEntry};
    use saddle_quant::format::hfq::HfqFile;

    fn make_tensor(name: &str, tag: u8, data_size: u64) -> TensorEntry {
        TensorEntry {
            name: name.to_string(),
            quant_tag: tag,
            quant_type: QuantType::from_tag(tag),
            shape: vec![1024, 1024],
            group_size: 256,
            data_offset: 0,
            data_size,
        }
    }

    fn make_hfq(tensors: Vec<TensorEntry>) -> HfqFile {
        HfqFile {
            version: 1,
            arch_id: 7,
            metadata_json: "{}".to_string(),
            tensors,
        }
    }

    #[test]
    fn clap_debug_assert() {
        Cli::command().debug_assert();
    }

    #[test]
    fn dtype_name_known_and_unknown() {
        assert_eq!(dtype_name(13), "Mq4G256");
        assert_eq!(dtype_name(3), "Q8F16");
        assert_eq!(dtype_name(1), "F16");
        assert_eq!(dtype_name(255), "unknown(255)");
    }

    #[test]
    fn histogram_row_contains_all_fields() {
        let row = format_histogram_row(13, 496, 675430400);
        assert!(row.contains("13"), "row missing tag: {row}");
        assert!(row.contains("Mq4G256"), "row missing dtype: {row}");
        assert!(row.contains("496"), "row missing count: {row}");
        assert!(row.contains("675430400"), "row missing bytes: {row}");
    }

    #[test]
    fn histogram_row_unknown() {
        let row = format_histogram_row(99, 1, 123);
        assert!(row.contains("99"));
        assert!(row.contains("unknown(99)"));
        assert!(row.contains("1"));
        assert!(row.contains("123"));
    }

    #[test]
    fn extract_slot_valid() {
        assert_eq!(
            extract_slot("blk.7.ssm_alpha.weight"),
            Some("ssm_alpha".to_string())
        );
        assert_eq!(
            extract_slot("blk.0.attn_q.weight"),
            Some("attn_q".to_string())
        );
        assert_eq!(
            extract_slot("blk.12.ffn_up.weight"),
            Some("ffn_up".to_string())
        );
    }

    #[test]
    fn extract_slot_invalid() {
        assert_eq!(extract_slot("token_embd.weight"), None);
        assert_eq!(extract_slot("blk.7.ssm_alpha"), None);
        assert_eq!(extract_slot("blk.7.ssm_alpha.bias"), None);
        assert_eq!(extract_slot("blk.ssm_alpha.weight"), None);
        assert_eq!(extract_slot("blk.7..weight"), None);
        assert_eq!(extract_slot("blk.7.ssm_alpha.weight.extra"), None);
        assert_eq!(extract_slot(""), None);
    }

    #[test]
    fn slot_histogram_derivation() {
        let names = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.1.attn_q.weight",
            "token_embd.weight",
            "blk.0.ffn_up.weight",
        ];
        let hist = slot_histogram_from_names(names.iter().copied());
        assert_eq!(hist.get("attn_q"), Some(&2));
        assert_eq!(hist.get("attn_k"), Some(&1));
        assert_eq!(hist.get("ffn_up"), Some(&1));
        assert_eq!(hist.get("token_embd"), None);
    }

    #[test]
    fn diff_hfq_one_row_on_tag_and_size_change() {
        // One tensor whose tag and size differ must produce exactly one row.
        let a = make_hfq(vec![make_tensor("lm_head.weight", 3, 1_350_860_800)]);
        let b = make_hfq(vec![make_tensor("lm_head.weight", 13, 675_430_400)]);
        let rows = diff_hfq_files(&a, &b);
        assert_eq!(rows.len(), 1, "expected single diff row, got {rows:?}");
        let r = &rows[0];
        assert_eq!(r.name, "lm_head.weight");
        assert_eq!(r.tag_a, Some(3));
        assert_eq!(r.tag_b, Some(13));
        assert_eq!(r.size_a, Some(1_350_860_800));
        assert_eq!(r.size_b, Some(675_430_400));
        // Formatted row must mention both dtypes and sizes.
        let formatted = format_hfq_diff_row(r);
        assert!(formatted.contains("Q8F16"), "missing Q8F16 in {formatted}");
        assert!(formatted.contains("Mq4G256"), "missing Mq4G256 in {formatted}");
        assert!(formatted.contains("1350860800"), "missing size A in {formatted}");
        assert!(formatted.contains("675430400"), "missing size B in {formatted}");
    }

    #[test]
    fn diff_hfq_identical_produces_no_row() {
        let a = make_hfq(vec![make_tensor("blk.0.attn_q.weight", 13, 1000)]);
        let b = make_hfq(vec![make_tensor("blk.0.attn_q.weight", 13, 1000)]);
        assert!(diff_hfq_files(&a, &b).is_empty());
    }

    #[test]
    fn diff_hfq_missing_tensors() {
        let a = make_hfq(vec![
            make_tensor("only_a.weight", 13, 100),
            make_tensor("common.weight", 13, 200),
        ]);
        let b = make_hfq(vec![make_tensor("common.weight", 13, 200)]);
        let rows = diff_hfq_files(&a, &b);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "only_a.weight");
        assert_eq!(rows[0].tag_a, Some(13));
        assert_eq!(rows[0].tag_b, None);
    }

    #[test]
    fn diff_hfq_only_size_change_counts() {
        let a = make_hfq(vec![make_tensor("x.weight", 13, 100)]);
        let b = make_hfq(vec![make_tensor("x.weight", 13, 200)]);
        let rows = diff_hfq_files(&a, &b);
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn diff_hfq_only_tag_change_counts() {
        let a = make_hfq(vec![make_tensor("x.weight", 13, 100)]);
        let b = make_hfq(vec![make_tensor("x.weight", 3, 100)]);
        let rows = diff_hfq_files(&a, &b);
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn inspect_text_contains_required_fields() {
        let file = make_hfq(vec![
            make_tensor("a.weight", 13, 100),
            make_tensor("b.weight", 1, 200),
        ]);
        let text = format_inspect_text(&file, Path::new("/tmp/foo.mq4"));
        assert!(text.contains("version:"), "missing version in {text}");
        assert!(text.contains("arch_id:"), "missing arch_id in {text}");
        assert!(text.contains("tensors:"), "missing tensors in {text}");
        assert!(
            text.contains("total_payload_bytes:"),
            "missing payload in {text}"
        );
        assert!(
            text.contains("bits_per_weight:"),
            "missing bpw in {text}"
        );
        assert!(text.contains("dtype_histogram"), "missing hist in {text}");
    }

    #[test]
    fn inspect_json_round_trips_histogram() {
        let file = make_hfq(vec![make_tensor("a.weight", 13, 100)]);
        let json = format_inspect_json(&file, Path::new("/tmp/foo.mq4")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["version"], 1);
        assert_eq!(v["tensor_count"], 1);
        assert!(v["histogram"].is_array());
    }

    #[test]
    fn format_imatrix_histogram_from_names_only() {
        // Pure helper does not need an Imatrix file.
        let hist = slot_histogram_from_names(
            [
                "blk.0.ffn_up.weight",
                "blk.0.ffn_down.weight",
                "blk.0.ffn_up.weight",
            ]
            .iter()
            .copied(),
        );
        assert_eq!(hist.get("ffn_up"), Some(&2));
        assert_eq!(hist.get("ffn_down"), Some(&1));
    }

    #[test]
    fn sort_arms_by_kld_ascending() {
        use saddle_quant::ArmScore;
        let arms = vec![
            ArmScore {
                label: "b".to_string(),
                mean_kld: 0.2,
                ci_lo: 0.1,
                ci_hi: 0.3,
                p99_kld: 0.4,
                mean_nll: 2.0,
                ppl: 7.389,
                n_chunks: 10,
            },
            ArmScore {
                label: "a".to_string(),
                mean_kld: 0.1,
                ci_lo: 0.05,
                ci_hi: 0.15,
                p99_kld: 0.3,
                mean_nll: 1.9,
                ppl: 6.686,
                n_chunks: 10,
            },
        ];
        let sorted = sort_arms_by_kld(arms);
        assert_eq!(sorted[0].label, "a");
        assert_eq!(sorted[1].label, "b");
    }

    #[test]
    fn reduce_table_and_compare_matrix_smoke() {
        use saddle_quant::ArmScore;
        let arms = vec![
            ArmScore {
                label: "arm1".to_string(),
                mean_kld: 0.05,
                ci_lo: 0.04,
                ci_hi: 0.06,
                p99_kld: 0.08,
                mean_nll: 2.0,
                ppl: 7.389,
                n_chunks: 24,
            },
            ArmScore {
                label: "arm2".to_string(),
                mean_kld: 0.06,
                ci_lo: 0.045,
                ci_hi: 0.075,
                p99_kld: 0.09,
                mean_nll: 2.1,
                ppl: 8.166,
                n_chunks: 24,
            },
        ];
        let table = format_reduce_table(&arms);
        assert!(table.contains("arm1"));
        assert!(table.contains("0.050000"));
        let matrix = format_compare_matrix(&arms);
        // The two arms overlap (ci 0.04-0.06 vs 0.045-0.075) => NOT SEPARATED
        assert!(matrix.contains("NOT SEPARATED") || matrix.contains("not separated"));
        assert!(matrix.contains("pairs not separated"));
    }

    #[test]
    fn format_identify_contains_fields() {
        let id = ArtifactId {
            path: "/tmp/foo.mq4".to_string(),
            sha256: "abc123".to_string(),
            bytes: 42,
        };
        let s = format_identify(&id);
        assert!(s.contains("/tmp/foo.mq4"));
        assert!(s.contains("abc123"));
        assert!(s.contains("42"));
    }
}
