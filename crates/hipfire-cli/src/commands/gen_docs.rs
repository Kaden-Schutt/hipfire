// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire gen-docs` (hidden) — render the CLI's clap definitions into
//! committed user-facing docs, so the `///` doc comments on the arg structs are
//! the single source of truth for `--help`, `docs/CLI.md`, and the man pages.
//!
//! Writes `docs/CLI.md` (via `clap-markdown`) and `man/hipfire*.1` (via
//! `clap_mangen`, one page for the root command and one per subcommand). With
//! `--check` it renders to memory and diffs against the committed files instead
//! of writing, exiting non-zero on drift — the freshness gate `tests/no-gpu-ci.sh`
//! runs so the docs can't silently fall out of sync with the code.

use std::path::Path;

use clap::CommandFactory;

use crate::Cli;

#[derive(Debug, clap::Args)]
pub struct GenDocsArgs {
    /// Directory for the generated Markdown reference (`CLI.md`).
    #[arg(long, default_value = "docs")]
    pub docs_dir: String,
    /// Directory for the generated man pages (`hipfire*.1`).
    #[arg(long, default_value = "man")]
    pub man_dir: String,
    /// Verify the committed docs match the current CLI without writing; exit
    /// non-zero on any drift (for CI).
    #[arg(long)]
    pub check: bool,
}

pub fn run(args: GenDocsArgs) -> anyhow::Result<()> {
    let markdown = render_markdown();
    let man_pages = render_man_pages()?;

    if args.check {
        let mut stale = Vec::new();
        check_file(
            Path::new(&args.docs_dir).join("CLI.md"),
            &markdown,
            &mut stale,
        );
        for (name, bytes) in &man_pages {
            check_file(Path::new(&args.man_dir).join(name), bytes, &mut stale);
        }
        if !stale.is_empty() {
            anyhow::bail!(
                "CLI docs are stale ({} file(s)): {}\n\
                 regenerate with `cargo run -p hipfire-cli -- gen-docs` and commit.",
                stale.len(),
                stale.join(", "),
            );
        }
        eprintln!("gen-docs: CLI docs are up to date");
        return Ok(());
    }

    std::fs::create_dir_all(&args.docs_dir)?;
    std::fs::create_dir_all(&args.man_dir)?;
    let cli_md = Path::new(&args.docs_dir).join("CLI.md");
    std::fs::write(&cli_md, markdown.as_bytes())?;
    eprintln!("gen-docs: wrote {}", cli_md.display());
    for (name, bytes) in &man_pages {
        let p = Path::new(&args.man_dir).join(name);
        std::fs::write(&p, bytes)?;
        eprintln!("gen-docs: wrote {}", p.display());
    }
    Ok(())
}

/// The full Markdown command reference, from the clap `Cli` definition.
fn render_markdown() -> String {
    clap_markdown::help_markdown::<Cli>()
}

/// One roff man page for the root command and one per subcommand, keyed by file
/// name (`hipfire.1`, `hipfire-<sub>.1`).
fn render_man_pages() -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    // The binary's runtime `--version` is a dynamic Git-derived identity
    // (`hipfire_build_info::VERSION`, e.g. `v0.3.0-957-g...`). Pin the *man page*
    // to the static crate version so the docs freshness gate stays
    // deterministic — otherwise every commit would render a new `.TH`/VERSION
    // line and the gate could never be satisfied.
    let cmd = Cli::command().version(env!("CARGO_PKG_VERSION"));
    let mut out = Vec::new();

    let mut root = Vec::new();
    clap_mangen::Man::new(cmd.clone()).render(&mut root)?;
    out.push(("hipfire.1".to_string(), root));

    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        let file = format!("hipfire-{}.1", sub.get_name());
        // `Command::name` wants `Into<Str>` (≈ `&'static str`); leak the short
        // per-subcommand title — this is a one-shot maintenance command.
        let title: &'static str = Box::leak(format!("hipfire-{}", sub.get_name()).into_boxed_str());
        let titled = sub.clone().name(title);
        let mut buf = Vec::new();
        clap_mangen::Man::new(titled).render(&mut buf)?;
        out.push((file, buf));
    }
    Ok(out)
}

fn check_file(path: std::path::PathBuf, expected: impl AsRef<[u8]>, stale: &mut Vec<String>) {
    let matches = std::fs::read(&path)
        .map(|got| got == expected.as_ref())
        .unwrap_or(false);
    if !matches {
        stale.push(path.display().to_string());
    }
}
