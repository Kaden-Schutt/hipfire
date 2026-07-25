import json, subprocess, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
G = [sys.executable, os.path.join(HERE, "no-cargo-fmt-guard.py")]
cases = [
 ("DENY","bare cargo fmt","cargo fmt"),
 ("DENY","cargo fmt --all","cargo fmt --all"),
 ("DENY","cargo +nightly fmt","cargo +nightly fmt --all"),
 ("DENY","cd then cargo fmt","cd crates/hip-bridge && cargo fmt"),
 ("DENY","chained after build","cargo build && cargo fmt --all"),
 ("DENY","cargo-fmt direct","cargo-fmt"),
 ("DENY","subshell","(cargo fmt)"),
 ("DENY","cargo fmt --check (still rewrites nothing but blocked by design)","cargo fmt --check"),
 ("DENY","rustfmt no targets","rustfmt"),
 ("DENY","rustfmt glob","rustfmt crates/**/*.rs"),
 ("DENY","rustfmt find -exec","find . -name '*.rs' -exec rustfmt {} +"),
 ("DENY","rustfmt xargs","git ls-files '*.rs' | xargs rustfmt"),
 ("DENY","rustfmt cmdsubst","rustfmt $(git ls-files '*.rs')"),
 ("DENY","rustfmt -r","rustfmt -r src"),
 ("DENY","semicolon chain","echo hi; cargo fmt"),
 ("ALLOW","sanctioned wrapper","scripts/fmt-changed.sh"),
 ("ALLOW","wrapper w/ BASE_REF","BASE_REF=origin/beta scripts/fmt-changed.sh"),
 ("ALLOW","ci wrapper","scripts/ci-rustfmt-changed.sh"),
 ("ALLOW","rustfmt --check","rustfmt --edition 2021 --check crates/a/src/lib.rs"),
 ("ALLOW","explicit files","rustfmt --edition 2021 --config skip_children=true crates/a/src/lib.rs crates/b/src/x.rs"),
 ("ALLOW","escape hatch","HIPFIRE_ALLOW_FMT=1 cargo fmt --all"),
 ("ALLOW","cargo build","cargo build --release --workspace --all-targets --locked"),
 ("ALLOW","cargo test","cargo test --lib"),
 ("ALLOW","cargo clippy","cargo clippy --workspace"),
 ("ALLOW","grep mentioning it","grep -rn 'cargo fmt' CLAUDE.md"),
 ("ALLOW","daemon build","cargo build --release --example daemon --features deltanet -p hipfire-runtime"),

 ("DENY","subshell trailing paren","(cargo fmt)"),
 ("DENY","backgrounded","cargo fmt &"),
 ("DENY","piped to tee","cargo fmt | tee log"),
 ("DENY","bash -lc wrapper","bash -lc 'cargo fmt --all'"),
 ("DENY","ssh remote","ssh hipx 'cd hipfire && cargo fmt'"),
 ("DENY","rustfmt in subshell","(rustfmt)"),
 ("ALLOW","rustfmt file in subshell","(rustfmt --edition 2021 crates/a/src/lib.rs)"),

 ("ALLOW","grep double quotes","grep -rn \"cargo fmt\" docs/"),
 ("ALLOW","rg search","rg 'cargo fmt' --type md"),
 ("ALLOW","echo mentioning","echo 'do not run cargo fmt'"),
 ("DENY","sh -c wrapper","sh -c 'cargo fmt'"),
 ("DENY","ssh double quote","ssh hiptrx \"cd hipfire && cargo fmt --all\""),
 ("ALLOW","ssh unrelated","ssh hipx 'cd hipfire && cargo build --release'"),

 ("ALLOW","hatch on later line","echo start\nHIPFIRE_ALLOW_FMT=1 cargo fmt --all"),
 ("ALLOW","hatch after &&","cd crates && HIPFIRE_ALLOW_FMT=1 cargo fmt"),
 ("DENY","multiline no hatch","echo start\ncargo fmt --all"),
]
fails=0
for exp,label,cmd in cases:
    p=subprocess.run(G,input=json.dumps({"tool_name":"Bash","tool_input":{"command":cmd}}),capture_output=True,text=True)
    got="DENY" if '"permissionDecision": "deny"' in p.stdout or '"permissionDecision":"deny"' in p.stdout else "ALLOW"
    ok = got==exp
    fails += 0 if ok else 1
    print(("  ok   " if ok else "  FAIL ")+f"[{got}] {label}"+("" if ok else f"  (expected {exp}) stderr={p.stderr[:200]}"))
print(("\nALL PASS" if not fails else f"\n{fails} FAILURES"))
sys.exit(1 if fails else 0)
