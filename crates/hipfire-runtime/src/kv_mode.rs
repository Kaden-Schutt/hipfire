//! Pure KV-mode resolution: maps a raw `HIPFIRE_KV_MODE` string + a per-site
//! `KvModePolicy` to a concrete, validated [`KvMode`]. No GPU, no env read, no
//! allocation — the caller reads the env/override and passes the raw string.
//!
//! This replaces 6 duplicated `match kv_mode.as_str()` ladders. The companion
//! dispatcher [`crate::llama::KvCache::from_mode`] turns the resolved mode into
//! the actual allocating constructor call.

pub use saddle_core::kv::KvMode;

/// Per-site alias table + accepted set + default. One const per load site.
pub struct KvModePolicy {
    /// Human label for diagnostics ("qwen35-hfq", "qwen35-pp", ...).
    pub site: &'static str,
    /// Site-LOCAL alias expansion: raw string → the [`KvMode`] it denotes ON
    /// THIS SITE, or `None` for "this site does not recognize this string".
    /// `"auto"` expands to `Asym3Auto` on site 3 but to `Q8` on site 6 — that
    /// divergence lives here, per-policy. There is no global alias table.
    pub normalize_alias: fn(&str) -> Option<KvMode>,
    /// Modes this site has a constructor for. `resolve` rejects anything else
    /// and falls to `default`. (The `Asym3Auto` sentinel is listed here on
    /// site 3 so it survives the accept check; `resolve` collapses it after.)
    pub accepted: &'static [KvMode],
    /// What an unrecognized / recognized-but-unaccepted string resolves to.
    pub default: KvMode,
}

/// The result of [`resolve`]: the concrete mode plus an optional operator-facing
/// warning (preserves each ladder's "unrecognized, defaulting to …" diagnostic).
pub struct ResolveResult {
    /// Guaranteed NOT `KvMode::Asym3Auto`.
    pub mode: KvMode,
    /// `Some` when a non-empty raw input was not honored as requested. `None`
    /// when the request was honored OR when the input was unset (`""`).
    pub warning: Option<&'static str>,
}

use KvMode::*;

/// Shared alias table for the two single-GPU full-ladder sites (1 & 2).
/// "auto"/"turbo"/"turbo3" mean asym3 HERE (qwen35 is always head_dim == 256),
/// so they map to the concrete `Asym3`, NOT the `Asym3Auto` sentinel.
fn normalize_full(raw: &str) -> Option<KvMode> {
    match raw {
        "q8" => Some(Q8),
        "asym2" | "turbo2" => Some(Asym2),
        "asym3" | "turbo3" | "turbo" | "auto" => Some(Asym3),
        "asym4" | "turbo4" => Some(Asym4),
        "fwht2" => Some(Fwht2),
        "fwht3" => Some(Fwht3),
        "fwht4" => Some(Fwht4),
        _ => None, // "" → default (silent); unrecognized → default (+warn)
    }
}

const FULL_LADDER: &[KvMode] = &[Q8, Asym2, Asym3, Asym4, Fwht2, Fwht3, Fwht4];

/// Site 1 — qwen35 HFQ in-place carrier (pp=1). carrier.rs:65. Default asym3.
pub const QWEN35_HFQ_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-hfq",
    normalize_alias: normalize_full,
    accepted: FULL_LADDER,
    default: Asym3,
};

/// Site 2 — qwen35 PaRo loader (filtered/capped). carriers.rs:333. Default q8.
/// `Asym3` is deliberately ABSENT from `accepted`: the real ladder has no bare
/// `"asym3"` arm, so `HIPFIRE_KV_MODE=asym3` yields q8 here. `normalize_full`
/// maps the string to `Some(Asym3)`, the accept check fails, default = q8.
pub const QWEN35_PARO_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-paro",
    normalize_alias: normalize_full,
    accepted: &[Q8, Asym2, Asym4, Fwht2, Fwht3, Fwht4],
    default: Q8,
};

/// Site 3 — Dir/safetensors llama/qwen3 PaRo (capped, flat n_layers).
/// carriers.rs:492. The ONLY site with a head_dim-conditional, carried by the
/// `Asym3Auto` sentinel. Explicit `"asym3"` is UNCONDITIONAL (panics @128 in
/// the constructor — current behavior, preserved).
fn normalize_dir(raw: &str) -> Option<KvMode> {
    match raw {
        "q8" => Some(Q8),
        "asym3" => Some(Asym3), // UNCONDITIONAL
        "asym4" | "turbo4" => Some(Asym4),
        "" | "auto" | "turbo" | "turbo3" => Some(Asym3Auto), // CONDITIONAL sentinel
        _ => None,                                           // asym2/fwht*/garbage → default
    }
}
pub const DIR_SAFETENSORS_POLICY: KvModePolicy = KvModePolicy {
    site: "dir-safetensors",
    normalize_alias: normalize_dir,
    accepted: &[Q8, Asym3, Asym4, Asym3Auto],
    default: Q8, // recognized-but-unaccepted AND unrecognized both → plain q8
};

/// Site 4 — llama HFQ carrier. carrier.rs:21. Default q8. Accepts the three
/// modes that have Flat KV constructors: Q8, Asym3, Asym4. Asym3/Asym4 are
/// gated in `KvCache::from_mode` with a clean error at head_dim≠256 (most llama
/// models are head_dim=128; only validated at 256 via Qwen 3.5). Asym2/Fwht*
/// have no Flat constructor and silently fall to q8 (unimplemented, not broken).
pub const LLAMA_HFQ_POLICY: KvModePolicy = KvModePolicy {
    site: "llama-hfq",
    normalize_alias: normalize_full,
    accepted: &[Q8, Asym3, Asym4],
    default: Q8,
};

/// Site 5 — minimax / lfm2moe HFQ carriers. minimax.rs:799, lfm2moe.rs:963.
/// Hardcoded q8 today (no env in scope; caller passes literal "").
pub const HFQ_Q8_ONLY_POLICY: KvModePolicy = KvModePolicy {
    site: "hfq-q8-only",
    normalize_alias: normalize_full,
    accepted: &[Q8],
    default: Q8,
};

/// Site 6 — qwen35 HFQ pp>1 (multi-GPU). carriers.rs:131. Default q8,
/// `"auto" → q8` (NOT asym3 — opposite of site 3), narrow accept
/// {q8, asym3, fwht3, fwht2}. This is why aliases can't be globally normalized.
fn normalize_pp(raw: &str) -> Option<KvMode> {
    match raw {
        "q8" | "auto" | "" => Some(Q8), // "auto" means q8 HERE
        "asym3" | "turbo3" | "turbo" => Some(Asym3),
        "fwht3" => Some(Fwht3),
        "fwht2" => Some(Fwht2),
        _ => None, // incl. asym2/asym4 → default (+warn "for pp>1")
    }
}
pub const QWEN35_PP_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-pp",
    normalize_alias: normalize_pp,
    accepted: &[Q8, Asym3, Fwht3, Fwht2],
    default: Q8,
};

/// Site 7 — maple (arch 15). Before this site existed, maple hardcoded
/// `KvCache::new_gpu_q8` and `--kv-mode` was a silent no-op for arch 15.
///
/// The accept set is deliberately just {Q8, Bf16}. Every other mode in the
/// ladder is a rotated or block-quantized tier whose attention kernels have NO
/// sliding-window variant, and Maple is 3:1 sliding(512)/global — a tier that
/// cannot carry the window would attend the full context on the sliding layers
/// and be silently WRONG at ctx > 512, not merely slower. Accepting them and
/// warning is the wrong trade here; refusing to the q8 default is right.
///
/// `"bf16"` is the only new name, and it is deliberately NOT added to
/// `normalize_full`: no other site can allocate a bf16 cache, so putting it
/// there would let `HIPFIRE_KV_MODE=bf16` on qwen35 normalize successfully and
/// then fall to that site's default — a silent downgrade instead of a warning.
fn normalize_maple(raw: &str) -> Option<KvMode> {
    match raw {
        "q8" | "auto" | "" => Some(Q8),
        "bf16" => Some(Bf16),
        _ => None, // every rotated/quantized tier → default (+warn)
    }
}
pub const MAPLE_POLICY: KvModePolicy = KvModePolicy {
    site: "maple",
    normalize_alias: normalize_maple,
    accepted: &[Q8, Bf16],
    default: Q8,
};

/// Pure: `&str + &'static policy + usize → ResolveResult`. No GPU, no env read.
pub fn resolve(raw: &str, policy: &KvModePolicy, head_dim: usize) -> ResolveResult {
    // 1. site-LOCAL alias expansion.
    let requested: Option<KvMode> = (policy.normalize_alias)(raw);

    // 2. accept only if the site supports it; else fall to the site's
    //    (unconditional) default. A non-empty raw that normalize rejected
    //    (None) OR that normalized to an unaccepted mode is a request the
    //    operator made but won't get — warn. Unset ("") defaults silently.
    let (mode, warning) = match requested {
        Some(m) if policy.accepted.contains(&m) => (m, None),
        _ => {
            let warning = if raw.is_empty() {
                None
            } else {
                Some("unrecognized or unsupported HIPFIRE_KV_MODE; using site default")
            };
            (policy.default, warning)
        }
    };

    // 3. collapse the head_dim-conditional sentinel. Only site 3 produces it.
    let mode = match mode {
        KvMode::Asym3Auto => {
            if head_dim == 256 {
                KvMode::Asym3
            } else {
                KvMode::Q8
            }
        }
        m => m,
    };

    ResolveResult { mode, warning }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truth_table_maple() {
        let p = &MAPLE_POLICY;
        assert_eq!(resolve("bf16", p, 128).mode, KvMode::Bf16);
        assert_eq!(resolve("q8", p, 128).mode, KvMode::Q8);
        // Unset and "auto" both mean q8, SILENTLY — this is the shipped
        // default and must not print a warning on every load.
        assert_eq!(resolve("", p, 128).mode, KvMode::Q8);
        assert!(resolve("", p, 128).warning.is_none());
        assert_eq!(resolve("auto", p, 128).mode, KvMode::Q8);
        assert!(resolve("auto", p, 128).warning.is_none());

        // Every ROTATED / block-quantized tier must be REFUSED to q8 and warn.
        // These have no sliding-window attention kernel, so silently accepting
        // one would make Maple's sliding layers attend the full context and be
        // wrong past 512 tokens rather than merely slower.
        for m in [
            "asym2", "asym3", "asym4", "fwht2", "fwht3", "fwht4", "turbo",
        ] {
            let r = resolve(m, p, 128);
            assert_eq!(r.mode, KvMode::Q8, "{m} must fall back to q8");
            assert!(r.warning.is_some(), "{m} must warn, not silently downgrade");
        }
        let garbage = resolve("garbage", p, 128);
        assert_eq!(garbage.mode, KvMode::Q8);
        assert!(garbage.warning.is_some());
    }

    #[test]
    fn bf16_is_maple_only() {
        // NEGATIVE CONTROL: "bf16" must not be a globally-known alias. No other
        // site can allocate a bf16 cache, so if `normalize_full` learned the
        // name, HIPFIRE_KV_MODE=bf16 on qwen35 would normalize fine and then
        // silently fall to that site's default. It must warn instead.
        for p in [
            &QWEN35_HFQ_POLICY,
            &QWEN35_PARO_POLICY,
            &LLAMA_HFQ_POLICY,
            &QWEN35_PP_POLICY,
            &DIR_SAFETENSORS_POLICY,
        ] {
            let r = resolve("bf16", p, 256);
            assert_ne!(r.mode, KvMode::Bf16, "site {} must not accept bf16", p.site);
            assert!(
                r.warning.is_some(),
                "site {} must WARN on bf16, not silently default",
                p.site
            );
        }
    }

    #[test]
    fn truth_table_qwen35_hfq() {
        let p = &QWEN35_HFQ_POLICY;
        assert_eq!(resolve("q8", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("asym4", p, 256).mode, KvMode::Asym4);
        assert_eq!(resolve("turbo4", p, 256).mode, KvMode::Asym4);
        assert_eq!(resolve("asym2", p, 256).mode, KvMode::Asym2);
        assert_eq!(resolve("turbo2", p, 256).mode, KvMode::Asym2);
        assert_eq!(resolve("asym3", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("turbo3", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("turbo", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("auto", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("fwht3", p, 256).mode, KvMode::Fwht3);
        assert_eq!(resolve("fwht2", p, 256).mode, KvMode::Fwht2);
        assert_eq!(resolve("fwht4", p, 256).mode, KvMode::Fwht4);
        assert_eq!(resolve("", p, 256).mode, KvMode::Asym3); // default, silent
        assert!(resolve("", p, 256).warning.is_none());
        assert_eq!(resolve("garbage", p, 256).mode, KvMode::Asym3); // unrecognized → default
        assert!(resolve("garbage", p, 256).warning.is_some()); // ...and WARNS
    }

    #[test]
    fn truth_table_qwen35_paro_no_asym3_arm() {
        let p = &QWEN35_PARO_POLICY;
        assert_eq!(resolve("asym3", p, 256).mode, KvMode::Q8); // no-asym3-arm quirk → q8
        assert!(resolve("asym3", p, 256).warning.is_some());
        assert_eq!(resolve("turbo3", p, 256).mode, KvMode::Q8); // alias of asym3 → q8 too
        assert_eq!(resolve("auto", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("asym4", p, 256).mode, KvMode::Asym4);
        assert_eq!(resolve("asym2", p, 256).mode, KvMode::Asym2);
        assert_eq!(resolve("fwht2", p, 256).mode, KvMode::Fwht2);
        assert_eq!(resolve("fwht3", p, 256).mode, KvMode::Fwht3);
        assert_eq!(resolve("fwht4", p, 256).mode, KvMode::Fwht4);
        assert_eq!(resolve("", p, 256).mode, KvMode::Q8); // default
    }

    #[test]
    fn truth_table_dir_safetensors_all_branches() {
        let p = &DIR_SAFETENSORS_POLICY;
        // auto-set: head_dim-CONDITIONAL
        assert_eq!(resolve("", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("", p, 128).mode, KvMode::Q8);
        assert_eq!(resolve("auto", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("auto", p, 128).mode, KvMode::Q8);
        assert_eq!(resolve("turbo", p, 128).mode, KvMode::Q8);
        assert_eq!(resolve("turbo3", p, 128).mode, KvMode::Q8);
        // explicit asym3: UNCONDITIONAL (asym3 even @128 — constructor then panics)
        assert_eq!(resolve("asym3", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("asym3", p, 128).mode, KvMode::Asym3);
        // asym4
        assert_eq!(resolve("asym4", p, 256).mode, KvMode::Asym4);
        assert_eq!(resolve("turbo4", p, 256).mode, KvMode::Asym4);
        // recognized-but-unaccepted + unrecognized: UNCONDITIONAL q8
        assert_eq!(resolve("asym2", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("fwht3", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("garbage", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("q8", p, 256).mode, KvMode::Q8);
    }

    #[test]
    fn truth_table_qwen35_pp_auto_is_q8() {
        let p = &QWEN35_PP_POLICY;
        assert_eq!(resolve("auto", p, 256).mode, KvMode::Q8); // "auto" → q8 here
        assert_eq!(resolve("", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("q8", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("asym3", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("turbo3", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("fwht2", p, 256).mode, KvMode::Fwht2);
        assert_eq!(resolve("fwht3", p, 256).mode, KvMode::Fwht3);
        assert_eq!(resolve("asym2", p, 256).mode, KvMode::Q8); // not accepted → default
        assert!(resolve("asym2", p, 256).warning.is_some());
        assert_eq!(resolve("asym4", p, 256).mode, KvMode::Q8); // not accepted → default
        assert!(resolve("asym4", p, 256).warning.is_some());
    }

    #[test]
    fn truth_table_llama_hfq_expanded() {
        let p = &LLAMA_HFQ_POLICY;
        // Default and explicit q8 → Q8.
        assert_eq!(resolve("", p, 128).mode, KvMode::Q8);
        assert_eq!(resolve("", p, 256).mode, KvMode::Q8); // default is Q8, not Asym3
        assert_eq!(resolve("q8", p, 128).mode, KvMode::Q8);
        // Asym3/Asym4 accepted — reach from_mode where head_dim gate fires.
        assert_eq!(resolve("asym3", p, 128).mode, KvMode::Asym3);
        assert_eq!(resolve("asym3", p, 256).mode, KvMode::Asym3);
        assert_eq!(resolve("asym4", p, 128).mode, KvMode::Asym4);
        assert_eq!(resolve("turbo3", p, 128).mode, KvMode::Asym3);
        assert_eq!(resolve("turbo4", p, 256).mode, KvMode::Asym4);
        // auto/turbo normalize to Asym3 (accepted → pass through).
        assert_eq!(resolve("auto", p, 256).mode, KvMode::Asym3);
        // Unaccepted modes (asym2/fwht*) silently fall to Q8 default.
        assert_eq!(resolve("asym2", p, 128).mode, KvMode::Q8);
        assert!(resolve("asym2", p, 128).warning.is_some());
        assert_eq!(resolve("fwht3", p, 256).mode, KvMode::Q8);
        // Unrecognized strings also fall to Q8 with warning.
        assert_eq!(resolve("garbage", p, 256).mode, KvMode::Q8);
        assert!(resolve("garbage", p, 256).warning.is_some());
    }

    #[test]
    fn truth_table_hfq_q8_only() {
        let p = &HFQ_Q8_ONLY_POLICY;
        assert_eq!(resolve("", p, 128).mode, KvMode::Q8);
        assert_eq!(resolve("q8", p, 128).mode, KvMode::Q8);
        assert_eq!(resolve("asym3", p, 128).mode, KvMode::Q8); // unaccepted → q8
        assert_eq!(resolve("fwht4", p, 256).mode, KvMode::Q8);
        assert_eq!(resolve("garbage", p, 256).mode, KvMode::Q8);
    }

    #[test]
    fn resolve_never_returns_sentinel() {
        // The sentinel must always collapse before return, on every policy.
        for raw in ["", "auto", "turbo", "turbo3", "asym3", "q8", "garbage"] {
            for hd in [128usize, 256] {
                assert_ne!(
                    resolve(raw, &DIR_SAFETENSORS_POLICY, hd).mode,
                    KvMode::Asym3Auto
                );
            }
        }
    }
}
