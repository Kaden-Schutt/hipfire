// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generation route matrix tests.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

    

    /// Baseline inputs that select nothing special (unknown arch, no EP/PP/spec).
    fn base() -> GenerationRouteInputs {
        GenerationRouteInputs {
            arch_id: 255,
            ep: false,
            pp: 1,
            has_speculator: false,
            speculator_is_mtp: false,
            deepseek4_spec_requested: false,
            ngram_can_sample: false,
            temp: 0.0,
            user_explicit_sampling: false,
            min_p: None,
            force_ar_chat: false,
            temp_spec_env_off: false,
            fast_sample_on: true,
            supports_temp_swor: false,
            kv_adaptive: false,
        }
    }

    #[test]
    fn dspark_request_is_independent_of_mtp_mode() {
        assert!(deepseek4_spec_requested_from_policy(
            Some("dspark"),
            "off",
            "off",
            false,
        ));
        assert!(!deepseek4_spec_requested_from_policy(
            None, "off", "auto", true,
        ));
        assert!(deepseek4_spec_requested_from_policy(
            None, "auto", "auto", true,
        ));
    }

    /// One canonical input row that selects each ALL variant (coverage guard).
    /// New enum variants must add a row here or `route_capability_table_covers_all_variants` fails.
    fn capability_rows() -> Vec<(GenerationRoute, GenerationRouteInputs)> {
        vec![
            (
                GenerationRoute::QwenAr,
                GenerationRouteInputs {
                    arch_id: 5,
                    ..base()
                },
            ),
            (
                GenerationRoute::QwenDflash,
                GenerationRouteInputs {
                    arch_id: 5,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::Qwen2Ar,
                GenerationRouteInputs {
                    arch_id: 7,
                    ..base()
                },
            ),
            (
                GenerationRoute::Qwen2Spec,
                GenerationRouteInputs {
                    arch_id: 7,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::Deepseek4Ar,
                GenerationRouteInputs {
                    arch_id: 9,
                    ..base()
                },
            ),
            (
                GenerationRoute::Deepseek4Ep,
                GenerationRouteInputs {
                    arch_id: 9,
                    ep: true,
                    // EP beats DS4 arch short-circuit even with spec flags set.
                    has_speculator: true,
                    deepseek4_spec_requested: true,
                    ..base()
                },
            ),
            (
                GenerationRoute::Deepseek4Spec,
                GenerationRouteInputs {
                    arch_id: 9,
                    has_speculator: true,
                    deepseek4_spec_requested: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::CohereAr,
                GenerationRouteInputs {
                    arch_id: 12,
                    ..base()
                },
            ),
            (
                GenerationRoute::CohereSpec,
                GenerationRouteInputs {
                    arch_id: 12,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::MiniMaxAr,
                GenerationRouteInputs {
                    arch_id: 10,
                    ..base()
                },
            ),
            (
                GenerationRoute::MiniMaxEp,
                GenerationRouteInputs {
                    arch_id: 10,
                    ep: true,
                    has_speculator: true,
                    ..base()
                },
            ),
            (
                GenerationRoute::MiniMaxSpec,
                GenerationRouteInputs {
                    arch_id: 10,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::LfmAr,
                GenerationRouteInputs {
                    arch_id: 11,
                    ..base()
                },
            ),
            (
                GenerationRoute::LfmSpec,
                GenerationRouteInputs {
                    arch_id: 11,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::LlamaAr,
                GenerationRouteInputs {
                    arch_id: 0,
                    ..base()
                },
            ),
            (
                GenerationRoute::LlamaSpec,
                GenerationRouteInputs {
                    arch_id: 0,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::PipelineParallel,
                GenerationRouteInputs {
                    arch_id: 5,
                    pp: 2,
                    // PP still beats spec when no arch short-circuit.
                    has_speculator: true,
                    ..base()
                },
            ),
            (
                GenerationRoute::DotsOcr,
                GenerationRouteInputs {
                    arch_id: 8,
                    ..base()
                },
            ),
            (
                GenerationRoute::GlimmerAr,
                GenerationRouteInputs {
                    arch_id: 14,
                    ..base()
                },
            ),
            (
                GenerationRoute::GlimmerSpec,
                GenerationRouteInputs {
                    arch_id: 14,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::Unknown,
                GenerationRouteInputs {
                    arch_id: 99,
                    ..base()
                },
            ),
        ]
    }

    /// Exact proven-safe producer set (contract).
    const SAFE_ROUTES: &[GenerationRoute] = &[
        GenerationRoute::QwenAr,
        GenerationRoute::QwenDflash,
        GenerationRoute::Deepseek4Ar,
        GenerationRoute::Deepseek4Ep,
        GenerationRoute::Deepseek4Spec,
        GenerationRoute::GlimmerAr,
        GenerationRoute::GlimmerSpec,
    ];

    /// Pure gate model mirroring generate()'s tools preflight:
    /// deny before RNG/gen_start when tools nonempty && !supports_tools.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct GateOutcome {
        allowed: bool,
        error_count: usize,
        class: Option<&'static str>,
        retryable: Option<bool>,
        mutated_generation_side: bool,
        route: GenerationRoute,
    }

    fn pure_tools_gate(route: GenerationRoute, tools_nonempty: bool) -> GateOutcome {
        if tools_nonempty && !route.supports_tools() {
            GateOutcome {
                allowed: false,
                error_count: 1,
                class: Some("unsupported"),
                retryable: Some(false),
                mutated_generation_side: false,
                route,
            }
        } else {
            GateOutcome {
                allowed: true,
                error_count: 0,
                class: None,
                retryable: None,
                mutated_generation_side: false,
                route,
            }
        }
    }

    #[test]
    fn route_capability_table_covers_all_variants() {
        let rows = capability_rows();
        assert_eq!(
            rows.len(),
            GenerationRoute::ALL.len(),
            "capability table must list every GenerationRoute::ALL variant"
        );
        for &variant in GenerationRoute::ALL {
            let hit = rows.iter().any(|(r, _)| *r == variant);
            assert!(
                hit,
                "missing capability row for {:?}; add an explicit selector input",
                variant
            );
        }
        // Each row's selector must actually produce the labeled route.
        for (expected, inputs) in &rows {
            let got = select_generation_route(inputs);
            assert_eq!(
                got, *expected,
                "capability row for {:?} selected {:?}",
                expected, got
            );
        }
    }

    #[test]
    fn route_matrix_tools_absent_and_present() {
        for (route, inputs) in capability_rows() {
            let selected = select_generation_route(&inputs);
            assert_eq!(selected, route);

            let safe = SAFE_ROUTES.contains(&route);
            assert_eq!(
                route.supports_tools(),
                safe,
                "{:?} supports_tools mismatch vs SAFE_ROUTES",
                route
            );

            // Tools absent: always allowed, zero errors, no mutation.
            let absent = pure_tools_gate(route, false);
            assert!(absent.allowed, "{:?} tools-absent must allow", route);
            assert_eq!(absent.error_count, 0);
            assert!(absent.class.is_none());
            assert!(!absent.mutated_generation_side);

            // Tools present: safe allows; unsafe emits exactly one nonretryable unsupported.
            let present = pure_tools_gate(route, true);
            if safe {
                assert!(present.allowed, "{:?} safe+tools must allow", route);
                assert_eq!(present.error_count, 0);
                assert!(!present.mutated_generation_side);
            } else {
                assert!(!present.allowed, "{:?} unsafe+tools must deny", route);
                assert_eq!(present.error_count, 1, "{:?} exactly one error", route);
                assert_eq!(present.class, Some("unsupported"));
                assert_eq!(present.retryable, Some(false));
                assert!(
                    !present.mutated_generation_side,
                    "{:?} deny must not mutate generation side",
                    route
                );
            }
        }
    }

    #[test]
    fn exact_safe_set_is_qwen_ar_dflash_ds4_ar_ep_spec_and_glimmer_ar_spec() {
        let mut from_all: Vec<GenerationRoute> = GenerationRoute::ALL
            .iter()
            .copied()
            .filter(|r| r.supports_tools())
            .collect();
        from_all.sort_by_key(|r| r.name());
        let mut expected = SAFE_ROUTES.to_vec();
        expected.sort_by_key(|r| r.name());
        assert_eq!(from_all, expected);
        assert_eq!(from_all.len(), 7);
        // Negative: every other ALL member is denied for tools.
        for &r in GenerationRoute::ALL {
            if !SAFE_ROUTES.contains(&r) {
                assert!(!r.supports_tools(), "{:?} must not be tool-safe", r);
            }
        }
    }

    #[test]
    fn precedence_ep_before_arch_short_circuit() {
        // EP on DS4 with spec requested → Deepseek4Ep, not Spec/Ar.
        let i = GenerationRouteInputs {
            arch_id: 9,
            ep: true,
            has_speculator: true,
            deepseek4_spec_requested: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Deepseek4Ep);
        // EP on MiniMax with n-gram spec → MiniMaxEp, not Spec.
        let i = GenerationRouteInputs {
            arch_id: 10,
            ep: true,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::MiniMaxEp);
        // EP on unregistered arch → Unknown (still EP-first).
        let i = GenerationRouteInputs {
            arch_id: 5,
            ep: true,
            has_speculator: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Unknown);
    }

    #[test]
    fn qwen_ep_batch_semantic_route_clears_ep_for_qwen_ar() {
        // Global selector: arch 6 + EP topology → Unknown (EP short-circuit).
        let with_ep = GenerationRouteInputs {
            arch_id: 6,
            ep: true,
            ..base()
        };
        assert_eq!(select_generation_route(&with_ep), GenerationRoute::Unknown);
        // Batch eligibility clears EP after independent topology gates so the
        // non-spec Qwen AR ladder remains reachable (exact callsite invariant).
        let cleared = GenerationRouteInputs {
            arch_id: 6,
            ep: false,
            ..base()
        };
        assert_eq!(select_generation_route(&cleared), GenerationRoute::QwenAr);
    }

    #[test]
    fn precedence_arch_short_circuit_before_pp() {
        // Qwen2 + pp>1 still short-circuits to Qwen2, never PipelineParallel.
        let i = GenerationRouteInputs {
            arch_id: 7,
            pp: 4,
            has_speculator: false,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Qwen2Ar);
        let i = GenerationRouteInputs {
            arch_id: 9,
            pp: 2,
            has_speculator: false,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Deepseek4Ar);
        let i = GenerationRouteInputs {
            arch_id: 11,
            pp: 2,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::LfmSpec);
        let i = GenerationRouteInputs {
            arch_id: 12,
            pp: 2,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::CohereAr);
        let i = GenerationRouteInputs {
            arch_id: 10,
            pp: 2,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::MiniMaxAr);
        let i = GenerationRouteInputs {
            arch_id: 8,
            pp: 2,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::DotsOcr);
    }

    #[test]
    fn precedence_pp_before_qwen_mtp() {
        let i = GenerationRouteInputs {
            arch_id: 5,
            pp: 2,
            temp: 0.0,
            has_speculator: true,
            ..base()
        };
        assert_eq!(
            select_generation_route(&i),
            GenerationRoute::PipelineParallel
        );
    }

    #[test]
    fn mtp_speculator_routes_through_qwen_dflash() {
        // Greedy MTP uses the generic QwenDflash wrapper.
        let i = GenerationRouteInputs {
            arch_id: 6,
            has_speculator: true,
            speculator_is_mtp: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
        // Sampled MTP with user-explicit sampling and min_p stays on spec
        // when supports_temp_verify — unlike DFlash-specific restrictions.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            speculator_is_mtp: true,
            supports_temp_swor: true,
            temp: 0.7,
            user_explicit_sampling: true,
            min_p: Some(0.05),
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
        // DFlash (not MTP) with user-explicit sampling still falls to AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            speculator_is_mtp: false,
            supports_temp_swor: true,
            temp: 0.7,
            user_explicit_sampling: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
        // MTP without supports_temp_verify at temp>0 falls to AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            speculator_is_mtp: true,
            supports_temp_swor: false,
            temp: 0.7,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
    }

    #[test]
    fn precedence_dflash_vs_ar() {
        // Qwen greedy + speculator → DFlash.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
        // force_ar_chat → AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            temp: 0.0,
            force_ar_chat: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
        // No speculator → AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: false,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
        // kv_adaptive blocks Qwen DFlash → AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            temp: 0.0,
            kv_adaptive: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
        // Llama greedy + spec → LlamaSpec; without → LlamaAr.
        let i = GenerationRouteInputs {
            arch_id: 1,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::LlamaSpec);
        let i = GenerationRouteInputs {
            arch_id: 1,
            has_speculator: false,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::LlamaAr);
    }

    #[test]
    fn precedence_arch_spec_vs_ar_matrix() {
        // Qwen2
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 7,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::Qwen2Spec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 7,
                has_speculator: true,
                temp: 0.7,
                ngram_can_sample: false,
                ..base()
            }),
            GenerationRoute::Qwen2Ar
        );
        // DeepSeek4
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 9,
                has_speculator: true,
                deepseek4_spec_requested: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::Deepseek4Spec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 9,
                has_speculator: true,
                deepseek4_spec_requested: false,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::Deepseek4Ar
        );
        // Cohere
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 12,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::CohereSpec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 12,
                has_speculator: false,
                ..base()
            }),
            GenerationRoute::CohereAr
        );
        // MiniMax
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 10,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::MiniMaxSpec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 10,
                has_speculator: false,
                ..base()
            }),
            GenerationRoute::MiniMaxAr
        );
        // LFM
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 11,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::LfmSpec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 11,
                has_speculator: true,
                temp: 0.8,
                ngram_can_sample: false,
                ..base()
            }),
            GenerationRoute::LfmAr
        );
        // dots + unknown
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 8,
                has_speculator: true,
                pp: 2,
                ..base()
            }),
            GenerationRoute::DotsOcr
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 42,
                ..base()
            }),
            GenerationRoute::Unknown
        );
    }

    #[test]
    fn pure_gate_unsafe_tools_one_nonretryable_no_mutation() {
        for &route in GenerationRoute::ALL {
            if route.supports_tools() {
                continue;
            }
            let o = pure_tools_gate(route, true);
            assert_eq!(o.error_count, 1);
            assert_eq!(o.class, Some("unsupported"));
            assert_eq!(o.retryable, Some(false));
            assert!(!o.allowed);
            assert!(!o.mutated_generation_side);
            // Correlated: outcome carries the denied route identity.
            assert_eq!(o.route, route);
        }
    }

    #[test]
    fn pure_gate_tools_absent_always_allowed() {
        for &route in GenerationRoute::ALL {
            let o = pure_tools_gate(route, false);
            assert!(o.allowed, "{:?} tools-absent", route);
            assert_eq!(o.error_count, 0);
            assert!(!o.mutated_generation_side);
        }
    }

    #[test]
    fn all_variant_count_is_twenty_one() {
        // Pin count so accidental ALL edits surface here too.
        assert_eq!(GenerationRoute::ALL.len(), 21);
        assert_eq!(capability_rows().len(), 21);
    }
