<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: LoaderDaemon

Client JSONL → daemon match(type) → load: tp>1 load_model_ep_with_kv_mode else load_model_with_gemma4_drafter → Carrier::load → LoadedModel{state|ep}. Post-load stage_continuous_batch may set EpArch::Qwen35.batch. Generate: set_active_attempt_id + activate_terminal_control → select_generation_route (EP only 9/10 named; 5|6 Unknown) → generate bodies. MoE EP intended serve is drive_qwen35_ep_continuous_batch only when batch staged and serve_continuous_batch. Errors: emit_active_attempt_error (TLS id) vs emit_uncorrelated_error (attempt 0) vs write_error_envelope (explicit). Unload: pflash drafter free before unload_model; ensure_vmm_ready_for_load on empty unload and pre-load.
