	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.text
	.protected	gemm_gate_up_hfq4g256_wmma_gfx12 ; -- Begin function gemm_gate_up_hfq4g256_wmma_gfx12
	.globl	gemm_gate_up_hfq4g256_wmma_gfx12
	.p2align	8
	.type	gemm_gate_up_hfq4g256_wmma_gfx12,@function
gemm_gate_up_hfq4g256_wmma_gfx12:       ; @gemm_gate_up_hfq4g256_wmma_gfx12
; %bb.0:
	s_load_b128 s[12:15], s[0:1], 0x28
	s_lshl_b32 s17, ttmp9, 4
	s_lshl_b32 s18, ttmp7, 4
	s_wait_kmcnt 0x0
	s_add_co_i32 s16, s13, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_cmp_ge_i32 s17, s16
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s15
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_23
; %bb.1:
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x20
	v_dual_mov_b32 v7, 0 :: v_dual_and_b32 v8, 15, v0
	v_lshrrev_b32_e32 v20, 4, v0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v19, s18, v8
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	v_mov_b32_e32 v0, 0
	s_cmp_lt_i32 s14, 0x100
	v_cmp_gt_i32_e32 vcc_lo, s15, v19
	s_cbranch_scc1 .LBB0_6
; %bb.2:                                ; %.lr.ph
	v_or_b32_e32 v0, s17, v8
	s_add_co_i32 s18, s16, -1
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v5, s5 :: v_dual_add_nc_u32 v2, s17, v8
	v_cndmask_b32_e32 v6, 0, v19, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v4, s18, v0
	s_ashr_i32 s1, s14, 31
	v_ashrrev_i32_e32 v3, 31, v2
	s_ashr_i32 s19, s18, 31
	s_lshr_b32 s5, s1, 24
	v_cmp_gt_i32_e64 s0, s12, v4
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_lt_i64_e64 s1, s[18:19], v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v7, s12, 0, s0
	v_cndmask_b32_e64 v9, v2, s18, s1
	v_cndmask_b32_e64 v3, v3, s19, s1
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_nc_u32_e32 v8, v4, v7
	v_cndmask_b32_e64 v4, s7, v5, s0
	v_mov_b32_e32 v5, s4
	v_mad_co_u64_u32 v[0:1], null, v6, s14, 0
	v_ashrrev_i32_e32 v6, 31, v6
	v_ashrrev_i32_e32 v10, 31, v7
	s_add_co_i32 s4, s14, s5
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s4, s4, 8
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[1:2], null, v6, s14, v[1:2]
	v_sub_co_u32 v2, s1, v9, v7
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v6, null, v3, v10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s1, s4, 0x88
	v_cndmask_b32_e64 v3, s6, v5, s0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s0, s1, 31
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	v_mul_lo_u32 v7, v6, s1
	v_mad_co_u64_u32 v[5:6], null, v2, s1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v11, v2, s0
	v_lshlrev_b32_e32 v2, 4, v20
	v_mad_co_i64_i32 v[9:10], null, v8, s1, v[3:4]
	v_add_co_u32 v0, s0, s8, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s9, v1, s0
	v_lshl_or_b32 v5, v20, 2, v5
	v_add3_u32 v6, v6, v11, v7
	v_add_co_u32 v11, s0, v0, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, 0, v1, s0
	v_add_co_u32 v1, s0, v3, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v4, v6, s0
	v_mov_b32_e32 v0, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v13, s0, v1, 32
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, 0, v2, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_mov_b32 s1, 0
.LBB0_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_4 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s0, s1, 0x88
	s_mov_b32 s5, -4
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v15, s0, v9, s0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, 0, v10, s0
	global_load_b64 v[17:18], v[15:16], off
	v_dual_mov_b32 v16, v12 :: v_dual_mov_b32 v15, v11
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v8.l, v17
	v_cvt_f16_f32_e32 v8.h, v18
	v_dual_mov_b32 v18, v14 :: v_dual_mov_b32 v17, v13
.LBB0_4:                                ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_clause 0x3
	global_load_b32 v37, v[17:18], off offset:-24
	global_load_b32 v38, v[17:18], off offset:-16
	global_load_b32 v39, v[17:18], off offset:-8
	global_load_b32 v40, v[17:18], off
	s_clause 0x3
	global_load_b128 v[21:24], v[15:16], off
	global_load_b128 v[25:28], v[15:16], off offset:32
	global_load_b128 v[29:32], v[15:16], off offset:64
	global_load_b128 v[33:36], v[15:16], off offset:96
	v_add_co_u32 v17, s0, v17, 32
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, 0, v18, s0
	v_add_co_u32 v15, s0, 0x80, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, 0, v16, s0
	s_add_co_i32 s5, s5, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s5, 11
	s_wait_loadcnt 0x7
	v_and_b32_e32 v41, 15, v37
	v_bfe_u32 v42, v37, 4, 4
	v_bfe_u32 v43, v37, 8, 4
	v_bfe_u32 v44, v37, 12, 4
	v_bfe_u32 v45, v37, 16, 4
	v_bfe_u32 v46, v37, 20, 4
	v_bfe_u32 v47, v37, 24, 4
	v_lshrrev_b32_e32 v37, 28, v37
	s_wait_loadcnt 0x6
	v_and_b32_e32 v48, 15, v38
	v_bfe_u32 v49, v38, 4, 4
	v_bfe_u32 v50, v38, 8, 4
	v_bfe_u32 v51, v38, 12, 4
	v_bfe_u32 v52, v38, 16, 4
	v_bfe_u32 v53, v38, 20, 4
	v_bfe_u32 v54, v38, 24, 4
	v_lshrrev_b32_e32 v38, 28, v38
	s_wait_loadcnt 0x5
	v_and_b32_e32 v55, 15, v39
	v_bfe_u32 v56, v39, 4, 4
	v_bfe_u32 v57, v39, 8, 4
	v_bfe_u32 v58, v39, 12, 4
	v_bfe_u32 v59, v39, 16, 4
	v_bfe_u32 v60, v39, 20, 4
	v_bfe_u32 v61, v39, 24, 4
	v_lshrrev_b32_e32 v39, 28, v39
	s_wait_loadcnt 0x4
	v_and_b32_e32 v62, 15, v40
	v_bfe_u32 v63, v40, 4, 4
	v_bfe_u32 v64, v40, 8, 4
	v_bfe_u32 v65, v40, 12, 4
	v_bfe_u32 v66, v40, 16, 4
	v_bfe_u32 v67, v40, 20, 4
	v_bfe_u32 v68, v40, 24, 4
	v_lshrrev_b32_e32 v40, 28, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v69, v37
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v70, v38
	v_cvt_f32_ubyte0_e32 v71, v39
	v_cvt_f32_ubyte0_e32 v72, v40
	v_cvt_f16_f32_e32 v37.l, v41
	v_cvt_f16_f32_e32 v37.h, v42
	v_cvt_f16_f32_e32 v38.l, v43
	v_cvt_f16_f32_e32 v38.h, v44
	v_cvt_f16_f32_e32 v39.l, v45
	v_cvt_f16_f32_e32 v39.h, v46
	v_cvt_f16_f32_e32 v40.l, v47
	v_cvt_f16_f32_e32 v40.h, v69
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f16_f32_e32 v41.l, v48
	v_cvt_f16_f32_e32 v41.h, v49
	v_cvt_f16_f32_e32 v42.l, v50
	v_cvt_f16_f32_e32 v42.h, v51
	v_cvt_f16_f32_e32 v43.l, v52
	v_cvt_f16_f32_e32 v43.h, v53
	v_cvt_f16_f32_e32 v44.l, v54
	v_cvt_f16_f32_e32 v44.h, v70
	v_fma_f16 v37.l, v8.l, v37.l, v8.h
	v_fma_f16 v37.h, v8.l, v37.h, v8.h
	v_fma_f16 v38.l, v8.l, v38.l, v8.h
	v_fma_f16 v38.h, v8.l, v38.h, v8.h
	v_fma_f16 v39.l, v8.l, v39.l, v8.h
	v_fma_f16 v39.h, v8.l, v39.h, v8.h
	v_fma_f16 v40.l, v8.l, v40.l, v8.h
	v_fma_f16 v40.h, v8.l, v40.h, v8.h
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f16_f32_e32 v45.l, v55
	v_cvt_f16_f32_e32 v45.h, v56
	v_cvt_f16_f32_e32 v46.l, v57
	v_cvt_f16_f32_e32 v46.h, v58
	v_cvt_f16_f32_e32 v47.l, v59
	v_cvt_f16_f32_e32 v47.h, v60
	v_cvt_f16_f32_e32 v48.l, v61
	v_cvt_f16_f32_e32 v48.h, v71
	v_fma_f16 v41.l, v8.l, v41.l, v8.h
	v_fma_f16 v41.h, v8.l, v41.h, v8.h
	v_fma_f16 v42.l, v8.l, v42.l, v8.h
	v_fma_f16 v42.h, v8.l, v42.h, v8.h
	v_fma_f16 v43.l, v8.l, v43.l, v8.h
	v_fma_f16 v43.h, v8.l, v43.h, v8.h
	v_fma_f16 v44.l, v8.l, v44.l, v8.h
	v_fma_f16 v44.h, v8.l, v44.h, v8.h
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_f16 v[0:7], v[37:40], v[21:24], v[0:7]
	v_cvt_f16_f32_e32 v49.l, v62
	v_cvt_f16_f32_e32 v49.h, v63
	v_cvt_f16_f32_e32 v50.l, v64
	v_cvt_f16_f32_e32 v50.h, v65
	v_cvt_f16_f32_e32 v51.l, v66
	v_cvt_f16_f32_e32 v51.h, v67
	v_cvt_f16_f32_e32 v52.l, v68
	v_cvt_f16_f32_e32 v52.h, v72
	v_fma_f16 v45.l, v8.l, v45.l, v8.h
	v_fma_f16 v45.h, v8.l, v45.h, v8.h
	v_fma_f16 v46.l, v8.l, v46.l, v8.h
	v_fma_f16 v46.h, v8.l, v46.h, v8.h
	v_fma_f16 v47.l, v8.l, v47.l, v8.h
	v_fma_f16 v47.h, v8.l, v47.h, v8.h
	v_fma_f16 v48.l, v8.l, v48.l, v8.h
	v_fma_f16 v48.h, v8.l, v48.h, v8.h
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_f16 v[0:7], v[41:44], v[25:28], v[0:7]
	v_fma_f16 v21.l, v8.l, v49.l, v8.h
	v_fma_f16 v21.h, v8.l, v49.h, v8.h
	v_fma_f16 v22.l, v8.l, v50.l, v8.h
	v_fma_f16 v22.h, v8.l, v50.h, v8.h
	v_fma_f16 v23.l, v8.l, v51.l, v8.h
	v_fma_f16 v23.h, v8.l, v51.h, v8.h
	v_fma_f16 v24.l, v8.l, v52.l, v8.h
	v_fma_f16 v24.h, v8.l, v52.h, v8.h
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[0:7], v[45:48], v[29:32], v[0:7]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[0:7], v[21:24], v[33:36], v[0:7]
	s_cbranch_scc0 .LBB0_4
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=1
	v_add_co_u32 v13, s0, 0x88, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, 0, v14, s0
	v_add_co_u32 v11, s0, 0x200, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, 0, v12, s0
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s4
	s_cbranch_scc0 .LBB0_3
.LBB0_6:                                ; %._crit_edge
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_23
; %bb.7:                                ; %.preheader
	v_lshl_add_u32 v8, v20, 3, s17
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s16, v8
	s_cbranch_execz .LBB0_9
; %bb.8:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s11 :: v_dual_mov_b32 v10, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v8
	v_mov_b32_e32 v14, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v13, s3, v9, vcc_lo
	v_cndmask_b32_e32 v9, s13, v10, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	v_cndmask_b32_e32 v14, s2, v14, vcc_lo
	v_mad_co_i64_i32 v[9:10], null, v9, v19, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v11, v8, v11
	v_ashrrev_i32_e32 v12, 31, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v9, vcc_lo, v14, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v13, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	global_store_b32 v[9:10], v0, off
.LBB0_9:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v9, 1, v8
	v_ashrrev_i32_e32 v0, 31, v8
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s16, v9
	s_cbranch_execz .LBB0_11
; %bb.10:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v10, s11 :: v_dual_mov_b32 v11, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v9
	v_mov_b32_e32 v14, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v9, s13, v11, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	v_cndmask_b32_e32 v13, s3, v10, vcc_lo
	v_cndmask_b32_e32 v14, s2, v14, vcc_lo
	v_mad_co_i64_i32 v[9:10], null, v9, v19, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v12, 31, v11
	v_sub_co_u32 v11, s0, v8, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v12, null, v0, v12, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v9, vcc_lo, v14, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v13, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	global_store_b32 v[9:10], v1, off offset:4
.LBB0_11:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 2, v8
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s16, v1
	s_cbranch_execz .LBB0_13
; %bb.12:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s11 :: v_dual_mov_b32 v10, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mov_b32_e32 v13, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v1, s3, v9, vcc_lo
	v_cndmask_b32_e32 v9, s13, v10, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	v_cndmask_b32_e32 v13, s2, v13, vcc_lo
	v_mad_co_i64_i32 v[9:10], null, v9, v19, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v12, 31, v11
	v_sub_co_u32 v11, s0, v8, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v12, null, v0, v12, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v9, vcc_lo, v13, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v12, vcc_lo
	global_store_b32 v[9:10], v2, off offset:8
.LBB0_13:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 3, v8
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s16, v1
	s_cbranch_execz .LBB0_15
; %bb.14:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s11 :: v_dual_mov_b32 v9, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mov_b32_e32 v12, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, s13, v9, vcc_lo
	v_cndmask_b32_e64 v9, s12, 0, vcc_lo
	v_cndmask_b32_e32 v11, s3, v2, vcc_lo
	v_cndmask_b32_e32 v12, s2, v12, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v19, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v10, 31, v9
	v_sub_co_u32 v9, s0, v8, v9
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v10, null, v0, v10, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, v12, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v11, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v10, vcc_lo
	global_store_b32 v[1:2], v3, off offset:12
.LBB0_15:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 4, v8
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s16, v1
	s_cbranch_execz .LBB0_17
; %bb.16:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s11 :: v_dual_mov_b32 v3, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mov_b32_e32 v12, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v1, s13, v3, vcc_lo
	v_cndmask_b32_e64 v3, s12, 0, vcc_lo
	v_cndmask_b32_e32 v11, s3, v2, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v19, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v10, 31, v3
	v_sub_co_u32 v9, s0, v8, v3
	v_cndmask_b32_e32 v3, s2, v12, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v10, null, v0, v10, s0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, v3, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v11, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v10, vcc_lo
	global_store_b32 v[1:2], v4, off offset:16
.LBB0_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 5, v8
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s16, v1
	s_cbranch_execz .LBB0_19
; %bb.18:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s11 :: v_dual_mov_b32 v3, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mov_b32_e32 v10, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, s13, v3, vcc_lo
	v_cndmask_b32_e64 v3, s12, 0, vcc_lo
	v_cndmask_b32_e32 v9, s3, v2, vcc_lo
	v_cndmask_b32_e32 v10, s2, v10, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v19, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v4, 31, v3
	v_sub_co_u32 v3, s0, v8, v3
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v4, null, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, v10, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v9, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	global_store_b32 v[1:2], v5, off offset:20
.LBB0_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 6, v8
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s16, v1
	s_cbranch_execz .LBB0_21
; %bb.20:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s11 :: v_dual_mov_b32 v3, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mov_b32_e32 v9, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, s13, v3, vcc_lo
	v_cndmask_b32_e64 v3, s12, 0, vcc_lo
	v_cndmask_b32_e32 v5, s3, v2, vcc_lo
	v_cndmask_b32_e32 v9, s2, v9, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v19, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v4, 31, v3
	v_sub_co_u32 v3, s0, v8, v3
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v4, null, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, v9, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	global_store_b32 v[1:2], v6, off offset:24
.LBB0_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 7, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s16, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_23
; %bb.22:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s11 :: v_dual_mov_b32 v3, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mov_b32_e32 v6, s10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, s13, v3, vcc_lo
	v_cndmask_b32_e64 v3, s12, 0, vcc_lo
	v_cndmask_b32_e32 v5, s3, v2, vcc_lo
	v_cndmask_b32_e32 v6, s2, v6, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v19, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v4, 31, v3
	v_sub_co_u32 v3, s0, v8, v3
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v4, null, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_b32 v[0:1], v7, off offset:28
.LBB0_23:                               ; %.loopexit
	s_endpgm
.Lfunc_end0:
	.size	gemm_gate_up_hfq4g256_wmma_gfx12, .Lfunc_end0-gemm_gate_up_hfq4g256_wmma_gfx12
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_gate_up_hfq4g256_wmma_gfx12
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 73
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 25
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.num_vgpr, 73
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.num_agpr, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.numbered_sgpr, 20
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.num_named_barrier, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.private_seg_size, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.uses_vcc, 1
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.uses_flat_scratch, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.has_dyn_sized_stack, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.has_recursion, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3140
; TotalNumSgprs: 22
; NumVgprs: 73
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 73
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage ; -- Begin function gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage
	.globl	gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage
	.p2align	8
	.type	gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage,@function
gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage: ; @gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage
; %bb.0:
	s_load_b128 s[12:15], s[0:1], 0x28
	s_lshl_b32 s16, ttmp9, 4
	s_lshl_b32 s17, ttmp7, 4
	s_wait_kmcnt 0x0
	s_add_co_i32 s18, s13, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_cmp_ge_i32 s16, s18
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s17, s15
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB1_23
; %bb.1:
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x20
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v9, 15, v0
	v_lshrrev_b32_e32 v14, 5, v0
	s_cmp_lt_i32 s14, 0x200
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v13, s17, v9
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	v_dual_mov_b32 v5, v8 :: v_dual_mov_b32 v4, v8
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v2, v8
	v_mov_b32_e32 v1, v8
	v_cmp_le_i32_e32 vcc_lo, s15, v13
	s_cbranch_scc1 .LBB1_6
; %bb.2:                                ; %.lr.ph
	v_mul_u32_u24_e32 v1, 0xf10, v0
	v_cndmask_b32_e64 v3, v13, 0, vcc_lo
	v_mov_b16_e32 v4.h, 0
	s_add_co_i32 s20, s18, -1
	s_ashr_i32 s0, s14, 31
	v_mov_b16_e32 v4.l, v1.h
	v_mad_co_u64_u32 v[1:2], null, v3, s14, 0
	v_ashrrev_i32_e32 v3, 31, v3
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v5, s5
	s_lshr_b32 s0, s0, 23
	v_add_nc_u32_e32 v6, s16, v4
	s_add_co_i32 s0, s14, s0
	v_mov_b32_e32 v7, s4
	s_ashr_i32 s19, s0, 9
	v_mad_co_u64_u32 v[2:3], null, v3, s14, v[2:3]
	v_min_i32_e32 v3, s20, v6
	v_mul_i32_i24_e32 v6, 0xffffffef, v4
	v_mul_u32_u24_e32 v10, 0x110, v4
	v_mul_u32_u24_e32 v9, 0x110, v9
	v_bfe_u32 v8, v0, 4, 1
	v_cmp_gt_i32_e64 s0, s12, v3
	v_add_lshl_u32 v4, v6, v0, 4
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_or_b32_e32 v19, 0x2000, v10
	s_mov_b32 s15, 0
	v_cndmask_b32_e64 v12, s7, v5, s0
	v_cndmask_b32_e64 v11, s6, v7, s0
	v_cndmask_b32_e64 v5, s12, 0, s0
	v_add_co_u32 v6, s0, s8, v1
	v_lshrrev_b32_e32 v7, 7, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s9, v2, s0
	s_or_b32 s0, s16, 15
	v_sub_nc_u32_e32 v3, v3, v5
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s0, s0, s20
	v_mad_u32_u24 v7, 0x88, v7, v9
	v_lshlrev_b32_e32 v9, 7, v14
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s0, s12
	v_ashrrev_i32_e32 v5, 31, v4
	s_cselect_b32 s8, 0, s12
	s_cselect_b32 s5, s5, s7
	s_cselect_b32 s4, s4, s6
	s_ashr_i32 s17, s16, 31
	s_lshr_b32 s14, s14, 8
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[16:17], 15
	s_ashr_i32 s21, s20, 31
	v_or_b32_e32 v16, 0x2000, v7
	v_add_co_u32 v9, s1, v6, v9
	v_mad_co_i64_i32 v[6:7], null, s14, v3, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_lt_i64_e64 s9, s[6:7], s[20:21]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v2, s1
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v2, 0x60, v0
	v_cmp_gt_u32_e64 s0, 16, v0
	s_and_b32 s1, s9, exec_lo
	s_cselect_b32 s7, s7, s21
	v_add_nc_u32_e32 v20, v16, v2
	v_mad_co_u64_u32 v[2:3], null, 0x88, v6, v[4:5]
	s_cselect_b32 s6, s6, s20
	s_ashr_i32 s9, s8, 31
	v_lshlrev_b32_e32 v17, 4, v8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_nc_u64 s[6:7], s[6:7], s[8:9]
	v_lshlrev_b32_e32 v15, 4, v0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[6:7], s[14:15], s[6:7]
	v_lshlrev_b32_e32 v8, 2, v8
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[6:7], s[6:7], 0x88
	v_mad_co_u64_u32 v[5:6], null, 0x88, v7, v[3:4]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[4:5], s[6:7]
	v_mov_b32_e32 v6, v1
	v_add_co_u32 v17, s1, v9, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, 0, v10, s1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v9, s1, s4, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s5, 0, s1
	v_add_nc_u32_e32 v19, v19, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v9, s1, v9, 16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s1
	v_add_co_u32 v11, s1, v11, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v5, s1
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v5, v1 :: v_dual_add_nc_u32 v20, v20, v8
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v7, v1
	v_mov_b32_e32 v8, v1
	s_mov_b32 s14, s15
	s_branch .LBB1_4
.LBB1_3:                                ;   in Loop: Header=BB1_4 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_lshl_b64 s[4:5], s[14:15], 1
	s_add_co_i32 s19, s19, -1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v33, s1, v17, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v34, null, s5, v18, s1
	v_add_co_u32 v9, s1, 0x110, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s1
	v_add_co_u32 v11, s1, 0x110, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, 0, v12, s1
	s_addk_co_i32 s14, 0x200
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b64 v[37:38], v16
	ds_load_2addr_b32 v[41:42], v20 offset0:2 offset1:4
	ds_load_2addr_b32 v[43:44], v20 offset0:6 offset1:8
	s_clause 0x3
	global_load_b128 v[21:24], v[33:34], off
	global_load_b128 v[25:28], v[33:34], off offset:32
	global_load_b128 v[29:32], v[33:34], off offset:64
	global_load_b128 v[33:36], v[33:34], off offset:96
	s_cmp_eq_u32 s19, 0
	s_wait_dscnt 0x2
	v_cvt_f16_f32_e32 v53.l, v37
	v_cvt_f16_f32_e32 v40.h, v38
	s_wait_dscnt 0x1
	v_and_b32_e32 v37, 15, v41
	v_bfe_u32 v38, v41, 4, 4
	v_bfe_u32 v39, v41, 8, 4
	v_bfe_u32 v45, v41, 12, 4
	v_bfe_u32 v46, v41, 16, 4
	v_bfe_u32 v47, v41, 20, 4
	v_bfe_u32 v48, v41, 24, 4
	v_lshrrev_b32_e32 v41, 28, v41
	v_and_b32_e32 v49, 15, v42
	v_bfe_u32 v50, v42, 4, 4
	v_bfe_u32 v51, v42, 8, 4
	v_bfe_u32 v52, v42, 12, 4
	v_bfe_u32 v54, v42, 16, 4
	v_bfe_u32 v55, v42, 20, 4
	v_bfe_u32 v56, v42, 24, 4
	v_lshrrev_b32_e32 v42, 28, v42
	s_wait_dscnt 0x0
	v_and_b32_e32 v57, 15, v43
	v_bfe_u32 v58, v43, 4, 4
	v_bfe_u32 v59, v43, 8, 4
	v_bfe_u32 v60, v43, 12, 4
	v_bfe_u32 v61, v43, 16, 4
	v_bfe_u32 v62, v43, 20, 4
	v_bfe_u32 v63, v43, 24, 4
	v_lshrrev_b32_e32 v43, 28, v43
	v_and_b32_e32 v64, 15, v44
	v_bfe_u32 v65, v44, 4, 4
	v_bfe_u32 v66, v44, 8, 4
	v_bfe_u32 v67, v44, 12, 4
	v_bfe_u32 v68, v44, 16, 4
	v_bfe_u32 v69, v44, 20, 4
	v_bfe_u32 v70, v44, 24, 4
	v_lshrrev_b32_e32 v44, 28, v44
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_cvt_f32_ubyte0_e32 v71, v44
	v_cvt_f16_f32_e32 v37.l, v37
	v_cvt_f16_f32_e32 v37.h, v38
	v_cvt_f16_f32_e32 v38.l, v39
	v_cvt_f16_f32_e32 v38.h, v45
	v_cvt_f16_f32_e32 v39.l, v46
	v_cvt_f16_f32_e32 v39.h, v47
	v_cvt_f16_f32_e32 v40.l, v48
	v_cvt_f16_f32_e32 v44.h, v41
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_cvt_f16_f32_e32 v45.l, v49
	v_cvt_f16_f32_e32 v45.h, v50
	v_cvt_f16_f32_e32 v46.l, v51
	v_cvt_f16_f32_e32 v46.h, v52
	v_cvt_f16_f32_e32 v47.l, v54
	v_cvt_f16_f32_e32 v47.h, v55
	v_cvt_f16_f32_e32 v48.l, v56
	v_cvt_f16_f32_e32 v48.h, v42
	v_cvt_f16_f32_e32 v52.h, v43
	v_fma_f16 v41.l, v53.l, v37.l, v40.h
	v_fma_f16 v41.h, v53.l, v37.h, v40.h
	v_fma_f16 v42.l, v53.l, v38.l, v40.h
	v_fma_f16 v42.h, v53.l, v38.h, v40.h
	v_fma_f16 v43.l, v53.l, v39.l, v40.h
	v_fma_f16 v43.h, v53.l, v39.h, v40.h
	v_fma_f16 v44.l, v53.l, v40.l, v40.h
	v_fma_f16 v44.h, v53.l, v44.h, v40.h
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f16_f32_e32 v49.l, v57
	v_cvt_f16_f32_e32 v49.h, v58
	v_cvt_f16_f32_e32 v50.l, v59
	v_cvt_f16_f32_e32 v50.h, v60
	v_cvt_f16_f32_e32 v51.l, v61
	v_cvt_f16_f32_e32 v51.h, v62
	v_cvt_f16_f32_e32 v52.l, v63
	v_fma_f16 v45.l, v53.l, v45.l, v40.h
	v_fma_f16 v45.h, v53.l, v45.h, v40.h
	v_fma_f16 v46.l, v53.l, v46.l, v40.h
	v_fma_f16 v46.h, v53.l, v46.h, v40.h
	v_fma_f16 v47.l, v53.l, v47.l, v40.h
	v_fma_f16 v47.h, v53.l, v47.h, v40.h
	v_fma_f16 v48.l, v53.l, v48.l, v40.h
	v_fma_f16 v48.h, v53.l, v48.h, v40.h
	v_cvt_f16_f32_e32 v53.h, v64
	v_cvt_f16_f32_e32 v54.l, v65
	v_cvt_f16_f32_e32 v54.h, v66
	v_cvt_f16_f32_e32 v55.l, v67
	v_cvt_f16_f32_e32 v55.h, v68
	v_cvt_f16_f32_e32 v56.l, v69
	v_cvt_f16_f32_e32 v56.h, v70
	v_cvt_f16_f32_e32 v57.l, v71
	v_fma_f16 v49.l, v53.l, v49.l, v40.h
	v_fma_f16 v49.h, v53.l, v49.h, v40.h
	v_fma_f16 v50.l, v53.l, v50.l, v40.h
	v_fma_f16 v50.h, v53.l, v50.h, v40.h
	v_fma_f16 v51.l, v53.l, v51.l, v40.h
	v_fma_f16 v51.h, v53.l, v51.h, v40.h
	v_fma_f16 v52.l, v53.l, v52.l, v40.h
	v_fma_f16 v52.h, v53.l, v52.h, v40.h
	v_fma_f16 v37.l, v53.l, v53.h, v40.h
	v_fma_f16 v37.h, v53.l, v54.l, v40.h
	v_fma_f16 v38.l, v53.l, v54.h, v40.h
	v_fma_f16 v38.h, v53.l, v55.l, v40.h
	v_fma_f16 v39.l, v53.l, v55.h, v40.h
	v_fma_f16 v39.h, v53.l, v56.l, v40.h
	v_fma_f16 v40.l, v53.l, v56.h, v40.h
	v_fmac_f16_e32 v40.h, v53.l, v57.l
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[21:24], v[1:8]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[25:28], v[1:8]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[29:32], v[1:8]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[37:40], v[33:36], v[1:8]
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_6
.LBB1_4:                                ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[21:24], v[11:12], off
	s_wait_loadcnt 0x0
	ds_store_b128 v19, v[21:24]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_3
; %bb.5:                                ;   in Loop: Header=BB1_4 Depth=1
	global_load_b128 v[21:24], v[9:10], off
	s_wait_loadcnt 0x0
	ds_store_b128 v15, v[21:24] offset:12288
	s_branch .LBB1_3
.LBB1_6:                                ; %Flow442
	v_lshlrev_b32_e32 v9, 5, v0
	v_cmp_eq_u32_e64 s0, 0, v14
	s_xor_b32 s1, vcc_lo, -1
	ds_store_b128 v9, v[1:4]
	ds_store_b128 v9, v[5:8] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_23
; %bb.7:                                ; %.preheader
	v_lshrrev_b32_e32 v0, 1, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v0, 0x78, v0, s16
	v_cmpx_gt_i32_e64 s18, v0
	s_cbranch_execz .LBB1_9
; %bb.8:
	ds_load_2addr_stride64_b32 v[1:2], v9 offset1:4
	ds_load_2addr_stride64_b32 v[3:4], v9 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[5:6], v9 offset0:16 offset1:20
	v_cmp_gt_i32_e32 vcc_lo, s12, v0
	s_wait_dscnt 0x2
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v8, s11 :: v_dual_add_f32 v7, v1, v2
	v_mov_b32_e32 v10, s12
	ds_load_2addr_stride64_b32 v[1:2], v9 offset0:24 offset1:28
	s_wait_dscnt 0x2
	v_add_f32_e32 v3, v7, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v7, s13, v10 :: v_dual_add_f32 v10, v3, v4
	v_mad_co_i64_i32 v[3:4], null, v7, v13, 0
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_f32 v5, v10, v5 :: v_dual_mov_b32 v10, s10
	v_cndmask_b32_e32 v11, s3, v8, vcc_lo
	v_cndmask_b32_e64 v8, s12, 0, vcc_lo
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v12, v5, v6
	v_cndmask_b32_e32 v10, s2, v10, vcc_lo
	v_sub_nc_u32_e32 v7, v0, v8
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v12, v1
	v_add_co_u32 v3, vcc_lo, v10, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v8, 31, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v11, v4, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 2, v[7:8]
	v_add_f32_e32 v7, v1, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v4, v6, vcc_lo
	global_store_b32 v[1:2], v7, off
.LBB1_9:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 1, v0
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s18, v2
	s_cbranch_execz .LBB1_11
; %bb.10:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v11, s11 :: v_dual_add_nc_u32 v10, 4, v9
	v_cmp_gt_i32_e32 vcc_lo, s12, v2
	v_mov_b32_e32 v12, s12
	ds_load_2addr_stride64_b32 v[3:4], v10 offset1:4
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:16 offset1:20
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v3, v4
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:24 offset1:28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, s3, v11, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v4, v4, v5 :: v_dual_cndmask_b32 v5, s13, v12
	v_mov_b32_e32 v12, s10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v14, 31, v11
	v_add_f32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, v5, v13, 0
	s_wait_dscnt 0x1
	v_add_f32_e32 v15, v6, v7
	v_sub_co_u32 v6, s0, v0, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v7, null, v1, v14, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_f32_e32 v8, v15, v8
	v_cndmask_b32_e32 v11, s2, v12, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v8, v2
	v_add_co_u32 v4, vcc_lo, v11, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v10, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v2, v3
	v_add_co_u32 v2, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, v5, v7, vcc_lo
	global_store_b32 v[2:3], v8, off offset:4
.LBB1_11:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, 2, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s18, v2
	s_cbranch_execz .LBB1_13
; %bb.12:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v11, s11 :: v_dual_add_nc_u32 v10, 8, v9
	v_cmp_gt_i32_e32 vcc_lo, s12, v2
	v_mov_b32_e32 v12, s12
	ds_load_2addr_stride64_b32 v[3:4], v10 offset1:4
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:16 offset1:20
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v3, v4
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:24 offset1:28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, s3, v11, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v4, v4, v5 :: v_dual_cndmask_b32 v5, s13, v12
	v_mov_b32_e32 v12, s10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v14, 31, v11
	v_add_f32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, v5, v13, 0
	s_wait_dscnt 0x1
	v_add_f32_e32 v15, v6, v7
	v_sub_co_u32 v6, s0, v0, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v7, null, v1, v14, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_f32_e32 v8, v15, v8
	v_cndmask_b32_e32 v11, s2, v12, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v8, v2
	v_add_co_u32 v4, vcc_lo, v11, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v10, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v2, v3
	v_add_co_u32 v2, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, v5, v7, vcc_lo
	global_store_b32 v[2:3], v8, off offset:8
.LBB1_13:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, 3, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s18, v2
	s_cbranch_execz .LBB1_15
; %bb.14:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v11, s11 :: v_dual_add_nc_u32 v10, 12, v9
	v_cmp_gt_i32_e32 vcc_lo, s12, v2
	v_mov_b32_e32 v12, s12
	ds_load_2addr_stride64_b32 v[3:4], v10 offset1:4
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:16 offset1:20
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v3, v4
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:24 offset1:28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, s3, v11, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v4, v4, v5 :: v_dual_cndmask_b32 v5, s13, v12
	v_mov_b32_e32 v12, s10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v14, 31, v11
	v_add_f32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, v5, v13, 0
	s_wait_dscnt 0x1
	v_add_f32_e32 v15, v6, v7
	v_sub_co_u32 v6, s0, v0, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v7, null, v1, v14, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_f32_e32 v8, v15, v8
	v_cndmask_b32_e32 v11, s2, v12, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v8, v2
	v_add_co_u32 v4, vcc_lo, v11, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v10, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v2, v3
	v_add_co_u32 v2, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, v5, v7, vcc_lo
	global_store_b32 v[2:3], v8, off offset:12
.LBB1_15:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, 4, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s18, v2
	s_cbranch_execz .LBB1_17
; %bb.16:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v11, s11 :: v_dual_add_nc_u32 v10, 16, v9
	v_cmp_gt_i32_e32 vcc_lo, s12, v2
	v_mov_b32_e32 v12, s12
	ds_load_2addr_stride64_b32 v[3:4], v10 offset1:4
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:16 offset1:20
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v3, v4
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:24 offset1:28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, s3, v11, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v4, v4, v5 :: v_dual_cndmask_b32 v5, s13, v12
	v_mov_b32_e32 v12, s10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v14, 31, v11
	v_add_f32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, v5, v13, 0
	s_wait_dscnt 0x1
	v_add_f32_e32 v15, v6, v7
	v_sub_co_u32 v6, s0, v0, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v7, null, v1, v14, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_f32_e32 v8, v15, v8
	v_cndmask_b32_e32 v11, s2, v12, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v8, v2
	v_add_co_u32 v4, vcc_lo, v11, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v10, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v2, v3
	v_add_co_u32 v2, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, v5, v7, vcc_lo
	global_store_b32 v[2:3], v8, off offset:16
.LBB1_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, 5, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s18, v2
	s_cbranch_execz .LBB1_19
; %bb.18:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v11, s11 :: v_dual_add_nc_u32 v10, 20, v9
	v_cmp_gt_i32_e32 vcc_lo, s12, v2
	v_mov_b32_e32 v12, s12
	ds_load_2addr_stride64_b32 v[3:4], v10 offset1:4
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:16 offset1:20
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v3, v4
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:24 offset1:28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, s3, v11, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v4, v4, v5 :: v_dual_cndmask_b32 v5, s13, v12
	v_mov_b32_e32 v12, s10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v14, 31, v11
	v_add_f32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, v5, v13, 0
	s_wait_dscnt 0x1
	v_add_f32_e32 v15, v6, v7
	v_sub_co_u32 v6, s0, v0, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v7, null, v1, v14, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_f32_e32 v8, v15, v8
	v_cndmask_b32_e32 v11, s2, v12, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v8, v2
	v_add_co_u32 v4, vcc_lo, v11, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v10, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v2, v3
	v_add_co_u32 v2, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, v5, v7, vcc_lo
	global_store_b32 v[2:3], v8, off offset:20
.LBB1_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, 6, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s18, v2
	s_cbranch_execz .LBB1_21
; %bb.20:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v11, s11 :: v_dual_add_nc_u32 v10, 24, v9
	v_cmp_gt_i32_e32 vcc_lo, s12, v2
	v_mov_b32_e32 v12, s12
	ds_load_2addr_stride64_b32 v[3:4], v10 offset1:4
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:16 offset1:20
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v3, v4
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:24 offset1:28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, s3, v11, vcc_lo
	v_cndmask_b32_e64 v11, s12, 0, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v4, v4, v5 :: v_dual_cndmask_b32 v5, s13, v12
	v_mov_b32_e32 v12, s10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v14, 31, v11
	v_add_f32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, v5, v13, 0
	s_wait_dscnt 0x1
	v_add_f32_e32 v15, v6, v7
	v_sub_co_u32 v6, s0, v0, v11
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v7, null, v1, v14, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_f32_e32 v8, v15, v8
	v_cndmask_b32_e32 v11, s2, v12, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v8, v2
	v_add_co_u32 v4, vcc_lo, v11, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v10, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v2, v3
	v_add_co_u32 v2, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, v5, v7, vcc_lo
	global_store_b32 v[2:3], v8, off offset:24
.LBB1_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, 7, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_23
; %bb.22:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v10, s11 :: v_dual_add_nc_u32 v9, 28, v9
	v_mov_b32_e32 v11, s12
	v_cmp_gt_i32_e32 vcc_lo, s12, v2
	ds_load_2addr_stride64_b32 v[3:4], v9 offset1:4
	ds_load_2addr_stride64_b32 v[5:6], v9 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[7:8], v9 offset0:16 offset1:20
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v3, v4
	ds_load_2addr_stride64_b32 v[2:3], v9 offset0:24 offset1:28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, s3, v10, vcc_lo
	v_cndmask_b32_e64 v10, s12, 0, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v4, v4, v5 :: v_dual_cndmask_b32 v5, s13, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v12, 31, v10
	v_sub_co_u32 v0, s0, v0, v10
	v_add_f32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_i64_i32 v[4:5], null, v5, v13, 0
	v_mov_b32_e32 v11, s10
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v1, null, v1, v12, s0
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	v_cndmask_b32_e32 v7, s2, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_f32_e32 v6, v6, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v7, v4
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v9, v5, vcc_lo
	v_add_co_u32 v0, vcc_lo, v4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v2, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off offset:28
.LBB1_23:                               ; %.loopexit
	s_endpgm
.Lfunc_end1:
	.size	gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage, .Lfunc_end1-gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage
		.amdhsa_group_segment_fixed_size 12544
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 72
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 32
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.num_vgpr, 72
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.num_agpr, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.numbered_sgpr, 22
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.num_named_barrier, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.private_seg_size, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.uses_vcc, 1
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.uses_flat_scratch, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.has_dyn_sized_stack, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.has_recursion, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4032
; TotalNumSgprs: 24
; NumVgprs: 72
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 12544 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 72
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_32b65b0c6441dc63,@object ; @__hip_cuid_32b65b0c6441dc63
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_32b65b0c6441dc63
__hip_cuid_32b65b0c6441dc63:
	.byte	0                               ; 0x0
	.size	__hip_cuid_32b65b0c6441dc63, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_32b65b0c6441dc63
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           gemm_gate_up_hfq4g256_wmma_gfx12
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         gemm_gate_up_hfq4g256_wmma_gfx12.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     73
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 12544
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
