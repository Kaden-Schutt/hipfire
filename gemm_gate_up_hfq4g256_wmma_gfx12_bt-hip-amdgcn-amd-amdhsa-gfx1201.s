	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.text
	.protected	gemm_gate_up_hfq4g256_wmma_gfx12_bt4 ; -- Begin function gemm_gate_up_hfq4g256_wmma_gfx12_bt4
	.globl	gemm_gate_up_hfq4g256_wmma_gfx12_bt4
	.p2align	8
	.type	gemm_gate_up_hfq4g256_wmma_gfx12_bt4,@function
gemm_gate_up_hfq4g256_wmma_gfx12_bt4:   ; @gemm_gate_up_hfq4g256_wmma_gfx12_bt4
; %bb.0:
	s_load_b128 s[12:15], s[0:1], 0x28
	s_lshl_b32 s20, ttmp9, 4
	s_lshl_b32 s18, ttmp7, 6
	s_wait_kmcnt 0x0
	s_add_co_i32 s19, s13, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_cmp_ge_i32 s20, s19
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s15
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_29
; %bb.1:                                ; %.preheader186
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	v_lshrrev_b32_e32 v1, 4, v0
	s_cmp_gt_i32 s14, 0xff
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v62, 3, v1
	s_cbranch_scc1 .LBB0_4
; %bb.2:                                ; %.preheader186..preheader182_crit_edge
	v_lshlrev_b32_e32 v32, 3, v1
	v_and_b32_e32 v61, 15, v0
	s_cbranch_execz .LBB0_5
; %bb.3:
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v16, v24 :: v_dual_mov_b32 v17, v25
	v_dual_mov_b32 v27, v25 :: v_dual_mov_b32 v26, v24
	v_dual_mov_b32 v29, v25 :: v_dual_mov_b32 v28, v24
	v_dual_mov_b32 v31, v25 :: v_dual_mov_b32 v30, v24
	v_dual_mov_b32 v18, v24 :: v_dual_mov_b32 v19, v25
	v_dual_mov_b32 v20, v24 :: v_dual_mov_b32 v21, v25
	v_dual_mov_b32 v22, v24 :: v_dual_mov_b32 v23, v25
	v_dual_mov_b32 v8, v24 :: v_dual_mov_b32 v9, v25
	v_dual_mov_b32 v10, v24 :: v_dual_mov_b32 v11, v25
	v_dual_mov_b32 v12, v24 :: v_dual_mov_b32 v13, v25
	v_dual_mov_b32 v14, v24 :: v_dual_mov_b32 v15, v25
	v_dual_mov_b32 v0, v24 :: v_dual_mov_b32 v1, v25
	v_dual_mov_b32 v2, v24 :: v_dual_mov_b32 v3, v25
	v_dual_mov_b32 v4, v24 :: v_dual_mov_b32 v5, v25
	v_dual_mov_b32 v6, v24 :: v_dual_mov_b32 v7, v25
	s_branch .LBB0_10
.LBB0_4:
                                        ; implicit-def: $vgpr32
	v_and_b32_e32 v61, 15, v0
.LBB0_5:                                ; %.lr.ph
	s_ashr_i32 s0, s14, 31
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v0, s20, v61
	v_or_b32_e32 v6, s18, v61
	s_lshr_b32 s0, s0, 24
	s_add_co_i32 s2, s19, -1
	s_add_co_i32 s0, s14, s0
	v_min_i32_e32 v0, s2, v0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s5 :: v_dual_mov_b32 v7, s4
	s_ashr_i32 s5, s0, 8
	v_cmp_gt_i32_e64 s0, s15, v6
	v_cmp_gt_i32_e32 vcc_lo, s12, v0
	v_or_b32_e32 v5, 16, v6
	v_or_b32_e32 v8, 32, v6
	v_or_b32_e32 v13, 48, v6
	v_cndmask_b32_e64 v4, 0, v6, s0
	v_cndmask_b32_e32 v11, s7, v2, vcc_lo
	v_cmp_gt_i32_e64 s0, s15, v5
	v_cndmask_b32_e64 v14, s12, 0, vcc_lo
	v_cndmask_b32_e32 v10, s6, v7, vcc_lo
	v_mad_co_u64_u32 v[2:3], null, v4, s14, 0
	v_ashrrev_i32_e32 v4, 31, v4
	v_cmp_gt_i32_e32 vcc_lo, s15, v8
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v9, 0, v5, s0
	v_sub_nc_u32_e32 v0, v0, v14
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s21, s5, 0x88
	s_ashr_i32 s3, s2, 31
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v8, vcc_lo
	v_mad_co_u64_u32 v[3:4], null, v4, s14, v[3:4]
	v_mad_co_u64_u32 v[4:5], null, v9, s14, 0
	v_cmp_gt_i32_e32 vcc_lo, s15, v13
	v_ashrrev_i32_e32 v9, 31, v9
	v_mad_co_u64_u32 v[7:8], null, v12, s14, 0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s4, s21, 31
	s_mov_b32 s1, 0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v13, vcc_lo
	v_mad_co_i64_i32 v[33:34], null, s21, v0, v[10:11]
	v_ashrrev_i32_e32 v0, 31, v12
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_mad_co_u64_u32 v[5:6], null, v9, s14, v[5:6]
	v_add_nc_u32_e32 v12, s20, v61
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s1
	v_mad_co_u64_u32 v[8:9], null, v0, s14, v[8:9]
	v_ashrrev_i32_e32 v0, 31, v15
	v_add_co_u32 v9, vcc_lo, s8, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s9, v3, vcc_lo
	v_mad_co_u64_u32 v[2:3], null, v15, s14, 0
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshlrev_b64_e32 v[6:7], 1, v[7:8]
	v_lshlrev_b32_e32 v15, 4, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_i64_e32 vcc_lo, s[2:3], v[12:13]
	v_add_co_u32 v8, s0, s8, v4
	v_mad_co_u64_u32 v[3:4], null, v0, s14, v[3:4]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s9, v5, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, v12, s2, vcc_lo
	v_cndmask_b32_e64 v0, v13, s3, vcc_lo
	v_ashrrev_i32_e32 v12, 31, v14
	v_add_co_u32 v6, vcc_lo, s8, v6
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s9, v7, vcc_lo
	v_sub_co_u32 v4, vcc_lo, v4, v14
	s_wait_alu depctr_va_vcc(0)
	v_sub_co_ci_u32_e64 v12, null, v0, v12, vcc_lo
	v_add_co_u32 v13, vcc_lo, s8, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s9, v3, vcc_lo
	v_mad_co_u64_u32 v[2:3], null, v4, s21, 0
	v_mul_lo_u32 v12, v12, s21
	v_mul_lo_u32 v17, v4, s4
	v_mov_b32_e32 v0, 0
	v_add_co_u32 v35, vcc_lo, v9, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, 0, v16, vcc_lo
	v_lshl_or_b32 v1, v1, 2, v2
	v_add_co_u32 v37, vcc_lo, v8, v15
	v_add3_u32 v3, v3, v17, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v38, null, 0, v5, vcc_lo
	v_add_co_u32 v39, vcc_lo, v6, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, 0, v7, vcc_lo
	v_add_co_u32 v1, vcc_lo, v10, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v11, v3, vcc_lo
	v_add_co_u32 v41, vcc_lo, v13, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, 0, v14, vcc_lo
	v_add_co_u32 v63, vcc_lo, v1, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, 0, v2, vcc_lo
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v8, v0
	v_dual_mov_b32 v9, v0 :: v_dual_mov_b32 v10, v0
	v_dual_mov_b32 v11, v0 :: v_dual_mov_b32 v12, v0
	v_dual_mov_b32 v13, v0 :: v_dual_mov_b32 v14, v0
	v_dual_mov_b32 v15, v0 :: v_dual_mov_b32 v16, v0
	v_dual_mov_b32 v17, v0 :: v_dual_mov_b32 v18, v0
	v_dual_mov_b32 v19, v0 :: v_dual_mov_b32 v20, v0
	v_dual_mov_b32 v21, v0 :: v_dual_mov_b32 v22, v0
	v_dual_mov_b32 v23, v0 :: v_dual_mov_b32 v24, v0
	v_dual_mov_b32 v25, v0 :: v_dual_mov_b32 v26, v0
	v_dual_mov_b32 v27, v0 :: v_dual_mov_b32 v28, v0
	v_dual_mov_b32 v29, v0 :: v_dual_mov_b32 v30, v0
	v_mov_b32_e32 v31, v0
	s_mov_b32 s0, s1
	s_mov_b32 s4, s1
.LBB0_6:                                ; %.preheader184
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_7 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s2, s6, 0x88
	v_dual_mov_b32 v46, v40 :: v_dual_mov_b32 v45, v39
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v43, vcc_lo, v33, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, 0, v34, vcc_lo
	v_dual_mov_b32 v48, v42 :: v_dual_mov_b32 v47, v41
	v_dual_mov_b32 v50, v38 :: v_dual_mov_b32 v49, v37
	global_load_b64 v[51:52], v[43:44], off
	v_add_co_u32 v43, vcc_lo, v63, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, 0, v64, vcc_lo
	s_lshl_b64 s[2:3], s[0:1], 1
	s_mov_b32 s7, -4
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v32.l, v51
	v_cvt_f16_f32_e32 v32.h, v52
	v_dual_mov_b32 v52, v36 :: v_dual_mov_b32 v51, v35
.LBB0_7:                                ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_clause 0x3
	global_load_b32 v54, v[43:44], off offset:-24
	global_load_b32 v74, v[43:44], off offset:-16
	global_load_b32 v75, v[43:44], off offset:-8
	global_load_b32 v65, v[43:44], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v57, vcc_lo, v51, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, s3, v52, vcc_lo
	v_add_co_u32 v59, vcc_lo, v49, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v60, null, s3, v50, vcc_lo
	s_add_co_i32 s7, s7, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s7, 11
	s_wait_loadcnt 0x3
	v_and_b32_e32 v53, 15, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v53, v53
	v_cvt_f16_f32_e32 v53.l, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v66.l, v32.l, v53.l, v32.h
	v_bfe_u32 v53, v54, 4, 4
	v_cvt_f32_ubyte0_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v53.l, v53
	v_fma_f16 v66.h, v32.l, v53.l, v32.h
	v_bfe_u32 v53, v54, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v53, v53
	v_cvt_f16_f32_e32 v53.l, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v67.l, v32.l, v53.l, v32.h
	v_bfe_u32 v53, v54, 12, 4
	v_cvt_f32_ubyte0_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v53.l, v53
	v_fma_f16 v67.h, v32.l, v53.l, v32.h
	v_bfe_u32 v53, v54, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v53, v53
	v_cvt_f16_f32_e32 v53.l, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v68.l, v32.l, v53.l, v32.h
	v_bfe_u32 v53, v54, 20, 4
	v_cvt_f32_ubyte0_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v53.l, v53
	v_fma_f16 v68.h, v32.l, v53.l, v32.h
	v_bfe_u32 v53, v54, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v53, v53
	v_cvt_f16_f32_e32 v53.l, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v69.l, v32.l, v53.l, v32.h
	v_lshrrev_b32_e32 v53, 28, v54
	v_cvt_f32_ubyte0_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v53.l, v53
	v_fma_f16 v69.h, v32.l, v53.l, v32.h
	global_load_b128 v[53:56], v[57:58], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[66:69], v[53:56], v[0:7]
	global_load_b128 v[53:56], v[59:60], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[66:69], v[53:56], v[8:15]
	v_add_co_u32 v55, vcc_lo, v45, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v56, null, s3, v46, vcc_lo
	v_add_co_u32 v53, vcc_lo, v47, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v54, null, s3, v48, vcc_lo
	global_load_b128 v[70:73], v[55:56], off
	v_add_co_u32 v43, vcc_lo, v43, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, 0, v44, vcc_lo
	v_add_co_u32 v51, vcc_lo, 0x80, v51
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, 0, v52, vcc_lo
	v_add_co_u32 v49, vcc_lo, 0x80, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, 0, v50, vcc_lo
	v_add_co_u32 v47, vcc_lo, 0x80, v47
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v48, null, 0, v48, vcc_lo
	v_add_co_u32 v45, vcc_lo, 0x80, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, 0, v46, vcc_lo
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[66:69], v[70:73], v[16:23]
	global_load_b128 v[70:73], v[53:54], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[66:69], v[70:73], v[24:31]
	v_and_b32_e32 v66, 15, v74
	v_bfe_u32 v67, v74, 4, 4
	v_bfe_u32 v68, v74, 12, 4
	v_bfe_u32 v69, v74, 20, 4
	v_lshrrev_b32_e32 v70, 28, v74
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f16_f32_e32 v66.l, v66
	v_cvt_f16_f32_e32 v66.h, v67
	v_bfe_u32 v67, v74, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v66.l, v32.l, v66.l, v32.h
	v_fma_f16 v66.h, v32.l, v66.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f16_f32_e32 v67.l, v67
	v_cvt_f16_f32_e32 v67.h, v68
	v_bfe_u32 v68, v74, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v67.l, v32.l, v67.l, v32.h
	v_fma_f16 v67.h, v32.l, v67.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f16_f32_e32 v68.l, v68
	v_cvt_f16_f32_e32 v68.h, v69
	v_bfe_u32 v69, v74, 24, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v68.l, v32.l, v68.l, v32.h
	v_fma_f16 v68.h, v32.l, v68.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f16_f32_e32 v69.l, v69
	v_cvt_f16_f32_e32 v69.h, v70
	global_load_b128 v[70:73], v[57:58], off offset:32
	v_fma_f16 v69.l, v32.l, v69.l, v32.h
	v_fma_f16 v69.h, v32.l, v69.h, v32.h
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[0:7], v[66:69], v[70:73], v[0:7]
	global_load_b128 v[70:73], v[59:60], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[66:69], v[70:73], v[8:15]
	global_load_b128 v[70:73], v[55:56], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[66:69], v[70:73], v[16:23]
	global_load_b128 v[70:73], v[53:54], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[66:69], v[70:73], v[24:31]
	v_and_b32_e32 v66, 15, v75
	v_bfe_u32 v67, v75, 4, 4
	v_bfe_u32 v68, v75, 12, 4
	v_bfe_u32 v69, v75, 20, 4
	v_lshrrev_b32_e32 v70, 28, v75
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f16_f32_e32 v66.l, v66
	v_cvt_f16_f32_e32 v66.h, v67
	v_bfe_u32 v67, v75, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v66.l, v32.l, v66.l, v32.h
	v_fma_f16 v66.h, v32.l, v66.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f16_f32_e32 v67.l, v67
	v_cvt_f16_f32_e32 v67.h, v68
	v_bfe_u32 v68, v75, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v67.l, v32.l, v67.l, v32.h
	v_fma_f16 v67.h, v32.l, v67.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f16_f32_e32 v68.l, v68
	v_cvt_f16_f32_e32 v68.h, v69
	v_bfe_u32 v69, v75, 24, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v68.l, v32.l, v68.l, v32.h
	v_fma_f16 v68.h, v32.l, v68.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f16_f32_e32 v69.l, v69
	v_cvt_f16_f32_e32 v69.h, v70
	global_load_b128 v[70:73], v[57:58], off offset:64
	v_fma_f16 v69.l, v32.l, v69.l, v32.h
	v_fma_f16 v69.h, v32.l, v69.h, v32.h
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[0:7], v[66:69], v[70:73], v[0:7]
	global_load_b128 v[70:73], v[59:60], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[66:69], v[70:73], v[8:15]
	global_load_b128 v[70:73], v[55:56], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[66:69], v[70:73], v[16:23]
	global_load_b128 v[70:73], v[53:54], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[66:69], v[70:73], v[24:31]
	global_load_b128 v[70:73], v[57:58], off offset:96
	global_load_b128 v[57:60], v[59:60], off offset:96
	v_and_b32_e32 v66, 15, v65
	v_bfe_u32 v67, v65, 4, 4
	v_bfe_u32 v68, v65, 12, 4
	v_bfe_u32 v69, v65, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v66.l, v66
	v_cvt_f16_f32_e32 v66.h, v67
	v_bfe_u32 v67, v65, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v66.l, v32.l, v66.l, v32.h
	v_fma_f16 v66.h, v32.l, v66.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f16_f32_e32 v67.l, v67
	v_cvt_f16_f32_e32 v67.h, v68
	v_bfe_u32 v68, v65, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v67.l, v32.l, v67.l, v32.h
	v_fma_f16 v67.h, v32.l, v67.h, v32.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f16_f32_e32 v68.l, v68
	v_cvt_f16_f32_e32 v68.h, v69
	v_bfe_u32 v69, v65, 24, 4
	v_lshrrev_b32_e32 v65, 28, v65
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v68.l, v32.l, v68.l, v32.h
	v_fma_f16 v68.h, v32.l, v68.h, v32.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v65, v65
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v69.l, v69
	v_cvt_f16_f32_e32 v65.l, v65
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v69.l, v32.l, v69.l, v32.h
	v_fma_f16 v69.h, v32.l, v65.l, v32.h
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[0:7], v[66:69], v[70:73], v[0:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[66:69], v[57:60], v[8:15]
	global_load_b128 v[55:58], v[55:56], off offset:96
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[66:69], v[55:58], v[16:23]
	global_load_b128 v[53:56], v[53:54], off offset:96
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[66:69], v[53:56], v[24:31]
	s_cbranch_scc0 .LBB0_7
; %bb.8:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_co_i32 s6, s6, 1
	s_addk_co_i32 s4, 0x88
	s_addk_co_i32 s0, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, s5
	s_cbranch_scc0 .LBB0_6
; %bb.9:                                ; %.preheader182.loopexit
	v_mov_b32_e32 v32, v62
.LBB0_10:                               ; %Flow463
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v52, s20, v32
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v50, s11
	v_mov_b32_e32 v56, s10
	v_mov_b32_e32 v58, s12
	v_or_b32_e32 v34, 1, v52
	v_cmp_gt_i32_e64 s0, s12, v52
	v_ashrrev_i32_e32 v53, 31, v52
	v_or_b32_e32 v42, 2, v52
	v_or_b32_e32 v59, 7, v52
	v_cmp_gt_i32_e64 s1, s12, v34
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v33, s12, 0, s0
	v_cndmask_b32_e64 v41, s17, v50, s0
	v_cndmask_b32_e64 v43, s16, v56, s0
	v_cndmask_b32_e64 v32, s13, v58, s0
	v_cndmask_b32_e64 v35, s12, 0, s1
	v_sub_nc_u32_e32 v33, v52, v33
	v_cmp_gt_i32_e64 s0, s19, v34
	v_cmp_gt_i32_e64 s2, s12, v42
	v_cndmask_b32_e64 v46, s16, v56, s1
	v_ashrrev_i32_e32 v36, 31, v35
	v_ashrrev_i32_e32 v34, 31, v33
	v_sub_co_u32 v35, s3, v52, v35
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v47, s12, 0, s2
	v_sub_co_ci_u32_e64 v36, null, v53, v36, s3
	v_lshlrev_b64_e32 v[37:38], 2, v[33:34]
	v_cndmask_b32_e64 v44, s17, v50, s1
	v_cndmask_b32_e64 v33, s13, v58, s1
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[39:40], 2, v[35:36]
	v_cndmask_b32_e64 v45, s17, v50, s2
	v_cmp_gt_i32_e64 s7, s12, v59
	v_add_co_u32 v34, s1, v43, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, v41, v38, s1
	v_add_co_u32 v36, s1, v46, v39
	v_ashrrev_i32_e32 v39, 31, v47
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v37, null, v44, v40, s1
	v_sub_co_u32 v38, s1, v52, v47
	v_or_b32_e32 v41, 3, v52
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v39, null, v53, v39, s1
	v_or_b32_e32 v46, 4, v52
	v_cmp_gt_i32_e64 s1, s19, v42
	v_cmp_gt_i32_e64 s3, s12, v41
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[39:40], 2, v[38:39]
	v_cndmask_b32_e64 v42, s16, v56, s2
	v_cmp_gt_i32_e64 s4, s12, v46
	v_cndmask_b32_e64 v38, s13, v58, s2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v43, s12, 0, s3
	v_cndmask_b32_e64 v47, s17, v50, s3
	v_add_co_u32 v39, s2, v42, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v40, null, v45, v40, s2
	v_cndmask_b32_e64 v45, s12, 0, s4
	v_ashrrev_i32_e32 v44, 31, v43
	v_sub_co_u32 v42, s2, v52, v43
	v_cndmask_b32_e64 v49, s16, v56, s3
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v51, 31, v45
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v43, null, v53, v44, s2
	v_cmp_gt_i32_e64 s2, s19, v41
	v_cndmask_b32_e64 v41, s13, v58, s3
	v_sub_co_u32 v44, s3, v52, v45
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v45, null, v53, v51, s3
	v_or_b32_e32 v51, 5, v52
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_cndmask_b32_e64 v48, s17, v50, s4
	v_cndmask_b32_e64 v63, s12, 0, s7
	v_cndmask_b32_e64 v62, s17, v50, s7
	v_cmp_gt_i32_e64 s5, s12, v51
	v_cmp_gt_i32_e32 vcc_lo, s19, v52
	v_add_co_u32 v42, s3, v49, v42
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v43, null, v47, v43, s3
	v_cmp_gt_i32_e64 s3, s19, v46
	v_lshlrev_b64_e32 v[45:46], 2, v[44:45]
	v_cndmask_b32_e64 v47, s16, v56, s4
	v_cndmask_b32_e64 v49, s12, 0, s5
	v_cndmask_b32_e64 v44, s13, v58, s4
	v_cndmask_b32_e64 v55, s16, v56, s5
	v_cndmask_b32_e64 v54, s17, v50, s5
	v_add_co_u32 v45, s4, v47, v45
	v_ashrrev_i32_e32 v47, 31, v49
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v46, null, v48, v46, s4
	v_cmp_gt_i32_e64 s4, s19, v51
	v_or_b32_e32 v51, 6, v52
	v_sub_co_u32 v48, s6, v52, v49
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v49, null, v53, v47, s6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s6, s12, v51
	v_cndmask_b32_e64 v47, s13, v58, s5
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v57, s12, 0, s6
	v_cndmask_b32_e64 v60, s17, v50, s6
	v_add_co_u32 v48, s5, v55, v48
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v49, null, v54, v49, s5
	v_cmp_gt_i32_e64 s5, s19, v51
	v_ashrrev_i32_e32 v51, 31, v57
	v_ashrrev_i32_e32 v54, 31, v63
	v_sub_co_u32 v50, s8, v52, v57
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_sub_co_ci_u32_e64 v51, null, v53, v51, s8
	v_sub_co_u32 v52, s8, v52, v63
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v53, null, v53, v54, s8
	v_lshlrev_b64_e32 v[54:55], 2, v[50:51]
	v_cndmask_b32_e64 v51, s16, v56, s6
	v_cndmask_b32_e64 v63, s16, v56, s7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[56:57], 2, v[52:53]
	v_cndmask_b32_e64 v50, s13, v58, s6
	v_cndmask_b32_e64 v53, s13, v58, s7
	v_add_co_u32 v51, s6, v51, v54
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v60, v55, s6
	v_add_co_u32 v54, s7, v63, v56
	v_cmp_gt_i32_e64 s6, s19, v59
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v55, null, v62, v57, s7
	v_add_nc_u32_e32 v56, s18, v61
	s_mov_b32 s8, 0
	s_branch .LBB0_12
.LBB0_11:                               ; %Flow460
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_add_nc_u32_e32 v56, 16, v56
	s_add_co_i32 s8, s8, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s8, 4
	s_cbranch_scc0 .LBB0_29
.LBB0_12:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s9, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v56
	s_cbranch_execz .LBB0_11
; %bb.13:                               ; %.preheader
                                        ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 m0, s8, 3
	v_movrels_b32_e32 v57, v7
	v_movrels_b32_e32 v58, v6
	v_movrels_b32_e32 v59, v5
	v_movrels_b32_e32 v60, v4
	v_movrels_b32_e32 v61, v3
	v_movrels_b32_e32 v62, v2
	v_movrels_b32_e32 v63, v1
	v_movrels_b32_e32 v64, v0
	s_and_saveexec_b32 s10, vcc_lo
	s_cbranch_execnz .LBB0_21
; %bb.14:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s0
	s_cbranch_execnz .LBB0_22
.LBB0_15:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s1
	s_cbranch_execnz .LBB0_23
.LBB0_16:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s2
	s_cbranch_execnz .LBB0_24
.LBB0_17:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s3
	s_cbranch_execnz .LBB0_25
.LBB0_18:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s4
	s_cbranch_execnz .LBB0_26
.LBB0_19:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s5
	s_cbranch_execnz .LBB0_27
.LBB0_20:                               ;   in Loop: Header=BB0_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB0_11
	s_branch .LBB0_28
.LBB0_21:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[65:66], null, v32, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	v_add_co_u32 v65, s7, v34, v65
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v66, null, v35, v66, s7
	global_store_b32 v[65:66], v64, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s0
	s_cbranch_execz .LBB0_15
.LBB0_22:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[64:65], null, v33, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v36, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v37, v65, s7
	global_store_b32 v[64:65], v63, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s1
	s_cbranch_execz .LBB0_16
.LBB0_23:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[63:64], null, v38, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[63:64], 2, v[63:64]
	v_add_co_u32 v63, s7, v39, v63
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v64, null, v40, v64, s7
	global_store_b32 v[63:64], v62, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s2
	s_cbranch_execz .LBB0_17
.LBB0_24:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[62:63], null, v41, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s7, v42, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, v43, v63, s7
	global_store_b32 v[62:63], v61, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s3
	s_cbranch_execz .LBB0_18
.LBB0_25:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[61:62], null, v44, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	v_add_co_u32 v61, s7, v45, v61
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v62, null, v46, v62, s7
	global_store_b32 v[61:62], v60, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s4
	s_cbranch_execz .LBB0_19
.LBB0_26:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[60:61], null, v47, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	v_add_co_u32 v60, s7, v48, v60
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, v49, v61, s7
	global_store_b32 v[60:61], v59, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s5
	s_cbranch_execz .LBB0_20
.LBB0_27:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[59:60], null, v50, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[59:60], 2, v[59:60]
	v_add_co_u32 v59, s7, v51, v59
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v60, null, v52, v60, s7
	global_store_b32 v[59:60], v58, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB0_11
.LBB0_28:                               ;   in Loop: Header=BB0_12 Depth=1
	v_mad_co_i64_i32 v[58:59], null, v53, v56, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	v_add_co_u32 v58, s7, v54, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v55, v59, s7
	global_store_b32 v[58:59], v57, off offset:28
	s_branch .LBB0_11
.LBB0_29:                               ; %.loopexit183
	s_endpgm
.Lfunc_end0:
	.size	gemm_gate_up_hfq4g256_wmma_gfx12_bt4, .Lfunc_end0-gemm_gate_up_hfq4g256_wmma_gfx12_bt4
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_gate_up_hfq4g256_wmma_gfx12_bt4
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
		.amdhsa_next_free_vgpr 76
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
		.amdhsa_inst_pref_size 34
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
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.num_vgpr, 76
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.num_agpr, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.numbered_sgpr, 22
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.num_named_barrier, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.private_seg_size, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.uses_vcc, 1
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.uses_flat_scratch, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.has_dyn_sized_stack, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.has_recursion, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4280
; TotalNumSgprs: 24
; NumVgprs: 76
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 76
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
	.protected	gemm_gate_up_hfq4g256_wmma_gfx12_bt8 ; -- Begin function gemm_gate_up_hfq4g256_wmma_gfx12_bt8
	.globl	gemm_gate_up_hfq4g256_wmma_gfx12_bt8
	.p2align	8
	.type	gemm_gate_up_hfq4g256_wmma_gfx12_bt8,@function
gemm_gate_up_hfq4g256_wmma_gfx12_bt8:   ; @gemm_gate_up_hfq4g256_wmma_gfx12_bt8
; %bb.0:
	s_load_b128 s[20:23], s[0:1], 0x28
	s_lshl_b32 s27, ttmp9, 4
	s_lshl_b32 s2, ttmp7, 7
	s_wait_kmcnt 0x0
	s_add_co_i32 s26, s21, s20
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_cmp_ge_i32 s27, s26
	s_cselect_b32 s3, -1, 0
	s_cmp_ge_i32 s2, s23
	s_cselect_b32 s4, -1, 0
	s_or_b32 s3, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB1_146
; %bb.1:                                ; %.preheader177
	v_and_b32_e32 v67, 15, v0
	s_clause 0x1
	s_load_b256 s[12:19], s[0:1], 0x0
	s_load_b64 s[24:25], s[0:1], 0x20
	v_lshrrev_b32_e32 v64, 4, v0
	s_cmp_gt_i32 s22, 0xff
	v_or_b32_e32 v126, s2, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v127, 3, v64
	v_or_b32_e32 v125, 16, v126
	v_or_b32_e32 v124, 32, v126
	v_or_b32_e32 v123, 48, v126
	v_or_b32_e32 v122, 64, v126
	v_or_b32_e32 v121, 0x50, v126
	v_or_b32_e32 v120, 0x60, v126
	v_or_b32_e32 v119, 0x70, v126
	v_cmp_gt_i32_e64 s7, s23, v126
	v_cmp_gt_i32_e64 s6, s23, v125
	v_cmp_gt_i32_e64 s5, s23, v124
	v_cmp_gt_i32_e64 s4, s23, v123
	v_cmp_gt_i32_e64 s3, s23, v122
	v_cmp_gt_i32_e64 s2, s23, v121
	v_cmp_gt_i32_e64 s1, s23, v120
	v_cmp_gt_i32_e64 s0, s23, v119
	s_cbranch_scc1 .LBB1_3
; %bb.2:                                ; %.preheader177..preheader175_crit_edge
	v_lshlrev_b32_e32 v65, 3, v64
	s_mov_b32 s8, 0
	s_branch .LBB1_4
.LBB1_3:
	s_mov_b32 s8, -1
                                        ; implicit-def: $vgpr65
.LBB1_4:                                ; %Flow851
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v8, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_cbranch_vccnz .LBB1_10
; %bb.5:                                ; %.lr.ph
	v_or_b32_e32 v0, s27, v67
	s_add_co_i32 s10, s26, -1
	v_cndmask_b32_e64 v4, 0, v126, s7
	s_ashr_i32 s8, s22, 31
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v3, s13
	v_min_i32_e32 v2, s10, v0
	v_mov_b32_e32 v5, s12
	s_lshr_b32 s8, s8, 24
	v_mad_co_u64_u32 v[0:1], null, v4, s22, 0
	s_add_co_i32 s8, s22, s8
	v_cmp_gt_i32_e32 vcc_lo, s20, v2
	s_ashr_i32 s28, s8, 8
	v_cndmask_b32_e64 v7, 0, v124, s5
	s_mul_i32 s9, s28, 0x88
	v_cndmask_b32_e64 v12, 0, v122, s3
	v_cndmask_b32_e64 v14, s20, 0, vcc_lo
	v_cndmask_b32_e32 v11, s15, v3, vcc_lo
	v_cndmask_b32_e32 v10, s14, v5, vcc_lo
	v_ashrrev_i32_e32 v3, 31, v4
	v_cndmask_b32_e64 v4, 0, v125, s6
	v_sub_nc_u32_e32 v2, v2, v14
	v_ashrrev_i32_e32 v13, 31, v7
	v_cndmask_b32_e64 v16, 0, v121, s2
	v_ashrrev_i32_e32 v20, 31, v12
	v_cndmask_b32_e64 v9, 0, v120, s1
	v_mad_co_i64_i32 v[65:66], null, s9, v2, v[10:11]
	v_mad_co_u64_u32 v[1:2], null, v3, s22, v[1:2]
	v_mad_co_u64_u32 v[2:3], null, v4, s22, 0
	v_ashrrev_i32_e32 v4, 31, v4
	v_cndmask_b32_e64 v15, 0, v119, s0
	s_ashr_i32 s11, s10, 31
	s_ashr_i32 s12, s9, 31
	s_mov_b32 s23, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[5:6], 1, v[0:1]
	v_mad_co_u64_u32 v[3:4], null, v4, s22, v[3:4]
	v_mad_co_u64_u32 v[0:1], null, v7, s22, 0
	v_cndmask_b32_e64 v4, 0, v123, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s29, s23
	v_add_co_u32 v17, vcc_lo, s16, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_ashrrev_i32_e32 v19, 31, v4
	v_lshlrev_b64_e32 v[7:8], 1, v[2:3]
	v_mad_co_u64_u32 v[1:2], null, v13, s22, v[1:2]
	v_mad_co_u64_u32 v[2:3], null, v4, s22, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v6, vcc_lo
	v_mad_co_u64_u32 v[5:6], null, v12, s22, 0
	v_add_co_u32 v21, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, s17, v8, vcc_lo
	v_lshlrev_b64_e32 v[12:13], 1, v[0:1]
	v_mad_co_u64_u32 v[3:4], null, v19, s22, v[3:4]
	v_mad_co_u64_u32 v[0:1], null, v16, s22, 0
	v_ashrrev_i32_e32 v4, 31, v16
	v_mad_co_u64_u32 v[6:7], null, v20, s22, v[6:7]
	v_mad_co_u64_u32 v[7:8], null, v9, s22, 0
	v_add_co_u32 v16, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v13, vcc_lo
	v_lshlrev_b64_e32 v[12:13], 1, v[2:3]
	v_mad_co_u64_u32 v[1:2], null, v4, s22, v[1:2]
	v_mad_co_u64_u32 v[2:3], null, v15, s22, 0
	v_ashrrev_i32_e32 v9, 31, v9
	v_ashrrev_i32_e32 v4, 31, v15
	v_lshlrev_b64_e32 v[5:6], 1, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[8:9], null, v9, s22, v[8:9]
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	v_mad_co_u64_u32 v[3:4], null, v4, s22, v[3:4]
	v_add_co_u32 v9, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v13, vcc_lo
	v_add_co_u32 v13, vcc_lo, s16, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s17, v6, vcc_lo
	v_lshlrev_b64_e32 v[4:5], 1, v[7:8]
	v_add_co_u32 v7, vcc_lo, s16, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v1, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 1, v[2:3]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_co_u32 v2, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s17, v5, vcc_lo
	v_lshlrev_b32_e32 v4, 4, v64
	v_add_co_u32 v5, vcc_lo, s16, v0
	v_add_nc_u32_e32 v0, s27, v67
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s17, v1, vcc_lo
	v_add_co_u32 v67, vcc_lo, v17, v4
	s_delay_alu instid0(VALU_DEP_3)
	v_ashrrev_i32_e32 v1, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, 0, v18, vcc_lo
	v_add_co_u32 v69, vcc_lo, v21, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, 0, v22, vcc_lo
	v_cmp_lt_i64_e32 vcc_lo, s[10:11], v[0:1]
	v_add_co_u32 v71, s8, v16, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v72, null, 0, v19, s8
	v_add_co_u32 v75, s8, v13, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v0, v0, s10, vcc_lo
	v_cndmask_b32_e64 v1, v1, s11, vcc_lo
	v_add_co_u32 v73, vcc_lo, v9, v4
	v_ashrrev_i32_e32 v9, 31, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, 0, v12, vcc_lo
	v_sub_co_u32 v12, vcc_lo, v0, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v76, null, 0, v6, s8
	s_wait_alu depctr_va_vcc(0)
	v_sub_co_ci_u32_e64 v6, null, v1, v9, vcc_lo
	v_mad_co_u64_u32 v[0:1], null, v12, s9, 0
	v_add_co_u32 v77, vcc_lo, v7, v4
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u32 v6, v6, s9
	v_mul_lo_u32 v7, v12, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, 0, v8, vcc_lo
	v_add_co_u32 v79, vcc_lo, v2, v4
	v_lshl_or_b32 v2, v64, 2, v0
	v_mov_b32_e32 v0, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v80, null, 0, v3, vcc_lo
	v_add3_u32 v1, v1, v7, v6
	v_add_co_u32 v81, vcc_lo, v5, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, 0, v15, vcc_lo
	v_add_co_u32 v8, vcc_lo, v10, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v11, v1, vcc_lo
	v_dual_mov_b32 v6, v0 :: v_dual_mov_b32 v7, v0
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_mov_b32_e32 v5, v0
	v_add_co_u32 v83, vcc_lo, v8, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, 0, v9, vcc_lo
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v8, v0
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v16, v0
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v24, v0
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v32, v0
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v40, v0
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v48, v0
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v56, v0
	s_mov_b32 s22, s23
.LBB1_6:                                ; %.preheader176
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_7 Depth 2
	s_mul_i32 s8, s29, 0x88
	v_dual_mov_b32 v88, v78 :: v_dual_mov_b32 v87, v77
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v85, vcc_lo, v65, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v86, null, 0, v66, vcc_lo
	v_dual_mov_b32 v90, v74 :: v_dual_mov_b32 v89, v73
	v_dual_mov_b32 v92, v80 :: v_dual_mov_b32 v91, v79
	global_load_b64 v[101:102], v[85:86], off
	v_dual_mov_b32 v86, v76 :: v_dual_mov_b32 v85, v75
	v_dual_mov_b32 v94, v72 :: v_dual_mov_b32 v93, v71
	v_dual_mov_b32 v96, v82 :: v_dual_mov_b32 v95, v81
	v_dual_mov_b32 v98, v70 :: v_dual_mov_b32 v97, v69
	v_dual_mov_b32 v100, v68 :: v_dual_mov_b32 v99, v67
	s_lshl_b64 s[16:17], s[22:23], 1
	s_mov_b32 s30, -4
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v64.l, v101
	v_cvt_f16_f32_e32 v64.h, v102
	v_dual_mov_b32 v102, v84 :: v_dual_mov_b32 v101, v83
.LBB1_7:                                ;   Parent Loop BB1_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_clause 0x1
	global_load_b32 v130, v[101:102], off offset:-24
	global_load_b32 v136, v[101:102], off offset:-16
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v117, vcc_lo, v99, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v118, null, s17, v100, vcc_lo
	v_add_co_u32 v115, s8, v97, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v116, null, s17, v98, s8
	v_add_co_u32 v113, s9, v93, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v114, null, s17, v94, s9
	v_add_co_u32 v111, s10, v89, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v112, null, s17, v90, s10
	v_add_co_u32 v109, s11, v85, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v110, null, s17, v86, s11
	v_add_co_u32 v107, s12, v87, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v108, null, s17, v88, s12
	v_add_co_u32 v105, s13, v91, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v106, null, s17, v92, s13
	v_add_co_u32 v103, s14, v95, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v104, null, s17, v96, s14
	global_load_b32 v137, v[101:102], off offset:-8
	v_add_co_u32 v99, s8, 0x80, v99
	v_add_co_u32 v97, s9, 0x80, v97
	v_add_co_u32 v95, s10, 0x80, v95
	v_add_co_u32 v93, s11, 0x80, v93
	v_add_co_u32 v91, s12, 0x80, v91
	v_add_co_u32 v89, s13, 0x80, v89
	v_add_co_u32 v87, s14, 0x80, v87
	v_add_co_u32 v85, s15, 0x80, v85
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v100, null, 0, v100, s8
	v_add_co_ci_u32_e64 v98, null, 0, v98, s9
	v_add_co_ci_u32_e64 v96, null, 0, v96, s10
	v_add_co_ci_u32_e64 v94, null, 0, v94, s11
	v_add_co_ci_u32_e64 v92, null, 0, v92, s12
	v_add_co_ci_u32_e64 v90, null, 0, v90, s13
	v_add_co_ci_u32_e64 v88, null, 0, v88, s14
	v_add_co_ci_u32_e64 v86, null, 0, v86, s15
	s_add_co_i32 s30, s30, 4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_u32 s30, 11
	s_wait_loadcnt 0x2
	v_and_b32_e32 v128, 15, v130
	v_bfe_u32 v129, v130, 4, 4
	v_bfe_u32 v131, v130, 8, 4
	v_bfe_u32 v132, v130, 12, 4
	v_bfe_u32 v133, v130, 16, 4
	v_cvt_f32_ubyte0_e32 v128, v128
	v_cvt_f32_ubyte0_e32 v129, v129
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v132, v132
	v_cvt_f32_ubyte0_e32 v133, v133
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e64 v128.l, v128
	v_cvt_f16_f32_e64 v128.h, v129
	v_cvt_f32_ubyte0_e32 v129, v131
	v_bfe_u32 v131, v130, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v128.l, v64.l, v128.l, v64.h
	v_fma_f16 v128.h, v64.l, v128.h, v64.h
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f16_f32_e64 v129.l, v129
	v_cvt_f16_f32_e64 v129.h, v132
	v_bfe_u32 v132, v130, 24, 4
	v_lshrrev_b32_e32 v130, 28, v130
	v_cvt_f32_ubyte0_e32 v131, v131
	v_fma_f16 v129.l, v64.l, v129.l, v64.h
	v_fma_f16 v129.h, v64.l, v129.h, v64.h
	v_cvt_f32_ubyte0_e32 v132, v132
	v_cvt_f32_ubyte0_e32 v134, v130
	v_cvt_f16_f32_e64 v130.l, v133
	v_cvt_f16_f32_e64 v130.h, v131
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e64 v131.l, v132
	v_cvt_f16_f32_e64 v131.h, v134
	global_load_b128 v[132:135], v[117:118], off
	v_fma_f16 v130.l, v64.l, v130.l, v64.h
	v_fma_f16 v130.h, v64.l, v130.h, v64.h
	v_fma_f16 v131.l, v64.l, v131.l, v64.h
	v_fma_f16 v131.h, v64.l, v131.h, v64.h
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[56:63], v[128:131], v[132:135], v[56:63]
	global_load_b128 v[132:135], v[115:116], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[48:55], v[128:131], v[132:135], v[48:55]
	global_load_b128 v[132:135], v[113:114], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[40:47], v[128:131], v[132:135], v[40:47]
	global_load_b128 v[132:135], v[111:112], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[32:39], v[128:131], v[132:135], v[32:39]
	global_load_b128 v[132:135], v[109:110], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[128:131], v[132:135], v[24:31]
	global_load_b128 v[132:135], v[107:108], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[128:131], v[132:135], v[16:23]
	global_load_b128 v[132:135], v[105:106], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[128:131], v[132:135], v[8:15]
	global_load_b128 v[132:135], v[103:104], off
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[128:131], v[132:135], v[0:7]
	v_and_b32_e32 v128, 15, v136
	v_bfe_u32 v129, v136, 4, 4
	v_bfe_u32 v130, v136, 8, 4
	v_bfe_u32 v131, v136, 12, 4
	v_bfe_u32 v132, v136, 16, 4
	v_cvt_f32_ubyte0_e32 v128, v128
	v_cvt_f32_ubyte0_e32 v129, v129
	v_lshrrev_b32_e32 v133, 28, v136
	v_cvt_f32_ubyte0_e32 v131, v131
	v_cvt_f32_ubyte0_e32 v132, v132
	v_cvt_f16_f32_e64 v128.l, v128
	v_cvt_f16_f32_e64 v128.h, v129
	v_cvt_f32_ubyte0_e32 v129, v130
	v_bfe_u32 v130, v136, 20, 4
	v_cvt_f32_ubyte0_e32 v133, v133
	v_fma_f16 v128.l, v64.l, v128.l, v64.h
	v_fma_f16 v128.h, v64.l, v128.h, v64.h
	v_cvt_f16_f32_e64 v129.l, v129
	v_cvt_f16_f32_e64 v129.h, v131
	v_bfe_u32 v131, v136, 24, 4
	v_cvt_f32_ubyte0_e32 v134, v130
	v_cvt_f16_f32_e64 v130.l, v132
	v_fma_f16 v129.l, v64.l, v129.l, v64.h
	v_fma_f16 v129.h, v64.l, v129.h, v64.h
	v_cvt_f32_ubyte0_e32 v131, v131
	v_cvt_f16_f32_e64 v130.h, v134
	v_fma_f16 v130.l, v64.l, v130.l, v64.h
	global_load_b32 v136, v[101:102], off
	v_add_co_u32 v101, vcc_lo, v101, 32
	v_cvt_f16_f32_e64 v131.l, v131
	v_cvt_f16_f32_e64 v131.h, v133
	global_load_b128 v[132:135], v[117:118], off offset:32
	v_fma_f16 v130.h, v64.l, v130.h, v64.h
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, 0, v102, vcc_lo
	v_fma_f16 v131.l, v64.l, v131.l, v64.h
	v_fma_f16 v131.h, v64.l, v131.h, v64.h
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[56:63], v[128:131], v[132:135], v[56:63]
	global_load_b128 v[132:135], v[115:116], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[48:55], v[128:131], v[132:135], v[48:55]
	global_load_b128 v[132:135], v[113:114], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[40:47], v[128:131], v[132:135], v[40:47]
	global_load_b128 v[132:135], v[111:112], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[32:39], v[128:131], v[132:135], v[32:39]
	global_load_b128 v[132:135], v[109:110], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[128:131], v[132:135], v[24:31]
	global_load_b128 v[132:135], v[107:108], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[128:131], v[132:135], v[16:23]
	global_load_b128 v[132:135], v[105:106], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[128:131], v[132:135], v[8:15]
	global_load_b128 v[132:135], v[103:104], off offset:32
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[128:131], v[132:135], v[0:7]
	v_and_b32_e32 v128, 15, v137
	v_bfe_u32 v129, v137, 4, 4
	v_bfe_u32 v130, v137, 8, 4
	v_bfe_u32 v131, v137, 12, 4
	v_bfe_u32 v132, v137, 16, 4
	v_cvt_f32_ubyte0_e32 v128, v128
	v_cvt_f32_ubyte0_e32 v129, v129
	v_lshrrev_b32_e32 v133, 28, v137
	v_cvt_f32_ubyte0_e32 v131, v131
	v_cvt_f32_ubyte0_e32 v132, v132
	v_cvt_f16_f32_e64 v128.l, v128
	v_cvt_f16_f32_e64 v128.h, v129
	v_cvt_f32_ubyte0_e32 v129, v130
	v_bfe_u32 v130, v137, 20, 4
	v_cvt_f32_ubyte0_e32 v133, v133
	v_fma_f16 v128.l, v64.l, v128.l, v64.h
	v_fma_f16 v128.h, v64.l, v128.h, v64.h
	v_cvt_f16_f32_e64 v129.l, v129
	v_cvt_f16_f32_e64 v129.h, v131
	v_bfe_u32 v131, v137, 24, 4
	v_cvt_f32_ubyte0_e32 v134, v130
	v_cvt_f16_f32_e64 v130.l, v132
	v_fma_f16 v129.l, v64.l, v129.l, v64.h
	v_fma_f16 v129.h, v64.l, v129.h, v64.h
	v_cvt_f32_ubyte0_e32 v131, v131
	v_cvt_f16_f32_e64 v130.h, v134
	v_fma_f16 v130.l, v64.l, v130.l, v64.h
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e64 v131.l, v131
	v_cvt_f16_f32_e64 v131.h, v133
	global_load_b128 v[132:135], v[117:118], off offset:64
	v_fma_f16 v130.h, v64.l, v130.h, v64.h
	v_fma_f16 v131.l, v64.l, v131.l, v64.h
	v_fma_f16 v131.h, v64.l, v131.h, v64.h
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[56:63], v[128:131], v[132:135], v[56:63]
	global_load_b128 v[132:135], v[115:116], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[48:55], v[128:131], v[132:135], v[48:55]
	global_load_b128 v[132:135], v[113:114], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[40:47], v[128:131], v[132:135], v[40:47]
	global_load_b128 v[132:135], v[111:112], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[32:39], v[128:131], v[132:135], v[32:39]
	global_load_b128 v[132:135], v[109:110], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[24:31], v[128:131], v[132:135], v[24:31]
	global_load_b128 v[132:135], v[107:108], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[16:23], v[128:131], v[132:135], v[16:23]
	global_load_b128 v[132:135], v[105:106], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[8:15], v[128:131], v[132:135], v[8:15]
	global_load_b128 v[132:135], v[103:104], off offset:64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[128:131], v[132:135], v[0:7]
	global_load_b128 v[128:131], v[117:118], off offset:96
	v_and_b32_e32 v117, 15, v136
	v_bfe_u32 v118, v136, 4, 4
	v_bfe_u32 v133, v136, 8, 4
	v_bfe_u32 v134, v136, 16, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v117.l, v117
	v_cvt_f16_f32_e32 v117.h, v118
	v_bfe_u32 v118, v136, 12, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v132.l, v64.l, v117.l, v64.h
	v_fma_f16 v132.h, v64.l, v117.h, v64.h
	v_cvt_f32_ubyte0_e32 v117, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v117.l, v117
	v_fma_f16 v133.l, v64.l, v117.l, v64.h
	v_cvt_f32_ubyte0_e32 v117, v118
	v_bfe_u32 v118, v136, 20, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v117.l, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v133.h, v64.l, v117.l, v64.h
	v_cvt_f32_ubyte0_e32 v117, v134
	v_cvt_f16_f32_e32 v117.l, v117
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v134.l, v64.l, v117.l, v64.h
	v_bfe_u32 v117, v136, 24, 4
	v_lshrrev_b32_e32 v136, 28, v136
	v_cvt_f32_ubyte0_e32 v135, v117
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v136, v136
	v_cvt_f16_f32_e32 v117.l, v118
	v_cvt_f16_f32_e64 v117.h, v135
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e64 v135.h, v136
	v_fma_f16 v134.h, v64.l, v117.l, v64.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v135.l, v64.l, v117.h, v64.h
	v_fma_f16 v135.h, v64.l, v135.h, v64.h
	global_load_b128 v[115:118], v[115:116], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[56:63], v[132:135], v[128:131], v[56:63]
	global_load_b128 v[128:131], v[113:114], off offset:96
	global_load_b128 v[111:114], v[111:112], off offset:96
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_f16 v[48:55], v[132:135], v[115:118], v[48:55]
	global_load_b128 v[115:118], v[109:110], off offset:96
	global_load_b128 v[107:110], v[107:108], off offset:96
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_f16 v[40:47], v[132:135], v[128:131], v[40:47]
	global_load_b128 v[128:131], v[105:106], off offset:96
	global_load_b128 v[103:106], v[103:104], off offset:96
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x16_f16 v[32:39], v[132:135], v[111:114], v[32:39]
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_f16 v[24:31], v[132:135], v[115:118], v[24:31]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_f16 v[16:23], v[132:135], v[107:110], v[16:23]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[8:15], v[132:135], v[128:131], v[8:15]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[132:135], v[103:106], v[0:7]
	s_cbranch_scc0 .LBB1_7
; %bb.8:                                ;   in Loop: Header=BB1_6 Depth=1
	v_add_co_u32 v83, vcc_lo, 0x88, v83
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, 0, v84, vcc_lo
	s_add_co_i32 s29, s29, 1
	s_addk_co_i32 s22, 0x100
	s_cmp_eq_u32 s29, s28
	s_cbranch_scc0 .LBB1_6
; %bb.9:                                ; %.preheader175.loopexit
	v_mov_b32_e32 v65, v127
.LBB1_10:                               ; %Flow852
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v64, s27, v65
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_27
; %bb.11:                               ; %.preheader
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_13
; %bb.12:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v65, s19 :: v_dual_mov_b32 v66, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v70, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v69, s25, v65, vcc_lo
	v_cndmask_b32_e32 v65, s21, v66, vcc_lo
	v_cndmask_b32_e64 v67, s20, 0, vcc_lo
	v_cndmask_b32_e32 v70, s24, v70, vcc_lo
	v_mad_co_i64_i32 v[65:66], null, v65, v126, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v67, v64, v67
	v_ashrrev_i32_e32 v68, 31, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	v_lshlrev_b64_e32 v[67:68], 2, v[67:68]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v65, vcc_lo, v70, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v69, v66, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v65, vcc_lo, v65, v67
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v66, v68, vcc_lo
	global_store_b32 v[65:66], v56, off
.LBB1_13:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v56, 1, v64
	s_mov_b32 s9, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v56
	s_cbranch_execz .LBB1_15
; %bb.14:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v65, s19 :: v_dual_mov_b32 v66, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v56
	v_ashrrev_i32_e32 v68, 31, v64
	v_mov_b32_e32 v70, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, s25, v65, vcc_lo
	v_cndmask_b32_e32 v65, s21, v66, vcc_lo
	v_cndmask_b32_e64 v67, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[65:66], null, v65, v126, 0
	v_ashrrev_i32_e32 v69, 31, v67
	v_sub_co_u32 v67, s7, v64, v67
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v68, null, v68, v69, s7
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	v_cndmask_b32_e32 v69, s24, v70, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[67:68], 2, v[67:68]
	v_add_co_u32 v65, vcc_lo, v69, v65
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v56, null, v56, v66, vcc_lo
	v_add_co_u32 v65, vcc_lo, v65, v67
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v66, null, v56, v68, vcc_lo
	global_store_b32 v[65:66], v57, off offset:4
.LBB1_15:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v56, 2, v64
	s_mov_b32 s9, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v56
	s_cbranch_execz .LBB1_17
; %bb.16:
	v_mov_b32_e32 v65, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v56
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v57, s19
	v_ashrrev_i32_e32 v66, 31, v64
	v_mov_b32_e32 v69, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, s21, v65, vcc_lo
	v_cndmask_b32_e64 v65, s20, 0, vcc_lo
	v_cndmask_b32_e32 v67, s25, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[56:57], null, v56, v126, 0
	v_ashrrev_i32_e32 v68, 31, v65
	v_sub_co_u32 v65, s7, v64, v65
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v66, null, v66, v68, s7
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_cndmask_b32_e32 v68, s24, v69, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	v_add_co_u32 v56, vcc_lo, v68, v56
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v67, v57, vcc_lo
	v_add_co_u32 v56, vcc_lo, v56, v65
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v57, v66, vcc_lo
	global_store_b32 v[56:57], v58, off offset:8
.LBB1_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v56, 3, v64
	s_mov_b32 s9, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v56
	s_cbranch_execz .LBB1_19
; %bb.18:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v57, s19 :: v_dual_mov_b32 v58, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v56
	v_ashrrev_i32_e32 v66, 31, v64
	v_mov_b32_e32 v69, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, s21, v58, vcc_lo
	v_cndmask_b32_e64 v58, s20, 0, vcc_lo
	v_cndmask_b32_e32 v67, s25, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[56:57], null, v56, v126, 0
	v_ashrrev_i32_e32 v68, 31, v58
	v_sub_co_u32 v65, s7, v64, v58
	v_cndmask_b32_e32 v58, s24, v69, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_ci_u32_e64 v66, null, v66, v68, s7
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v56, vcc_lo, v58, v56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, v67, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v56, vcc_lo, v56, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, v57, v66, vcc_lo
	global_store_b32 v[56:57], v59, off offset:12
.LBB1_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v56, 4, v64
	s_mov_b32 s9, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v56
	s_cbranch_execz .LBB1_21
; %bb.20:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v57, s19 :: v_dual_mov_b32 v58, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v56
	v_ashrrev_i32_e32 v59, 31, v64
	v_mov_b32_e32 v67, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, s21, v58, vcc_lo
	v_cndmask_b32_e64 v58, s20, 0, vcc_lo
	v_cndmask_b32_e32 v65, s25, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[56:57], null, v56, v126, 0
	v_ashrrev_i32_e32 v66, 31, v58
	v_sub_co_u32 v58, s7, v64, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v59, null, v59, v66, s7
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_cndmask_b32_e32 v66, s24, v67, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	v_add_co_u32 v56, vcc_lo, v66, v56
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v65, v57, vcc_lo
	v_add_co_u32 v56, vcc_lo, v56, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v57, v59, vcc_lo
	global_store_b32 v[56:57], v60, off offset:16
.LBB1_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v56, 5, v64
	s_mov_b32 s9, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v56
	s_cbranch_execz .LBB1_23
; %bb.22:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v57, s19 :: v_dual_mov_b32 v58, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v56
	v_ashrrev_i32_e32 v59, 31, v64
	v_mov_b32_e32 v66, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, s21, v58, vcc_lo
	v_cndmask_b32_e64 v58, s20, 0, vcc_lo
	v_cndmask_b32_e32 v60, s25, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[56:57], null, v56, v126, 0
	v_ashrrev_i32_e32 v65, 31, v58
	v_sub_co_u32 v58, s7, v64, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v59, null, v59, v65, s7
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_cndmask_b32_e32 v65, s24, v66, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	v_add_co_u32 v56, vcc_lo, v65, v56
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v60, v57, vcc_lo
	v_add_co_u32 v56, vcc_lo, v56, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v57, v59, vcc_lo
	global_store_b32 v[56:57], v61, off offset:20
.LBB1_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v56, 6, v64
	s_mov_b32 s9, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v56
	s_cbranch_execz .LBB1_25
; %bb.24:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v57, s19 :: v_dual_mov_b32 v58, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v56
	v_ashrrev_i32_e32 v59, 31, v64
	v_mov_b32_e32 v65, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, s21, v58, vcc_lo
	v_cndmask_b32_e64 v58, s20, 0, vcc_lo
	v_cndmask_b32_e32 v60, s25, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[56:57], null, v56, v126, 0
	v_ashrrev_i32_e32 v61, 31, v58
	v_sub_co_u32 v58, s7, v64, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v59, null, v59, v61, s7
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_cndmask_b32_e32 v61, s24, v65, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	v_add_co_u32 v56, vcc_lo, v61, v56
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v60, v57, vcc_lo
	v_add_co_u32 v56, vcc_lo, v56, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v57, v59, vcc_lo
	global_store_b32 v[56:57], v62, off offset:24
.LBB1_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v56, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v56
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_27
; %bb.26:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v57, s19 :: v_dual_mov_b32 v58, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v56
	v_ashrrev_i32_e32 v59, 31, v64
	v_mov_b32_e32 v62, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, s21, v58, vcc_lo
	v_cndmask_b32_e64 v58, s20, 0, vcc_lo
	v_cndmask_b32_e32 v60, s25, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[56:57], null, v56, v126, 0
	v_ashrrev_i32_e32 v61, 31, v58
	v_sub_co_u32 v58, s7, v64, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v59, null, v59, v61, s7
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_cndmask_b32_e32 v61, s24, v62, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	v_add_co_u32 v56, vcc_lo, v61, v56
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v60, v57, vcc_lo
	v_add_co_u32 v56, vcc_lo, v56, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v57, v59, vcc_lo
	global_store_b32 v[56:57], v63, off offset:28
.LBB1_27:                               ; %Flow850
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB1_44
; %bb.28:                               ; %.preheader.1
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_30
; %bb.29:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v56, s19 :: v_dual_mov_b32 v57, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v61, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v60, s25, v56, vcc_lo
	v_cndmask_b32_e32 v56, s21, v57, vcc_lo
	v_cndmask_b32_e64 v58, s20, 0, vcc_lo
	v_cndmask_b32_e32 v61, s24, v61, vcc_lo
	v_mad_co_i64_i32 v[56:57], null, v56, v125, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v58, v64, v58
	v_ashrrev_i32_e32 v59, 31, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v56, vcc_lo, v61, v56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, v60, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v56, vcc_lo, v56, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, v57, v59, vcc_lo
	global_store_b32 v[56:57], v48, off
.LBB1_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v48, 1, v64
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v48
	s_cbranch_execz .LBB1_32
; %bb.31:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v56, s19 :: v_dual_mov_b32 v57, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v48
	v_ashrrev_i32_e32 v59, 31, v64
	v_mov_b32_e32 v61, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, s25, v56, vcc_lo
	v_cndmask_b32_e32 v56, s21, v57, vcc_lo
	v_cndmask_b32_e64 v58, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[56:57], null, v56, v125, 0
	v_ashrrev_i32_e32 v60, 31, v58
	v_sub_co_u32 v58, s6, v64, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v59, null, v59, v60, s6
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_cndmask_b32_e32 v60, s24, v61, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	v_add_co_u32 v56, vcc_lo, v60, v56
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v48, null, v48, v57, vcc_lo
	v_add_co_u32 v56, vcc_lo, v56, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v48, v59, vcc_lo
	global_store_b32 v[56:57], v49, off offset:4
.LBB1_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v48, 2, v64
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v48
	s_cbranch_execz .LBB1_34
; %bb.33:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v49, s19 :: v_dual_mov_b32 v56, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v48
	v_ashrrev_i32_e32 v57, 31, v64
	v_mov_b32_e32 v60, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, s21, v56, vcc_lo
	v_cndmask_b32_e64 v56, s20, 0, vcc_lo
	v_cndmask_b32_e32 v58, s25, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[48:49], null, v48, v125, 0
	v_ashrrev_i32_e32 v59, 31, v56
	v_sub_co_u32 v56, s6, v64, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v57, null, v57, v59, s6
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_cndmask_b32_e32 v59, s24, v60, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v48, vcc_lo, v59, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v58, v49, vcc_lo
	v_add_co_u32 v48, vcc_lo, v48, v56
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v49, v57, vcc_lo
	global_store_b32 v[48:49], v50, off offset:8
.LBB1_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v48, 3, v64
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v48
	s_cbranch_execz .LBB1_36
; %bb.35:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v49, s19 :: v_dual_mov_b32 v50, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v48
	v_ashrrev_i32_e32 v57, 31, v64
	v_mov_b32_e32 v60, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, s21, v50, vcc_lo
	v_cndmask_b32_e64 v50, s20, 0, vcc_lo
	v_cndmask_b32_e32 v58, s25, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[48:49], null, v48, v125, 0
	v_ashrrev_i32_e32 v59, 31, v50
	v_sub_co_u32 v56, s6, v64, v50
	v_cndmask_b32_e32 v50, s24, v60, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_ci_u32_e64 v57, null, v57, v59, s6
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v48, vcc_lo, v50, v48
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v58, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v48, vcc_lo, v48, v56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v49, v57, vcc_lo
	global_store_b32 v[48:49], v51, off offset:12
.LBB1_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v48, 4, v64
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v48
	s_cbranch_execz .LBB1_38
; %bb.37:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v49, s19 :: v_dual_mov_b32 v50, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v48
	v_ashrrev_i32_e32 v51, 31, v64
	v_mov_b32_e32 v58, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, s21, v50, vcc_lo
	v_cndmask_b32_e64 v50, s20, 0, vcc_lo
	v_cndmask_b32_e32 v56, s25, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[48:49], null, v48, v125, 0
	v_ashrrev_i32_e32 v57, 31, v50
	v_sub_co_u32 v50, s6, v64, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v51, null, v51, v57, s6
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_cndmask_b32_e32 v57, s24, v58, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v48, vcc_lo, v57, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v56, v49, vcc_lo
	v_add_co_u32 v48, vcc_lo, v48, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v49, v51, vcc_lo
	global_store_b32 v[48:49], v52, off offset:16
.LBB1_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v48, 5, v64
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v48
	s_cbranch_execz .LBB1_40
; %bb.39:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v49, s19 :: v_dual_mov_b32 v50, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v48
	v_ashrrev_i32_e32 v51, 31, v64
	v_mov_b32_e32 v57, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, s21, v50, vcc_lo
	v_cndmask_b32_e64 v50, s20, 0, vcc_lo
	v_cndmask_b32_e32 v52, s25, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[48:49], null, v48, v125, 0
	v_ashrrev_i32_e32 v56, 31, v50
	v_sub_co_u32 v50, s6, v64, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v51, null, v51, v56, s6
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_cndmask_b32_e32 v56, s24, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v48, vcc_lo, v56, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v52, v49, vcc_lo
	v_add_co_u32 v48, vcc_lo, v48, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v49, v51, vcc_lo
	global_store_b32 v[48:49], v53, off offset:20
.LBB1_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v48, 6, v64
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v48
	s_cbranch_execz .LBB1_42
; %bb.41:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v49, s19 :: v_dual_mov_b32 v50, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v48
	v_ashrrev_i32_e32 v51, 31, v64
	v_mov_b32_e32 v56, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, s21, v50, vcc_lo
	v_cndmask_b32_e64 v50, s20, 0, vcc_lo
	v_cndmask_b32_e32 v52, s25, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[48:49], null, v48, v125, 0
	v_ashrrev_i32_e32 v53, 31, v50
	v_sub_co_u32 v50, s6, v64, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v51, null, v51, v53, s6
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_cndmask_b32_e32 v53, s24, v56, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v48, vcc_lo, v53, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v52, v49, vcc_lo
	v_add_co_u32 v48, vcc_lo, v48, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v49, v51, vcc_lo
	global_store_b32 v[48:49], v54, off offset:24
.LBB1_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v48, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v48
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_44
; %bb.43:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v49, s19 :: v_dual_mov_b32 v50, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v48
	v_ashrrev_i32_e32 v51, 31, v64
	v_mov_b32_e32 v54, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, s21, v50, vcc_lo
	v_cndmask_b32_e64 v50, s20, 0, vcc_lo
	v_cndmask_b32_e32 v52, s25, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[48:49], null, v48, v125, 0
	v_ashrrev_i32_e32 v53, 31, v50
	v_sub_co_u32 v50, s6, v64, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v51, null, v51, v53, s6
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_cndmask_b32_e32 v53, s24, v54, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v48, vcc_lo, v53, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v52, v49, vcc_lo
	v_add_co_u32 v48, vcc_lo, v48, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v49, v51, vcc_lo
	global_store_b32 v[48:49], v55, off offset:28
.LBB1_44:                               ; %Flow848
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_61
; %bb.45:                               ; %.preheader.2
	s_mov_b32 s5, exec_lo
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_47
; %bb.46:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v48, s19 :: v_dual_mov_b32 v49, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v53, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v52, s25, v48, vcc_lo
	v_cndmask_b32_e32 v48, s21, v49, vcc_lo
	v_cndmask_b32_e64 v50, s20, 0, vcc_lo
	v_cndmask_b32_e32 v53, s24, v53, vcc_lo
	v_mad_co_i64_i32 v[48:49], null, v48, v124, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v50, v64, v50
	v_ashrrev_i32_e32 v51, 31, v50
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v48, vcc_lo, v53, v48
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v52, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v48, vcc_lo, v48, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v49, v51, vcc_lo
	global_store_b32 v[48:49], v40, off
.LBB1_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v40, 1, v64
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v40
	s_cbranch_execz .LBB1_49
; %bb.48:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v48, s19 :: v_dual_mov_b32 v49, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v40
	v_ashrrev_i32_e32 v51, 31, v64
	v_mov_b32_e32 v53, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, s25, v48, vcc_lo
	v_cndmask_b32_e32 v48, s21, v49, vcc_lo
	v_cndmask_b32_e64 v50, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[48:49], null, v48, v124, 0
	v_ashrrev_i32_e32 v52, 31, v50
	v_sub_co_u32 v50, s5, v64, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v51, null, v51, v52, s5
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_cndmask_b32_e32 v52, s24, v53, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v48, vcc_lo, v52, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v40, null, v40, v49, vcc_lo
	v_add_co_u32 v48, vcc_lo, v48, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v40, v51, vcc_lo
	global_store_b32 v[48:49], v41, off offset:4
.LBB1_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v40, 2, v64
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v40
	s_cbranch_execz .LBB1_51
; %bb.50:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v41, s19 :: v_dual_mov_b32 v48, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v40
	v_ashrrev_i32_e32 v49, 31, v64
	v_mov_b32_e32 v52, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, s21, v48, vcc_lo
	v_cndmask_b32_e64 v48, s20, 0, vcc_lo
	v_cndmask_b32_e32 v50, s25, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[40:41], null, v40, v124, 0
	v_ashrrev_i32_e32 v51, 31, v48
	v_sub_co_u32 v48, s5, v64, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v49, null, v49, v51, s5
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_cndmask_b32_e32 v51, s24, v52, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v40, vcc_lo, v51, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v50, v41, vcc_lo
	v_add_co_u32 v40, vcc_lo, v40, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v41, v49, vcc_lo
	global_store_b32 v[40:41], v42, off offset:8
.LBB1_51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v40, 3, v64
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v40
	s_cbranch_execz .LBB1_53
; %bb.52:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v41, s19 :: v_dual_mov_b32 v42, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v40
	v_ashrrev_i32_e32 v49, 31, v64
	v_mov_b32_e32 v52, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, s21, v42, vcc_lo
	v_cndmask_b32_e64 v42, s20, 0, vcc_lo
	v_cndmask_b32_e32 v50, s25, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[40:41], null, v40, v124, 0
	v_ashrrev_i32_e32 v51, 31, v42
	v_sub_co_u32 v48, s5, v64, v42
	v_cndmask_b32_e32 v42, s24, v52, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_ci_u32_e64 v49, null, v49, v51, s5
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v40, vcc_lo, v42, v40
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v50, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v40, vcc_lo, v40, v48
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v41, v49, vcc_lo
	global_store_b32 v[40:41], v43, off offset:12
.LBB1_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v40, 4, v64
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v40
	s_cbranch_execz .LBB1_55
; %bb.54:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v41, s19 :: v_dual_mov_b32 v42, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v40
	v_ashrrev_i32_e32 v43, 31, v64
	v_mov_b32_e32 v50, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, s21, v42, vcc_lo
	v_cndmask_b32_e64 v42, s20, 0, vcc_lo
	v_cndmask_b32_e32 v48, s25, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[40:41], null, v40, v124, 0
	v_ashrrev_i32_e32 v49, 31, v42
	v_sub_co_u32 v42, s5, v64, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v43, null, v43, v49, s5
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_cndmask_b32_e32 v49, s24, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v40, vcc_lo, v49, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v48, v41, vcc_lo
	v_add_co_u32 v40, vcc_lo, v40, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v41, v43, vcc_lo
	global_store_b32 v[40:41], v44, off offset:16
.LBB1_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v40, 5, v64
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v40
	s_cbranch_execz .LBB1_57
; %bb.56:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v41, s19 :: v_dual_mov_b32 v42, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v40
	v_ashrrev_i32_e32 v43, 31, v64
	v_mov_b32_e32 v49, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, s21, v42, vcc_lo
	v_cndmask_b32_e64 v42, s20, 0, vcc_lo
	v_cndmask_b32_e32 v44, s25, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[40:41], null, v40, v124, 0
	v_ashrrev_i32_e32 v48, 31, v42
	v_sub_co_u32 v42, s5, v64, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v43, null, v43, v48, s5
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_cndmask_b32_e32 v48, s24, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v40, vcc_lo, v48, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v44, v41, vcc_lo
	v_add_co_u32 v40, vcc_lo, v40, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v41, v43, vcc_lo
	global_store_b32 v[40:41], v45, off offset:20
.LBB1_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v40, 6, v64
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v40
	s_cbranch_execz .LBB1_59
; %bb.58:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v41, s19 :: v_dual_mov_b32 v42, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v40
	v_ashrrev_i32_e32 v43, 31, v64
	v_mov_b32_e32 v48, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, s21, v42, vcc_lo
	v_cndmask_b32_e64 v42, s20, 0, vcc_lo
	v_cndmask_b32_e32 v44, s25, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[40:41], null, v40, v124, 0
	v_ashrrev_i32_e32 v45, 31, v42
	v_sub_co_u32 v42, s5, v64, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v43, null, v43, v45, s5
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_cndmask_b32_e32 v45, s24, v48, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v40, vcc_lo, v45, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v44, v41, vcc_lo
	v_add_co_u32 v40, vcc_lo, v40, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v41, v43, vcc_lo
	global_store_b32 v[40:41], v46, off offset:24
.LBB1_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v40, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v40
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_61
; %bb.60:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v41, s19 :: v_dual_mov_b32 v42, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v40
	v_ashrrev_i32_e32 v43, 31, v64
	v_mov_b32_e32 v46, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, s21, v42, vcc_lo
	v_cndmask_b32_e64 v42, s20, 0, vcc_lo
	v_cndmask_b32_e32 v44, s25, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[40:41], null, v40, v124, 0
	v_ashrrev_i32_e32 v45, 31, v42
	v_sub_co_u32 v42, s5, v64, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v43, null, v43, v45, s5
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_cndmask_b32_e32 v45, s24, v46, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v40, vcc_lo, v45, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v44, v41, vcc_lo
	v_add_co_u32 v40, vcc_lo, v40, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v41, v43, vcc_lo
	global_store_b32 v[40:41], v47, off offset:28
.LBB1_61:                               ; %Flow846
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_78
; %bb.62:                               ; %.preheader.3
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_64
; %bb.63:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v40, s19 :: v_dual_mov_b32 v41, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v45, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v44, s25, v40, vcc_lo
	v_cndmask_b32_e32 v40, s21, v41, vcc_lo
	v_cndmask_b32_e64 v42, s20, 0, vcc_lo
	v_cndmask_b32_e32 v45, s24, v45, vcc_lo
	v_mad_co_i64_i32 v[40:41], null, v40, v123, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v42, v64, v42
	v_ashrrev_i32_e32 v43, 31, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v40, vcc_lo, v45, v40
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v44, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v40, vcc_lo, v40, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v41, v43, vcc_lo
	global_store_b32 v[40:41], v32, off
.LBB1_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v32, 1, v64
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v32
	s_cbranch_execz .LBB1_66
; %bb.65:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v40, s19 :: v_dual_mov_b32 v41, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v32
	v_ashrrev_i32_e32 v43, 31, v64
	v_mov_b32_e32 v45, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, s25, v40, vcc_lo
	v_cndmask_b32_e32 v40, s21, v41, vcc_lo
	v_cndmask_b32_e64 v42, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[40:41], null, v40, v123, 0
	v_ashrrev_i32_e32 v44, 31, v42
	v_sub_co_u32 v42, s4, v64, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v43, null, v43, v44, s4
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_cndmask_b32_e32 v44, s24, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v40, vcc_lo, v44, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v32, null, v32, v41, vcc_lo
	v_add_co_u32 v40, vcc_lo, v40, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v32, v43, vcc_lo
	global_store_b32 v[40:41], v33, off offset:4
.LBB1_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v32, 2, v64
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v32
	s_cbranch_execz .LBB1_68
; %bb.67:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v33, s19 :: v_dual_mov_b32 v40, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v32
	v_ashrrev_i32_e32 v41, 31, v64
	v_mov_b32_e32 v44, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, s21, v40, vcc_lo
	v_cndmask_b32_e64 v40, s20, 0, vcc_lo
	v_cndmask_b32_e32 v42, s25, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[32:33], null, v32, v123, 0
	v_ashrrev_i32_e32 v43, 31, v40
	v_sub_co_u32 v40, s4, v64, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v41, null, v41, v43, s4
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_cndmask_b32_e32 v43, s24, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v32, vcc_lo, v43, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v42, v33, vcc_lo
	v_add_co_u32 v32, vcc_lo, v32, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v33, v41, vcc_lo
	global_store_b32 v[32:33], v34, off offset:8
.LBB1_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v32, 3, v64
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v32
	s_cbranch_execz .LBB1_70
; %bb.69:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v33, s19 :: v_dual_mov_b32 v34, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v32
	v_ashrrev_i32_e32 v41, 31, v64
	v_mov_b32_e32 v44, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, s21, v34, vcc_lo
	v_cndmask_b32_e64 v34, s20, 0, vcc_lo
	v_cndmask_b32_e32 v42, s25, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[32:33], null, v32, v123, 0
	v_ashrrev_i32_e32 v43, 31, v34
	v_sub_co_u32 v40, s4, v64, v34
	v_cndmask_b32_e32 v34, s24, v44, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_ci_u32_e64 v41, null, v41, v43, s4
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, v34, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v42, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v32, v40
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v33, v41, vcc_lo
	global_store_b32 v[32:33], v35, off offset:12
.LBB1_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v32, 4, v64
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v32
	s_cbranch_execz .LBB1_72
; %bb.71:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v33, s19 :: v_dual_mov_b32 v34, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v32
	v_ashrrev_i32_e32 v35, 31, v64
	v_mov_b32_e32 v42, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, s21, v34, vcc_lo
	v_cndmask_b32_e64 v34, s20, 0, vcc_lo
	v_cndmask_b32_e32 v40, s25, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[32:33], null, v32, v123, 0
	v_ashrrev_i32_e32 v41, 31, v34
	v_sub_co_u32 v34, s4, v64, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v35, null, v35, v41, s4
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_cndmask_b32_e32 v41, s24, v42, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v32, vcc_lo, v41, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v40, v33, vcc_lo
	v_add_co_u32 v32, vcc_lo, v32, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v33, v35, vcc_lo
	global_store_b32 v[32:33], v36, off offset:16
.LBB1_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v32, 5, v64
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v32
	s_cbranch_execz .LBB1_74
; %bb.73:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v33, s19 :: v_dual_mov_b32 v34, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v32
	v_ashrrev_i32_e32 v35, 31, v64
	v_mov_b32_e32 v41, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, s21, v34, vcc_lo
	v_cndmask_b32_e64 v34, s20, 0, vcc_lo
	v_cndmask_b32_e32 v36, s25, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[32:33], null, v32, v123, 0
	v_ashrrev_i32_e32 v40, 31, v34
	v_sub_co_u32 v34, s4, v64, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v35, null, v35, v40, s4
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_cndmask_b32_e32 v40, s24, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v32, vcc_lo, v40, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v36, v33, vcc_lo
	v_add_co_u32 v32, vcc_lo, v32, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v33, v35, vcc_lo
	global_store_b32 v[32:33], v37, off offset:20
.LBB1_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v32, 6, v64
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v32
	s_cbranch_execz .LBB1_76
; %bb.75:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v33, s19 :: v_dual_mov_b32 v34, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v32
	v_ashrrev_i32_e32 v35, 31, v64
	v_mov_b32_e32 v40, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, s21, v34, vcc_lo
	v_cndmask_b32_e64 v34, s20, 0, vcc_lo
	v_cndmask_b32_e32 v36, s25, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[32:33], null, v32, v123, 0
	v_ashrrev_i32_e32 v37, 31, v34
	v_sub_co_u32 v34, s4, v64, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v35, null, v35, v37, s4
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_cndmask_b32_e32 v37, s24, v40, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v32, vcc_lo, v37, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v36, v33, vcc_lo
	v_add_co_u32 v32, vcc_lo, v32, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v33, v35, vcc_lo
	global_store_b32 v[32:33], v38, off offset:24
.LBB1_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v32, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v32
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_78
; %bb.77:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v33, s19 :: v_dual_mov_b32 v34, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v32
	v_ashrrev_i32_e32 v35, 31, v64
	v_mov_b32_e32 v38, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, s21, v34, vcc_lo
	v_cndmask_b32_e64 v34, s20, 0, vcc_lo
	v_cndmask_b32_e32 v36, s25, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[32:33], null, v32, v123, 0
	v_ashrrev_i32_e32 v37, 31, v34
	v_sub_co_u32 v34, s4, v64, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v35, null, v35, v37, s4
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_cndmask_b32_e32 v37, s24, v38, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v32, vcc_lo, v37, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v36, v33, vcc_lo
	v_add_co_u32 v32, vcc_lo, v32, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v33, v35, vcc_lo
	global_store_b32 v[32:33], v39, off offset:28
.LBB1_78:                               ; %Flow844
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_95
; %bb.79:                               ; %.preheader.4
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_81
; %bb.80:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v32, s19 :: v_dual_mov_b32 v33, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v37, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v36, s25, v32, vcc_lo
	v_cndmask_b32_e32 v32, s21, v33, vcc_lo
	v_cndmask_b32_e64 v34, s20, 0, vcc_lo
	v_cndmask_b32_e32 v37, s24, v37, vcc_lo
	v_mad_co_i64_i32 v[32:33], null, v32, v122, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v34, v64, v34
	v_ashrrev_i32_e32 v35, 31, v34
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, v37, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v36, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v32, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v33, v35, vcc_lo
	global_store_b32 v[32:33], v24, off
.LBB1_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v24, 1, v64
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v24
	s_cbranch_execz .LBB1_83
; %bb.82:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v32, s19 :: v_dual_mov_b32 v33, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v24
	v_ashrrev_i32_e32 v35, 31, v64
	v_mov_b32_e32 v37, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, s25, v32, vcc_lo
	v_cndmask_b32_e32 v32, s21, v33, vcc_lo
	v_cndmask_b32_e64 v34, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[32:33], null, v32, v122, 0
	v_ashrrev_i32_e32 v36, 31, v34
	v_sub_co_u32 v34, s3, v64, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v35, null, v35, v36, s3
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_cndmask_b32_e32 v36, s24, v37, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v32, vcc_lo, v36, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v24, null, v24, v33, vcc_lo
	v_add_co_u32 v32, vcc_lo, v32, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v24, v35, vcc_lo
	global_store_b32 v[32:33], v25, off offset:4
.LBB1_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v24, 2, v64
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v24
	s_cbranch_execz .LBB1_85
; %bb.84:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v25, s19 :: v_dual_mov_b32 v32, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v24
	v_ashrrev_i32_e32 v33, 31, v64
	v_mov_b32_e32 v36, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, s21, v32, vcc_lo
	v_cndmask_b32_e64 v32, s20, 0, vcc_lo
	v_cndmask_b32_e32 v34, s25, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[24:25], null, v24, v122, 0
	v_ashrrev_i32_e32 v35, 31, v32
	v_sub_co_u32 v32, s3, v64, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v33, null, v33, v35, s3
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_cndmask_b32_e32 v35, s24, v36, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v24, vcc_lo, v35, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v34, v25, vcc_lo
	v_add_co_u32 v24, vcc_lo, v24, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v25, v33, vcc_lo
	global_store_b32 v[24:25], v26, off offset:8
.LBB1_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v24, 3, v64
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v24
	s_cbranch_execz .LBB1_87
; %bb.86:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v25, s19 :: v_dual_mov_b32 v26, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v24
	v_ashrrev_i32_e32 v33, 31, v64
	v_mov_b32_e32 v36, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, s21, v26, vcc_lo
	v_cndmask_b32_e64 v26, s20, 0, vcc_lo
	v_cndmask_b32_e32 v34, s25, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[24:25], null, v24, v122, 0
	v_ashrrev_i32_e32 v35, 31, v26
	v_sub_co_u32 v32, s3, v64, v26
	v_cndmask_b32_e32 v26, s24, v36, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_ci_u32_e64 v33, null, v33, v35, s3
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v24, vcc_lo, v26, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v34, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v24, vcc_lo, v24, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v25, v33, vcc_lo
	global_store_b32 v[24:25], v27, off offset:12
.LBB1_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v24, 4, v64
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v24
	s_cbranch_execz .LBB1_89
; %bb.88:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v25, s19 :: v_dual_mov_b32 v26, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v24
	v_ashrrev_i32_e32 v27, 31, v64
	v_mov_b32_e32 v34, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, s21, v26, vcc_lo
	v_cndmask_b32_e64 v26, s20, 0, vcc_lo
	v_cndmask_b32_e32 v32, s25, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[24:25], null, v24, v122, 0
	v_ashrrev_i32_e32 v33, 31, v26
	v_sub_co_u32 v26, s3, v64, v26
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v27, null, v27, v33, s3
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_cndmask_b32_e32 v33, s24, v34, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	v_add_co_u32 v24, vcc_lo, v33, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v32, v25, vcc_lo
	v_add_co_u32 v24, vcc_lo, v24, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v25, v27, vcc_lo
	global_store_b32 v[24:25], v28, off offset:16
.LBB1_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v24, 5, v64
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v24
	s_cbranch_execz .LBB1_91
; %bb.90:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v25, s19 :: v_dual_mov_b32 v26, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v24
	v_ashrrev_i32_e32 v27, 31, v64
	v_mov_b32_e32 v33, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, s21, v26, vcc_lo
	v_cndmask_b32_e64 v26, s20, 0, vcc_lo
	v_cndmask_b32_e32 v28, s25, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[24:25], null, v24, v122, 0
	v_ashrrev_i32_e32 v32, 31, v26
	v_sub_co_u32 v26, s3, v64, v26
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v27, null, v27, v32, s3
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_cndmask_b32_e32 v32, s24, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	v_add_co_u32 v24, vcc_lo, v32, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v28, v25, vcc_lo
	v_add_co_u32 v24, vcc_lo, v24, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v25, v27, vcc_lo
	global_store_b32 v[24:25], v29, off offset:20
.LBB1_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v24, 6, v64
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v24
	s_cbranch_execz .LBB1_93
; %bb.92:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v25, s19 :: v_dual_mov_b32 v26, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v24
	v_ashrrev_i32_e32 v27, 31, v64
	v_mov_b32_e32 v32, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, s21, v26, vcc_lo
	v_cndmask_b32_e64 v26, s20, 0, vcc_lo
	v_cndmask_b32_e32 v28, s25, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[24:25], null, v24, v122, 0
	v_ashrrev_i32_e32 v29, 31, v26
	v_sub_co_u32 v26, s3, v64, v26
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v27, null, v27, v29, s3
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_cndmask_b32_e32 v29, s24, v32, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	v_add_co_u32 v24, vcc_lo, v29, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v28, v25, vcc_lo
	v_add_co_u32 v24, vcc_lo, v24, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v25, v27, vcc_lo
	global_store_b32 v[24:25], v30, off offset:24
.LBB1_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v24, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v24
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_95
; %bb.94:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v25, s19 :: v_dual_mov_b32 v26, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v24
	v_ashrrev_i32_e32 v27, 31, v64
	v_mov_b32_e32 v30, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, s21, v26, vcc_lo
	v_cndmask_b32_e64 v26, s20, 0, vcc_lo
	v_cndmask_b32_e32 v28, s25, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[24:25], null, v24, v122, 0
	v_ashrrev_i32_e32 v29, 31, v26
	v_sub_co_u32 v26, s3, v64, v26
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v27, null, v27, v29, s3
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_cndmask_b32_e32 v29, s24, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	v_add_co_u32 v24, vcc_lo, v29, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v28, v25, vcc_lo
	v_add_co_u32 v24, vcc_lo, v24, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v25, v27, vcc_lo
	global_store_b32 v[24:25], v31, off offset:28
.LBB1_95:                               ; %Flow842
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_112
; %bb.96:                               ; %.preheader.5
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_98
; %bb.97:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v24, s19 :: v_dual_mov_b32 v25, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v29, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v28, s25, v24, vcc_lo
	v_cndmask_b32_e32 v24, s21, v25, vcc_lo
	v_cndmask_b32_e64 v26, s20, 0, vcc_lo
	v_cndmask_b32_e32 v29, s24, v29, vcc_lo
	v_mad_co_i64_i32 v[24:25], null, v24, v121, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v26, v64, v26
	v_ashrrev_i32_e32 v27, 31, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v24, vcc_lo, v29, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v28, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v24, vcc_lo, v24, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v25, v27, vcc_lo
	global_store_b32 v[24:25], v16, off
.LBB1_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v16, 1, v64
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v16
	s_cbranch_execz .LBB1_100
; %bb.99:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v24, s19 :: v_dual_mov_b32 v25, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v16
	v_ashrrev_i32_e32 v27, 31, v64
	v_mov_b32_e32 v29, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, s25, v24, vcc_lo
	v_cndmask_b32_e32 v24, s21, v25, vcc_lo
	v_cndmask_b32_e64 v26, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[24:25], null, v24, v121, 0
	v_ashrrev_i32_e32 v28, 31, v26
	v_sub_co_u32 v26, s2, v64, v26
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v27, null, v27, v28, s2
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_cndmask_b32_e32 v28, s24, v29, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	v_add_co_u32 v24, vcc_lo, v28, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v25, vcc_lo
	v_add_co_u32 v24, vcc_lo, v24, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v16, v27, vcc_lo
	global_store_b32 v[24:25], v17, off offset:4
.LBB1_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v16, 2, v64
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v16
	s_cbranch_execz .LBB1_102
; %bb.101:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v17, s19 :: v_dual_mov_b32 v24, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v16
	v_ashrrev_i32_e32 v25, 31, v64
	v_mov_b32_e32 v28, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, s21, v24, vcc_lo
	v_cndmask_b32_e64 v24, s20, 0, vcc_lo
	v_cndmask_b32_e32 v26, s25, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[16:17], null, v16, v121, 0
	v_ashrrev_i32_e32 v27, 31, v24
	v_sub_co_u32 v24, s2, v64, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v25, null, v25, v27, s2
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_cndmask_b32_e32 v27, s24, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v16, vcc_lo, v27, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v26, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v25, vcc_lo
	global_store_b32 v[16:17], v18, off offset:8
.LBB1_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v16, 3, v64
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v16
	s_cbranch_execz .LBB1_104
; %bb.103:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v17, s19 :: v_dual_mov_b32 v18, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v16
	v_ashrrev_i32_e32 v25, 31, v64
	v_mov_b32_e32 v28, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, s21, v18, vcc_lo
	v_cndmask_b32_e64 v18, s20, 0, vcc_lo
	v_cndmask_b32_e32 v26, s25, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[16:17], null, v16, v121, 0
	v_ashrrev_i32_e32 v27, 31, v18
	v_sub_co_u32 v24, s2, v64, v18
	v_cndmask_b32_e32 v18, s24, v28, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_ci_u32_e64 v25, null, v25, v27, s2
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v16, vcc_lo, v18, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v26, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, v16, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v25, vcc_lo
	global_store_b32 v[16:17], v19, off offset:12
.LBB1_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v16, 4, v64
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v16
	s_cbranch_execz .LBB1_106
; %bb.105:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v17, s19 :: v_dual_mov_b32 v18, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v16
	v_ashrrev_i32_e32 v19, 31, v64
	v_mov_b32_e32 v26, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, s21, v18, vcc_lo
	v_cndmask_b32_e64 v18, s20, 0, vcc_lo
	v_cndmask_b32_e32 v24, s25, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[16:17], null, v16, v121, 0
	v_ashrrev_i32_e32 v25, 31, v18
	v_sub_co_u32 v18, s2, v64, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v19, null, v19, v25, s2
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_cndmask_b32_e32 v25, s24, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	v_add_co_u32 v16, vcc_lo, v25, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v24, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	global_store_b32 v[16:17], v20, off offset:16
.LBB1_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v16, 5, v64
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v16
	s_cbranch_execz .LBB1_108
; %bb.107:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v17, s19 :: v_dual_mov_b32 v18, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v16
	v_ashrrev_i32_e32 v19, 31, v64
	v_mov_b32_e32 v25, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, s21, v18, vcc_lo
	v_cndmask_b32_e64 v18, s20, 0, vcc_lo
	v_cndmask_b32_e32 v20, s25, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[16:17], null, v16, v121, 0
	v_ashrrev_i32_e32 v24, 31, v18
	v_sub_co_u32 v18, s2, v64, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v19, null, v19, v24, s2
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_cndmask_b32_e32 v24, s24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	v_add_co_u32 v16, vcc_lo, v24, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v20, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	global_store_b32 v[16:17], v21, off offset:20
.LBB1_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v16, 6, v64
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v16
	s_cbranch_execz .LBB1_110
; %bb.109:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v17, s19 :: v_dual_mov_b32 v18, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v16
	v_ashrrev_i32_e32 v19, 31, v64
	v_mov_b32_e32 v24, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, s21, v18, vcc_lo
	v_cndmask_b32_e64 v18, s20, 0, vcc_lo
	v_cndmask_b32_e32 v20, s25, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[16:17], null, v16, v121, 0
	v_ashrrev_i32_e32 v21, 31, v18
	v_sub_co_u32 v18, s2, v64, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v19, null, v19, v21, s2
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_cndmask_b32_e32 v21, s24, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	v_add_co_u32 v16, vcc_lo, v21, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v20, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	global_store_b32 v[16:17], v22, off offset:24
.LBB1_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v16, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v16
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_112
; %bb.111:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v17, s19 :: v_dual_mov_b32 v18, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v16
	v_ashrrev_i32_e32 v19, 31, v64
	v_mov_b32_e32 v22, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, s21, v18, vcc_lo
	v_cndmask_b32_e64 v18, s20, 0, vcc_lo
	v_cndmask_b32_e32 v20, s25, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[16:17], null, v16, v121, 0
	v_ashrrev_i32_e32 v21, 31, v18
	v_sub_co_u32 v18, s2, v64, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v19, null, v19, v21, s2
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_cndmask_b32_e32 v21, s24, v22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	v_add_co_u32 v16, vcc_lo, v21, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v20, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	global_store_b32 v[16:17], v23, off offset:28
.LBB1_112:                              ; %Flow840
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_129
; %bb.113:                              ; %.preheader.6
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_115
; %bb.114:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v16, s19 :: v_dual_mov_b32 v17, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v21, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v20, s25, v16, vcc_lo
	v_cndmask_b32_e32 v16, s21, v17, vcc_lo
	v_cndmask_b32_e64 v18, s20, 0, vcc_lo
	v_cndmask_b32_e32 v21, s24, v21, vcc_lo
	v_mad_co_i64_i32 v[16:17], null, v16, v120, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v18, v64, v18
	v_ashrrev_i32_e32 v19, 31, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v16, vcc_lo, v21, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v20, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	global_store_b32 v[16:17], v8, off
.LBB1_115:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v8, 1, v64
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v8
	s_cbranch_execz .LBB1_117
; %bb.116:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v16, s19 :: v_dual_mov_b32 v17, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_ashrrev_i32_e32 v19, 31, v64
	v_mov_b32_e32 v21, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, s25, v16, vcc_lo
	v_cndmask_b32_e32 v16, s21, v17, vcc_lo
	v_cndmask_b32_e64 v18, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[16:17], null, v16, v120, 0
	v_ashrrev_i32_e32 v20, 31, v18
	v_sub_co_u32 v18, s1, v64, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v19, null, v19, v20, s1
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_cndmask_b32_e32 v20, s24, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	v_add_co_u32 v16, vcc_lo, v20, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v8, v19, vcc_lo
	global_store_b32 v[16:17], v9, off offset:4
.LBB1_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 2, v64
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v8
	s_cbranch_execz .LBB1_119
; %bb.118:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s19 :: v_dual_mov_b32 v16, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_ashrrev_i32_e32 v17, 31, v64
	v_mov_b32_e32 v20, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, s21, v16, vcc_lo
	v_cndmask_b32_e64 v16, s20, 0, vcc_lo
	v_cndmask_b32_e32 v18, s25, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[8:9], null, v8, v120, 0
	v_ashrrev_i32_e32 v19, 31, v16
	v_sub_co_u32 v16, s1, v64, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v17, null, v17, v19, s1
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_cndmask_b32_e32 v19, s24, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v8, vcc_lo, v19, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v18, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	global_store_b32 v[8:9], v10, off offset:8
.LBB1_119:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 3, v64
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v8
	s_cbranch_execz .LBB1_121
; %bb.120:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s19 :: v_dual_mov_b32 v10, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_ashrrev_i32_e32 v17, 31, v64
	v_mov_b32_e32 v20, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, s21, v10, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v18, s25, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[8:9], null, v8, v120, 0
	v_ashrrev_i32_e32 v19, 31, v10
	v_sub_co_u32 v16, s1, v64, v10
	v_cndmask_b32_e32 v10, s24, v20, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_ci_u32_e64 v17, null, v17, v19, s1
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v8, vcc_lo, v10, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v18, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	global_store_b32 v[8:9], v11, off offset:12
.LBB1_121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 4, v64
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v8
	s_cbranch_execz .LBB1_123
; %bb.122:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s19 :: v_dual_mov_b32 v10, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_ashrrev_i32_e32 v11, 31, v64
	v_mov_b32_e32 v18, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, s21, v10, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v16, s25, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[8:9], null, v8, v120, 0
	v_ashrrev_i32_e32 v17, 31, v10
	v_sub_co_u32 v10, s1, v64, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v11, null, v11, v17, s1
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_cndmask_b32_e32 v17, s24, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v8, vcc_lo, v17, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v16, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	global_store_b32 v[8:9], v12, off offset:16
.LBB1_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 5, v64
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v8
	s_cbranch_execz .LBB1_125
; %bb.124:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s19 :: v_dual_mov_b32 v10, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_ashrrev_i32_e32 v11, 31, v64
	v_mov_b32_e32 v17, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, s21, v10, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v12, s25, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[8:9], null, v8, v120, 0
	v_ashrrev_i32_e32 v16, 31, v10
	v_sub_co_u32 v10, s1, v64, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v11, null, v11, v16, s1
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_cndmask_b32_e32 v16, s24, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v8, vcc_lo, v16, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v12, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	global_store_b32 v[8:9], v13, off offset:20
.LBB1_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 6, v64
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v8
	s_cbranch_execz .LBB1_127
; %bb.126:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s19 :: v_dual_mov_b32 v10, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_ashrrev_i32_e32 v11, 31, v64
	v_mov_b32_e32 v16, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, s21, v10, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v12, s25, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[8:9], null, v8, v120, 0
	v_ashrrev_i32_e32 v13, 31, v10
	v_sub_co_u32 v10, s1, v64, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v11, null, v11, v13, s1
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_cndmask_b32_e32 v13, s24, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v8, vcc_lo, v13, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v12, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	global_store_b32 v[8:9], v14, off offset:24
.LBB1_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v8
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_129
; %bb.128:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s19 :: v_dual_mov_b32 v10, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_ashrrev_i32_e32 v11, 31, v64
	v_mov_b32_e32 v14, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, s21, v10, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v12, s25, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_i64_i32 v[8:9], null, v8, v120, 0
	v_ashrrev_i32_e32 v13, 31, v10
	v_sub_co_u32 v10, s1, v64, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v11, null, v11, v13, s1
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_cndmask_b32_e32 v13, s24, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v8, vcc_lo, v13, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v12, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	global_store_b32 v[8:9], v15, off offset:28
.LBB1_129:                              ; %Flow838
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_146
; %bb.130:                              ; %.preheader.7
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s26, v64
	s_cbranch_execz .LBB1_132
; %bb.131:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v8, s19 :: v_dual_mov_b32 v9, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v64
	v_mov_b32_e32 v13, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v12, s25, v8, vcc_lo
	v_cndmask_b32_e32 v8, s21, v9, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v13, s24, v13, vcc_lo
	v_mad_co_i64_i32 v[8:9], null, v8, v119, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v10, v64, v10
	v_ashrrev_i32_e32 v11, 31, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v8, vcc_lo, v13, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v12, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	global_store_b32 v[8:9], v0, off
.LBB1_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 1, v64
	v_ashrrev_i32_e32 v0, 31, v64
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s26, v8
	s_cbranch_execz .LBB1_134
; %bb.133:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s19 :: v_dual_mov_b32 v10, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v8
	v_mov_b32_e32 v13, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v8, s21, v10, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v12, s25, v9, vcc_lo
	v_cndmask_b32_e32 v13, s24, v13, vcc_lo
	v_mad_co_i64_i32 v[8:9], null, v8, v119, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v11, 31, v10
	v_sub_co_u32 v10, s0, v64, v10
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v11, null, v0, v11, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v8, vcc_lo, v13, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v12, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	global_store_b32 v[8:9], v1, off offset:4
.LBB1_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 2, v64
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v1
	s_cbranch_execz .LBB1_136
; %bb.135:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v8, s19 :: v_dual_mov_b32 v9, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v1
	v_mov_b32_e32 v12, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v1, s25, v8, vcc_lo
	v_cndmask_b32_e32 v8, s21, v9, vcc_lo
	v_cndmask_b32_e64 v10, s20, 0, vcc_lo
	v_cndmask_b32_e32 v12, s24, v12, vcc_lo
	v_mad_co_i64_i32 v[8:9], null, v8, v119, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v11, 31, v10
	v_sub_co_u32 v10, s0, v64, v10
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v11, null, v0, v11, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v8, vcc_lo, v12, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v1, v11, vcc_lo
	global_store_b32 v[8:9], v2, off offset:8
.LBB1_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 3, v64
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v1
	s_cbranch_execz .LBB1_138
; %bb.137:
	v_mov_b32_e32 v8, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v1
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s19 :: v_dual_mov_b32 v11, s18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, s21, v8, vcc_lo
	v_cndmask_b32_e64 v8, s20, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v10, s25, v2, vcc_lo
	v_cndmask_b32_e32 v11, s24, v11, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v119, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v9, 31, v8
	v_sub_co_u32 v8, s0, v64, v8
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v9, null, v0, v9, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, v11, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v10, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v9, vcc_lo
	global_store_b32 v[1:2], v3, off offset:12
.LBB1_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 4, v64
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v1
	s_cbranch_execz .LBB1_140
; %bb.139:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s19 :: v_dual_mov_b32 v3, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v1
	v_mov_b32_e32 v11, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v1, s21, v3, vcc_lo
	v_cndmask_b32_e64 v3, s20, 0, vcc_lo
	v_cndmask_b32_e32 v10, s25, v2, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v119, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v9, 31, v3
	v_sub_co_u32 v8, s0, v64, v3
	v_cndmask_b32_e32 v3, s24, v11, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v9, null, v0, v9, s0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, vcc_lo, v3, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v10, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v9, vcc_lo
	global_store_b32 v[1:2], v4, off offset:16
.LBB1_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 5, v64
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v1
	s_cbranch_execz .LBB1_142
; %bb.141:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s19 :: v_dual_mov_b32 v3, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v1
	v_mov_b32_e32 v9, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, s21, v3, vcc_lo
	v_cndmask_b32_e64 v3, s20, 0, vcc_lo
	v_cndmask_b32_e32 v8, s25, v2, vcc_lo
	v_cndmask_b32_e32 v9, s24, v9, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v119, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v4, 31, v3
	v_sub_co_u32 v3, s0, v64, v3
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v4, null, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, v9, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v8, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	global_store_b32 v[1:2], v5, off offset:20
.LBB1_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 6, v64
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s26, v1
	s_cbranch_execz .LBB1_144
; %bb.143:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s19 :: v_dual_mov_b32 v3, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v1
	v_mov_b32_e32 v8, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, s21, v3, vcc_lo
	v_cndmask_b32_e64 v3, s20, 0, vcc_lo
	v_cndmask_b32_e32 v5, s25, v2, vcc_lo
	v_cndmask_b32_e32 v8, s24, v8, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v119, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v4, 31, v3
	v_sub_co_u32 v3, s0, v64, v3
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v4, null, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, v8, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	global_store_b32 v[1:2], v6, off offset:24
.LBB1_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v1, 7, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s26, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_146
; %bb.145:
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v2, s19 :: v_dual_mov_b32 v3, s20
	v_cmp_gt_i32_e32 vcc_lo, s20, v1
	v_mov_b32_e32 v6, s18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, s21, v3, vcc_lo
	v_cndmask_b32_e64 v3, s20, 0, vcc_lo
	v_cndmask_b32_e32 v5, s25, v2, vcc_lo
	v_cndmask_b32_e32 v6, s24, v6, vcc_lo
	v_mad_co_i64_i32 v[1:2], null, v1, v119, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v4, 31, v3
	v_sub_co_u32 v3, s0, v64, v3
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
.LBB1_146:                              ; %.loopexit.7
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end1:
	.size	gemm_gate_up_hfq4g256_wmma_gfx12_bt8, .Lfunc_end1-gemm_gate_up_hfq4g256_wmma_gfx12_bt8
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_gate_up_hfq4g256_wmma_gfx12_bt8
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
		.amdhsa_next_free_vgpr 138
		.amdhsa_next_free_sgpr 31
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 125
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
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.num_vgpr, 138
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.num_agpr, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.numbered_sgpr, 31
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.num_named_barrier, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.private_seg_size, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.uses_vcc, 1
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.uses_flat_scratch, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.has_dyn_sized_stack, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.has_recursion, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt8.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15940
; TotalNumSgprs: 33
; NumVgprs: 138
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 17
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 138
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_gate_up_hfq4g256_wmma_gfx12_bt12 ; -- Begin function gemm_gate_up_hfq4g256_wmma_gfx12_bt12
	.globl	gemm_gate_up_hfq4g256_wmma_gfx12_bt12
	.p2align	8
	.type	gemm_gate_up_hfq4g256_wmma_gfx12_bt12,@function
gemm_gate_up_hfq4g256_wmma_gfx12_bt12:  ; @gemm_gate_up_hfq4g256_wmma_gfx12_bt12
; %bb.0:
	s_load_b128 s[12:15], s[0:1], 0x28
	s_lshl_b32 s19, ttmp9, 4
	s_mul_i32 s2, ttmp7, 0xc0
	s_wait_kmcnt 0x0
	s_add_co_i32 s18, s13, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_cmp_ge_i32 s19, s18
	s_cselect_b32 s3, -1, 0
	s_cmp_ge_i32 s2, s15
	s_cselect_b32 s4, -1, 0
	s_or_b32 s3, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB2_130
; %bb.1:                                ; %.preheader177
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	v_and_b32_e32 v107, 15, v0
	v_lshrrev_b32_e32 v96, 4, v0
	s_cmp_gt_i32 s14, 0xff
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v176, s2, v107
	v_lshlrev_b32_e32 v177, 3, v96
	s_cbranch_scc1 .LBB2_3
; %bb.2:                                ; %.preheader177..preheader175_crit_edge
	v_lshlrev_b32_e32 v97, 3, v96
	s_mov_b32 s1, 0
	s_branch .LBB2_4
.LBB2_3:
	s_mov_b32 s1, -1
                                        ; implicit-def: $vgpr97
.LBB2_4:                                ; %Flow1056
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0
	v_cmp_gt_i32_e64 s0, s15, v176
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v8, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v88, 0
	v_add_nc_u32_e32 v165, 0xb0, v176
	v_add_nc_u32_e32 v166, 0xa0, v176
	v_add_nc_u32_e32 v167, 0x90, v176
	v_add_nc_u32_e32 v168, 0x80, v176
	v_add_nc_u32_e32 v169, 0x70, v176
	v_add_nc_u32_e32 v170, 0x60, v176
	v_add_nc_u32_e32 v171, 0x50, v176
	v_add_nc_u32_e32 v172, 64, v176
	v_or_b32_e32 v173, 48, v176
	v_or_b32_e32 v174, 32, v176
	v_or_b32_e32 v175, 16, v176
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB2_10
; %bb.5:                                ; %.lr.ph
	v_or_b32_e32 v0, s19, v107
	s_add_co_i32 s2, s18, -1
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v1, s5 :: v_dual_mov_b32 v2, s4
	s_ashr_i32 s1, s14, 31
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v0, s2, v0
	s_lshr_b32 s1, s1, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_add_co_i32 s3, s14, s1
	s_mov_b32 s1, 0
	v_cmp_gt_i32_e32 vcc_lo, s12, v0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s4, s3, 8
	s_ashr_i32 s3, s2, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s5, s4, 0x88
	v_cndmask_b32_e64 v16, s12, 0, vcc_lo
	v_cndmask_b32_e32 v11, s7, v1, vcc_lo
	v_cndmask_b32_e32 v10, s6, v2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v165
	v_cndmask_b32_e64 v2, 0, v176, s0
	v_sub_nc_u32_e32 v0, v0, v16
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s6, s5, 31
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, 0, v165, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v166
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v166, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v167
	v_mad_co_i64_i32 v[97:98], null, s5, v0, v[10:11]
	v_mad_co_u64_u32 v[0:1], null, v2, s14, 0
	v_ashrrev_i32_e32 v2, 31, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v167, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v168
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[1:2], null, v2, s14, v[1:2]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, 0, v168, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v169
	v_ashrrev_i32_e32 v30, 31, v15
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0, v169, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v175
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, 0, v175, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v170
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[2:3], null, v4, s14, 0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v20, 0, v170, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v171
	v_ashrrev_i32_e32 v4, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v21, 0, v171, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v172
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[3:4], null, v4, s14, v[3:4]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, 0, v172, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v174
	v_lshlrev_b64_e32 v[12:13], 1, v[2:3]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, 0, v174, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s15, v173
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[0:1], null, v5, s14, 0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, 0, v173, vcc_lo
	v_ashrrev_i32_e32 v22, 31, v5
	v_add_co_u32 v24, vcc_lo, s8, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, s9, v7, vcc_lo
	v_mad_co_u64_u32 v[4:5], null, v8, s14, 0
	v_ashrrev_i32_e32 v23, 31, v8
	v_mad_co_u64_u32 v[1:2], null, v22, s14, v[1:2]
	v_mad_co_u64_u32 v[7:8], null, v9, s14, 0
	v_add_co_u32 v22, vcc_lo, s8, v12
	v_ashrrev_i32_e32 v12, 31, v19
	v_mad_co_u64_u32 v[5:6], null, v23, s14, v[5:6]
	v_ashrrev_i32_e32 v6, 31, v9
	v_lshlrev_b64_e32 v[2:3], 1, v[0:1]
	v_mad_co_u64_u32 v[0:1], null, v21, s14, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, s9, v13, vcc_lo
	v_mad_co_u64_u32 v[8:9], null, v6, s14, v[8:9]
	v_ashrrev_i32_e32 v9, 31, v21
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]
	v_add_co_u32 v21, vcc_lo, s8, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s9, v3, vcc_lo
	v_mad_co_u64_u32 v[1:2], null, v9, s14, v[1:2]
	v_lshlrev_b64_e32 v[6:7], 1, v[7:8]
	v_mad_co_u64_u32 v[2:3], null, v20, s14, 0
	v_add_co_u32 v27, vcc_lo, s8, v4
	v_ashrrev_i32_e32 v4, 31, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s9, v5, vcc_lo
	v_add_co_u32 v20, vcc_lo, s8, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s9, v7, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]
	v_mad_co_u64_u32 v[3:4], null, v4, s14, v[3:4]
	v_mad_co_u64_u32 v[0:1], null, v19, s14, 0
	v_mad_co_u64_u32 v[4:5], null, v18, s14, 0
	v_ashrrev_i32_e32 v13, 31, v18
	v_add_co_u32 v18, vcc_lo, s8, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s9, v7, vcc_lo
	v_lshlrev_b64_e32 v[8:9], 1, v[2:3]
	v_mad_co_u64_u32 v[1:2], null, v12, s14, v[1:2]
	v_mad_co_u64_u32 v[5:6], null, v13, s14, v[5:6]
	v_mad_co_u64_u32 v[2:3], null, v15, s14, 0
	v_mad_co_u64_u32 v[6:7], null, v14, s14, 0
	v_add_co_u32 v31, vcc_lo, s8, v8
	v_ashrrev_i32_e32 v8, 31, v14
	v_lshlrev_b64_e32 v[12:13], 1, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 1, v[4:5]
	v_mad_co_u64_u32 v[3:4], null, v30, s14, v[3:4]
	v_mad_co_u64_u32 v[0:1], null, v17, s14, 0
	v_ashrrev_i32_e32 v5, 31, v17
	v_mad_co_u64_u32 v[7:8], null, v8, s14, v[7:8]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, s8, v12
	v_lshlrev_b64_e32 v[3:4], 1, v[2:3]
	v_mad_co_u64_u32 v[1:2], null, v5, s14, v[1:2]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s9, v13, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 1, v[6:7]
	v_add_co_u32 v2, vcc_lo, s8, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s9, v15, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	v_add_co_u32 v3, vcc_lo, s8, v3
	v_lshlrev_b32_e32 v7, 4, v96
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s9, v4, vcc_lo
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	v_add_co_u32 v14, vcc_lo, s8, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s9, v1, vcc_lo
	v_add_co_u32 v99, vcc_lo, v24, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v100, null, 0, v25, vcc_lo
	v_add_co_u32 v101, vcc_lo, v22, v7
	v_add_nc_u32_e32 v0, s19, v107
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, 0, v23, vcc_lo
	v_add_co_u32 v103, vcc_lo, v21, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v104, null, 0, v26, vcc_lo
	v_add_co_u32 v105, vcc_lo, v27, v7
	v_ashrrev_i32_e32 v1, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v106, null, 0, v28, vcc_lo
	v_add_co_u32 v107, vcc_lo, v20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v108, null, 0, v29, vcc_lo
	v_add_co_u32 v109, vcc_lo, v18, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v110, null, 0, v19, vcc_lo
	v_cmp_lt_i64_e32 vcc_lo, s[2:3], v[0:1]
	v_add_co_u32 v111, s0, v31, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v112, null, 0, v9, s0
	v_add_co_u32 v115, s0, v2, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v0, v0, s2, vcc_lo
	v_cndmask_b32_e64 v1, v1, s3, vcc_lo
	v_add_co_u32 v113, vcc_lo, v8, v7
	v_ashrrev_i32_e32 v8, 31, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v114, null, 0, v12, vcc_lo
	v_sub_co_u32 v9, vcc_lo, v0, v16
	s_wait_alu depctr_va_vcc(0)
	v_sub_co_ci_u32_e64 v2, null, v1, v8, vcc_lo
	v_add_co_u32 v117, vcc_lo, v3, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[0:1], null, v9, s5, 0
	v_mul_lo_u32 v2, v2, s5
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v3, v9, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v118, null, 0, v4, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v116, null, 0, v13, s0
	s_mov_b32 s0, s1
	v_lshl_or_b32 v4, v96, 2, v0
	v_mov_b32_e32 v0, 0
	v_add_co_u32 v119, vcc_lo, v5, v7
	v_add3_u32 v1, v1, v3, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v120, null, 0, v6, vcc_lo
	v_add_co_u32 v121, vcc_lo, v14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, 0, v15, vcc_lo
	v_add_co_u32 v8, vcc_lo, v10, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v11, v1, vcc_lo
	v_mov_b32_e32 v7, v0
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_add_co_u32 v123, vcc_lo, v8, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v124, null, 0, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v79, v7 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v87, v7 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v95, v7 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v8, v0
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v16, v0
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v24, v0
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v32, v0
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v40, v0
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v48, v0
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v56, v0
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v64, v0
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v72, v0
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v80, v0
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v88, v0
	s_mov_b32 s5, s1
.LBB2_6:                                ; %.preheader176
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_7 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s2, s5, 0x88
	v_dual_mov_b32 v128, v114 :: v_dual_mov_b32 v127, v113
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v125, vcc_lo, v97, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v126, null, 0, v98, vcc_lo
	v_dual_mov_b32 v130, v110 :: v_dual_mov_b32 v129, v109
	v_dual_mov_b32 v132, v116 :: v_dual_mov_b32 v131, v115
	global_load_b64 v[149:150], v[125:126], off
	v_dual_mov_b32 v126, v112 :: v_dual_mov_b32 v125, v111
	v_dual_mov_b32 v134, v108 :: v_dual_mov_b32 v133, v107
	v_dual_mov_b32 v136, v118 :: v_dual_mov_b32 v135, v117
	v_dual_mov_b32 v138, v106 :: v_dual_mov_b32 v137, v105
	v_dual_mov_b32 v140, v120 :: v_dual_mov_b32 v139, v119
	v_dual_mov_b32 v142, v104 :: v_dual_mov_b32 v141, v103
	v_dual_mov_b32 v144, v122 :: v_dual_mov_b32 v143, v121
	v_dual_mov_b32 v146, v102 :: v_dual_mov_b32 v145, v101
	v_dual_mov_b32 v148, v100 :: v_dual_mov_b32 v147, v99
	s_lshl_b64 s[2:3], s[0:1], 1
	s_mov_b32 s6, -4
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e64 v96.l, v149
	v_cvt_f16_f32_e64 v96.h, v150
	v_dual_mov_b32 v150, v124 :: v_dual_mov_b32 v149, v123
.LBB2_7:                                ;   Parent Loop BB2_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_clause 0x3
	global_load_b32 v152, v[149:150], off offset:-24
	global_load_b32 v180, v[149:150], off offset:-16
	global_load_b32 v179, v[149:150], off offset:-8
	global_load_b32 v178, v[149:150], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v237, vcc_lo, v147, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v238, null, s3, v148, vcc_lo
	s_add_co_i32 s6, s6, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s6, 11
	s_wait_loadcnt 0x3
	v_and_b32_e32 v151, 15, v152
	v_bfe_u32 v153, v152, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v151, v151
	v_cvt_f32_ubyte0_e32 v153, v153
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e64 v151.l, v151
	v_cvt_f16_f32_e64 v151.h, v153
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v181.l, v96.l, v151.l, v96.h
	v_fma_f16 v181.h, v96.l, v151.h, v96.h
	v_bfe_u32 v151, v152, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v151, v151
	v_cvt_f16_f32_e64 v151.l, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v182.l, v96.l, v151.l, v96.h
	v_bfe_u32 v151, v152, 12, 4
	v_cvt_f32_ubyte0_e32 v151, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e64 v151.l, v151
	v_fma_f16 v182.h, v96.l, v151.l, v96.h
	v_bfe_u32 v151, v152, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v151, v151
	v_cvt_f16_f32_e64 v151.l, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v183.l, v96.l, v151.l, v96.h
	v_bfe_u32 v151, v152, 20, 4
	v_cvt_f32_ubyte0_e32 v151, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e64 v151.l, v151
	v_fma_f16 v183.h, v96.l, v151.l, v96.h
	v_bfe_u32 v151, v152, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v151, v151
	v_cvt_f16_f32_e64 v151.l, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v184.l, v96.l, v151.l, v96.h
	v_lshrrev_b32_e32 v151, 28, v152
	v_cvt_f32_ubyte0_e32 v151, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e64 v151.l, v151
	v_fma_f16 v184.h, v96.l, v151.l, v96.h
	s_clause 0x1
	global_load_b128 v[151:154], v[237:238], off
	global_load_b128 v[185:188], v[237:238], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[88:95], v[181:184], v[151:154], v[88:95]
	v_add_co_u32 v151, vcc_lo, v145, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v152, null, s3, v146, vcc_lo
	s_clause 0x1
	global_load_b128 v[153:156], v[151:152], off
	global_load_b128 v[189:192], v[151:152], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[80:87], v[181:184], v[153:156], v[80:87]
	v_add_co_u32 v153, vcc_lo, v141, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v154, null, s3, v142, vcc_lo
	s_clause 0x1
	global_load_b128 v[155:158], v[153:154], off
	global_load_b128 v[193:196], v[153:154], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[72:79], v[181:184], v[155:158], v[72:79]
	v_add_co_u32 v155, vcc_lo, v137, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v156, null, s3, v138, vcc_lo
	s_clause 0x1
	global_load_b128 v[157:160], v[155:156], off
	global_load_b128 v[197:200], v[155:156], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[64:71], v[181:184], v[157:160], v[64:71]
	v_add_co_u32 v157, vcc_lo, v133, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v158, null, s3, v134, vcc_lo
	s_clause 0x1
	global_load_b128 v[159:162], v[157:158], off
	global_load_b128 v[201:204], v[157:158], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[56:63], v[181:184], v[159:162], v[56:63]
	v_add_co_u32 v159, vcc_lo, v129, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v160, null, s3, v130, vcc_lo
	s_clause 0x1
	global_load_b128 v[161:164], v[159:160], off
	global_load_b128 v[205:208], v[159:160], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[48:55], v[181:184], v[161:164], v[48:55]
	v_add_co_u32 v161, vcc_lo, v125, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v162, null, s3, v126, vcc_lo
	s_clause 0x1
	global_load_b128 v[209:212], v[161:162], off
	global_load_b128 v[213:216], v[161:162], off offset:32
	v_add_co_u32 v163, vcc_lo, v127, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v164, null, s3, v128, vcc_lo
	v_add_co_u32 v239, vcc_lo, v131, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v240, null, s3, v132, vcc_lo
	v_add_co_u32 v241, vcc_lo, v135, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v242, null, s3, v136, vcc_lo
	v_add_co_u32 v243, vcc_lo, v139, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v244, null, s3, v140, vcc_lo
	v_add_co_u32 v245, vcc_lo, v143, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v246, null, s3, v144, vcc_lo
	v_add_co_u32 v149, vcc_lo, v149, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v150, null, 0, v150, vcc_lo
	v_add_co_u32 v147, vcc_lo, 0x80, v147
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v148, null, 0, v148, vcc_lo
	v_add_co_u32 v145, vcc_lo, 0x80, v145
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v146, null, 0, v146, vcc_lo
	v_add_co_u32 v143, vcc_lo, 0x80, v143
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v144, null, 0, v144, vcc_lo
	v_add_co_u32 v141, vcc_lo, 0x80, v141
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v142, null, 0, v142, vcc_lo
	v_add_co_u32 v139, vcc_lo, 0x80, v139
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v140, null, 0, v140, vcc_lo
	v_add_co_u32 v137, vcc_lo, 0x80, v137
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, 0, v138, vcc_lo
	v_add_co_u32 v135, vcc_lo, 0x80, v135
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, 0, v136, vcc_lo
	v_add_co_u32 v133, vcc_lo, 0x80, v133
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v134, null, 0, v134, vcc_lo
	v_add_co_u32 v131, vcc_lo, 0x80, v131
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v132, null, 0, v132, vcc_lo
	v_add_co_u32 v129, vcc_lo, 0x80, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, 0, v130, vcc_lo
	v_add_co_u32 v127, vcc_lo, 0x80, v127
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v128, null, 0, v128, vcc_lo
	v_add_co_u32 v125, vcc_lo, 0x80, v125
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v126, null, 0, v126, vcc_lo
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[40:47], v[181:184], v[209:212], v[40:47]
	s_clause 0x1
	global_load_b128 v[209:212], v[163:164], off
	global_load_b128 v[217:220], v[163:164], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[32:39], v[181:184], v[209:212], v[32:39]
	s_clause 0x1
	global_load_b128 v[209:212], v[239:240], off
	global_load_b128 v[221:224], v[239:240], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[24:31], v[181:184], v[209:212], v[24:31]
	s_clause 0x1
	global_load_b128 v[209:212], v[241:242], off
	global_load_b128 v[225:228], v[241:242], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[16:23], v[181:184], v[209:212], v[16:23]
	s_clause 0x1
	global_load_b128 v[209:212], v[243:244], off
	global_load_b128 v[229:232], v[243:244], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[8:15], v[181:184], v[209:212], v[8:15]
	s_clause 0x1
	global_load_b128 v[209:212], v[245:246], off
	global_load_b128 v[233:236], v[245:246], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[0:7], v[181:184], v[209:212], v[0:7]
	v_and_b32_e32 v181, 15, v180
	v_bfe_u32 v182, v180, 4, 4
	v_bfe_u32 v183, v180, 12, 4
	v_bfe_u32 v184, v180, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v181, v181
	v_cvt_f32_ubyte0_e32 v182, v182
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v183, v183
	v_cvt_f32_ubyte0_e32 v184, v184
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e64 v181.l, v181
	v_cvt_f16_f32_e64 v181.h, v182
	v_bfe_u32 v182, v180, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v181.l, v96.l, v181.l, v96.h
	v_fma_f16 v181.h, v96.l, v181.h, v96.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v182, v182
	v_cvt_f16_f32_e64 v182.l, v182
	v_cvt_f16_f32_e64 v182.h, v183
	v_bfe_u32 v183, v180, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v182.l, v96.l, v182.l, v96.h
	v_fma_f16 v182.h, v96.l, v182.h, v96.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v183, v183
	v_cvt_f16_f32_e64 v183.l, v183
	v_cvt_f16_f32_e64 v183.h, v184
	v_bfe_u32 v184, v180, 24, 4
	v_lshrrev_b32_e32 v180, 28, v180
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v183.l, v96.l, v183.l, v96.h
	v_fma_f16 v183.h, v96.l, v183.h, v96.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v184, v184
	v_cvt_f32_ubyte0_e32 v180, v180
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e64 v184.l, v184
	v_cvt_f16_f32_e64 v180.l, v180
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v184.l, v96.l, v184.l, v96.h
	v_fma_f16 v184.h, v96.l, v180.l, v96.h
	v_and_b32_e32 v180, 15, v179
	s_delay_alu instid0(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[88:95], v[181:184], v[185:188], v[88:95]
	v_wmma_f32_16x16x16_f16 v[80:87], v[181:184], v[189:192], v[80:87]
	v_wmma_f32_16x16x16_f16 v[72:79], v[181:184], v[193:196], v[72:79]
	v_wmma_f32_16x16x16_f16 v[64:71], v[181:184], v[197:200], v[64:71]
	v_wmma_f32_16x16x16_f16 v[56:63], v[181:184], v[201:204], v[56:63]
	v_wmma_f32_16x16x16_f16 v[48:55], v[181:184], v[205:208], v[48:55]
	v_wmma_f32_16x16x16_f16 v[40:47], v[181:184], v[213:216], v[40:47]
	v_wmma_f32_16x16x16_f16 v[32:39], v[181:184], v[217:220], v[32:39]
	v_wmma_f32_16x16x16_f16 v[24:31], v[181:184], v[221:224], v[24:31]
	v_wmma_f32_16x16x16_f16 v[16:23], v[181:184], v[225:228], v[16:23]
	v_wmma_f32_16x16x16_f16 v[8:15], v[181:184], v[229:232], v[8:15]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[181:184], v[233:236], v[0:7]
	s_clause 0x1
	global_load_b128 v[184:187], v[237:238], off offset:64
	global_load_b128 v[188:191], v[237:238], off offset:96
	v_bfe_u32 v181, v179, 4, 4
	v_cvt_f32_ubyte0_e32 v180, v180
	v_bfe_u32 v182, v179, 12, 4
	v_bfe_u32 v183, v179, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v181, v181
	v_cvt_f16_f32_e64 v180.l, v180
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v182, v182
	v_cvt_f32_ubyte0_e32 v183, v183
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e64 v180.h, v181
	v_bfe_u32 v181, v179, 8, 4
	v_fma_f16 v180.l, v96.l, v180.l, v96.h
	v_fma_f16 v180.h, v96.l, v180.h, v96.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v181, v181
	v_cvt_f16_f32_e64 v181.l, v181
	v_cvt_f16_f32_e64 v181.h, v182
	v_bfe_u32 v182, v179, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v181.l, v96.l, v181.l, v96.h
	v_fma_f16 v181.h, v96.l, v181.h, v96.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v182, v182
	v_cvt_f16_f32_e64 v182.l, v182
	v_cvt_f16_f32_e64 v182.h, v183
	v_bfe_u32 v183, v179, 24, 4
	v_lshrrev_b32_e32 v179, 28, v179
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v182.l, v96.l, v182.l, v96.h
	v_fma_f16 v182.h, v96.l, v182.h, v96.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v183, v183
	v_cvt_f32_ubyte0_e32 v179, v179
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e64 v183.l, v183
	v_cvt_f16_f32_e64 v179.l, v179
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v183.l, v96.l, v183.l, v96.h
	v_fma_f16 v183.h, v96.l, v179.l, v96.h
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[88:95], v[180:183], v[184:187], v[88:95]
	s_clause 0x1
	global_load_b128 v[184:187], v[151:152], off offset:64
	global_load_b128 v[192:195], v[151:152], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[80:87], v[180:183], v[184:187], v[80:87]
	s_clause 0x1
	global_load_b128 v[184:187], v[153:154], off offset:64
	global_load_b128 v[151:154], v[153:154], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[72:79], v[180:183], v[184:187], v[72:79]
	s_clause 0x1
	global_load_b128 v[184:187], v[155:156], off offset:64
	global_load_b128 v[196:199], v[155:156], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[64:71], v[180:183], v[184:187], v[64:71]
	s_clause 0x1
	global_load_b128 v[184:187], v[157:158], off offset:64
	global_load_b128 v[155:158], v[157:158], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[56:63], v[180:183], v[184:187], v[56:63]
	s_clause 0x1
	global_load_b128 v[184:187], v[159:160], off offset:64
	global_load_b128 v[200:203], v[159:160], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[48:55], v[180:183], v[184:187], v[48:55]
	s_clause 0x1
	global_load_b128 v[184:187], v[161:162], off offset:64
	global_load_b128 v[159:162], v[161:162], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[40:47], v[180:183], v[184:187], v[40:47]
	s_clause 0x1
	global_load_b128 v[184:187], v[163:164], off offset:64
	global_load_b128 v[204:207], v[163:164], off offset:96
	v_and_b32_e32 v163, 15, v178
	v_bfe_u32 v164, v178, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v163, v163
	v_cvt_f32_ubyte0_e32 v164, v164
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e64 v163.l, v163
	v_cvt_f16_f32_e64 v163.h, v164
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v179.l, v96.l, v163.l, v96.h
	v_fma_f16 v179.h, v96.l, v163.h, v96.h
	v_bfe_u32 v163, v178, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v163, v163
	v_cvt_f16_f32_e64 v163.l, v163
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[32:39], v[180:183], v[184:187], v[32:39]
	s_clause 0x1
	global_load_b128 v[184:187], v[239:240], off offset:64
	global_load_b128 v[208:211], v[239:240], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[24:31], v[180:183], v[184:187], v[24:31]
	s_clause 0x1
	global_load_b128 v[184:187], v[241:242], off offset:64
	global_load_b128 v[212:215], v[241:242], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[16:23], v[180:183], v[184:187], v[16:23]
	s_clause 0x1
	global_load_b128 v[184:187], v[243:244], off offset:64
	global_load_b128 v[216:219], v[243:244], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[8:15], v[180:183], v[184:187], v[8:15]
	s_clause 0x1
	global_load_b128 v[184:187], v[245:246], off offset:64
	global_load_b128 v[220:223], v[245:246], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_f16 v[0:7], v[180:183], v[184:187], v[0:7]
	v_fma_f16 v180.l, v96.l, v163.l, v96.h
	v_bfe_u32 v163, v178, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v163, v163
	v_cvt_f16_f32_e64 v163.l, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v180.h, v96.l, v163.l, v96.h
	v_bfe_u32 v163, v178, 16, 4
	v_cvt_f32_ubyte0_e32 v163, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e64 v163.l, v163
	v_fma_f16 v181.l, v96.l, v163.l, v96.h
	v_bfe_u32 v163, v178, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v163, v163
	v_cvt_f16_f32_e64 v163.l, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v181.h, v96.l, v163.l, v96.h
	v_bfe_u32 v163, v178, 24, 4
	v_cvt_f32_ubyte0_e32 v163, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e64 v163.l, v163
	v_fma_f16 v182.l, v96.l, v163.l, v96.h
	v_lshrrev_b32_e32 v163, 28, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v163, v163
	v_cvt_f16_f32_e64 v163.l, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f16 v182.h, v96.l, v163.l, v96.h
	v_wmma_f32_16x16x16_f16 v[88:95], v[179:182], v[188:191], v[88:95]
	v_wmma_f32_16x16x16_f16 v[80:87], v[179:182], v[192:195], v[80:87]
	v_wmma_f32_16x16x16_f16 v[72:79], v[179:182], v[151:154], v[72:79]
	v_wmma_f32_16x16x16_f16 v[64:71], v[179:182], v[196:199], v[64:71]
	v_wmma_f32_16x16x16_f16 v[56:63], v[179:182], v[155:158], v[56:63]
	v_wmma_f32_16x16x16_f16 v[48:55], v[179:182], v[200:203], v[48:55]
	v_wmma_f32_16x16x16_f16 v[40:47], v[179:182], v[159:162], v[40:47]
	v_wmma_f32_16x16x16_f16 v[32:39], v[179:182], v[204:207], v[32:39]
	v_wmma_f32_16x16x16_f16 v[24:31], v[179:182], v[208:211], v[24:31]
	v_wmma_f32_16x16x16_f16 v[16:23], v[179:182], v[212:215], v[16:23]
	v_wmma_f32_16x16x16_f16 v[8:15], v[179:182], v[216:219], v[8:15]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[179:182], v[220:223], v[0:7]
	s_cbranch_scc0 .LBB2_7
; %bb.8:                                ;   in Loop: Header=BB2_6 Depth=1
	v_add_co_u32 v123, vcc_lo, 0x88, v123
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v124, null, 0, v124, vcc_lo
	s_add_co_i32 s5, s5, 1
	s_addk_co_i32 s0, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, s4
	s_cbranch_scc0 .LBB2_6
; %bb.9:                                ; %.preheader175.loopexit
	v_mov_b32_e32 v97, v177
.LBB2_10:                               ; %Flow1057
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v120, s10 :: v_dual_add_nc_u32 v113, s19, v97
	v_dual_mov_b32 v115, s11 :: v_dual_mov_b32 v122, s12
	v_or_b32_e32 v97, 1, v113
	v_cmp_gt_i32_e64 s0, s12, v113
	v_ashrrev_i32_e32 v119, 31, v113
	v_or_b32_e32 v105, 2, v113
	v_or_b32_e32 v114, 4, v113
	v_cmp_gt_i32_e64 s1, s12, v97
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v96, s12, 0, s0
	v_cndmask_b32_e64 v104, s17, v115, s0
	v_cndmask_b32_e64 v106, s16, v120, s0
	v_cndmask_b32_e64 v99, s13, v122, s0
	v_cndmask_b32_e64 v98, s12, 0, s1
	v_sub_nc_u32_e32 v96, v113, v96
	v_cmp_gt_i32_e64 s0, s18, v97
	v_cmp_gt_i32_e64 s2, s12, v105
	v_cndmask_b32_e64 v109, s16, v120, s1
	v_ashrrev_i32_e32 v101, 31, v98
	v_ashrrev_i32_e32 v97, 31, v96
	v_sub_co_u32 v100, s3, v113, v98
	v_cndmask_b32_e64 v107, s17, v115, s1
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v101, null, v119, v101, s3
	v_lshlrev_b64_e32 v[97:98], 2, v[96:97]
	v_cndmask_b32_e64 v110, s12, 0, s2
	v_cndmask_b32_e64 v96, s13, v122, s1
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[102:103], 2, v[100:101]
	v_cndmask_b32_e64 v108, s17, v115, s2
	v_or_b32_e32 v117, 5, v113
	v_add_co_u32 v100, s1, v106, v97
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v101, null, v104, v98, s1
	v_add_co_u32 v97, s1, v109, v102
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v98, null, v107, v103, s1
	v_ashrrev_i32_e32 v103, 31, v110
	v_or_b32_e32 v106, 3, v113
	v_sub_co_u32 v102, s1, v113, v110
	v_cndmask_b32_e64 v107, s16, v120, s2
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v103, null, v119, v103, s1
	v_cmp_gt_i32_e64 s3, s12, v106
	v_cmp_gt_i32_e64 s1, s18, v105
	v_cmp_gt_i32_e64 s5, s12, v117
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[104:105], 2, v[102:103]
	v_cndmask_b32_e64 v103, s13, v122, s2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v102, s12, 0, s3
	v_cmp_gt_i32_e64 s2, s12, v114
	v_cndmask_b32_e64 v110, s17, v115, s3
	v_or_b32_e32 v121, 6, v113
	v_add_co_u32 v107, s4, v107, v104
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v108, null, v108, v105, s4
	v_ashrrev_i32_e32 v105, 31, v102
	v_cndmask_b32_e64 v109, s12, 0, s2
	v_sub_co_u32 v104, s4, v113, v102
	v_cndmask_b32_e64 v102, s13, v122, s3
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v105, null, v119, v105, s4
	v_ashrrev_i32_e32 v112, 31, v109
	v_cmp_gt_i32_e64 s4, s18, v106
	v_cndmask_b32_e64 v106, s16, v120, s3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[104:105], 2, v[104:105]
	v_sub_co_u32 v111, s3, v113, v109
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v112, null, v119, v112, s3
	v_cndmask_b32_e64 v116, s17, v115, s2
	v_add_co_u32 v109, s3, v106, v104
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v110, null, v110, v105, s3
	v_lshlrev_b64_e32 v[105:106], 2, v[111:112]
	v_cndmask_b32_e64 v112, s12, 0, s5
	v_cndmask_b32_e64 v111, s16, v120, s2
	v_cmp_gt_i32_e64 s3, s18, v114
	v_cndmask_b32_e64 v104, s13, v122, s2
	v_or_b32_e32 v123, 7, v113
	v_ashrrev_i32_e32 v114, 31, v112
	v_add_co_u32 v105, s2, v111, v105
	v_sub_co_u32 v111, s6, v113, v112
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_sub_co_ci_u32_e64 v112, null, v119, v114, s6
	v_cmp_gt_i32_e64 s6, s12, v121
	v_cmp_gt_i32_e64 s7, s12, v123
	v_add_co_ci_u32_e64 v106, null, v116, v106, s2
	v_lshlrev_b64_e32 v[111:112], 2, v[111:112]
	v_cndmask_b32_e64 v116, s16, v120, s5
	v_cndmask_b32_e64 v118, s17, v115, s5
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v125, s12, 0, s6
	v_cndmask_b32_e64 v124, s17, v115, s6
	v_cndmask_b32_e64 v126, s17, v115, s7
	v_cndmask_b32_e64 v115, s12, 0, s7
	v_cndmask_b32_e64 v114, s13, v122, s5
	v_add_co_u32 v116, s5, v116, v111
	v_cmp_gt_i32_e64 s2, s18, v117
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v117, null, v118, v112, s5
	v_ashrrev_i32_e32 v112, 31, v125
	v_cmp_gt_i32_e64 s5, s18, v121
	v_ashrrev_i32_e32 v121, 31, v115
	v_sub_co_u32 v111, s8, v113, v125
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v112, null, v119, v112, s8
	v_sub_co_u32 v118, s8, v113, v115
	s_wait_alu depctr_va_sdst(0)
	v_sub_co_ci_u32_e64 v119, null, v119, v121, s8
	v_cmp_gt_i32_e32 vcc_lo, s18, v113
	v_lshlrev_b64_e32 v[111:112], 2, v[111:112]
	v_cndmask_b32_e64 v113, s16, v120, s6
	v_cndmask_b32_e64 v125, s16, v120, s7
	v_lshlrev_b64_e32 v[120:121], 2, v[118:119]
	v_cndmask_b32_e64 v115, s13, v122, s6
	s_mov_b32 s8, exec_lo
	v_add_co_u32 v118, s6, v113, v111
	v_cndmask_b32_e64 v113, s13, v122, s7
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v111, s7, v125, v120
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v119, null, v124, v112, s6
	v_cmp_gt_i32_e64 s6, s18, v123
	v_add_co_ci_u32_e64 v112, null, v126, v121, s7
	v_cmpx_gt_i32_e64 s15, v176
	s_cbranch_execz .LBB2_20
; %bb.11:                               ; %.preheader
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_131
; %bb.12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_132
.LBB2_13:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_133
.LBB2_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_134
.LBB2_15:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_135
.LBB2_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_136
.LBB2_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_137
.LBB2_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_20
.LBB2_19:
	v_mad_co_i64_i32 v[88:89], null, v113, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v111, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v89, null, v112, v89, s7
	global_store_b32 v[88:89], v95, off offset:28
.LBB2_20:                               ; %Flow1055
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v175
	s_cbranch_execz .LBB2_30
; %bb.21:                               ; %.preheader.1
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_138
; %bb.22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_139
.LBB2_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_140
.LBB2_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_141
.LBB2_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_142
.LBB2_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_143
.LBB2_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_144
.LBB2_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_30
.LBB2_29:
	v_mad_co_i64_i32 v[80:81], null, v113, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v111, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v81, null, v112, v81, s7
	global_store_b32 v[80:81], v87, off offset:28
.LBB2_30:                               ; %Flow1053
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v174
	s_cbranch_execz .LBB2_40
; %bb.31:                               ; %.preheader.2
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_145
; %bb.32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_146
.LBB2_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_147
.LBB2_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_148
.LBB2_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_149
.LBB2_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_150
.LBB2_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_151
.LBB2_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_40
.LBB2_39:
	v_mad_co_i64_i32 v[72:73], null, v113, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v111, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v112, v73, s7
	global_store_b32 v[72:73], v79, off offset:28
.LBB2_40:                               ; %Flow1051
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v173
	s_cbranch_execz .LBB2_50
; %bb.41:                               ; %.preheader.3
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_152
; %bb.42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_153
.LBB2_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_154
.LBB2_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_155
.LBB2_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_156
.LBB2_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_157
.LBB2_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_158
.LBB2_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_50
.LBB2_49:
	v_mad_co_i64_i32 v[64:65], null, v113, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v111, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v112, v65, s7
	global_store_b32 v[64:65], v71, off offset:28
.LBB2_50:                               ; %Flow1049
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v172
	s_cbranch_execz .LBB2_60
; %bb.51:                               ; %.preheader.4
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_159
; %bb.52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_160
.LBB2_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_161
.LBB2_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_162
.LBB2_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_163
.LBB2_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_164
.LBB2_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_165
.LBB2_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_60
.LBB2_59:
	v_mad_co_i64_i32 v[56:57], null, v113, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v111, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v112, v57, s7
	global_store_b32 v[56:57], v63, off offset:28
.LBB2_60:                               ; %Flow1047
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v171
	s_cbranch_execz .LBB2_70
; %bb.61:                               ; %.preheader.5
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_166
; %bb.62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_167
.LBB2_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_168
.LBB2_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_169
.LBB2_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_170
.LBB2_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_171
.LBB2_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_172
.LBB2_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_70
.LBB2_69:
	v_mad_co_i64_i32 v[48:49], null, v113, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v111, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v112, v49, s7
	global_store_b32 v[48:49], v55, off offset:28
.LBB2_70:                               ; %Flow1045
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v170
	s_cbranch_execz .LBB2_80
; %bb.71:                               ; %.preheader.6
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_173
; %bb.72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_174
.LBB2_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_175
.LBB2_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_176
.LBB2_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_177
.LBB2_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_178
.LBB2_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_179
.LBB2_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_80
.LBB2_79:
	v_mad_co_i64_i32 v[40:41], null, v113, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v111, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v112, v41, s7
	global_store_b32 v[40:41], v47, off offset:28
.LBB2_80:                               ; %Flow1043
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v169
	s_cbranch_execz .LBB2_90
; %bb.81:                               ; %.preheader.7
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_180
; %bb.82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_181
.LBB2_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_182
.LBB2_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_183
.LBB2_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_184
.LBB2_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_185
.LBB2_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_186
.LBB2_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_90
.LBB2_89:
	v_mad_co_i64_i32 v[32:33], null, v113, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v111, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v112, v33, s7
	global_store_b32 v[32:33], v39, off offset:28
.LBB2_90:                               ; %Flow1041
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v168
	s_cbranch_execz .LBB2_100
; %bb.91:                               ; %.preheader.8
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_187
; %bb.92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_188
.LBB2_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_189
.LBB2_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_190
.LBB2_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_191
.LBB2_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_192
.LBB2_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_193
.LBB2_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_100
.LBB2_99:
	v_mad_co_i64_i32 v[24:25], null, v113, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v111, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v112, v25, s7
	global_store_b32 v[24:25], v31, off offset:28
.LBB2_100:                              ; %Flow1039
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v167
	s_cbranch_execz .LBB2_110
; %bb.101:                              ; %.preheader.9
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_194
; %bb.102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_195
.LBB2_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_196
.LBB2_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_197
.LBB2_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_198
.LBB2_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_199
.LBB2_107:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_200
.LBB2_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_110
.LBB2_109:
	v_mad_co_i64_i32 v[16:17], null, v113, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v111, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v112, v17, s7
	global_store_b32 v[16:17], v23, off offset:28
.LBB2_110:                              ; %Flow1037
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v166
	s_cbranch_execz .LBB2_120
; %bb.111:                              ; %.preheader.10
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execnz .LBB2_201
; %bb.112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execnz .LBB2_202
.LBB2_113:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execnz .LBB2_203
.LBB2_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execnz .LBB2_204
.LBB2_115:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execnz .LBB2_205
.LBB2_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execnz .LBB2_206
.LBB2_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execnz .LBB2_207
.LBB2_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_120
.LBB2_119:
	v_mad_co_i64_i32 v[8:9], null, v113, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s7, v111, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v112, v9, s7
	global_store_b32 v[8:9], v15, off offset:28
.LBB2_120:                              ; %Flow1035
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s15, v165
	s_cbranch_execz .LBB2_130
; %bb.121:                              ; %.preheader.11
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execnz .LBB2_208
; %bb.122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s7, s0
	s_cbranch_execnz .LBB2_209
.LBB2_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_210
.LBB2_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execnz .LBB2_211
.LBB2_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execnz .LBB2_212
.LBB2_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s2
	s_cbranch_execnz .LBB2_213
.LBB2_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB2_214
.LBB2_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB2_130
.LBB2_129:
	v_mad_co_i64_i32 v[0:1], null, v113, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v111, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v112, v1, vcc_lo
	global_store_b32 v[0:1], v7, off offset:28
.LBB2_130:                              ; %.loopexit.11
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB2_131:
	v_mad_co_i64_i32 v[120:121], null, v99, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[120:121], 2, v[120:121]
	v_add_co_u32 v120, s7, v100, v120
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v121, null, v101, v121, s7
	global_store_b32 v[120:121], v88, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_13
.LBB2_132:
	v_mad_co_i64_i32 v[120:121], null, v96, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[120:121], 2, v[120:121]
	v_add_co_u32 v120, s7, v97, v120
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v121, null, v98, v121, s7
	global_store_b32 v[120:121], v89, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_14
.LBB2_133:
	v_mad_co_i64_i32 v[88:89], null, v103, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v107, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v89, null, v108, v89, s7
	global_store_b32 v[88:89], v90, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_15
.LBB2_134:
	v_mad_co_i64_i32 v[88:89], null, v102, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v109, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v89, null, v110, v89, s7
	global_store_b32 v[88:89], v91, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_16
.LBB2_135:
	v_mad_co_i64_i32 v[88:89], null, v104, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v105, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v89, null, v106, v89, s7
	global_store_b32 v[88:89], v92, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_17
.LBB2_136:
	v_mad_co_i64_i32 v[88:89], null, v114, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v116, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v89, null, v117, v89, s7
	global_store_b32 v[88:89], v93, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_18
.LBB2_137:
	v_mad_co_i64_i32 v[88:89], null, v115, v176, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v118, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v89, null, v119, v89, s7
	global_store_b32 v[88:89], v94, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_19
	s_branch .LBB2_20
.LBB2_138:
	v_mad_co_i64_i32 v[88:89], null, v99, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v100, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v89, null, v101, v89, s7
	global_store_b32 v[88:89], v80, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_23
.LBB2_139:
	v_mad_co_i64_i32 v[88:89], null, v96, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]
	v_add_co_u32 v88, s7, v97, v88
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v89, null, v98, v89, s7
	global_store_b32 v[88:89], v81, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_24
.LBB2_140:
	v_mad_co_i64_i32 v[80:81], null, v103, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v107, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v81, null, v108, v81, s7
	global_store_b32 v[80:81], v82, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_25
.LBB2_141:
	v_mad_co_i64_i32 v[80:81], null, v102, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v109, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v81, null, v110, v81, s7
	global_store_b32 v[80:81], v83, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_26
.LBB2_142:
	v_mad_co_i64_i32 v[80:81], null, v104, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v105, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v81, null, v106, v81, s7
	global_store_b32 v[80:81], v84, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_27
.LBB2_143:
	v_mad_co_i64_i32 v[80:81], null, v114, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v116, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v81, null, v117, v81, s7
	global_store_b32 v[80:81], v85, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_28
.LBB2_144:
	v_mad_co_i64_i32 v[80:81], null, v115, v175, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v118, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v81, null, v119, v81, s7
	global_store_b32 v[80:81], v86, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_29
	s_branch .LBB2_30
.LBB2_145:
	v_mad_co_i64_i32 v[80:81], null, v99, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v100, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v81, null, v101, v81, s7
	global_store_b32 v[80:81], v72, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_33
.LBB2_146:
	v_mad_co_i64_i32 v[80:81], null, v96, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	v_add_co_u32 v80, s7, v97, v80
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v81, null, v98, v81, s7
	global_store_b32 v[80:81], v73, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_34
.LBB2_147:
	v_mad_co_i64_i32 v[72:73], null, v103, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v107, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v108, v73, s7
	global_store_b32 v[72:73], v74, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_35
.LBB2_148:
	v_mad_co_i64_i32 v[72:73], null, v102, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v109, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v110, v73, s7
	global_store_b32 v[72:73], v75, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_36
.LBB2_149:
	v_mad_co_i64_i32 v[72:73], null, v104, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v105, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v106, v73, s7
	global_store_b32 v[72:73], v76, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_37
.LBB2_150:
	v_mad_co_i64_i32 v[72:73], null, v114, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v116, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v117, v73, s7
	global_store_b32 v[72:73], v77, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_38
.LBB2_151:
	v_mad_co_i64_i32 v[72:73], null, v115, v174, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v118, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v73, null, v119, v73, s7
	global_store_b32 v[72:73], v78, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_39
	s_branch .LBB2_40
.LBB2_152:
	v_mad_co_i64_i32 v[72:73], null, v99, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v100, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v101, v73, s7
	global_store_b32 v[72:73], v64, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_43
.LBB2_153:
	v_mad_co_i64_i32 v[72:73], null, v96, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, s7, v97, v72
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v98, v73, s7
	global_store_b32 v[72:73], v65, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_44
.LBB2_154:
	v_mad_co_i64_i32 v[64:65], null, v103, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v107, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v108, v65, s7
	global_store_b32 v[64:65], v66, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_45
.LBB2_155:
	v_mad_co_i64_i32 v[64:65], null, v102, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v109, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v110, v65, s7
	global_store_b32 v[64:65], v67, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_46
.LBB2_156:
	v_mad_co_i64_i32 v[64:65], null, v104, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v105, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v106, v65, s7
	global_store_b32 v[64:65], v68, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_47
.LBB2_157:
	v_mad_co_i64_i32 v[64:65], null, v114, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v116, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v117, v65, s7
	global_store_b32 v[64:65], v69, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_48
.LBB2_158:
	v_mad_co_i64_i32 v[64:65], null, v115, v173, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v118, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v65, null, v119, v65, s7
	global_store_b32 v[64:65], v70, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_49
	s_branch .LBB2_50
.LBB2_159:
	v_mad_co_i64_i32 v[64:65], null, v99, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v100, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v101, v65, s7
	global_store_b32 v[64:65], v56, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_53
.LBB2_160:
	v_mad_co_i64_i32 v[64:65], null, v96, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s7, v97, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v98, v65, s7
	global_store_b32 v[64:65], v57, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_54
.LBB2_161:
	v_mad_co_i64_i32 v[56:57], null, v103, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v107, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v108, v57, s7
	global_store_b32 v[56:57], v58, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_55
.LBB2_162:
	v_mad_co_i64_i32 v[56:57], null, v102, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v109, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v110, v57, s7
	global_store_b32 v[56:57], v59, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_56
.LBB2_163:
	v_mad_co_i64_i32 v[56:57], null, v104, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v105, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v106, v57, s7
	global_store_b32 v[56:57], v60, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_57
.LBB2_164:
	v_mad_co_i64_i32 v[56:57], null, v114, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v116, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v117, v57, s7
	global_store_b32 v[56:57], v61, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_58
.LBB2_165:
	v_mad_co_i64_i32 v[56:57], null, v115, v172, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v118, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v57, null, v119, v57, s7
	global_store_b32 v[56:57], v62, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_59
	s_branch .LBB2_60
.LBB2_166:
	v_mad_co_i64_i32 v[56:57], null, v99, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v100, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v101, v57, s7
	global_store_b32 v[56:57], v48, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_63
.LBB2_167:
	v_mad_co_i64_i32 v[56:57], null, v96, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]
	v_add_co_u32 v56, s7, v97, v56
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v57, null, v98, v57, s7
	global_store_b32 v[56:57], v49, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_64
.LBB2_168:
	v_mad_co_i64_i32 v[48:49], null, v103, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v107, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v108, v49, s7
	global_store_b32 v[48:49], v50, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_65
.LBB2_169:
	v_mad_co_i64_i32 v[48:49], null, v102, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v109, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v110, v49, s7
	global_store_b32 v[48:49], v51, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_66
.LBB2_170:
	v_mad_co_i64_i32 v[48:49], null, v104, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v105, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v106, v49, s7
	global_store_b32 v[48:49], v52, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_67
.LBB2_171:
	v_mad_co_i64_i32 v[48:49], null, v114, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v116, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v117, v49, s7
	global_store_b32 v[48:49], v53, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_68
.LBB2_172:
	v_mad_co_i64_i32 v[48:49], null, v115, v171, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v118, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v49, null, v119, v49, s7
	global_store_b32 v[48:49], v54, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_69
	s_branch .LBB2_70
.LBB2_173:
	v_mad_co_i64_i32 v[48:49], null, v99, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v100, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v101, v49, s7
	global_store_b32 v[48:49], v40, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_73
.LBB2_174:
	v_mad_co_i64_i32 v[48:49], null, v96, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]
	v_add_co_u32 v48, s7, v97, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v49, null, v98, v49, s7
	global_store_b32 v[48:49], v41, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_74
.LBB2_175:
	v_mad_co_i64_i32 v[40:41], null, v103, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v107, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v108, v41, s7
	global_store_b32 v[40:41], v42, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_75
.LBB2_176:
	v_mad_co_i64_i32 v[40:41], null, v102, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v109, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v110, v41, s7
	global_store_b32 v[40:41], v43, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_76
.LBB2_177:
	v_mad_co_i64_i32 v[40:41], null, v104, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v105, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v106, v41, s7
	global_store_b32 v[40:41], v44, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_77
.LBB2_178:
	v_mad_co_i64_i32 v[40:41], null, v114, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v116, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v117, v41, s7
	global_store_b32 v[40:41], v45, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_78
.LBB2_179:
	v_mad_co_i64_i32 v[40:41], null, v115, v170, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v118, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v41, null, v119, v41, s7
	global_store_b32 v[40:41], v46, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_79
	s_branch .LBB2_80
.LBB2_180:
	v_mad_co_i64_i32 v[40:41], null, v99, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v100, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v101, v41, s7
	global_store_b32 v[40:41], v32, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_83
.LBB2_181:
	v_mad_co_i64_i32 v[40:41], null, v96, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, s7, v97, v40
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, v98, v41, s7
	global_store_b32 v[40:41], v33, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_84
.LBB2_182:
	v_mad_co_i64_i32 v[32:33], null, v103, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v107, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v108, v33, s7
	global_store_b32 v[32:33], v34, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_85
.LBB2_183:
	v_mad_co_i64_i32 v[32:33], null, v102, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v109, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v110, v33, s7
	global_store_b32 v[32:33], v35, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_86
.LBB2_184:
	v_mad_co_i64_i32 v[32:33], null, v104, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v105, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v106, v33, s7
	global_store_b32 v[32:33], v36, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_87
.LBB2_185:
	v_mad_co_i64_i32 v[32:33], null, v114, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v116, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v117, v33, s7
	global_store_b32 v[32:33], v37, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_88
.LBB2_186:
	v_mad_co_i64_i32 v[32:33], null, v115, v169, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v118, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v33, null, v119, v33, s7
	global_store_b32 v[32:33], v38, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_89
	s_branch .LBB2_90
.LBB2_187:
	v_mad_co_i64_i32 v[32:33], null, v99, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v100, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v101, v33, s7
	global_store_b32 v[32:33], v24, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_93
.LBB2_188:
	v_mad_co_i64_i32 v[32:33], null, v96, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, s7, v97, v32
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v98, v33, s7
	global_store_b32 v[32:33], v25, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_94
.LBB2_189:
	v_mad_co_i64_i32 v[24:25], null, v103, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v107, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v108, v25, s7
	global_store_b32 v[24:25], v26, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_95
.LBB2_190:
	v_mad_co_i64_i32 v[24:25], null, v102, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v109, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v110, v25, s7
	global_store_b32 v[24:25], v27, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_96
.LBB2_191:
	v_mad_co_i64_i32 v[24:25], null, v104, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v105, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v106, v25, s7
	global_store_b32 v[24:25], v28, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_97
.LBB2_192:
	v_mad_co_i64_i32 v[24:25], null, v114, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v116, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v117, v25, s7
	global_store_b32 v[24:25], v29, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_98
.LBB2_193:
	v_mad_co_i64_i32 v[24:25], null, v115, v168, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v118, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v25, null, v119, v25, s7
	global_store_b32 v[24:25], v30, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_99
	s_branch .LBB2_100
.LBB2_194:
	v_mad_co_i64_i32 v[24:25], null, v99, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v100, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v101, v25, s7
	global_store_b32 v[24:25], v16, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_103
.LBB2_195:
	v_mad_co_i64_i32 v[24:25], null, v96, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_add_co_u32 v24, s7, v97, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, v98, v25, s7
	global_store_b32 v[24:25], v17, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_104
.LBB2_196:
	v_mad_co_i64_i32 v[16:17], null, v103, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v107, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v108, v17, s7
	global_store_b32 v[16:17], v18, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_105
.LBB2_197:
	v_mad_co_i64_i32 v[16:17], null, v102, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v109, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v110, v17, s7
	global_store_b32 v[16:17], v19, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_106
.LBB2_198:
	v_mad_co_i64_i32 v[16:17], null, v104, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v105, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v106, v17, s7
	global_store_b32 v[16:17], v20, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_107
.LBB2_199:
	v_mad_co_i64_i32 v[16:17], null, v114, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v116, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v117, v17, s7
	global_store_b32 v[16:17], v21, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_108
.LBB2_200:
	v_mad_co_i64_i32 v[16:17], null, v115, v167, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v118, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v17, null, v119, v17, s7
	global_store_b32 v[16:17], v22, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_109
	s_branch .LBB2_110
.LBB2_201:
	v_mad_co_i64_i32 v[16:17], null, v99, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v100, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v101, v17, s7
	global_store_b32 v[16:17], v8, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_113
.LBB2_202:
	v_mad_co_i64_i32 v[16:17], null, v96, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, v97, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v98, v17, s7
	global_store_b32 v[16:17], v9, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s1
	s_cbranch_execz .LBB2_114
.LBB2_203:
	v_mad_co_i64_i32 v[8:9], null, v103, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s7, v107, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v108, v9, s7
	global_store_b32 v[8:9], v10, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB2_115
.LBB2_204:
	v_mad_co_i64_i32 v[8:9], null, v102, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s7, v109, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v110, v9, s7
	global_store_b32 v[8:9], v11, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s3
	s_cbranch_execz .LBB2_116
.LBB2_205:
	v_mad_co_i64_i32 v[8:9], null, v104, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s7, v105, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v106, v9, s7
	global_store_b32 v[8:9], v12, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s2
	s_cbranch_execz .LBB2_117
.LBB2_206:
	v_mad_co_i64_i32 v[8:9], null, v114, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s7, v116, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v117, v9, s7
	global_store_b32 v[8:9], v13, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s5
	s_cbranch_execz .LBB2_118
.LBB2_207:
	v_mad_co_i64_i32 v[8:9], null, v115, v166, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s7, v118, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v9, null, v119, v9, s7
	global_store_b32 v[8:9], v14, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_119
	s_branch .LBB2_120
.LBB2_208:
	v_mad_co_i64_i32 v[8:9], null, v99, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, v100, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v101, v9, vcc_lo
	global_store_b32 v[8:9], v0, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB2_123
.LBB2_209:
	v_mad_co_i64_i32 v[8:9], null, v96, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, v97, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v98, v9, vcc_lo
	global_store_b32 v[8:9], v1, off offset:4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_124
.LBB2_210:
	v_mad_co_i64_i32 v[0:1], null, v103, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v107, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v108, v1, vcc_lo
	global_store_b32 v[0:1], v2, off offset:8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB2_125
.LBB2_211:
	v_mad_co_i64_i32 v[0:1], null, v102, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v109, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v110, v1, vcc_lo
	global_store_b32 v[0:1], v3, off offset:12
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB2_126
.LBB2_212:
	v_mad_co_i64_i32 v[0:1], null, v104, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v105, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v106, v1, vcc_lo
	global_store_b32 v[0:1], v4, off offset:16
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB2_127
.LBB2_213:
	v_mad_co_i64_i32 v[0:1], null, v114, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v116, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v117, v1, vcc_lo
	global_store_b32 v[0:1], v5, off offset:20
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execz .LBB2_128
.LBB2_214:
	v_mad_co_i64_i32 v[0:1], null, v115, v165, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v118, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v1, null, v119, v1, vcc_lo
	global_store_b32 v[0:1], v6, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB2_129
	s_branch .LBB2_130
.Lfunc_end2:
	.size	gemm_gate_up_hfq4g256_wmma_gfx12_bt12, .Lfunc_end2-gemm_gate_up_hfq4g256_wmma_gfx12_bt12
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_gate_up_hfq4g256_wmma_gfx12_bt12
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
		.amdhsa_next_free_vgpr 247
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
		.amdhsa_inst_pref_size 112
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
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.num_vgpr, 247
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.num_agpr, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.numbered_sgpr, 20
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.num_named_barrier, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.private_seg_size, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.uses_vcc, 1
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.uses_flat_scratch, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.has_dyn_sized_stack, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.has_recursion, 0
	.set .Lgemm_gate_up_hfq4g256_wmma_gfx12_bt12.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14284
; TotalNumSgprs: 22
; NumVgprs: 247
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 247
; Occupancy: 5
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
	.type	__hip_cuid_5acb86ef4699f94f,@object ; @__hip_cuid_5acb86ef4699f94f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5acb86ef4699f94f
__hip_cuid_5acb86ef4699f94f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5acb86ef4699f94f, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5acb86ef4699f94f
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
    .name:           gemm_gate_up_hfq4g256_wmma_gfx12_bt4
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         gemm_gate_up_hfq4g256_wmma_gfx12_bt4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     76
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           gemm_gate_up_hfq4g256_wmma_gfx12_bt8
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         gemm_gate_up_hfq4g256_wmma_gfx12_bt8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     138
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           gemm_gate_up_hfq4g256_wmma_gfx12_bt12
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         gemm_gate_up_hfq4g256_wmma_gfx12_bt12.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     247
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
