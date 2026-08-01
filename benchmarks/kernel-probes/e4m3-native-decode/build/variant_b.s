	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gemv_mfp4g32_e8         ; -- Begin function gemv_mfp4g32_e8
	.globl	gemv_mfp4g32_e8
	.p2align	8
	.type	gemv_mfp4g32_e8,@function
gemv_mfp4g32_e8:                        ; @gemv_mfp4g32_e8
; %bb.0:
	s_load_b64 s[10:11], s[0:1], 0x18
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 ttmp9, s10
	s_cbranch_scc1 .LBB0_12
; %bb.1:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_ashr_i32 s10, s11, 31
	s_mov_b32 s2, ttmp9
	s_lshr_b32 s3, s10, 27
	v_dual_mov_b32 v37, 0 :: v_dual_lshlrev_b32 v42, 3, v0
	s_add_co_i32 s3, s11, s3
	v_lshrrev_b32_e32 v2, 2, v0
	s_ashr_i32 s3, s3, 5
	v_dual_mov_b32 v38, 0 :: v_dual_lshlrev_b32 v3, 2, v0
	s_mul_i32 s8, s3, 17
	s_ashr_i32 s3, ttmp9, 31
	s_ashr_i32 s9, s8, 31
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v44, 12, v3
	s_add_nc_u64 s[8:9], s[8:9], 16
	v_mul_u32_u24_e32 v43, 17, v2
	s_mul_u64 s[12:13], s[8:9], s[2:3]
	s_load_b64 s[8:9], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[4:5], s[12:13]
	s_lshr_b32 s0, s10, 24
	global_load_d16_b16 v1, v37, s[4:5]
	s_add_co_i32 s1, s11, s0
	v_mov_b32_e32 v40, 0
	s_ashr_i32 s10, s1, 10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s10, 1
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v41, v1.l
	s_cbranch_scc1 .LBB0_4
; %bb.2:
	v_add_co_u32 v1, s0, s4, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v2, null, s5, 0, s0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v38, 0
	v_add_co_u32 v33, vcc_lo, 0x120, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v34, null, 0, v2, vcc_lo
	v_dual_mov_b32 v35, v42 :: v_dual_mov_b32 v40, 0
	v_mov_b32_e32 v39, 0
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v1, vcc_lo, v33, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v34, vcc_lo
	s_clause 0x3
	global_load_u8 v48, v[33:34], off offset:-272
	global_load_u8 v49, v[33:34], off offset:-136
	global_load_u8 v50, v[33:34], off
	global_load_u8 v51, v[33:34], off offset:136
	v_ashrrev_i32_e32 v36, 31, v35
	v_mov_b16_e32 v53.h, 0
	s_clause 0x3
	global_load_b32 v45, v[1:2], off offset:-271
	global_load_b32 v46, v[1:2], off offset:-135
	global_load_b32 v47, v[1:2], off offset:1
	global_load_b32 v52, v[1:2], off offset:137
	s_add_co_i32 s10, s10, -1
	v_lshlrev_b64_e32 v[1:2], 2, v[35:36]
	v_add_nc_u32_e32 v35, 0x400, v35
	s_cmp_eq_u32 s10, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v13, vcc_lo, s6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s7, v2, vcc_lo
	s_clause 0x7
	global_load_b128 v[17:20], v[13:14], off
	global_load_b128 v[21:24], v[13:14], off offset:1024
	global_load_b128 v[25:28], v[13:14], off offset:2048
	global_load_b128 v[1:4], v[13:14], off offset:16
	global_load_b128 v[5:8], v[13:14], off offset:1040
	global_load_b128 v[9:12], v[13:14], off offset:2064
	global_load_b128 v[29:32], v[13:14], off offset:3072
	global_load_b128 v[13:16], v[13:14], off offset:3088
	v_add_co_u32 v33, vcc_lo, 0x220, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, 0, v34, vcc_lo
	s_wait_loadcnt 0xf
	v_cvt_f32_fp8_e32 v36, v48
	s_wait_loadcnt 0xe
	v_cvt_f32_fp8_e32 v48, v49
	s_wait_loadcnt 0xd
	v_cvt_f32_fp8_e32 v49, v50
	s_wait_loadcnt 0xb
	v_mov_b16_e32 v53.l, v45.h
	v_lshrrev_b32_e32 v56, 12, v45
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v36, v36, v41 :: v_dual_and_b32 v85, 15, v47
	v_and_b32_e32 v59, 15, v45
	v_cvt_f32_fp8_e32 v50, v51
	v_dual_mul_f32 v49, v49, v41 :: v_dual_and_b32 v72, 15, v46
	v_lshrrev_b32_e32 v54, 24, v45
	v_lshrrev_b32_e32 v55, 20, v45
	v_lshrrev_b32_e32 v69, 12, v46
	v_mul_f32_e32 v36, 0x3f6147ae, v36
	v_add3_u32 v56, v56, v59, v53
	v_mov_b16_e32 v53.l, v46.h
	v_lshrrev_b32_e32 v57, 8, v45
	v_lshrrev_b32_e32 v58, 4, v45
	v_bfe_u32 v61, v45, 20, 4
	v_bfe_u32 v63, v45, 12, 4
	v_mul_f32_e32 v50, v50, v41
	v_lshrrev_b32_e32 v82, 12, v47
	v_add3_u32 v54, v56, v55, v54
	v_add3_u32 v69, v69, v72, v53
	v_mov_b16_e32 v53.l, v47.h
	v_mul_f32_e32 v48, v48, v41
	v_bfe_u32 v60, v45, 24, 4
	s_wait_loadcnt 0x8
	v_lshrrev_b32_e32 v56, 12, v52
	v_add3_u32 v54, v54, v58, v57
	v_and_b32_e32 v57, 15, v52
	v_add3_u32 v82, v82, v85, v53
	v_mov_b16_e32 v53.l, v52.h
	v_add_nc_u32_e32 v61, -7, v61
	v_dual_mul_f32 v50, 0x3f6147ae, v50 :: v_dual_add_nc_u32 v63, -7, v63
	v_bfe_u32 v62, v45, 16, 4
	v_lshrrev_b32_e32 v93, 24, v52
	v_lshrrev_b32_e32 v55, 20, v52
	v_add3_u32 v53, v56, v57, v53
	v_cvt_f32_i32_e32 v61, v61
	v_add_nc_u32_e32 v60, -7, v60
	v_bfe_u32 v64, v45, 8, 4
	v_bfe_u32 v65, v45, 4, 4
	v_lshrrev_b32_e32 v80, 24, v47
	v_lshrrev_b32_e32 v81, 20, v47
	v_add3_u32 v53, v53, v55, v93
	v_add_f32_e32 v93, 0.5, v61
	v_cvt_f32_i32_e32 v60, v60
	v_add_nc_u32_e32 v62, -7, v62
	v_lshrrev_b32_e32 v67, 24, v46
	v_lshrrev_b32_e32 v68, 20, v46
	v_lshrrev_b32_e32 v83, 8, v47
	v_lshrrev_b32_e32 v84, 4, v47
	v_add3_u32 v80, v82, v81, v80
	v_cvt_f32_i32_e32 v63, v63
	v_add_nc_u32_e32 v65, -7, v65
	v_add_f32_e32 v55, 0.5, v60
	v_cvt_f32_i32_e32 v62, v62
	v_add_nc_u32_e32 v64, -7, v64
	v_lshrrev_b32_e32 v51, 27, v45
	v_bfe_u32 v74, v46, 20, 4
	v_add3_u32 v67, v69, v68, v67
	v_lshrrev_b32_e32 v68, 8, v52
	v_lshrrev_b32_e32 v69, 4, v52
	v_add3_u32 v80, v80, v84, v83
	v_add_f32_e32 v83, 0.5, v62
	v_cvt_f32_i32_e32 v64, v64
	v_add_nc_u32_e32 v59, -7, v59
	v_add_f32_e32 v84, 0.5, v63
	v_cvt_f32_i32_e32 v65, v65
	v_and_b32_e32 v54, 1, v54
	v_lshrrev_b32_e32 v70, 8, v46
	v_lshrrev_b32_e32 v71, 4, v46
	v_bfe_u32 v73, v46, 24, 4
	v_add_nc_u32_e32 v74, -7, v74
	v_add3_u32 v53, v53, v69, v68
	v_add_f32_e32 v68, 0.5, v64
	v_cvt_f32_i32_e32 v59, v59
	v_add_f32_e32 v69, 0.5, v65
	v_and_or_b32 v51, v51, 14, v54
	v_cmp_gt_i32_e64 s0, 0, v45
	v_bfe_u32 v77, v46, 8, 4
	v_bfe_u32 v78, v46, 4, 4
	v_add3_u32 v67, v67, v71, v70
	v_dual_add_f32 v54, 0.5, v59 :: v_dual_add_nc_u32 v73, -7, v73
	v_cvt_f32_i32_e32 v74, v74
	v_add_nc_u32_e32 v72, -7, v72
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v65, v65, v69, s0
	v_add_nc_u32_e32 v51, -7, v51
	v_lshrrev_b32_e32 v66, 27, v46
	v_lshrrev_b32_e32 v79, 27, v47
	v_bfe_u32 v88, v47, 16, 4
	v_bfe_u32 v81, v52, 12, 4
	v_and_b32_e32 v80, 1, v80
	v_cvt_f32_i32_e32 v51, v51
	v_add_nc_u32_e32 v77, -7, v77
	v_add_f32_e32 v45, 0.5, v74
	v_cvt_f32_i32_e32 v72, v72
	v_add_nc_u32_e32 v78, -7, v78
	v_cndmask_b32_e64 v54, v59, v54, s0
	v_and_b32_e32 v67, 1, v67
	v_bfe_u32 v76, v46, 12, 4
	v_bfe_u32 v86, v47, 24, 4
	v_bfe_u32 v87, v47, 20, 4
	v_bfe_u32 v58, v52, 24, 4
	v_and_or_b32 v79, v79, 14, v80
	v_add_f32_e32 v80, 0.5, v51
	v_cvt_f32_i32_e32 v77, v77
	v_cndmask_b32_e64 v62, v62, v83, s0
	v_add_nc_u32_e32 v81, -7, v81
	v_add_f32_e32 v83, 0.5, v72
	v_cvt_f32_i32_e32 v78, v78
	v_add_nc_u32_e32 v88, -7, v88
	v_mul_f32_e32 v54, v36, v54
	v_and_or_b32 v66, v66, 14, v67
	v_bfe_u32 v75, v46, 16, 4
	v_bfe_u32 v91, v47, 4, 4
	v_add_nc_u32_e32 v87, -7, v87
	v_cvt_f32_i32_e32 v73, v73
	v_add_nc_u32_e32 v76, -7, v76
	v_cndmask_b32_e64 v51, v51, v80, s0
	v_cndmask_b32_e64 v55, v60, v55, s0
	v_add_nc_u32_e32 v58, -7, v58
	v_add_f32_e32 v60, 0.5, v77
	v_cndmask_b32_e64 v61, v61, v93, s0
	v_add_f32_e32 v93, 0.5, v78
	v_cvt_f32_i32_e32 v88, v88
	v_add_nc_u32_e32 v86, -7, v86
	v_cndmask_b32_e64 v63, v63, v84, s0
	v_cndmask_b32_e64 v64, v64, v68, s0
	v_cmp_gt_i32_e64 s0, 0, v46
	v_mul_f32_e32 v62, v36, v62
	v_cvt_f32_i32_e32 v81, v81
	v_add_nc_u32_e32 v66, -7, v66
	v_bfe_u32 v70, v52, 20, 4
	v_bfe_u32 v56, v52, 4, 4
	v_add_f32_e32 v67, 0.5, v73
	v_cvt_f32_i32_e32 v76, v76
	v_add_nc_u32_e32 v75, -7, v75
	v_add_f32_e32 v69, 0.5, v88
	v_cvt_f32_i32_e32 v86, v86
	v_add_nc_u32_e32 v91, -7, v91
	v_mul_f32_e32 v55, v36, v55
	v_cvt_f32_i32_e32 v58, v58
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v60, v77, v60, s0
	v_add_f32_e32 v77, 0.5, v81
	v_cvt_f32_i32_e32 v66, v66
	v_bfe_u32 v89, v47, 12, 4
	v_bfe_u32 v90, v47, 8, 4
	v_lshrrev_b32_e32 v92, 27, v52
	v_dual_mul_f32 v61, v36, v61 :: v_dual_add_nc_u32 v70, -7, v70
	v_dual_add_f32 v59, 0.5, v66 :: v_dual_add_nc_u32 v56, -7, v56
	v_cvt_f32_i32_e32 v87, v87
	v_dual_add_f32 v80, 0.5, v76 :: v_dual_add_nc_u32 v85, -7, v85
	v_cvt_f32_i32_e32 v75, v75
	v_add_f32_e32 v84, 0.5, v86
	v_cvt_f32_i32_e32 v91, v91
	v_and_b32_e32 v53, 1, v53
	v_cndmask_b32_e64 v45, v74, v45, s0
	v_dual_add_f32 v74, 0.5, v58 :: v_dual_add_nc_u32 v79, -7, v79
	v_bfe_u32 v82, v52, 8, 4
	v_dual_mul_f32 v48, 0x3f6147ae, v48 :: v_dual_mul_f32 v49, 0x3f6147ae, v49
	v_add_nc_u32_e32 v89, -7, v89
	v_dual_mul_f32 v51, v36, v51 :: v_dual_add_nc_u32 v90, -7, v90
	v_cvt_f32_i32_e32 v70, v70
	v_add_f32_e32 v68, 0.5, v87
	v_cvt_f32_i32_e32 v85, v85
	v_cndmask_b32_e64 v59, v66, v59, s0
	v_add_f32_e32 v66, 0.5, v91
	v_and_or_b32 v53, v92, 14, v53
	v_add_f32_e32 v92, 0.5, v75
	v_cndmask_b32_e64 v76, v76, v80, s0
	v_cvt_f32_i32_e32 v79, v79
	v_bfe_u32 v71, v52, 16, 4
	v_dual_mul_f32 v63, v36, v63 :: v_dual_add_nc_u32 v82, -7, v82
	v_dual_mul_f32 v64, v36, v64 :: v_dual_add_nc_u32 v57, -7, v57
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v56, v56
	v_mul_f32_e32 v36, v36, v65
	v_cndmask_b32_e64 v67, v73, v67, s0
	v_add_f32_e32 v73, 0.5, v85
	v_cndmask_b32_e64 v75, v75, v92, s0
	v_add_f32_e32 v92, 0.5, v70
	v_cndmask_b32_e64 v78, v78, v93, s0
	v_cndmask_b32_e64 v72, v72, v83, s0
	v_add_f32_e32 v83, 0.5, v79
	v_cmp_gt_i32_e64 s0, 0, v47
	v_dual_mul_f32 v76, v48, v76 :: v_dual_add_nc_u32 v53, -7, v53
	v_dual_add_f32 v46, 0.5, v89 :: v_dual_add_nc_u32 v71, -7, v71
	v_cvt_f32_i32_e32 v57, v57
	v_add_f32_e32 v65, 0.5, v90
	v_add_f32_e32 v47, 0.5, v56
	v_mul_f32_e32 v59, v48, v59
	v_cvt_f32_i32_e32 v53, v53
	v_mul_f32_e32 v60, v48, v60
	s_wait_loadcnt 0x7
	v_mul_f32_e32 v18, v36, v18
	v_mul_f32_e32 v67, v48, v67
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, v79, v83, s0
	v_mul_f32_e32 v45, v48, v45
	v_cndmask_b32_e64 v68, v87, v68, s0
	v_mul_f32_e32 v75, v48, v75
	v_cndmask_b32_e64 v73, v85, v73, s0
	v_mul_f32_e32 v72, v48, v72
	v_mul_f32_e32 v48, v48, v78
	v_cndmask_b32_e64 v79, v86, v84, s0
	v_cndmask_b32_e64 v46, v89, v46, s0
	v_cndmask_b32_e64 v65, v90, v65, s0
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v18, v54, v17 :: v_dual_mul_f32 v17, v48, v22
	v_cndmask_b32_e64 v69, v88, v69, s0
	v_cndmask_b32_e64 v66, v91, v66, s0
	v_cmp_gt_i32_e64 s0, 0, v52
	v_mul_f32_e32 v22, v49, v36
	v_mul_f32_e32 v36, v49, v79
	v_mul_f32_e32 v54, v49, v65
	v_mul_f32_e32 v65, v49, v73
	v_dual_mul_f32 v46, v49, v46 :: v_dual_fmac_f32 v17, v72, v21
	v_dual_add_f32 v83, 0.5, v53 :: v_dual_mul_f32 v48, v49, v68
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v58, v58, v74, s0
	v_mul_f32_e32 v52, v49, v69
	v_dual_mul_f32 v49, v49, v66 :: v_dual_fmac_f32 v18, v64, v19
	v_fmac_f32_e32 v17, v60, v23
	v_cvt_f32_i32_e32 v71, v71
	v_cvt_f32_i32_e32 v82, v82
	v_add_f32_e32 v78, 0.5, v57
	v_cndmask_b32_e64 v47, v56, v47, s0
	v_fmac_f32_e32 v18, v63, v20
	v_dual_fmac_f32 v17, v76, v24 :: v_dual_add_f32 v80, 0.5, v71
	v_add_f32_e32 v93, 0.5, v82
	v_cndmask_b32_e64 v53, v53, v83, s0
	v_cndmask_b32_e64 v56, v57, v78, s0
	v_mul_f32_e32 v21, v50, v47
	s_wait_loadcnt 0x4
	v_fmac_f32_e32 v18, v62, v1
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v17, v75, v5
	v_cndmask_b32_e64 v66, v70, v92, s0
	v_cndmask_b32_e64 v68, v71, v80, s0
	v_cndmask_b32_e64 v70, v82, v93, s0
	v_dual_mul_f32 v19, v49, v26 :: v_dual_mul_f32 v26, v50, v53
	v_dual_mul_f32 v53, v50, v56 :: v_dual_fmac_f32 v18, v61, v2
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v21, v21, v30 :: v_dual_mul_f32 v30, v50, v68
	v_fmac_f32_e32 v17, v45, v6
	v_cndmask_b32_e64 v69, v81, v77, s0
	v_dual_mul_f32 v56, v50, v70 :: v_dual_fmac_f32 v19, v65, v25
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v21, v53, v29 :: v_dual_fmac_f32 v18, v55, v3
	v_fmac_f32_e32 v17, v67, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v25, v50, v69
	v_mul_f32_e32 v49, v50, v66
	v_dual_mul_f32 v47, v50, v58 :: v_dual_fmac_f32 v18, v51, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v59, v8
	v_fmac_f32_e32 v21, v56, v31
	v_dual_add_f32 v37, v37, v18 :: v_dual_add_f32 v38, v38, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v25, v32
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v21, v30, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v49, v14
	v_fmac_f32_e32 v21, v47, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v26, v16
	v_dual_fmac_f32 v19, v54, v27 :: v_dual_add_f32 v40, v40, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v46, v28
	v_fmac_f32_e32 v19, v52, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v48, v10
	v_fmac_f32_e32 v19, v36, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v22, v12
	v_add_f32_e32 v39, v39, v19
	s_cbranch_scc0 .LBB0_3
.LBB0_4:
	s_ashr_i32 s0, s1, 8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s0, 3
	s_cbranch_scc0 .LBB0_6
; %bb.5:
	s_and_b32 s12, s0, -4
	v_mov_b16_e32 v11.h, 0
	s_mul_i32 s10, s12, 0x88
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s11, s10, 31
	s_add_nc_u64 s[10:11], s[4:5], s[10:11]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s13, s10, v43
	v_add_co_ci_u32_e64 v2, null, s11, 0, s13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x1
	global_load_b32 v9, v[1:2], off offset:17
	global_load_u8 v10, v43, s[10:11] offset:16
	v_lshl_or_b32 v1, s12, 8, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[5:8], v[5:6], off offset:16
	s_wait_loadcnt 0x3
	v_mov_b16_e32 v11.l, v9.h
	s_wait_loadcnt 0x2
	v_cvt_f32_fp8_e32 v10, v10
	v_lshrrev_b32_e32 v12, 12, v9
	v_lshrrev_b32_e32 v14, 24, v9
	v_lshrrev_b32_e32 v15, 20, v9
	v_lshrrev_b32_e32 v16, 4, v9
	v_mul_f32_e32 v10, v10, v41
	v_and_b32_e32 v13, 15, v9
	v_bfe_u32 v18, v9, 8, 4
	v_bfe_u32 v17, v9, 12, 4
	v_cmp_gt_i32_e32 vcc_lo, 0, v9
	v_mul_f32_e32 v10, 0x3f6147ae, v10
	v_add3_u32 v11, v12, v13, v11
	v_lshrrev_b32_e32 v12, 8, v9
	v_add_nc_u32_e32 v13, -7, v13
	v_add_nc_u32_e32 v18, -7, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add3_u32 v11, v11, v15, v14
	v_bfe_u32 v15, v9, 4, 4
	v_cvt_f32_i32_e32 v13, v13
	v_bfe_u32 v14, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add3_u32 v11, v11, v16, v12
	v_lshrrev_b32_e32 v12, 27, v9
	v_add_nc_u32_e32 v15, -7, v15
	v_bfe_u32 v16, v9, 16, 4
	v_and_b32_e32 v11, 1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_and_or_b32 v11, v12, 14, v11
	v_bfe_u32 v12, v9, 20, 4
	v_add_nc_u32_e32 v9, -7, v17
	v_cvt_f32_i32_e32 v17, v18
	v_add_f32_e32 v18, 0.5, v13
	v_cvt_f32_i32_e32 v15, v15
	v_add_nc_u32_e32 v14, -7, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v19, 0.5, v15 :: v_dual_add_nc_u32 v16, -7, v16
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v15, v15, v19 :: v_dual_add_nc_u32 v12, -7, v12
	v_add_f32_e32 v19, 0.5, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v10, v15
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v2, v15, v2
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v15, 0.5, v16
	v_cndmask_b32_e32 v13, v13, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v18, 0.5, v9 :: v_dual_add_nc_u32 v11, -7, v11
	v_mul_f32_e32 v13, v10, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_i32_e32 v11, v11
	v_cndmask_b32_e32 v9, v9, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v2, v13, v1 :: v_dual_add_f32 v13, 0.5, v12
	v_cndmask_b32_e32 v17, v17, v19, vcc_lo
	v_mul_f32_e32 v9, v10, v9
	v_cvt_f32_i32_e32 v1, v14
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo
	v_dual_mul_f32 v17, v10, v17 :: v_dual_add_f32 v20, 0.5, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v13, v10, v14 :: v_dual_fmac_f32 v2, v17, v3
	v_add_f32_e32 v3, 0.5, v1
	v_cndmask_b32_e32 v11, v11, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v2, v9, v4
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v3, v10, v12 :: v_dual_fmac_f32 v2, v13, v5
	v_dual_mul_f32 v1, v10, v1 :: v_dual_fmac_f32 v2, v3, v6
	v_mul_f32_e32 v3, v10, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v1, v7
	v_fmac_f32_e32 v2, v3, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v2
.LBB0_6:
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s1, 2
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	s_and_b32 s10, s0, -4
	v_mov_b16_e32 v11.h, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s12, s10, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s10, s12, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s11, s10, 31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[4:5], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s13, s10, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s11, 0, s13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x1
	global_load_b32 v9, v[1:2], off offset:17
	global_load_u8 v10, v43, s[10:11] offset:16
	v_lshl_or_b32 v1, s12, 8, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[5:8], v[5:6], off offset:16
	s_wait_loadcnt 0x3
	v_mov_b16_e32 v11.l, v9.h
	s_wait_loadcnt 0x2
	v_cvt_f32_fp8_e32 v10, v10
	v_lshrrev_b32_e32 v12, 12, v9
	v_lshrrev_b32_e32 v14, 24, v9
	v_lshrrev_b32_e32 v15, 20, v9
	v_lshrrev_b32_e32 v16, 4, v9
	v_mul_f32_e32 v10, v10, v41
	v_and_b32_e32 v13, 15, v9
	v_bfe_u32 v18, v9, 8, 4
	v_bfe_u32 v17, v9, 12, 4
	v_cmp_gt_i32_e32 vcc_lo, 0, v9
	v_mul_f32_e32 v10, 0x3f6147ae, v10
	v_add3_u32 v11, v12, v13, v11
	v_lshrrev_b32_e32 v12, 8, v9
	v_add_nc_u32_e32 v13, -7, v13
	v_add_nc_u32_e32 v18, -7, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add3_u32 v11, v11, v15, v14
	v_bfe_u32 v15, v9, 4, 4
	v_cvt_f32_i32_e32 v13, v13
	v_bfe_u32 v14, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add3_u32 v11, v11, v16, v12
	v_lshrrev_b32_e32 v12, 27, v9
	v_add_nc_u32_e32 v15, -7, v15
	v_bfe_u32 v16, v9, 16, 4
	v_and_b32_e32 v11, 1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_and_or_b32 v11, v12, 14, v11
	v_bfe_u32 v12, v9, 20, 4
	v_add_nc_u32_e32 v9, -7, v17
	v_cvt_f32_i32_e32 v17, v18
	v_add_f32_e32 v18, 0.5, v13
	v_cvt_f32_i32_e32 v15, v15
	v_add_nc_u32_e32 v14, -7, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v19, 0.5, v15 :: v_dual_add_nc_u32 v16, -7, v16
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v15, v15, v19 :: v_dual_add_nc_u32 v12, -7, v12
	v_add_f32_e32 v19, 0.5, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v10, v15
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v2, v15, v2
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v15, 0.5, v16
	v_cndmask_b32_e32 v13, v13, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v18, 0.5, v9 :: v_dual_add_nc_u32 v11, -7, v11
	v_mul_f32_e32 v13, v10, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_i32_e32 v11, v11
	v_cndmask_b32_e32 v9, v9, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v2, v13, v1 :: v_dual_add_f32 v13, 0.5, v12
	v_cndmask_b32_e32 v17, v17, v19, vcc_lo
	v_mul_f32_e32 v9, v10, v9
	v_cvt_f32_i32_e32 v1, v14
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo
	v_dual_mul_f32 v17, v10, v17 :: v_dual_add_f32 v20, 0.5, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v13, v10, v14 :: v_dual_fmac_f32 v2, v17, v3
	v_add_f32_e32 v3, 0.5, v1
	v_cndmask_b32_e32 v11, v11, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v2, v9, v4
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v3, v10, v12 :: v_dual_fmac_f32 v2, v13, v5
	v_dual_mul_f32 v1, v10, v1 :: v_dual_fmac_f32 v2, v3, v6
	v_mul_f32_e32 v3, v10, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v1, v7
	v_fmac_f32_e32 v2, v3, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v2
.LBB0_8:
	s_cmp_lg_u32 s1, 3
	s_cbranch_scc1 .LBB0_10
; %bb.9:
	s_and_b32 s10, s0, -2
	v_mov_b16_e32 v11.h, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s0, s10, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[4:5], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s4, s0, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s1, 0, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x1
	global_load_b32 v9, v[1:2], off offset:17
	global_load_u8 v10, v43, s[0:1] offset:16
	v_lshl_or_b32 v1, s10, 8, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[5:8], v[5:6], off offset:16
	s_wait_loadcnt 0x3
	v_mov_b16_e32 v11.l, v9.h
	s_wait_loadcnt 0x2
	v_cvt_f32_fp8_e32 v10, v10
	v_lshrrev_b32_e32 v12, 12, v9
	v_lshrrev_b32_e32 v14, 24, v9
	v_lshrrev_b32_e32 v15, 20, v9
	v_lshrrev_b32_e32 v16, 4, v9
	v_mul_f32_e32 v10, v10, v41
	v_and_b32_e32 v13, 15, v9
	v_bfe_u32 v18, v9, 8, 4
	v_bfe_u32 v17, v9, 12, 4
	v_cmp_gt_i32_e32 vcc_lo, 0, v9
	v_mul_f32_e32 v10, 0x3f6147ae, v10
	v_add3_u32 v11, v12, v13, v11
	v_lshrrev_b32_e32 v12, 8, v9
	v_add_nc_u32_e32 v13, -7, v13
	v_add_nc_u32_e32 v18, -7, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add3_u32 v11, v11, v15, v14
	v_bfe_u32 v15, v9, 4, 4
	v_cvt_f32_i32_e32 v13, v13
	v_bfe_u32 v14, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add3_u32 v11, v11, v16, v12
	v_lshrrev_b32_e32 v12, 27, v9
	v_add_nc_u32_e32 v15, -7, v15
	v_bfe_u32 v16, v9, 16, 4
	v_and_b32_e32 v11, 1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_and_or_b32 v11, v12, 14, v11
	v_bfe_u32 v12, v9, 20, 4
	v_add_nc_u32_e32 v9, -7, v17
	v_cvt_f32_i32_e32 v17, v18
	v_add_f32_e32 v18, 0.5, v13
	v_cvt_f32_i32_e32 v15, v15
	v_add_nc_u32_e32 v14, -7, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v19, 0.5, v15 :: v_dual_add_nc_u32 v16, -7, v16
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v15, v15, v19 :: v_dual_add_nc_u32 v12, -7, v12
	v_add_f32_e32 v19, 0.5, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v10, v15
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v2, v15, v2
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v15, 0.5, v16
	v_cndmask_b32_e32 v13, v13, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v18, 0.5, v9 :: v_dual_add_nc_u32 v11, -7, v11
	v_mul_f32_e32 v13, v10, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_i32_e32 v11, v11
	v_cndmask_b32_e32 v9, v9, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v2, v13, v1 :: v_dual_add_f32 v13, 0.5, v12
	v_cndmask_b32_e32 v17, v17, v19, vcc_lo
	v_mul_f32_e32 v9, v10, v9
	v_cvt_f32_i32_e32 v1, v14
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo
	v_dual_mul_f32 v17, v10, v17 :: v_dual_add_f32 v20, 0.5, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v13, v10, v14 :: v_dual_fmac_f32 v2, v17, v3
	v_add_f32_e32 v3, 0.5, v1
	v_cndmask_b32_e32 v11, v11, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v2, v9, v4
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v3, v10, v12 :: v_dual_fmac_f32 v2, v13, v5
	v_dual_mul_f32 v1, v10, v1 :: v_dual_fmac_f32 v2, v3, v6
	v_mul_f32_e32 v3, v10, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v1, v7
	v_fmac_f32_e32 v2, v3, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v2
.LBB0_10:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_add_f32 v1, v37, v38 :: v_dual_add_f32 v2, v40, v39
	v_mbcnt_lo_u32_b32 v3, -1, 0
	s_mov_b32 s0, exec_lo
	v_add_f32_e32 v1, v1, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v2, v3, 2, 64
	v_cmp_gt_u32_e32 vcc_lo, 24, v3
	ds_bpermute_b32 v2, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 28, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_add_lshl_u32 v4, v4, v3, 2
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 30, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_add_lshl_u32 v4, v4, v3, 2
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_add_lshl_u32 v4, v4, v3, 2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v4, v1
	s_wait_dscnt 0x0
	v_dual_add_f32 v1, v1, v2 :: v_dual_lshlrev_b32 v2, 2, v3
	ds_bpermute_b32 v2, v2, v1
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB0_12
; %bb.11:
	s_wait_dscnt 0x0
	v_dual_add_f32 v0, v1, v2 :: v_dual_mov_b32 v1, 0
	s_lshl_b64 s[0:1], s[2:3], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]
	global_store_b32 v1, v0, s[0:1]
.LBB0_12:
	s_endpgm
.Lfunc_end0:
	.size	gemv_mfp4g32_e8, .Lfunc_end0-gemv_mfp4g32_e8
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemv_mfp4g32_e8
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 94
		.amdhsa_next_free_sgpr 14
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 33
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
	.set .Lgemv_mfp4g32_e8.num_vgpr, 94
	.set .Lgemv_mfp4g32_e8.num_agpr, 0
	.set .Lgemv_mfp4g32_e8.numbered_sgpr, 14
	.set .Lgemv_mfp4g32_e8.num_named_barrier, 0
	.set .Lgemv_mfp4g32_e8.private_seg_size, 0
	.set .Lgemv_mfp4g32_e8.uses_vcc, 1
	.set .Lgemv_mfp4g32_e8.uses_flat_scratch, 0
	.set .Lgemv_mfp4g32_e8.has_dyn_sized_stack, 0
	.set .Lgemv_mfp4g32_e8.has_recursion, 0
	.set .Lgemv_mfp4g32_e8.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4180
; TotalNumSgprs: 16
; NumVgprs: 94
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 94
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
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
	.type	__hip_cuid_d122204b9ad7516,@object ; @__hip_cuid_d122204b9ad7516
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d122204b9ad7516
__hip_cuid_d122204b9ad7516:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d122204b9ad7516, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_d122204b9ad7516
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           gemv_mfp4g32_e8
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         gemv_mfp4g32_e8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     94
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
