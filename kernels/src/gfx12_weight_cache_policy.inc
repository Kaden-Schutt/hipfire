// Architecture-neutral HFQ4 weight-buffer addressing, with per-ISA cache
// policy selection.
//
// HIPFIRE_WEIGHT_CPOL_AUX is the raw-buffer cache-policy immediate. GFX12 uses
// the TH/SCOPE encoding:
//   16 = TH_RT    + SCOPE_DEV (buffer-addressing control)
//   18 = TH_HT    + SCOPE_DEV
//   20 = TH_NT_RT + SCOPE_DEV
//   22 = TH_NT_HT + SCOPE_DEV
// Earlier RDNA generations use cache-policy zero for ordinary temporal reuse.
//
// This preamble is concatenated ahead of selected HFQ4 kernel sources by
// kernels.rs. gfx1201 defaults to buffer RT after matched model-level gates;
// gfx1100 now defaults to temporal buffer addressing after its own exact
// model-level A/B. Other targets remain byte-identical unless Radiowave
// explicitly selects HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN. Flat gfx1100 GEMV
// stays global unless HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN is set.
// HIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS keeps the historical gfx12 control.
#if defined(HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE) && \
    (defined(__gfx1201__) || defined(__gfx1100__) || \
     defined(HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN)) && \
    (!defined(HIPFIRE_WEIGHT_CACHE_FLAT_GEMV) || \
     defined(HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN)) && \
    !defined(HIPFIRE_GFX11_WEIGHT_GLOBAL_LOADS) && \
    !defined(HIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS)
#ifndef HIPFIRE_WEIGHT_CPOL_AUX
#if defined(__gfx1201__)
#define HIPFIRE_WEIGHT_CPOL_AUX 16
#else
#define HIPFIRE_WEIGHT_CPOL_AUX 0
#endif
#endif
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 1
#define HIPFIRE_WEIGHT_RSRC(base) \
    __builtin_amdgcn_make_buffer_rsrc( \
        const_cast<char*>(base), 0, 0xffffffffu, 0x31004000u)
#define HIPFIRE_WEIGHT_LOAD_U32(rsrc, byte_offset) \
    __builtin_amdgcn_raw_buffer_load_b32( \
        (rsrc), (unsigned int)(byte_offset), 0, HIPFIRE_WEIGHT_CPOL_AUX)
#define HIPFIRE_WEIGHT_LOAD_SPLIT_U32(rsrc, vector_offset, scalar_offset) \
    __builtin_amdgcn_raw_buffer_load_b32( \
        (rsrc), (unsigned int)(vector_offset), (unsigned int)(scalar_offset), \
        HIPFIRE_WEIGHT_CPOL_AUX)
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    HIPFIRE_WEIGHT_LOAD_U32((rsrc), (byte_offset))
#define HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, ptr, vector_offset, scalar_offset) \
    HIPFIRE_WEIGHT_LOAD_SPLIT_U32((rsrc), (vector_offset), (scalar_offset))
#else
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#define HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, ptr, vector_offset, scalar_offset) \
    (*(const unsigned int*)(ptr))
#endif
