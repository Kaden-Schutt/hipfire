// Opt-in GFX12 weight-load cache-policy experiment.
//
// HIPFIRE_GFX12_WEIGHT_CPOL_AUX is the raw-buffer cache-policy immediate:
//   16 = TH_RT    + SCOPE_DEV (buffer-addressing control)
//   18 = TH_HT    + SCOPE_DEV
//   20 = TH_NT_RT + SCOPE_DEV
//   22 = TH_NT_HT + SCOPE_DEV
//
// This preamble is concatenated ahead of selected HFQ4 kernel sources by
// kernels.rs. gfx1201 defaults to buffer RT after matched model-level gates;
// HIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS preserves the old global-load control.
#if defined(__gfx1201__) && \
    defined(HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE) && \
    !defined(HIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS)
#ifndef HIPFIRE_GFX12_WEIGHT_CPOL_AUX
#define HIPFIRE_GFX12_WEIGHT_CPOL_AUX 16
#endif
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 1
#define HIPFIRE_WEIGHT_RSRC(base) \
    __builtin_amdgcn_make_buffer_rsrc( \
        const_cast<char*>(base), 0, 0xffffffffu, 0x31004000u)
#define HIPFIRE_WEIGHT_LOAD_U32(rsrc, byte_offset) \
    __builtin_amdgcn_raw_buffer_load_b32( \
        (rsrc), (unsigned int)(byte_offset), 0, HIPFIRE_GFX12_WEIGHT_CPOL_AUX)
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    HIPFIRE_WEIGHT_LOAD_U32((rsrc), (byte_offset))
#else
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif
