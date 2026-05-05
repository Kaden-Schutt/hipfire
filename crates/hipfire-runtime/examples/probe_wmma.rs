//! Probe WMMA thread-to-output mapping empirically.
//! A = identity rows, B = columns with unique values.
//! Determines: which acc[j] in which thread corresponds to which C[m][n].

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    let src = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

extern "C" __launch_bounds__(32)
__global__ void probe_wmma(float* __restrict__ out) {
    const int tid = threadIdx.x;
    // A: row (tid%16) of 16x16 identity
    half16_t a;
    for (int i = 0; i < 16; i++) a[i] = (i == (tid & 15)) ? (_Float16)1.0f : (_Float16)0.0f;
    // B: column (tid%16) filled with value (tid%16 + 1) * 100
    half16_t b;
    for (int i = 0; i < 16; i++) b[i] = (_Float16)(float)((tid & 15) + 1) * (_Float16)100.0f;

    float8_t acc = {0,0,0,0,0,0,0,0};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    for (int j = 0; j < 8; j++) out[tid * 8 + j] = acc[j];
}
"#;

    gpu.ensure_kernel_public("probe_wmma", src, "probe_wmma").unwrap();

    let d_out = gpu.zeros(&[32 * 8], rdna_compute::DType::F32).unwrap();
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(d_out.buf.as_ptr());
    gpu.launch_kernel_blob("probe_wmma", [1, 1, 1], [32, 1, 1], 0, kernargs.as_mut_slice())
        .unwrap();

    let out = gpu.download_f32(&d_out).unwrap();

    // C = A * B where A = I, B[k][n] = (n+1)*100
    // So C[m][n] = B[m][n] = (n+1)*100
    // We expect: if thread t, acc[j] = C[m][n], then the value is (n+1)*100
    // n = batch/column dimension, m = row dimension

    eprintln!("WMMA output mapping (A=identity, B[k][col]=(col+1)*100):");
    eprintln!("Expected C[m][n] = (n+1)*100");
    eprintln!();
    for tid in 0..32 {
        let vals: Vec<f32> = (0..8).map(|j| out[tid * 8 + j]).collect();
        let row = tid & 15;
        let half = tid >> 4;
        // Decode: value = (n+1)*100, so n = value/100 - 1
        let cols: Vec<i32> = vals.iter().map(|&v| (v / 100.0 + 0.5) as i32 - 1).collect();
        eprintln!("  tid={tid:2} (row={row:2}, half={half}): values={:.0?} → cols={cols:?}",
            vals.iter().map(|v| *v as i32).collect::<Vec<_>>());
    }
}
