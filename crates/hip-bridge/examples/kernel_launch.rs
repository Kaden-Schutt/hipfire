//! Test: compile a HIP kernel, load it via hip-bridge, launch, verify results.

use std::process::Command;

fn main() {
    let hip = hip_bridge::HipRuntime::load().expect("failed to load HIP runtime");
    hip.set_device(0).unwrap();

    // Write kernel source to temp file
    let tmp = std::env::temp_dir().join("hipfire_test");
    std::fs::create_dir_all(&tmp).unwrap();
    let src_path = tmp.join("vadd.hip");
    let obj_path = tmp.join("vadd.hsaco");

    std::fs::write(
        &src_path,
        r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#,
    )
    .unwrap();

    // Compile to code object
    println!("Compiling kernel...");
    let status = Command::new("hipcc")
        .args([
            "--genco",
            "--offload-arch=gfx1010",
            "-o",
            obj_path.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run hipcc");
    assert!(status.success(), "hipcc compilation failed");

    // Load module from file
    println!("Loading module...");
    let module = hip.module_load(obj_path.to_str().unwrap()).expect("module_load failed");

    // Get kernel function
    let func = hip
        .module_get_function(&module, "vector_add")
        .expect("get_function failed");
    println!("Got kernel function handle");

    // Prepare data
    let n: i32 = 65536;
    let size = (n as usize) * std::mem::size_of::<f32>();
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    // Allocate GPU buffers
    let d_a = hip.malloc(size).unwrap();
    let d_b = hip.malloc(size).unwrap();
    let d_c = hip.malloc(size).unwrap();

    // Upload data
    let a_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, size) };
    let b_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, size) };
    hip.memcpy_htod(&d_a, a_bytes).unwrap();
    hip.memcpy_htod(&d_b, b_bytes).unwrap();
    println!("Data uploaded to GPU");

    // Launch kernel
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;

    let mut d_a_ptr = d_a.as_ptr();
    let mut d_b_ptr = d_b.as_ptr();
    let mut d_c_ptr = d_c.as_ptr();
    let mut n_val = n;

    let mut params: Vec<*mut std::ffi::c_void> = vec![
        &mut d_a_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_b_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_c_ptr as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];

    println!("Launching kernel: grid={grid_size}, block={block_size}");
    unsafe {
        hip.launch_kernel(
            &func,
            [grid_size, 1, 1],
            [block_size, 1, 1],
            0,
            None,
            &mut params,
        )
        .expect("kernel launch failed");
    }

    // Read back results
    let mut c = vec![0.0f32; n as usize];
    let c_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut u8, size) };
    hip.memcpy_dtoh(c_bytes, &d_c).unwrap();

    // Verify: a[i] + b[i] = i + (n - i) = n for all i
    let mut errors = 0;
    for i in 0..n as usize {
        if (c[i] - n as f32).abs() > 0.001 {
            errors += 1;
            if errors <= 5 {
                eprintln!("  mismatch at {i}: expected {n}, got {}", c[i]);
            }
        }
    }

    // Cleanup
    hip.free(d_a).unwrap();
    hip.free(d_b).unwrap();
    hip.free(d_c).unwrap();

    if errors == 0 {
        println!("Kernel result VERIFIED: {n} elements, 0 errors");
        println!("\nkernel_launch test: PASS");
    } else {
        eprintln!("FAIL: {errors} errors out of {n} elements");
        std::process::exit(1);
    }
}
