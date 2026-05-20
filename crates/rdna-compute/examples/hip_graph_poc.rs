// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipGraph POC for the redline-dispatch branch.
//!
//! Demonstrates the bit-exact graph capture/replay path for hipfire kernels.
//!
//! ## History: why the first version of this POC failed
//!
//! The original POC tried to capture `gpu.mul_f32(&a, &b, &scratch)` directly.
//! That dispatch helper packs kernargs via `Vec<*mut c_void>` where each
//! entry points at a stack-local variable. Under `hipStreamBeginCapture` on
//! gfx1100 / ROCm 6.3, `hipModuleLaunchKernel` captured those stack pointers
//! *by reference* into the graph node — by the time the graph was replayed,
//! the enclosing stack frame was gone and the kernel read garbage. The POC
//! showed `graph vs reference: 1/4096 match` (only index 0, which happened
//! to be zero in both) and the kernel effectively did nothing on replay.
//!
//! ## The fix
//!
//! `HipRuntime::launch_kernel_blob` uses the `extra` path of
//! `hipModuleLaunchKernel`, passing a contiguous kernarg byte buffer. HIP
//! copies the blob contents into the kernel node at capture time (the blob
//! pointer itself still has to stay alive, but the caller owns the buffer
//! and can hold it for the graph's full lifetime). Combined with the
//! `hip_bridge::KernargBlob` helper, dispatch paths can build correctly
//! aligned kernarg buffers without resorting to stack-local addresses.
//!
//! This POC proves the fix: it captures a `mul_f32` launch via the blob
//! path, replays it, and confirms every element of the output matches the
//! sequential reference.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example hip_graph_poc

use hip_bridge::KernargBlob;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("[hip_graph_poc] arch={}", gpu.arch);

    let dim = 4096usize;

    // Distinct buffers so the result has a unique signature.
    let init_a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let init_b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.0007 + 0.5).collect();
    let zero: Vec<f32> = vec![0.0; dim];

    let a = gpu.upload_f32(&init_a, &[dim]).unwrap();
    let b = gpu.upload_f32(&init_b, &[dim]).unwrap();
    let scratch = gpu.upload_f32(&zero, &[dim]).unwrap();

    // ─── Reference: run one mul_f32 on the null stream ──────────────────
    //
    // This ALSO serves as the kernel-warmup path — after this call,
    // `mul_f32` is compiled, the module is loaded, and the function is
    // cached in `gpu.functions`, so the subsequent `launch_kernel_blob`
    // lookup by name will succeed.
    eprintln!("\n--- Reference (null stream via dispatch.rs) ---");
    gpu.mul_f32(&a, &b, &scratch).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let reference = gpu.download_f32(&scratch).unwrap();
    eprintln!("  reference[0..4] = {:?}", &reference[..4]);

    // ─── Zero the scratch so we can see the kernel writing to it ────────
    gpu.hip.memcpy_htod(&scratch.buf, as_bytes(&zero)).unwrap();
    let zeroed = gpu.download_f32(&scratch).unwrap();
    assert_eq!(zeroed[1], 0.0, "reset failed");

    // ─── Build a kernarg blob for mul_f32 ───────────────────────────────
    //
    // Kernel signature:
    //   extern "C" __global__ void mul_f32(
    //       const float* a, const float* b, float* c, int n);
    //
    // ABI layout: 3 × 8-byte pointers + 1 × 4-byte i32 = 28 bytes.
    let n = dim as i32;
    let mut blob = KernargBlob::new();
    blob.push_ptr(a.buf.as_ptr());
    blob.push_ptr(b.buf.as_ptr());
    blob.push_ptr(scratch.buf.as_ptr());
    blob.push_i32(n);
    eprintln!("\n--- Kernarg blob ---");
    eprintln!("  size = {} bytes (expected 28)", blob.len());
    assert_eq!(blob.len(), 28);

    // ─── Blob-path launch on a dedicated stream ─────────────────────────
    //
    // Sanity check: the blob path works OUTSIDE of graph capture too,
    // which confirms the kernarg layout is correct before we trust it
    // inside a captured graph.
    eprintln!("\n--- Blob-path launch (no capture) ---");
    gpu.hip.memcpy_htod(&scratch.buf, as_bytes(&zero)).unwrap();
    let stream = gpu.hip.stream_create().unwrap();
    gpu.active_stream = Some(stream);

    let block = [256u32, 1, 1];
    let grid = [((dim as u32) + 255) / 256, 1, 1];
    gpu.launch_kernel_blob("mul_f32", grid, block, 0, blob.as_mut_slice())
        .unwrap();
    gpu.hip
        .stream_synchronize(gpu.active_stream.as_ref().unwrap())
        .unwrap();

    let blob_direct = gpu.download_f32(&scratch).unwrap();
    let bad = (0..dim)
        .filter(|&i| (blob_direct[i] - reference[i]).abs() > 1e-6)
        .count();
    eprintln!("  blob-direct vs reference: {}/{} match", dim - bad, dim);
    assert_eq!(bad, 0, "blob-path launch outside capture already differs");

    // ─── Graph capture via the blob path ────────────────────────────────
    eprintln!("\n--- hipGraph capture + replay (blob path) ---");
    gpu.hip.memcpy_htod(&scratch.buf, as_bytes(&zero)).unwrap();

    gpu.hip
        .stream_begin_capture(gpu.active_stream.as_ref().unwrap(), 0)
        .expect("stream_begin_capture");

    // Rebuild the blob — we'll hand this one to the graph and keep it
    // alive until graph destruction so the captured node's kernarg
    // pointer stays valid.
    let mut graph_blob = KernargBlob::new();
    graph_blob.push_ptr(a.buf.as_ptr());
    graph_blob.push_ptr(b.buf.as_ptr());
    graph_blob.push_ptr(scratch.buf.as_ptr());
    graph_blob.push_i32(n);

    gpu.launch_kernel_blob("mul_f32", grid, block, 0, graph_blob.as_mut_slice())
        .expect("launch during capture");

    let graph = gpu
        .hip
        .stream_end_capture(gpu.active_stream.as_ref().unwrap())
        .expect("stream_end_capture");
    eprintln!("  capture succeeded");

    let exec = gpu.hip.graph_instantiate(&graph).expect("graph_instantiate");
    eprintln!("  instantiated");

    // First replay
    gpu.hip.memcpy_htod(&scratch.buf, as_bytes(&zero)).unwrap();
    gpu.hip
        .graph_launch(&exec, gpu.active_stream.as_ref().unwrap())
        .unwrap();
    gpu.hip
        .stream_synchronize(gpu.active_stream.as_ref().unwrap())
        .unwrap();

    let graph_out = gpu.download_f32(&scratch).unwrap();
    let bad = (0..dim)
        .filter(|&i| (graph_out[i] - reference[i]).abs() > 1e-6)
        .count();
    eprintln!("  graph replay vs reference: {}/{} match", dim - bad, dim);

    if bad > 0 {
        eprintln!("  FIRST 8 MISMATCHES:");
        for i in (0..dim).filter(|&i| (graph_out[i] - reference[i]).abs() > 1e-6).take(8) {
            eprintln!(
                "    [{i}] graph={} ref={} delta={}",
                graph_out[i],
                reference[i],
                graph_out[i] - reference[i]
            );
        }
        eprintln!("\n=== POC FAILED — kernarg blob fix does not survive graph replay ===");
        std::process::exit(1);
    }

    // ─── Second replay to prove it's repeatable ─────────────────────────
    gpu.hip.memcpy_htod(&scratch.buf, as_bytes(&zero)).unwrap();
    gpu.hip
        .graph_launch(&exec, gpu.active_stream.as_ref().unwrap())
        .unwrap();
    gpu.hip
        .stream_synchronize(gpu.active_stream.as_ref().unwrap())
        .unwrap();
    let graph_out2 = gpu.download_f32(&scratch).unwrap();
    let bad2 = (0..dim)
        .filter(|&i| (graph_out2[i] - reference[i]).abs() > 1e-6)
        .count();
    eprintln!("  second replay: {}/{} match", dim - bad2, dim);
    assert_eq!(bad2, 0);

    // ─── Timing: sequential vs graph replay ─────────────────────────────
    let burst = 200u32;
    let mut burst_lat = Vec::with_capacity(50);
    for _ in 0..50 {
        let t = Instant::now();
        for _ in 0..burst {
            gpu.hip
                .graph_launch(&exec, gpu.active_stream.as_ref().unwrap())
                .unwrap();
        }
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();
        burst_lat.push(t.elapsed());
    }
    burst_lat.sort();
    let burst_total_us = burst_lat[burst_lat.len() / 2].as_secs_f64() * 1_000_000.0;
    let burst_per_replay = burst_total_us / burst as f64;
    eprintln!(
        "\n  graph burst: {:.1} µs total / {burst} replays → {:.2} µs/replay",
        burst_total_us, burst_per_replay
    );

    // Sequential reference: blob-path launch, no graph.
    let mut seq_lat = Vec::with_capacity(50);
    for _ in 0..50 {
        let t = Instant::now();
        for _ in 0..burst {
            gpu.launch_kernel_blob("mul_f32", grid, block, 0, blob.as_mut_slice())
                .unwrap();
        }
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();
        seq_lat.push(t.elapsed());
    }
    seq_lat.sort();
    let seq_total_us = seq_lat[seq_lat.len() / 2].as_secs_f64() * 1_000_000.0;
    let seq_per_launch = seq_total_us / burst as f64;
    eprintln!(
        "  sequential (blob-direct) burst: {:.1} µs total / {burst} launches → {:.2} µs/launch",
        seq_total_us, seq_per_launch
    );
    let speedup = seq_per_launch / burst_per_replay;
    eprintln!("  graph-replay speedup: {:.2}x", speedup);

    // Cleanup the single-node graph
    gpu.hip.graph_exec_destroy(exec).unwrap();
    gpu.hip.graph_destroy(graph).unwrap();

    // ─── Multi-node graph test ───────────────────────────────────────────
    //
    // The single-node result above showed graph replay ≫ sequential launch
    // on wall time, which is suspicious. The theory is that graph_launch
    // has a fixed per-invocation overhead (issuing the command to the GPU's
    // command processor) that gets amortized across all nodes inside the
    // graph. A forward pass has ~400 kernels in one graph — we need to
    // measure the per-NODE cost, not the per-graph_launch cost.
    //
    // This block captures N copies of mul_f32 into one graph, launches the
    // graph once per iter, and computes per-node walk cost by subtracting
    // the single-node number.
    eprintln!("\n--- Multi-node graph (N kernels per graph_launch) ---");
    for n_nodes in [1usize, 10, 50, 200] {
        // Keep all kernargs alive for the life of the graph.
        let mut blobs: Vec<KernargBlob> = (0..n_nodes)
            .map(|_| {
                let mut k = KernargBlob::new();
                k.push_ptr(a.buf.as_ptr());
                k.push_ptr(b.buf.as_ptr());
                k.push_ptr(scratch.buf.as_ptr());
                k.push_i32(n);
                k
            })
            .collect();

        gpu.hip
            .stream_begin_capture(gpu.active_stream.as_ref().unwrap(), 0)
            .unwrap();
        for blob in blobs.iter_mut() {
            gpu.launch_kernel_blob("mul_f32", grid, block, 0, blob.as_mut_slice())
                .unwrap();
        }
        let multi_graph = gpu
            .hip
            .stream_end_capture(gpu.active_stream.as_ref().unwrap())
            .unwrap();
        let multi_exec = gpu.hip.graph_instantiate(&multi_graph).unwrap();

        // Warm up
        for _ in 0..5 {
            gpu.hip
                .graph_launch(&multi_exec, gpu.active_stream.as_ref().unwrap())
                .unwrap();
        }
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();

        let iters = 100u32;
        let mut lat = Vec::with_capacity(20);
        for _ in 0..20 {
            let t = Instant::now();
            for _ in 0..iters {
                gpu.hip
                    .graph_launch(&multi_exec, gpu.active_stream.as_ref().unwrap())
                    .unwrap();
            }
            gpu.hip
                .stream_synchronize(gpu.active_stream.as_ref().unwrap())
                .unwrap();
            lat.push(t.elapsed());
        }
        lat.sort();
        let total_us = lat[lat.len() / 2].as_secs_f64() * 1_000_000.0;
        let per_launch = total_us / iters as f64;
        let per_node = per_launch / n_nodes as f64;
        eprintln!(
            "  N={n_nodes:4} nodes: {per_launch:7.2} µs/graph_launch, {per_node:6.2} µs/node"
        );

        gpu.hip.graph_exec_destroy(multi_exec).unwrap();
        gpu.hip.graph_destroy(multi_graph).unwrap();
        drop(blobs);
    }

    // Cleanup
    let stream = gpu.active_stream.take().unwrap();
    gpu.hip.stream_destroy(stream).unwrap();

    // Keep blob alive until the graph that captured it is destroyed.
    drop(blob);
    drop(graph_blob);

    eprintln!("\n=== POC PASSED — hipGraph capture + blob-path kernargs work bit-exact ===");
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
