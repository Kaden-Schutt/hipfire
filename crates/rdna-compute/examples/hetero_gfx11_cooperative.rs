// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-target gfx1100/gfx1151 cooperative-layer fixture.
//!
//! This is G1 of the DeepSeek V4 heterogeneous-compute campaign. It proves the
//! complete device-resident ownership chain before model lowering:
//!
//! ```text
//! gfx1100 producer -> ROCr SDMA activation -> gfx1151 routed expert
//!                  \-> concurrent gfx1100 shared branch
//! gfx1151 result   -> ROCr SDMA result     -> gfx1100 ordered join
//! ```
//!
//! The two code objects are compiled separately for their exact targets. AQL
//! completion signals feed public ROCr asynchronous copies directly, so the
//! cooperative path has no host wait in the layer loop. Two packet/result/state
//! slots are reused by parity across all 43 layers. The serial AQL arm moves the
//! activation-copy dependency from producer completion to shared-branch
//! completion; the host-sync arm launches the same exact-target kernels but
//! waits after every phase. All three arms are checked against one CPU oracle.

use hip_bridge::{DeviceBuffer, Function, HipRuntime, Module};
use rdna_compute::KernelCompiler;
use redline_rocr::packet::{BarrierAndPacket, KernelDispatchPacket, LaunchGeometry, PacketImage};
use redline_rocr::{
    CompletionSignal, Executable, GpuDevice, GpuSelector, KernargBuffer, KernargPool, Kernel,
    QueueSet, Runtime,
};
use std::ffi::c_void;
use std::sync::Arc;
use std::time::{Duration, Instant};

const HIDDEN: usize = 4096;
const ROUTES: usize = 6;
const PACKET_METADATA_WORDS: usize = 16;
const PACKET_WORDS: usize = HIDDEN + PACKET_METADATA_WORDS;
const DEFAULT_LAYERS: usize = 43;
const DEFAULT_SHARED_MIB: usize = 84;
const DEFAULT_EXPERT_MIB: usize = 42;
const BLOCK: u16 = 256;
const COPY_TO_EXPERT_ENGINE: u32 = 0x2;
const COPY_TO_DENSE_ENGINE: u32 = 0x1;

const SOURCE: &str = r#"
#include <hip/hip_runtime.h>

static __device__ __forceinline__ unsigned int rotl32(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32u - n));
}

static __device__ __forceinline__ unsigned int layer_key(unsigned int layer) {
    return (layer + 1u) * 0x9e3779b9u + 0x7f4a7c15u;
}

extern "C" __global__ void hetero_gfx11_producer(
    const unsigned int* input,
    unsigned int* packet,
    unsigned int layer,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int key = layer_key(layer);
    if (i < n) {
        const unsigned int shift = layer % 31u + 1u;
        packet[i] = rotl32(input[i] ^ key, shift) + i * 0x045d9f3bu;
    }
    if (i < 16u) {
        if (i < 6u) {
            packet[n + i] = (layer * 7u + i * 13u) % 256u;
        } else if (i < 12u) {
            const unsigned int route = i - 6u;
            packet[n + i] = rotl32(key ^ (route + 1u) * 0x27d4eb2du, route + 3u);
        } else {
            packet[n + i] = key ^ i * 0x85ebca6bu;
        }
    }
}

extern "C" __global__ void hetero_gfx11_shared(
    const unsigned int* packet,
    const unsigned int* weights,
    unsigned int* output,
    unsigned int layer,
    unsigned int n,
    unsigned int weight_words
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int checksum = 0u;
    for (unsigned int j = i; j < weight_words; j += n) {
        checksum ^= weights[j];
    }
    output[i] = rotl32(packet[i] ^ checksum ^ layer_key(layer), 5u);
}

extern "C" __global__ void hetero_gfx11_expert(
    const unsigned int* packet,
    const unsigned int* weights,
    unsigned int* output,
    unsigned int layer,
    unsigned int n,
    unsigned int weight_words
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int checksum = 0u;
    for (unsigned int j = i; j < weight_words; j += n) {
        checksum ^= weights[j];
    }
    unsigned int route_mix = 0u;
    #pragma unroll
    for (unsigned int route = 0u; route < 6u; ++route) {
        route_mix ^= rotl32(packet[n + route] * 0x9e3779b9u ^ packet[n + 6u + route], route + 1u);
    }
    output[i] = rotl32(packet[i] ^ checksum ^ route_mix ^ layer_key(layer), 11u);
}

extern "C" __global__ void hetero_gfx11_join(
    const unsigned int* shared,
    const unsigned int* routed,
    unsigned int* output,
    unsigned int layer,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = rotl32(shared[i] + routed[i], 3u) ^ layer_key(layer);
}

// Batched prefill fixture. Each output element performs a genuine row-by-
// matrix reduction over K; unsigned arithmetic makes the cross-architecture
// raw-bit oracle deterministic while retaining GEMM-shaped activation/weight
// access. The B=1 decode kernels above remain byte-for-byte independent.
extern "C" __global__ void hetero_gfx11_prefill_producer(
    const unsigned int* input,
    unsigned int* packet,
    unsigned int layer,
    unsigned int hidden,
    unsigned int batch
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= batch || col >= hidden) return;
    const unsigned int key = layer_key(layer);
    const unsigned int input_index = row * hidden + col;
    const unsigned int packet_stride = hidden + 16u;
    packet[row * packet_stride + col] =
        rotl32(input[input_index] ^ key ^ row * 0x85ebca6bu, layer % 31u + 1u)
        + col * 0x045d9f3bu;
    if (col < 16u) {
        const unsigned int meta = row * packet_stride + hidden + col;
        if (col < 6u) {
            packet[meta] = (layer * 7u + row * 11u + col * 13u) % 256u;
        } else if (col < 12u) {
            const unsigned int route = col - 6u;
            packet[meta] = rotl32(
                key ^ row * 0x165667b1u ^ (route + 1u) * 0x27d4eb2du,
                route + 3u
            );
        } else {
            packet[meta] = key ^ row * 0xc2b2ae35u ^ col * 0x85ebca6bu;
        }
    }
}

extern "C" __global__ void hetero_gfx11_prefill_shared(
    const unsigned int* packet,
    const unsigned int* weights,
    unsigned int* output,
    unsigned int layer,
    unsigned int hidden,
    unsigned int batch,
    unsigned int k_dim
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= batch || col >= hidden) return;
    const unsigned int packet_stride = hidden + 16u;
    unsigned int acc = layer_key(layer) ^ row * 0x9e3779b9u ^ col;
    for (unsigned int k = 0u; k < k_dim; ++k) {
        const unsigned int activation = packet[
            row * packet_stride + (col + k * 17u) % hidden
        ];
        const unsigned int weight = weights[col * k_dim + k];
        acc = acc * 0x0019660du + activation * (weight | 1u) + k;
    }
    output[row * hidden + col] = rotl32(acc, 5u);
}

extern "C" __global__ void hetero_gfx11_prefill_expert(
    const unsigned int* packet,
    const unsigned int* weights,
    unsigned int* output,
    unsigned int layer,
    unsigned int hidden,
    unsigned int batch,
    unsigned int k_dim
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= batch || col >= hidden) return;
    const unsigned int packet_stride = hidden + 16u;
    unsigned int route_mix = 0u;
    #pragma unroll
    for (unsigned int route = 0u; route < 6u; ++route) {
        route_mix ^= rotl32(
            packet[row * packet_stride + hidden + route] * 0x9e3779b9u ^
            packet[row * packet_stride + hidden + 6u + route],
            route + 1u
        );
    }
    unsigned int acc = layer_key(layer) ^ route_mix ^ row * 0x27d4eb2du ^ col;
    for (unsigned int k = 0u; k < k_dim; ++k) {
        const unsigned int activation = packet[
            row * packet_stride + (col + k * 29u) % hidden
        ];
        const unsigned int weight = weights[col * k_dim + k];
        acc = acc * 0x0019660du + activation * (weight | 1u) + k;
    }
    output[row * hidden + col] = rotl32(acc, 11u);
}

extern "C" __global__ void hetero_gfx11_prefill_join(
    const unsigned int* shared,
    const unsigned int* routed,
    unsigned int* output,
    unsigned int layer,
    unsigned int hidden,
    unsigned int batch
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= batch || col >= hidden) return;
    const unsigned int i = row * hidden + col;
    output[i] = rotl32(shared[i] + routed[i], 3u) ^ layer_key(layer);
}
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Schedule {
    Overlap,
    Serial,
}

impl Schedule {
    fn label(self) -> &'static str {
        match self {
            Self::Overlap => "device_overlap",
            Self::Serial => "device_serial",
        }
    }
}

#[derive(Debug)]
struct Config {
    layers: usize,
    warmups: usize,
    samples: usize,
    host_samples: usize,
    sync_samples: usize,
    shared_mib: usize,
    expert_mib: usize,
    prefill_batch: Option<usize>,
    prefill_depth: Option<usize>,
    prefill_shared_k: usize,
    prefill_expert_k: usize,
    expect_arch0: String,
    expect_arch1: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            layers: DEFAULT_LAYERS,
            warmups: 2,
            samples: 7,
            host_samples: 3,
            sync_samples: 20,
            shared_mib: DEFAULT_SHARED_MIB,
            expert_mib: DEFAULT_EXPERT_MIB,
            prefill_batch: None,
            prefill_depth: None,
            prefill_shared_k: 256,
            prefill_expert_k: 128,
            expect_arch0: "gfx1100".to_owned(),
            expect_arch1: "gfx1151".to_owned(),
        }
    }
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self::default();
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let mut index = 0;
        while index < args.len() {
            let flag = &args[index];
            let value = |index: &mut usize| -> Result<&str, String> {
                *index += 1;
                args.get(*index)
                    .map(String::as_str)
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            match flag.as_str() {
                "--layers" => config.layers = parse_positive(flag, value(&mut index)?)?,
                "--warmups" => config.warmups = parse_usize(flag, value(&mut index)?)?,
                "--samples" => config.samples = parse_positive(flag, value(&mut index)?)?,
                "--host-samples" => config.host_samples = parse_positive(flag, value(&mut index)?)?,
                "--sync-samples" => config.sync_samples = parse_positive(flag, value(&mut index)?)?,
                "--shared-mib" => config.shared_mib = parse_positive(flag, value(&mut index)?)?,
                "--expert-mib" => config.expert_mib = parse_positive(flag, value(&mut index)?)?,
                "--prefill-batch" => {
                    config.prefill_batch = Some(parse_positive(flag, value(&mut index)?)?)
                }
                "--prefill-depth" => {
                    config.prefill_depth = Some(parse_positive(flag, value(&mut index)?)?)
                }
                "--prefill-shared-k" => {
                    config.prefill_shared_k = parse_positive(flag, value(&mut index)?)?
                }
                "--prefill-expert-k" => {
                    config.prefill_expert_k = parse_positive(flag, value(&mut index)?)?
                }
                "--expect-arch0" => config.expect_arch0 = value(&mut index)?.to_owned(),
                "--expect-arch1" => config.expect_arch1 = value(&mut index)?.to_owned(),
                "-h" | "--help" => {
                    println!(
                        "hetero_gfx11_cooperative [options]\n\
                         --layers N         cooperative layers (default 43)\n\
                         --warmups N        untimed cooperative runs (default 2)\n\
                         --samples N        samples per AQL arm (default 7)\n\
                         --host-samples N   host-sync control samples (default 3)\n\
                         --sync-samples N   one-byte dependency-chain samples (default 20)\n\
                         --shared-mib N     gfx1100 bytes read per layer (default 84)\n\
                         --expert-mib N     gfx1151 bytes read per layer (default 42)\n\
                         --prefill-batch N  run G3 batched fixture instead of B=1 decode\n\
                         --prefill-depth N  repeat batched graphs to this prompt depth\n\
                         --prefill-shared-k N  gfx1100 matrix K (default 256)\n\
                         --prefill-expert-k N  gfx1151 matrix K (default 128)\n\
                         --expect-arch0 A   exact dense-device arch (default gfx1100)\n\
                         --expect-arch1 A   exact expert-device arch (default gfx1151)"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {flag:?}; use --help")),
            }
            index += 1;
        }
        if config.layers > 60 {
            return Err(
                "--layers must be <=60 so every persistent batch fits a 256-packet AQL queue"
                    .to_owned(),
            );
        }
        if config.prefill_batch.is_some_and(|batch| batch > 1024) {
            return Err("--prefill-batch must be <=1024".to_owned());
        }
        if config.prefill_depth.is_some() && config.prefill_batch.is_none() {
            return Err("--prefill-depth requires --prefill-batch".to_owned());
        }
        if config.prefill_shared_k > 4096 || config.prefill_expert_k > 4096 {
            return Err("prefill K dimensions must be <=4096".to_owned());
        }
        Ok(config)
    }
}

fn parse_positive(flag: &str, raw: &str) -> Result<usize, String> {
    let parsed = parse_usize(flag, raw)?;
    if parsed == 0 {
        Err(format!("{flag} must be positive"))
    } else {
        Ok(parsed)
    }
}

fn parse_usize(flag: &str, raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|error| format!("invalid {flag} value {raw:?}: {error}"))
}

fn layer_key(layer: u32) -> u32 {
    layer
        .wrapping_add(1)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(0x7f4a_7c15)
}

fn make_words(words: usize, salt: u32) -> Vec<u32> {
    (0..words)
        .map(|index| {
            let mut value = (index as u32).wrapping_mul(0x27d4_eb2d).wrapping_add(salt);
            value ^= value >> 16;
            value = value.wrapping_mul(0x85eb_ca6b);
            value ^ (value >> 13)
        })
        .collect()
}

fn xor_by_hidden(weights: &[u32]) -> Vec<u32> {
    let mut checksums = vec![0_u32; HIDDEN];
    for (index, &weight) in weights.iter().enumerate() {
        checksums[index % HIDDEN] ^= weight;
    }
    checksums
}

fn oracle(
    initial: &[u32],
    shared_checksums: &[u32],
    expert_checksums: &[u32],
    layers: usize,
) -> Vec<u32> {
    let mut state = initial.to_vec();
    let mut packet = vec![0_u32; PACKET_WORDS];
    for layer in 0..layers {
        let layer = layer as u32;
        let key = layer_key(layer);
        let shift = layer % 31 + 1;
        for index in 0..HIDDEN {
            packet[index] = (state[index] ^ key)
                .rotate_left(shift)
                .wrapping_add((index as u32).wrapping_mul(0x045d_9f3b));
        }
        for route in 0..ROUTES {
            packet[HIDDEN + route] = (layer * 7 + route as u32 * 13) % 256;
            packet[HIDDEN + ROUTES + route] =
                (key ^ (route as u32 + 1).wrapping_mul(0x27d4_eb2d)).rotate_left(route as u32 + 3);
        }
        for index in 12..PACKET_METADATA_WORDS {
            packet[HIDDEN + index] = key ^ (index as u32).wrapping_mul(0x85eb_ca6b);
        }
        let mut route_mix = 0_u32;
        for route in 0..ROUTES {
            route_mix ^= (packet[HIDDEN + route].wrapping_mul(0x9e37_79b9)
                ^ packet[HIDDEN + ROUTES + route])
                .rotate_left(route as u32 + 1);
        }
        for index in 0..HIDDEN {
            let shared = (packet[index] ^ shared_checksums[index] ^ key).rotate_left(5);
            let expert =
                (packet[index] ^ expert_checksums[index] ^ route_mix ^ key).rotate_left(11);
            state[index] = shared.wrapping_add(expert).rotate_left(3) ^ key;
        }
    }
    state
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    // SAFETY: all u32 bit patterns are valid bytes and the extent is exact.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn u32_bytes_mut(values: &mut [u32]) -> &mut [u8] {
    // SAFETY: same representation argument as `u32_bytes`, with unique access.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn pointer(buffer: &DeviceBuffer) -> u64 {
    buffer.as_ptr() as usize as u64
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u32_if_present(bytes: &mut [u8], offset: usize, value: u32) {
    if let Some(destination) = bytes.get_mut(offset..offset + 4) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
}

fn write_u16_if_present(bytes: &mut [u8], offset: usize, value: u16) {
    if let Some(destination) = bytes.get_mut(offset..offset + 2) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
}

fn kernargs(
    pool: &KernargPool,
    kernel: &Kernel,
    pointers: &[u64],
    scalars: &[u32],
) -> Result<KernargBuffer, Box<dyn std::error::Error>> {
    let geometry = LaunchGeometry::new([HIDDEN as u32, 1, 1], [BLOCK, 1, 1])?;
    kernargs_with_geometry(pool, kernel, pointers, scalars, geometry)
}

fn kernargs_with_geometry(
    pool: &KernargPool,
    kernel: &Kernel,
    pointers: &[u64],
    scalars: &[u32],
    geometry: LaunchGeometry,
) -> Result<KernargBuffer, Box<dyn std::error::Error>> {
    let mut buffer = pool.allocate_for(kernel.metadata())?;
    let mut offset = 0;
    for &value in pointers {
        write_u64(buffer.as_mut_bytes(), offset, value);
        offset += 8;
    }
    for &value in scalars {
        write_u32(buffer.as_mut_bytes(), offset, value);
        offset += 4;
    }
    // Clang Code Object V5/V6 appends hidden launch geometry after the
    // explicit parameters. HIP fills these fields before module launch; raw
    // AQL must do the same or blockIdx.x remains zero and only the first
    // workgroup's output survives. Keep every store length-guarded because a
    // kernel which does not consume a hidden suffix may report only its
    // explicit kernarg extent.
    let hidden = offset.next_multiple_of(8);
    let bytes = buffer.as_mut_bytes();
    for axis in 0..3 {
        let groups = geometry.grid_workitems[axis].div_ceil(u32::from(geometry.workgroup[axis]));
        write_u32_if_present(bytes, hidden + axis * 4, groups);
        write_u16_if_present(bytes, hidden + 12 + axis * 2, geometry.workgroup[axis]);
    }
    write_u16_if_present(bytes, hidden + 64, 1);
    Ok(buffer)
}

struct Buffers {
    state0: [DeviceBuffer; 2],
    packet0: [DeviceBuffer; 2],
    shared0: [DeviceBuffer; 2],
    routed0: [DeviceBuffer; 2],
    packet1: [DeviceBuffer; 2],
    expert1: [DeviceBuffer; 2],
    shared_weights0: DeviceBuffer,
    expert_weights1: DeviceBuffer,
}

impl Buffers {
    fn allocate(
        hip: &HipRuntime,
        shared_bytes: usize,
        expert_bytes: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        hip.set_device(0)?;
        let state0 = [hip.malloc(HIDDEN * 4)?, hip.malloc(HIDDEN * 4)?];
        let packet0 = [hip.malloc(PACKET_WORDS * 4)?, hip.malloc(PACKET_WORDS * 4)?];
        let shared0 = [hip.malloc(HIDDEN * 4)?, hip.malloc(HIDDEN * 4)?];
        let routed0 = [hip.malloc(HIDDEN * 4)?, hip.malloc(HIDDEN * 4)?];
        let shared_weights0 = hip.malloc(shared_bytes)?;

        hip.set_device(1)?;
        let packet1 = [hip.malloc(PACKET_WORDS * 4)?, hip.malloc(PACKET_WORDS * 4)?];
        let expert1 = [hip.malloc(HIDDEN * 4)?, hip.malloc(HIDDEN * 4)?];
        let expert_weights1 = hip.malloc(expert_bytes)?;
        Ok(Self {
            state0,
            packet0,
            shared0,
            routed0,
            packet1,
            expert1,
            shared_weights0,
            expert_weights1,
        })
    }

    fn initialize(
        &self,
        hip: &HipRuntime,
        initial: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        hip.set_device(0)?;
        hip.memcpy_htod(&self.state0[0], u32_bytes(initial))?;
        hip.memset(&self.state0[1], 0, HIDDEN * 4)?;
        for slot in 0..2 {
            hip.memset(&self.packet0[slot], 0, PACKET_WORDS * 4)?;
            hip.memset(&self.shared0[slot], 0, HIDDEN * 4)?;
            hip.memset(&self.routed0[slot], 0, HIDDEN * 4)?;
        }
        hip.device_synchronize()?;
        hip.set_device(1)?;
        for slot in 0..2 {
            hip.memset(&self.packet1[slot], 0, PACKET_WORDS * 4)?;
            hip.memset(&self.expert1[slot], 0, HIDDEN * 4)?;
        }
        hip.device_synchronize()?;
        Ok(())
    }

    fn assert_output(
        &self,
        hip: &HipRuntime,
        layers: usize,
        expected: &[u32],
        label: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        hip.set_device(0)?;
        let mut actual = vec![0_u32; HIDDEN];
        hip.memcpy_dtoh(u32_bytes_mut(&mut actual), &self.state0[layers % 2])?;
        if actual != expected {
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err(format!(
                "{label}: raw-bit mismatch at {first}: got={:#010x}, expected={:#010x}",
                actual[first], expected[first]
            )
            .into());
        }
        Ok(())
    }
}

struct PrefillBuffers {
    state0: [DeviceBuffer; 2],
    packet0: [DeviceBuffer; 2],
    shared0: [DeviceBuffer; 2],
    routed0: [DeviceBuffer; 2],
    packet1: [DeviceBuffer; 2],
    expert1: [DeviceBuffer; 2],
    shared_weights0: DeviceBuffer,
    expert_weights0: DeviceBuffer,
    expert_weights1: DeviceBuffer,
    elements: usize,
    packet_words: usize,
}

impl PrefillBuffers {
    fn allocate(
        hip: &HipRuntime,
        batch: usize,
        shared_k: usize,
        expert_k: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let elements = batch
            .checked_mul(HIDDEN)
            .ok_or("prefill element count overflow")?;
        let packet_words = batch
            .checked_mul(HIDDEN + PACKET_METADATA_WORDS)
            .ok_or("prefill packet count overflow")?;
        hip.set_device(0)?;
        let state0 = [hip.malloc(elements * 4)?, hip.malloc(elements * 4)?];
        let packet0 = [hip.malloc(packet_words * 4)?, hip.malloc(packet_words * 4)?];
        let shared0 = [hip.malloc(elements * 4)?, hip.malloc(elements * 4)?];
        let routed0 = [hip.malloc(elements * 4)?, hip.malloc(elements * 4)?];
        let shared_weights0 = hip.malloc(HIDDEN * shared_k * 4)?;
        let expert_weights0 = hip.malloc(HIDDEN * expert_k * 4)?;

        hip.set_device(1)?;
        let packet1 = [hip.malloc(packet_words * 4)?, hip.malloc(packet_words * 4)?];
        let expert1 = [hip.malloc(elements * 4)?, hip.malloc(elements * 4)?];
        let expert_weights1 = hip.malloc(HIDDEN * expert_k * 4)?;
        Ok(Self {
            state0,
            packet0,
            shared0,
            routed0,
            packet1,
            expert1,
            shared_weights0,
            expert_weights0,
            expert_weights1,
            elements,
            packet_words,
        })
    }

    fn upload_weights(
        &self,
        hip: &HipRuntime,
        shared: &[u32],
        expert: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        hip.set_device(0)?;
        hip.memcpy_htod(&self.shared_weights0, u32_bytes(shared))?;
        hip.memcpy_htod(&self.expert_weights0, u32_bytes(expert))?;
        hip.device_synchronize()?;
        hip.set_device(1)?;
        hip.memcpy_htod(&self.expert_weights1, u32_bytes(expert))?;
        hip.device_synchronize()?;
        Ok(())
    }

    fn initialize(
        &self,
        hip: &HipRuntime,
        initial: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if initial.len() != self.elements {
            return Err("prefill initial-state length mismatch".into());
        }
        hip.set_device(0)?;
        hip.memcpy_htod(&self.state0[0], u32_bytes(initial))?;
        hip.memset(&self.state0[1], 0, self.elements * 4)?;
        for slot in 0..2 {
            hip.memset(&self.packet0[slot], 0, self.packet_words * 4)?;
            hip.memset(&self.shared0[slot], 0, self.elements * 4)?;
            hip.memset(&self.routed0[slot], 0, self.elements * 4)?;
        }
        hip.device_synchronize()?;
        hip.set_device(1)?;
        for slot in 0..2 {
            hip.memset(&self.packet1[slot], 0, self.packet_words * 4)?;
            hip.memset(&self.expert1[slot], 0, self.elements * 4)?;
        }
        hip.device_synchronize()?;
        Ok(())
    }

    fn read_output(
        &self,
        hip: &HipRuntime,
        layers: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        hip.set_device(0)?;
        let mut output = vec![0_u32; self.elements];
        hip.memcpy_dtoh(u32_bytes_mut(&mut output), &self.state0[layers % 2])?;
        Ok(output)
    }

    fn assert_output(
        &self,
        hip: &HipRuntime,
        layers: usize,
        expected: &[u32],
        label: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let actual = self.read_output(hip, layers)?;
        if actual != expected {
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err(format!(
                "{label}: raw-bit mismatch at {first}: got={:#010x}, expected={:#010x}",
                actual[first], expected[first]
            )
            .into());
        }
        Ok(())
    }
}

struct KernelSet {
    producer: Kernel,
    shared: Kernel,
    expert: Kernel,
    join: Kernel,
    prefill_producer: Kernel,
    prefill_shared: Kernel,
    prefill_expert: Kernel,
    prefill_join: Kernel,
}

struct Graph {
    schedule: Schedule,
    runtime: Runtime,
    dev0: GpuDevice,
    dev1: GpuDevice,
    queues: QueueSet,
    batches: [Vec<PacketImage>; 2],
    producer_signals: Vec<CompletionSignal>,
    shared_signals: Vec<CompletionSignal>,
    activation_signals: Vec<CompletionSignal>,
    expert_signals: Vec<CompletionSignal>,
    result_signals: Vec<CompletionSignal>,
    join_signals: Vec<CompletionSignal>,
    _kernargs: Vec<KernargBuffer>,
    layers: usize,
}

impl Graph {
    #[allow(clippy::too_many_arguments)]
    fn new(
        schedule: Schedule,
        runtime: &Runtime,
        dev0: &GpuDevice,
        dev1: &GpuDevice,
        kernels: &KernelSet,
        buffers: &Buffers,
        layers: usize,
        shared_words: u32,
        expert_words: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pool0 = KernargPool::discover(dev0)?;
        let pool1 = KernargPool::discover(dev1)?;
        let queues = QueueSet::create_for_devices(&[dev0.clone(), dev1.clone()], 256)?;
        queues.set_profiling(true)?;
        let mut batches = [
            Vec::with_capacity(layers * 4),
            Vec::with_capacity(layers * 2),
        ];
        let mut producer_signals = Vec::with_capacity(layers);
        let mut shared_signals = Vec::with_capacity(layers);
        let mut activation_signals = Vec::with_capacity(layers);
        let mut expert_signals = Vec::with_capacity(layers);
        let mut result_signals = Vec::with_capacity(layers);
        let mut join_signals = Vec::with_capacity(layers);
        for _ in 0..layers {
            producer_signals.push(CompletionSignal::new(dev0)?);
            shared_signals.push(CompletionSignal::new(dev0)?);
            activation_signals.push(CompletionSignal::new(dev1)?);
            expert_signals.push(CompletionSignal::new(dev1)?);
            result_signals.push(CompletionSignal::new(dev0)?);
            join_signals.push(CompletionSignal::new(dev0)?);
        }

        let geometry = LaunchGeometry::new([HIDDEN as u32, 1, 1], [BLOCK, 1, 1])?;
        let mut all_kernargs = Vec::with_capacity(layers * 4);
        for layer in 0..layers {
            let slot = layer % 2;
            let input_slot = layer % 2;
            let output_slot = (layer + 1) % 2;
            let layer_u32 = layer as u32;

            let producer_args = kernargs(
                &pool0,
                &kernels.producer,
                &[
                    pointer(&buffers.state0[input_slot]),
                    pointer(&buffers.packet0[slot]),
                ],
                &[layer_u32, HIDDEN as u32],
            )?;
            let producer_packet = KernelDispatchPacket::new(
                kernels.producer.metadata(),
                geometry,
                0,
                producer_args.address(),
                producer_signals[layer].raw(),
            )?;
            batches[0].push(PacketImage::kernel(&producer_packet));
            all_kernargs.push(producer_args);

            let shared_args = kernargs(
                &pool0,
                &kernels.shared,
                &[
                    pointer(&buffers.packet0[slot]),
                    pointer(&buffers.shared_weights0),
                    pointer(&buffers.shared0[slot]),
                ],
                &[layer_u32, HIDDEN as u32, shared_words],
            )?;
            let shared_packet = KernelDispatchPacket::new(
                kernels.shared.metadata(),
                geometry,
                0,
                shared_args.address(),
                shared_signals[layer].raw(),
            )?;
            batches[0].push(PacketImage::kernel(&shared_packet));
            all_kernargs.push(shared_args);

            let join_barrier = BarrierAndPacket::new(
                &[result_signals[layer].raw()],
                redline_rocr::abi::Signal(0),
            )?;
            batches[0].push(PacketImage::barrier(&join_barrier));
            let join_args = kernargs(
                &pool0,
                &kernels.join,
                &[
                    pointer(&buffers.shared0[slot]),
                    pointer(&buffers.routed0[slot]),
                    pointer(&buffers.state0[output_slot]),
                ],
                &[layer_u32, HIDDEN as u32],
            )?;
            let join_packet = KernelDispatchPacket::new(
                kernels.join.metadata(),
                geometry,
                0,
                join_args.address(),
                join_signals[layer].raw(),
            )?;
            batches[0].push(PacketImage::kernel(&join_packet));
            all_kernargs.push(join_args);

            let activation_barrier = BarrierAndPacket::new(
                &[activation_signals[layer].raw()],
                redline_rocr::abi::Signal(0),
            )?;
            batches[1].push(PacketImage::barrier(&activation_barrier));
            let expert_args = kernargs(
                &pool1,
                &kernels.expert,
                &[
                    pointer(&buffers.packet1[slot]),
                    pointer(&buffers.expert_weights1),
                    pointer(&buffers.expert1[slot]),
                ],
                &[layer_u32, HIDDEN as u32, expert_words],
            )?;
            let expert_packet = KernelDispatchPacket::new(
                kernels.expert.metadata(),
                geometry,
                0,
                expert_args.address(),
                expert_signals[layer].raw(),
            )?;
            batches[1].push(PacketImage::kernel(&expert_packet));
            all_kernargs.push(expert_args);
        }
        Ok(Self {
            schedule,
            runtime: runtime.clone(),
            dev0: dev0.clone(),
            dev1: dev1.clone(),
            queues,
            batches,
            producer_signals,
            shared_signals,
            activation_signals,
            expert_signals,
            result_signals,
            join_signals,
            _kernargs: all_kernargs,
            layers,
        })
    }

    fn reset(&mut self) {
        for layer in 0..self.layers {
            self.producer_signals[layer].reset();
            self.shared_signals[layer].reset();
            self.activation_signals[layer].reset();
            self.expert_signals[layer].reset();
            self.result_signals[layer].reset();
            self.join_signals[layer].reset();
        }
    }

    fn run(&mut self, buffers: &Buffers) -> Result<RunReport, Box<dyn std::error::Error>> {
        self.reset();
        let started = Instant::now();
        for layer in 0..self.layers {
            let slot = layer % 2;
            let activation_dependency = match self.schedule {
                Schedule::Overlap => &self.producer_signals[layer],
                Schedule::Serial => &self.shared_signals[layer],
            };
            // SAFETY: every HIP allocation remains live and peer-accessible;
            // persistent dependency/completion signals are reset only after
            // the preceding graph has completed.
            unsafe {
                self.dev1.memory_async_copy(
                    buffers.packet1[slot].as_ptr(),
                    &self.dev0,
                    buffers.packet0[slot].as_ptr(),
                    PACKET_WORDS * 4,
                    &[activation_dependency],
                    &self.activation_signals[layer],
                    Some(COPY_TO_EXPERT_ENGINE),
                )?;
                self.dev0.memory_async_copy(
                    buffers.routed0[slot].as_ptr(),
                    &self.dev1,
                    buffers.expert1[slot].as_ptr(),
                    HIDDEN * 4,
                    &[&self.expert_signals[layer]],
                    &self.result_signals[layer],
                    Some(COPY_TO_DENSE_ENGINE),
                )?;
            }
        }
        self.queues.prepare_batches(&self.batches)?;
        self.queues.ring_prepared()?;
        let host_enqueue_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        self.queues
            .wait_signal(&self.join_signals[self.layers - 1], Duration::from_secs(30))?;
        for signals in [
            &self.producer_signals,
            &self.shared_signals,
            &self.activation_signals,
            &self.expert_signals,
            &self.result_signals,
            &self.join_signals,
        ] {
            for signal in signals {
                if signal.value_scacquire() != 0 {
                    return Err(format!(
                        "{} completion signal ended at {}",
                        self.schedule.label(),
                        signal.value_scacquire()
                    )
                    .into());
                }
            }
        }
        Ok(RunReport {
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            host_enqueue_us,
        })
    }

    fn timeline_report(&self) -> Result<TimelineReport, Box<dyn std::error::Error>> {
        let frequency = self.dev0.timestamp_frequency_hz()?;
        let mut shared_ticks = 0_u64;
        let mut expert_ticks = 0_u64;
        let mut activation_copy_ticks = 0_u64;
        let mut result_copy_ticks = 0_u64;
        let mut shared_expert_overlap_ticks = 0_u64;
        let mut activation_shared_overlap_ticks = 0_u64;
        let mut overlap_layers = 0_usize;
        let mut producer_to_shared_gap_ticks = 0_i128;
        let mut producer_to_activation_gap_ticks = 0_i128;
        let mut activation_to_expert_gap_ticks = 0_i128;
        let mut expert_to_result_gap_ticks = 0_i128;
        let mut result_to_join_gap_ticks = 0_i128;
        let mut shared_to_join_gap_ticks = 0_i128;
        for layer in 0..self.layers {
            let producer = self.dev0.dispatch_time(&self.producer_signals[layer])?;
            let shared = self.dev0.dispatch_time(&self.shared_signals[layer])?;
            let activation = self
                .runtime
                .async_copy_time(&self.activation_signals[layer])?;
            let expert = self.dev1.dispatch_time(&self.expert_signals[layer])?;
            let result = self.runtime.async_copy_time(&self.result_signals[layer])?;
            let join = self.dev0.dispatch_time(&self.join_signals[layer])?;
            shared_ticks += shared.end - shared.start;
            expert_ticks += expert.end - expert.start;
            activation_copy_ticks += activation.end - activation.start;
            result_copy_ticks += result.end - result.start;
            let shared_expert_overlap = shared
                .end
                .min(expert.end)
                .saturating_sub(shared.start.max(expert.start));
            let activation_shared_overlap = activation
                .end
                .min(shared.end)
                .saturating_sub(activation.start.max(shared.start));
            shared_expert_overlap_ticks += shared_expert_overlap;
            activation_shared_overlap_ticks += activation_shared_overlap;
            overlap_layers += usize::from(shared_expert_overlap != 0);
            producer_to_shared_gap_ticks += signed_gap(shared.start, producer.end);
            producer_to_activation_gap_ticks += signed_gap(activation.start, producer.end);
            activation_to_expert_gap_ticks += signed_gap(expert.start, activation.end);
            expert_to_result_gap_ticks += signed_gap(result.start, expert.end);
            result_to_join_gap_ticks += signed_gap(join.start, result.end);
            shared_to_join_gap_ticks += signed_gap(join.start, shared.end);
        }
        let to_us = |ticks: u64| ticks as f64 * 1_000_000.0 / frequency as f64;
        let signed_to_us = |ticks: i128| ticks as f64 * 1_000_000.0 / frequency as f64;
        Ok(TimelineReport {
            shared_us: to_us(shared_ticks),
            expert_us: to_us(expert_ticks),
            activation_copy_us: to_us(activation_copy_ticks),
            result_copy_us: to_us(result_copy_ticks),
            shared_expert_overlap_us: to_us(shared_expert_overlap_ticks),
            activation_shared_overlap_us: to_us(activation_shared_overlap_ticks),
            overlap_layers,
            producer_to_shared_gap_us: signed_to_us(producer_to_shared_gap_ticks),
            producer_to_activation_gap_us: signed_to_us(producer_to_activation_gap_ticks),
            activation_to_expert_gap_us: signed_to_us(activation_to_expert_gap_ticks),
            expert_to_result_gap_us: signed_to_us(expert_to_result_gap_ticks),
            result_to_join_gap_us: signed_to_us(result_to_join_gap_ticks),
            shared_to_join_gap_us: signed_to_us(shared_to_join_gap_ticks),
        })
    }
}

struct PrefillGraph {
    schedule: Schedule,
    runtime: Runtime,
    dev0: GpuDevice,
    dev1: GpuDevice,
    queues: QueueSet,
    batches: [Vec<PacketImage>; 2],
    producer_signals: Vec<CompletionSignal>,
    shared_signals: Vec<CompletionSignal>,
    activation_signals: Vec<CompletionSignal>,
    expert_signals: Vec<CompletionSignal>,
    result_signals: Vec<CompletionSignal>,
    join_signals: Vec<CompletionSignal>,
    _kernargs: Vec<KernargBuffer>,
    layers: usize,
    activation_bytes: usize,
    result_bytes: usize,
}

impl PrefillGraph {
    #[allow(clippy::too_many_arguments)]
    fn new(
        schedule: Schedule,
        runtime: &Runtime,
        dev0: &GpuDevice,
        dev1: &GpuDevice,
        kernels: &KernelSet,
        buffers: &PrefillBuffers,
        layers: usize,
        batch: usize,
        shared_k: usize,
        expert_k: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pool0 = KernargPool::discover(dev0)?;
        let pool1 = KernargPool::discover(dev1)?;
        let queues = QueueSet::create_for_devices(&[dev0.clone(), dev1.clone()], 256)?;
        queues.set_profiling(true)?;
        let mut batches = [
            Vec::with_capacity(layers * 4),
            Vec::with_capacity(layers * 2),
        ];
        let mut producer_signals = Vec::with_capacity(layers);
        let mut shared_signals = Vec::with_capacity(layers);
        let mut activation_signals = Vec::with_capacity(layers);
        let mut expert_signals = Vec::with_capacity(layers);
        let mut result_signals = Vec::with_capacity(layers);
        let mut join_signals = Vec::with_capacity(layers);
        for _ in 0..layers {
            producer_signals.push(CompletionSignal::new(dev0)?);
            shared_signals.push(CompletionSignal::new(dev0)?);
            activation_signals.push(CompletionSignal::new(dev1)?);
            expert_signals.push(CompletionSignal::new(dev1)?);
            result_signals.push(CompletionSignal::new(dev0)?);
            join_signals.push(CompletionSignal::new(dev0)?);
        }
        let batch_u32 = u32::try_from(batch)?;
        let shared_k_u32 = u32::try_from(shared_k)?;
        let expert_k_u32 = u32::try_from(expert_k)?;
        let geometry = LaunchGeometry::new([HIDDEN as u32, batch_u32, 1], [BLOCK, 1, 1])?;
        let mut all_kernargs = Vec::with_capacity(layers * 4);
        for layer in 0..layers {
            let slot = layer % 2;
            let output_slot = (layer + 1) % 2;
            let layer_u32 = layer as u32;
            let common = [layer_u32, HIDDEN as u32, batch_u32];

            let producer_args = kernargs_with_geometry(
                &pool0,
                &kernels.prefill_producer,
                &[
                    pointer(&buffers.state0[slot]),
                    pointer(&buffers.packet0[slot]),
                ],
                &common,
                geometry,
            )?;
            let producer_packet = KernelDispatchPacket::new(
                kernels.prefill_producer.metadata(),
                geometry,
                0,
                producer_args.address(),
                producer_signals[layer].raw(),
            )?;
            batches[0].push(PacketImage::kernel(&producer_packet));
            all_kernargs.push(producer_args);

            let shared_args = kernargs_with_geometry(
                &pool0,
                &kernels.prefill_shared,
                &[
                    pointer(&buffers.packet0[slot]),
                    pointer(&buffers.shared_weights0),
                    pointer(&buffers.shared0[slot]),
                ],
                &[layer_u32, HIDDEN as u32, batch_u32, shared_k_u32],
                geometry,
            )?;
            let shared_packet = KernelDispatchPacket::new(
                kernels.prefill_shared.metadata(),
                geometry,
                0,
                shared_args.address(),
                shared_signals[layer].raw(),
            )?;
            batches[0].push(PacketImage::kernel(&shared_packet));
            all_kernargs.push(shared_args);

            let join_barrier = BarrierAndPacket::new(
                &[result_signals[layer].raw()],
                redline_rocr::abi::Signal(0),
            )?;
            batches[0].push(PacketImage::barrier(&join_barrier));
            let join_args = kernargs_with_geometry(
                &pool0,
                &kernels.prefill_join,
                &[
                    pointer(&buffers.shared0[slot]),
                    pointer(&buffers.routed0[slot]),
                    pointer(&buffers.state0[output_slot]),
                ],
                &common,
                geometry,
            )?;
            let join_packet = KernelDispatchPacket::new(
                kernels.prefill_join.metadata(),
                geometry,
                0,
                join_args.address(),
                join_signals[layer].raw(),
            )?;
            batches[0].push(PacketImage::kernel(&join_packet));
            all_kernargs.push(join_args);

            let activation_barrier = BarrierAndPacket::new(
                &[activation_signals[layer].raw()],
                redline_rocr::abi::Signal(0),
            )?;
            batches[1].push(PacketImage::barrier(&activation_barrier));
            let expert_args = kernargs_with_geometry(
                &pool1,
                &kernels.prefill_expert,
                &[
                    pointer(&buffers.packet1[slot]),
                    pointer(&buffers.expert_weights1),
                    pointer(&buffers.expert1[slot]),
                ],
                &[layer_u32, HIDDEN as u32, batch_u32, expert_k_u32],
                geometry,
            )?;
            let expert_packet = KernelDispatchPacket::new(
                kernels.prefill_expert.metadata(),
                geometry,
                0,
                expert_args.address(),
                expert_signals[layer].raw(),
            )?;
            batches[1].push(PacketImage::kernel(&expert_packet));
            all_kernargs.push(expert_args);
        }
        Ok(Self {
            schedule,
            runtime: runtime.clone(),
            dev0: dev0.clone(),
            dev1: dev1.clone(),
            queues,
            batches,
            producer_signals,
            shared_signals,
            activation_signals,
            expert_signals,
            result_signals,
            join_signals,
            _kernargs: all_kernargs,
            layers,
            activation_bytes: buffers.packet_words * 4,
            result_bytes: buffers.elements * 4,
        })
    }

    fn reset(&mut self) {
        for layer in 0..self.layers {
            self.producer_signals[layer].reset();
            self.shared_signals[layer].reset();
            self.activation_signals[layer].reset();
            self.expert_signals[layer].reset();
            self.result_signals[layer].reset();
            self.join_signals[layer].reset();
        }
    }

    fn run(&mut self, buffers: &PrefillBuffers) -> Result<RunReport, Box<dyn std::error::Error>> {
        self.reset();
        let started = Instant::now();
        for layer in 0..self.layers {
            let slot = layer % 2;
            let activation_dependency = match self.schedule {
                Schedule::Overlap => &self.producer_signals[layer],
                Schedule::Serial => &self.shared_signals[layer],
            };
            // SAFETY: all allocations and dependency signals are persistent
            // through terminal completion; the previous graph is quiescent
            // before signals or parity slots are reused.
            unsafe {
                self.dev1.memory_async_copy(
                    buffers.packet1[slot].as_ptr(),
                    &self.dev0,
                    buffers.packet0[slot].as_ptr(),
                    self.activation_bytes,
                    &[activation_dependency],
                    &self.activation_signals[layer],
                    Some(COPY_TO_EXPERT_ENGINE),
                )?;
                self.dev0.memory_async_copy(
                    buffers.routed0[slot].as_ptr(),
                    &self.dev1,
                    buffers.expert1[slot].as_ptr(),
                    self.result_bytes,
                    &[&self.expert_signals[layer]],
                    &self.result_signals[layer],
                    Some(COPY_TO_DENSE_ENGINE),
                )?;
            }
        }
        self.queues.prepare_batches(&self.batches)?;
        self.queues.ring_prepared()?;
        let host_enqueue_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        self.queues.wait_signal(
            &self.join_signals[self.layers - 1],
            Duration::from_secs(120),
        )?;
        for signals in [
            &self.producer_signals,
            &self.shared_signals,
            &self.activation_signals,
            &self.expert_signals,
            &self.result_signals,
            &self.join_signals,
        ] {
            for signal in signals {
                if signal.value_scacquire() != 0 {
                    return Err(format!(
                        "prefill {} completion signal ended at {}",
                        self.schedule.label(),
                        signal.value_scacquire()
                    )
                    .into());
                }
            }
        }
        Ok(RunReport {
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            host_enqueue_us,
        })
    }

    fn timeline_report(&self) -> Result<TimelineReport, Box<dyn std::error::Error>> {
        timeline_report(
            &self.runtime,
            &self.dev0,
            &self.dev1,
            &self.producer_signals,
            &self.shared_signals,
            &self.activation_signals,
            &self.expert_signals,
            &self.result_signals,
            &self.join_signals,
        )
    }

    fn pcie_bytes_per_graph(&self) -> usize {
        self.layers * (self.activation_bytes + self.result_bytes)
    }

    fn pcie_transactions_per_graph(&self) -> usize {
        self.layers * 2
    }
}

#[derive(Clone, Copy, Debug)]
struct RunReport {
    wall_ms: f64,
    host_enqueue_us: f64,
}

#[derive(Clone, Copy, Debug)]
struct TimelineReport {
    shared_us: f64,
    expert_us: f64,
    activation_copy_us: f64,
    result_copy_us: f64,
    shared_expert_overlap_us: f64,
    activation_shared_overlap_us: f64,
    overlap_layers: usize,
    producer_to_shared_gap_us: f64,
    producer_to_activation_gap_us: f64,
    activation_to_expert_gap_us: f64,
    expert_to_result_gap_us: f64,
    result_to_join_gap_us: f64,
    shared_to_join_gap_us: f64,
}

fn signed_gap(start: u64, end: u64) -> i128 {
    i128::from(start) - i128::from(end)
}

#[allow(clippy::too_many_arguments)]
fn timeline_report(
    runtime: &Runtime,
    dev0: &GpuDevice,
    dev1: &GpuDevice,
    producer_signals: &[CompletionSignal],
    shared_signals: &[CompletionSignal],
    activation_signals: &[CompletionSignal],
    expert_signals: &[CompletionSignal],
    result_signals: &[CompletionSignal],
    join_signals: &[CompletionSignal],
) -> Result<TimelineReport, Box<dyn std::error::Error>> {
    let layers = producer_signals.len();
    if [
        shared_signals.len(),
        activation_signals.len(),
        expert_signals.len(),
        result_signals.len(),
        join_signals.len(),
    ]
    .into_iter()
    .any(|len| len != layers)
    {
        return Err("heterogeneous timeline signal-count mismatch".into());
    }
    let frequency = dev0.timestamp_frequency_hz()?;
    let mut shared_ticks = 0_u64;
    let mut expert_ticks = 0_u64;
    let mut activation_copy_ticks = 0_u64;
    let mut result_copy_ticks = 0_u64;
    let mut shared_expert_overlap_ticks = 0_u64;
    let mut activation_shared_overlap_ticks = 0_u64;
    let mut overlap_layers = 0_usize;
    let mut producer_to_shared_gap_ticks = 0_i128;
    let mut producer_to_activation_gap_ticks = 0_i128;
    let mut activation_to_expert_gap_ticks = 0_i128;
    let mut expert_to_result_gap_ticks = 0_i128;
    let mut result_to_join_gap_ticks = 0_i128;
    let mut shared_to_join_gap_ticks = 0_i128;
    for layer in 0..layers {
        let producer = dev0.dispatch_time(&producer_signals[layer])?;
        let shared = dev0.dispatch_time(&shared_signals[layer])?;
        let activation = runtime.async_copy_time(&activation_signals[layer])?;
        let expert = dev1.dispatch_time(&expert_signals[layer])?;
        let result = runtime.async_copy_time(&result_signals[layer])?;
        let join = dev0.dispatch_time(&join_signals[layer])?;
        shared_ticks += shared.end - shared.start;
        expert_ticks += expert.end - expert.start;
        activation_copy_ticks += activation.end - activation.start;
        result_copy_ticks += result.end - result.start;
        let shared_expert_overlap = shared
            .end
            .min(expert.end)
            .saturating_sub(shared.start.max(expert.start));
        let activation_shared_overlap = activation
            .end
            .min(shared.end)
            .saturating_sub(activation.start.max(shared.start));
        shared_expert_overlap_ticks += shared_expert_overlap;
        activation_shared_overlap_ticks += activation_shared_overlap;
        overlap_layers += usize::from(shared_expert_overlap != 0);
        producer_to_shared_gap_ticks += signed_gap(shared.start, producer.end);
        producer_to_activation_gap_ticks += signed_gap(activation.start, producer.end);
        activation_to_expert_gap_ticks += signed_gap(expert.start, activation.end);
        expert_to_result_gap_ticks += signed_gap(result.start, expert.end);
        result_to_join_gap_ticks += signed_gap(join.start, result.end);
        shared_to_join_gap_ticks += signed_gap(join.start, shared.end);
    }
    let to_us = |ticks: u64| ticks as f64 * 1_000_000.0 / frequency as f64;
    let signed_to_us = |ticks: i128| ticks as f64 * 1_000_000.0 / frequency as f64;
    Ok(TimelineReport {
        shared_us: to_us(shared_ticks),
        expert_us: to_us(expert_ticks),
        activation_copy_us: to_us(activation_copy_ticks),
        result_copy_us: to_us(result_copy_ticks),
        shared_expert_overlap_us: to_us(shared_expert_overlap_ticks),
        activation_shared_overlap_us: to_us(activation_shared_overlap_ticks),
        overlap_layers,
        producer_to_shared_gap_us: signed_to_us(producer_to_shared_gap_ticks),
        producer_to_activation_gap_us: signed_to_us(producer_to_activation_gap_ticks),
        activation_to_expert_gap_us: signed_to_us(activation_to_expert_gap_ticks),
        expert_to_result_gap_us: signed_to_us(expert_to_result_gap_ticks),
        result_to_join_gap_us: signed_to_us(result_to_join_gap_ticks),
        shared_to_join_gap_us: signed_to_us(shared_to_join_gap_ticks),
    })
}

struct HipFunctions {
    _module0: Module,
    _module1: Module,
    producer: Function,
    shared: Function,
    expert: Function,
    join: Function,
    prefill_producer: Function,
    prefill_shared: Function,
    prefill_expert0: Function,
    prefill_join: Function,
}

impl HipFunctions {
    fn load(
        hip: &HipRuntime,
        image0: &[u8],
        image1: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        hip.set_device(0)?;
        let module0 = hip.module_load_data(image0)?;
        let producer = hip.module_get_function(&module0, "hetero_gfx11_producer")?;
        let shared = hip.module_get_function(&module0, "hetero_gfx11_shared")?;
        let join = hip.module_get_function(&module0, "hetero_gfx11_join")?;
        let prefill_producer =
            hip.module_get_function(&module0, "hetero_gfx11_prefill_producer")?;
        let prefill_shared = hip.module_get_function(&module0, "hetero_gfx11_prefill_shared")?;
        let prefill_expert0 = hip.module_get_function(&module0, "hetero_gfx11_prefill_expert")?;
        let prefill_join = hip.module_get_function(&module0, "hetero_gfx11_prefill_join")?;
        hip.set_device(1)?;
        let module1 = hip.module_load_data(image1)?;
        let expert = hip.module_get_function(&module1, "hetero_gfx11_expert")?;
        Ok(Self {
            _module0: module0,
            _module1: module1,
            producer,
            shared,
            expert,
            join,
            prefill_producer,
            prefill_shared,
            prefill_expert0,
            prefill_join,
        })
    }
}

fn launch_producer(
    hip: &HipRuntime,
    function: &Function,
    input: &DeviceBuffer,
    packet: &DeviceBuffer,
    layer: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut input = input.as_ptr();
    let mut packet = packet.as_ptr();
    let mut layer = layer;
    let mut n = HIDDEN as u32;
    let mut params = [
        (&mut input as *mut *mut c_void).cast(),
        (&mut packet as *mut *mut c_void).cast(),
        (&mut layer as *mut u32).cast(),
        (&mut n as *mut u32).cast(),
    ];
    // SAFETY: parameters and extents exactly match the compiled kernel ABI.
    unsafe {
        hip.launch_kernel(
            function,
            [(HIDDEN as u32).div_ceil(BLOCK as u32), 1, 1],
            [BLOCK as u32, 1, 1],
            0,
            None,
            &mut params,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_branch(
    hip: &HipRuntime,
    function: &Function,
    packet: &DeviceBuffer,
    weights: &DeviceBuffer,
    output: &DeviceBuffer,
    layer: u32,
    weight_words: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut packet = packet.as_ptr();
    let mut weights = weights.as_ptr();
    let mut output = output.as_ptr();
    let mut layer = layer;
    let mut n = HIDDEN as u32;
    let mut weight_words = weight_words;
    let mut params = [
        (&mut packet as *mut *mut c_void).cast(),
        (&mut weights as *mut *mut c_void).cast(),
        (&mut output as *mut *mut c_void).cast(),
        (&mut layer as *mut u32).cast(),
        (&mut n as *mut u32).cast(),
        (&mut weight_words as *mut u32).cast(),
    ];
    // SAFETY: parameters and extents exactly match both branch kernel ABIs.
    unsafe {
        hip.launch_kernel(
            function,
            [(HIDDEN as u32).div_ceil(BLOCK as u32), 1, 1],
            [BLOCK as u32, 1, 1],
            0,
            None,
            &mut params,
        )?;
    }
    Ok(())
}

fn launch_join(
    hip: &HipRuntime,
    function: &Function,
    shared: &DeviceBuffer,
    routed: &DeviceBuffer,
    output: &DeviceBuffer,
    layer: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut shared = shared.as_ptr();
    let mut routed = routed.as_ptr();
    let mut output = output.as_ptr();
    let mut layer = layer;
    let mut n = HIDDEN as u32;
    let mut params = [
        (&mut shared as *mut *mut c_void).cast(),
        (&mut routed as *mut *mut c_void).cast(),
        (&mut output as *mut *mut c_void).cast(),
        (&mut layer as *mut u32).cast(),
        (&mut n as *mut u32).cast(),
    ];
    // SAFETY: parameters and extents exactly match the join kernel ABI.
    unsafe {
        hip.launch_kernel(
            function,
            [(HIDDEN as u32).div_ceil(BLOCK as u32), 1, 1],
            [BLOCK as u32, 1, 1],
            0,
            None,
            &mut params,
        )?;
    }
    Ok(())
}

fn launch_prefill_producer(
    hip: &HipRuntime,
    function: &Function,
    input: &DeviceBuffer,
    packet: &DeviceBuffer,
    layer: u32,
    batch: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut input = input.as_ptr();
    let mut packet = packet.as_ptr();
    let mut layer = layer;
    let mut hidden = HIDDEN as u32;
    let mut batch = batch;
    let mut params = [
        (&mut input as *mut *mut c_void).cast(),
        (&mut packet as *mut *mut c_void).cast(),
        (&mut layer as *mut u32).cast(),
        (&mut hidden as *mut u32).cast(),
        (&mut batch as *mut u32).cast(),
    ];
    // SAFETY: parameters and 2-D geometry match the compiled kernel ABI.
    unsafe {
        hip.launch_kernel(
            function,
            [(HIDDEN as u32).div_ceil(BLOCK as u32), batch, 1],
            [BLOCK as u32, 1, 1],
            0,
            None,
            &mut params,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_prefill_branch(
    hip: &HipRuntime,
    function: &Function,
    packet: &DeviceBuffer,
    weights: &DeviceBuffer,
    output: &DeviceBuffer,
    layer: u32,
    batch: u32,
    k_dim: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut packet = packet.as_ptr();
    let mut weights = weights.as_ptr();
    let mut output = output.as_ptr();
    let mut layer = layer;
    let mut hidden = HIDDEN as u32;
    let mut batch = batch;
    let mut k_dim = k_dim;
    let mut params = [
        (&mut packet as *mut *mut c_void).cast(),
        (&mut weights as *mut *mut c_void).cast(),
        (&mut output as *mut *mut c_void).cast(),
        (&mut layer as *mut u32).cast(),
        (&mut hidden as *mut u32).cast(),
        (&mut batch as *mut u32).cast(),
        (&mut k_dim as *mut u32).cast(),
    ];
    // SAFETY: parameters and 2-D geometry match both prefill branch ABIs.
    unsafe {
        hip.launch_kernel(
            function,
            [(HIDDEN as u32).div_ceil(BLOCK as u32), batch, 1],
            [BLOCK as u32, 1, 1],
            0,
            None,
            &mut params,
        )?;
    }
    Ok(())
}

fn launch_prefill_join(
    hip: &HipRuntime,
    function: &Function,
    shared: &DeviceBuffer,
    routed: &DeviceBuffer,
    output: &DeviceBuffer,
    layer: u32,
    batch: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut shared = shared.as_ptr();
    let mut routed = routed.as_ptr();
    let mut output = output.as_ptr();
    let mut layer = layer;
    let mut hidden = HIDDEN as u32;
    let mut batch = batch;
    let mut params = [
        (&mut shared as *mut *mut c_void).cast(),
        (&mut routed as *mut *mut c_void).cast(),
        (&mut output as *mut *mut c_void).cast(),
        (&mut layer as *mut u32).cast(),
        (&mut hidden as *mut u32).cast(),
        (&mut batch as *mut u32).cast(),
    ];
    // SAFETY: parameters and 2-D geometry match the compiled join ABI.
    unsafe {
        hip.launch_kernel(
            function,
            [(HIDDEN as u32).div_ceil(BLOCK as u32), batch, 1],
            [BLOCK as u32, 1, 1],
            0,
            None,
            &mut params,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn single_device_prefill_oracle(
    hip: &HipRuntime,
    functions: &HipFunctions,
    buffers: &PrefillBuffers,
    layers: usize,
    batch: usize,
    shared_k: usize,
    expert_k: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    hip.set_device(0)?;
    let batch = u32::try_from(batch)?;
    let shared_k = u32::try_from(shared_k)?;
    let expert_k = u32::try_from(expert_k)?;
    for layer in 0..layers {
        let slot = layer % 2;
        launch_prefill_producer(
            hip,
            &functions.prefill_producer,
            &buffers.state0[slot],
            &buffers.packet0[slot],
            layer as u32,
            batch,
        )?;
        launch_prefill_branch(
            hip,
            &functions.prefill_shared,
            &buffers.packet0[slot],
            &buffers.shared_weights0,
            &buffers.shared0[slot],
            layer as u32,
            batch,
            shared_k,
        )?;
        launch_prefill_branch(
            hip,
            &functions.prefill_expert0,
            &buffers.packet0[slot],
            &buffers.expert_weights0,
            &buffers.routed0[slot],
            layer as u32,
            batch,
            expert_k,
        )?;
        launch_prefill_join(
            hip,
            &functions.prefill_join,
            &buffers.shared0[slot],
            &buffers.routed0[slot],
            &buffers.state0[(layer + 1) % 2],
            layer as u32,
            batch,
        )?;
    }
    hip.device_synchronize()?;
    buffers.read_output(hip, layers)
}

#[allow(clippy::too_many_arguments)]
fn host_sync_sample(
    hip: &HipRuntime,
    functions: &HipFunctions,
    dev0: &GpuDevice,
    dev1: &GpuDevice,
    buffers: &Buffers,
    layers: usize,
    shared_words: u32,
    expert_words: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut activation = CompletionSignal::new(dev1)?;
    let mut result = CompletionSignal::new(dev0)?;
    let started = Instant::now();
    for layer in 0..layers {
        let slot = layer % 2;
        hip.set_device(0)?;
        launch_producer(
            hip,
            &functions.producer,
            &buffers.state0[slot],
            &buffers.packet0[slot],
            layer as u32,
        )?;
        hip.device_synchronize()?;
        launch_branch(
            hip,
            &functions.shared,
            &buffers.packet0[slot],
            &buffers.shared_weights0,
            &buffers.shared0[slot],
            layer as u32,
            shared_words,
        )?;
        hip.device_synchronize()?;
        activation.reset();
        // SAFETY: producer/shared have completed, allocations remain live.
        unsafe {
            dev1.memory_async_copy(
                buffers.packet1[slot].as_ptr(),
                dev0,
                buffers.packet0[slot].as_ptr(),
                PACKET_WORDS * 4,
                &[],
                &activation,
                Some(COPY_TO_EXPERT_ENGINE),
            )?;
        }
        activation.wait_timeout(Duration::from_secs(30))?;

        hip.set_device(1)?;
        launch_branch(
            hip,
            &functions.expert,
            &buffers.packet1[slot],
            &buffers.expert_weights1,
            &buffers.expert1[slot],
            layer as u32,
            expert_words,
        )?;
        hip.device_synchronize()?;
        result.reset();
        // SAFETY: expert has completed and both allocations remain live.
        unsafe {
            dev0.memory_async_copy(
                buffers.routed0[slot].as_ptr(),
                dev1,
                buffers.expert1[slot].as_ptr(),
                HIDDEN * 4,
                &[],
                &result,
                Some(COPY_TO_DENSE_ENGINE),
            )?;
        }
        result.wait_timeout(Duration::from_secs(30))?;

        hip.set_device(0)?;
        launch_join(
            hip,
            &functions.join,
            &buffers.shared0[slot],
            &buffers.routed0[slot],
            &buffers.state0[(layer + 1) % 2],
            layer as u32,
        )?;
        hip.device_synchronize()?;
    }
    Ok(started.elapsed().as_secs_f64() * 1000.0)
}

fn sync_only_sample(
    dev0: &GpuDevice,
    dev1: &GpuDevice,
    buffers: &Buffers,
    signals: &mut [CompletionSignal],
) -> Result<f64, Box<dyn std::error::Error>> {
    for signal in signals.iter_mut() {
        signal.reset();
    }
    let started = Instant::now();
    for copy in 0..signals.len() {
        let dependency = if copy == 0 {
            Vec::new()
        } else {
            vec![&signals[copy - 1]]
        };
        if copy % 2 == 0 {
            // SAFETY: one live byte is copied in each direction; signals are
            // persistent for the entire chain.
            unsafe {
                dev1.memory_async_copy(
                    buffers.packet1[0].as_ptr(),
                    dev0,
                    buffers.packet0[0].as_ptr(),
                    1,
                    &dependency,
                    &signals[copy],
                    Some(COPY_TO_EXPERT_ENGINE),
                )?;
            }
        } else {
            // SAFETY: same contract in the reverse direction.
            unsafe {
                dev0.memory_async_copy(
                    buffers.packet0[1].as_ptr(),
                    dev1,
                    buffers.packet1[0].as_ptr(),
                    1,
                    &dependency,
                    &signals[copy],
                    Some(COPY_TO_DENSE_ENGINE),
                )?;
            }
        }
    }
    signals[signals.len() - 1].wait_timeout(Duration::from_secs(30))?;
    Ok(started.elapsed().as_secs_f64() * 1000.0)
}

fn p50(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn compile_exact(
    arch: &str,
    module: &str,
) -> Result<(Arc<[u8]>, String), Box<dyn std::error::Error>> {
    let mut compiler = KernelCompiler::new(arch, "-mcode-object-version=6".to_owned())?;
    let artifact = compiler.compile(module, SOURCE)?.to_owned();
    let image: Arc<[u8]> = Arc::from(std::fs::read(&artifact)?);
    Ok((image, artifact.display().to_string()))
}

#[allow(clippy::too_many_arguments)]
fn run_prefill(
    config: &Config,
    batch: usize,
    hip: &HipRuntime,
    runtime: &Runtime,
    dev0: &GpuDevice,
    dev1: &GpuDevice,
    kernels: &KernelSet,
    functions: &HipFunctions,
    artifact0: &str,
    artifact1: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let shared_weights = make_words(HIDDEN * config.prefill_shared_k, 0x1357_9bdf);
    let expert_weights = make_words(HIDDEN * config.prefill_expert_k, 0x2468_ace0);
    let initial = make_words(batch * HIDDEN, 0xa5a5_5a5a ^ batch as u32);
    let buffers =
        PrefillBuffers::allocate(hip, batch, config.prefill_shared_k, config.prefill_expert_k)?;
    buffers.upload_weights(hip, &shared_weights, &expert_weights)?;
    drop(shared_weights);
    drop(expert_weights);

    buffers.initialize(hip, &initial)?;
    let expected = single_device_prefill_oracle(
        hip,
        functions,
        &buffers,
        config.layers,
        batch,
        config.prefill_shared_k,
        config.prefill_expert_k,
    )?;
    println!(
        "prefill_identity arch0={} pci0={} hsaco0={} arch1={} pci1={} hsaco1={} layers={} batch={} hidden={} shared_k={} expert_k={} activation_bytes_per_layer={} result_bytes_per_layer={} persistent_device_allocations=15 persistent_kernargs={} persistent_completion_signals={} persistent_queues=2 parity_slots=2 oracle=single_gfx1100",
        dev0.name(),
        dev0.pci_bus_id(),
        artifact0,
        dev1.name(),
        dev1.pci_bus_id(),
        artifact1,
        config.layers,
        batch,
        HIDDEN,
        config.prefill_shared_k,
        config.prefill_expert_k,
        buffers.packet_words * 4,
        buffers.elements * 4,
        config.layers * 4,
        config.layers * 6,
    );

    let mut serial_p50 = None;
    for schedule in [Schedule::Serial, Schedule::Overlap] {
        let mut graph = PrefillGraph::new(
            schedule,
            runtime,
            dev0,
            dev1,
            kernels,
            &buffers,
            config.layers,
            batch,
            config.prefill_shared_k,
            config.prefill_expert_k,
        )?;
        for _ in 0..config.warmups {
            buffers.initialize(hip, &initial)?;
            graph.run(&buffers)?;
            buffers.assert_output(hip, config.layers, &expected, schedule.label())?;
        }
        let mut samples = Vec::with_capacity(config.samples);
        let mut last_timeline = None;
        let mut enqueue_samples = Vec::with_capacity(config.samples);
        for sample in 0..config.samples {
            buffers.initialize(hip, &initial)?;
            let run = graph.run(&buffers)?;
            buffers.assert_output(hip, config.layers, &expected, schedule.label())?;
            let timeline = graph.timeline_report()?;
            println!(
                "prefill_arm={} batch={} sample={} ms={:.6} host_enqueue_us={:.3} shared_us={:.3} expert_us={:.3} activation_copy_us={:.3} result_copy_us={:.3} shared_expert_overlap_us={:.3} activation_shared_overlap_us={:.3} overlap_layers={}/{} producer_shared_gap_us={:.3} producer_activation_gap_us={:.3} activation_expert_gap_us={:.3} expert_result_gap_us={:.3} result_join_gap_us={:.3} shared_join_gap_us={:.3} pcie_bytes={} pcie_transactions={} raw_bits=PASS",
                schedule.label(),
                batch,
                sample + 1,
                run.wall_ms,
                run.host_enqueue_us,
                timeline.shared_us,
                timeline.expert_us,
                timeline.activation_copy_us,
                timeline.result_copy_us,
                timeline.shared_expert_overlap_us,
                timeline.activation_shared_overlap_us,
                timeline.overlap_layers,
                config.layers,
                timeline.producer_to_shared_gap_us,
                timeline.producer_to_activation_gap_us,
                timeline.activation_to_expert_gap_us,
                timeline.expert_to_result_gap_us,
                timeline.result_to_join_gap_us,
                timeline.shared_to_join_gap_us,
                graph.pcie_bytes_per_graph(),
                graph.pcie_transactions_per_graph(),
            );
            samples.push(run.wall_ms);
            enqueue_samples.push(run.host_enqueue_us);
            last_timeline = Some(timeline);
        }
        let median = p50(&samples);
        let enqueue_p50 = p50(&enqueue_samples);
        let tokens_per_second = batch as f64 * 1000.0 / median;
        let timeline = last_timeline.expect("at least one prefill sample");
        println!(
            "prefill_arm={} batch={} samples={} p50_ms={median:.6} host_enqueue_p50_us={enqueue_p50:.3} rows_per_second={tokens_per_second:.3} overlap_us={:.3} overlap_layers={}/{}",
            schedule.label(),
            batch,
            samples.len(),
            timeline.shared_expert_overlap_us,
            timeline.overlap_layers,
            config.layers,
        );
        match schedule {
            Schedule::Serial => serial_p50 = Some(median),
            Schedule::Overlap => {
                if timeline.overlap_layers == 0 {
                    return Err("prefill overlap arm showed zero shared/expert overlap".into());
                }
                let serial = serial_p50.expect("prefill serial arm runs first");
                let speedup = serial / median;
                println!(
                    "prefill_gate batch={} vs_serial_speedup={speedup:.6} raw_bits=PASS no_host_wait_inside_graph=true stable_allocations=true",
                    batch
                );
                if speedup < 1.02 {
                    return Err(format!(
                        "prefill overlap does not beat serial scheduling by 2%: {speedup:.4}x"
                    )
                    .into());
                }
            }
        }
    }
    if let Some(depth) = config.prefill_depth {
        let chunks = depth.div_ceil(batch);
        let processed_rows = chunks * batch;
        let mut graph = PrefillGraph::new(
            Schedule::Overlap,
            runtime,
            dev0,
            dev1,
            kernels,
            &buffers,
            config.layers,
            batch,
            config.prefill_shared_k,
            config.prefill_expert_k,
        )?;
        buffers.initialize(hip, &initial)?;
        let mut depth_expected = Vec::new();
        for _ in 0..chunks {
            depth_expected = single_device_prefill_oracle(
                hip,
                functions,
                &buffers,
                config.layers,
                batch,
                config.prefill_shared_k,
                config.prefill_expert_k,
            )?;
        }
        buffers.initialize(hip, &initial)?;
        graph.run(&buffers)?;
        buffers.assert_output(hip, config.layers, &expected, "prefill_depth_warmup")?;
        buffers.initialize(hip, &initial)?;
        let started = Instant::now();
        let mut host_enqueue_us = 0.0;
        for _ in 0..chunks {
            host_enqueue_us += graph.run(&buffers)?.host_enqueue_us;
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        buffers.assert_output(hip, config.layers, &depth_expected, "prefill_depth")?;
        let timeline = graph.timeline_report()?;
        let rows_per_second = depth as f64 * 1000.0 / elapsed_ms;
        println!(
            "prefill_depth batch={} requested_rows={} processed_rows={} chunks={} elapsed_ms={elapsed_ms:.6} requested_rows_per_second={rows_per_second:.3} host_enqueue_us={host_enqueue_us:.3} pcie_bytes={} pcie_transactions={} last_chunk_overlap_us={:.3} last_chunk_overlap_layers={}/{} terminal_chunk_waits={} host_wait_inside_graph=false raw_bits=PASS stable_allocations=true",
            batch,
            depth,
            processed_rows,
            chunks,
            graph.pcie_bytes_per_graph() * chunks,
            graph.pcie_transactions_per_graph() * chunks,
            timeline.shared_expert_overlap_us,
            timeline.overlap_layers,
            config.layers,
            chunks,
        );
    }
    println!("hetero_gfx11_prefill batch={batch}: PASS");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse().map_err(|error| format!("hetero_gfx11_cooperative: {error}"))?;
    let hip = HipRuntime::load()?;
    if hip.device_count()? < 2 {
        return Err("two visible GPUs are required".into());
    }
    let arch0 = hip.get_arch(0)?;
    let arch1 = hip.get_arch(1)?;
    if arch0 != config.expect_arch0 || arch1 != config.expect_arch1 {
        return Err(format!(
            "logical GPU identity mismatch: device0={arch0} expected {}, device1={arch1} expected {}",
            config.expect_arch0, config.expect_arch1
        )
        .into());
    }
    hip.set_device(0)?;
    if !hip.can_access_peer(0, 1)? {
        return Err("gfx1100 cannot access gfx1151 peer allocations".into());
    }
    hip.enable_peer_access(1)?;
    hip.set_device(1)?;
    if !hip.can_access_peer(1, 0)? {
        return Err("gfx1151 cannot access gfx1100 peer allocations".into());
    }
    hip.enable_peer_access(0)?;

    let runtime = Runtime::initialize(redline_rocr::load_symbols()?)?;
    runtime.set_async_copy_profiling(true)?;
    let dev0 = runtime.select_gpu(GpuSelector::NameContains(&config.expect_arch0))?;
    let dev1 = runtime.select_gpu(GpuSelector::NameContains(&config.expect_arch1))?;
    let mask01 = dev1.copy_engine_mask(&dev0)?;
    let mask10 = dev0.copy_engine_mask(&dev1)?;
    if mask01 & COPY_TO_EXPERT_ENGINE == 0 || mask10 & COPY_TO_DENSE_ENGINE == 0 {
        return Err(format!(
            "selected G0 SDMA engines unavailable: 0->1 mask={mask01:#x}, 1->0 mask={mask10:#x}"
        )
        .into());
    }

    let (image0, artifact0) = compile_exact(&config.expect_arch0, "hetero_g1_gfx1100")?;
    let (image1, artifact1) = compile_exact(&config.expect_arch1, "hetero_g1_gfx1151")?;
    if Arc::ptr_eq(&image0, &image1) || artifact0 == artifact1 {
        return Err("exact-target code objects unexpectedly alias".into());
    }
    let executable0 = Executable::load(&dev0, image0.clone())?;
    let executable1 = Executable::load(&dev1, image1.clone())?;
    let kernels = KernelSet {
        producer: executable0.kernel("hetero_gfx11_producer.kd")?,
        shared: executable0.kernel("hetero_gfx11_shared.kd")?,
        expert: executable1.kernel("hetero_gfx11_expert.kd")?,
        join: executable0.kernel("hetero_gfx11_join.kd")?,
        prefill_producer: executable0.kernel("hetero_gfx11_prefill_producer.kd")?,
        prefill_shared: executable0.kernel("hetero_gfx11_prefill_shared.kd")?,
        prefill_expert: executable1.kernel("hetero_gfx11_prefill_expert.kd")?,
        prefill_join: executable0.kernel("hetero_gfx11_prefill_join.kd")?,
    };
    let hip_functions = HipFunctions::load(&hip, &image0, &image1)?;

    if let Some(batch) = config.prefill_batch {
        return run_prefill(
            &config,
            batch,
            &hip,
            &runtime,
            &dev0,
            &dev1,
            &kernels,
            &hip_functions,
            &artifact0,
            &artifact1,
        );
    }

    let shared_bytes = config.shared_mib * 1024 * 1024;
    let expert_bytes = config.expert_mib * 1024 * 1024;
    if !shared_bytes.is_multiple_of(4) || !expert_bytes.is_multiple_of(4) {
        return Err("weight byte counts must be divisible by four".into());
    }
    let shared_words = u32::try_from(shared_bytes / 4)?;
    let expert_words = u32::try_from(expert_bytes / 4)?;
    let shared_weights = make_words(shared_words as usize, 0x1357_9bdf);
    let expert_weights = make_words(expert_words as usize, 0x2468_ace0);
    let initial = make_words(HIDDEN, 0xa5a5_5a5a);
    let expected = oracle(
        &initial,
        &xor_by_hidden(&shared_weights),
        &xor_by_hidden(&expert_weights),
        config.layers,
    );
    let buffers = Buffers::allocate(&hip, shared_bytes, expert_bytes)?;
    hip.set_device(0)?;
    hip.memcpy_htod(&buffers.shared_weights0, u32_bytes(&shared_weights))?;
    hip.device_synchronize()?;
    hip.set_device(1)?;
    hip.memcpy_htod(&buffers.expert_weights1, u32_bytes(&expert_weights))?;
    hip.device_synchronize()?;
    drop(shared_weights);
    drop(expert_weights);

    println!(
        "identity arch0={} pci0={} hsaco0={} arch1={} pci1={} hsaco1={} layers={} hidden={} routes={} shared_mib={} expert_mib={} engine01={:#x} engine10={:#x}",
        arch0,
        dev0.pci_bus_id(),
        artifact0,
        arch1,
        dev1.pci_bus_id(),
        artifact1,
        config.layers,
        HIDDEN,
        ROUTES,
        config.shared_mib,
        config.expert_mib,
        COPY_TO_EXPERT_ENGINE,
        COPY_TO_DENSE_ENGINE,
    );

    let mut sync_signals = Vec::with_capacity(config.layers * 2);
    for copy in 0..config.layers * 2 {
        sync_signals.push(CompletionSignal::new(if copy % 2 == 0 {
            &dev1
        } else {
            &dev0
        })?);
    }
    let mut sync_samples = Vec::with_capacity(config.sync_samples);
    for _ in 0..config.sync_samples {
        sync_samples.push(sync_only_sample(&dev0, &dev1, &buffers, &mut sync_signals)?);
    }
    println!(
        "arm=sync_only_one_byte samples={} p50_ms={:.6}",
        sync_samples.len(),
        p50(&sync_samples)
    );

    let mut host_samples = Vec::with_capacity(config.host_samples);
    for sample in 0..config.host_samples {
        buffers.initialize(&hip, &initial)?;
        let elapsed = host_sync_sample(
            &hip,
            &hip_functions,
            &dev0,
            &dev1,
            &buffers,
            config.layers,
            shared_words,
            expert_words,
        )?;
        buffers.assert_output(&hip, config.layers, &expected, "host_sync")?;
        println!(
            "arm=host_sync sample={} ms={elapsed:.6} raw_bits=PASS",
            sample + 1
        );
        host_samples.push(elapsed);
    }
    let host_p50 = p50(&host_samples);
    println!(
        "arm=host_sync samples={} p50_ms={host_p50:.6}",
        host_samples.len()
    );

    let mut serial_p50 = None;
    for schedule in [Schedule::Serial, Schedule::Overlap] {
        let mut graph = Graph::new(
            schedule,
            &runtime,
            &dev0,
            &dev1,
            &kernels,
            &buffers,
            config.layers,
            shared_words,
            expert_words,
        )?;
        for _ in 0..config.warmups {
            buffers.initialize(&hip, &initial)?;
            graph.run(&buffers)?;
            buffers.assert_output(&hip, config.layers, &expected, schedule.label())?;
        }
        let mut samples = Vec::with_capacity(config.samples);
        let mut last_overlap = None;
        for sample in 0..config.samples {
            buffers.initialize(&hip, &initial)?;
            let run = graph.run(&buffers)?;
            buffers.assert_output(&hip, config.layers, &expected, schedule.label())?;
            let timeline = graph.timeline_report()?;
            println!(
                "arm={} sample={} ms={:.6} host_enqueue_us={:.3} shared_us={:.3} expert_us={:.3} activation_copy_us={:.3} result_copy_us={:.3} shared_expert_overlap_us={:.3} activation_shared_overlap_us={:.3} overlap_layers={}/{} producer_shared_gap_us={:.3} producer_activation_gap_us={:.3} activation_expert_gap_us={:.3} expert_result_gap_us={:.3} result_join_gap_us={:.3} shared_join_gap_us={:.3} raw_bits=PASS",
                schedule.label(),
                sample + 1,
                run.wall_ms,
                run.host_enqueue_us,
                timeline.shared_us,
                timeline.expert_us,
                timeline.activation_copy_us,
                timeline.result_copy_us,
                timeline.shared_expert_overlap_us,
                timeline.activation_shared_overlap_us,
                timeline.overlap_layers,
                config.layers,
                timeline.producer_to_shared_gap_us,
                timeline.producer_to_activation_gap_us,
                timeline.activation_to_expert_gap_us,
                timeline.expert_to_result_gap_us,
                timeline.result_to_join_gap_us,
                timeline.shared_to_join_gap_us,
            );
            samples.push(run.wall_ms);
            last_overlap = Some(timeline);
        }
        let median = p50(&samples);
        let speedup = host_p50 / median;
        let overlap = last_overlap.expect("at least one sample");
        println!(
            "arm={} samples={} p50_ms={median:.6} vs_host_speedup={speedup:.6} overlap_us={:.3} overlap_layers={}/{}",
            schedule.label(),
            samples.len(),
            overlap.shared_expert_overlap_us,
            overlap.overlap_layers,
            config.layers,
        );
        if schedule == Schedule::Overlap && overlap.overlap_layers == 0 {
            return Err("device-overlap arm showed zero shared/expert dispatch overlap".into());
        }
        match schedule {
            Schedule::Serial => serial_p50 = Some(median),
            Schedule::Overlap => {
                let serial = serial_p50.expect("serial arm runs first");
                let vs_serial = serial / median;
                println!(
                    "gate=device_overlap vs_serial_speedup={vs_serial:.6} vs_host_speedup={speedup:.6}"
                );
                if speedup < 1.05 {
                    return Err(format!(
                        "cooperative path is not materially faster than host-sync control: {speedup:.4}x"
                    )
                    .into());
                }
                if vs_serial < 1.02 {
                    return Err(format!(
                        "overlap does not beat device-serial scheduling by 2%: {vs_serial:.4}x"
                    )
                    .into());
                }
            }
        }
    }

    println!("hetero_gfx11_cooperative: PASS");
    Ok(())
}
