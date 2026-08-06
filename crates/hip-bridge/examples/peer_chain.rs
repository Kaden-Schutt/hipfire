// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Persistent bidirectional HIP peer-channel benchmark.
//!
//! This is the H0 transport gate for the DeepSeek V4 heterogeneous
//! gfx1100/gfx1151 route. It measures the actual decode boundary shape:
//! one `[batch, hidden=4096]` F32 payload in each direction across a
//! 43-layer dependency chain. Streams, timing events, and buffers are
//! allocated once and reused for every sample. The default synchronization
//! mode uses GPU-visible signal memory; host stream synchronization remains a
//! correctness control. Host/device synchronization appears only outside the
//! timed chain to order initialization and at the terminal timing event.
//!
//! Example:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 cargo run --release -p hip-bridge \
//!   --example peer_chain -- \
//!   --expect-arch0 gfx1100 --expect-arch1 gfx1151
//! ```

use hip_bridge::{
    DeviceBuffer, Event, HipError, HipResult, HipRuntime, RcclComms, RcclDataType, Stream,
    HIP_EVENT_DISABLE_TIMING, HIP_EVENT_RELEASE_TO_SYSTEM,
};
use redline_rocr::packet::{BarrierAndPacket, PacketImage};
use redline_rocr::{CompletionSignal, GpuDevice, GpuSelector, QueueSet, Runtime};
use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

const HIDDEN: usize = 4096;
const F32_BYTES: usize = 4;
const SIGNAL_BYTES: usize = std::mem::size_of::<u64>();
const DEFAULT_LAYERS: usize = 43;
const DEFAULT_BATCHES: &[usize] = &[1, 16, 128, 512, 1024];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SyncMode {
    Event,
    Host,
    Rccl,
    Rocr,
    RocrAql,
    RocrSdma,
    Signal,
}

impl SyncMode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "event" => Ok(Self::Event),
            "host" => Ok(Self::Host),
            "rccl" => Ok(Self::Rccl),
            "rocr" => Ok(Self::Rocr),
            "rocr-aql" => Ok(Self::RocrAql),
            "rocr-sdma" => Ok(Self::RocrSdma),
            "signal" => Ok(Self::Signal),
            _ => Err(format!(
                "unsupported --sync value {raw:?}; use event, signal, rccl, rocr, rocr-aql, rocr-sdma, or host"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Event => "system_scope_event",
            Self::Host => "host_stream_sync",
            Self::Rccl => "rccl_grouped_p2p",
            Self::Rocr => "rocr_async_copy_auto",
            Self::RocrAql => "rocr_sdma_dual_aql",
            Self::RocrSdma => "rocr_async_copy_sdma",
            Self::Signal => "signal_memory",
        }
    }

    fn is_rocr(self) -> bool {
        matches!(self, Self::Rocr | Self::RocrAql | Self::RocrSdma)
    }

    fn supports_sync_only(self) -> bool {
        !self.is_rocr() || self == Self::RocrAql
    }
}

#[derive(Debug)]
struct Config {
    warmups: usize,
    one_way_samples: usize,
    chain_samples: usize,
    exactness_samples: usize,
    exactness_seed_base: u64,
    sync: SyncMode,
    layers: usize,
    batches: Vec<usize>,
    expect_arch0: Option<String>,
    expect_arch1: Option<String>,
    rocr_engine_0_to_1: Option<u32>,
    rocr_engine_1_to_0: Option<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            warmups: 10,
            one_way_samples: 100,
            chain_samples: 50,
            exactness_samples: 10,
            exactness_seed_base: 0,
            sync: SyncMode::Signal,
            layers: DEFAULT_LAYERS,
            batches: DEFAULT_BATCHES.to_vec(),
            expect_arch0: None,
            expect_arch1: None,
            rocr_engine_0_to_1: None,
            rocr_engine_1_to_0: None,
        }
    }
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut cfg = Self::default();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let flag = &args[i];
            let value = |i: &mut usize| -> Result<&str, String> {
                *i += 1;
                args.get(*i)
                    .map(String::as_str)
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            match flag.as_str() {
                "--warmups" => cfg.warmups = parse_positive(flag, value(&mut i)?)?,
                "--one-way-samples" => cfg.one_way_samples = parse_positive(flag, value(&mut i)?)?,
                "--chain-samples" => cfg.chain_samples = parse_positive(flag, value(&mut i)?)?,
                "--exactness-samples" => {
                    cfg.exactness_samples = parse_positive(flag, value(&mut i)?)?
                }
                "--exactness-seed-base" => {
                    cfg.exactness_seed_base = parse_u64(flag, value(&mut i)?)?
                }
                "--sync" => cfg.sync = SyncMode::parse(value(&mut i)?)?,
                "--layers" => cfg.layers = parse_positive(flag, value(&mut i)?)?,
                "--batches" => {
                    cfg.batches = value(&mut i)?
                        .split(',')
                        .map(|raw| parse_positive("--batches", raw))
                        .collect::<Result<Vec<_>, _>>()?;
                    cfg.batches.sort_unstable();
                    cfg.batches.dedup();
                }
                "--expect-arch0" => cfg.expect_arch0 = Some(value(&mut i)?.to_string()),
                "--expect-arch1" => cfg.expect_arch1 = Some(value(&mut i)?.to_string()),
                "--rocr-engine-0-to-1" => {
                    cfg.rocr_engine_0_to_1 = Some(parse_u32_auto(flag, value(&mut i)?)?)
                }
                "--rocr-engine-1-to-0" => {
                    cfg.rocr_engine_1_to_0 = Some(parse_u32_auto(flag, value(&mut i)?)?)
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {flag:?}; use --help")),
            }
            i += 1;
        }
        if cfg.batches.is_empty() {
            return Err("--batches must contain at least one positive value".to_string());
        }
        if (cfg.rocr_engine_0_to_1.is_some() || cfg.rocr_engine_1_to_0.is_some())
            && !matches!(cfg.sync, SyncMode::RocrSdma | SyncMode::RocrAql)
        {
            return Err("ROCr engine overrides require --sync rocr-sdma".to_string());
        }
        Ok(cfg)
    }
}

fn parse_positive(flag: &str, raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|e| format!("invalid {flag} value {raw:?}: {e}"))?;
    if value == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(value)
}

fn parse_u64(flag: &str, raw: &str) -> Result<u64, String> {
    raw.parse::<u64>()
        .map_err(|e| format!("invalid {flag} value {raw:?}: {e}"))
}

fn parse_u32_auto(flag: &str, raw: &str) -> Result<u32, String> {
    let parsed = raw
        .strip_prefix("0x")
        .map_or_else(|| raw.parse::<u32>(), |hex| u32::from_str_radix(hex, 16))
        .map_err(|e| format!("invalid {flag} value {raw:?}: {e}"))?;
    if parsed == 0 || !parsed.is_power_of_two() {
        return Err(format!("{flag} must be one nonzero SDMA-engine bit"));
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "peer_chain [options]\n\
         \n\
         Options:\n\
           --warmups N            warm chain/one-way iterations (default 10)\n\
           --one-way-samples N    samples per direction and size (default 100)\n\
           --chain-samples N      43-layer chain samples per size (default 50)\n\
           --exactness-samples N  post-timing exactness stress samples (default 10)\n\
           --exactness-seed-base N first unique stress payload seed (default 0)\n\
           --sync MODE            signal, event, rccl, rocr, rocr-aql, rocr-sdma, or host\n\
           --layers N             round-trip dependency count (default 43)\n\
           --batches CSV          batch rows for [B,4096] F32 (default 1,16,128,512,1024)\n\
           --expect-arch0 ARCH    fail unless logical device 0 matches\n\
           --expect-arch1 ARCH    fail unless logical device 1 matches\n\
           --rocr-engine-0-to-1 N explicit SDMA engine bit (rocr-sdma only)\n\
           --rocr-engine-1-to-0 N explicit SDMA engine bit (rocr-sdma only)"
    );
}

#[derive(Clone, Copy, Debug)]
struct Distribution {
    min_us: f64,
    p50_us: f64,
    p95_us: f64,
    max_us: f64,
}

impl Distribution {
    fn from_ms(samples: &[f64]) -> Self {
        assert!(!samples.is_empty());
        let mut us: Vec<f64> = samples.iter().map(|ms| ms * 1000.0).collect();
        us.sort_by(f64::total_cmp);
        Self {
            min_us: us[0],
            p50_us: percentile(&us, 0.50),
            p95_us: percentile(&us, 0.95),
            max_us: us[us.len() - 1],
        }
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let rank = q * (sorted.len().saturating_sub(1)) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn pattern(size: usize, seed: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size);
    let mut state = seed;
    while bytes.len() < size {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut word = state;
        word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        word ^= word >> 31;
        let remaining = size - bytes.len();
        bytes.extend_from_slice(&word.to_le_bytes()[..remaining.min(8)]);
    }
    bytes
}

fn assert_bytes(
    hip: &HipRuntime,
    device: i32,
    actual: &DeviceBuffer,
    expected: &[u8],
    label: &str,
) -> HipResult<()> {
    hip.set_device(device)?;
    let mut got = vec![0u8; expected.len()];
    hip.memcpy_dtoh(&mut got, actual)?;
    if got != expected {
        let first = got
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "{label}: byte mismatch at {first}: got={} expected={}",
            got[first], expected[first]
        );
    }
    Ok(())
}

fn initialize_chain_inputs(
    hip: &HipRuntime,
    dev0_a: &DeviceBuffer,
    dev0_b: &DeviceBuffer,
    dev1: &DeviceBuffer,
    expected: &[u8],
) -> HipResult<()> {
    hip.set_device(0)?;
    hip.memcpy_htod(dev0_a, expected)?;
    hip.memset(dev0_b, 0, expected.len())?;
    hip.device_synchronize()?;

    hip.set_device(1)?;
    hip.memset(dev1, 0, expected.len())?;
    hip.device_synchronize()?;
    Ok(())
}

struct Direction<'a> {
    src_device: i32,
    dst_device: i32,
    src: &'a DeviceBuffer,
    dst: &'a DeviceBuffer,
    stream: &'a Stream,
    start: &'a Event,
    stop: &'a Event,
}

fn one_way_sample(hip: &HipRuntime, d: &Direction<'_>, size: usize) -> HipResult<(f64, f64)> {
    hip.set_device(d.src_device)?;
    let host_start = Instant::now();
    hip.event_record(d.start, Some(d.stream))?;
    hip.memcpy_peer_async(d.dst, d.dst_device, d.src, d.src_device, size, d.stream)?;
    hip.event_record(d.stop, Some(d.stream))?;
    hip.event_synchronize(d.stop)?;
    let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = hip.event_elapsed_ms(d.start, d.stop)? as f64;
    Ok((gpu_ms, host_ms))
}

struct Chain<'a> {
    stream0: &'a Stream,
    stream1: &'a Stream,
    start0: &'a Event,
    stop0: &'a Event,
    signal_to1: Option<&'a DeviceBuffer>,
    signal_to0: Option<&'a DeviceBuffer>,
    events_to1: &'a [Event],
    events_to0: &'a [Event],
    rccl: Option<&'a RcclComms>,
    rocr: Option<&'a RocrChannel>,
    dev0_a: &'a DeviceBuffer,
    dev0_b: &'a DeviceBuffer,
    dev1: &'a DeviceBuffer,
    layers: usize,
    next_epoch: Cell<u32>,
    sync: SyncMode,
}

struct RocrChannel {
    dev0: GpuDevice,
    dev1: GpuDevice,
    completions: RefCell<Vec<CompletionSignal>>,
    engine_0_to_1: Option<u32>,
    engine_1_to_0: Option<u32>,
    mask_0_to_1: u32,
    mask_1_to_0: u32,
    preferred_0_to_1: u32,
    preferred_1_to_0: u32,
    aql: Option<RocrAqlState>,
}

struct RocrAqlState {
    queues: RefCell<QueueSet>,
    barrier_completions: RefCell<Vec<CompletionSignal>>,
    sync_completions: RefCell<Vec<CompletionSignal>>,
    payload_batches: [Vec<PacketImage>; 2],
    sync_batches: [Vec<PacketImage>; 2],
}

impl RocrChannel {
    fn new(
        layers: usize,
        explicit_sdma: bool,
        dual_aql: bool,
        override_0_to_1: Option<u32>,
        override_1_to_0: Option<u32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = Runtime::initialize(redline_rocr::load_symbols()?)?;
        let dev0 = runtime.select_gpu(GpuSelector::NameContains("gfx1100"))?;
        let dev1 = runtime.select_gpu(GpuSelector::NameContains("gfx1151"))?;
        let mask_0_to_1 = dev1.copy_engine_mask(&dev0)?;
        let mask_1_to_0 = dev0.copy_engine_mask(&dev1)?;
        let preferred_0_to_1 = dev1.preferred_copy_engine_mask(&dev0)?;
        let preferred_1_to_0 = dev0.preferred_copy_engine_mask(&dev1)?;
        let choose = |available: u32, preferred: u32| -> Result<u32, String> {
            let candidates = available & preferred;
            let candidates = if candidates != 0 {
                candidates
            } else {
                available
            };
            if candidates == 0 {
                Err("ROCr reported no available SDMA copy engine".to_string())
            } else {
                Ok(1_u32 << candidates.trailing_zeros())
            }
        };
        let validate_override = |requested: Option<u32>, available: u32, label: &str| {
            if requested.is_some_and(|engine| available & engine == 0) {
                Err(format!(
                    "requested {label} engine {requested:?} is outside available mask {available:#x}"
                ))
            } else {
                Ok(requested)
            }
        };
        let engine_0_to_1 =
            validate_override(override_0_to_1, mask_0_to_1, "0->1")?.or(explicit_sdma
                .then(|| choose(mask_0_to_1, preferred_0_to_1))
                .transpose()?);
        let engine_1_to_0 =
            validate_override(override_1_to_0, mask_1_to_0, "1->0")?.or(explicit_sdma
                .then(|| choose(mask_1_to_0, preferred_1_to_0))
                .transpose()?);
        let mut completions = Vec::with_capacity(layers * 2);
        for index in 0..layers * 2 {
            let owner = if index % 2 == 0 { &dev1 } else { &dev0 };
            completions.push(CompletionSignal::new(owner)?);
        }
        let aql = if dual_aql {
            let mut barrier_completions = Vec::with_capacity(layers * 2);
            let mut sync_completions = Vec::with_capacity(layers * 2);
            for index in 0..layers * 2 {
                let owner = if index % 2 == 0 { &dev1 } else { &dev0 };
                barrier_completions.push(CompletionSignal::new(owner)?);
                sync_completions.push(CompletionSignal::new(owner)?);
            }
            let mut payload_batches = [Vec::with_capacity(layers), Vec::with_capacity(layers)];
            let mut sync_batches = [Vec::with_capacity(layers), Vec::with_capacity(layers)];
            for copy in 0..layers * 2 {
                let lane = if copy % 2 == 0 { 1 } else { 0 };
                let payload_packet = BarrierAndPacket::new(
                    &[completions[copy].raw()],
                    barrier_completions[copy].raw(),
                )?;
                payload_batches[lane].push(PacketImage::barrier(&payload_packet));
                let sync_dependencies = if copy == 0 {
                    Vec::new()
                } else {
                    vec![sync_completions[copy - 1].raw()]
                };
                let sync_packet =
                    BarrierAndPacket::new(&sync_dependencies, sync_completions[copy].raw())?;
                sync_batches[lane].push(PacketImage::barrier(&sync_packet));
            }
            Some(RocrAqlState {
                queues: RefCell::new(QueueSet::create_for_devices(
                    &[dev0.clone(), dev1.clone()],
                    256,
                )?),
                barrier_completions: RefCell::new(barrier_completions),
                sync_completions: RefCell::new(sync_completions),
                payload_batches,
                sync_batches,
            })
        } else {
            None
        };
        Ok(Self {
            dev0,
            dev1,
            completions: RefCell::new(completions),
            engine_0_to_1,
            engine_1_to_0,
            mask_0_to_1,
            mask_1_to_0,
            preferred_0_to_1,
            preferred_1_to_0,
            aql,
        })
    }

    fn identity(&self) -> String {
        format!(
            "rocr_dev0={} rocr_pci0={} rocr_dev1={} rocr_pci1={} \
             engine_mask_0_to_1={:#x} engine_mask_1_to_0={:#x} \
             preferred_0_to_1={:#x} preferred_1_to_0={:#x} \
             selected_0_to_1={:?} selected_1_to_0={:?}",
            self.dev0.name(),
            self.dev0.pci_bus_id(),
            self.dev1.name(),
            self.dev1.pci_bus_id(),
            self.mask_0_to_1,
            self.mask_1_to_0,
            self.preferred_0_to_1,
            self.preferred_1_to_0,
            self.engine_0_to_1,
            self.engine_1_to_0,
        )
    }

    fn one_way(
        &self,
        direction_0_to_1: bool,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
        size: usize,
    ) -> HipResult<f64> {
        let mut completions = self.completions.borrow_mut();
        let completion = &mut completions[0];
        completion.reset();
        let started = Instant::now();
        let result = if direction_0_to_1 {
            // SAFETY: HIP buffers are live peer-accessible allocations owned
            // by the matching HSA agents for the duration of this call.
            unsafe {
                self.dev1.memory_async_copy(
                    dst.as_ptr(),
                    &self.dev0,
                    src.as_ptr(),
                    size,
                    &[],
                    completion,
                    self.engine_0_to_1,
                )
            }
        } else {
            // SAFETY: same invariant, reversed direction.
            unsafe {
                self.dev0.memory_async_copy(
                    dst.as_ptr(),
                    &self.dev1,
                    src.as_ptr(),
                    size,
                    &[],
                    completion,
                    self.engine_1_to_0,
                )
            }
        };
        result.map_err(rocr_as_hip)?;
        completion
            .wait_timeout(Duration::from_secs(30))
            .map_err(rocr_as_hip)?;
        check_rocr_completion(completion)?;
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }

    fn chain(
        &self,
        dev0_a: &DeviceBuffer,
        dev0_b: &DeviceBuffer,
        dev1: &DeviceBuffer,
        layers: usize,
        size: usize,
        copy_payload: bool,
    ) -> HipResult<f64> {
        if let Some(aql) = &self.aql {
            return self.aql_chain(aql, dev0_a, dev0_b, dev1, layers, size, copy_payload);
        }
        if !copy_payload {
            return Err(HipError {
                code: 1,
                message: "raw ROCr has no payload-free dependency chain".to_string(),
            });
        }
        let mut completions = self.completions.borrow_mut();
        for signal in completions.iter_mut().take(layers * 2) {
            signal.reset();
        }
        let started = Instant::now();
        for copy in 0..layers * 2 {
            let layer = copy / 2;
            let (src0, dst0) = if layer % 2 == 0 {
                (dev0_a, dev0_b)
            } else {
                (dev0_b, dev0_a)
            };
            let deps: Vec<&CompletionSignal> = if copy == 0 {
                Vec::new()
            } else {
                vec![&completions[copy - 1]]
            };
            let completion = &completions[copy];
            let result = if copy % 2 == 0 {
                // SAFETY: live peer-accessible HIP allocations are associated
                // with the matching agents; dependency signals remain live.
                unsafe {
                    self.dev1.memory_async_copy(
                        dev1.as_ptr(),
                        &self.dev0,
                        src0.as_ptr(),
                        size,
                        &deps,
                        completion,
                        self.engine_0_to_1,
                    )
                }
            } else {
                // SAFETY: same invariant, reversed direction.
                unsafe {
                    self.dev0.memory_async_copy(
                        dst0.as_ptr(),
                        &self.dev1,
                        dev1.as_ptr(),
                        size,
                        &deps,
                        completion,
                        self.engine_1_to_0,
                    )
                }
            };
            result.map_err(rocr_as_hip)?;
        }
        let terminal = &completions[layers * 2 - 1];
        terminal
            .wait_timeout(Duration::from_secs(30))
            .map_err(rocr_as_hip)?;
        for completion in completions.iter().take(layers * 2) {
            check_rocr_completion(completion)?;
        }
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }

    fn aql_chain(
        &self,
        aql: &RocrAqlState,
        dev0_a: &DeviceBuffer,
        dev0_b: &DeviceBuffer,
        dev1: &DeviceBuffer,
        layers: usize,
        size: usize,
        copy_payload: bool,
    ) -> HipResult<f64> {
        let started = Instant::now();
        if !copy_payload {
            let mut sync = aql.sync_completions.borrow_mut();
            for signal in sync.iter_mut().take(layers * 2) {
                signal.reset();
            }
            let mut queues = aql.queues.borrow_mut();
            queues
                .prepare_batches(&aql.sync_batches)
                .map_err(rocr_as_hip)?;
            queues.ring_prepared().map_err(rocr_as_hip)?;
            sync[layers * 2 - 1]
                .wait_timeout(Duration::from_secs(30))
                .map_err(rocr_as_hip)?;
            for completion in sync.iter().take(layers * 2) {
                check_rocr_completion(completion)?;
            }
            return Ok(started.elapsed().as_secs_f64() * 1000.0);
        }

        let mut copies = self.completions.borrow_mut();
        let mut barriers = aql.barrier_completions.borrow_mut();
        for signal in copies.iter_mut().take(layers * 2) {
            signal.reset();
        }
        for signal in barriers.iter_mut().take(layers * 2) {
            signal.reset();
        }
        for copy in 0..layers * 2 {
            let layer = copy / 2;
            let (src0, dst0) = if layer % 2 == 0 {
                (dev0_a, dev0_b)
            } else {
                (dev0_b, dev0_a)
            };
            let dependencies: Vec<&CompletionSignal> = if copy == 0 {
                Vec::new()
            } else {
                vec![&barriers[copy - 1]]
            };
            let result = if copy % 2 == 0 {
                // SAFETY: the AQL barrier on the destination agent performs a
                // System acquire before publishing the next dependency.
                unsafe {
                    self.dev1.memory_async_copy(
                        dev1.as_ptr(),
                        &self.dev0,
                        src0.as_ptr(),
                        size,
                        &dependencies,
                        &copies[copy],
                        self.engine_0_to_1,
                    )
                }
            } else {
                // SAFETY: same ownership and system-scope contract reversed.
                unsafe {
                    self.dev0.memory_async_copy(
                        dst0.as_ptr(),
                        &self.dev1,
                        dev1.as_ptr(),
                        size,
                        &dependencies,
                        &copies[copy],
                        self.engine_1_to_0,
                    )
                }
            };
            result.map_err(rocr_as_hip)?;
        }
        let mut queues = aql.queues.borrow_mut();
        queues
            .prepare_batches(&aql.payload_batches)
            .map_err(rocr_as_hip)?;
        queues.ring_prepared().map_err(rocr_as_hip)?;
        barriers[layers * 2 - 1]
            .wait_timeout(Duration::from_secs(30))
            .map_err(rocr_as_hip)?;
        for completion in copies.iter().take(layers * 2) {
            check_rocr_completion(completion)?;
        }
        for completion in barriers.iter().take(layers * 2) {
            check_rocr_completion(completion)?;
        }
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }
}

fn check_rocr_completion(signal: &CompletionSignal) -> HipResult<()> {
    let value = signal.value_scacquire();
    if value == 0 {
        Ok(())
    } else {
        Err(HipError {
            code: 1,
            message: format!("ROCr async-copy completion signal ended at {value}, expected 0"),
        })
    }
}

fn rocr_as_hip(error: redline_rocr::RuntimeError) -> HipError {
    HipError {
        code: 1,
        message: error.to_string(),
    }
}

fn chain_sample(
    hip: &HipRuntime,
    chain: &Chain<'_>,
    size: usize,
    copy_payload: bool,
) -> HipResult<(f64, f64)> {
    const WAIT_EQ: u32 = 0x1;
    const SIGNAL_FLAGS: u32 = 0;
    const SIGNAL_MASK: u32 = u32::MAX;

    let first_epoch = chain.next_epoch.get();
    let final_epoch = first_epoch
        .checked_add(chain.layers as u32)
        .expect("peer-chain signal epoch overflow");
    hip.set_device(0)?;
    let host_start = Instant::now();

    if chain.sync.is_rocr() {
        let rocr = chain.rocr.expect("ROCr mode requires channel");
        let wall_ms = rocr.chain(
            chain.dev0_a,
            chain.dev0_b,
            chain.dev1,
            chain.layers,
            size,
            copy_payload,
        )?;
        return Ok((wall_ms, wall_ms));
    }
    hip.event_record(chain.start0, Some(chain.stream0))?;

    if chain.sync == SyncMode::Rccl {
        let rccl = chain.rccl.expect("RCCL mode requires communicators");
        rccl.group_start().map_err(rccl_as_hip)?;
        let body = (|| -> HipResult<()> {
            for layer in 0..chain.layers {
                let (src0, dst0) = if layer % 2 == 0 {
                    (chain.dev0_a, chain.dev0_b)
                } else {
                    (chain.dev0_b, chain.dev0_a)
                };
                if copy_payload {
                    let count = size / F32_BYTES;
                    hip.set_device(0)?;
                    unsafe {
                        rccl.send(
                            0,
                            src0.as_ptr(),
                            count,
                            RcclDataType::Float32,
                            1,
                            chain.stream0.as_raw(),
                        )
                    }
                    .map_err(rccl_as_hip)?;
                    hip.set_device(1)?;
                    unsafe {
                        rccl.recv(
                            1,
                            chain.dev1.as_ptr(),
                            count,
                            RcclDataType::Float32,
                            0,
                            chain.stream1.as_raw(),
                        )
                    }
                    .map_err(rccl_as_hip)?;
                    unsafe {
                        rccl.send(
                            1,
                            chain.dev1.as_ptr(),
                            count,
                            RcclDataType::Float32,
                            0,
                            chain.stream1.as_raw(),
                        )
                    }
                    .map_err(rccl_as_hip)?;
                    hip.set_device(0)?;
                    unsafe {
                        rccl.recv(
                            0,
                            dst0.as_ptr(),
                            count,
                            RcclDataType::Float32,
                            1,
                            chain.stream0.as_raw(),
                        )
                    }
                    .map_err(rccl_as_hip)?;
                }
            }
            Ok(())
        })();
        let close = rccl.group_end().map_err(rccl_as_hip);
        body?;
        close?;

        hip.set_device(0)?;
        hip.event_record(chain.stop0, Some(chain.stream0))?;
        hip.event_synchronize(chain.stop0)?;
        let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = hip.event_elapsed_ms(chain.start0, chain.stop0)? as f64;
        return Ok((gpu_ms, host_ms));
    }

    for layer in 0..chain.layers {
        let epoch = first_epoch + layer as u32 + 1;
        let (src0, dst0) = if layer % 2 == 0 {
            (chain.dev0_a, chain.dev0_b)
        } else {
            (chain.dev0_b, chain.dev0_a)
        };

        match chain.sync {
            SyncMode::Event => {
                let to1 = &chain.events_to1[layer];
                let to0 = &chain.events_to0[layer];
                hip.set_device(0)?;
                if copy_payload {
                    hip.memcpy_peer_async(chain.dev1, 1, src0, 0, size, chain.stream0)?;
                }
                hip.event_record(to1, Some(chain.stream0))?;

                hip.set_device(1)?;
                hip.stream_wait_event(chain.stream1, to1)?;
                if copy_payload {
                    hip.memcpy_peer_async(dst0, 0, chain.dev1, 1, size, chain.stream1)?;
                }
                hip.event_record(to0, Some(chain.stream1))?;

                hip.set_device(0)?;
                hip.stream_wait_event(chain.stream0, to0)?;
            }
            SyncMode::Host => {
                hip.set_device(0)?;
                if copy_payload {
                    hip.memcpy_peer_async(chain.dev1, 1, src0, 0, size, chain.stream0)?;
                }
                hip.stream_synchronize(chain.stream0)?;

                hip.set_device(1)?;
                if copy_payload {
                    hip.memcpy_peer_async(dst0, 0, chain.dev1, 1, size, chain.stream1)?;
                }
                hip.stream_synchronize(chain.stream1)?;
            }
            SyncMode::Rccl => unreachable!("RCCL chain is handled as one persistent group"),
            SyncMode::Rocr | SyncMode::RocrAql | SyncMode::RocrSdma => {
                unreachable!("ROCr chain is handled before HIP stream setup")
            }
            SyncMode::Signal => {
                let signal_to1 = chain.signal_to1.expect("signal mode requires to1 memory");
                let signal_to0 = chain.signal_to0.expect("signal mode requires to0 memory");
                hip.set_device(0)?;
                if copy_payload {
                    hip.memcpy_peer_async(chain.dev1, 1, src0, 0, size, chain.stream0)?;
                }
                hip.stream_write_value32(chain.stream0, signal_to1, epoch, SIGNAL_FLAGS)?;

                hip.set_device(1)?;
                hip.stream_wait_value32(chain.stream1, signal_to1, epoch, WAIT_EQ, SIGNAL_MASK)?;
                if copy_payload {
                    hip.memcpy_peer_async(dst0, 0, chain.dev1, 1, size, chain.stream1)?;
                }
                hip.stream_write_value32(chain.stream1, signal_to0, epoch, SIGNAL_FLAGS)?;

                hip.set_device(0)?;
                hip.stream_wait_value32(chain.stream0, signal_to0, epoch, WAIT_EQ, SIGNAL_MASK)?;
            }
        }
    }

    hip.event_record(chain.stop0, Some(chain.stream0))?;
    hip.event_synchronize(chain.stop0)?;
    chain.next_epoch.set(final_epoch);
    let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = hip.event_elapsed_ms(chain.start0, chain.stop0)? as f64;
    Ok((gpu_ms, host_ms))
}

fn rccl_as_hip(error: hip_bridge::RcclError) -> HipError {
    HipError {
        code: error.status,
        message: error.to_string(),
    }
}

fn print_row(
    kind: &str,
    direction: &str,
    batch: usize,
    bytes_per_copy: usize,
    copies: usize,
    gpu: Distribution,
    host: Distribution,
) {
    let total_bytes = bytes_per_copy.saturating_mul(copies);
    let gbps_at_p50 = if gpu.p50_us > 0.0 {
        total_bytes as f64 / (gpu.p50_us * 1000.0)
    } else {
        f64::INFINITY
    };
    println!(
        "row kind={kind} direction={direction} batch={batch} bytes_per_copy={bytes_per_copy} \
         copies={copies} total_bytes={total_bytes} gpu_min_us={:.3} gpu_p50_us={:.3} \
         gpu_p95_us={:.3} gpu_max_us={:.3} host_min_us={:.3} host_p50_us={:.3} \
         host_p95_us={:.3} host_max_us={:.3} effective_gbps_p50={gbps_at_p50:.3}",
        gpu.min_us,
        gpu.p50_us,
        gpu.p95_us,
        gpu.max_us,
        host.min_us,
        host.p50_us,
        host.p95_us,
        host.max_us,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::parse().map_err(|e| format!("peer_chain: {e}"))?;
    let hip = HipRuntime::load()?;
    let device_count = hip.device_count()?;
    if device_count < 2 {
        return Err(format!(
            "peer_chain requires at least two visible devices, got {device_count}; \
             set HIP_VISIBLE_DEVICES to the intended pair"
        )
        .into());
    }

    let arch0 = hip.get_arch(0)?;
    let arch1 = hip.get_arch(1)?;
    if let Some(expected) = cfg.expect_arch0.as_deref() {
        if arch0 != expected {
            return Err(format!("logical device 0 is {arch0}, expected {expected}").into());
        }
    }
    if let Some(expected) = cfg.expect_arch1.as_deref() {
        if arch1 != expected {
            return Err(format!("logical device 1 is {arch1}, expected {expected}").into());
        }
    }

    let can_0_to_1 = hip.can_access_peer(0, 1)?;
    let can_1_to_0 = hip.can_access_peer(1, 0)?;
    if !can_0_to_1 || !can_1_to_0 {
        return Err(format!(
            "bidirectional peer access required: 0->1={can_0_to_1} 1->0={can_1_to_0}"
        )
        .into());
    }

    let rccl = if cfg.sync == SyncMode::Rccl {
        Some(RcclComms::init_all(&[0, 1])?)
    } else {
        None
    };
    let rccl_version = rccl.as_ref().map(RcclComms::version).transpose()?;
    let rocr = if cfg.sync.is_rocr() {
        Some(RocrChannel::new(
            cfg.layers,
            matches!(cfg.sync, SyncMode::RocrSdma | SyncMode::RocrAql),
            cfg.sync == SyncMode::RocrAql,
            cfg.rocr_engine_0_to_1,
            cfg.rocr_engine_1_to_0,
        )?)
    } else {
        None
    };

    let max_batch = *cfg.batches.iter().max().unwrap();
    let max_bytes = max_batch
        .checked_mul(HIDDEN)
        .and_then(|n| n.checked_mul(F32_BYTES))
        .ok_or("payload byte-size overflow")?;

    println!(
        "identity arch0={arch0} arch1={arch1} visible_devices={device_count} \
         peer_0_to_1={can_0_to_1} peer_1_to_0={can_1_to_0} sync={} hidden={HIDDEN} \
         layers={} warmups={} one_way_samples={} chain_samples={} \
         exactness_samples={} exactness_seed_base={} rccl_version={rccl_version:?} \
         max_bytes={max_bytes}",
        cfg.sync.label(),
        cfg.layers,
        cfg.warmups,
        cfg.one_way_samples,
        cfg.chain_samples,
        cfg.exactness_samples,
        cfg.exactness_seed_base
    );
    if let Some(rocr) = &rocr {
        println!("{}", rocr.identity());
    }

    hip.set_device(0)?;
    let dev0_a = hip.malloc(max_bytes)?;
    let dev0_b = hip.malloc(max_bytes)?;
    let signal_to0 = if cfg.sync == SyncMode::Signal {
        let signal = hip.malloc_signal(SIGNAL_BYTES)?;
        hip.memset(&signal, 0, SIGNAL_BYTES)?;
        Some(signal)
    } else {
        None
    };
    let stream0 = hip.stream_create()?;
    hip.enable_peer_access(1)?;
    // Exercise idempotence so the product path can call this after every load.
    hip.enable_peer_access(1)?;

    hip.set_device(1)?;
    let dev1_a = hip.malloc(max_bytes)?;
    let dev1_b = hip.malloc(max_bytes)?;
    let signal_to1 = if cfg.sync == SyncMode::Signal {
        let signal = hip.malloc_signal(SIGNAL_BYTES)?;
        hip.memset(&signal, 0, SIGNAL_BYTES)?;
        Some(signal)
    } else {
        None
    };
    let stream1 = hip.stream_create()?;
    hip.enable_peer_access(0)?;
    hip.enable_peer_access(0)?;

    hip.set_device(0)?;
    let one_start0 = hip.event_create()?;
    let one_stop0 = hip.event_create()?;
    let chain_start0 = hip.event_create()?;
    let chain_stop0 = hip.event_create()?;
    let mut events_to1 = Vec::new();
    if cfg.sync == SyncMode::Event {
        events_to1.reserve(cfg.layers);
        for _ in 0..cfg.layers {
            events_to1.push(
                hip.event_create_with_flags(
                    HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM,
                )?,
            );
        }
    }

    hip.set_device(1)?;
    let one_start1 = hip.event_create()?;
    let one_stop1 = hip.event_create()?;
    let mut events_to0 = Vec::new();
    if cfg.sync == SyncMode::Event {
        events_to0.reserve(cfg.layers);
        for _ in 0..cfg.layers {
            events_to0.push(
                hip.event_create_with_flags(
                    HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM,
                )?,
            );
        }
    }

    let direction_0_to_1 = Direction {
        src_device: 0,
        dst_device: 1,
        src: &dev0_a,
        dst: &dev1_a,
        stream: &stream0,
        start: &one_start0,
        stop: &one_stop0,
    };
    let direction_1_to_0 = Direction {
        src_device: 1,
        dst_device: 0,
        src: &dev1_b,
        dst: &dev0_b,
        stream: &stream1,
        start: &one_start1,
        stop: &one_stop1,
    };
    let chain = Chain {
        stream0: &stream0,
        stream1: &stream1,
        start0: &chain_start0,
        stop0: &chain_stop0,
        signal_to1: signal_to1.as_ref(),
        signal_to0: signal_to0.as_ref(),
        events_to1: &events_to1,
        events_to0: &events_to0,
        rccl: rccl.as_ref(),
        rocr: rocr.as_ref(),
        dev0_a: &dev0_a,
        dev0_b: &dev0_b,
        dev1: &dev1_a,
        layers: cfg.layers,
        next_epoch: Cell::new(0),
        sync: cfg.sync,
    };

    for &batch in &cfg.batches {
        let size = batch * HIDDEN * F32_BYTES;
        let expected_0 = pattern(size, 7);
        let expected_1 = pattern(size, 113);

        hip.set_device(0)?;
        hip.memcpy_htod(&dev0_a, &expected_0)?;
        hip.memset(&dev0_b, 0, size)?;
        hip.device_synchronize()?;
        hip.set_device(1)?;
        hip.memset(&dev1_a, 0, size)?;
        hip.memcpy_htod(&dev1_b, &expected_1)?;
        hip.device_synchronize()?;

        // Directional correctness before warm/timed samples.
        if let Some(rocr) = &rocr {
            rocr.one_way(true, &dev1_a, &dev0_a, size)?;
        } else {
            one_way_sample(&hip, &direction_0_to_1, size)?;
        }
        assert_bytes(&hip, 1, &dev1_a, &expected_0, "0->1")?;
        if let Some(rocr) = &rocr {
            rocr.one_way(false, &dev0_b, &dev1_b, size)?;
        } else {
            one_way_sample(&hip, &direction_1_to_0, size)?;
        }
        assert_bytes(&hip, 0, &dev0_b, &expected_1, "1->0")?;

        // Chain correctness starts with one patterned and two zeroed buffers;
        // a silently skipped copy therefore cannot pass by stale equality.
        initialize_chain_inputs(&hip, &dev0_a, &dev0_b, &dev1_a, &expected_0)?;
        chain_sample(&hip, &chain, size, true)?;
        let final_dev0 = if cfg.layers % 2 == 0 {
            &dev0_a
        } else {
            &dev0_b
        };
        assert_bytes(&hip, 0, final_dev0, &expected_0, "round-trip chain")?;

        for _ in 0..cfg.warmups {
            if let Some(rocr) = &rocr {
                rocr.one_way(true, &dev1_a, &dev0_a, size)?;
                rocr.one_way(false, &dev0_b, &dev1_b, size)?;
            } else {
                one_way_sample(&hip, &direction_0_to_1, size)?;
                one_way_sample(&hip, &direction_1_to_0, size)?;
                if cfg.sync.supports_sync_only() {
                    chain_sample(&hip, &chain, size, false)?;
                }
            }
            chain_sample(&hip, &chain, size, true)?;
        }

        let mut gpu_0_to_1 = Vec::with_capacity(cfg.one_way_samples);
        let mut host_0_to_1 = Vec::with_capacity(cfg.one_way_samples);
        let mut gpu_1_to_0 = Vec::with_capacity(cfg.one_way_samples);
        let mut host_1_to_0 = Vec::with_capacity(cfg.one_way_samples);
        for _ in 0..cfg.one_way_samples {
            let (gpu_ms, host_ms) = if let Some(rocr) = &rocr {
                let wall_ms = rocr.one_way(true, &dev1_a, &dev0_a, size)?;
                (wall_ms, wall_ms)
            } else {
                one_way_sample(&hip, &direction_0_to_1, size)?
            };
            gpu_0_to_1.push(gpu_ms);
            host_0_to_1.push(host_ms);
            let (gpu_ms, host_ms) = if let Some(rocr) = &rocr {
                let wall_ms = rocr.one_way(false, &dev0_b, &dev1_b, size)?;
                (wall_ms, wall_ms)
            } else {
                one_way_sample(&hip, &direction_1_to_0, size)?
            };
            gpu_1_to_0.push(gpu_ms);
            host_1_to_0.push(host_ms);
        }
        print_row(
            "one_way",
            "0_to_1",
            batch,
            size,
            1,
            Distribution::from_ms(&gpu_0_to_1),
            Distribution::from_ms(&host_0_to_1),
        );
        print_row(
            "one_way",
            "1_to_0",
            batch,
            size,
            1,
            Distribution::from_ms(&gpu_1_to_0),
            Distribution::from_ms(&host_1_to_0),
        );

        let mut event_gpu = Vec::with_capacity(cfg.chain_samples);
        let mut event_host = Vec::with_capacity(cfg.chain_samples);
        let mut chain_gpu = Vec::with_capacity(cfg.chain_samples);
        let mut chain_host = Vec::with_capacity(cfg.chain_samples);
        for _ in 0..cfg.chain_samples {
            if cfg.sync.supports_sync_only() {
                let (gpu_ms, host_ms) = chain_sample(&hip, &chain, size, false)?;
                event_gpu.push(gpu_ms);
                event_host.push(host_ms);
            }
            let (gpu_ms, host_ms) = chain_sample(&hip, &chain, size, true)?;
            chain_gpu.push(gpu_ms);
            chain_host.push(host_ms);
        }
        if cfg.sync.supports_sync_only() {
            print_row(
                match cfg.sync {
                    SyncMode::Event => "event_chain",
                    SyncMode::Host => "host_sync_chain",
                    SyncMode::Rccl => "rccl_grouped_chain",
                    SyncMode::RocrAql => "rocr_aql_barrier_chain",
                    SyncMode::Rocr | SyncMode::RocrSdma => unreachable!(),
                    SyncMode::Signal => "signal_chain",
                },
                "round_trip",
                batch,
                0,
                cfg.layers * 2,
                Distribution::from_ms(&event_gpu),
                Distribution::from_ms(&event_host),
            );
        }
        print_row(
            "copy_chain",
            "round_trip",
            batch,
            size,
            cfg.layers * 2,
            Distribution::from_ms(&chain_gpu),
            Distribution::from_ms(&chain_host),
        );
        // Reinitialize with a distinct payload before every post-timing chain.
        // This catches intermittent dependency-event failures that a stable
        // repeated payload could hide after the next successful round trip.
        for sample in 0..cfg.exactness_samples {
            let seed = cfg
                .exactness_seed_base
                .checked_add(sample as u64)
                .expect("exactness payload seed overflow");
            let expected = pattern(size, seed);
            initialize_chain_inputs(&hip, &dev0_a, &dev0_b, &dev1_a, &expected)?;
            chain_sample(&hip, &chain, size, true)?;
            let final_dev0 = if cfg.layers % 2 == 0 {
                &dev0_a
            } else {
                &dev0_b
            };
            assert_bytes(
                &hip,
                0,
                final_dev0,
                &expected,
                &format!("round-trip stress batch={batch} sample={sample} seed={seed}"),
            )?;
        }
        println!(
            "exactness batch={batch} samples={} seed_base={} status=PASS",
            cfg.exactness_samples + 1,
            cfg.exactness_seed_base
        );
    }

    hip.set_device(0)?;
    hip.event_destroy(chain_start0)?;
    hip.event_destroy(chain_stop0)?;
    hip.event_destroy(one_start0)?;
    hip.event_destroy(one_stop0)?;
    for event in events_to1 {
        hip.event_destroy(event)?;
    }
    hip.stream_destroy(stream0)?;
    hip.free(dev0_a)?;
    hip.free(dev0_b)?;
    if let Some(signal) = signal_to0 {
        hip.free(signal)?;
    }

    hip.set_device(1)?;
    hip.event_destroy(one_start1)?;
    hip.event_destroy(one_stop1)?;
    for event in events_to0 {
        hip.event_destroy(event)?;
    }
    hip.stream_destroy(stream1)?;
    hip.free(dev1_a)?;
    hip.free(dev1_b)?;
    if let Some(signal) = signal_to1 {
        hip.free(signal)?;
    }

    println!("peer_chain: PASS");
    Ok(())
}
