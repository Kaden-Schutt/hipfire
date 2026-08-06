// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Persistent bidirectional HIP peer-channel benchmark.
//!
//! This is the H0 transport gate for the DeepSeek V4 heterogeneous
//! gfx1100/gfx1151 route. It measures the actual decode boundary shape:
//! one `[batch, hidden=4096]` F32 payload in each direction across a
//! 43-layer dependency chain. Streams, timing events, dependency events,
//! and buffers are allocated once and reused for every sample.
//!
//! Example:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 cargo run --release -p hip-bridge \
//!   --example peer_chain -- \
//!   --expect-arch0 gfx1100 --expect-arch1 gfx1151
//! ```

use hip_bridge::{DeviceBuffer, Event, HipResult, HipRuntime, Stream};
use std::time::Instant;

const HIDDEN: usize = 4096;
const F32_BYTES: usize = 4;
const DEFAULT_LAYERS: usize = 43;
const DEFAULT_BATCHES: &[usize] = &[1, 16, 128, 512, 1024];

#[derive(Debug)]
struct Config {
    warmups: usize,
    one_way_samples: usize,
    chain_samples: usize,
    layers: usize,
    batches: Vec<usize>,
    expect_arch0: Option<String>,
    expect_arch1: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            warmups: 10,
            one_way_samples: 100,
            chain_samples: 50,
            layers: DEFAULT_LAYERS,
            batches: DEFAULT_BATCHES.to_vec(),
            expect_arch0: None,
            expect_arch1: None,
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

fn print_help() {
    println!(
        "peer_chain [options]\n\
         \n\
         Options:\n\
           --warmups N            warm chain/one-way iterations (default 10)\n\
           --one-way-samples N    samples per direction and size (default 100)\n\
           --chain-samples N      43-layer chain samples per size (default 50)\n\
           --layers N             round-trip dependency count (default 43)\n\
           --batches CSV          batch rows for [B,4096] F32 (default 1,16,128,512,1024)\n\
           --expect-arch0 ARCH    fail unless logical device 0 matches\n\
           --expect-arch1 ARCH    fail unless logical device 1 matches"
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

fn pattern(size: usize, salt: u8) -> Vec<u8> {
    (0..size)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(salt as usize)) & 0xff) as u8)
        .collect()
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
    to1_ready: &'a [Event],
    to0_ready: &'a [Event],
    dev0_a: &'a DeviceBuffer,
    dev0_b: &'a DeviceBuffer,
    dev1: &'a DeviceBuffer,
}

fn chain_sample(
    hip: &HipRuntime,
    chain: &Chain<'_>,
    size: usize,
    copy_payload: bool,
) -> HipResult<(f64, f64)> {
    debug_assert_eq!(chain.to1_ready.len(), chain.to0_ready.len());
    hip.set_device(0)?;
    let host_start = Instant::now();
    hip.event_record(chain.start0, Some(chain.stream0))?;

    for layer in 0..chain.to1_ready.len() {
        let (src0, dst0) = if layer % 2 == 0 {
            (chain.dev0_a, chain.dev0_b)
        } else {
            (chain.dev0_b, chain.dev0_a)
        };

        hip.set_device(0)?;
        if copy_payload {
            hip.memcpy_peer_async(chain.dev1, 1, src0, 0, size, chain.stream0)?;
        }
        hip.event_record(&chain.to1_ready[layer], Some(chain.stream0))?;

        hip.set_device(1)?;
        hip.stream_wait_event(chain.stream1, &chain.to1_ready[layer])?;
        if copy_payload {
            hip.memcpy_peer_async(dst0, 0, chain.dev1, 1, size, chain.stream1)?;
        }
        hip.event_record(&chain.to0_ready[layer], Some(chain.stream1))?;

        hip.set_device(0)?;
        hip.stream_wait_event(chain.stream0, &chain.to0_ready[layer])?;
    }

    hip.event_record(chain.stop0, Some(chain.stream0))?;
    hip.event_synchronize(chain.stop0)?;
    let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = hip.event_elapsed_ms(chain.start0, chain.stop0)? as f64;
    Ok((gpu_ms, host_ms))
}

fn create_events(hip: &HipRuntime, device: i32, n: usize) -> HipResult<Vec<Event>> {
    hip.set_device(device)?;
    (0..n).map(|_| hip.event_create()).collect()
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

    let max_batch = *cfg.batches.iter().max().unwrap();
    let max_bytes = max_batch
        .checked_mul(HIDDEN)
        .and_then(|n| n.checked_mul(F32_BYTES))
        .ok_or("payload byte-size overflow")?;

    println!(
        "identity arch0={arch0} arch1={arch1} visible_devices={device_count} \
         peer_0_to_1={can_0_to_1} peer_1_to_0={can_1_to_0} hidden={HIDDEN} \
         layers={} warmups={} one_way_samples={} chain_samples={} max_bytes={max_bytes}",
        cfg.layers, cfg.warmups, cfg.one_way_samples, cfg.chain_samples
    );

    hip.set_device(0)?;
    let dev0_a = hip.malloc(max_bytes)?;
    let dev0_b = hip.malloc(max_bytes)?;
    let stream0 = hip.stream_create()?;
    hip.enable_peer_access(1)?;
    // Exercise idempotence so the product path can call this after every load.
    hip.enable_peer_access(1)?;

    hip.set_device(1)?;
    let dev1_a = hip.malloc(max_bytes)?;
    let dev1_b = hip.malloc(max_bytes)?;
    let stream1 = hip.stream_create()?;
    hip.enable_peer_access(0)?;
    hip.enable_peer_access(0)?;

    hip.set_device(0)?;
    let one_start0 = hip.event_create()?;
    let one_stop0 = hip.event_create()?;
    let chain_start0 = hip.event_create()?;
    let chain_stop0 = hip.event_create()?;
    let to1_ready = create_events(&hip, 0, cfg.layers)?;

    hip.set_device(1)?;
    let one_start1 = hip.event_create()?;
    let one_stop1 = hip.event_create()?;
    let to0_ready = create_events(&hip, 1, cfg.layers)?;

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
        to1_ready: &to1_ready,
        to0_ready: &to0_ready,
        dev0_a: &dev0_a,
        dev0_b: &dev0_b,
        dev1: &dev1_a,
    };

    for &batch in &cfg.batches {
        let size = batch * HIDDEN * F32_BYTES;
        let expected_0 = pattern(size, 7);
        let expected_1 = pattern(size, 113);

        hip.set_device(0)?;
        hip.memcpy_htod(&dev0_a, &expected_0)?;
        hip.memset(&dev0_b, 0, size)?;
        hip.set_device(1)?;
        hip.memset(&dev1_a, 0, size)?;
        hip.memcpy_htod(&dev1_b, &expected_1)?;

        // Directional correctness before warm/timed samples.
        one_way_sample(&hip, &direction_0_to_1, size)?;
        assert_bytes(&hip, 1, &dev1_a, &expected_0, "0->1")?;
        one_way_sample(&hip, &direction_1_to_0, size)?;
        assert_bytes(&hip, 0, &dev0_b, &expected_1, "1->0")?;

        // Chain correctness starts with one patterned and two zeroed buffers;
        // a silently skipped copy therefore cannot pass by stale equality.
        hip.set_device(0)?;
        hip.memcpy_htod(&dev0_a, &expected_0)?;
        hip.memset(&dev0_b, 0, size)?;
        hip.set_device(1)?;
        hip.memset(&dev1_a, 0, size)?;
        chain_sample(&hip, &chain, size, true)?;
        let final_dev0 = if cfg.layers % 2 == 0 {
            &dev0_a
        } else {
            &dev0_b
        };
        assert_bytes(&hip, 0, final_dev0, &expected_0, "round-trip chain")?;

        for _ in 0..cfg.warmups {
            one_way_sample(&hip, &direction_0_to_1, size)?;
            one_way_sample(&hip, &direction_1_to_0, size)?;
            chain_sample(&hip, &chain, size, false)?;
            chain_sample(&hip, &chain, size, true)?;
        }

        let mut gpu_0_to_1 = Vec::with_capacity(cfg.one_way_samples);
        let mut host_0_to_1 = Vec::with_capacity(cfg.one_way_samples);
        let mut gpu_1_to_0 = Vec::with_capacity(cfg.one_way_samples);
        let mut host_1_to_0 = Vec::with_capacity(cfg.one_way_samples);
        for _ in 0..cfg.one_way_samples {
            let (gpu_ms, host_ms) = one_way_sample(&hip, &direction_0_to_1, size)?;
            gpu_0_to_1.push(gpu_ms);
            host_0_to_1.push(host_ms);
            let (gpu_ms, host_ms) = one_way_sample(&hip, &direction_1_to_0, size)?;
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
            let (gpu_ms, host_ms) = chain_sample(&hip, &chain, size, false)?;
            event_gpu.push(gpu_ms);
            event_host.push(host_ms);
            let (gpu_ms, host_ms) = chain_sample(&hip, &chain, size, true)?;
            chain_gpu.push(gpu_ms);
            chain_host.push(host_ms);
        }
        print_row(
            "event_chain",
            "round_trip",
            batch,
            0,
            cfg.layers * 2,
            Distribution::from_ms(&event_gpu),
            Distribution::from_ms(&event_host),
        );
        print_row(
            "copy_chain",
            "round_trip",
            batch,
            size,
            cfg.layers * 2,
            Distribution::from_ms(&chain_gpu),
            Distribution::from_ms(&chain_host),
        );
        println!("exactness batch={batch} status=PASS");
    }

    hip.set_device(0)?;
    for event in to1_ready {
        hip.event_destroy(event)?;
    }
    hip.event_destroy(chain_start0)?;
    hip.event_destroy(chain_stop0)?;
    hip.event_destroy(one_start0)?;
    hip.event_destroy(one_stop0)?;
    hip.stream_destroy(stream0)?;
    hip.free(dev0_a)?;
    hip.free(dev0_b)?;

    hip.set_device(1)?;
    for event in to0_ready {
        hip.event_destroy(event)?;
    }
    hip.event_destroy(one_start1)?;
    hip.event_destroy(one_stop1)?;
    hip.stream_destroy(stream1)?;
    hip.free(dev1_a)?;
    hip.free(dev1_b)?;

    println!("peer_chain: PASS");
    Ok(())
}
