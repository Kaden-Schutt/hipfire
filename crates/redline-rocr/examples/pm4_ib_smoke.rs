// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt

use std::time::Duration;

use redline_rocr::packet::PacketImage;
use redline_rocr::{CompletionSignal, GpuSelector, KernargPool, QueueSet, Runtime, load_symbols};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::initialize(load_symbols()?)?;
    let device = runtime.select_gpu(GpuSelector::Ordinal(0))?;
    let pool = KernargPool::discover(&device)?;
    let mut indirect = pool.allocate_executable_bytes(8)?;
    // Ordinary two-dword PACKET3_NOP: count=0 plus one zero payload dword.
    indirect
        .as_mut_bytes()
        .copy_from_slice(&[0x00, 0x10, 0x00, 0xc0, 0, 0, 0, 0]);
    let completion = CompletionSignal::new(&device)?;
    let packet = PacketImage::pm4_indirect_buffer(indirect.address(), 2, completion.raw())?;
    let queue_size = *device.queue_size_range().start();
    let mut queues = QueueSet::create(&device, 1, queue_size)?;
    queues.prepare_batches(&[vec![packet]])?;
    queues.ring_prepared()?;
    queues.wait_signal(&completion, Duration::from_secs(5))?;
    println!(
        "pm4-ib-smoke: pass queue={} ib=0x{:x}",
        queues.queue_ids().next().unwrap(),
        indirect.address() as usize,
    );
    drop(queues);
    drop(completion);
    drop(indirect);
    drop(pool);
    drop(device);
    drop(runtime);
    Ok(())
}
