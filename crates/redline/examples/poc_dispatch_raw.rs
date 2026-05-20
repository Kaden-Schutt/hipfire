// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ultimate bisection: hand-assembled s_endpgm shader.
//! No hipcc, no ELF parsing, no kernel descriptors.
//! Just raw machine code + absolute minimum PM4.

use redline::device::Device;
use redline::queue::ComputeQueue;

fn main() {
    eprintln!("=== redline: raw s_endpgm dispatch ===\n");

    let dev = Device::open(None).unwrap();
    let queue = ComputeQueue::new(&dev).unwrap();

    // Hand-assemble the simplest possible GFX10 compute shader:
    //   s_endpgm    ; encoding: 0xBF810000
    // Placed at offset 0 of a page-aligned VRAM buffer.
    let mut code = vec![0u8; 256]; // pad to 256 bytes for alignment
    code[0..4].copy_from_slice(&0xBF810000u32.to_le_bytes()); // s_endpgm

    let code_buf = dev.alloc_vram(4096).unwrap();
    dev.upload(&code_buf, &code).unwrap();
    let code_va = code_buf.gpu_addr; // at offset 0, page-aligned
    eprintln!("code_va=0x{:x} (aligned? {})", code_va, code_va & 0xFF == 0);
    eprintln!("code bytes: {:02x} {:02x} {:02x} {:02x}", code[0], code[1], code[2], code[3]);

    // Also prepare a "marker" buffer to verify execution
    let marker_buf = dev.alloc_vram(4096).unwrap();
    dev.upload(&marker_buf, &vec![0xADu8; 4096]).unwrap();

    let hdr = |opcode: u32, ndw: u32| -> u32 {
        (3u32 << 30) | ((ndw - 1) << 16) | (opcode << 8) | (1 << 1) // type3, compute
    };

    // === Test A: WRITE_DATA (sanity check — should work) ===
    {
        let mut pm4: Vec<u32> = Vec::new();
        let control = (5u32 << 8) | (1 << 20); // dst_sel=memory_async, wr_confirm
        pm4.push(hdr(0x37, 4));
        pm4.push(control);
        pm4.push(marker_buf.gpu_addr as u32);
        pm4.push((marker_buf.gpu_addr >> 32) as u32);
        pm4.push(0xAAAA_BBBB);

        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();
        eprint!("Test A (WRITE_DATA): ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &marker_buf]) {
            Ok(()) => {
                let mut rb = vec![0u8; 4];
                dev.download(&marker_buf, &mut rb).unwrap();
                let val = u32::from_le_bytes([rb[0], rb[1], rb[2], rb[3]]);
                eprintln!("OK (0x{:08x})", val);
            }
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }

    // === Test B: Dispatch s_endpgm with minimal PM4 ===
    eprintln!("\nTest B: Minimal dispatch (s_endpgm, no user data, wave64)");
    {
        let mut pm4: Vec<u32> = Vec::new();

        // PGM_LO/HI
        pm4.push(hdr(0x76, 3));
        pm4.push(0x020C); // COMPUTE_PGM_LO offset
        pm4.push((code_va >> 8) as u32);
        pm4.push((code_va >> 40) as u32);

        // PGM_RSRC1: all zeros — minimal VGPRs/SGPRs
        pm4.push(hdr(0x76, 3));
        pm4.push(0x0212); // COMPUTE_PGM_RSRC1 offset
        pm4.push(0x00000000); // RSRC1: VGPRS=0→8, SGPRS=0→8, all defaults
        pm4.push(0x00000000); // RSRC2: no scratch, no user SGPRs, no TG IDs

        // PGM_RSRC3
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0228);
        pm4.push(0);

        // TMPRING_SIZE
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0218);
        pm4.push(0);

        // NUM_THREAD_X/Y/Z
        pm4.push(hdr(0x76, 4));
        pm4.push(0x0207); // COMPUTE_NUM_THREAD_X offset
        pm4.push(1); // 1 thread
        pm4.push(1);
        pm4.push(1);

        // RESOURCE_LIMITS
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0215);
        pm4.push(0);

        // DISPATCH_DIRECT — NO CS_W32_EN (wave64 default)
        let di = 1u32; // just CS_EN
        pm4.push(hdr(0x15, 4));
        pm4.push(1); // 1 group
        pm4.push(1);
        pm4.push(1);
        pm4.push(di);

        eprintln!("PM4: {} dwords", pm4.len());

        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();

        eprint!("Submitting... ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &code_buf]) {
            Ok(()) => eprintln!("OK!"),
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }

    // === Test C: Same but with WRITE_DATA after dispatch (check if CP continues) ===
    eprintln!("\nTest C: WRITE_DATA before dispatch + WRITE_DATA after dispatch");
    {
        // Reset marker
        dev.upload(&marker_buf, &vec![0xADu8; 16]).unwrap();

        let mut pm4: Vec<u32> = Vec::new();
        let control = (5u32 << 8) | (1 << 20);

        // Write marker BEFORE dispatch
        pm4.push(hdr(0x37, 4));
        pm4.push(control);
        pm4.push(marker_buf.gpu_addr as u32);
        pm4.push((marker_buf.gpu_addr >> 32) as u32);
        pm4.push(0x1111_1111);

        // PGM_LO/HI
        pm4.push(hdr(0x76, 3));
        pm4.push(0x020C);
        pm4.push((code_va >> 8) as u32);
        pm4.push((code_va >> 40) as u32);

        // PGM_RSRC1/2
        pm4.push(hdr(0x76, 3));
        pm4.push(0x0212);
        pm4.push(0x00000000);
        pm4.push(0x00000000);

        // PGM_RSRC3
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0228);
        pm4.push(0);

        // TMPRING_SIZE
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0218);
        pm4.push(0);

        // NUM_THREAD
        pm4.push(hdr(0x76, 4));
        pm4.push(0x0207);
        pm4.push(1);
        pm4.push(1);
        pm4.push(1);

        // RESOURCE_LIMITS
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0215);
        pm4.push(0);

        // DISPATCH
        pm4.push(hdr(0x15, 4));
        pm4.push(1);
        pm4.push(1);
        pm4.push(1);
        pm4.push(1u32); // CS_EN only

        // Write marker AFTER dispatch
        pm4.push(hdr(0x37, 4));
        pm4.push(control);
        pm4.push((marker_buf.gpu_addr + 4) as u32);
        pm4.push(((marker_buf.gpu_addr + 4) >> 32) as u32);
        pm4.push(0x2222_2222);

        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();

        eprint!("Submitting... ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &code_buf, &marker_buf]) {
            Ok(()) => {
                let mut rb = vec![0u8; 8];
                dev.download(&marker_buf, &mut rb).unwrap();
                let pre = u32::from_le_bytes([rb[0], rb[1], rb[2], rb[3]]);
                let post = u32::from_le_bytes([rb[4], rb[5], rb[6], rb[7]]);
                eprintln!("OK  pre=0x{:08x} post=0x{:08x}", pre, post);
            }
            Err(e) => {
                // Even on timeout, check what was written
                let mut rb = vec![0u8; 8];
                let _ = dev.download(&marker_buf, &mut rb);
                let pre = u32::from_le_bytes([rb[0], rb[1], rb[2], rb[3]]);
                let post = u32::from_le_bytes([rb[4], rb[5], rb[6], rb[7]]);
                eprintln!("FAIL: {e}");
                eprintln!("  pre-dispatch marker: 0x{:08x} (expect 0x11111111)", pre);
                eprintln!("  post-dispatch marker: 0x{:08x} (expect 0x22222222)", post);
                eprintln!("  If pre=0x11111111 but post=0xADADADAD: GPU hung during dispatch");
                eprintln!("  If both 0xADADADAD: CP never reached the WRITE_DATA packets");
            }
        }
    }

    // === Test D: s_endpgm but with HSACO's RSRC1/2 values ===
    eprintln!("\nTest D: s_endpgm + RSRC1=0x60af0000, RSRC2=0x00000088 (from hipcc noop)");
    {
        let mut pm4: Vec<u32> = Vec::new();
        pm4.push(hdr(0x76, 3));
        pm4.push(0x020C);
        pm4.push((code_va >> 8) as u32);
        pm4.push((code_va >> 40) as u32);
        pm4.push(hdr(0x76, 3));
        pm4.push(0x0212);
        pm4.push(0x60af0000); // hipcc's RSRC1
        pm4.push(0x00000088); // hipcc's RSRC2 (USER_SGPR=4, TGID_X_EN=1)
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0228);
        pm4.push(0);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0218);
        pm4.push(0);
        pm4.push(hdr(0x76, 4));
        pm4.push(0x0207);
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0215);
        pm4.push(0);
        // USER_DATA: 4 zeros for private segment buffer (USER_SGPR=4)
        pm4.push(hdr(0x76, 5));
        pm4.push(0x0240);
        pm4.push(0); pm4.push(0); pm4.push(0); pm4.push(0);
        pm4.push(hdr(0x15, 4));
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(1u32); // CS_EN only, wave64
        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();
        eprint!("Submitting... ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &code_buf]) {
            Ok(()) => eprintln!("OK!"),
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }

    // === Test E: actual HSACO code at computed offset ===
    eprintln!("\nTest E: HSACO noop kernel code from ELF at offset 0x1600");
    {
        // Compile noop kernel
        let hip_src = "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void noop_kernel() {}\n";
        std::fs::write("/tmp/redline_noop.hip", hip_src).unwrap();
        let out = std::process::Command::new("hipcc")
            .args(["--genco", "--offload-arch=gfx1010", "-O3",
                   "-o", "/tmp/redline_noop.hsaco", "/tmp/redline_noop.hip"])
            .output().expect("hipcc");
        assert!(out.status.success(), "hipcc failed");
        let module = redline::hsaco::HsacoModule::from_file("/tmp/redline_noop.hsaco").unwrap();
        let k = &module.kernels[0];

        let elf_buf = dev.alloc_vram(module.elf.len() as u64).unwrap();
        dev.upload(&elf_buf, &module.elf).unwrap();
        let elf_code_va = elf_buf.gpu_addr + k.code_offset;
        eprintln!("  elf_code_va=0x{:x} (aligned? {})", elf_code_va, elf_code_va & 0xFF == 0);

        // Verify: dump first 16 bytes of code from the ELF
        let co = k.code_offset as usize;
        if co + 16 <= module.elf.len() {
            eprintln!("  code bytes at offset 0x{:x}: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                co, module.elf[co], module.elf[co+1], module.elf[co+2], module.elf[co+3],
                module.elf[co+4], module.elf[co+5], module.elf[co+6], module.elf[co+7]);
        }

        let mut pm4: Vec<u32> = Vec::new();
        pm4.push(hdr(0x76, 3));
        pm4.push(0x020C);
        pm4.push((elf_code_va >> 8) as u32);
        pm4.push((elf_code_va >> 40) as u32);
        pm4.push(hdr(0x76, 3));
        pm4.push(0x0212);
        pm4.push(k.pgm_rsrc1);
        pm4.push(k.pgm_rsrc2);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0228);
        pm4.push(0);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0218);
        pm4.push(0);
        pm4.push(hdr(0x76, 4));
        pm4.push(0x0207);
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0215);
        pm4.push(0);
        pm4.push(hdr(0x76, 5));
        pm4.push(0x0240);
        pm4.push(0); pm4.push(0); pm4.push(0); pm4.push(0);
        pm4.push(hdr(0x15, 4));
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(1u32);

        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();
        eprint!("Submitting... ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &elf_buf]) {
            Ok(()) => eprintln!("OK!"),
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }

    // === Test F: s_endpgm at HSACO code offset (ELF bytes but our code) ===
    // Tests if the offset/alignment is the issue vs the code itself
    eprintln!("\nTest F: s_endpgm placed at same offset as HSACO code (0x1600)");
    {
        let mut custom_buf = vec![0u8; 0x2000];
        // Place s_endpgm at offset 0x1600
        custom_buf[0x1600..0x1604].copy_from_slice(&0xBF810000u32.to_le_bytes());
        let fbuf = dev.alloc_vram(custom_buf.len() as u64).unwrap();
        dev.upload(&fbuf, &custom_buf).unwrap();
        let fcode_va = fbuf.gpu_addr + 0x1600;
        eprintln!("  fcode_va=0x{:x} (aligned? {})", fcode_va, fcode_va & 0xFF == 0);

        let mut pm4: Vec<u32> = Vec::new();
        pm4.push(hdr(0x76, 3));
        pm4.push(0x020C);
        pm4.push((fcode_va >> 8) as u32);
        pm4.push((fcode_va >> 40) as u32);
        pm4.push(hdr(0x76, 3));
        pm4.push(0x0212);
        pm4.push(0x60af0000);
        pm4.push(0x00000088);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0228);
        pm4.push(0);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0218);
        pm4.push(0);
        pm4.push(hdr(0x76, 4));
        pm4.push(0x0207);
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0215);
        pm4.push(0);
        pm4.push(hdr(0x76, 5));
        pm4.push(0x0240);
        pm4.push(0); pm4.push(0); pm4.push(0); pm4.push(0);
        pm4.push(hdr(0x15, 4));
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(1u32);

        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();
        eprint!("Submitting... ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &fbuf]) {
            Ok(()) => eprintln!("OK!"),
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }

    // === Test G: HSACO noop kernel (same as poc_dispatch_noop, but in this process) ===
    eprintln!("\nTest G: HSACO noop kernel from ELF (full dispatch)");
    {
        let hip_src = "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void noop_kernel() {}\n";
        std::fs::write("/tmp/redline_noop.hip", hip_src).unwrap();
        let out = std::process::Command::new("hipcc")
            .args(["--genco", "--offload-arch=gfx1010", "-O3",
                   "-o", "/tmp/redline_noop.hsaco", "/tmp/redline_noop.hip"])
            .output().expect("hipcc");
        assert!(out.status.success());
        let module = redline::hsaco::HsacoModule::from_file("/tmp/redline_noop.hsaco").unwrap();
        let k = &module.kernels[0];
        eprintln!("  kernel: {} rsrc1=0x{:08x} rsrc2=0x{:08x}", k.name, k.pgm_rsrc1, k.pgm_rsrc2);
        eprintln!("  kd_offset=0x{:x} code_offset=0x{:x}", k.kd_offset, k.code_offset);
        eprintln!("  elf size={} bytes", module.elf.len());

        // Verify code bytes
        let co = k.code_offset as usize;
        if co + 4 <= module.elf.len() {
            let instr = u32::from_le_bytes([module.elf[co], module.elf[co+1], module.elf[co+2], module.elf[co+3]]);
            eprintln!("  code[0] = 0x{:08x} (expect 0xBF810000 = s_endpgm)", instr);
        } else {
            eprintln!("  ERROR: code_offset 0x{:x} past ELF end 0x{:x}", co, module.elf.len());
        }

        let elf_buf = dev.alloc_vram(module.elf.len() as u64).unwrap();
        dev.upload(&elf_buf, &module.elf).unwrap();
        let elf_code_va = elf_buf.gpu_addr + k.code_offset;
        eprintln!("  elf_code_va=0x{:x}", elf_code_va);

        let mut pm4: Vec<u32> = Vec::new();
        // Use EXACTLY the same PM4 as working Test D
        pm4.push(hdr(0x76, 3));
        pm4.push(0x020C);
        pm4.push((elf_code_va >> 8) as u32);
        pm4.push((elf_code_va >> 40) as u32);
        pm4.push(hdr(0x76, 3));
        pm4.push(0x0212);
        pm4.push(k.pgm_rsrc1);
        pm4.push(k.pgm_rsrc2);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0228);
        pm4.push(0);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0218);
        pm4.push(0);
        pm4.push(hdr(0x76, 4));
        pm4.push(0x0207);
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0215);
        pm4.push(0);
        pm4.push(hdr(0x76, 5));
        pm4.push(0x0240);
        pm4.push(0); pm4.push(0); pm4.push(0); pm4.push(0);
        pm4.push(hdr(0x15, 4));
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(1u32); // CS_EN only, same as Test D

        // Print PM4 for comparison
        eprintln!("  PM4 ({} dwords): {:08x} {:08x} {:08x} {:08x} ...",
            pm4.len(), pm4[0], pm4[1], pm4[2], pm4[3]);

        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();
        eprint!("Submitting... ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &elf_buf]) {
            Ok(()) => eprintln!("OK!"),
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }

    // === Test H: Copy just the s_endpgm from HSACO .text into a clean buffer ===
    eprintln!("\nTest H: Copy HSACO .text bytes into fresh buffer, dispatch from there");
    {
        let module = redline::hsaco::HsacoModule::from_file("/tmp/redline_noop.hsaco").unwrap();
        let k = &module.kernels[0];
        let co = k.code_offset as usize;

        // Allocate a page-aligned buffer, copy .text at the same file offset
        let mut clean = vec![0u8; 4096];
        let text_bytes = &module.elf[co..std::cmp::min(co + 256, module.elf.len())];
        clean[co..co + text_bytes.len()].copy_from_slice(text_bytes);
        eprintln!("  Copied {} bytes to offset 0x{:x} in clean buffer", text_bytes.len(), co);
        eprintln!("  code[0] = 0x{:08x}", u32::from_le_bytes([clean[co], clean[co+1], clean[co+2], clean[co+3]]));

        let hbuf = dev.alloc_vram(4096).unwrap();
        dev.upload(&hbuf, &clean).unwrap();
        let hcode_va = hbuf.gpu_addr + co as u64;
        eprintln!("  hcode_va=0x{:x} (aligned? {})", hcode_va, hcode_va & 0xFF == 0);

        let mut pm4: Vec<u32> = Vec::new();
        pm4.push(hdr(0x76, 3));
        pm4.push(0x020C);
        pm4.push((hcode_va >> 8) as u32);
        pm4.push((hcode_va >> 40) as u32);
        pm4.push(hdr(0x76, 3));
        pm4.push(0x0212);
        pm4.push(k.pgm_rsrc1);
        pm4.push(k.pgm_rsrc2);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0228);
        pm4.push(0);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0218);
        pm4.push(0);
        pm4.push(hdr(0x76, 4));
        pm4.push(0x0207);
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(hdr(0x76, 2));
        pm4.push(0x0215);
        pm4.push(0);
        pm4.push(hdr(0x76, 5));
        pm4.push(0x0240);
        pm4.push(0); pm4.push(0); pm4.push(0); pm4.push(0);
        pm4.push(hdr(0x15, 4));
        pm4.push(1); pm4.push(1); pm4.push(1);
        pm4.push(1u32);

        let ib = dev.alloc_vram(4096).unwrap();
        let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
        dev.upload(&ib, &ib_bytes).unwrap();
        eprint!("Submitting... ");
        match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &hbuf]) {
            Ok(()) => eprintln!("OK!"),
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }

    queue.destroy(&dev);
}
