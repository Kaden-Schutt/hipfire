//! Validate the async NPU dispatch split (submit / poll / wait) on hardware:
//!  1. submit -> non-blocking poll loop -> wait, and the output is correct;
//!  2. TWO in-flight dispatches at once (distinct C buffers, per-handle command BOs),
//!     polled and waited in order, both correct, tags preserved — the multi-in-flight
//!     path a GPU‖NPU microbatch pipeline needs.
//!
//! Run: cargo run -p hipfire-xdna --example async_smoke -- <dir> <asz> <wsz> <csz> <expect_c0>

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        let a: Vec<String> = std::env::args().collect();
        if a.len() < 6 {
            eprintln!("usage: async_smoke <dir> <asz> <wsz> <csz> <expect_c0>");
            std::process::exit(2);
        }
        let dir = &a[1];
        let (asz, wsz, csz): (usize, usize, usize) = (
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
        );
        let expect: i32 = a[5].parse().unwrap();
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");
        let c0 = |c: &hipfire_xdna::DeviceBuffer| unsafe { *(c.as_slice().as_ptr() as *const i32) };

        let mut aw = k.alloc_arg(asz).unwrap();
        let mut ww = k.alloc_arg(wsz).unwrap();
        aw.as_mut_slice().fill(1);
        ww.as_mut_slice().fill(0x11);

        // 1) submit -> poll -> wait
        let mut c1 = k.alloc_arg(csz).unwrap();
        c1.as_mut_slice().fill(0);
        let f = k.submit(&[&aw, &ww, &c1]).expect("submit");
        let mut polls = 0u64;
        while !k.poll(&f).expect("poll") {
            polls += 1;
        }
        k.wait(f).expect("wait");
        let r1 = c0(&c1);
        println!("[1] submit/poll/wait: polls-before-done={polls}  C[0]={r1} (expect {expect})");
        assert_eq!(r1, expect, "async single-dispatch result wrong");

        // 2) two in-flight at once (distinct C buffers + per-handle command BOs), tagged
        let mut c2 = k.alloc_arg(csz).unwrap();
        let mut c3 = k.alloc_arg(csz).unwrap();
        c2.as_mut_slice().fill(0);
        c3.as_mut_slice().fill(0);
        let f2 = k.submit_tagged(&[&aw, &ww, &c2], 101).expect("submit c2");
        let f3 = k.submit_tagged(&[&aw, &ww, &c3], 202).expect("submit c3");
        println!(
            "[2] two in-flight: seq2={} tag2={}  seq3={} tag3={}  (seq strictly increasing: {})",
            f2.seq(),
            f2.tag(),
            f3.seq(),
            f3.tag(),
            f3.seq() > f2.seq()
        );
        assert!(
            f3.seq() > f2.seq(),
            "second submit must have a later timeline seq"
        );
        assert_eq!((f2.tag(), f3.tag()), (101, 202), "tags not preserved");
        k.wait(f2).expect("wait c2");
        k.wait(f3).expect("wait c3");
        let (r2, r3) = (c0(&c2), c0(&c3));
        println!("    C2[0]={r2}  C3[0]={r3} (expect {expect} each)");
        assert!(
            r2 == expect && r3 == expect,
            "multi-in-flight results wrong"
        );

        println!("async dispatch OK (submit/poll/wait + multi-in-flight + tags)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
