# NPU to GPU dma-buf results 20260601T172728Z

- Host date UTC: 20260601T172728Z
- Kernel release: 7.0.0-15-generic
- HIP compiler: HIP version: 7.13.26176-79e85e1468
- GPU arch: gfx1151
- XRT version: 2.25.0
- amdxdna version: 2.25.0_20260601, 627cee46c6c40fd92147ba64cb1c596538aad750
- XRT payload dir: /home/sadara/xdna-driver/build/vtd_extract/strx
- XCLBIN: /home/sadara/xdna-driver/build/vtd_extract/strx/validate_df_bandwidth.xclbin
- ELF: /home/sadara/xdna-driver/build/vtd_extract/strx/df_bw.elf
- Build status: 0
- xrt-smi examine status: 0
- xrt-smi validate df-bw status: 1
- single matrix status: 0
- pingpong matrix status: 0

Raw logs are in `raw/`. JSON benchmark reports are next to this README.

Commands:

```bash
benchmarks/npu_gpu_dmabuf/build.sh
xrt-smi examine --batch
xrt-smi validate -r df-bw -p /home/sadara/xdna-driver/build/vtd_extract/strx
target/npu_gpu_dmabuf/npu_gpu_dmabuf_bench --mode single --size <4K|64K|1M|16M|64M> --iters 1 --xclbin /home/sadara/xdna-driver/build/vtd_extract/strx/validate_df_bandwidth.xclbin --elf /home/sadara/xdna-driver/build/vtd_extract/strx/df_bw.elf --json <out>.json
target/npu_gpu_dmabuf/npu_gpu_dmabuf_bench --mode pingpong --size <1M|16M|64M> --iters 100 --xclbin /home/sadara/xdna-driver/build/vtd_extract/strx/validate_df_bandwidth.xclbin --elf /home/sadara/xdna-driver/build/vtd_extract/strx/df_bw.elf --json <out>.json
```

The requested df-bw preflight command is recorded even if this installed
`xrt-smi` does not support `-r df-bw -p <dir>`.
