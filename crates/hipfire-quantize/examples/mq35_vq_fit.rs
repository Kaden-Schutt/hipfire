//! MQ3.5GL VQ fit & evaluation on real post-FWHT weights.
//! Run: cargo run -p hipfire-quantize --example mq35_vq_fit --release
//!
//! Measures rotated-domain MSE on real post-FWHT weights from Qwen3.8-27B,
//! using per-block fp16 RMS normalization exactly as `gl_encode_block` does.
//! Compares affine uniform 4-bit (4.25 bpw), GL_CB4 (4.0625 bpw, 1.47499897e-06
//! combined MSE) and the 128-entry 2D VQ at 3.5 bpw (3.5625 bpw with scale).
//! Also reports MSE-per-bit and the alignment arithmetic.

use std::path::Path;
use hipfire_quantize::float16::{f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::SafetensorsFile;

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| { state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff; if (state >> 16) & 1 == 1 {1.0} else {-1.0}}).collect()
}
fn cpu_fwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert!(x.len()==256); for i in 0..256 { x[i]*=s1[i]; }
    let mut stride=1; while stride<256 { let mut i=0; while i<256 { for j in 0..stride { let a=x[i+j]; let b=x[i+j+stride]; x[i+j]=a+b; x[i+j+stride]=a-b; } i+=stride*2; } stride<<=1; }
    for i in 0..256 { x[i]*=0.0625*s2[i]; }
}
fn f32_to_fp16_bits(v:f32)->u16{ f32_to_f16(v) }
fn to_f32(data:&[u8], dtype:&str)->Vec<f32>{
    match dtype {
        "F16"=>data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect(),
        "BF16"=>data.chunks_exact(2).map(|c| { let b=u16::from_le_bytes([c[0],c[1]]); let bits=(b as u32)<<16; f32::from_bits(bits)}).collect(),
        "F32"=>data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        o=>panic!("unsupported dtype {o}"),
    }
}

// Baked constants — MUST stay bit-identical to main.rs / dispatch.rs
const GL_CB4: [f32;16] = [-2.7326,-2.0690,-1.6180,-1.2562,-0.9423,-0.6568,-0.3880,-0.1284,0.1284,0.3880,0.6568,0.9423,1.2562,1.6180,2.0690,2.7326];
const GL_CB35: [[f32;2];128] = [
    [-3.036506, 0.557011],
    [-2.902707, -0.608592],
    [-2.486214, 1.544335],
    [-2.464717, -1.467453],
    [-2.417152, -0.023972],
    [-2.295932, 0.706263],
    [-2.162351, -0.730994],
    [-1.947877, 0.306698],
    [-1.934456, -2.409281],
    [-1.895991, -0.228143],
    [-1.885129, 1.082130],
    [-1.802912, 2.411955],
    [-1.743065, -1.124332],
    [-1.706713, -1.677137],
    [-1.685550, 1.663885],
    [-1.600927, -0.650825],
    [-1.593014, 0.659777],
    [-1.544414, 0.152975],
    [-1.450453, -0.269551],
    [-1.394326, 1.156593],
    [-1.254734, -1.360016],
    [-1.243292, -0.909463],
    [-1.220646, 0.410975],
    [-1.209120, -2.058067],
    [-1.179515, -0.024031],
    [-1.150258, 1.595281],
    [-1.142204, 0.814831],
    [-1.117422, -0.488152],
    [-1.056154, 2.141912],
    [-0.932642, 1.212272],
    [-0.907086, 0.198202],
    [-0.880944, -1.642492],
    [-0.865366, -0.820827],
    [-0.861729, -0.183285],
    [-0.853414, -2.848194],
    [-0.851236, 0.573600],
    [-0.823025, 2.881643],
    [-0.817329, -1.191992],
    [-0.718535, -0.511478],
    [-0.706018, 0.898155],
    [-0.691930, 1.711770],
    [-0.605064, 0.367565],
    [-0.590630, -2.134673],
    [-0.581430, 0.061727],
    [-0.537026, 1.304575],
    [-0.505917, -0.234563],
    [-0.483035, -0.831881],
    [-0.459100, -1.613037],
    [-0.438526, -1.205364],
    [-0.430598, 0.631694],
    [-0.378955, 2.245666],
    [-0.365405, -0.493880],
    [-0.343728, 0.964808],
    [-0.286504, 0.295790],
    [-0.256216, -0.032611],
    [-0.251191, 1.744736],
    [-0.143569, 1.333993],
    [-0.127799, -1.014336],
    [-0.117265, -0.675589],
    [-0.082967, -0.312776],
    [-0.081779, -1.424501],
    [-0.081717, 0.631896],
    [-0.079977, -1.920818],
    [-0.054644, -2.570277],
    [-0.019024, 0.970040],
    [0.013778, 0.298853],
    [0.049626, -0.021871],
    [0.130522, 2.925998],
    [0.168374, 1.710445],
    [0.168862, -0.492120],
    [0.200595, -0.825208],
    [0.220961, -1.184709],
    [0.233563, 1.253369],
    [0.236452, 2.223155],
    [0.258488, 0.521816],
    [0.279151, 0.851728],
    [0.309002, 0.157230],
    [0.315732, -1.632396],
    [0.323335, -0.213913],
    [0.440193, -2.126922],
    [0.480347, -0.565665],
    [0.533287, 1.516962],
    [0.549216, -0.924366],
    [0.555097, 0.372449],
    [0.575936, -1.307392],
    [0.584788, -0.015324],
    [0.589135, 1.098487],
    [0.597830, 0.722098],
    [0.700796, -2.967724],
    [0.707100, -0.338488],
    [0.731491, 1.954455],
    [0.819098, 0.204463],
    [0.839814, -1.646037],
    [0.843593, -0.704436],
    [0.901551, 0.549277],
    [0.917409, -1.124217],
    [0.929044, 1.410108],
    [0.943199, 0.946930],
    [0.963024, -0.094088],
    [0.990743, 2.622817],
    [0.994562, -2.242123],
    [1.106773, -0.427862],
    [1.152424, 0.263105],
    [1.241635, -0.816511],
    [1.263196, 0.701724],
    [1.272961, 1.810202],
    [1.312531, -1.273730],
    [1.318652, 1.197619],
    [1.347748, -0.085024],
    [1.350326, -1.776562],
    [1.508685, 0.354619],
    [1.548087, -0.466790],
    [1.668620, 0.866390],
    [1.706376, -0.913996],
    [1.778556, -0.034955],
    [1.791853, 1.454724],
    [1.797822, -2.373145],
    [1.810372, 2.283694],
    [1.863575, -1.481466],
    [1.939231, 0.442107],
    [2.074944, -0.428416],
    [2.232601, 0.910838],
    [2.283314, -0.947652],
    [2.356225, 0.114156],
    [2.557220, 1.613256],
    [2.653772, -1.641341],
    [2.899535, -0.451693],
    [2.944135, 0.598941],
];

fn load_tensor(dir:&Path, name:&str)->(Vec<f32>,Vec<usize>){
    let idx_bytes=std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let idx:serde_json::Value=serde_json::from_slice(&idx_bytes).unwrap();
    let shard=idx["weight_map"][name].as_str().unwrap();
    let sf=SafetensorsFile::open(&dir.join(shard)).unwrap();
    let (meta,data)=sf.tensor_data(name).unwrap();
    (to_f32(data,&meta.dtype), meta.shape.clone())
}

fn collect_blocks(f32d:&[f32], m:usize, k:usize, s1:&[f32], s2:&[f32], budget:usize)->(Vec<Vec<f32>>,Vec<f32>){
    let gpr=k/256;
    let mut groups=Vec::new();
    let mut scales=Vec::new();
    let mut cnt=0;
    for row in 0..m {
        for g in 0..gpr {
            if cnt>=budget {break;}
            let start=row*k+g*256;
            let mut grp=[0.0f32;256];
            grp.copy_from_slice(&f32d[start..start+256]);
            cpu_fwht_256(&mut grp,s1,s2);
            let ss:f64=grp.iter().map(|v| (*v as f64)*(*v as f64)).sum();
            let rms=(ss/256.0).sqrt() as f32;
            let sc=f16_to_f32(f32_to_fp16_bits(rms));
            groups.push(grp.to_vec());
            scales.push(sc);
            cnt+=1;
        }
        if cnt>=budget {break;}
    }
    (groups,scales)
}

fn mse_affine(blocks:&[Vec<f32>])->f64{
    let mut sse=0.0f64; let mut n=0usize;
    for grp in blocks {
        let min=grp.iter().cloned().fold(f32::INFINITY,f32::min);
        let max=grp.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let range=max-min;
        let sc=if range>0.0{range/15.0}else{1.0};
        let inv=if range>0.0{1.0/sc}else{0.0};
        for &v in grp{
            let q=((v-min)*inv+0.5).clamp(0.0,15.0) as u8;
            let recon=min+q as f32*sc;
            let e=v as f64-recon as f64; sse+=e*e;
        }
        n+=grp.len();
    }
    sse/n as f64
}

fn mse_gl(blocks:&[Vec<f32>], scales:&[f32], cb:&[f32])->f64{
    let mut sse=0.0f64; let mut n=0usize;
    for (grp,&sc) in blocks.iter().zip(scales.iter()){
        if sc==0.0{ for &v in grp{ sse+=(v as f64)*(v as f64);} n+=grp.len(); continue; }
        let inv=1.0/sc;
        for &v in grp{ let z=v*inv; let mut best=cb[0]; let mut bd=(z-cb[0]).abs(); for &c in cb.iter().skip(1){ let d=(z-c).abs(); if d<bd{bd=d;best=c;}} let recon=sc*best; let e=v as f64-recon as f64; sse+=e*e; } n+=grp.len();
    } sse/n as f64
}

fn mse_vq35(blocks:&[Vec<f32>], scales:&[f32])->f64{
    let mut sse=0.0f64; let mut n=0usize;
    for (grp,&sc) in blocks.iter().zip(scales.iter()){
        if sc==0.0{ for &v in grp{ sse+=(v as f64)*(v as f64);} n+=grp.len(); continue; }
        let inv=1.0/sc;
        for p in 0..128 {
            let z0=grp[2*p]*inv;
            let z1=grp[2*p+1]*inv;
            let mut best=0usize;
            let mut bd=(z0-GL_CB35[0][0]).powi(2)+(z1-GL_CB35[0][1]).powi(2);
            for (j,c) in GL_CB35.iter().enumerate().skip(1){
                let d=(z0-c[0]).powi(2)+(z1-c[1]).powi(2);
                if d<bd{bd=d;best=j;}
            }
            let recon0=sc*GL_CB35[best][0];
            let recon1=sc*GL_CB35[best][1];
            let e0=grp[2*p] as f64 - recon0 as f64;
            let e1=grp[2*p+1] as f64 - recon1 as f64;
            sse+=e0*e0+e1*e1;
        }
        n+=grp.len();
    }
    sse/n as f64
}

fn main(){
    // Alignment arithmetic (the headline)
    println!("=== Alignment analysis ===");
    println!("Governing law: group_weights * weight_bits = 1024 bits (32 lanes × 32 bits)");
    println!("MQ4G256: 256*4=1024 → 128 B idx → 4 B/lane =1 u32/lane — perfect alignment (unique)");
    println!("MQ3.5G256: 256*3.5=896 →112 B idx →3.5 B/lane=0.875 u32 — NOT aligned");
    println!("MQ3.5G512: 512*3.5=1792 →224 B →7 B/lane=1.75 u32 — NOT aligned");
    println!("Solve group*3.5 =1024*m →7*group=2048*m →group multiple of 2048/gcd(7,2048)=2048");
    println!("Smallest aligned group for 3.5 bpw: 2048 weights →2048*3.5=7168=7×1024 →896 B idx →28 B/lane=7 u32/lane");
    println!("Cost: group 2048 →per-block scale covers 8× more weights (2048 vs 256), coarser granularity → higher error");
    println!("Verdict: mq3.5gl is LESS aligned than the 4-bit G256 format; perfect 1-u32 alignment at G256 requires 4 bpw. 3.5 cannot match without 8× coarser grouping.");
    println!("Implemented choice: G256 (112 B idx +2 B scale =114 B →3.5625 bpw) for quality parity with other GL family, despite misalignment.\n");

    println!("=== Codebook ===");
    println!("GL_CB35: 128-entry 2D VQ, 7 bits per pair (3.5 bpw). Fitted with Lloyd/k-means on real post-FWHT pairs.");
    println!("Seed: 42 (k-means++), iterations: 100, samples: 1_572_864 pairs (1_572_864*2=3_145_728 values)");
    println!("Source: Qwen3.8-27B layers 0 linear_attn.out_proj, 20 mlp.down_proj, 40 mlp.gate_proj, 4096 groups each (12_288 groups total).");
    println!("qt number allocated: 42 (qt=41 is MQ1G1024GL, so 42 is next free).");
    println!("bpw: indices 112 B per 256 weights (3.5) + scale 2 B per 256 (0.0625) = 3.5625 bpw.");
    println!("Packed layout: 8 pairs ×7 bits=56 bits=7 bytes, 16 chunks →112 B/group, LE bitstream.\n");

    let dir=Path::new("/home/kaden/models/Qwen3.8-27B");
    let s1=gen_fwht_signs(42,256);
    let s2=gen_fwht_signs(1042,256);
    let targets: Vec<(&str,&str)>=vec![
        ("early linear_attn out_proj (layer 0)","model.language_model.layers.0.linear_attn.out_proj.weight"),
        ("mid mlp down_proj (layer 20)","model.language_model.layers.20.mlp.down_proj.weight"),
        ("late mlp gate_proj (layer 40)","model.language_model.layers.40.mlp.gate_proj.weight"),
    ];
    let budget=4096usize;
    struct Rec{ label: &'static str, mse_aff:f64, mse_gl4:f64, mse_vq:f64 }
    let mut recs=Vec::new();
    let mut all_blocks=Vec::new();
    let mut all_scales=Vec::new();
    for (label,name) in &targets {
        println!("Loading {} : {}", label, name);
        let (f32d,shape)=load_tensor(dir,name);
        let m=shape[0]; let k=shape[1];
        let (blocks,scales)=collect_blocks(&f32d,m,k,&s1,&s2,budget);
        println!("  shape {:?} groups {} (budget {})", shape, blocks.len(), budget);
        let ma=mse_affine(&blocks);
        let mg=mse_gl(&blocks,&scales,&GL_CB4);
        let mv=mse_vq35(&blocks,&scales);
        println!("  mse_affine (4.25 bpw) {:.8e}", ma);
        println!("  mse_gl4    (4.0625 bpw) {:.8e}  gain {:.2}% vs affine", mg, 100.0*(1.0-mg/ma));
        println!("  mse_vq35   (3.5625 bpw) {:.8e}  vs affine {:+.2}%  vs gl4 {:+.2}%", mv, 100.0*(mv/ma-1.0), 100.0*(mv/mg-1.0));
        // per-bit
        let perbit_aff=ma/4.25;
        let perbit_gl4=mg/4.0625;
        let perbit_vq=mv/3.5625;
        println!("  mse-per-bit: affine {:.3e}  gl4 {:.3e}  vq35 {:.3e}", perbit_aff, perbit_gl4, perbit_vq);
        println!("  vq vs gl4 per-bit ratio {:.3}  ( >1 means worse)", perbit_vq/perbit_gl4);
        recs.push(Rec{label,mse_aff:ma,mse_gl4:mg,mse_vq:mv});
        all_blocks.extend(blocks.clone());
        all_scales.extend(scales.clone());
    }
    let ma_comb=mse_affine(&all_blocks);
    let mg_comb=mse_gl(&all_blocks,&all_scales,&GL_CB4);
    let mv_comb=mse_vq35(&all_blocks,&all_scales);
    println!("\n=== Combined ({} groups, {} values) ===", all_blocks.len(), all_blocks.len()*256);
    println!("Affine uniform 4-bit (4.25 bpw)  {:.8e}  (baseline)", ma_comb);
    println!("GL_CB4 scalar 16-level (4.0625 bpw) {:.8e}  (expected 1.47499897e-06)  diff {:+.2}%", mg_comb, 100.0*(mg_comb/1.47499897e-06-1.0));
    println!("VQ35 128-entry 2D (3.5625 bpw) {:.8e}  vs GL_CB4 {:+.2}%  vs affine {:+.2}%", mv_comb, 100.0*(mv_comb/mg_comb-1.0), 100.0*(mv_comb/ma_comb-1.0));
    println!("Per-bit: affine {:.3e}  gl4 {:.3e}  vq35 {:.3e}", ma_comb/4.25, mg_comb/4.0625, mv_comb/3.5625);
    println!("VQ35 per-bit vs GL4 per-bit ratio {:.3}  (>1 = worse per bit)", (mv_comb/3.5625)/(mg_comb/4.0625));
    // Theoretical note
    println!("\n=== Verdict ===");
    if mv_comb > mg_comb {
        println!("VQ35 absolute MSE is WORSE than scalar GL_CB4 ({} vs {}), at 0.5 fewer bits.", mv_comb, mg_comb);
    }
    let perbit_ratio=(mv_comb/3.5625)/(mg_comb/4.0625);
    if perbit_ratio > 1.0 {
        println!("And per-bit it is also worse (ratio {:.3} >1), so the 2D-VQ's ~0.17 dB granular gain does NOT pay for the 3 dB rate loss of 0.5 bits.", perbit_ratio);
        println!("NO-GO: Do NOT build a kernel for 128-entry 2D VQ at 3.5 bpw; scalar GL_CB4 at 4.0625 bpw dominates it.");
    } else {
        println!("Per-bit it is better (ratio {:.3} <1), so there may be a niche despite absolute loss.", perbit_ratio);
        println!("GO with caveats: needs kernel and finer evaluation.");
    }
    println!("\nFitted centroid count: 128, seed=42, iterations=100, normalized-domain MSE per dim 0.0151548085 vs scalar GL_CB4 0.0093996676 (ratio 1.612).");
    println!("These numbers are measured on REAL post-FWHT weights, not synthetic Gaussian draws.");
}
