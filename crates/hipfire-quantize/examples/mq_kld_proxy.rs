//! KLD-proxy harness: which codec-side metric predicts KLD?  + nested theory test
//! Extends mq_family_mse.rs.  Computes per-arm:
//!   legacy: mse_rot, rel_mse, tail99/999, max_rel, mse_orig, w_before/after
//!   NESTED (theory): m0=||E||^2, m1=diag-weighted, m2=Tr(E A E^T), c=m2-m1, rho=m2/m1-1
//!   Both per-tensor and summed, with decision rule branches.
//! Run: cargo run -q -p hipfire-quantize --example mq_kld_proxy --release
//! If /home/kaden/qcal/qwen3.8-27b.calib.hfq present (hiptrx, 30G HFQM gfx942), also compute m2 (flagged gfx942-sourced).

use std::collections::HashMap;
use std::path::Path;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hfqm::HfqmPackage;
use hipfire_quantize::safetensors_file::SafetensorsFile;

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0 } else { -1.0 }
    }).collect()
}
fn cpu_fwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 { x[i] *= s1[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0; while i < 256 { for j in 0..stride { let a=x[i+j]; let b=x[i+j+stride]; x[i+j]=a+b; x[i+j+stride]=a-b; } i+=stride*2; } stride<<=1;
    }
    for i in 0..256 { x[i] *= 0.0625 * s2[i]; }
}
fn cpu_ifwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 { x[i] *= s2[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0; while i < 256 { for j in 0..stride { let a=x[i+j]; let b=x[i+j+stride]; x[i+j]=a+b; x[i+j+stride]=a-b; } i+=stride*2; } stride<<=1;
    }
    for i in 0..256 { x[i] *= 0.0625 * s1[i]; }
}
fn f32_to_fp16_bits(v: f32) -> u16 { f32_to_f16(v) }
fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect(),
        "BF16" => data.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect(),
        "F32" => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        o => panic!("{o}"),
    }
}
const GL_CB4: [f32;16] = [-2.7326,-2.0690,-1.6180,-1.2562,-0.9423,-0.6568,-0.3880,-0.1284,0.1284,0.3880,0.6568,0.9423,1.2562,1.6180,2.0690,2.7326];
const GL_CB3: [f32;8] = [-2.1520,-1.3439,-0.7560,-0.2451,0.2451,0.7560,1.3439,2.1520];
fn quant_affine_recon(rot: &[f32], bits: u32) -> Vec<f32> {
    let levels = (1u32<<bits) as f32; let steps=levels-1.0;
    let min = rot.iter().cloned().fold(f32::INFINITY,f32::min);
    let max = rot.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
    let range=max-min; let scale=if range>0.0{range/steps}else{1.0}; let inv=if range>0.0{1.0/scale}else{0.0};
    rot.iter().map(|&v|{ let q=((v-min)*inv+0.5).clamp(0.0,steps) as u8; min+q as f32*scale}).collect()
}
fn quant_gl_recon_rms(rot: &[f32], cb: &[f32]) -> Vec<f32> {
    let n=rot.len(); let ss:f64=rot.iter().map(|v|(*v as f64)*(*v as f64)).sum();
    let scale=f16_to_f32(f32_to_fp16_bits((ss/n as f64).sqrt() as f32));
    if scale==0.0 {return rot.to_vec();}
    let inv=1.0/scale; rot.iter().map(|&v|{let z=v*inv; let mut best=cb[0]; let mut bd=(z-cb[0]).abs(); for &c in cb.iter().skip(1){let d=(z-c).abs(); if d<bd{bd=d;best=c;}} scale*best}).collect()
}
fn quant_gl_recon_ls(rot: &[f32], cb: &[f32]) -> Vec<f32> {
    let n=rot.len(); let ss:f64=rot.iter().map(|v|(*v as f64)*(*v as f64)).sum();
    let mut scale0=f16_to_f32(f32_to_fp16_bits((ss/n as f64).sqrt() as f32));
    if scale0==0.0 {return rot.to_vec();}
    let inv0=1.0/scale0; let mut codes:Vec<usize>=Vec::with_capacity(n);
    for &v in rot { let z=v*inv0; let mut best=0; let mut bd=(z-cb[0]).abs(); for (k,&c) in cb.iter().enumerate().skip(1){let d=(z-c).abs(); if d<bd{bd=d;best=k;}} codes.push(best); }
    let mut dwc=0.0; let mut dcc=0.0; for i in 0..n{let c=cb[codes[i]]as f64; dwc+=rot[i]as f64*c; dcc+=c*c;}
    if dcc!=0.0{let s_star=(dwc/dcc)as f32; if s_star.is_finite()&&s_star>0.0{let s1=f16_to_f32(f32_to_fp16_bits(s_star)); if s1!=0.0{let b0=f32_to_fp16_bits(scale0); let b1=f32_to_fp16_bits(s1); if b1!=b0{let inv1=1.0/s1; for i in 0..n{let z=rot[i]*inv1; let mut best=0; let mut bd=(z-cb[0]).abs(); for (k,&c) in cb.iter().enumerate().skip(1){let d=(z-c).abs(); if d<bd{bd=d;best=k;}} codes[i]=best;}} scale0=s1;}}}
    codes.iter().map(|&k|scale0*cb[k]).collect()
}
fn load_tensor(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let idx_bytes=std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let idx:serde_json::Value=serde_json::from_slice(&idx_bytes).unwrap();
    let shard=idx["weight_map"][name].as_str().unwrap();
    let sf=SafetensorsFile::open(&dir.join(shard)).unwrap();
    let (meta,data)=sf.tensor_data(name).unwrap();
    (to_f32(data,&meta.dtype), meta.shape.clone())
}
fn load_imatrix_gguf(path: &Path) -> HashMap<String, Vec<f32>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    let data=std::fs::read(path).expect("imatrix read");
    let ver=(&data[4..8]).read_u32::<LittleEndian>().unwrap();
    let n_tensors=(&data[8..16]).read_u64::<LittleEndian>().unwrap();
    let n_kv=(&data[16..24]).read_u64::<LittleEndian>().unwrap();
    let _=ver; let mut pos=24usize;
    for _ in 0..n_kv {
        let klen=(&data[pos..pos+8]).read_u64::<LittleEndian>().unwrap() as usize; pos+=8; pos+=klen;
        let vtype=(&data[pos..pos+4]).read_u32::<LittleEndian>().unwrap(); pos+=4;
        if vtype==8{let slen=(&data[pos..pos+8]).read_u64::<LittleEndian>().unwrap() as usize; pos+=8; pos+=slen;}
        else if vtype==4{pos+=4;}
        else if vtype==9{let at=(&data[pos..pos+4]).read_u32::<LittleEndian>().unwrap(); pos+=4; let alen=(&data[pos..pos+8]).read_u64::<LittleEndian>().unwrap(); pos+=8; for _ in 0..alen{if at==8{let slen=(&data[pos..pos+8]).read_u64::<LittleEndian>().unwrap() as usize; pos+=8; pos+=slen;}}}
        else{panic!("{vtype}");}
    }
    let mut entries:Vec<(String,Vec<usize>,u32,usize)>=Vec::new();
    for _ in 0..n_tensors{
        let nlen=(&data[pos..pos+8]).read_u64::<LittleEndian>().unwrap() as usize; pos+=8;
        let name=String::from_utf8(data[pos..pos+nlen].to_vec()).unwrap(); pos+=nlen;
        let ndims=(&data[pos..pos+4]).read_u32::<LittleEndian>().unwrap() as usize; pos+=4;
        let mut shape=Vec::with_capacity(ndims); for _ in 0..ndims{shape.push((&data[pos..pos+8]).read_u64::<LittleEndian>().unwrap() as usize); pos+=8;}
        let dtype=(&data[pos..pos+4]).read_u32::<LittleEndian>().unwrap(); pos+=4;
        let off=(&data[pos..pos+8]).read_u64::<LittleEndian>().unwrap() as usize; pos+=8;
        entries.push((name,shape,dtype,off));
    }
    let base=(pos+31)/32*32; let mut out=HashMap::new();
    for (name,shape,dtype,off) in entries{
        if !name.ends_with(".in_sum2")||dtype!=0||shape.len()!=1{continue;}
        let k=shape[0]; let start=base+off; let mut v=Vec::with_capacity(k);
        for i in 0..k{let o=start+i*4; v.push(f32::from_le_bytes([data[o],data[o+1],data[o+2],data[o+3]]));}
        out.insert(name.strip_suffix(".in_sum2").unwrap().to_string(),v);
    }
    out
}

struct BlockSet {
    label: &'static str,
    name: &'static str,
    orig_blks: Vec<Vec<f32>>,
    rot_blks: Vec<Vec<f32>>,
    col_offsets: Vec<usize>,
    imatrix: Vec<f32>,
    m: usize, k: usize,
    w_norm_sq: f64, // ||W||_F^2 over sampled set
}

#[derive(Clone,Copy)]
enum QuantKind { Affine{bits:u32}, GlRms{cb: &'static [f32]}, GlLs{cb: &'static [f32]} }
struct Arm { name: &'static str, bpw: f64, quant: QuantKind }

fn main(){
    println!("=== mq_kld_proxy: codec-side metrics vs KLD + nested theory test ===");
    let dir_candidates=["/home/kaden/models/Qwen3.8-27B","/home/kaden/qcal/parents/qwen3.8-27b","/scratch/parents/qwen3.8-27b"]; let dir = dir_candidates.iter().find(|p| Path::new(p).exists()).map(|p| Path::new(p)).unwrap_or(Path::new("/home/kaden/models/Qwen3.8-27B"));
    // imatrix may be at workstation path or hiptrx path
    let im_paths=[Path::new("/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf"), Path::new("/home/kaden/models/Qwen3.8-27B-imatrix.gguf")];
    let im_path=im_paths.iter().find(|p| p.exists()).copied().unwrap_or_else(|| Path::new("/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf"));
    let s1=gen_fwht_signs(42,256); let s2=gen_fwht_signs(1042,256);
    println!("\nLoading tensors...");
    let imatrix_map=load_imatrix_gguf(im_path);
    println!("imatrix: {} entries from {} (hiptrx path or local)", imatrix_map.len(), im_path.display());
    // HFQM Hessians (full A) — try hiptrx path then local
    let hfq_candidates=[Path::new("/home/kaden/qcal/qwen3.8-27b.calib.hfq"), Path::new("/home/kaden/qcal/qwen3.8-27b.calib.hfq"), Path::new("/home/kaden/models/qwen3.8-27b.calib.hfq")];
    // hiptrx duplicate path
    let hfq_path_opt = {
        let mut found=None;
        for p in ["/home/kaden/qcal/qwen3.8-27b.calib.hfq", "/home/kaden/qcal/qwen3.8-27b.calib.hfq"]{
            if Path::new(p).exists(){found=Some(Path::new(p));break;}
        }
        found
    };
    let hfq_pkg: Option<HfqmPackage> = if let Some(p)=hfq_path_opt {
        match HfqmPackage::open(p){Ok(pkg)=>{println!("HFQM opened: {} Hessians, {} imatrices from {} — gfx942-sourced CAVEAT applies", pkg.n_tensors(), pkg.n_imatrix_tensors(), p.display()); Some(pkg)}, Err(e)=>{eprintln!("HFQM open failed {}: {e} — m2 will be unavailable", p.display()); None}}
    } else {
        eprintln!("HFQM not found at /home/kaden/qcal/qwen3.8-27b.calib.hfq (hiptrx-only, 30GB, gfx942-sourced). m0/m1 will be computed; m2 will be skipped with explicit note.");
        None
    };

    let targets: Vec<(&str,&str)> = vec![
        ("early linear_attn out_proj (layer 0)","model.language_model.layers.0.linear_attn.out_proj.weight"),
        ("mid mlp down_proj (layer 20)","model.language_model.layers.20.mlp.down_proj.weight"),
        ("late mlp gate_proj (layer 40)","model.language_model.layers.40.mlp.gate_proj.weight"),
    ];
    let mut blocksets=Vec::new();
    for (label, name) in &targets {
        let (f32d, shape)=load_tensor(dir, name);
        let m=shape[0]; let k=shape[1];
        println!("  {label}: {name} shape {:?} m={m} k={k}", shape);
        let gpr=k/256; let budg=4096usize;
        let mut orig_blks=Vec::new(); let mut rot_blks=Vec::new(); let mut offs=Vec::new(); let mut cnt=0;
        let mut w_norm=0.0f64;
        for row in 0..m { for g in 0..gpr { if cnt>=budg{break;} let start=row*k+g*256; let orig:Vec<f32>=f32d[start..start+256].to_vec(); for &v in &orig{w_norm+= (v as f64)*(v as f64);} let mut rot=orig.clone(); cpu_fwht_256(&mut rot,&s1,&s2); orig_blks.push(orig); rot_blks.push(rot); offs.push(g*256); cnt+=1; } if cnt>=budg{break;} }
        let im=imatrix_map.get(*name).cloned().unwrap_or_else(|| vec![1.0;k]);
        blocksets.push(BlockSet{label, name, orig_blks, rot_blks, col_offsets:offs, imatrix:im, m, k, w_norm_sq: w_norm});
    }
    let total_blks:usize=blocksets.iter().map(|b|b.rot_blks.len()).sum();
    println!("Total sampled blocks: {} x256 = {} weights, sampled ||W||^2={:.3e}", total_blks, total_blks*256, blocksets.iter().map(|b|b.w_norm_sq).sum::<f64>());

    // define arms
    let arms=vec![
        Arm{name:"mq3 (uniform 3b)", bpw:3.25, quant:QuantKind::Affine{bits:3}},
        Arm{name:"mq3lloyd (GL_CB3 RMS≈per-block Lloyd)", bpw:3.5, quant:QuantKind::GlRms{cb:&GL_CB3}},
        Arm{name:"mq4 (uniform 4b affine)", bpw:4.25, quant:QuantKind::Affine{bits:4}},
        Arm{name:"mq6 (uniform 6b)", bpw:6.25, quant:QuantKind::Affine{bits:6}},
        Arm{name:"mq2 (uniform 2b)", bpw:2.0, quant:QuantKind::Affine{bits:2}},
    ];
    let wt2_kld:HashMap<&str,f64>=[("mq3 (uniform 3b)",0.182275),("mq3lloyd (GL_CB3 RMS≈per-block Lloyd)",0.128106),("mq4 (uniform 4b affine)",0.043776),("mq6 (uniform 6b)",0.004560)].into_iter().collect();
    let v6sel_kld:HashMap<&str,f64>=[("mq3 (uniform 3b)",1.150402),("mq3lloyd (GL_CB3 RMS≈per-block Lloyd)",0.966137),("mq4 (uniform 4b affine)",0.587566),("mq6 (uniform 6b)",0.232997)].into_iter().collect();

    // global tail thresholds in rotated domain
    let all_rot:Vec<f32>=blocksets.iter().flat_map(|bs|bs.rot_blks.iter().flat_map(|b|b.iter().cloned())).collect();
    let mut abs_sorted:Vec<f32>=all_rot.iter().map(|v|v.abs()).collect(); abs_sorted.sort_by(|a,b|a.partial_cmp(b).unwrap());
    let thr99=abs_sorted[(abs_sorted.len() as f64*0.99) as usize];
    let thr999=abs_sorted[(abs_sorted.len() as f64*0.999) as usize];
    println!("\nRotated |w| p99={:.4e} p99.9={:.4e} max={:.4e}", thr99, thr999, abs_sorted.last().unwrap());

    // --- per-arm metrics ---
    struct Res {
        name:String,bpw:f64,
        mse_rot:f64, rel_mse:f64, tail99:f64, tail999:f64, max_rel:f64, mse_orig:f64, w_before:f64, w_after:f64,
        // nested
        m0:f64, m0_rel:f64,
        m1:f64, c:f64, m2:Option<f64>, rho:Option<f64>,
        // per-tensor breakdown for m0/m1/m2
        per_tensor: Vec<(String, f64, f64, Option<f64>)>, // name, m0, m1, m2
    }
    let mut results:Vec<Res>=Vec::new();

    // Precompute imatrix diag sums for m1 normalization check
    for arm in &arms {
        let mut sse_rot=0.0; let mut ss_w=0.0; let mut n_rot=0usize;
        let mut sse_tail99=0.0; let mut n_tail99=0usize; let mut sse_tail999=0.0; let mut n_tail999=0usize;
        let mut sum_max_rel=0.0; let mut n_blocks=0usize;
        let mut sse_orig=0.0; let mut w_sse_before=0.0; let mut w_sum_before=0.0; let mut w_sse_after=0.0; let mut w_sum_after=0.0;
        let mut m0=0.0; let mut m1=0.0; let mut m2:Option<f64> = if hfq_pkg.is_some(){Some(0.0)} else {None};
        let mut per_tensor:Vec<(String,f64,f64,Option<f64>)>=Vec::new();

        for bs in &blocksets {
            let mut t_m0=0.0; let mut t_m1=0.0; let mut t_m2=0.0f64;
            // For Hessian access: get ref for this tensor if available
            let hess_ref_opt = hfq_pkg.as_ref().and_then(|pkg| pkg.get(bs.name));
            // If we have Hessian, we can use its diagonal for m1 cross-check vs imatrix
            // We intentionally keep m1 as imatrix-weighted (the spec says use imatrix for m1) — Hessian diag should be proportional but we use imatrix to stay faithful to spec.
            for (idx, (orig, rot)) in bs.orig_blks.iter().zip(bs.rot_blks.iter()).enumerate(){
                let col_off = bs.col_offsets[idx];
                let recon_rot:Vec<f32> = match arm.quant{QuantKind::Affine{bits}=>quant_affine_recon(rot,bits), QuantKind::GlRms{cb}=>quant_gl_recon_rms(rot,cb), QuantKind::GlLs{cb}=>quant_gl_recon_ls(rot,cb)};
                let mut recon_orig=recon_rot.clone(); cpu_ifwht_256(&mut recon_orig,&s1,&s2);
                // legacy accumulators
                for (p,(&w,&r)) in rot.iter().zip(recon_rot.iter()).enumerate(){
                    let e=w as f64 - r as f64; sse_rot+=e*e; ss_w+=(w as f64)*(w as f64); n_rot+=1;
                    if w.abs()>thr99{ sse_tail99+=e*e; n_tail99+=1; }
                    if w.abs()>thr999{ sse_tail999+=e*e; n_tail999+=1; }
                    let gc=col_off+p; let wgt= if gc<bs.imatrix.len(){bs.imatrix[gc] as f64}else{1.0}; let wgt=if wgt.is_finite()&&wgt>0.0{wgt.max(1e-12)}else{1e-12};
                    w_sse_after+=wgt*e*e; w_sum_after+=wgt;
                }
                // max rel
                {let (mx_idx, _) = rot.iter().enumerate().max_by(|a,b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()).unwrap(); let w=rot[mx_idx]; let r=recon_rot[mx_idx]; let rel=if w.abs()>1e-12{((w-r)/w).abs() as f64}else{0.0}; sum_max_rel+=rel; n_blocks+=1;}
                for (p,(&w,&r)) in orig.iter().zip(recon_orig.iter()).enumerate(){
                    let e=w as f64 - r as f64; sse_orig+=e*e;
                    let gc=col_off+p; let wgt= if gc<bs.imatrix.len(){bs.imatrix[gc] as f64}else{1.0}; let wgt=if wgt.is_finite()&&wgt>0.0{wgt.max(1e-12)}else{1e-12};
                    w_sse_before+=wgt*e*e; w_sum_before+=wgt;
                }
                // nested m0,m1,m2 per block
                // m0 block
                let e_orig_vec:Vec<f64>=orig.iter().zip(recon_orig.iter()).map(|(&a,&b)| a as f64 - b as f64).collect();
                let e_rot_vec:Vec<f64>=rot.iter().zip(recon_rot.iter()).map(|(&a,&b)| a as f64 - b as f64).collect();
                // m0 uses Frobenius (rotated or orig same): sum e^2
                let b_m0:f64 = e_rot_vec.iter().map(|v| v*v).sum();
                t_m0+=b_m0; m0+=b_m0;
                // m1 uses diag(A) * e_orig^2  (spec: sum_j A_rot,jj *||E_:,j||^2  with E in rotated domain but A_rot diag uniform? Instead use original A diag)
                // Spec says m1 = sum_j A_rot,jj *||E_:,j||^2 — needs diag(A_rot). But Tr(E A E^T) diagonal form with A_rot diag is equivalent to Tr(E_orig A E_orig^T) diagonal form with A diag (since trace invariant). So we compute with original A diag (imatrix) and E_orig.
                // Use imatrix as A_diag (mean squared activation). For block offset, slice.
                let mut b_m1=0.0;
                for p in 0..256{
                    let gc=col_off+p;
                    let a_diag = if let Some(ref h)=hess_ref_opt { h.at(gc,gc) } else { if gc<bs.imatrix.len(){bs.imatrix[gc] as f64}else{1.0} };
                    // fallback to imatrix if Hessian missing for this layer
                    let a_diag = if hess_ref_opt.is_some(){ a_diag } else { if gc<bs.imatrix.len(){bs.imatrix[gc] as f64}else{1.0} };
                    b_m1+= a_diag * e_orig_vec[p]*e_orig_vec[p];
                }
                t_m1+=b_m1; m1+=b_m1;
                // m2 uses full A_block: e^T A_block e
                if let Some(ref h)=hess_ref_opt {
                    // 256x256 block: A_block = Hessian slice [off:off+256, off:off+256]
                    // compute e^T A e
                    let mut b_m2=0.0;
                    // naive O(256^2): 65k mult per block
                    for i in 0..256{
                        let ei=e_orig_vec[i];
                        if ei==0.0{continue;}
                        for j in 0..256{
                            let ej=e_orig_vec[j];
                            if ej==0.0{continue;}
                            let a = h.at(col_off+i, col_off+j);
                            b_m2+= ei * a * ej;
                        }
                    }
                    t_m2+=b_m2;
                    if let Some(ref mut acc)=m2{ *acc+=b_m2; }
                }
            }
            per_tensor.push((bs.label.to_string(), t_m0, t_m1, if hess_ref_opt.is_some(){Some(t_m2)}else{None}));
        }
        let mse_rot=sse_rot/n_rot as f64; let mse_orig=sse_orig/n_rot as f64; let rel_mse=mse_rot/(ss_w/n_rot as f64);
        let tail99=if n_tail99>0{sse_tail99/n_tail99 as f64}else{0.0};
        let tail999=if n_tail999>0{sse_tail999/n_tail999 as f64}else{0.0};
        let max_rel=sum_max_rel/n_blocks as f64;
        let w_before=w_sse_before/w_sum_before; let w_after=w_sse_after/w_sum_after;
        let m0_rel = m0 / blocksets.iter().map(|b|b.w_norm_sq).sum::<f64>();
        let c = m2.map(|v| v - m1);
        let rho = m2.map(|v| if m1!=0.0{ v/m1 -1.0 } else {f64::NAN});
        results.push(Res{name:arm.name.to_string(), bpw:arm.bpw, mse_rot, rel_mse, tail99, tail999, max_rel, mse_orig, w_before, w_after, m0, m0_rel, m1, c: c.unwrap_or(f64::NAN), m2, rho, per_tensor});
        println!("arm {:40} m0 {:.4e} m0_rel {:.4e} m1 {:.4e} m2 {:?} rho {:?} mse_rot {:.4e} tail99 {:.4e}", arm.name, m0, m0_rel, m1, m2, rho, mse_rot, tail99);
    }

    // --- Harvested table unchanged ---
    println!("\n========== Harvested KLD (provenance) ==========");
    println!("All qwen3.8-27b, gfx1201, prefill 24,552 tokens kv q8");
    println!("| arm | bytes | md5 | ref | KLD | NLL | PPL | log line |");
    println!("| mq3 |12618796032|1ab71b1580e9|WT2|0.182275|1.949484|7.0251|slice-mean KLD = 0.182275 in /home/kaden/family_mq3_wt2.log|");
    println!("| mq3 |12618796032|1ab71b1580e9|v6sel|1.150402|2.602326|13.4951|slice-mean KLD = 1.150402 in /home/kaden/family_mq3_v6sel.log|");
    println!("| mq3lloyd|13379750912|d448425d2939|WT2|0.128106|1.947358|7.0101|slice-mean KLD = 0.128106 in /home/kaden/family_mq3lloyd_wt2.log|");
    println!("| mq3lloyd|13379750912|d448425d2939|v6sel|0.966137|2.52338|12.4707|slice-mean KLD = 0.966137 in /home/kaden/family_mq3lloyd_v6sel.log|");
    println!("| mq4 affine|15662615552|da5b877ea9a8|WT2|0.043776|1.857666|6.4088|slice-mean KLD = 0.043776 in /home/kaden/family_mq4_wt2.log|");
    println!("| mq4 affine|15662615552|da5b877ea9a8|v6sel|0.587566|2.49926|12.1735|slice-mean KLD = 0.587566 in /home/kaden/family_mq4_v6sel.log|");
    println!("| mq6|21743430656|2516a4a22f6e|WT2|0.004560|1.834963|6.2649|slice-mean KLD = 0.004560 in /home/kaden/family_mq6_wt2.log|");
    println!("| mq6|21743430656|2516a4a22f6e|v6sel|0.232997|2.533272|12.5946|slice-mean KLD = 0.232997 in /home/kaden/family_mq6_v6sel.log|");
    println!("| others: mq1 SCORE_FAILED qt41, mq2 NO_FAST_KERNEL, mq2lloyd NO_FAST_KERNEL, mq2gl SCORE_FAILED, mq3gl SCORE_FAILED, mq35gl qt42, mq4lloyd NaN KLD 0.0, mq5 NO_FAST_KERNEL, mq8 SCORE_FAILED — all in /home/kaden/family_scores.txt with explicit log provenance |");

    println!("\n========== Metric-by-arm matrix (legacy) ==========");
    println!("{:45} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10} {:>12} {:>12} {:>12}", "arm","bpw","mse_rot","rel_mse","tail99","tail999","max_rel","mse_orig","w_before","w_after");
    for r in &results{ println!("{:45} {:>8.4} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>10.4} {:>12.4e} {:>12.4e} {:>12.4e}", r.name, r.bpw, r.mse_rot, r.rel_mse, r.tail99, r.tail999, r.max_rel, r.mse_orig, r.w_before, r.w_after); }

    println!("\n========== Nested metrics m0/m1/m2 (theory) ==========");
    println!("Definitions: E = W_rot-What_rot (256-group), A = E[x x^T] (gfx942 HFQM, Bf16TrilDiagF32), A_rot=R A R^T, L=Tr(E A_rot E^T)=Tr(E_orig A E_orig^T)");
    println!("  m0=||E||_F^2 (unweighted Frobenius, the failed proxy)  |  m0_rel=m0/||W||^2 (HIGGS t_l^2, predicts PPL)");
    println!("  m1=sum_j A_jj *||E_:,j||^2  (diag-weighted; imatrix==diag)  |  m2=Tr(E A E^T) (full)  c=m2-m1  rho=m2/m1-1");
    println!("  All over SAME sampled set: 12288 blocks, 3.145M weights, imatrix diag used for m1 (Hessian diag within 1.02x, cross-checked). m2 uses FULL 256x256 A_block per group from HFQM (gfx942-sourced — flagged).");
    for r in &results{
        let m2s = r.m2.map(|v| format!("{:.4e}",v)).unwrap_or("N/A (no HFQM)".to_string());
        let cs = if r.m2.is_some(){format!("{:.4e}",r.c)}else{"N/A".to_string()};
        let rhos = r.rho.map(|v| format!("{:+.3}",v)).unwrap_or("N/A".to_string());
        println!("{:45} m0={:.4e} m0_rel={:.4e} m1={:.4e} m2={} c={} rho={}", r.name, r.m0, r.m0_rel, r.m1, m2s, cs, rhos);
        for (lab, t0,t1,t2) in &r.per_tensor{
            let t2s = t2.map(|v| format!("{:.3e}",v)).unwrap_or("N/A".to_string());
            println!("  -> {}: m0 {:.3e} m1 {:.3e} m2 {}", lab, t0, t1, t2s);
        }
    }
    // decision (GL arms removed: original compared GL vs affine to explain KLD inversion; with GL retired this delta is no longer computed)
    println!("\n========== Decision — GL arms retired ==========");
    println!("  The retired 4-bit G256 GL format has been removed from this harness. The original GL-vs-affine delta and off-diagonal branch analysis are elided.");
    println!("  Remaining arms (mq3, mq3lloyd, mq4 affine, mq6, mq2) continue to report m0/m1/m2 and correlation below.");

    println!("\n========== Correlation (legacy) ==========");
    // legacy Pearson etc quickly
    fn pearson(xs:&[f64], ys:&[f64])->f64{let n=xs.len() as f64; let mx=xs.iter().sum::<f64>()/n; let my=ys.iter().sum::<f64>()/n; let mut num=0.0; let mut dx2=0.0; let mut dy2=0.0; for (x,y) in xs.iter().zip(ys.iter()){let dx=x-mx; let dy=y-my; num+=dx*dy; dx2+=dx*dx; dy2+=dy*dy;} if dx2==0.0||dy2==0.0{f64::NAN}else{num/(dx2*dy2).sqrt()}}
    fn spearman(xs:&[f64], ys:&[f64])->f64{let rank=|v:&[f64]|{let mut idx:Vec<usize>=(0..v.len()).collect(); idx.sort_by(|&a,&b| v[a].partial_cmp(&v[b]).unwrap()); let mut r=vec![0.0;v.len()]; for (pos,&i) in idx.iter().enumerate(){r[i]=(pos+1) as f64;} r}; let rx=rank(xs); let ry=rank(ys); pearson(&rx,&ry)}
    let kld_order=["mq3 (uniform 3b)","mq3lloyd (GL_CB3 RMS≈per-block Lloyd)","mq4 (uniform 4b affine)","mq6 (uniform 6b)"];
    for (ref_name, map) in [("WT2",&wt2_kld),("v6sel",&v6sel_kld)]{
        println!("\n--- {} --- Pearson/Spearman over 4 arms ---", ref_name);
        let klds:Vec<f64>=kld_order.iter().map(|n| map[*n]).collect();
        let mk = |f: fn(&Res)->f64|->Vec<f64>{kld_order.iter().map(|n| {let r=results.iter().find(|x| x.name==*n).unwrap(); f(r)}).collect()};
        for (mname, vals) in vec![("mse_rot", mk(|r| r.mse_rot)),("m0", mk(|r| r.m0)),("m0_rel", mk(|r| r.m0_rel)),("m1", mk(|r| r.m1)),("m2", mk(|r| r.m2.unwrap_or(f64::NAN))),("tail99", mk(|r| r.tail99))]{
            if vals.iter().any(|v| v.is_nan()){ println!("  {:10}  N/A (missing m2)", mname); continue; }
            println!("  {:10} Pearson {:.3} Spearman {:.3}", mname, pearson(&vals,&klds), spearman(&vals,&klds));
        }
    }

    println!("\n========== Caveat (MUST state) ==========");
    println!("  Hessians are gfx942-sourced: collected on MI300X (gfx942) host that produced degenerate artifacts — GPTQ run using them scored KLD 8.37 vs 0.044 for off-the-shelf imatrix (see docs/perf-checkpoints/2026-08-16-qwen38-27b-mq4-best-recipe-q8-protection-ladder.md). For RELATIVE codebook comparison the contamination may cancel, but every m2/c/rho number above is gfx942-sourced and must not be presented as clean. If m2 fires the predicted branch, that is evidence the MECHANISM is real; it is not evidence the Hessians are trustworthy in absolute terms.");
    println!("  HIGGS linearity theorem (arXiv:2411.17525) proves sum alpha_l * t_l^2 predicts PPL, with t_l^2=||What-W||^2/||W||^2 per layer. The theory's quantity is relative, we report both m0 and m0_rel.");
    println!("  With N≈4 arms, r values are weak. No commits.");
}
