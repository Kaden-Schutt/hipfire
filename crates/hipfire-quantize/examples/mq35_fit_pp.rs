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
    match dtype {"F16"=>data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect(),"BF16"=>data.chunks_exact(2).map(|c| { let b=u16::from_le_bytes([c[0],c[1]]); let bits=(b as u32)<<16; f32::from_bits(bits)}).collect(),"F32"=>data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),o=>panic!("{o}")}
}
fn load_tensor(dir:&Path, name:&str)->(Vec<f32>,Vec<usize>){
    let idx_bytes=std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let idx:serde_json::Value=serde_json::from_slice(&idx_bytes).unwrap();
    let shard=idx["weight_map"][name].as_str().unwrap();
    let sf=SafetensorsFile::open(&dir.join(shard)).unwrap();
    let (meta,data)=sf.tensor_data(name).unwrap();
    (to_f32(data,&meta.dtype), meta.shape.clone())
}
fn collect_pair_samples(f32d:&[f32], m:usize, k:usize, s1:&[f32], s2:&[f32], budget:usize)->Vec<[f32;2]> {
    let gpr=k/256;
    let mut pairs=Vec::new();
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
            if sc>0.0 {
                let inv=1.0/sc;
                for i in (0..256).step_by(2) { pairs.push([grp[i]*inv, grp[i+1]*inv]); }
            } else { for _ in 0..128 { pairs.push([0.0,0.0]); } }
            cnt+=1;
        }
        if cnt>=budget {break;}
    }
    pairs
}
fn lcg_next(state: &mut u64)->u64 { *state = state.wrapping_mul(6364136223846793005).wrapping_add(1); *state }
fn kmeans_pp(samples:&[[f32;2]], k:usize, seed:u64)->Vec<[f32;2]> {
    let n=samples.len();
    let mut centroids=Vec::with_capacity(k);
    let mut state=seed;
    let first=(lcg_next(&mut state) as usize)%n;
    centroids.push(samples[first]);
    let mut dists=vec![f32::INFINITY; n];
    for _ in 1..k {
        let last=centroids.last().unwrap();
        for (i,s) in samples.iter().enumerate() {
            let d=(s[0]-last[0]).powi(2)+(s[1]-last[1]).powi(2);
            if d < dists[i] { dists[i]=d; }
        }
        // weighted sampling: cumulative sum
        let sum:f64=dists.iter().map(|&d| d as f64).sum();
        let r=(lcg_next(&mut state) as f64)/(u64::MAX as f64) * sum;
        let mut cum=0.0;
        let mut chosen=n-1;
        for (i,&d) in dists.iter().enumerate() {
            cum+=d as f64;
            if cum>=r { chosen=i; break; }
        }
        centroids.push(samples[chosen]);
    }
    centroids
}
fn kmeans_2d(samples:&[[f32;2]], k:usize, iters:usize, seed:u64)->Vec<[f32;2]> {
    let n=samples.len();
    let mut centroids=kmeans_pp(samples,k,seed);
    // also add deterministic tie: sort?
    for iter in 0..iters {
        let mut sums=vec![[0.0f64;2]; k];
        let mut counts=vec![0u64; k];
        for &s in samples {
            let mut best=0;
            let mut best_d=(s[0]-centroids[0][0]).powi(2)+(s[1]-centroids[0][1]).powi(2);
            for j in 1..k { let d=(s[0]-centroids[j][0]).powi(2)+(s[1]-centroids[j][1]).powi(2); if d<best_d {best_d=d;best=j;} }
            sums[best][0]+=s[0] as f64; sums[best][1]+=s[1] as f64; counts[best]+=1;
        }
        let mut moved=false;
        let mut state=seed.wrapping_add(iter as u64 * 0x9E3779B97F4A7C15);
        for j in 0..k {
            if counts[j]>0 {
                let nx=(sums[j][0]/counts[j] as f64) as f32;
                let ny=(sums[j][1]/counts[j] as f64) as f32;
                if (nx-centroids[j][0]).abs()>1e-7 || (ny-centroids[j][1]).abs()>1e-7 {moved=true;}
                centroids[j]=[nx,ny];
            } else {
                let idx=(lcg_next(&mut state) as usize)%n;
                centroids[j]=samples[idx];
                moved=true;
            }
        }
        if iter%10==0 || iter==iters-1 {
            let mut sse=0.0f64;
            for &s in samples { let mut bd=f32::INFINITY; for c in &centroids { let d=(s[0]-c[0]).powi(2)+(s[1]-c[1]).powi(2); if d<bd{bd=d;} } sse+=bd as f64; }
            let mse=sse/(samples.len() as f64 *2.0);
            eprintln!("iter {} mse {:.8e} empty {}", iter, mse, counts.iter().filter(|&&c| c==0).count());
        }
        if !moved { eprintln!("converged {}", iter); break; }
    }
    centroids.sort_by(|a,b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));
    centroids
}
fn main(){
    let dir=Path::new("/home/kaden/models/Qwen3.8-27B");
    let s1=gen_fwht_signs(42,256); let s2=gen_fwht_signs(1042,256);
    let targets:Vec<(&str,&str)>=vec![("early linear_attn out_proj (layer 0)","model.language_model.layers.0.linear_attn.out_proj.weight"),("mid mlp down_proj (layer 20)","model.language_model.layers.20.mlp.down_proj.weight"),("late mlp gate_proj (layer 40)","model.language_model.layers.40.mlp.gate_proj.weight")];
    let budget=4096usize;
    let mut all_pairs=Vec::new();
    for (_l,name) in &targets { let (f32d,shape)=load_tensor(dir,name); let m=shape[0]; let k=shape[1]; let p=collect_pair_samples(&f32d,m,k,&s1,&s2,budget); eprintln!("{} pairs {}", name, p.len()); all_pairs.extend(p); }
    eprintln!("total {} pairs", all_pairs.len());
    let seed=42u64;
    let iters=100;
    eprintln!("kmeans++ seed {} iters {}", seed, iters);
    let centroids=kmeans_2d(&all_pairs,128,iters,seed);
    let mut sse=0.0f64; let mut cnt=0usize; for &p in &all_pairs { let mut bd=f32::INFINITY; for c in &centroids { let d=(p[0]-c[0]).powi(2)+(p[1]-c[1]).powi(2); if d<bd{bd=d;}} sse+=bd as f64; cnt+=2; }
    let mse=sse/(cnt as f64);
    eprintln!("final VQ mse per dim {:.8e}", mse);
    const GL_CB4:[f32;16]=[-2.7326,-2.0690,-1.6180,-1.2562,-0.9423,-0.6568,-0.3880,-0.1284,0.1284,0.3880,0.6568,0.9423,1.2562,1.6180,2.0690,2.7326];
    let mut sse_s=0.0f64; let mut n_s=0usize; for &p in &all_pairs { for &v in &[p[0],p[1]] { let mut bd=(v-GL_CB4[0]).abs(); let mut best=GL_CB4[0]; for &c in GL_CB4.iter().skip(1){ let d=(v-c).abs(); if d<bd{bd=d;best=c;}} let e=v as f64-best as f64; sse_s+=e*e; n_s+=1; } }
    let mse_s=sse_s/(n_s as f64);
    eprintln!("scalar GL_CB4 mse {:.8e} ratio {:.4}", mse_s, mse/mse_s);
    println!("pub const GL_CB35: [[f32;2];128] = [");
    for c in &centroids { println!("    [{:.6}, {:.6}],", c[0], c[1]); }
    println!("];");
}
