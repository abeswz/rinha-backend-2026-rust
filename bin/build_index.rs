use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

const K: usize = 4096;
const D: usize = 14;
const INIT_SAMPLE: usize = 50_000;
const LLOYD_ITERS: usize = 25;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; D],
    label: String,
}

fn round4(x: f32) -> f32 { (x * 10000.0).round() * 0.0001 }

fn sq_dist(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..D { let d = a[i] - b[i]; s += d * d; }
    s
}

fn nearest_centroid(v: &[f32; D], centroids: &[[f32; D]]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_dist(v, c);
        if d < best_d { best_d = d; best = i; }
    }
    best
}

fn kmeans_plus_plus_init(vecs: &[[f32; D]], k: usize, sample_n: usize) -> Vec<[f32; D]> {
    let step = (vecs.len() / sample_n).max(1);
    let sample: Vec<&[f32; D]> = vecs.iter().step_by(step).take(sample_n).collect();
    let n = sample.len();

    let mut rng = 0xdeadbeef_u64;
    let lcg = |r: &mut u64| -> usize {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*r >> 33) as usize
    };

    let mut centers: Vec<[f32; D]> = Vec::with_capacity(k);
    centers.push(*sample[lcg(&mut rng) % n]);

    let mut dists = vec![f32::INFINITY; n];

    for _ in 1..k {
        let last = centers.last().unwrap();
        for (i, v) in sample.iter().enumerate() {
            let d = sq_dist(v, last);
            if d < dists[i] { dists[i] = d; }
        }
        let total: f32 = dists.iter().sum();
        let mut threshold = (lcg(&mut rng) as f32 / u32::MAX as f32) * total;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 { chosen = i; break; }
        }
        centers.push(*sample[chosen]);
    }

    centers
}

fn lloyd_assign(vecs: &[[f32; D]], centroids: &[[f32; D]]) -> Vec<usize> {
    let n = vecs.len();
    let nthreads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(16);
    let chunk = (n + nthreads - 1) / nthreads;
    let mut assignments = vec![0usize; n];

    std::thread::scope(|s| {
        let chunks: Vec<_> = assignments.chunks_mut(chunk).enumerate().collect();
        let mut handles = Vec::new();
        for (ci, chunk_slice) in chunks {
            let start = ci * chunk;
            let end = (start + chunk_slice.len()).min(n);
            let vslice = &vecs[start..end];
            handles.push(s.spawn(move || {
                for (j, v) in vslice.iter().enumerate() {
                    chunk_slice[j] = nearest_centroid(v, centroids);
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
    });

    assignments
}

fn lloyd_update(vecs: &[[f32; D]], assignments: &[usize], k: usize) -> Vec<[f32; D]> {
    let mut sums = vec![[0.0f32; D]; k];
    let mut counts = vec![0u64; k];
    for (v, &ci) in vecs.iter().zip(assignments.iter()) {
        for d in 0..D { sums[ci][d] += v[d]; }
        counts[ci] += 1;
    }
    for (ci, sum) in sums.iter_mut().enumerate() {
        let c = counts[ci].max(1) as f32;
        for d in 0..D { sum[d] /= c; }
    }
    sums
}

fn write_ivf1(
    centroids: &[[f32; D]],
    assignments: &[usize],
    vecs: &[[f32; D]],
    labels: &[u8],
    k: usize,
    out_path: &str,
) {
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &ci) in assignments.iter().enumerate() {
        groups[ci].push(i);
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(k + 1);
    let mut label_buf: Vec<u8> = Vec::new();
    let mut block_buf: Vec<i16> = Vec::new();

    let mut block_idx: u32 = 0;
    for g in &groups {
        offsets.push(block_idx);
        let nblocks = (g.len() + 7) / 8;
        for b in 0..nblocks {
            for slot in 0..8usize {
                let vec_idx = b * 8 + slot;
                let label = if vec_idx < g.len() { labels[g[vec_idx]] } else { 0 };
                label_buf.push(label);
            }
            for d in 0..D {
                for slot in 0..8usize {
                    let vec_idx = b * 8 + slot;
                    let val = if vec_idx < g.len() {
                        (vecs[g[vec_idx]][d] * 10000.0).round() as i16
                    } else {
                        0i16
                    };
                    block_buf.push(val);
                }
            }
        }
        block_idx += nblocks as u32;
    }
    offsets.push(block_idx);

    let mut centroid_buf: Vec<f32> = Vec::with_capacity(D * k);
    for d in 0..D {
        for c in centroids {
            centroid_buf.push(c[d]);
        }
    }

    let n = vecs.len() as u32;

    let out = File::create(out_path).expect("cannot create output");
    let gz = GzEncoder::new(BufWriter::new(out), Compression::best());
    let mut w = BufWriter::new(gz);

    w.write_all(b"IVF1").unwrap();
    w.write_all(&n.to_le_bytes()).unwrap();
    w.write_all(&(k as u32).to_le_bytes()).unwrap();
    w.write_all(&(D as u32).to_le_bytes()).unwrap();
    for &f in &centroid_buf { w.write_all(&f.to_le_bytes()).unwrap(); }
    for &o in &offsets { w.write_all(&o.to_le_bytes()).unwrap(); }
    for &l in &label_buf { w.write_all(&[l]).unwrap(); }
    for &b in &block_buf { w.write_all(&b.to_le_bytes()).unwrap(); }
    w.flush().unwrap();

    println!("IVF1 written to {out_path}: n={n}, k={k}, blocks={block_idx}");
}

fn main() {
    let in_path = "resources/references.json.gz";
    let out_path = "data/index.bin.gz";

    eprintln!("reading {in_path}...");
    let file = File::open(in_path).expect("cannot open references.json.gz");
    let gz = GzDecoder::new(BufReader::new(file));
    let refs: Vec<Reference> = serde_json::from_reader(gz).expect("failed to parse references JSON");

    let n = refs.len();
    eprintln!("loaded {n} reference vectors");

    let vecs: Vec<[f32; D]> = refs.iter().map(|r| {
        let mut v = r.vector;
        for x in v.iter_mut() { *x = round4(*x); }
        v
    }).collect();

    let labels: Vec<u8> = refs.iter().map(|r| if r.label == "fraud" { 1u8 } else { 0u8 }).collect();

    eprintln!("running kmeans++ init (K={K}, sample={INIT_SAMPLE})...");
    let mut centroids = kmeans_plus_plus_init(&vecs, K, INIT_SAMPLE);

    for iter in 0..LLOYD_ITERS {
        let assignments = lloyd_assign(&vecs, &centroids);
        centroids = lloyd_update(&vecs, &assignments, K);
        if iter % 5 == 0 || iter == LLOYD_ITERS - 1 {
            eprintln!("  iter {}/{LLOYD_ITERS}", iter + 1);
        }
        if iter == LLOYD_ITERS - 1 {
            write_ivf1(&centroids, &assignments, &vecs, &labels, K, out_path);
        }
    }
}
