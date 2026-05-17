use smallvec::SmallVec;
use std::cell::RefCell;
use std::path::Path;

pub struct IvfIndex {
    k: usize,
    #[allow(dead_code)]
    n: usize,
    nprobe_fast: usize,
    nprobe_slow: usize,
    centroids: Vec<f32>,
    offsets: Vec<u32>,
    labels: Vec<u8>,
    blocks: Vec<i16>,
}

struct CentroidBufs {
    dists: Vec<f32>,
}

thread_local! {
    static CENTROID_BUFS: RefCell<CentroidBufs> = RefCell::new(CentroidBufs {
        dists: Vec::with_capacity(4096),
    });
}

fn centroid_dists_scalar(query: &[f32; 14], centroids: &[f32], k: usize, dists: &mut Vec<f32>) {
    dists.clear();
    dists.resize(k, 0.0);
    for d in 0..14usize {
        let qd = query[d];
        let base = d * k;
        for ci in 0..k {
            let diff = centroids[base + ci] - qd;
            dists[ci] += diff * diff;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn centroid_dists_simd(query: &[f32; 14], centroids: &[f32], k: usize, dists: &mut Vec<f32>) {
    use std::arch::x86_64::*;

    dists.clear();
    dists.resize(k, 0.0);
    let dp = dists.as_mut_ptr();
    let cp = centroids.as_ptr();

    // Dim 0: initialize (mul, not fmadd)
    {
        let qd = _mm256_set1_ps(query[0]);
        let mut ci = 0usize;
        while ci + 16 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci)), qd);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci + 8)), qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_mul_ps(d1, d1));
            ci += 16;
        }
        while ci + 8 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci)), qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(ci) - query[0];
            *dp.add(ci) = diff * diff;
            ci += 1;
        }
    }

    // Dims 1..14: accumulate with fmadd
    for d in 1..14usize {
        let base = d * k;
        let qd = _mm256_set1_ps(query[d]);
        let mut ci = 0usize;
        while ci + 16 <= k {
            let cv0 = _mm256_loadu_ps(cp.add(base + ci));
            let cv1 = _mm256_loadu_ps(cp.add(base + ci + 8));
            let dv0 = _mm256_sub_ps(cv0, qd);
            let dv1 = _mm256_sub_ps(cv1, qd);
            let a0 = _mm256_loadu_ps(dp.add(ci));
            let a1 = _mm256_loadu_ps(dp.add(ci + 8));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(dv0, dv0, a0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_fmadd_ps(dv1, dv1, a1));
            ci += 16;
        }
        while ci + 8 <= k {
            let cv = _mm256_loadu_ps(cp.add(base + ci));
            let dv = _mm256_sub_ps(cv, qd);
            let a = _mm256_loadu_ps(dp.add(ci));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(dv, dv, a));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(base + ci) - query[d];
            *dp.add(ci) += diff * diff;
            ci += 1;
        }
    }
}

fn fill_centroid_dists(query: &[f32; 14], centroids: &[f32], k: usize, dists: &mut Vec<f32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { centroid_dists_simd(query, centroids, k, dists) };
        }
    }
    centroid_dists_scalar(query, centroids, k, dists);
}

fn block_scan_scalar(
    query: &[f32; 14],
    offsets: &[u32],
    labels: &[u8],
    blocks: &[i16],
    probed: &[usize],
    k: usize,
) -> SmallVec<[u8; 5]> {
    let mut top: SmallVec<[(u32, u8); 6]> = SmallVec::new();

    for &ci in probed {
        let block_start = offsets[ci] as usize;
        let block_end = offsets[ci + 1] as usize;

        for block_idx in block_start..block_end {
            let block_base = block_idx * 14 * 8;
            let label_base = block_idx * 8;

            for slot in 0..8 {
                let mut sq = 0.0f32;
                for d in 0..14usize {
                    let raw = blocks[block_base + d * 8 + slot] as f32;
                    let diff = query[d] - raw * 0.0001;
                    sq += diff * diff;
                }
                if sq.is_nan() {
                    continue;
                }
                let dist_bits = sq.to_bits();
                let label = labels[label_base + slot];
                if top.len() < k {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, label));
                } else if dist_bits < top[top.len() - 1].0 {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, label));
                    top.truncate(k);
                }
            }
        }
    }

    top.iter().map(|&(_, label)| label).collect()
}

impl IvfIndex {
    pub fn load(path: &Path, nprobe_fast: usize, nprobe_slow: usize) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let mut pos = 0;

        macro_rules! need {
            ($n:expr, $msg:literal) => {
                if data.len() < pos + $n {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, $msg));
                }
            };
        }
        macro_rules! read_u32 {
            () => {{
                need!(4, "truncated");
                let v = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
                v as usize
            }};
        }

        need!(4, "missing magic");
        if &data[..4] != b"IVF2" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "expected IVF2 magic",
            ));
        }
        pos = 4;

        let n = read_u32!();
        let k = read_u32!();
        let d = read_u32!();
        if d != 14 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected d=14, got {d}"),
            ));
        }

        need!(d * k * 4, "truncated: centroids");
        let mut centroids = Vec::with_capacity(d * k);
        for _ in 0..d * k {
            centroids.push(f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }

        need!((k + 1) * 4, "truncated: offsets");
        let mut offsets = Vec::with_capacity(k + 1);
        for _ in 0..=k {
            offsets.push(u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }

        let total_blocks = offsets[k] as usize;

        need!(total_blocks * 8, "truncated: labels");
        let labels = data[pos..pos + total_blocks * 8].to_vec();
        pos += total_blocks * 8;

        let block_i16_count = total_blocks * d * 8;
        need!(block_i16_count * 2, "truncated: blocks");
        let mut blocks = Vec::with_capacity(block_i16_count);
        for _ in 0..block_i16_count {
            blocks.push(i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()));
            pos += 2;
        }

        Ok(Self { k, n, nprobe_fast, nprobe_slow, centroids, offsets, labels, blocks })
    }

    pub fn knn(&self, query: &[f32; 14], k: usize, nprobe: usize) -> SmallVec<[u8; 5]> {
        let nprobe = nprobe.min(self.k);

        CENTROID_BUFS.with(|bufs| {
            let mut bufs = bufs.borrow_mut();
            fill_centroid_dists(query, &self.centroids, self.k, &mut bufs.dists);

            // Use a local vec for indices (avoids re-borrow conflict)
            let mut indices: Vec<usize> = (0..self.k).collect();

            if nprobe < self.k {
                let dists = &bufs.dists;
                indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
                    dists[a].partial_cmp(&dists[b]).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            block_scan_scalar(
                query,
                &self.offsets,
                &self.labels,
                &self.blocks,
                &indices[..nprobe],
                k,
            )
        })
    }

    pub fn knn_adaptive(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        let stage1 = self.knn(query, k, self.nprobe_fast);
        let fraud_votes = stage1.iter().filter(|&&l| l == 1).count();
        if fraud_votes <= 1 || fraud_votes >= k.saturating_sub(1) {
            return stage1;
        }
        self.knn(query, k, self.nprobe_slow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ivf2_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"IVF2");
        buf.extend_from_slice(&16u32.to_le_bytes()); // n
        buf.extend_from_slice(&2u32.to_le_bytes());  // k
        buf.extend_from_slice(&14u32.to_le_bytes()); // d

        // centroids column-major: [C0_d0, C1_d0, C0_d1, C1_d1, ...]
        for _ in 0..14 {
            buf.extend_from_slice(&0.0f32.to_le_bytes()); // C0
            buf.extend_from_slice(&2.0f32.to_le_bytes()); // C1
        }

        // block_offsets: [0, 1, 2]
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());

        // labels: 16 bytes (2 blocks × 8 slots)
        for _ in 0..8 { buf.push(0u8); } // block 0: legit
        for _ in 0..8 { buf.push(1u8); } // block 1: fraud

        // blocks: 2 × 14 × 8 i16
        let legit_val: i16 = 1000;  // round(0.1 * 10000)
        let fraud_val: i16 = 20000; // round(2.0 * 10000)
        for _ in 0..112 { buf.extend_from_slice(&legit_val.to_le_bytes()); } // block 0
        for _ in 0..112 { buf.extend_from_slice(&fraud_val.to_le_bytes()); } // block 1

        buf
    }

    fn write_ivf2(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, make_ivf2_bytes()).unwrap();
        path
    }

    fn make_staged_ivf2_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"IVF2");
        buf.extend_from_slice(&48u32.to_le_bytes()); // n
        buf.extend_from_slice(&6u32.to_le_bytes());  // k
        buf.extend_from_slice(&14u32.to_le_bytes()); // d

        let centroid_vals = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        for _ in 0..14 {
            for &v in &centroid_vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        for i in 0u32..=6 {
            buf.extend_from_slice(&i.to_le_bytes());
        }

        // C0: all legit
        for _ in 0..8 { buf.push(0u8); }
        // C1: all legit
        for _ in 0..8 { buf.push(0u8); }
        // C2: slot 0 = fraud, slots 1-7 = legit
        buf.push(1u8);
        for _ in 0..7 { buf.push(0u8); }
        // C3: slot 0 = fraud, slots 1-7 = legit
        buf.push(1u8);
        for _ in 0..7 { buf.push(0u8); }
        // C4: all legit
        for _ in 0..8 { buf.push(0u8); }
        // C5: all fraud
        for _ in 0..8 { buf.push(1u8); }

        // blocks: 6 × 112 i16
        // C0 block: all dim=0.1 → i16=1000
        for _ in 0..112 { buf.extend_from_slice(&1000i16.to_le_bytes()); }
        // C1 block: all dim=0.2 → i16=2000
        for _ in 0..112 { buf.extend_from_slice(&2000i16.to_le_bytes()); }
        // C2 block: slot 0 = 2400 (0.24), slots 1-7 = 3000 (0.3)
        for _d in 0..14usize {
            buf.extend_from_slice(&2400i16.to_le_bytes()); // slot 0: fraud
            for _ in 0..7 { buf.extend_from_slice(&3000i16.to_le_bytes()); }
        }
        // C3 block: slot 0 = 2600 (0.26), slots 1-7 = 4000 (0.4)
        for _d in 0..14usize {
            buf.extend_from_slice(&2600i16.to_le_bytes()); // slot 0: fraud
            for _ in 0..7 { buf.extend_from_slice(&4000i16.to_le_bytes()); }
        }
        // C4 block: all dim=0.5 → i16=5000
        for _ in 0..112 { buf.extend_from_slice(&5000i16.to_le_bytes()); }
        // C5 block: all dim=0.25 → i16=2500
        for _ in 0..112 { buf.extend_from_slice(&2500i16.to_le_bytes()); }

        buf
    }

    fn write_staged_ivf2(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, make_staged_ivf2_bytes()).unwrap();
        path
    }

    #[test]
    fn test_ivf2_load_parses_header() {
        let path = write_ivf2("ivf2_header.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        assert_eq!(idx.k, 2);
        assert_eq!(idx.n, 16);
        assert_eq!(idx.offsets.len(), 3);
        assert_eq!(idx.labels.len(), 16);
        assert_eq!(idx.blocks.len(), 224);
        assert_eq!(idx.centroids.len(), 28);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ivf2_load_rejects_ivf1_magic() {
        let path = std::env::temp_dir().join("ivf2_bad_magic.bin");
        let mut bad = make_ivf2_bytes();
        bad[..4].copy_from_slice(b"IVF1");
        std::fs::write(&path, &bad).unwrap();
        assert!(IvfIndex::load(&path, 5, 24).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ivf2_load_rejects_wrong_dimensions() {
        let path = std::env::temp_dir().join("ivf2_bad_d.bin");
        let mut bad = make_ivf2_bytes();
        bad[12..16].copy_from_slice(&13u32.to_le_bytes());
        std::fs::write(&path, &bad).unwrap();
        assert!(IvfIndex::load(&path, 5, 24).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ivf2_load_rejects_truncated_file() {
        let path = std::env::temp_dir().join("ivf2_truncated.bin");
        std::fs::write(&path, &[0u8; 10]).unwrap();
        assert!(IvfIndex::load(&path, 5, 24).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_centroid_scan_column_major_matches_brute_force() {
        let path = write_ivf2("ivf2_centroid_scan.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        let query = [0.0f32; 14];
        let expected_c0_dist: f32 = 0.0;
        let expected_c1_dist: f32 = 14.0 * 2.0f32.powi(2);
        let labels = idx.knn(&query, 5, 1);
        assert_eq!(labels.len(), 5);
        assert!(
            labels.iter().all(|&l| l == 0),
            "nprobe=1 near C0 → all legit; expected_c0={expected_c0_dist}, expected_c1={expected_c1_dist}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_block_scan_8vec_matches_brute_force() {
        let path = write_ivf2("ivf2_block_scan.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 5, 1);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().all(|&l| l == 0));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_query_near_fraud_cluster() {
        let path = write_ivf2("ivf2_fraud_cluster.bin");
        let idx = IvfIndex::load(&path, 5, 1).unwrap();
        let query = [2.0f32; 14];
        let labels = idx.knn(&query, 5, 1);
        assert_eq!(labels.len(), 5);
        assert!(
            labels.iter().all(|&l| l == 1),
            "query near fraud centroid [2.0;14] → all fraud neighbors"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_nprobe_clamped_to_k() {
        let path = write_ivf2("ivf2_clamp.bin");
        let idx = IvfIndex::load(&path, 5, 999).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 5, 999);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().filter(|&&l| l == 0).count() >= 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_adaptive_unambiguous_legit_uses_stage1() {
        let path = write_ivf2("ivf2_adapt_legit.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn_adaptive(&query, 5);
        assert_eq!(labels.len(), 5);
        assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_adaptive_unambiguous_fraud_uses_stage1() {
        let path = write_ivf2("ivf2_adapt_fraud.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        let query = [2.0f32; 14];
        let labels = idx.knn_adaptive(&query, 5);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().filter(|&&l| l == 1).count() >= 4);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_simd_centroid_dists_matches_scalar() {
        let path = write_ivf2("ivf2_simd_centroid.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();

        let query = [0.3f32; 14];

        // Fixture: C0=[0.0;14] (dist=14*0.09=1.26), C1=[2.0;14] (dist=14*2.89=40.46)
        // Top-1 centroid = C0, so top-5 neighbors are all legit.
        let scalar_labels = idx.knn(&query, 5, 2);
        assert_eq!(scalar_labels.len(), 5);
        assert!(
            scalar_labels.iter().all(|&l| l == 0),
            "SIMD centroid scan must route to legit cluster"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_adaptive_ambiguous_triggers_stage2() {
        let path = write_staged_ivf2("ivf2_adapt_staged.bin");
        let idx = IvfIndex::load(&path, 5, 6).unwrap();
        let query = [0.25f32; 14];

        let stage1 = idx.knn(&query, 5, 5);
        let stage1_fraud = stage1.iter().filter(|&&l| l == 1).count();
        assert_eq!(stage1_fraud, 2, "stage1 must be ambiguous (2 fraud), got {stage1_fraud}");

        let labels = idx.knn_adaptive(&query, 5);
        let fraud_count = labels.iter().filter(|&&l| l == 1).count();
        assert!(
            fraud_count >= 4,
            "stage2 must find C5 straggler fraud vectors, got {fraud_count} fraud"
        );
        std::fs::remove_file(&path).ok();
    }
}
