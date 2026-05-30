use crate::fraud::data::Dataset;
use std::cell::UnsafeCell;

pub const FAST_NPROBE: usize = 8;
pub const FULL_NPROBE: usize = 24;
pub const K: usize = 4096; // must match build_index K; guarded by assert in data::init()

thread_local! {
    static DISTS: UnsafeCell<[f32; K]> = const { UnsafeCell::new([0.0f32; K]) };
}

pub fn knn5_ivf(q: &[f32; 14], ds: &Dataset) -> u8 {
    let fast = probe(q, ds, FAST_NPROBE);
    let fraud_count = count_fraud(fast);
    if fraud_count == 2 || fraud_count == 3 {
        let full = probe(q, ds, FULL_NPROBE);
        count_fraud(full) as u8
    } else {
        fraud_count as u8
    }
}


#[inline(always)]
fn count_fraud(labels: [u8; 5]) -> usize {
    let packed = u64::from_le_bytes([
        labels[0], labels[1], labels[2], labels[3], labels[4], 0, 0, 0,
    ]);
    (packed & 0x0000_0001_0101_0101).count_ones() as usize
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn centroid_dists_avx2(q: &[f32; 14], centroids: *const f32, dists: *mut f32) {
    use std::arch::x86_64::*;
    for i in (0..K).step_by(8) {
        _mm256_storeu_ps(dists.add(i), _mm256_setzero_ps());
    }
    #[allow(clippy::needless_range_loop)]
    for d in 0..14usize {
        let qd = _mm256_set1_ps(q[d]);
        let base = d * K;
        for ci in (0..K).step_by(8) {
            let v = _mm256_loadu_ps(centroids.add(base + ci));
            let diff = _mm256_sub_ps(qd, v);
            let acc = _mm256_loadu_ps(dists.add(ci));
            let acc = _mm256_fmadd_ps(diff, diff, acc);
            _mm256_storeu_ps(dists.add(ci), acc);
        }
    }
}


fn top_n_centroids_fast(dists: &[f32; K], nprobe: usize) -> [u16; FULL_NPROBE] {
    let nprobe = nprobe.min(FULL_NPROBE);
    // Stack-allocated: 4096 × 6B = 24KB, well within 512KB thread stack
    let mut indexed = [(0u32, 0u16); K];
    for (ci, &d) in dists.iter().enumerate() {
        indexed[ci] = (d.to_bits(), ci as u16);
    }
    // O(K) expected partial sort via PDQ; top nprobe entries unsorted among themselves
    indexed.select_nth_unstable_by_key(nprobe - 1, |&(b, _)| b);
    indexed[..nprobe].sort_unstable_by_key(|&(b, _)| b);
    let mut result = [0u16; FULL_NPROBE];
    for i in 0..nprobe {
        result[i] = indexed[i].1;
    }
    result
}

fn probe(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        probe_avx2(q, ds, nprobe)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (q, ds, nprobe);
        unimplemented!("requires x86_64")
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn probe_avx2(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    let dists_ptr = DISTS.with(|cell| cell.get() as *mut f32);
    centroid_dists_avx2(q, ds.centroids.as_ptr(), dists_ptr);
    let dists: &[f32; K] = &*(dists_ptr as *const [f32; K]);
    let top = top_n_centroids_fast(dists, nprobe);
    scan_blocks_avx2(q, ds, &top[..nprobe])
}


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_blocks_avx2(q: &[f32; 14], ds: &Dataset, probed: &[u16]) -> [u8; 5] {
    use std::arch::x86_64::*;

    // Pre-quantize query to i16 (values ∈ [-100,100]; diff ∈ [-200,200] needs i16 to avoid overflow)
    let mut q_i16 = [0i16; 14];
    for d in 0..14 {
        q_i16[d] = (q[d] * 100.0).round() as i16;
    }

    const K_NEIGHBORS: usize = 5;
    let mut top: [(u32, u8); K_NEIGHBORS] = [(u32::MAX, 0u8); K_NEIGHBORS];
    let mut worst_u32: u32 = u32::MAX;
    let bp = ds.blocks.as_ptr(); // *const i8
    let lp = ds.labels.as_ptr();

    for &ci in probed {
        let ci = ci as usize;
        let block_start = ds.offsets[ci] as usize;
        let block_end = ds.offsets[ci + 1] as usize;

        'block: for block_i in block_start..block_end {
            if block_i + 4 < block_end {
                _mm_prefetch(bp.add((block_i + 4) * 112) as *const i8, _MM_HINT_T0);
            }

            // Each block: 8 slots × 14 dims = 112 bytes
            let bb = block_i * 112;
            let mut acc = _mm256_setzero_si256(); // 8 × i32 squared-distance accumulators

            macro_rules! acc_dim {
                ($d:expr) => {{
                    let raw = _mm_loadl_epi64(bp.add(bb + $d * 8) as *const __m128i);
                    let v16 = _mm_cvtepi8_epi16(raw);            // 8×i8 → 8×i16
                    let q16 = _mm_set1_epi16(q_i16[$d]);         // broadcast query value
                    let diff = _mm_sub_epi16(q16, v16);          // 8×i16 diffs ∈ [-200,200]
                    let diff32 = _mm256_cvtepi16_epi32(diff);    // 8×i32
                    let sq = _mm256_mullo_epi32(diff32, diff32); // 8×i32 squares ≤ 40000
                    acc = _mm256_add_epi32(acc, sq);             // max sum 14×40000=560000 < i32::MAX
                }};
            }

            acc_dim!(0);
            acc_dim!(1);
            acc_dim!(2);
            acc_dim!(3);
            acc_dim!(4);
            acc_dim!(5);
            acc_dim!(6);
            acc_dim!(7);

            // Early exit: if no vector can beat current worst after 8 dims, skip remaining 6
            if worst_u32 < u32::MAX {
                // worst_u32 ≤ 560000 < i32::MAX so cast is safe; signed cmp works for non-negative values
                let threshold = _mm256_set1_epi32(worst_u32 as i32);
                let cmp_can_win = _mm256_cmpgt_epi32(threshold, acc); // threshold > acc → might win
                if _mm256_movemask_epi8(cmp_can_win) == 0 {
                    continue 'block;
                }
            }

            acc_dim!(8);
            acc_dim!(9);
            acc_dim!(10);
            acc_dim!(11);
            acc_dim!(12);
            acc_dim!(13);

            let mut dists_i32 = [0i32; 8];
            _mm256_storeu_si256(dists_i32.as_mut_ptr() as *mut __m256i, acc);
            let labels_ptr = lp.add(block_i * 8);

            #[allow(clippy::needless_range_loop)]
            for slot in 0..8usize {
                let d_u32 = dists_i32[slot] as u32; // safe: squared distances are non-negative
                if d_u32 < worst_u32 {
                    let label = *labels_ptr.add(slot);
                    let insert_pos = top.partition_point(|&(d, _)| d <= d_u32);
                    if insert_pos < K_NEIGHBORS {
                        top[insert_pos..].rotate_right(1);
                        top[insert_pos] = (d_u32, label);
                        worst_u32 = top[K_NEIGHBORS - 1].0;
                    }
                }
            }
        }
    }

    [top[0].1, top[1].1, top[2].1, top[3].1, top[4].1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fraud::data;

    #[test]
    fn count_fraud_correct() {
        assert_eq!(count_fraud([1, 0, 1, 0, 1]), 3);
        assert_eq!(count_fraud([0, 0, 0, 0, 0]), 0);
        assert_eq!(count_fraud([1, 1, 1, 1, 1]), 5);
        assert_eq!(count_fraud([1, 0, 0, 0, 0]), 1);
    }

    #[test]
    fn top_n_centroids_fast_smallest_first() {
        let mut dists = [100.0f32; K];
        dists[10] = 0.5;
        dists[7] = 1.0;
        dists[42] = 2.0;
        let top = top_n_centroids_fast(&dists, 3);
        assert_eq!(top[0], 10u16); // dist=0.5, smallest
        assert_eq!(top[1], 7u16); // dist=1.0
        assert_eq!(top[2], 42u16); // dist=2.0
    }

    #[test]
    fn smoke_zero_query() {
        data::init();
        let q = [0.0f32; 14];
        let ds = data::dataset();
        let result = knn5_ivf(&q, ds);
        assert!(result <= 5, "knn5_ivf must return 0..=5, got {result}");
    }

    #[test]
    fn smoke_fraud_heavy_query() {
        data::init();
        let q = [1.0f32; 14];
        let ds = data::dataset();
        let result = knn5_ivf(&q, ds);
        assert!(result <= 5, "knn5_ivf must return 0..=5, got {result}");
    }

    #[test]
    fn model_gen_sanity() {
        let fraud_q = [1.0f64; 14];
        let p_fraud = crate::fraud::model_gen::predict_fraud(&fraud_q);
        assert!(p_fraud > 0.5, "all-ones features got P(fraud)={p_fraud:.4}, expected > 0.5");

        let legit_q = [0.0f64; 14];
        let p_legit = crate::fraud::model_gen::predict_fraud(&legit_q);
        assert!(p_legit < 0.5, "all-zeros features got P(fraud)={p_legit:.4}, expected < 0.5");
    }

    #[test]
    fn lgbm_decides_ambiguous_cases() {
        data::init();
        let ds = data::dataset();
        let test_queries: &[[f32; 14]] = &[
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 0.5, 0.5],
            [0.3, 0.1, 0.3, 0.6, 0.4, 0.2, 0.1, 0.3, 0.4, 0.0, 1.0, 0.0, 0.3, 0.1],
            [0.8, 0.0, 0.8, 0.3, 0.7, 0.5, 0.5, 0.8, 0.3, 1.0, 1.0, 1.0, 0.8, 0.0],
        ];
        for q in test_queries {
            let result = knn5_ivf(q, ds);
            assert!(result <= 5, "knn5_ivf returned {result} (must be 0..=5)");
        }
    }

}
