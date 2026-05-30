use crate::fraud::data::Dataset;
use std::cell::UnsafeCell;

pub const K: usize = 4096;

thread_local! {
    static DISTS: UnsafeCell<[f32; K]> = const { UnsafeCell::new([0.0f32; K]) };
}

pub fn knn5_exact(q: &[f32; 14], ds: &Dataset) -> u8 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        knn5_exact_impl(q, ds)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (q, ds);
        unimplemented!("requires x86_64")
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

fn sorted_centroids(dists: &[f32; K]) -> [u16; K] {
    // Stack: [(u32, u16); 4096] = 24KB + [u16; 4096] = 8KB, well within 512KB thread stack
    let mut indexed = [(0u32, 0u16); K];
    for (ci, &d) in dists.iter().enumerate() {
        indexed[ci] = (d.to_bits(), ci as u16);
    }
    indexed.sort_unstable_by_key(|&(b, _)| b);
    let mut result = [0u16; K];
    for i in 0..K {
        result[i] = indexed[i].1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_exact_impl(q: &[f32; 14], ds: &Dataset) -> u8 {
    let dists_ptr = DISTS.with(|cell| cell.get() as *mut f32);
    centroid_dists_avx2(q, ds.centroids.as_ptr(), dists_ptr);
    let dists: &[f32; K] = &*(dists_ptr as *const [f32; K]);
    let sorted = sorted_centroids(dists);
    let labels = scan_exact_avx2(q, ds, dists, &sorted);
    count_fraud(labels) as u8
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_exact_avx2(
    q: &[f32; 14],
    ds: &Dataset,
    centroid_dists_sq: &[f32; K],
    sorted: &[u16; K],
) -> [u8; 5] {
    use std::arch::x86_64::*;

    // Pre-quantize query to i16
    // Scale=3000, features in [-1,1] → i16 in [-3000,3000]. Max diff=6000 (fits i16).
    // Max sq diff = 36_000_000 < i32::MAX. Max sum over 14 dims = 504_000_000 < i32::MAX.
    let mut q_i16 = [0i16; 14];
    for d in 0..14 {
        q_i16[d] = (q[d] * 3000.0).round() as i16;
    }

    const K_NEIGHBORS: usize = 5;
    let mut top: [(u32, u8); K_NEIGHBORS] = [(u32::MAX, 0u8); K_NEIGHBORS];
    let mut worst_i16_sq: u32 = u32::MAX;
    let bp = ds.blocks.as_ptr() as *const i8;
    let lp = ds.labels.as_ptr();

    'cluster: for &ci_u16 in sorted.iter() {
        let ci = ci_u16 as usize;

        // Bounding-ball lower bound: min possible dist from q to any vector in cluster ci.
        // worst_f32_sq: convert worst i16-sq distance back to f32-sq space for comparison.
        // Features in [-1,1]: max lb² ≈ 56 << u32::MAX/9e6 ≈ 477, so no false prune when heap not full.
        let worst_f32_sq = worst_i16_sq as f32 / 9_000_000.0_f32;
        let lb = (centroid_dists_sq[ci].sqrt() - ds.radii[ci]).max(0.0);
        if lb * lb > worst_f32_sq {
            continue 'cluster;
        }

        let block_start = ds.offsets[ci] as usize;
        let block_end = ds.offsets[ci + 1] as usize;

        'block: for block_i in block_start..block_end {
            if block_i + 4 < block_end {
                _mm_prefetch(bp.add((block_i + 4) * 224), _MM_HINT_T0);
            }

            // Each block: 8 slots × 14 dims × 2 bytes (i16) = 224 bytes
            let bb = block_i * 224;
            let mut acc = _mm256_setzero_si256();

            macro_rules! acc_dim {
                ($d:expr) => {{
                    let v16 = _mm_loadu_si128(bp.add(bb + $d * 16) as *const __m128i);
                    let q16 = _mm_set1_epi16(q_i16[$d]);
                    let diff = _mm_sub_epi16(q16, v16);
                    let diff32 = _mm256_cvtepi16_epi32(diff);
                    let sq = _mm256_mullo_epi32(diff32, diff32);
                    acc = _mm256_add_epi32(acc, sq);
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

            // Early exit: if no slot in this block can beat current worst after 8 dims, skip
            if worst_i16_sq < u32::MAX {
                let threshold = _mm256_set1_epi32(worst_i16_sq as i32);
                let cmp_can_win = _mm256_cmpgt_epi32(threshold, acc);
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
                let d_u32 = dists_i32[slot] as u32;
                if d_u32 < worst_i16_sq {
                    let label = *labels_ptr.add(slot);
                    let insert_pos = top.partition_point(|&(d, _)| d <= d_u32);
                    if insert_pos < K_NEIGHBORS {
                        top[insert_pos..].rotate_right(1);
                        top[insert_pos] = (d_u32, label);
                        worst_i16_sq = top[K_NEIGHBORS - 1].0;
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
    fn sorted_centroids_smallest_first() {
        let mut dists = [100.0f32; K];
        dists[10] = 0.5;
        dists[7] = 1.0;
        dists[42] = 2.0;
        let sorted = sorted_centroids(&dists);
        assert_eq!(sorted[0], 10u16);
        assert_eq!(sorted[1], 7u16);
        assert_eq!(sorted[2], 42u16);
    }

    #[test]
    fn exact_smoke_zero_query() {
        data::init();
        let q = [0.0f32; 14];
        let ds = data::dataset();
        let result = knn5_exact(&q, ds);
        assert!(result <= 5, "knn5_exact must return 0..=5, got {result}");
    }

    #[test]
    fn exact_smoke_fraud_heavy() {
        data::init();
        let q = [1.0f32; 14];
        let ds = data::dataset();
        let result = knn5_exact(&q, ds);
        assert!(result <= 5, "knn5_exact must return 0..=5, got {result}");
    }

    #[test]
    fn exact_deterministic() {
        data::init();
        let q = [
            0.3, -0.7, 0.2, 0.5, -0.1, 1.0, -0.5, 0.2, 0.8, -0.1, 0.4, 0.6, -0.3, 0.9f32,
        ];
        let ds = data::dataset();
        let r1 = knn5_exact(&q, ds);
        let r2 = knn5_exact(&q, ds);
        assert_eq!(r1, r2, "knn5_exact must be deterministic");
    }
}
