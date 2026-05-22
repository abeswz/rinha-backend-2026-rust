use crate::fraud::data::Dataset;
use std::cell::UnsafeCell;

pub const FAST_NPROBE: usize = 5;
pub const FULL_NPROBE: usize = 24;
const K: usize = 4096; // must match build_index K; guarded by assert in warmup()

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

pub fn warmup() {
    let ds = crate::fraud::data::dataset();
    assert_eq!(
        ds.k, K,
        "index k={} != compiled K={K}; rebuild index or update K const",
        ds.k
    );
    let mut x = 0x12345678u32;
    for _ in 0..500 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let mut q = [0.0f32; 14];
        let mut s = x;
        for v in q.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = (s & 0xFFFF) as f32 / 65535.0;
        }
        let _ = knn5_ivf(&q, ds);
    }
}

#[inline(always)]
fn count_fraud(labels: [u8; 5]) -> usize {
    let packed = u64::from_le_bytes([
        labels[0], labels[1], labels[2], labels[3], labels[4], 0, 0, 0,
    ]);
    // labels are 0 or 1; bit 0 of each byte is the value.
    // mask selects bit 0 of bytes 0-4.
    (packed & 0x0000_0001_0101_0101).count_ones() as usize
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn centroid_dists_avx2(q: &[f32; 14], centroids: *const f32, dists: *mut f32) {
    use std::arch::x86_64::*;
    // zero accumulator — 512 stores (K=4096, step=8)
    for i in (0..K).step_by(8) {
        _mm256_storeu_ps(dists.add(i), _mm256_setzero_ps());
    }
    // accumulate squared diffs dimension by dimension
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

#[allow(dead_code)]
fn centroid_dists_scalar(q: &[f32; 14], centroids: *const f32, dists: *mut f32) {
    for i in 0..K {
        unsafe {
            *dists.add(i) = 0.0f32;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for d in 0..14usize {
        let qd = q[d];
        let base = d * K;
        for ci in 0..K {
            let diff = unsafe { *centroids.add(base + ci) } - qd;
            unsafe {
                *dists.add(ci) += diff * diff;
            }
        }
    }
}

fn top_n_centroids_fast(dists: &[f32; K], nprobe: usize) -> [u16; FULL_NPROBE] {
    let nprobe = nprobe.min(FULL_NPROBE);
    let mut top = [(u32::MAX, 0u16); FULL_NPROBE];
    let mut worst = u32::MAX;
    for (ci, &d) in dists.iter().enumerate() {
        let bits = d.to_bits();
        if bits < worst {
            let pos = top[..nprobe].partition_point(|&(b, _)| b <= bits);
            if pos < nprobe {
                top[pos..nprobe].rotate_right(1);
                top[pos] = (bits, ci as u16);
                worst = top[nprobe - 1].0;
            }
        }
    }
    top.map(|(_, idx)| idx)
}

fn probe(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { probe_avx2(q, ds, nprobe) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    probe_scalar(q, ds, nprobe)
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

#[cfg(not(target_arch = "x86_64"))]
fn probe_scalar(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    let dists_ptr = DISTS.with(|cell| cell.get() as *mut f32);
    centroid_dists_scalar(q, ds.centroids.as_ptr(), dists_ptr);
    let dists: &[f32; K] = unsafe { &*(dists_ptr as *const [f32; K]) };
    let top = top_n_centroids_fast(dists, nprobe);
    scan_blocks_scalar(q, ds, &top[..nprobe])
}

#[cfg(not(target_arch = "x86_64"))]
fn scan_blocks_scalar(q: &[f32; 14], ds: &Dataset, probed: &[u16]) -> [u8; 5] {
    const K_NEIGHBORS: usize = 5;
    let mut top: [(u32, u8); 5] = [(u32::MAX, 0u8); K_NEIGHBORS];
    let mut worst_bits = u32::MAX;

    for &ci in probed {
        let ci = ci as usize;
        let block_start = ds.offsets[ci] as usize;
        let block_end = ds.offsets[ci + 1] as usize;

        for block_i in block_start..block_end {
            let bb = block_i * 14 * 8;
            let lb = block_i * 8;

            let mut partial = 0.0f32;
            for d in 0..8usize {
                let raw = ds.blocks[bb + d * 8] as f32;
                let diff = q[d] - raw * 0.0001;
                partial += diff * diff;
            }
            if partial.to_bits() >= worst_bits && top[K_NEIGHBORS - 1].0 < u32::MAX {
                continue;
            }

            for slot in 0..8usize {
                let mut sq = 0.0f32;
                for d in 0..14usize {
                    let raw = ds.blocks[bb + d * 8 + slot] as f32;
                    let diff = q[d] - raw * 0.0001;
                    sq += diff * diff;
                }
                let bits = sq.to_bits();
                let label = ds.labels[lb + slot];
                if bits < worst_bits {
                    let insert_pos = top.partition_point(|&(d, _)| d <= bits);
                    if insert_pos < K_NEIGHBORS {
                        top[insert_pos..].rotate_right(1);
                        top[insert_pos] = (bits, label);
                        worst_bits = top[K_NEIGHBORS - 1].0;
                    }
                }
            }
        }
    }

    [top[0].1, top[1].1, top[2].1, top[3].1, top[4].1]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_blocks_avx2(q: &[f32; 14], ds: &Dataset, probed: &[u16]) -> [u8; 5] {
    use std::arch::x86_64::*;

    const K_NEIGHBORS: usize = 5;
    let scale = _mm256_set1_ps(0.0001);
    let mut q_vecs = [_mm256_setzero_ps(); 14];
    for d in 0..14usize {
        q_vecs[d] = _mm256_set1_ps(q[d]);
    }

    let mut top: [(u32, u8); 5] = [(u32::MAX, 0u8); K_NEIGHBORS];
    let mut worst_bits = u32::MAX;
    let bp = ds.blocks.as_ptr();
    let lp = ds.labels.as_ptr();

    for &ci in probed {
        let ci = ci as usize;
        let block_start = ds.offsets[ci] as usize;
        let block_end = ds.offsets[ci + 1] as usize;

        'block: for block_i in block_start..block_end {
            if block_i + 4 < block_end {
                _mm_prefetch(bp.add((block_i + 4) * 112) as *const i8, _MM_HINT_T0);
            }

            let bb = block_i * 112;
            let threshold = _mm256_set1_ps(f32::from_bits(worst_bits));

            macro_rules! load_dim {
                ($d:expr) => {{
                    let raw = _mm_loadu_si128(bp.add(bb + $d * 8) as *const _);
                    let i32s = _mm256_cvtepi16_epi32(raw);
                    _mm256_mul_ps(_mm256_cvtepi32_ps(i32s), scale)
                }};
            }
            macro_rules! fmadd_diff {
                ($acc:expr, $d:expr) => {{
                    let v = load_dim!($d);
                    let diff = _mm256_sub_ps(q_vecs[$d], v);
                    _mm256_fmadd_ps(diff, diff, $acc)
                }};
            }

            let mut acc = _mm256_setzero_ps();
            acc = fmadd_diff!(acc, 0);
            acc = fmadd_diff!(acc, 1);
            acc = fmadd_diff!(acc, 2);
            acc = fmadd_diff!(acc, 3);
            acc = fmadd_diff!(acc, 4);
            acc = fmadd_diff!(acc, 5);
            acc = fmadd_diff!(acc, 6);
            acc = fmadd_diff!(acc, 7);

            if top[K_NEIGHBORS - 1].0 < u32::MAX {
                let cmp = _mm256_cmp_ps::<_CMP_GE_OQ>(acc, threshold);
                if _mm256_movemask_ps(cmp) == 0xFF {
                    continue 'block;
                }
            }

            acc = fmadd_diff!(acc, 8);
            acc = fmadd_diff!(acc, 9);
            acc = fmadd_diff!(acc, 10);
            acc = fmadd_diff!(acc, 11);
            acc = fmadd_diff!(acc, 12);
            acc = fmadd_diff!(acc, 13);

            let mut dists = [0.0f32; 8];
            _mm256_storeu_ps(dists.as_mut_ptr(), acc);
            let labels_ptr = lp.add(block_i * 8);

            #[allow(clippy::needless_range_loop)]
            for slot in 0..8usize {
                let bits = dists[slot].to_bits();
                if bits < worst_bits {
                    let label = *labels_ptr.add(slot);
                    let insert_pos = top.partition_point(|&(d, _)| d <= bits);
                    if insert_pos < K_NEIGHBORS {
                        top[insert_pos..].rotate_right(1);
                        top[insert_pos] = (bits, label);
                        worst_bits = top[K_NEIGHBORS - 1].0;
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
    fn smoke_warmup_and_query() {
        data::init();
        warmup();
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
}
