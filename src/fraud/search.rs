use crate::fraud::data::{
    IvfIndex, CID_BITS, CID_MASK, IDX_BITS, K_NEIGHBORS, N_PAIRS, N_PROBE_INITIAL,
    N_PROBE_REPAIR_MAX, N_PROBE_REPAIR_MIN,
};

// ── Scalar fallbacks (used in tests and non-x86 builds) ──────────────────────

fn dist_l2_i16q_scalar(flat_vec: &[i16], base: usize, q: &[i16; 16]) -> i32 {
    let mut d = 0i32;
    for i in 0..16 {
        let diff = flat_vec[base + i] as i32 - q[i] as i32;
        d = d.wrapping_add(diff * diff);
    }
    d
}

fn compute_cluster_batch8_scalar(
    min_soa: &[i16],
    max_soa: &[i16],
    q: &[i16; 16],
    lbs: &mut [i32; 8],
) {
    lbs.fill(0);
    for p in 0..N_PAIRS {
        for l in 0..8usize {
            let di = p * 16 + l * 2;
            for d in 0..2usize {
                let qd = q[2 * p + d] as i32;
                let lo = min_soa[di + d] as i32;
                let hi = max_soa[di + d] as i32;
                let gap = if qd < lo { lo - qd } else if qd > hi { qd - hi } else { 0 };
                lbs[l] = lbs[l].wrapping_add(gap * gap);
            }
        }
    }
}

// ── AVX2 implementations ──────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_l2_i16q_avx2(flat_vec: &[i16], base: usize, q: &[i16; 16]) -> i32 {
    use std::arch::x86_64::*;
    let v = _mm256_loadu_si256(flat_vec.as_ptr().add(base) as *const __m256i);
    let qv = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
    let diff = _mm256_sub_epi16(v, qv);
    let sq = _mm256_madd_epi16(diff, diff);
    let hi = _mm256_extracti128_si256(sq, 1);
    let lo = _mm256_castsi256_si128(sq);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_hadd_epi32(s, s);
    let s = _mm_hadd_epi32(s, s);
    _mm_cvtsi128_si32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_cluster_batch8_avx2(
    min_soa: &[i16],
    max_soa: &[i16],
    q: &[i16; 16],
    lbs: &mut [i32; 8],
) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let mut acc = zero;

    for p in 0..N_PAIRS {
        let off = p * 16;
        let min_vec = _mm256_loadu_si256(min_soa.as_ptr().add(off) as *const __m256i);
        let max_vec = _mm256_loadu_si256(max_soa.as_ptr().add(off) as *const __m256i);

        let q0 = q[2 * p] as i16;
        let q1 = q[2 * p + 1] as i16;
        // Broadcast pair: [q0, q1, q0, q1, ...] × 8  (set_epi16 fills e15→e0)
        let q_pair = _mm256_set_epi16(
            q1, q0, q1, q0, q1, q0, q1, q0,
            q1, q0, q1, q0, q1, q0, q1, q0,
        );

        let lb_min = _mm256_max_epi16(_mm256_sub_epi16(min_vec, q_pair), zero);
        let lb_max = _mm256_max_epi16(_mm256_sub_epi16(q_pair, max_vec), zero);
        let lb = _mm256_add_epi16(lb_min, lb_max);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lb, lb));
    }

    _mm256_storeu_si256(lbs.as_mut_ptr() as *mut __m256i, acc);
}

// ── Dispatch wrappers ─────────────────────────────────────────────────────────

#[inline(always)]
fn dist_l2_i16q(flat_vec: &[i16], base: usize, q: &[i16; 16]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        dist_l2_i16q_avx2(flat_vec, base, q)
    }
    #[cfg(not(target_arch = "x86_64"))]
    dist_l2_i16q_scalar(flat_vec, base, q)
}

#[inline(always)]
fn compute_cluster_batch8(
    min_soa: &[i16],
    max_soa: &[i16],
    q: &[i16; 16],
    lbs: &mut [i32; 8],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        compute_cluster_batch8_avx2(min_soa, max_soa, q, lbs)
    }
    #[cfg(not(target_arch = "x86_64"))]
    compute_cluster_batch8_scalar(min_soa, max_soa, q, lbs)
}

// ── IvfIndex search ───────────────────────────────────────────────────────────

impl IvfIndex {
    pub fn search(&self, q: &[i16; 16]) -> u8 {
        let mut topk_keys = [i64::MAX; K_NEIGHBORS];
        let mut topk_labels = [0u8; K_NEIGHBORS];

        self.search_core(q, N_PROBE_INITIAL, &mut topk_keys, &mut topk_labels);

        let count: usize = topk_labels.iter().map(|&l| l as usize).sum();
        if count >= N_PROBE_REPAIR_MIN && count <= N_PROBE_REPAIR_MAX {
            // Uncertain result — sweep all clusters
            topk_keys = [i64::MAX; K_NEIGHBORS];
            topk_labels = [0u8; K_NEIGHBORS];
            self.search_core(q, self.n_clusters, &mut topk_keys, &mut topk_labels);
        }

        topk_labels.iter().sum()
    }

    fn search_core(
        &self,
        q: &[i16; 16],
        max_probes: usize,
        topk_keys: &mut [i64; K_NEIGHBORS],
        topk_labels: &mut [u8; K_NEIGHBORS],
    ) {
        // Compute packed bounding-box lower bounds for all clusters
        let mut packed = vec![i64::MAX; self.n_clusters];
        let n_groups = (self.n_clusters + 7) / 8;
        let mut lbs = [0i32; 8];

        for g in 0..n_groups {
            let off = g * N_PAIRS * 16;
            compute_cluster_batch8(
                &self.bpsoa_min[off..off + N_PAIRS * 16],
                &self.bpsoa_max[off..off + N_PAIRS * 16],
                q,
                &mut lbs,
            );
            let base = g * 8;
            for l in 0..8 {
                let c = base + l;
                if c >= self.n_clusters {
                    break;
                }
                // Negative means i32 overflow → very far cluster
                let lb = if lbs[l] < 0 { i64::MAX >> CID_BITS } else { lbs[l] as i64 };
                packed[c] = (lb << CID_BITS) | c as i64;
            }
        }

        let mut worst_key = i64::MAX;
        let mut probe = 0usize;

        loop {
            // Find best (min) remaining cluster
            let best = packed.iter().copied().min().unwrap_or(i64::MAX);
            if best == i64::MAX {
                break;
            }
            // Pruning: cluster lb (scaled) >= worst neighbor dist
            let best_lb = best >> CID_BITS;
            if (best_lb << IDX_BITS) >= worst_key {
                break;
            }

            let cid = (best & CID_MASK) as usize;
            packed[cid] = i64::MAX;

            self.scan_cluster(cid, q, topk_keys, topk_labels, &mut worst_key);

            probe += 1;
            if probe >= max_probes {
                break;
            }
        }
    }

    fn scan_cluster(
        &self,
        cid: usize,
        q: &[i16; 16],
        topk_keys: &mut [i64; K_NEIGHBORS],
        topk_labels: &mut [u8; K_NEIGHBORS],
        worst_key: &mut i64,
    ) {
        let start = self.cluster_offsets[cid] as usize;
        let end = self.cluster_offsets[cid + 1] as usize;
        let wk = *worst_key;

        for vi in start..end {
            let base = vi * 16;
            let dist = dist_l2_i16q(self.flat_vec, base, q);
            let key = (dist as u32 as i64) << IDX_BITS | vi as i64;
            if key >= wk {
                continue;
            }
            // Replace the worst neighbor in topk
            let (wi, _) = topk_keys
                .iter()
                .enumerate()
                .max_by_key(|&(_, &k)| k)
                .unwrap();
            topk_keys[wi] = key;
            topk_labels[wi] = self.labels[vi];
            *worst_key = *topk_keys.iter().max().unwrap();
        }
    }
}

// ── Fallback: tag-based lookup with clear-bit retry ───────────────────────────

pub fn search_with_fallback(tag: usize, q: &[i16; 16]) -> u8 {
    use crate::fraud::data::index_for_tag;

    if let Some(ix) = index_for_tag(tag) {
        return ix.search(q);
    }
    // Clear card_present bit and retry
    let tag2 = tag & !0b1000;
    if let Some(ix) = index_for_tag(tag2) {
        return ix.search(q);
    }
    // Clear is_online bit and retry
    let tag3 = tag2 & !0b0100;
    if let Some(ix) = index_for_tag(tag3) {
        return ix.search(q);
    }
    // Approve by default
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fraud::data::IvfIndex;

    // ── dist_l2_i16q tests ────────────────────────────────────────────────────

    #[test]
    fn dist_l2_zero_query_zero_vec() {
        let vecs = vec![0i16; 32];
        let q = [0i16; 16];
        assert_eq!(dist_l2_i16q(&vecs, 0, &q), 0);
    }

    #[test]
    fn dist_l2_matches_scalar() {
        let vecs: Vec<i16> = (0..32).map(|i| (i * 300 - 5000) as i16).collect();
        let q: [i16; 16] = core::array::from_fn(|i| (i as i16) * 200);
        let avx = dist_l2_i16q(&vecs, 0, &q);
        let scalar = dist_l2_i16q_scalar(&vecs, 0, &q);
        assert_eq!(avx, scalar, "AVX2 and scalar must agree");
    }

    #[test]
    fn dist_l2_sentinel_same() {
        let vecs = vec![-10000i16; 32];
        let q = [-10000i16; 16];
        assert_eq!(dist_l2_i16q(&vecs, 0, &q), 0);
    }

    #[test]
    fn dist_l2_max_no_overflow() {
        // Worst realistic: 12 dims at diff 10000, 2 sentinel dims at diff 20000, 2 padding at 0
        // dims 0-4, 7-13 (12 total): diff 10000 → 12 × 100_000_000 = 1_200_000_000
        // dims 5,6 (2 sentinel): diff 20000 → 2 × 400_000_000 = 800_000_000
        // dims 14,15 (2 padding): diff 0 → 0
        // Total: 2_000_000_000 (fits in i32 max = 2_147_483_647)
        let mut vecs = vec![10000i16; 32];
        let mut q = [0i16; 16];
        vecs[5] = 10000; q[5] = -10000;
        vecs[6] = 10000; q[6] = -10000;
        // Zero out padding dims 14,15
        vecs[14] = 0; vecs[15] = 0;
        q[14] = 0; q[15] = 0;
        let result = dist_l2_i16q(&vecs, 0, &q);
        let expected = dist_l2_i16q_scalar(&vecs, 0, &q);
        assert_eq!(result, expected, "must match scalar");
        assert_eq!(result, 2_000_000_000i32, "exact expected value");
    }

    // ── compute_cluster_batch8 tests ──────────────────────────────────────────

    #[test]
    fn batch8_query_inside_all_bboxes() {
        let min_soa = vec![0i16; N_PAIRS * 16];
        let max_soa = vec![5000i16; N_PAIRS * 16];
        let q: [i16; 16] = core::array::from_fn(|_| 2500i16);
        let mut lbs = [0i32; 8];
        compute_cluster_batch8(&min_soa, &max_soa, &q, &mut lbs);
        for l in 0..8 {
            assert_eq!(lbs[l], 0, "query inside bbox → lb must be 0 for cluster {l}");
        }
    }

    #[test]
    fn batch8_matches_scalar() {
        let mut min_soa = vec![0i16; N_PAIRS * 16];
        let mut max_soa = vec![5000i16; N_PAIRS * 16];
        min_soa[0] = 3000; max_soa[0] = 4000;
        min_soa[2] = 1000; max_soa[2] = 2000;
        let q: [i16; 16] = core::array::from_fn(|i| (i as i16) * 500);
        let mut lbs_avx = [0i32; 8];
        let mut lbs_scalar = [0i32; 8];
        compute_cluster_batch8(&min_soa, &max_soa, &q, &mut lbs_avx);
        compute_cluster_batch8_scalar(&min_soa, &max_soa, &q, &mut lbs_scalar);
        assert_eq!(lbs_avx, lbs_scalar, "AVX2 and scalar must agree");
    }

    // ── IvfIndex::search tests ────────────────────────────────────────────────

    #[test]
    fn search_returns_valid_count() {
        let ix = IvfIndex::open("index/index_p0.bin").unwrap();
        let q = [0i16; 16];
        let result = ix.search(&q);
        assert!(result <= 5, "search result must be 0..=5, got {result}");
    }

    #[test]
    fn search_deterministic() {
        let ix = IvfIndex::open("index/index_p1.bin").unwrap();
        let q: [i16; 16] = core::array::from_fn(|i| (i as i16) * 700);
        let r1 = ix.search(&q);
        let r2 = ix.search(&q);
        assert_eq!(r1, r2, "search must be deterministic");
    }

    #[test]
    fn search_with_fallback_no_panic() {
        crate::fraud::data::init_indices();
        for tag in 0..16 {
            let q = [0i16; 16];
            let _ = search_with_fallback(tag, &q);
        }
    }
}
