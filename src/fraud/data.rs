use memmap2::Mmap;
use std::fs::File;

pub const N_DIMS: usize = 14;
pub const N_PAIRS: usize = 7;     // 7 dim-pairs: (0,1),(2,3)...(12,13)
pub const N_PROBE_INITIAL: usize = 12;
pub const N_PROBE_REPAIR_MIN: usize = 1;
pub const N_PROBE_REPAIR_MAX: usize = 4;
pub const CID_BITS: u32 = 12;
pub const IDX_BITS: u32 = 22;
pub const CID_MASK: i64 = 0xFFF;
pub const K_NEIGHBORS: usize = 5;

const MAGIC: &[u8; 8] = b"RNH5-IDX";

fn align64(x: usize) -> usize {
    (x + 63) & !63
}

pub struct IvfIndex {
    _mmap: Mmap,
    pub n_clusters: usize,
    pub n_vectors: usize,
    pub cluster_offsets: &'static [u32],
    pub bbox_min: &'static [i16],   // n_clusters * 16
    pub bbox_max: &'static [i16],   // n_clusters * 16
    pub flat_vec: &'static [i16],   // (n_vectors+1) * 16
    pub labels: &'static [u8],
    pub bpsoa_min: Vec<i16>,        // n_groups * N_PAIRS * 16
    pub bpsoa_max: Vec<i16>,
}

// SAFETY: IvfIndex's 'static slices point into self._mmap which this struct owns.
// We never move the Mmap out, so the pointer stays valid for the struct's lifetime.
unsafe impl Send for IvfIndex {}
unsafe impl Sync for IvfIndex {}

impl IvfIndex {
    pub fn open(path: &str) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {path}: {e}"))?;

        if mmap.len() < 64 || &mmap[0..8] != MAGIC {
            return Err(format!("bad magic in {path}"));
        }

        let n_clusters = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let n_vectors  = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;

        if n_clusters < 1 || n_clusters > 4096 {
            return Err(format!("n_clusters={n_clusters} out of range in {path}"));
        }

        // Parse section offsets
        let off_offsets = 64usize;
        let off_bbox_min = align64(off_offsets + (n_clusters + 1) * 4);
        let off_bbox_max = align64(off_bbox_min + n_clusters * 32);
        let off_flat_vec = align64(off_bbox_max + n_clusters * 32);
        let off_labels   = align64(off_flat_vec + (n_vectors + 1) * 32);

        // SAFETY: We extend lifetimes to 'static because:
        // 1. `_mmap` is stored in the same struct, keeping the mapping alive.
        // 2. We never allow &mut access to these slices.
        // 3. The struct is not Send/Sync by default; we add explicit impls above.
        let data: &'static [u8] = unsafe {
            std::slice::from_raw_parts(mmap.as_ptr(), mmap.len())
        };

        let cluster_offsets: &'static [u32] = unsafe {
            std::slice::from_raw_parts(
                data[off_offsets..].as_ptr() as *const u32,
                n_clusters + 1,
            )
        };
        let bbox_min: &'static [i16] = unsafe {
            std::slice::from_raw_parts(
                data[off_bbox_min..].as_ptr() as *const i16,
                n_clusters * 16,
            )
        };
        let bbox_max: &'static [i16] = unsafe {
            std::slice::from_raw_parts(
                data[off_bbox_max..].as_ptr() as *const i16,
                n_clusters * 16,
            )
        };
        let flat_vec: &'static [i16] = unsafe {
            std::slice::from_raw_parts(
                data[off_flat_vec..].as_ptr() as *const i16,
                (n_vectors + 1) * 16,
            )
        };
        let labels: &'static [u8] = &data[off_labels..off_labels + n_vectors];

        // Lock pages and advise kernel
        unsafe {
            libc::mlock(mmap.as_ptr() as *const libc::c_void, mmap.len());
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_HUGEPAGE);
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_WILLNEED);
        }

        let mut ix = IvfIndex {
            _mmap: mmap,
            n_clusters,
            n_vectors,
            cluster_offsets,
            bbox_min,
            bbox_max,
            flat_vec,
            labels,
            bpsoa_min: Vec::new(),
            bpsoa_max: Vec::new(),
        };
        ix.build_bpsoa();
        Ok(ix)
    }

    fn build_bpsoa(&mut self) {
        let k = self.n_clusters;
        let n_groups = (k + 7) / 8;
        self.bpsoa_min = vec![0i16; n_groups * N_PAIRS * 16];
        self.bpsoa_max = vec![0i16; n_groups * N_PAIRS * 16];

        for g in 0..n_groups {
            for p in 0..N_PAIRS {
                let dst = (g * N_PAIRS + p) * 16;
                for l in 0..8usize {
                    let c = g * 8 + l;
                    let di = dst + l * 2;
                    if c < k {
                        self.bpsoa_min[di]     = self.bbox_min[c * 16 + 2 * p];
                        self.bpsoa_min[di + 1] = self.bbox_min[c * 16 + 2 * p + 1];
                        self.bpsoa_max[di]     = self.bbox_max[c * 16 + 2 * p];
                        self.bpsoa_max[di + 1] = self.bbox_max[c * 16 + 2 * p + 1];
                    } else {
                        // Padding clusters: infinite bbox → always outside query bbox
                        self.bpsoa_min[di]     = i16::MAX;
                        self.bpsoa_min[di + 1] = i16::MAX;
                        self.bpsoa_max[di]     = i16::MIN;
                        self.bpsoa_max[di + 1] = i16::MIN;
                    }
                }
            }
        }
    }
}

// TAG_PATHS maps the 4-bit tag to partition file name (or None for tags 12-15).
// Tag = card_present<<3 | is_online<<2 | is_unknown<<1 | has_last
static TAG_PATHS: [Option<&str>; 16] = [
    Some("index/index_p0.bin"),
    Some("index/index_p1.bin"),
    Some("index/index_p2.bin"),
    Some("index/index_p3.bin"),
    Some("index/index_p4.bin"),
    Some("index/index_p5.bin"),
    Some("index/index_p6.bin"),
    Some("index/index_p7.bin"),
    Some("index/index_p8.bin"),
    Some("index/index_p9.bin"),
    Some("index/index_p10.bin"),
    Some("index/index_p11.bin"),
    None, None, None, None,
];

// SAFETY: IvfIndex contains only immutable mmaps after init; global access is read-only.
#[allow(static_mut_refs)]
static mut INDICES: [Option<IvfIndex>; 16] = [
    None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None,
];

pub fn init_indices() {
    for (tag, maybe_path) in TAG_PATHS.iter().enumerate() {
        if let Some(path) = maybe_path {
            match IvfIndex::open(path) {
                Ok(ix) => unsafe { INDICES[tag] = Some(ix) },
                Err(e) => panic!("failed to load index for tag {tag}: {e}"),
            }
        }
    }
}

#[allow(static_mut_refs)]
pub fn index_for_tag(tag: usize) -> Option<&'static IvfIndex> {
    unsafe { INDICES[tag].as_ref() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_index_loads(tag: usize, path: &str) {
        let ix = IvfIndex::open(path).unwrap_or_else(|e| panic!("tag {tag}: {e}"));
        assert!(ix.n_clusters >= 1, "tag {tag}: n_clusters must be >= 1");
        assert!(ix.n_vectors >= 1, "tag {tag}: n_vectors must be >= 1");
        assert_eq!(
            ix.cluster_offsets.len(), ix.n_clusters + 1,
            "tag {tag}: cluster_offsets len"
        );
        assert_eq!(
            ix.bbox_min.len(), ix.n_clusters * 16,
            "tag {tag}: bbox_min len"
        );
        assert_eq!(
            ix.flat_vec.len(), (ix.n_vectors + 1) * 16,
            "tag {tag}: flat_vec len"
        );
        assert_eq!(ix.labels.len(), ix.n_vectors, "tag {tag}: labels len");
        // bpsoa shape
        let n_groups = (ix.n_clusters + 7) / 8;
        assert_eq!(
            ix.bpsoa_min.len(), n_groups * N_PAIRS * 16,
            "tag {tag}: bpsoa_min len"
        );
    }

    #[test]
    fn all_12_partitions_load() {
        for (tag, maybe_path) in TAG_PATHS.iter().enumerate() {
            if let Some(path) = maybe_path {
                assert_index_loads(tag, path);
            }
        }
    }

    #[test]
    fn bpsoa_padding_clusters_have_sentinel() {
        // Find a partition with K not a multiple of 8
        for (tag, maybe_path) in TAG_PATHS.iter().enumerate() {
            let Some(path) = maybe_path else { continue };
            let ix = IvfIndex::open(path).unwrap();
            let k = ix.n_clusters;
            if k % 8 != 0 {
                let n_groups = (k + 7) / 8;
                let g = n_groups - 1;
                let last_real = k % 8;
                let p = 0usize;
                let l = last_real;
                let di = (g * N_PAIRS + p) * 16 + l * 2;
                assert_eq!(ix.bpsoa_min[di], i16::MAX, "tag {tag}: padding min should be i16::MAX");
                assert_eq!(ix.bpsoa_max[di], i16::MIN, "tag {tag}: padding max should be i16::MIN");
                return;
            }
        }
        // All partitions have K % 8 == 0; acceptable
    }

    #[test]
    fn cluster_offsets_monotone() {
        let ix = IvfIndex::open("index/index_p0.bin").unwrap();
        let offs = ix.cluster_offsets;
        for i in 1..offs.len() {
            assert!(offs[i] >= offs[i-1], "offsets must be monotonically non-decreasing at {i}");
        }
        assert_eq!(offs[ix.n_clusters] as usize, ix.n_vectors, "last offset must equal n_vectors");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fraud::knn::K;

    #[test]
    fn ivf3_radii_loaded() {
        init();
        let ds = dataset();
        assert_eq!(ds.radii.len(), K, "radii count must equal K");
        assert!(
            ds.radii.iter().all(|&r| r >= 0.0 && r.is_finite()),
            "all radii must be non-negative and finite"
        );
    }
}
