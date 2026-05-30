use aligned_vec::{AVec, ConstAlign};
use flate2::read::GzDecoder;
use std::io::{BufReader, Read};
use std::sync::OnceLock;

static INDEX_GZ: &[u8] = include_bytes!("../../data/index.bin.gz");
static DATASET: OnceLock<Dataset> = OnceLock::new();

#[allow(dead_code)]
pub struct Dataset {
    pub n: usize,
    pub k: usize,
    pub centroids: AVec<f32, ConstAlign<32>>,
    pub offsets: Vec<u32>,
    pub labels: Vec<u8>,
    pub blocks: AVec<i16, ConstAlign<32>>,
}

pub fn dataset() -> &'static Dataset {
    DATASET.get().expect("call data::init() before dataset()")
}

pub fn init() {
    DATASET.get_or_init(decode);
}

// Read `count` elements of T directly into the spare capacity of v (no intermediate buffer).
unsafe fn fill_vec<T>(r: &mut impl Read, v: &mut Vec<T>, count: usize) {
    let spare = v.spare_capacity_mut();
    assert!(spare.len() >= count);
    let ptr = spare.as_mut_ptr() as *mut u8;
    r.read_exact(std::slice::from_raw_parts_mut(
        ptr,
        count * std::mem::size_of::<T>(),
    ))
    .expect("truncated index");
    v.set_len(v.len() + count);
}

// Same as fill_vec but for AVec (which has as_mut_ptr + set_len directly).
unsafe fn fill_avec<T>(r: &mut impl Read, v: &mut AVec<T, ConstAlign<32>>, count: usize) {
    assert!(v.capacity() >= v.len() + count);
    let ptr = v.as_mut_ptr().add(v.len()) as *mut u8;
    r.read_exact(std::slice::from_raw_parts_mut(
        ptr,
        count * std::mem::size_of::<T>(),
    ))
    .expect("truncated index");
    v.set_len(v.len() + count);
}

fn decode() -> Dataset {
    let mut gz = BufReader::new(GzDecoder::new(INDEX_GZ));

    let mut hdr = [0u8; 16];
    gz.read_exact(&mut hdr).expect("truncated header");
    assert_eq!(&hdr[..4], b"IVF1", "bad IVF1 magic");
    let n = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    assert_eq!(d, 14, "expected d=14, got {d}");

    // All reads go directly into pre-allocated memory. Peak RSS = binary + final arrays only.
    let centroid_count = d * k;
    let mut centroids: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, centroid_count);
    unsafe {
        fill_avec(&mut gz, &mut centroids, centroid_count);
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(k + 1);
    unsafe {
        fill_vec(&mut gz, &mut offsets, k + 1);
    }

    let total_blocks = offsets[k] as usize;
    let mut labels: Vec<u8> = Vec::with_capacity(total_blocks * 8);
    unsafe {
        fill_vec(&mut gz, &mut labels, total_blocks * 8);
    }

    let block_i16_count = total_blocks * d * 8;
    let mut blocks: AVec<i16, ConstAlign<32>> = AVec::with_capacity(32, block_i16_count);
    unsafe {
        fill_avec(&mut gz, &mut blocks, block_i16_count);
    }

    // Guard: if index was built with a different K, knn5_ivf silently gives wrong results.
    assert_eq!(
        k,
        crate::fraud::knn::K,
        "index k={k} != compiled K={}; rebuild index or update K const",
        crate::fraud::knn::K
    );

    Dataset {
        n,
        k,
        centroids,
        offsets,
        labels,
        blocks,
    }
}
