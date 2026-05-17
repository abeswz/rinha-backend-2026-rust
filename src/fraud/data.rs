use aligned_vec::{AVec, ConstAlign};
use flate2::read::GzDecoder;
use std::io::Read;
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

fn decode() -> Dataset {
    let mut gz = GzDecoder::new(INDEX_GZ);
    let mut raw: Vec<u8> = Vec::new();
    gz.read_to_end(&mut raw).expect("failed to decompress index");

    assert_eq!(&raw[..4], b"IVF1", "bad IVF1 magic");
    let mut pos = 4usize;

    macro_rules! read_u32 {
        () => {{
            let v = u32::from_le_bytes(raw[pos..pos+4].try_into().unwrap());
            pos += 4;
            v
        }};
    }

    let n = read_u32!() as usize;
    let k = read_u32!() as usize;
    let d = read_u32!() as usize;
    assert_eq!(d, 14, "expected d=14, got {d}");

    let centroid_count = d * k;
    let mut centroids: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, centroid_count);
    for _ in 0..centroid_count {
        centroids.push(f32::from_le_bytes(raw[pos..pos+4].try_into().unwrap()));
        pos += 4;
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(k + 1);
    for _ in 0..=k {
        offsets.push(u32::from_le_bytes(raw[pos..pos+4].try_into().unwrap()));
        pos += 4;
    }

    let total_blocks = offsets[k] as usize;
    let labels = raw[pos..pos + total_blocks * 8].to_vec();
    pos += total_blocks * 8;

    let block_i16_count = total_blocks * d * 8;
    let mut blocks: AVec<i16, ConstAlign<32>> = AVec::with_capacity(32, block_i16_count);
    for _ in 0..block_i16_count {
        blocks.push(i16::from_le_bytes(raw[pos..pos+2].try_into().unwrap()));
        pos += 2;
    }

    Dataset { n, k, centroids, offsets, labels, blocks }
}
