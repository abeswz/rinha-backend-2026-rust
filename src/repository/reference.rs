use super::ivf::IvfIndex;
use smallvec::SmallVec;
use std::path::Path;

pub struct ReferenceRepository {
    ivf: IvfIndex,
}

impl ReferenceRepository {
    pub fn from_file(path: &Path, nprobe_fast: usize, nprobe_slow: usize) -> std::io::Result<Self> {
        let ivf = IvfIndex::load(path, nprobe_fast, nprobe_slow)?;
        Ok(Self { ivf })
    }

    pub fn knn_adaptive(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        self.ivf.knn_adaptive(query, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tiny_repo_ivf(name: &str) -> std::path::PathBuf {
        let mut buf: Vec<u8> = Vec::new();
        // IVF2 format
        buf.extend_from_slice(b"IVF2");
        buf.extend_from_slice(&8u32.to_le_bytes()); // n=8 (1 block per cluster)
        buf.extend_from_slice(&2u32.to_le_bytes()); // k=2
        buf.extend_from_slice(&14u32.to_le_bytes()); // d=14

        // centroids column-major: C0=[0.0;14], C1=[10.0;14]
        for _ in 0..14 {
            buf.extend_from_slice(&0.0f32.to_le_bytes()); // C0
            buf.extend_from_slice(&10.0f32.to_le_bytes()); // C1
        }

        // offsets: [0, 1, 2]
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());

        // labels: 2 blocks × 8 slots
        for _ in 0..8 { buf.push(0u8); } // block 0: legit
        for _ in 0..8 { buf.push(1u8); } // block 1: fraud

        // blocks: 2 × 14 × 8 i16
        let legit_val: i16 = 1000;  // 0.1
        let fraud_val: i16 = i16::MAX; // padding / far away
        for _ in 0..112 { buf.extend_from_slice(&legit_val.to_le_bytes()); } // block 0
        for _ in 0..112 { buf.extend_from_slice(&fraud_val.to_le_bytes()); } // block 1

        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, buf).unwrap();
        path
    }

    #[test]
    fn test_knn_adaptive_legit_query() {
        let path = write_tiny_repo_ivf("repo_adapt_legit.bin");
        let repo = ReferenceRepository::from_file(&path, 5, 24).unwrap();
        let query = [0.0f32; 14];
        let labels = repo.knn_adaptive(&query, 5);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().filter(|&&l| l == 1).count() <= 2);
        std::fs::remove_file(&path).ok();
    }
}
