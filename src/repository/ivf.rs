use half::f16;
use smallvec::SmallVec;
use std::path::Path;

pub struct IvfIndex {
    k: usize,
    nprobe: usize,
    centroids: Vec<[f32; 14]>,
    lists: Vec<Vec<([f16; 14], u8)>>,
}

impl IvfIndex {
    pub fn load(path: &Path, nprobe: usize) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ivf_index.bin too short: missing header",
            ));
        }
        let mut pos = 0;

        let k = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let d = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if d != 14 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected D=14, got {d}"),
            ));
        }

        let centroid_bytes = k * 14 * 4;
        if data.len() < pos + centroid_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ivf_index.bin truncated: centroids",
            ));
        }
        let mut centroids = Vec::with_capacity(k);
        for _ in 0..k {
            let mut c = [0.0f32; 14];
            for elem in &mut c {
                *elem = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
            }
            centroids.push(c);
        }

        if data.len() < pos + k * 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ivf_index.bin truncated: list sizes",
            ));
        }
        let mut list_sizes = Vec::with_capacity(k);
        for _ in 0..k {
            let sz = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            list_sizes.push(sz);
            pos += 4;
        }

        let mut lists = Vec::with_capacity(k);
        for (i, &sz) in list_sizes.iter().enumerate() {
            let entry_bytes = sz * (14 * 2 + 1);
            if data.len() < pos + entry_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("ivf_index.bin truncated: cluster {i} entries"),
                ));
            }
            let mut list = Vec::with_capacity(sz);
            for _ in 0..sz {
                let mut vec = [f16::ZERO; 14];
                for elem in &mut vec {
                    *elem = f16::from_le_bytes([data[pos], data[pos + 1]]);
                    pos += 2;
                }
                let label = data[pos];
                pos += 1;
                list.push((vec, label));
            }
            lists.push(list);
        }

        Ok(Self {
            k,
            nprobe,
            centroids,
            lists,
        })
    }

    pub fn knn(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        let mut centroid_dists: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (centroid_sq_dist(query, c), i))
            .collect();
        centroid_dists
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let nprobe = self.nprobe.min(self.k);

        let mut top: Vec<(u32, u8)> = Vec::with_capacity(k + 1);
        for &(_, ci) in &centroid_dists[..nprobe] {
            for (vec, label) in &self.lists[ci] {
                let dist = vec_sq_dist(query, vec);
                if dist.is_nan() {
                    continue;
                }
                let dist_bits = dist.to_bits();
                if top.len() < k {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, *label));
                } else if dist_bits < top[top.len() - 1].0 {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, *label));
                    top.truncate(k);
                }
            }
        }

        top.iter().map(|&(_, label)| label).collect()
    }
}

fn centroid_sq_dist(query: &[f32; 14], centroid: &[f32; 14]) -> f32 {
    query
        .iter()
        .zip(centroid.iter())
        .map(|(&q, &c)| {
            let d = q - c;
            d * d
        })
        .sum()
}

fn vec_sq_dist(query: &[f32; 14], vec: &[f16; 14]) -> f32 {
    query
        .iter()
        .zip(vec.iter())
        .map(|(&q, &r)| {
            let d = q - r.to_f32();
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny_ivf_bytes() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&14u32.to_le_bytes());
        for _ in 0..14 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..14 {
            buf.extend_from_slice(&10.0f32.to_le_bytes());
        }
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        for _ in 0..3 {
            for _ in 0..14 {
                buf.extend_from_slice(&f16::from_f32(0.1).to_le_bytes());
            }
            buf.push(0u8);
        }
        for _ in 0..3 {
            for _ in 0..14 {
                buf.extend_from_slice(&f16::from_f32(10.0).to_le_bytes());
            }
            buf.push(1u8);
        }
        buf
    }

    fn write_tiny_ivf(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, make_tiny_ivf_bytes()).unwrap();
        path
    }

    #[test]
    fn test_load_parses_header() {
        let path = write_tiny_ivf("test_ivf_header.bin");
        let idx = IvfIndex::load(&path, 1).unwrap();
        assert_eq!(idx.k, 2);
        assert_eq!(idx.lists.len(), 2);
        assert_eq!(idx.lists[0].len(), 3);
        assert_eq!(idx.lists[1].len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_query_near_legit_cluster() {
        let path = write_tiny_ivf("test_ivf_legit.bin");
        let idx = IvfIndex::load(&path, 1).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 3);
        assert_eq!(labels.len(), 3);
        assert!(
            labels.iter().all(|&l| l == 0),
            "all neighbors near zero centroid should be legit"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_query_near_fraud_cluster() {
        let path = write_tiny_ivf("test_ivf_fraud.bin");
        let idx = IvfIndex::load(&path, 1).unwrap();
        let query = [10.0f32; 14];
        let labels = idx.knn(&query, 3);
        assert_eq!(labels.len(), 3);
        assert!(
            labels.iter().all(|&l| l == 1),
            "all neighbors near 10.0 centroid should be fraud"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_nprobe_2_returns_from_both_clusters() {
        let path = write_tiny_ivf("test_ivf_nprobe2.bin");
        let idx = IvfIndex::load(&path, 2).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 3);
        assert_eq!(labels.len(), 3);
        assert!(
            labels.iter().all(|&l| l == 0),
            "top-3 when near zero centroid should all be legit even with nprobe=2"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_rejects_truncated_file() {
        let path = std::env::temp_dir().join("test_ivf_truncated.bin");
        std::fs::write(&path, [0u8; 4]).unwrap();
        assert!(IvfIndex::load(&path, 1).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_nprobe_clamped_to_k() {
        let path = write_tiny_ivf("test_ivf_clamp");
        let idx = IvfIndex::load(&path, 999).unwrap(); // nprobe >> K
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 3);
        assert_eq!(labels.len(), 3);
        // With nprobe clamped to K=2, probes both clusters; nearest are legit
        assert!(labels.iter().all(|&l| l == 0), "top-3 should be legit");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_mixed_labels_ordered_by_distance() {
        let path = write_tiny_ivf("test_ivf_mixed");
        // nprobe=2: probes both clusters
        // Cluster 0 (near [0.0;14]): 3 legit vectors at ~0.1
        // Cluster 1 (near [10.0;14]): 3 fraud vectors at ~10.0
        // Query at [0.0;14]: top-5 should be: 3 legit (close) then 2 fraud (far)
        let idx = IvfIndex::load(&path, 2).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 5);
        assert_eq!(labels.len(), 5);
        // First 3 should be legit (closer), last 2 should be fraud (farther)
        let legit_count = labels.iter().filter(|&&l| l == 0).count();
        let fraud_count = labels.iter().filter(|&&l| l == 1).count();
        assert_eq!(legit_count, 3);
        assert_eq!(fraud_count, 2);
        // Verify order: all legit come before any fraud
        let first_fraud_pos = labels.iter().position(|&l| l == 1).unwrap_or(5);
        let last_legit_pos = labels.iter().rposition(|&l| l == 0).unwrap_or(0);
        assert!(
            last_legit_pos < first_fraud_pos,
            "all legit neighbors should rank before fraud neighbors"
        );
        std::fs::remove_file(&path).ok();
    }
}
