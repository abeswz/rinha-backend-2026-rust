use half::f16;
use smallvec::SmallVec;
use std::path::Path;

pub struct ReferenceRepository {
    vectors: Box<[[f16; 14]]>,
    labels: Box<[u8]>,
}

impl ReferenceRepository {
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;

        if data.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "refs.bin too short",
            ));
        }

        let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let vec_bytes = n * 14 * 2;
        let expected = 4 + vec_bytes + n;

        if data.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "refs.bin size mismatch: expected {expected}, got {}",
                    data.len()
                ),
            ));
        }

        let vectors: Vec<[f16; 14]> = data[4..4 + vec_bytes]
            .chunks_exact(28) // 14 × 2
            .map(|chunk| {
                let mut arr = [f16::ZERO; 14];
                for (i, b) in chunk.chunks_exact(2).enumerate() {
                    arr[i] = f16::from_le_bytes([b[0], b[1]]);
                }
                arr
            })
            .collect();

        let labels: Vec<u8> = data[4 + vec_bytes..4 + vec_bytes + n].to_vec();

        Ok(Self {
            vectors: vectors.into_boxed_slice(),
            labels: labels.into_boxed_slice(),
        })
    }

    pub fn knn(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        let mut top: Vec<(u32, u8)> = Vec::with_capacity(k + 1);

        for (vec, &label) in self.vectors.iter().zip(self.labels.iter()) {
            let dist = squared_dist(query, vec);
            let dist_bits = dist.to_bits();

            if top.len() < k {
                let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                top.insert(pos, (dist_bits, label));
            } else if dist_bits < top[top.len() - 1].0 {
                let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                top.insert(pos, (dist_bits, label));
                top.truncate(k);
            }
        }

        top.iter().map(|&(_, label)| label).collect()
    }
}

#[inline(always)]
fn squared_dist(query: &[f32; 14], reference: &[f16; 14]) -> f32 {
    query
        .iter()
        .zip(reference.iter())
        .map(|(&q, &r)| {
            let diff = q - r.to_f32();
            diff * diff
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_repo(vectors_f32: Vec<[f32; 14]>, labels: Vec<u8>) -> ReferenceRepository {
        let vectors: Vec<[f16; 14]> = vectors_f32
            .iter()
            .map(|v| {
                let mut arr = [f16::ZERO; 14];
                for (i, &f) in v.iter().enumerate() {
                    arr[i] = f16::from_f32(f);
                }
                arr
            })
            .collect();
        ReferenceRepository {
            vectors: vectors.into_boxed_slice(),
            labels: labels.into_boxed_slice(),
        }
    }

    fn query_zeros() -> [f32; 14] {
        [0.0f32; 14]
    }

    #[test]
    fn test_knn_all_fraud() {
        let vecs = vec![[0.0f32; 14]; 10];
        let labels = vec![1u8; 10];
        let repo = make_repo(vecs, labels);
        let result = repo.knn(&query_zeros(), 5);
        assert_eq!(result.len(), 5);
        assert!(
            result.iter().all(|&l| l == 1),
            "all neighbors should be fraud"
        );
    }

    #[test]
    fn test_knn_all_legit() {
        let vecs = vec![[0.0f32; 14]; 10];
        let labels = vec![0u8; 10];
        let repo = make_repo(vecs, labels);
        let result = repo.knn(&query_zeros(), 5);
        assert_eq!(result.len(), 5);
        assert!(
            result.iter().all(|&l| l == 0),
            "all neighbors should be legit"
        );
    }

    #[test]
    fn test_knn_threshold_3_of_5() {
        let mut vecs = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..3 {
            let mut v = [0.0f32; 14];
            v[0] = 0.1;
            vecs.push(v);
            labels.push(1u8);
        }
        for _ in 0..2 {
            let mut v = [0.0f32; 14];
            v[0] = 0.5;
            vecs.push(v);
            labels.push(0u8);
        }
        for _ in 0..5 {
            let mut v = [0.0f32; 14];
            v[0] = 2.0;
            vecs.push(v);
            labels.push(0u8);
        }
        let repo = make_repo(vecs, labels);
        let result = repo.knn(&query_zeros(), 5);
        let fraud_count = result.iter().filter(|&&l| l == 1).count();
        assert_eq!(fraud_count, 3, "exactly 3 of 5 neighbors should be fraud");
    }

    #[test]
    fn test_knn_threshold_2_of_5() {
        let mut vecs = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..2 {
            let mut v = [0.0f32; 14];
            v[0] = 0.1;
            vecs.push(v);
            labels.push(1u8);
        }
        for _ in 0..3 {
            let mut v = [0.0f32; 14];
            v[0] = 0.5;
            vecs.push(v);
            labels.push(0u8);
        }
        for _ in 0..5 {
            let mut v = [0.0f32; 14];
            v[0] = 2.0;
            vecs.push(v);
            labels.push(1u8);
        }
        let repo = make_repo(vecs, labels);
        let result = repo.knn(&query_zeros(), 5);
        let fraud_count = result.iter().filter(|&&l| l == 1).count();
        assert_eq!(fraud_count, 2, "exactly 2 of 5 neighbors should be fraud");
    }
}
