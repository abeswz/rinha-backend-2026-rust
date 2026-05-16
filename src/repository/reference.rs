use super::ivf::IvfIndex;
use smallvec::SmallVec;
use std::path::Path;

pub struct ReferenceRepository {
    ivf: IvfIndex,
}

impl ReferenceRepository {
    pub fn from_file(path: &Path, nprobe: usize) -> std::io::Result<Self> {
        let ivf = IvfIndex::load(path, nprobe)?;
        Ok(Self { ivf })
    }

    pub fn knn(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        self.ivf.knn(query, k, self.ivf.nprobe_slow)
    }
}
