pub mod config;
pub mod domain;
pub mod error;
pub mod repository;
pub mod service;
pub mod usecase;
pub mod web;

use config::Config;
use repository::reference::ReferenceRepository;
use service::vectorizer::{MccRiskMap, NormalizationConstants, Vectorizer};
use usecase::score_fraud::ScoreFraudUseCase;

pub struct AppState {
    pub use_case: ScoreFraudUseCase,
}

impl AppState {
    pub fn build(config: &Config) -> std::io::Result<Self> {
        if config.nprobe_slow == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "IVF_NPROBE must be >= 1",
            ));
        }
        let repository = ReferenceRepository::from_file(&config.ivf_path, config.nprobe_slow)?;
        let norm = NormalizationConstants::from_file(&config.norm_path)?;
        let mcc_risk = MccRiskMap::from_file(&config.mcc_path)?;
        let vectorizer = Vectorizer::new(norm, mcc_risk);
        let state = Self {
            use_case: ScoreFraudUseCase {
                vectorizer,
                repository,
            },
        };
        // Prime CPU branch predictors and L2/L3 caches before serving traffic
        let warmup_query = [0.0f32; 14];
        for _ in 0..500 {
            let _ = state.use_case.repository.knn_adaptive(&warmup_query, 5);
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    #[test]
    fn test_appstate_build_warmup_does_not_panic() {
        // Warmup should complete without panic when index is loaded
        let config = Config {
            port: 3000,
            ivf_path: PathBuf::from("resources/ivf_index.bin"),
            mcc_path: PathBuf::from("resources/mcc_risk.json"),
            norm_path: PathBuf::from("resources/normalization.json"),
            nprobe_slow: 24,
        };
        // Only run if resources exist (CI/CD may not have them)
        if config.ivf_path.exists() {
            let state = AppState::build(&config);
            assert!(
                state.is_ok(),
                "AppState::build should succeed: {:?}",
                state.err()
            );
        }
    }
}
