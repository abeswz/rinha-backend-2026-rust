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
        if config.nprobe == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "IVF_NPROBE must be >= 1",
            ));
        }
        let repository = ReferenceRepository::from_file(&config.ivf_path, config.nprobe)?;
        let norm = NormalizationConstants::from_file(&config.norm_path)?;
        let mcc_risk = MccRiskMap::from_file(&config.mcc_path)?;
        let vectorizer = Vectorizer::new(norm, mcc_risk);
        Ok(Self {
            use_case: ScoreFraudUseCase {
                vectorizer,
                repository,
            },
        })
    }
}
