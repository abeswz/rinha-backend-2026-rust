#![allow(dead_code)]

mod config;
mod domain;
mod error;
mod repository;
mod service;
mod usecase;
mod web;

use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

use config::Config;
use repository::reference::ReferenceRepository;
use service::vectorizer::{MccRiskMap, NormalizationConstants, Vectorizer};
use usecase::score_fraud::ScoreFraudUseCase;

pub struct AppState {
    pub use_case: ScoreFraudUseCase,
}

impl AppState {
    pub fn build(config: &Config) -> std::io::Result<Self> {
        let repository = ReferenceRepository::from_file(&config.refs_path)?;
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

#[tokio::main]
async fn main() {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let config = Config::from_env();
    let state = Arc::new(AppState::build(&config).expect("failed to initialize AppState"));
    let addr = format!("0.0.0.0:{}", config.port);
    let router = web::router::build_router(state);

    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
