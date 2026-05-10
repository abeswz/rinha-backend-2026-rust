use super::handlers::{fraud_score_handler, ready_handler};
use crate::AppState;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/ready", get(ready_handler))
        .route("/fraud-score", post(fraud_score_handler))
        .with_state(state)
}
