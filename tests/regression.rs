use axum_test::TestServer;
use fraud_detection::{config::Config, web::router::build_router, AppState};
use once_cell::sync::Lazy;
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

static STATE: Lazy<Arc<AppState>> = Lazy::new(|| {
    let config = Config {
        port: 3000,
        ivf_path: PathBuf::from("resources/ivf_index.bin"),
        mcc_path: PathBuf::from("resources/mcc_risk.json"),
        norm_path: PathBuf::from("resources/normalization.json"),
        nprobe_fast: 3,
        nprobe_slow: 8,
    };
    Arc::new(AppState::build(&config).expect("AppState init failed"))
});

fn test_server() -> TestServer {
    TestServer::new(build_router(STATE.clone()))
}

#[tokio::test]
async fn test_high_velocity_customer() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-velocity",
            "transaction": { "amount": 50.0, "installments": 1, "requested_at": "2026-03-11T14:00:00Z" },
            "customer": { "avg_amount": 50.0, "tx_count_24h": 20, "known_merchants": ["MERC-001"] },
            "merchant": { "id": "MERC-001", "mcc": "5411", "avg_amount": 50.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 2.0 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert!(body["approved"].is_boolean());
    assert!(body["fraud_score"].is_number());
}

#[tokio::test]
async fn test_suspicious_value_spike() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-spike",
            "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
            "customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008"] },
            "merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert_eq!(
        body["approved"], false,
        "suspicious value spike should be rejected"
    );
}

#[tokio::test]
async fn test_unknown_merchant_far_from_home() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-far",
            "transaction": { "amount": 500.0, "installments": 3, "requested_at": "2026-03-14T02:00:00Z" },
            "customer": { "avg_amount": 100.0, "tx_count_24h": 5, "known_merchants": ["MERC-001"] },
            "merchant": { "id": "MERC-999", "mcc": "7995", "avg_amount": 500.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.0 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    let score = body["fraud_score"].as_f64().unwrap();
    assert!(
        score > 0.4,
        "unknown merchant + far from home should yield high fraud_score, got {score}"
    );
}

#[tokio::test]
async fn test_first_time_customer() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-first",
            "transaction": { "amount": 45.0, "installments": 1, "requested_at": "2026-03-11T10:00:00Z" },
            "customer": { "avg_amount": 50.0, "tx_count_24h": 1, "known_merchants": ["MERC-003"] },
            "merchant": { "id": "MERC-003", "mcc": "5411", "avg_amount": 50.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 10.0 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert_eq!(
        body["approved"], true,
        "first-time customer with modest legit profile should be approved"
    );
}

#[tokio::test]
async fn test_all_fraud_signals() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-all-fraud",
            "transaction": { "amount": 10000.0, "installments": 12, "requested_at": "2026-03-14T03:00:00Z" },
            "customer": { "avg_amount": 10.0, "tx_count_24h": 20, "known_merchants": [] },
            "merchant": { "id": "MERC-UNKNOWN", "mcc": "7995", "avg_amount": 10000.0 },
            "terminal": { "is_online": true, "card_present": false, "km_from_home": 1000.0 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert_eq!(
        body["approved"], false,
        "all-fraud-signals should not be approved"
    );
    let score = body["fraud_score"].as_f64().unwrap();
    assert!(
        score >= 0.6,
        "all-fraud-signals score should be >= 0.6, got {score}"
    );
}
