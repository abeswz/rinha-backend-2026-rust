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
        nprobe_slow: 8,
    };
    Arc::new(
        AppState::build(&config)
            .expect("AppState init failed — run `python tools/build_ivf.py` first"),
    )
});

fn test_server() -> TestServer {
    TestServer::new(build_router(STATE.clone()))
}

#[tokio::test]
async fn test_ready_endpoint() {
    let server = test_server();
    let resp = server.get("/ready").await;
    resp.assert_status_ok();
    resp.assert_text("ok");
}

#[tokio::test]
async fn test_legit_transaction_from_docs() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-1329056812",
            "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
            "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
            "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.2331036248 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert_eq!(
        body["approved"], true,
        "legit transaction should be approved"
    );
    assert_eq!(
        body["fraud_score"], 0.0,
        "legit transaction should have fraud_score=0.0"
    );
}

#[tokio::test]
async fn test_fraud_transaction_from_docs() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-3330991687",
            "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
            "customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
            "merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert_eq!(
        body["approved"], false,
        "fraud transaction should not be approved"
    );
    assert_eq!(
        body["fraud_score"], 1.0,
        "fraud transaction should have fraud_score=1.0"
    );
}

#[tokio::test]
async fn test_null_last_transaction_no_panic() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-null-test",
            "transaction": { "amount": 100.0, "installments": 1, "requested_at": "2026-03-11T12:00:00Z" },
            "customer": { "avg_amount": 100.0, "tx_count_24h": 1, "known_merchants": ["MERC-001"] },
            "merchant": { "id": "MERC-001", "mcc": "5411", "avg_amount": 100.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 5.0 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status_ok();
}

#[tokio::test]
async fn test_malformed_missing_field() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-bad",
            "transaction": { "amount": 100.0, "installments": 1, "requested_at": "2026-03-11T12:00:00Z" },
            "customer": { "avg_amount": 100.0, "tx_count_24h": 1, "known_merchants": [] },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 5.0 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status(axum::http::StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_malformed_invalid_timestamp() {
    let server = test_server();
    let resp = server
        .post("/fraud-score")
        .json(&json!({
            "id": "tx-bad-ts",
            "transaction": { "amount": 100.0, "installments": 1, "requested_at": "not-a-date" },
            "customer": { "avg_amount": 100.0, "tx_count_24h": 1, "known_merchants": [] },
            "merchant": { "id": "MERC-001", "mcc": "5411", "avg_amount": 100.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 5.0 },
            "last_transaction": null
        }))
        .await;
    resp.assert_status(axum::http::StatusCode::UNPROCESSABLE_ENTITY);
}
