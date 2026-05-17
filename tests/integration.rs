use axum_test::TestServer;
use fraud_detection::{config::Config, web::router::build_router, AppState};
use once_cell::sync::Lazy;
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

/// Builds a minimal valid IVF2 binary for integration tests.
///
/// Two clusters:
///   C0 (legit):  centroid near the legit example transaction vector
///   C1 (fraud):  centroid near the fraud example transaction vector
///
/// Both example transactions have last_transaction=null → dims 5,6 = -1.0.
/// Centroids use -1.0 at dims 5,6 so the nearest-cluster selection is not
/// dominated by those two dimensions.
///
/// Legit vector (approx):  [0.004, 0.167, 0.05, 0.78, 0.33, -1.0, -1.0, 0.03, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006]
/// Fraud vector (approx):  [0.95,  0.83,  1.0,  0.22, 0.83, -1.0, -1.0, 0.95, 1.0,  0.0, 1.0, 1.0, 0.75, 0.005]
fn make_test_ivf2() -> std::path::PathBuf {
    // Centroids, per dimension (column-major: [C0_d, C1_d] for each d)
    let legit_centroid: [f32; 14] = [0.004, 0.167, 0.05, 0.78, 0.33, -1.0, -1.0, 0.03, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006];
    let fraud_centroid: [f32; 14] = [0.95,  0.83,  1.0,  0.22, 0.83, -1.0, -1.0, 0.95, 1.0,  0.0, 1.0, 1.0, 0.75, 0.005];

    // i16 encoding of centroids for block vectors: round(v * 10000)
    let legit_i16: [i16; 14] = legit_centroid.map(|v| (v * 10000.0) as i16);
    let fraud_i16: [i16; 14] = fraud_centroid.map(|v| (v * 10000.0) as i16);

    let mut buf = Vec::new();
    buf.extend_from_slice(b"IVF2");
    buf.extend_from_slice(&16u32.to_le_bytes()); // n
    buf.extend_from_slice(&2u32.to_le_bytes());  // k
    buf.extend_from_slice(&14u32.to_le_bytes()); // d

    // centroids column-major: [C0_d0, C1_d0, C0_d1, C1_d1, ...]
    for d in 0..14 {
        buf.extend_from_slice(&legit_centroid[d].to_le_bytes());
        buf.extend_from_slice(&fraud_centroid[d].to_le_bytes());
    }

    // block_offsets: [0, 1, 2] — 1 block per cluster
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());

    // labels: 2 blocks × 8 slots
    buf.extend_from_slice(&[0u8; 8]); // block 0: legit
    buf.extend_from_slice(&[1u8; 8]); // block 1: fraud

    // blocks: 2 × 14 × 8 i16, layout: for each dim, 8 slots
    // block 0 (legit): 8 identical vectors at legit_centroid
    for val in legit_i16 {
        for _ in 0..8 {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    // block 1 (fraud): 8 identical vectors at fraud_centroid
    for val in fraud_i16 {
        for _ in 0..8 {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

    let path = std::env::temp_dir().join("integration_test_ivf2.bin");
    std::fs::write(&path, &buf).expect("write test IVF2");
    path
}

static STATE: Lazy<Arc<AppState>> = Lazy::new(|| {
    let config = Config {
        port: 3000,
        ivf_path: make_test_ivf2(),
        mcc_path: PathBuf::from("resources/mcc_risk.json"),
        norm_path: PathBuf::from("resources/normalization.json"),
        nprobe_fast: 3,
        nprobe_slow: 8,
    };
    Arc::new(AppState::build(&config).expect("AppState::build failed in integration test"))
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
