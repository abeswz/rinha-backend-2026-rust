use axum_test::TestServer;
use fraud_detection::{config::Config, web::router::build_router, AppState};
use once_cell::sync::Lazy;
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

/// Builds a minimal valid IVF2 binary for regression tests.
///
/// Two clusters based on the canonical legit/fraud transaction vectors
/// (both with last_transaction=null, so dims 5,6 = -1.0 in centroids).
///
/// Legit centroid: [0.004, 0.167, 0.05, 0.78, 0.33, -1.0, -1.0, 0.03, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006]
/// Fraud centroid: [0.95,  0.83,  1.0,  0.22, 0.83, -1.0, -1.0, 0.95, 1.0,  0.0, 1.0, 1.0, 0.75, 0.005]
fn make_test_ivf2() -> std::path::PathBuf {
    let legit_centroid: [f32; 14] = [0.004, 0.167, 0.05, 0.78, 0.33, -1.0, -1.0, 0.03, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006];
    let fraud_centroid: [f32; 14] = [0.95,  0.83,  1.0,  0.22, 0.83, -1.0, -1.0, 0.95, 1.0,  0.0, 1.0, 1.0, 0.75, 0.005];
    let legit_i16: [i16; 14] = legit_centroid.map(|v| (v * 10000.0) as i16);
    let fraud_i16: [i16; 14] = fraud_centroid.map(|v| (v * 10000.0) as i16);

    let mut buf = Vec::new();
    buf.extend_from_slice(b"IVF2");
    buf.extend_from_slice(&16u32.to_le_bytes()); // n
    buf.extend_from_slice(&2u32.to_le_bytes());  // k
    buf.extend_from_slice(&14u32.to_le_bytes()); // d

    // centroids column-major
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
    for val in legit_i16 {
        for _ in 0..8 { buf.extend_from_slice(&val.to_le_bytes()); }
    }
    for val in fraud_i16 {
        for _ in 0..8 { buf.extend_from_slice(&val.to_le_bytes()); }
    }

    let path = std::env::temp_dir().join("regression_test_ivf2.bin");
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
    Arc::new(AppState::build(&config).expect("AppState::build failed in regression test"))
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
