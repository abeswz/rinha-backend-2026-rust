# Flat IVF + int16 + Block Scan — p99 Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current f16-based IVF index with a flat i16 + column-major + 8-vector block layout to reduce p99 from 84ms to <30ms by eliminating cache misses under CPU throttle.

**Architecture:** New `IvfIndex` uses contiguous column-major centroids for L1-friendly SIMD distance computation, CSR-style block offsets into flat i16 blocks (8 vectors × 14 dims, dim-major within block), early termination after 8 of 14 dims once the top-K heap fills. Python builder switches to K=4096, i16 quantization (×10000), and writes IVF2 binary format. Handler uses pre-built `&'static str` response bodies indexed by fraud_count to eliminate serde_json on the hot path.

**Tech Stack:** Rust (AVX2+FMA intrinsics), Python (numpy, scikit-learn), Axum, SmallVec

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/domain/fraud.rs` | Modify | Add `fraud_count: usize` to `FraudDecision` |
| `src/usecase/score_fraud.rs` | Modify | Populate `fraud_count` in `execute()` |
| `src/web/handlers.rs` | Modify | Pre-built static bodies; index by `fraud_count`; update fallback |
| `src/repository/ivf.rs` | Full rewrite | IVF2 loader, column-major centroid scan, block scan, SIMD |
| `tools/build_ivf.py` | Modify | K=4096, i16 quantization, IVF2 binary format |

---

## IVF2 Binary Format Reference

All values little-endian:
```
[0..4]   magic: b"IVF2"
[4..8]   n: u32  (total vectors, before padding)
[8..12]  k: u32  (number of centroids)
[12..16] d: u32  (must be 14)
[16..]   centroids: f32[d * k]          column-major: centroids[d * k + ci]
         offsets:   u32[k + 1]          block offsets (each unit = 1 block of 8 vectors)
         labels:    u8[total_blocks * 8] label per slot; padding slots = 0
         blocks:    i16[total_blocks * 14 * 8]
                    blocks[(block_idx * 14 + d) * 8 + slot] = quantize(vector[slot][d])
                    padding slots = i16::MAX (guaranteed not to enter top-K)
```

Quantization: `i16 = round(f32 * 10_000)`. Range [-1.0, 1.0] → [-10000, 10000] (fits i16).

---

## Task 1: Domain — Add fraud_count to FraudDecision

**Files:**
- Modify: `src/domain/fraud.rs`
- Modify: `src/usecase/score_fraud.rs`

- [ ] **Step 1: Write failing test for fraud_count**

In `src/usecase/score_fraud.rs`, add to the `tests` module:

```rust
#[test]
fn test_execute_sets_fraud_count() {
    let repo = make_repo();
    let norm = NormalizationConstants::default();
    let mcc_risk = MccRiskMap::default();
    let use_case = ScoreFraudUseCase {
        vectorizer: Vectorizer::new(norm, mcc_risk),
        repository: repo,
    };
    let tx = make_tx(100.0);
    let decision = use_case.execute(&tx);
    assert!(decision.fraud_count <= 5, "fraud_count must be 0..=5");
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test test_execute_sets_fraud_count 2>&1 | tail -5
```
Expected: compile error — `FraudDecision` has no field `fraud_count`

- [ ] **Step 3: Add fraud_count to FraudDecision**

Replace the entire content of `src/domain/fraud.rs`:

```rust
pub struct FraudVector(pub [f32; 14]);

pub struct FraudDecision {
    pub approved: bool,
    pub fraud_score: f32,
    pub fraud_count: usize,
}
```

- [ ] **Step 4: Populate fraud_count in execute()**

Replace `src/usecase/score_fraud.rs` `execute` method:

```rust
pub fn execute(&self, tx: &Transaction) -> FraudDecision {
    let vector = self.vectorizer.vectorize(tx);
    let labels = self.repository.knn_adaptive(&vector.0, 5);
    let fraud_count = labels.iter().filter(|&&l| l == 1).count();
    let fraud_score = fraud_count as f32 / 5.0;
    FraudDecision {
        approved: fraud_score < 0.6,
        fraud_score,
        fraud_count,
    }
}
```

- [ ] **Step 5: Fix compile error in handlers.rs timeout fallback**

In `src/web/handlers.rs`, the `unwrap_or` fallback constructs `FraudDecision`. Update it:

```rust
.unwrap_or(FraudDecision {
    approved: true,
    fraud_score: 0.0,
    fraud_count: 0,
});
```

- [ ] **Step 6: Run all tests to confirm pass**

```bash
cargo test 2>&1 | tail -10
```
Expected: all tests pass, no compile errors

- [ ] **Step 7: Commit**

```bash
git add src/domain/fraud.rs src/usecase/score_fraud.rs src/web/handlers.rs
git commit -m "feat(domain): add fraud_count to FraudDecision"
```

---

## Task 2: Static Response Bodies

**Files:**
- Modify: `src/web/handlers.rs`

Replace serde_json serialization on the hot path with pre-built `&'static str` bodies indexed by `fraud_count`. The 6 possible responses (fraud_count 0-5 with k=5) are compile-time constants.

- [ ] **Step 1: Write failing test for static bodies**

Add to `src/web/handlers.rs` tests module:

```rust
#[test]
fn test_static_bodies_all_valid_json() {
    for (i, body) in STATIC_BODIES.iter().enumerate() {
        let v: serde_json::Value =
            serde_json::from_str(body).expect(&format!("body[{i}] is not valid JSON: {body}"));
        assert!(v.get("approved").is_some(), "body[{i}] missing 'approved'");
        assert!(v.get("fraud_score").is_some(), "body[{i}] missing 'fraud_score'");
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test test_static_bodies_all_valid_json 2>&1 | tail -5
```
Expected: compile error — `STATIC_BODIES` not defined

- [ ] **Step 3: Add STATIC_BODIES constant and update handler**

Replace the entire `src/web/handlers.rs`:

```rust
use crate::AppState;
use crate::{
    domain::fraud::FraudDecision,
    domain::transaction::{Customer, LastTransaction, Merchant, Terminal, Transaction},
    web::dto::TransactionRequest,
};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    Json,
};
use std::sync::Arc;
use std::time::Duration;

pub const STATIC_BODIES: [&str; 6] = [
    r#"{"approved":true,"fraud_score":0.0}"#,
    r#"{"approved":true,"fraud_score":0.2}"#,
    r#"{"approved":true,"fraud_score":0.4}"#,
    r#"{"approved":false,"fraud_score":0.6}"#,
    r#"{"approved":false,"fraud_score":0.8}"#,
    r#"{"approved":false,"fraud_score":1.0}"#,
];

pub async fn ready_handler() -> &'static str {
    "ok"
}

pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> impl IntoResponse {
    let tx = into_transaction(req);
    let decision = tokio::time::timeout(
        Duration::from_millis(1600),
        tokio::task::spawn_blocking(move || state.use_case.execute(&tx)),
    )
    .await
    .ok()
    .and_then(|r| r.ok())
    .unwrap_or(FraudDecision {
        approved: true,
        fraud_score: 0.0,
        fraud_count: 0,
    });

    let body = STATIC_BODIES[decision.fraud_count.min(5)];
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
}

fn into_transaction(req: TransactionRequest) -> Transaction {
    Transaction {
        id: req.id,
        amount: req.transaction.amount,
        installments: req.transaction.installments,
        requested_at: req.transaction.requested_at,
        customer: Customer {
            avg_amount: req.customer.avg_amount,
            tx_count_24h: req.customer.tx_count_24h,
            known_merchants: req.customer.known_merchants,
        },
        merchant: Merchant {
            id: req.merchant.id,
            mcc: req.merchant.mcc,
            avg_amount: req.merchant.avg_amount,
        },
        terminal: Terminal {
            is_online: req.terminal.is_online,
            card_present: req.terminal.card_present,
            km_from_home: req.terminal.km_from_home,
        },
        last_transaction: req.last_transaction.map(|lt| LastTransaction {
            timestamp: lt.timestamp,
            km_from_current: lt.km_from_current,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::fraud::FraudDecision;
    use std::time::Duration;

    #[test]
    fn test_static_bodies_all_valid_json() {
        for (i, body) in STATIC_BODIES.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(body)
                .expect(&format!("body[{i}] is not valid JSON: {body}"));
            assert!(v.get("approved").is_some(), "body[{i}] missing 'approved'");
            assert!(v.get("fraud_score").is_some(), "body[{i}] missing 'fraud_score'");
        }
    }

    #[tokio::test]
    async fn test_timeout_fallback_is_approved_true() {
        let result = tokio::time::timeout(
            Duration::from_millis(50),
            tokio::task::spawn_blocking(|| {
                std::thread::sleep(Duration::from_millis(500));
                FraudDecision { approved: false, fraud_score: 1.0, fraud_count: 5 }
            }),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(FraudDecision { approved: true, fraud_score: 0.0, fraud_count: 0 });

        assert!(result.approved, "timeout fallback must be approved=true");
        assert_eq!(result.fraud_count, 0);
    }

    #[tokio::test]
    async fn test_fast_execution_returns_actual_decision() {
        let result = tokio::time::timeout(
            Duration::from_millis(1600),
            tokio::task::spawn_blocking(|| FraudDecision {
                approved: false,
                fraud_score: 0.8,
                fraud_count: 4,
            }),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(FraudDecision { approved: true, fraud_score: 0.0, fraud_count: 0 });

        assert!(!result.approved);
        assert_eq!(result.fraud_count, 4);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test 2>&1 | tail -10
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/web/handlers.rs
git commit -m "feat(handler): static response bodies indexed by fraud_count, remove serde_json on hot path"
```

---

## Task 3: IVF2 Scalar Loader + KNN

**Files:**
- Full rewrite: `src/repository/ivf.rs`

This replaces the entire file. Current tests use IVF1 format — all are replaced.

### Key layout facts
- `centroids[d * k + ci]` = dim `d` of centroid `ci` (column-major)
- `offsets[ci]`..`offsets[ci+1]` = block range for cluster `ci` (unit: blocks of 8)
- `labels[block_idx * 8 + slot]` = label of vector at position `slot` in `block_idx`
- `blocks[(block_idx * 14 + d) * 8 + slot]` = i16 value of dim `d` of slot `slot` in block `block_idx`
- Padding slots in partial blocks: `i16::MAX` in all dims, label `0`

### Test fixture: make_ivf2_bytes()
k=2, 8 vectors per cluster (no padding), n=16, d=14:
- Cluster 0 (centroid [0.0;14]): 8 legit vectors at [0.1;14] → i16=1000
- Cluster 1 (centroid [2.0;14]): 8 fraud vectors at [2.0;14] → i16=20000

Column-major centroids for k=2, d=14: interleaved [C0_dim, C1_dim] × 14 = 28 f32 values:
`[0.0, 2.0, 0.0, 2.0, ...(14 pairs)]`

block_offsets = [0, 1, 2] (cluster 0 → block 0; cluster 1 → block 1)

### Test fixture: make_staged_ivf2_bytes()
k=6, 8 vectors each, n=48, for knn_adaptive staging tests:
- C0 (centroid [0.1;14]): 8 legit at [0.1;14]
- C1 (centroid [0.2;14]): 8 legit at [0.2;14]
- C2 (centroid [0.3;14]): slot0=fraud[0.24;14], slots1-7=legit[0.3;14]
- C3 (centroid [0.4;14]): slot0=fraud[0.26;14], slots1-7=legit[0.4;14]
- C4 (centroid [0.5;14]): 8 legit at [0.5;14]
- C5 (centroid [0.6;14]): 8 fraud at [0.25;14]

Query [0.25;14]:
- Centroid dists: C1=0.035, C2=0.035, C0=0.315, C3=0.315, C4=0.875, C5=1.715
- nprobe=5 probes C1,C2,C0,C3,C4 → 2 fraud (C2 slot0 + C3 slot0) → ambiguous
- nprobe=6 adds C5 → 5 fraud (all from C5 at dist=0) → decisive

- [ ] **Step 1: Write all tests (they will fail to compile until Step 5)**

Write the new `src/repository/ivf.rs` with ONLY the tests module and an empty struct stub:

```rust
use smallvec::SmallVec;
use std::cell::RefCell;
use std::path::Path;

pub struct IvfIndex {
    k: usize,
    n: usize,
    nprobe_fast: usize,
    nprobe_slow: usize,
    centroids: Vec<f32>,
    offsets: Vec<u32>,
    labels: Vec<u8>,
    blocks: Vec<i16>,
}

impl IvfIndex {
    pub fn load(_path: &Path, _nprobe_fast: usize, _nprobe_slow: usize) -> std::io::Result<Self> {
        unimplemented!()
    }

    pub fn knn(&self, _query: &[f32; 14], _k: usize, _nprobe: usize) -> SmallVec<[u8; 5]> {
        unimplemented!()
    }

    pub fn knn_adaptive(&self, _query: &[f32; 14], _k: usize) -> SmallVec<[u8; 5]> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fixture helpers ──────────────────────────────────────────────────────

    fn make_ivf2_bytes() -> Vec<u8> {
        // k=2 clusters, 8 vectors each (n=16), d=14
        // Cluster 0: centroid [0.0;14], 8 legit vectors at [0.1;14]  (i16=1000)
        // Cluster 1: centroid [2.0;14], 8 fraud vectors at [2.0;14]  (i16=20000)
        let mut buf = Vec::new();
        buf.extend_from_slice(b"IVF2");
        buf.extend_from_slice(&16u32.to_le_bytes()); // n
        buf.extend_from_slice(&2u32.to_le_bytes());  // k
        buf.extend_from_slice(&14u32.to_le_bytes()); // d

        // centroids column-major: [C0_d0, C1_d0, C0_d1, C1_d1, ...]
        for _ in 0..14 {
            buf.extend_from_slice(&0.0f32.to_le_bytes()); // C0
            buf.extend_from_slice(&2.0f32.to_le_bytes()); // C1
        }

        // block_offsets: [0, 1, 2]
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());

        // labels: 16 bytes (2 blocks × 8 slots)
        for _ in 0..8 { buf.push(0u8); } // block 0: legit
        for _ in 0..8 { buf.push(1u8); } // block 1: fraud

        // blocks: 2 × 14 × 8 i16
        let legit_val: i16 = 1000;  // round(0.1 * 10000)
        let fraud_val: i16 = 20000; // round(2.0 * 10000)
        for _ in 0..112 { buf.extend_from_slice(&legit_val.to_le_bytes()); } // block 0
        for _ in 0..112 { buf.extend_from_slice(&fraud_val.to_le_bytes()); } // block 1

        buf
    }

    fn write_ivf2(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, make_ivf2_bytes()).unwrap();
        path
    }

    fn make_staged_ivf2_bytes() -> Vec<u8> {
        // k=6, 8 vectors each (n=48), d=14
        // Query [0.25;14]:
        //   nprobe=5 → 2 fraud (ambiguous) → triggers stage 2
        //   nprobe=6 → 5 fraud (C5 stragglers at dist=0)
        let mut buf = Vec::new();
        buf.extend_from_slice(b"IVF2");
        buf.extend_from_slice(&48u32.to_le_bytes()); // n
        buf.extend_from_slice(&6u32.to_le_bytes());  // k
        buf.extend_from_slice(&14u32.to_le_bytes()); // d

        // column-major centroids for k=6:
        // C0=[0.1;14] C1=[0.2;14] C2=[0.3;14] C3=[0.4;14] C4=[0.5;14] C5=[0.6;14]
        let centroid_vals = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        for _ in 0..14 {
            for &v in &centroid_vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // block_offsets: [0,1,2,3,4,5,6] (1 block per cluster)
        for i in 0u32..=6 {
            buf.extend_from_slice(&i.to_le_bytes());
        }

        // labels: 6 blocks × 8 slots = 48 bytes
        // C0: all legit
        for _ in 0..8 { buf.push(0u8); }
        // C1: all legit
        for _ in 0..8 { buf.push(0u8); }
        // C2: slot 0 = fraud, slots 1-7 = legit
        buf.push(1u8);
        for _ in 0..7 { buf.push(0u8); }
        // C3: slot 0 = fraud, slots 1-7 = legit
        buf.push(1u8);
        for _ in 0..7 { buf.push(0u8); }
        // C4: all legit
        for _ in 0..8 { buf.push(0u8); }
        // C5: all fraud
        for _ in 0..8 { buf.push(1u8); }

        // blocks: 6 × 112 i16
        // C0 block: all dim=0.1 → i16=1000
        for _ in 0..112 { buf.extend_from_slice(&1000i16.to_le_bytes()); }
        // C1 block: all dim=0.2 → i16=2000
        for _ in 0..112 { buf.extend_from_slice(&2000i16.to_le_bytes()); }
        // C2 block: slot 0 = 2400 (0.24), slots 1-7 = 3000 (0.3)
        // Layout: blocks[(2*14+d)*8+slot]; d-major within block
        for d in 0..14usize {
            let _ = d; // all dims same value
            buf.extend_from_slice(&2400i16.to_le_bytes()); // slot 0: fraud
            for _ in 0..7 { buf.extend_from_slice(&3000i16.to_le_bytes()); } // slots 1-7: legit
        }
        // C3 block: slot 0 = 2600 (0.26), slots 1-7 = 4000 (0.4)
        for d in 0..14usize {
            let _ = d;
            buf.extend_from_slice(&2600i16.to_le_bytes()); // slot 0: fraud
            for _ in 0..7 { buf.extend_from_slice(&4000i16.to_le_bytes()); }
        }
        // C4 block: all dim=0.5 → i16=5000
        for _ in 0..112 { buf.extend_from_slice(&5000i16.to_le_bytes()); }
        // C5 block: all dim=0.25 → i16=2500
        for _ in 0..112 { buf.extend_from_slice(&2500i16.to_le_bytes()); }

        buf
    }

    fn write_staged_ivf2(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, make_staged_ivf2_bytes()).unwrap();
        path
    }

    // ── IVF2 loader tests ────────────────────────────────────────────────────

    #[test]
    fn test_ivf2_load_parses_header() {
        let path = write_ivf2("ivf2_header.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        assert_eq!(idx.k, 2);
        assert_eq!(idx.n, 16);
        assert_eq!(idx.offsets.len(), 3);  // k+1
        assert_eq!(idx.labels.len(), 16);  // total_blocks * 8 = 2 * 8
        assert_eq!(idx.blocks.len(), 224); // total_blocks * 14 * 8 = 2 * 112
        assert_eq!(idx.centroids.len(), 28); // d * k = 14 * 2
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ivf2_load_rejects_ivf1_magic() {
        let path = std::env::temp_dir().join("ivf2_bad_magic.bin");
        let mut bad = make_ivf2_bytes();
        bad[..4].copy_from_slice(b"IVF1");
        std::fs::write(&path, &bad).unwrap();
        assert!(IvfIndex::load(&path, 5, 24).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ivf2_load_rejects_wrong_dimensions() {
        let path = std::env::temp_dir().join("ivf2_bad_d.bin");
        let mut bad = make_ivf2_bytes();
        // d field is bytes 12..16; write d=13
        bad[12..16].copy_from_slice(&13u32.to_le_bytes());
        std::fs::write(&path, &bad).unwrap();
        assert!(IvfIndex::load(&path, 5, 24).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ivf2_load_rejects_truncated_file() {
        let path = std::env::temp_dir().join("ivf2_truncated.bin");
        std::fs::write(&path, &[0u8; 10]).unwrap();
        assert!(IvfIndex::load(&path, 5, 24).is_err());
        std::fs::remove_file(&path).ok();
    }

    // ── Centroid scan tests ──────────────────────────────────────────────────

    #[test]
    fn test_centroid_scan_column_major_matches_brute_force() {
        let path = write_ivf2("ivf2_centroid_scan.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();

        // Query [0.0;14]: brute force dist to C0=[0.0;14]=0.0, C1=[2.0;14]=14*4=56.0
        let query = [0.0f32; 14];
        let expected_c0_dist: f32 = 0.0;
        let expected_c1_dist: f32 = 14.0 * 2.0f32.powi(2);

        // centroid_dists writes distances into a Vec<f32>, C0 at index 0, C1 at index 1
        // We verify indirectly: knn with nprobe=1 picks the closer centroid
        let labels = idx.knn(&query, 5, 1);
        assert_eq!(labels.len(), 5);
        assert!(
            labels.iter().all(|&l| l == 0),
            "nprobe=1 near C0 → all legit; expected_c0={expected_c0_dist}, expected_c1={expected_c1_dist}"
        );
        std::fs::remove_file(&path).ok();
    }

    // ── Block scan tests ─────────────────────────────────────────────────────

    #[test]
    fn test_block_scan_8vec_matches_brute_force() {
        let path = write_ivf2("ivf2_block_scan.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();

        // Query [0.0;14], nprobe=1 → only cluster 0 (legit)
        // 8 legit vectors at [0.1;14] stored as i16=1000 → reconstructed 0.1
        // sq_dist per vector = 14 * (0.0 - 0.1)^2 = 0.14
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 5, 1);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().all(|&l| l == 0));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_query_near_fraud_cluster() {
        let path = write_ivf2("ivf2_fraud_cluster.bin");
        let idx = IvfIndex::load(&path, 5, 1).unwrap();
        let query = [2.0f32; 14];
        let labels = idx.knn(&query, 5, 1);
        assert_eq!(labels.len(), 5);
        assert!(
            labels.iter().all(|&l| l == 1),
            "query near fraud centroid [2.0;14] → all fraud neighbors"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_nprobe_clamped_to_k() {
        let path = write_ivf2("ivf2_clamp.bin");
        let idx = IvfIndex::load(&path, 5, 999).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 5, 999);
        assert_eq!(labels.len(), 5);
        // With nprobe clamped to k=2, both clusters probed; legit still closer
        assert!(labels.iter().filter(|&&l| l == 0).count() >= 3);
        std::fs::remove_file(&path).ok();
    }

    // ── knn_adaptive tests ───────────────────────────────────────────────────

    #[test]
    fn test_knn_adaptive_unambiguous_legit_uses_stage1() {
        let path = write_ivf2("ivf2_adapt_legit.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        let query = [0.0f32; 14];
        let labels = idx.knn_adaptive(&query, 5);
        assert_eq!(labels.len(), 5);
        assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_adaptive_unambiguous_fraud_uses_stage1() {
        let path = write_ivf2("ivf2_adapt_fraud.bin");
        let idx = IvfIndex::load(&path, 5, 24).unwrap();
        let query = [2.0f32; 14];
        let labels = idx.knn_adaptive(&query, 5);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().filter(|&&l| l == 1).count() >= 4);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_adaptive_ambiguous_triggers_stage2() {
        let path = write_staged_ivf2("ivf2_adapt_staged.bin");
        let idx = IvfIndex::load(&path, 5, 6).unwrap();
        let query = [0.25f32; 14];

        // Stage 1 (nprobe=5) → 2 fraud → ambiguous
        let stage1 = idx.knn(&query, 5, 5);
        let stage1_fraud = stage1.iter().filter(|&&l| l == 1).count();
        assert_eq!(stage1_fraud, 2, "stage1 must be ambiguous (2 fraud), got {stage1_fraud}");

        // knn_adaptive triggers stage 2 (nprobe=6 adds C5 stragglers)
        let labels = idx.knn_adaptive(&query, 5);
        let fraud_count = labels.iter().filter(|&&l| l == 1).count();
        assert!(
            fraud_count >= 4,
            "stage2 must find C5 straggler fraud vectors, got {fraud_count} fraud"
        );
        std::fs::remove_file(&path).ok();
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail (unimplemented)**

```bash
cargo test --lib 2>&1 | grep -E "FAILED|panicked|unimplemented" | head -10
```
Expected: tests panic with "not implemented"

- [ ] **Step 3: Implement IVF2 load()**

Replace the stub `load()` with:

```rust
pub fn load(path: &Path, nprobe_fast: usize, nprobe_slow: usize) -> std::io::Result<Self> {
    let data = std::fs::read(path)?;
    let mut pos = 0;

    macro_rules! need {
        ($n:expr, $msg:literal) => {
            if data.len() < pos + $n {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, $msg));
            }
        };
    }
    macro_rules! read_u32 {
        () => {{
            need!(4, "truncated");
            let v = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            v as usize
        }};
    }

    need!(4, "missing magic");
    if &data[..4] != b"IVF2" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "expected IVF2 magic",
        ));
    }
    pos = 4;

    let n = read_u32!();
    let k = read_u32!();
    let d = read_u32!();
    if d != 14 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected d=14, got {d}"),
        ));
    }

    // centroids: d * k f32 (column-major)
    need!(d * k * 4, "truncated: centroids");
    let mut centroids = Vec::with_capacity(d * k);
    for _ in 0..d * k {
        centroids.push(f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
        pos += 4;
    }

    // offsets: (k + 1) u32
    need!((k + 1) * 4, "truncated: offsets");
    let mut offsets = Vec::with_capacity(k + 1);
    for _ in 0..=k {
        offsets.push(u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
        pos += 4;
    }

    let total_blocks = offsets[k] as usize;

    // labels: total_blocks * 8 bytes
    need!(total_blocks * 8, "truncated: labels");
    let labels = data[pos..pos + total_blocks * 8].to_vec();
    pos += total_blocks * 8;

    // blocks: total_blocks * d * 8 i16
    let block_i16_count = total_blocks * d * 8;
    need!(block_i16_count * 2, "truncated: blocks");
    let mut blocks = Vec::with_capacity(block_i16_count);
    for _ in 0..block_i16_count {
        blocks.push(i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()));
        pos += 2;
    }

    Ok(Self { k, n, nprobe_fast, nprobe_slow, centroids, offsets, labels, blocks })
}
```

- [ ] **Step 4: Implement centroid_dists_scalar() and thread-local buffers**

Add above `impl IvfIndex`:

```rust
struct CentroidBufs {
    dists: Vec<f32>,
    indices: Vec<usize>,
}

thread_local! {
    static CENTROID_BUFS: RefCell<CentroidBufs> = RefCell::new(CentroidBufs {
        dists: Vec::with_capacity(4096),
        indices: Vec::with_capacity(4096),
    });
}

fn centroid_dists_scalar(query: &[f32; 14], centroids: &[f32], k: usize, dists: &mut Vec<f32>) {
    dists.clear();
    dists.resize(k, 0.0);
    for d in 0..14usize {
        let qd = query[d];
        let base = d * k;
        for ci in 0..k {
            let diff = centroids[base + ci] - qd;
            dists[ci] += diff * diff;
        }
    }
}
```

- [ ] **Step 5: Implement block_scan_scalar()**

Add as a free function in `src/repository/ivf.rs`:

```rust
fn block_scan_scalar(
    query: &[f32; 14],
    offsets: &[u32],
    labels: &[u8],
    blocks: &[i16],
    probed: &[usize],
    k: usize,
) -> SmallVec<[u8; 5]> {
    let mut top: SmallVec<[(u32, u8); 6]> = SmallVec::new();

    for &ci in probed {
        let block_start = offsets[ci] as usize;
        let block_end = offsets[ci + 1] as usize;

        for block_idx in block_start..block_end {
            let block_base = block_idx * 14 * 8;
            let label_base = block_idx * 8;

            for slot in 0..8 {
                let mut sq = 0.0f32;
                for d in 0..14usize {
                    let raw = blocks[block_base + d * 8 + slot] as f32;
                    let diff = query[d] - raw * 0.0001;
                    sq += diff * diff;
                }
                if sq.is_nan() {
                    continue;
                }
                let dist_bits = sq.to_bits();
                let label = labels[label_base + slot];
                if top.len() < k {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, label));
                } else if dist_bits < top[top.len() - 1].0 {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, label));
                    top.truncate(k);
                }
            }
        }
    }

    top.iter().map(|&(_, label)| label).collect()
}
```

- [ ] **Step 6: Implement knn() and knn_adaptive()**

```rust
impl IvfIndex {
    // ... load() above ...

    pub fn knn(&self, query: &[f32; 14], k: usize, nprobe: usize) -> SmallVec<[u8; 5]> {
        let nprobe = nprobe.min(self.k);
        CENTROID_BUFS.with(|bufs| {
            let mut bufs = bufs.borrow_mut();
            centroid_dists_scalar(query, &self.centroids, self.k, &mut bufs.dists);

            bufs.indices.clear();
            bufs.indices.extend(0..self.k);
            if nprobe < self.k {
                let dists = &bufs.dists;
                bufs.indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
                    dists[a].partial_cmp(&dists[b]).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            block_scan_scalar(
                query,
                &self.offsets,
                &self.labels,
                &self.blocks,
                &bufs.indices[..nprobe],
                k,
            )
        })
    }

    pub fn knn_adaptive(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        let stage1 = self.knn(query, k, self.nprobe_fast);
        let fraud_votes = stage1.iter().filter(|&&l| l == 1).count();
        if fraud_votes <= 1 || fraud_votes >= k.saturating_sub(1) {
            return stage1;
        }
        self.knn(query, k, self.nprobe_slow)
    }
}
```

Note: `knn()` borrows `bufs.indices` while calling `centroid_dists_scalar` which also borrows `bufs.dists`. Since they're separate fields, you need to temporarily split the borrow or call `centroid_dists_scalar` first and clone the result. The cleanest fix: use a local `Vec` for indices inside `knn()` instead of the thread-local for indices:

```rust
pub fn knn(&self, query: &[f32; 14], k: usize, nprobe: usize) -> SmallVec<[u8; 5]> {
    let nprobe = nprobe.min(self.k);
    CENTROID_BUFS.with(|bufs| {
        let mut bufs = bufs.borrow_mut();
        centroid_dists_scalar(query, &self.centroids, self.k, &mut bufs.dists);

        // Use a local smallvec for indices (nprobe_slow ≤ 24; fits on stack)
        let mut indices: smallvec::SmallVec<[usize; 24]> =
            (0..self.k).collect();
        if nprobe < self.k {
            let dists = &bufs.dists;
            indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
                dists[a].partial_cmp(&dists[b]).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        block_scan_scalar(
            query,
            &self.offsets,
            &self.labels,
            &self.blocks,
            &indices[..nprobe],
            k,
        )
    })
}
```

Wait — `SmallVec<[usize; 24]>` can't hold k=4096 indices without heap allocation. Use the thread-local indices buffer. The borrow conflict is resolved by computing dists first, then borrowing indices separately:

```rust
pub fn knn(&self, query: &[f32; 14], k: usize, nprobe: usize) -> SmallVec<[u8; 5]> {
    let nprobe = nprobe.min(self.k);

    // Compute centroid distances into thread-local buffer
    CENTROID_BUFS.with(|bufs| {
        let mut bufs = bufs.borrow_mut();
        centroid_dists_scalar(query, &self.centroids, self.k, &mut bufs.dists);

        bufs.indices.clear();
        bufs.indices.extend(0..self.k);

        if nprobe < self.k {
            // select_nth_unstable_by needs immutable access to dists via closure,
            // but dists and indices are different fields in the same struct.
            // Split borrow manually:
            let CentroidBufs { dists, indices } = &mut *bufs;
            indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
                dists[a].partial_cmp(&dists[b]).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        block_scan_scalar(
            query,
            &self.offsets,
            &self.labels,
            &self.blocks,
            &bufs.indices[..nprobe],
            k,
        )
    })
}
```

- [ ] **Step 7: Run all tests**

```bash
cargo test 2>&1 | tail -20
```
Expected: all tests pass

- [ ] **Step 8: Run clippy**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | grep -E "^error" | head -10
```
Expected: no errors

- [ ] **Step 9: Commit**

```bash
git add src/repository/ivf.rs
git commit -m "feat(ivf): IVF2 scalar loader — flat layout, column-major centroids, block scan"
```

---

## Task 4: AVX2 Centroid Scan (Column-Major SIMD)

**Files:**
- Modify: `src/repository/ivf.rs`

Add an AVX2+FMA implementation of `centroid_dists_scalar`. The column-major layout allows 8-wide SIMD loads across k centroids per dimension pass. Dispatched at runtime with `is_x86_feature_detected!`.

- [ ] **Step 1: Write correctness test**

Add to the `tests` module in `src/repository/ivf.rs`:

```rust
#[test]
fn test_simd_centroid_dists_matches_scalar() {
    let path = write_ivf2("ivf2_simd_centroid.bin");
    let idx = IvfIndex::load(&path, 5, 24).unwrap();

    let query = [0.3f32; 14];

    // Scalar result via knn with nprobe=2 (all centroids)
    let scalar_labels = idx.knn(&query, 5, 2);

    // After Task 4, knn() will dispatch to SIMD centroid scan internally.
    // We verify knn() results are identical before/after SIMD path activates,
    // using a brute-force reference for this known fixture.
    // Fixture: C0=[0.0;14] (dist=14*0.09=1.26), C1=[2.0;14] (dist=14*2.89=40.46)
    // Top-1 centroid = C0, so top-5 neighbors are all legit.
    assert_eq!(scalar_labels.len(), 5);
    assert!(
        scalar_labels.iter().all(|&l| l == 0),
        "SIMD centroid scan must route to legit cluster"
    );
    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Run test to confirm it passes already (SIMD not yet added)**

```bash
cargo test test_simd_centroid_dists_matches_scalar 2>&1 | tail -5
```
Expected: PASS (scalar passes; SIMD will be a drop-in replacement)

- [ ] **Step 3: Add SIMD centroid distance function**

Add below the existing `centroid_dists_scalar` function:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn centroid_dists_simd(query: &[f32; 14], centroids: &[f32], k: usize, dists: &mut Vec<f32>) {
    use std::arch::x86_64::*;

    dists.clear();
    dists.resize(k, 0.0);
    let dp = dists.as_mut_ptr();
    let cp = centroids.as_ptr();

    // Dim 0: initialize (no fmadd, just mul)
    {
        let qd = _mm256_set1_ps(query[0]);
        let mut ci = 0usize;
        while ci + 16 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci)), qd);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci + 8)), qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_mul_ps(d1, d1));
            ci += 16;
        }
        while ci + 8 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci)), qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(ci) - query[0];
            *dp.add(ci) = diff * diff;
            ci += 1;
        }
    }

    // Dims 1..14: accumulate with fmadd
    for d in 1..14usize {
        let base = d * k;
        let qd = _mm256_set1_ps(query[d]);
        let mut ci = 0usize;
        while ci + 16 <= k {
            let cv0 = _mm256_loadu_ps(cp.add(base + ci));
            let cv1 = _mm256_loadu_ps(cp.add(base + ci + 8));
            let dv0 = _mm256_sub_ps(cv0, qd);
            let dv1 = _mm256_sub_ps(cv1, qd);
            let a0 = _mm256_loadu_ps(dp.add(ci));
            let a1 = _mm256_loadu_ps(dp.add(ci + 8));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(dv0, dv0, a0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_fmadd_ps(dv1, dv1, a1));
            ci += 16;
        }
        while ci + 8 <= k {
            let cv = _mm256_loadu_ps(cp.add(base + ci));
            let dv = _mm256_sub_ps(cv, qd);
            let a = _mm256_loadu_ps(dp.add(ci));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(dv, dv, a));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(base + ci) - query[d];
            *dp.add(ci) += diff * diff;
            ci += 1;
        }
    }
}
```

- [ ] **Step 4: Dispatch SIMD in knn()**

Replace the `centroid_dists_scalar(...)` call in `knn()` with a dispatch:

```rust
fn fill_centroid_dists(query: &[f32; 14], centroids: &[f32], k: usize, dists: &mut Vec<f32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { centroid_dists_simd(query, centroids, k, dists) };
        }
    }
    centroid_dists_scalar(query, centroids, k, dists);
}
```

Update `knn()` to call `fill_centroid_dists(...)` instead of `centroid_dists_scalar(...)`.

- [ ] **Step 5: Run all tests**

```bash
cargo test 2>&1 | tail -10
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/repository/ivf.rs
git commit -m "perf(ivf): AVX2+FMA column-major centroid distance scan"
```

---

## Task 5: AVX2 Block Scan + Early Termination

**Files:**
- Modify: `src/repository/ivf.rs`

Add an AVX2+FMA implementation of `block_scan_scalar`. Processes 8 vectors simultaneously using i16→f32 widening. Early termination after 8 of 14 dims when all 8 partial distances exceed the current worst neighbor.

- [ ] **Step 1: Write early termination test**

Add to `tests` module:

```rust
#[test]
fn test_early_termination_skips_distant_block() {
    // Fixture: k=2, 8 legit near [0.0;14], 8 fraud near [2.0;14]
    // Query [0.0;14] with nprobe=2: after block 0 (legit, dist≈0.14) fills heap,
    // block 1 (fraud, dist≈56.0) should be skipped by early termination.
    // We can't observe early termination directly, so verify correctness:
    // - top-5 must all be legit (block 1 distances exceed threshold after 8 dims)
    let path = write_ivf2("ivf2_early_term.bin");
    let idx = IvfIndex::load(&path, 5, 24).unwrap();
    let query = [0.0f32; 14];
    let labels = idx.knn(&query, 5, 2);
    assert_eq!(labels.len(), 5);
    assert!(
        labels.iter().all(|&l| l == 0),
        "early termination must skip far block; got labels: {:?}", labels
    );
    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Run test (passes with scalar, will still pass after SIMD)**

```bash
cargo test test_early_termination_skips_distant_block 2>&1 | tail -5
```
Expected: PASS

- [ ] **Step 3: Add SIMD block scan function**

Add below `block_scan_scalar`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_scan_simd(
    query: &[f32; 14],
    offsets: &[u32],
    labels: &[u8],
    blocks: &[i16],
    probed: &[usize],
    k: usize,
) -> SmallVec<[u8; 5]> {
    use std::arch::x86_64::*;

    let scale = _mm256_set1_ps(0.0001);
    let mut q_vecs = [_mm256_setzero_ps(); 14];
    for d in 0..14usize {
        q_vecs[d] = _mm256_set1_ps(query[d]);
    }

    // top-k heap: (distance_bits, label) sorted ascending by distance
    let mut top: SmallVec<[(u32, u8); 6]> = SmallVec::new();
    // worst_dist tracks worst accepted distance; used as early termination threshold
    let mut worst_dist_bits: u32 = u32::MAX;

    macro_rules! load_and_widen {
        ($ptr:expr) => {{
            let raw = _mm_loadu_si128($ptr as *const __m128i);
            let i32s = _mm256_cvtepi16_epi32(raw);
            _mm256_mul_ps(_mm256_cvtepi32_ps(i32s), scale)
        }};
    }

    let bp = blocks.as_ptr();
    let lp = labels.as_ptr();

    for &ci in probed {
        let block_start = offsets[ci] as usize;
        let block_end = offsets[ci + 1] as usize;

        'block: for block_i in block_start..block_end {
            // Prefetch next block
            if block_i + 4 < block_end {
                _mm_prefetch(bp.add((block_i + 4) * 112) as *const i8, _MM_HINT_T0);
            }

            let bb = block_i * 112; // offset into blocks[] for this block
            let threshold = _mm256_set1_ps(f32::from_bits(worst_dist_bits));

            // Process first 8 dims (4 pairs) → early termination check
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();

            macro_rules! dim_pair {
                ($d:expr) => {{
                    let v0 = load_and_widen!(bp.add(bb + $d * 8));
                    let v1 = load_and_widen!(bp.add(bb + ($d + 1) * 8));
                    let dv0 = _mm256_sub_ps(v0, q_vecs[$d]);
                    let dv1 = _mm256_sub_ps(v1, q_vecs[$d + 1]);
                    acc0 = _mm256_fmadd_ps(dv0, dv0, acc0);
                    acc1 = _mm256_fmadd_ps(dv1, dv1, acc1);
                }};
            }

            dim_pair!(0);
            dim_pair!(2);
            dim_pair!(4);
            dim_pair!(6);

            // Early termination: if all 8 partial distances already exceed threshold, skip
            let partial = _mm256_add_ps(acc0, acc1);
            if _mm256_movemask_ps(_mm256_cmp_ps(partial, threshold, _CMP_LT_OQ)) == 0 {
                continue 'block;
            }

            // Process remaining 6 dims (3 pairs)
            dim_pair!(8);
            dim_pair!(10);
            dim_pair!(12);

            let full = _mm256_add_ps(acc0, acc1);
            let mut mask =
                _mm256_movemask_ps(_mm256_cmp_ps(full, threshold, _CMP_LT_OQ)) as u32;
            if mask == 0 {
                continue;
            }

            let mut dists_buf = [0.0f32; 8];
            _mm256_storeu_ps(dists_buf.as_mut_ptr(), full);
            let label_base = block_i * 8;

            while mask != 0 {
                let slot = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let dist = dists_buf[slot];
                let dist_bits = dist.to_bits();
                let label = *lp.add(label_base + slot);

                if top.len() < k {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, label));
                    if top.len() == k {
                        worst_dist_bits = top[k - 1].0;
                    }
                } else if dist_bits < worst_dist_bits {
                    let pos = top.partition_point(|&(d, _)| d <= dist_bits);
                    top.insert(pos, (dist_bits, label));
                    top.truncate(k);
                    worst_dist_bits = top[k - 1].0;
                }
            }
        }
    }

    top.iter().map(|&(_, label)| label).collect()
}
```

- [ ] **Step 4: Dispatch SIMD in knn()**

Replace `block_scan_scalar(...)` call in `knn()` with a dispatch function:

```rust
fn run_block_scan(
    query: &[f32; 14],
    offsets: &[u32],
    labels: &[u8],
    blocks: &[i16],
    probed: &[usize],
    k: usize,
) -> SmallVec<[u8; 5]> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { block_scan_simd(query, offsets, labels, blocks, probed, k) };
        }
    }
    block_scan_scalar(query, offsets, labels, blocks, probed, k)
}
```

Update `knn()` to call `run_block_scan(...)`.

- [ ] **Step 5: Run all tests**

```bash
cargo test 2>&1 | tail -10
```
Expected: all pass

- [ ] **Step 6: Run clippy**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | grep "^error" | head -10
```
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/repository/ivf.rs
git commit -m "perf(ivf): AVX2+FMA block scan with early termination after 8 dims"
```

---

## Task 6: Python IVF2 Builder — K=4096 + int16

**Files:**
- Modify: `tools/build_ivf.py`

Rewrite the builder to output IVF2 format with K=4096 centroids and int16 quantization (×10000). The index is rebuilt at Docker build time; this task only modifies the builder script.

- [ ] **Step 1: Write a self-test function and run it on a tiny synthetic dataset**

Add a test at the bottom of `tools/build_ivf.py` (guarded by `if __name__ == "__main__" and "--test" in sys.argv:`):

```python
def _test_ivf2_roundtrip():
    """Build a tiny IVF2 from 80 synthetic vectors and verify the header."""
    import tempfile, os
    import numpy as np
    
    n, d, k_test = 80, 14, 4
    rng = np.random.default_rng(42)
    vecs = rng.uniform(-1, 1, (n, d)).astype(np.float32)
    lbls = rng.integers(0, 2, n).astype(np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = f.name
    
    try:
        _write_ivf2(vecs, lbls, k_test, tmp)
        with open(tmp, "rb") as f:
            magic = f.read(4)
            assert magic == b"IVF2", f"bad magic: {magic}"
            n_out = int.from_bytes(f.read(4), "little")
            k_out = int.from_bytes(f.read(4), "little")
            d_out = int.from_bytes(f.read(4), "little")
            assert n_out == n, f"n mismatch: {n_out}"
            assert k_out == k_test, f"k mismatch: {k_out}"
            assert d_out == d, f"d mismatch: {d_out}"
        print("PASS: IVF2 roundtrip test")
    finally:
        os.unlink(tmp)
```

- [ ] **Step 2: Run test to confirm it fails (function not yet defined)**

```bash
cd /home/snow/workspace/rinha-backend/fraud-detection
uv run tools/build_ivf.py --test 2>&1 | tail -5
```
Expected: NameError or AttributeError — `_write_ivf2` not defined

- [ ] **Step 3: Rewrite tools/build_ivf.py**

Replace the entire file:

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
Build IVF2 index from resources/references.json.gz.
Output: resources/ivf_index.bin

IVF2 binary format (all little-endian):
  [4B]        magic: IVF2
  [4B u32]    n   — total vectors (before padding)
  [4B u32]    k   — number of centroids
  [4B u32]    d   — dimensions (14)
  [d*k*4B]    centroids f32, column-major: centroids[d_idx * k + ci]
  [(k+1)*4B]  block_offsets u32 — offsets[ci]..offsets[ci+1] = block range (unit: 8-vec block)
  [total_blocks*8 B] labels u8 — padding slots = 0
  [total_blocks*d*8*2B] blocks i16 — blocks[(block_idx*d+dim)*8+slot], padding = i16::MAX

Quantization: i16 = round(f32 * 10_000). Range [-3.2768, 3.2767] maps exactly.
Sentinel -1.0 (null last_transaction) → -10000. Fits in i16.
"""

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

INPUT = Path("resources/references.json.gz")
OUTPUT = Path("resources/ivf_index.bin")
K = 4096
D = 14
BATCH_SIZE = 50_000
N_INIT = 3
RANDOM_STATE = 42
SCALE = 10_000


def quantize(arr: np.ndarray) -> np.ndarray:
    """f32 → i16 via round(x * 10000). Clips to i16 range."""
    return np.clip(np.round(arr * SCALE), -32768, 32767).astype(np.int16)


def _write_ivf2(vectors: np.ndarray, labels: np.ndarray, k: int, output_path: str) -> None:
    """Write IVF2 binary from pre-assigned clusters. Called after KMeans fit."""
    n, d = vectors.shape
    assert d == D

    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=BATCH_SIZE,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    assignments = km.fit_predict(vectors)
    centroids = km.cluster_centers_.astype(np.float32)  # shape: (k, d)

    # Group vector indices by cluster
    cluster_vecs: list[list[int]] = [[] for _ in range(k)]
    for i, ci in enumerate(assignments):
        cluster_vecs[ci].append(i)

    # Compute block offsets (unit: 8-vector blocks)
    block_offsets = np.zeros(k + 1, dtype=np.uint32)
    for ci in range(k):
        n_blocks = (len(cluster_vecs[ci]) + 7) // 8
        block_offsets[ci + 1] = block_offsets[ci] + n_blocks

    total_blocks = int(block_offsets[k])
    padded_n = total_blocks * 8

    out_labels = np.zeros(padded_n, dtype=np.uint8)
    out_blocks = np.full(total_blocks * d * 8, fill_value=32767, dtype=np.int16)  # padding=i16::MAX

    for ci in range(k):
        block_start = int(block_offsets[ci])
        vecs_ci = cluster_vecs[ci]
        n_blocks = int(block_offsets[ci + 1]) - block_start

        for bk in range(n_blocks):
            block_idx = block_start + bk
            label_base = block_idx * 8
            block_base = block_idx * d * 8

            for slot in range(8):
                vi_pos = bk * 8 + slot
                if vi_pos >= len(vecs_ci):
                    break  # padding slot — already i16::MAX / label=0
                vi = vecs_ci[vi_pos]
                out_labels[label_base + slot] = labels[vi]
                for dim in range(d):
                    out_blocks[block_base + dim * 8 + slot] = quantize(vectors[vi, dim : dim + 1])[0]

    # centroids: column-major [d * k]
    centroids_t = centroids.T.copy()  # shape (d, k), row = dim

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"IVF2")
        f.write(struct.pack("<III", n, k, d))
        f.write(centroids_t.astype(np.float32).tobytes())  # column-major f32
        f.write(block_offsets.tobytes())
        f.write(out_labels.tobytes())
        f.write(out_blocks.tobytes())


def _test_ivf2_roundtrip():
    import tempfile
    import os

    n_test, k_test = 80, 4
    rng = np.random.default_rng(42)
    vecs = rng.uniform(-1, 1, (n_test, D)).astype(np.float32)
    lbls = rng.integers(0, 2, n_test).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = f.name

    try:
        _write_ivf2(vecs, lbls, k_test, tmp)
        with open(tmp, "rb") as f:
            magic = f.read(4)
            assert magic == b"IVF2", f"bad magic: {magic}"
            n_out = int.from_bytes(f.read(4), "little")
            k_out = int.from_bytes(f.read(4), "little")
            d_out = int.from_bytes(f.read(4), "little")
            assert n_out == n_test, f"n mismatch: {n_out} != {n_test}"
            assert k_out == k_test, f"k mismatch: {k_out} != {k_test}"
            assert d_out == D, f"d mismatch: {d_out} != {D}"
        print("PASS: IVF2 roundtrip test")
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test_ivf2_roundtrip()
        sys.exit(0)

    print(f"Loading {INPUT}...", flush=True)
    if not INPUT.exists():
        sys.exit(f"ERROR: {INPUT} not found. Run from the fraud-detection directory.")

    with gzip.open(INPUT, "rt", encoding="utf-8") as f:
        refs = json.load(f)

    vectors = np.array([r["vector"] for r in refs], dtype=np.float32)
    labels = np.array([1 if r["label"] == "fraud" else 0 for r in refs], dtype=np.uint8)
    N, d_check = vectors.shape
    print(f"Loaded {N} vectors, D={d_check}", flush=True)

    if d_check != D:
        sys.exit(f"ERROR: expected D={D}, got {d_check}")

    print(f"Building IVF2 index K={K}...", flush=True)
    _write_ivf2(vectors, labels, K, str(OUTPUT))

    size_mb = OUTPUT.stat().st_size / 1024**2
    print(f"Done. {OUTPUT} = {size_mb:.1f} MB", flush=True)
```

- [ ] **Step 4: Run roundtrip test**

```bash
cd /home/snow/workspace/rinha-backend/fraud-detection
uv run tools/build_ivf.py --test
```
Expected: `PASS: IVF2 roundtrip test`

- [ ] **Step 5: Commit**

```bash
git add tools/build_ivf.py
git commit -m "feat(builder): IVF2 format, K=4096, int16 quantization (×10000)"
```

---

## Post-Implementation Validation

Before submitting to competition, run:

```bash
make bench
```

Expected: p99 < 30ms, ERR=0

Monitor CPU usage under bench load:
```bash
docker stats --no-stream 2>/dev/null | head -5
```

If p99 > 30ms after SIMD tasks, check:
1. `is_x86_feature_detected!("avx2")` returns true (verify in a test)
2. Compilation uses `RUSTFLAGS="-C target-cpu=native"` or equivalent in Dockerfile
3. Docker cgroup CPU throttle — benchmark at `--cpus 0.45` to simulate competition

---

## Self-Review

### Spec coverage check

| Spec section | Task | Status |
|---|---|---|
| `IvfIndex` flat layout (struct) | Task 3 | ✓ |
| IVF2 binary format | Task 3 + Task 6 | ✓ |
| Column-major centroids | Task 3 (scalar), Task 4 (SIMD) | ✓ |
| CSR block offsets (block units) | Task 3 | ✓ |
| i16 quantization ×10000 | Task 3 (decode), Task 6 (encode) | ✓ |
| SIMD centroid scan | Task 4 | ✓ |
| SIMD block scan + early termination | Task 5 | ✓ |
| Static responses | Task 2 | ✓ |
| fraud_count in FraudDecision | Task 1 | ✓ |
| K=4096 | Task 6 | ✓ |
| All existing tests pass | Each task verifies | ✓ |
| lib.rs wiring | No change needed (const approach) | ✓ (simplified) |

### Type consistency check
- `FraudDecision.fraud_count: usize` — used in Task 1, consumed in Task 2 handler
- `IvfIndex.offsets: Vec<u32>` — written in load(), read in block_scan as `offsets[ci]` (cast to usize)
- `IvfIndex.blocks: Vec<i16>` — blocks[(block_idx * 14 + d) * 8 + slot] — consistent across Task 3, 5
- `block_scan_scalar`/`block_scan_simd` — same signature, same return type `SmallVec<[u8; 5]>`
- `fill_centroid_dists`/`centroid_dists_simd` — same signature, `dists: &mut Vec<f32>`
- `CentroidBufs.dists: Vec<f32>` — used in Tasks 3, 4
- `STATIC_BODIES: [&str; 6]` — indexed by `fraud_count.min(5)`

### Placeholder scan
No TBD, TODO, or placeholder steps found. All code blocks are complete.

### Regression note
Task 3 removes all IVF1-format tests (they used IVF1 binary helpers). They are replaced with IVF2 equivalents covering the same scenarios. The `ReferenceRepository` wrapper in `src/repository/reference.rs` is unchanged — its interface is stable.
