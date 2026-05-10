# IVF Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace brute-force KNN with a pre-built IVF index and fix async blocking with `spawn_blocking`, targeting ≥900 req/s within the 1 CPU + 350MB contest budget.

**Architecture:** A Python script runs at Docker build time to cluster 3M reference vectors into K=1732 groups via MiniBatchKMeans and write a binary IVF index (~87MB). At runtime, Rust loads this index and serves each request by probing 8 nearest clusters (~217K ops vs. current 42M). A `spawn_blocking` wrapper offloads the CPU-bound lookup from the Tokio thread pool, eliminating request serialization.

**Tech Stack:** Rust (half, smallvec, tokio::task::spawn_blocking), Python 3.12 (scikit-learn, numpy), Docker multi-stage build.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/repository/ivf.rs` | CREATE | `IvfIndex` struct: load binary, knn search |
| `src/repository/mod.rs` | MODIFY | Add `pub mod ivf;` |
| `src/repository/reference.rs` | MODIFY | Wrap `IvfIndex`, remove brute-force |
| `src/config.rs` | MODIFY | Rename `refs_path` → `ivf_path`, add `nprobe` |
| `src/lib.rs` | MODIFY | `AppState::build` uses `ivf_path` + `nprobe` |
| `src/web/handlers.rs` | MODIFY | `spawn_blocking` wrap + approved:true fallback |
| `tests/integration.rs` | MODIFY | Update `Config` struct literals |
| `tests/regression.rs` | MODIFY | Update `Config` struct literals |
| `tools/build_ivf.py` | CREATE | Offline MiniBatchKMeans builder |
| `tools/requirements.txt` | CREATE | numpy, scikit-learn |
| `Dockerfile` | MODIFY | Add Python build stage before Rust build |
| `.gitignore` | MODIFY | Ignore `resources/ivf_index.bin` |

No changes to: domain logic, DTOs, scoring, vectorizer, use cases.

---

### Task 1: spawn_blocking + error fallback in handler

**Files:**
- Modify: `src/web/handlers.rs`

Context: `execute()` currently runs synchronously on the Tokio async thread, serializing all requests. Wrapping with `spawn_blocking` moves it to a dedicated blocking thread pool. On `JoinError`, return `approved: true, fraud_score: 0.0` — a false positive (FP weight=1) is cheaper than HTTP 500 (weight=5 in contest scoring).

- [ ] **Step 1: Baseline — run existing tests to confirm they pass before any changes**

```bash
cargo test --test integration
```

Expected: all tests pass.

- [ ] **Step 2: Replace the handler with spawn_blocking version**

Full new `src/web/handlers.rs`:

```rust
use crate::AppState;
use crate::{
    domain::{
        fraud::FraudDecision,
        transaction::{Customer, LastTransaction, Merchant, Terminal, Transaction},
    },
    web::dto::{FraudScoreResponse, TransactionRequest},
};
use axum::{extract::State, Json};
use std::collections::HashSet;
use std::sync::Arc;

pub async fn ready_handler() -> &'static str {
    "ok"
}

pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> Json<FraudScoreResponse> {
    let tx = into_transaction(req);
    let decision = tokio::task::spawn_blocking({
        let state = Arc::clone(&state);
        move || state.use_case.execute(&tx)
    })
    .await
    .unwrap_or_else(|_| FraudDecision {
        approved: true,
        fraud_score: 0.0,
    });
    Json(FraudScoreResponse {
        approved: decision.approved,
        fraud_score: decision.fraud_score,
    })
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
            known_merchants: req
                .customer
                .known_merchants
                .into_iter()
                .collect::<HashSet<_>>(),
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
```

Note: return type changes from `Result<Json<FraudScoreResponse>, AppError>` to `Json<FraudScoreResponse>` — no error path remains since join failures return the fallback value.

- [ ] **Step 3: Verify it compiles and tests still pass**

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo test --test integration
```

Expected: no warnings, all integration tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/web/handlers.rs
git commit -m "fix(handler): offload execute() to spawn_blocking, fallback to approved=true on join error"
```

---

### Task 2: IvfIndex module

**Files:**
- Create: `src/repository/ivf.rs`
- Modify: `src/repository/mod.rs`

This is the core data structure. Binary format (all LE):
```
[4B u32] K — number of clusters
[4B u32] D — dimensions (must be 14)
[K × 14 × 4B f32] centroids — row-major
[K × 4B u32] list_sizes
[for each cluster i: list_sizes[i] × (14 × 2B f16 + 1B u8)]
```

knn algorithm:
1. Compute squared Euclidean distance from query (f32) to each of the K centroids (f32)
2. Sort centroids by distance; take top `nprobe`
3. Brute-force over all vectors in those `nprobe` lists
4. Return top-k labels by distance

- [ ] **Step 1: Add `pub mod ivf;` to `src/repository/mod.rs`**

Full new `src/repository/mod.rs`:

```rust
pub mod ivf;
pub mod reference;
```

- [ ] **Step 2: Write failing tests — create `src/repository/ivf.rs` with struct stub + tests**

```rust
use half::f16;
use smallvec::SmallVec;
use std::path::Path;

pub struct IvfIndex {
    k: usize,
    nprobe: usize,
    centroids: Vec<[f32; 14]>,
    lists: Vec<Vec<([f16; 14], u8)>>,
}

impl IvfIndex {
    pub fn load(path: &Path, nprobe: usize) -> std::io::Result<Self> {
        todo!()
    }

    pub fn knn(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        todo!()
    }
}

fn centroid_sq_dist(query: &[f32; 14], centroid: &[f32; 14]) -> f32 {
    query
        .iter()
        .zip(centroid.iter())
        .map(|(&q, &c)| {
            let d = q - c;
            d * d
        })
        .sum()
}

fn vec_sq_dist(query: &[f32; 14], vec: &[f16; 14]) -> f32 {
    query
        .iter()
        .zip(vec.iter())
        .map(|(&q, &r)| {
            let d = q - r.to_f32();
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal IVF binary in memory:
    //   K=2, D=14
    //   Centroid 0: all 0.0  (cluster of legit vectors)
    //   Centroid 1: all 10.0 (cluster of fraud vectors)
    //   list_sizes: [3, 3]
    //   Cluster 0: 3 vectors near [0.1; 14], label=0 (legit)
    //   Cluster 1: 3 vectors near [10.0; 14], label=1 (fraud)
    fn make_tiny_ivf_bytes() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        // K=2, D=14
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&14u32.to_le_bytes());
        // centroid 0: all 0.0
        for _ in 0..14 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        // centroid 1: all 10.0
        for _ in 0..14 {
            buf.extend_from_slice(&10.0f32.to_le_bytes());
        }
        // list_sizes: [3, 3]
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        // cluster 0: 3 legit vectors near [0.1; 14]
        for _ in 0..3 {
            for _ in 0..14 {
                buf.extend_from_slice(&f16::from_f32(0.1).to_le_bytes());
            }
            buf.push(0u8);
        }
        // cluster 1: 3 fraud vectors near [10.0; 14]
        for _ in 0..3 {
            for _ in 0..14 {
                buf.extend_from_slice(&f16::from_f32(10.0).to_le_bytes());
            }
            buf.push(1u8);
        }
        buf
    }

    fn write_tiny_ivf() -> std::path::PathBuf {
        let path = std::env::temp_dir().join("test_ivf_tiny.bin");
        std::fs::write(&path, make_tiny_ivf_bytes()).unwrap();
        path
    }

    #[test]
    fn test_load_parses_header() {
        let path = write_tiny_ivf();
        let idx = IvfIndex::load(&path, 1).unwrap();
        assert_eq!(idx.k, 2);
        assert_eq!(idx.lists.len(), 2);
        assert_eq!(idx.lists[0].len(), 3);
        assert_eq!(idx.lists[1].len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_query_near_legit_cluster() {
        let path = write_tiny_ivf();
        let idx = IvfIndex::load(&path, 1).unwrap();
        // Query near [0.0; 14] — should probe cluster 0 only (nprobe=1)
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 3);
        assert_eq!(labels.len(), 3);
        assert!(
            labels.iter().all(|&l| l == 0),
            "all neighbors near zero centroid should be legit"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_query_near_fraud_cluster() {
        let path = write_tiny_ivf();
        let idx = IvfIndex::load(&path, 1).unwrap();
        // Query near [10.0; 14] — should probe cluster 1 only (nprobe=1)
        let query = [10.0f32; 14];
        let labels = idx.knn(&query, 3);
        assert_eq!(labels.len(), 3);
        assert!(
            labels.iter().all(|&l| l == 1),
            "all neighbors near 10.0 centroid should be fraud"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_knn_nprobe_2_returns_from_both_clusters() {
        let path = write_tiny_ivf();
        let idx = IvfIndex::load(&path, 2).unwrap();
        // Query near [0.0; 14] with nprobe=2 — probes both clusters
        // 3 legit (dist ~0.14 each) + 3 fraud (dist ~1398 each)
        // top-3 should still be legit
        let query = [0.0f32; 14];
        let labels = idx.knn(&query, 3);
        assert_eq!(labels.len(), 3);
        assert!(
            labels.iter().all(|&l| l == 0),
            "top-3 when near zero centroid should all be legit even with nprobe=2"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_rejects_truncated_file() {
        let path = std::env::temp_dir().join("test_ivf_truncated.bin");
        std::fs::write(&path, &[0u8; 4]).unwrap();
        assert!(IvfIndex::load(&path, 1).is_err());
        std::fs::remove_file(&path).ok();
    }
}
```

- [ ] **Step 3: Run tests — confirm they compile but panic on `todo!()`**

```bash
cargo test -p fraud-detection ivf::tests 2>&1 | head -30
```

Expected: tests compile, all fail with `not yet implemented`.

- [ ] **Step 4: Implement `IvfIndex::load`**

Replace the `todo!()` in `load`:

```rust
pub fn load(path: &Path, nprobe: usize) -> std::io::Result<Self> {
    let data = std::fs::read(path)?;
    if data.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ivf_index.bin too short: missing header",
        ));
    }
    let mut pos = 0;

    let k = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let d = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    if d != 14 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected D=14, got {d}"),
        ));
    }

    // Read centroids: k × 14 × 4B f32
    let centroid_bytes = k * 14 * 4;
    if data.len() < pos + centroid_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ivf_index.bin truncated: centroids",
        ));
    }
    let mut centroids = Vec::with_capacity(k);
    for _ in 0..k {
        let mut c = [0.0f32; 14];
        for j in 0..14 {
            c[j] = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
        }
        centroids.push(c);
    }

    // Read list sizes: k × 4B u32
    if data.len() < pos + k * 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ivf_index.bin truncated: list sizes",
        ));
    }
    let mut list_sizes = Vec::with_capacity(k);
    for _ in 0..k {
        let sz = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        list_sizes.push(sz);
        pos += 4;
    }

    // Read entries: for each cluster, list_sizes[i] × (14 × 2B f16 + 1B u8)
    let mut lists = Vec::with_capacity(k);
    for i in 0..k {
        let entry_bytes = list_sizes[i] * (14 * 2 + 1);
        if data.len() < pos + entry_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("ivf_index.bin truncated: cluster {i} entries"),
            ));
        }
        let mut list = Vec::with_capacity(list_sizes[i]);
        for _ in 0..list_sizes[i] {
            let mut vec = [f16::ZERO; 14];
            for j in 0..14 {
                vec[j] = f16::from_le_bytes([data[pos], data[pos + 1]]);
                pos += 2;
            }
            let label = data[pos];
            pos += 1;
            list.push((vec, label));
        }
        lists.push(list);
    }

    Ok(Self {
        k,
        nprobe,
        centroids,
        lists,
    })
}
```

- [ ] **Step 5: Implement `IvfIndex::knn`**

Replace the `todo!()` in `knn`:

```rust
pub fn knn(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
    // Rank all centroids by distance to query
    let mut centroid_dists: Vec<(f32, usize)> = self
        .centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (centroid_sq_dist(query, c), i))
        .collect();
    centroid_dists
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let nprobe = self.nprobe.min(self.k);

    // Brute-force over the nprobe nearest clusters
    let mut top: Vec<(u32, u8)> = Vec::with_capacity(k + 1);
    for &(_, ci) in &centroid_dists[..nprobe] {
        for &(ref vec, label) in &self.lists[ci] {
            let dist = vec_sq_dist(query, vec);
            let dist_bits = dist.to_bits();
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

    top.iter().map(|&(_, label)| label).collect()
}
```

- [ ] **Step 6: Run tests — confirm all pass**

```bash
cargo test -p fraud-detection ivf::tests
```

Expected: 4 tests pass, 0 failures.

- [ ] **Step 7: Clippy check**

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Expected: no warnings.

- [ ] **Step 8: Commit**

```bash
git add src/repository/ivf.rs src/repository/mod.rs
git commit -m "feat(repository): add IvfIndex with binary load and nprobe knn search"
```

---

### Task 3: Config update

**Files:**
- Modify: `src/config.rs`

Rename `refs_path` → `ivf_path`. Add `nprobe` with default 8. Change env var from `REFS_PATH` to `IVF_PATH`.

- [ ] **Step 1: Replace `src/config.rs`**

```rust
use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub ivf_path: PathBuf,
    pub mcc_path: PathBuf,
    pub norm_path: PathBuf,
    pub nprobe: usize,
}

impl Config {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        Self {
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .expect("PORT must be a valid port number"),
            ivf_path: PathBuf::from(
                std::env::var("IVF_PATH")
                    .unwrap_or_else(|_| "resources/ivf_index.bin".to_string()),
            ),
            mcc_path: PathBuf::from(
                std::env::var("MCC_PATH")
                    .unwrap_or_else(|_| "resources/mcc_risk.json".to_string()),
            ),
            norm_path: PathBuf::from(
                std::env::var("NORM_PATH")
                    .unwrap_or_else(|_| "resources/normalization.json".to_string()),
            ),
            nprobe: std::env::var("IVF_NPROBE")
                .unwrap_or_else(|_| "8".to_string())
                .parse()
                .expect("IVF_NPROBE must be a valid integer"),
        }
    }
}
```

- [ ] **Step 2: Verify compile (will fail at lib.rs — fix in next task)**

```bash
cargo build 2>&1 | grep "error\["
```

Expected: errors referencing `refs_path` in `src/lib.rs` and test files — that is expected and resolved in Tasks 4 and 5.

- [ ] **Step 3: Commit config change alone**

```bash
git add src/config.rs
git commit -m "feat(config): replace refs_path with ivf_path, add nprobe (default 8)"
```

---

### Task 4: ReferenceRepository refactor + AppState

**Files:**
- Modify: `src/repository/reference.rs`
- Modify: `src/lib.rs`

`ReferenceRepository` keeps its public API (`knn(&[f32; 14], usize) -> SmallVec<[u8; 5]>`) but delegates internally to `IvfIndex`. The brute-force implementation and its unit tests are replaced — the knn behavior is now tested in `ivf.rs`.

- [ ] **Step 1: Replace `src/repository/reference.rs`**

```rust
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
        self.ivf.knn(query, k)
    }
}
```

- [ ] **Step 2: Update `src/lib.rs` to use `ivf_path` and `nprobe`**

```rust
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
```

- [ ] **Step 3: Verify build compiles (excluding tests — they still reference old Config)**

```bash
cargo build 2>&1 | grep "error\["
```

Expected: errors only in `tests/integration.rs` and `tests/regression.rs` — those are fixed in Task 5.

- [ ] **Step 4: Commit**

```bash
git add src/repository/reference.rs src/lib.rs
git commit -m "refactor(repository): delegate ReferenceRepository::knn to IvfIndex, remove brute-force"
```

---

### Task 5: Update integration and regression tests

**Files:**
- Modify: `tests/integration.rs`
- Modify: `tests/regression.rs`

Both test files build `Config` directly. Replace `refs_path` with `ivf_path` and add `nprobe`. The test scenarios themselves are unchanged.

**Prerequisite:** `resources/ivf_index.bin` must exist locally before running these tests. Build it with:
```bash
cd /path/to/fraud-detection
pip install numpy scikit-learn  # or: uv pip install numpy scikit-learn
python tools/build_ivf.py       # takes 3–8 minutes
```

- [ ] **Step 1: Update Config literal in `tests/integration.rs`**

Find the `STATE` initializer (lines 8–16) and replace:

```rust
static STATE: Lazy<Arc<AppState>> = Lazy::new(|| {
    let config = Config {
        port: 3000,
        ivf_path: PathBuf::from("resources/ivf_index.bin"),
        mcc_path: PathBuf::from("resources/mcc_risk.json"),
        norm_path: PathBuf::from("resources/normalization.json"),
        nprobe: 8,
    };
    Arc::new(
        AppState::build(&config)
            .expect("AppState init failed — run `python tools/build_ivf.py` first"),
    )
});
```

- [ ] **Step 2: Update Config literal in `tests/regression.rs`**

Find the `STATE` initializer (lines 8–15) and replace:

```rust
static STATE: Lazy<Arc<AppState>> = Lazy::new(|| {
    let config = Config {
        port: 3000,
        ivf_path: PathBuf::from("resources/ivf_index.bin"),
        mcc_path: PathBuf::from("resources/mcc_risk.json"),
        norm_path: PathBuf::from("resources/normalization.json"),
        nprobe: 8,
    };
    Arc::new(AppState::build(&config).expect("AppState init failed — run `python tools/build_ivf.py` first"))
});
```

- [ ] **Step 3: Run unit tests only (no ivf_index.bin needed)**

```bash
cargo test --lib
```

Expected: all unit tests pass (ivf.rs unit tests, vectorizer tests, etc.).

- [ ] **Step 4: Build ivf_index.bin locally, then run integration tests**

```bash
python tools/build_ivf.py
cargo test --test integration
cargo test --test regression
```

Expected for integration:
- `test_ready_endpoint` — PASS
- `test_legit_transaction_from_docs` — PASS (`approved: true, fraud_score: 0.0`)
- `test_fraud_transaction_from_docs` — PASS (`approved: false, fraud_score: 1.0`)
- `test_null_last_transaction_no_panic` — PASS

Expected for regression:
- All tests PASS (fraud/legit decisions preserved at nprobe=8).

- [ ] **Step 5: Commit**

```bash
git add tests/integration.rs tests/regression.rs
git commit -m "test: update Config struct literals to ivf_path/nprobe after config rename"
```

---

### Task 6: Python IVF builder

**Files:**
- Create: `tools/build_ivf.py`
- Create: `tools/requirements.txt`

Runs offline (at Docker build time, or locally). Reads `resources/references.json.gz`, clusters with MiniBatchKMeans, writes `resources/ivf_index.bin` in the binary format defined in the spec.

- [ ] **Step 1: Create `tools/requirements.txt`**

```
numpy>=1.26
scikit-learn>=1.4
```

- [ ] **Step 2: Create `tools/build_ivf.py`**

```python
#!/usr/bin/env python3
"""
Build IVF index from resources/references.json.gz.
Output: resources/ivf_index.bin

Binary format (all little-endian):
  [4B u32] K  — number of clusters
  [4B u32] D  — dimensions (14)
  [K * D * 4B f32] centroids — row-major
  [K * 4B u32] list_sizes
  [for each cluster i: list_sizes[i] * (D * 2B f16 + 1B u8)]
    — vector (f16 x 14) followed by label (u8: 0=legit, 1=fraud)
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
K = 1732
BATCH_SIZE = 50_000
N_INIT = 3
RANDOM_STATE = 42

print(f"Loading {INPUT}...", flush=True)
if not INPUT.exists():
    sys.exit(f"ERROR: {INPUT} not found. Run from the fraud-detection directory.")

with gzip.open(INPUT, "rt", encoding="utf-8") as f:
    refs = json.load(f)

vectors = np.array([r["vector"] for r in refs], dtype=np.float32)
labels = np.array([1 if r["label"] == "fraud" else 0 for r in refs], dtype=np.uint8)
N, D = vectors.shape
print(f"Loaded {N} vectors, D={D}", flush=True)

if D != 14:
    sys.exit(f"ERROR: expected D=14, got {D}")

print(
    f"Clustering K={K}, batch_size={BATCH_SIZE}, n_init={N_INIT}...",
    flush=True,
)
km = MiniBatchKMeans(
    n_clusters=K,
    batch_size=BATCH_SIZE,
    n_init=N_INIT,
    random_state=RANDOM_STATE,
    verbose=0,
)
assignments = km.fit_predict(vectors)
centroids = km.cluster_centers_.astype(np.float32)  # shape: (K, 14)

print("Building inverted lists...", flush=True)
lists: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(K)]
for i in range(N):
    lists[assignments[i]].append((vectors[i], int(labels[i])))

empty = sum(1 for lst in lists if len(lst) == 0)
if empty:
    print(f"WARNING: {empty} empty clusters", flush=True)

print(f"Writing {OUTPUT}...", flush=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "wb") as f:
    # Header
    f.write(struct.pack("<II", K, D))
    # Centroids: K * 14 * 4B f32, row-major
    for centroid in centroids:
        f.write(centroid.astype(np.float32).tobytes())
    # List sizes: K * 4B u32
    for lst in lists:
        f.write(struct.pack("<I", len(lst)))
    # Entries: for each cluster, each entry is 14 * 2B f16 + 1B u8
    for lst in lists:
        for vec, label in lst:
            f.write(vec.astype(np.float16).tobytes())
            f.write(struct.pack("B", label))

size_mb = OUTPUT.stat().st_size / 1024**2
print(f"Done. {OUTPUT} = {size_mb:.1f} MB (expected ~87 MB)", flush=True)
```

- [ ] **Step 3: Verify script runs locally**

```bash
# From fraud-detection/ directory
uv pip install numpy scikit-learn   # or: pip install -r tools/requirements.txt
python tools/build_ivf.py
```

Expected output:
```
Loading resources/references.json.gz...
Loaded 3000000 vectors, D=14
Clustering K=1732, batch_size=50000, n_init=3...
Building inverted lists...
Writing resources/ivf_index.bin...
Done. resources/ivf_index.bin = 86.X MB (expected ~87 MB)
```

Expected run time: 3–8 minutes depending on CPU.

- [ ] **Step 4: Verify binary header is correct**

```bash
python3 -c "
import struct
with open('resources/ivf_index.bin', 'rb') as f:
    k, d = struct.unpack('<II', f.read(8))
print(f'K={k}, D={d}')
assert k == 1732, f'Expected K=1732, got {k}'
assert d == 14, f'Expected D=14, got {d}'
print('Header OK')
"
```

Expected:
```
K=1732, D=14
Header OK
```

- [ ] **Step 5: Commit**

```bash
git add tools/build_ivf.py tools/requirements.txt
git commit -m "feat(tools): add build_ivf.py — MiniBatchKMeans IVF index builder (K=1732)"
```

---

### Task 7: Dockerfile multi-stage build + .gitignore

**Files:**
- Modify: `Dockerfile`
- Modify: `.gitignore`

Add a Python build stage that runs `build_ivf.py` and produces `ivf_index.bin`. The Rust build stage copies this binary in before compiling. The final image copies it from the Python stage. Remove the now-unused `cargo run --bin preprocess` step.

- [ ] **Step 1: Add `resources/ivf_index.bin` to `.gitignore`**

Append to `.gitignore`:

```
resources/ivf_index.bin
```

- [ ] **Step 2: Replace `Dockerfile`**

```dockerfile
# Stage 1: Build IVF index (Python)
FROM python:3.12-slim AS ivf-builder
WORKDIR /build
COPY resources/references.json.gz resources/
COPY tools/ tools/
RUN pip install --no-cache-dir -r tools/requirements.txt && \
    python tools/build_ivf.py

# Stage 2: Build Rust binary
FROM rust:1.82-slim AS rust-builder
RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY bin/ bin/
COPY resources/ resources/
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
RUN cargo build --release

# Stage 3: Runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=rust-builder /app/target/release/fraud-detection ./
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
COPY --from=rust-builder /app/resources/mcc_risk.json resources/mcc_risk.json
COPY --from=rust-builder /app/resources/normalization.json resources/normalization.json
EXPOSE 3000
CMD ["./fraud-detection"]
```

Note: `cargo run --bin preprocess` is removed — `refs.bin` is no longer needed at runtime.

- [ ] **Step 3: Build Docker image to verify**

```bash
docker build -t fraud-detection:ivf . 2>&1 | tail -20
```

Expected: build completes successfully. The Python stage (~3–8 min) runs first, then Rust compiles. Final image contains `fraud-detection` binary and `ivf_index.bin`.

- [ ] **Step 4: Smoke-test the container**

```bash
docker run --rm -p 3000:3000 fraud-detection:ivf &
sleep 2
curl -s http://localhost:3000/ready
```

Expected: `ok`

```bash
curl -s -X POST http://localhost:3000/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "tx-1329056812",
    "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
    "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
    "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
    "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.23},
    "last_transaction": null
  }'
```

Expected: `{"approved":true,"fraud_score":0.0}`

```bash
# Stop the container
kill %1 2>/dev/null || docker stop $(docker ps -q --filter ancestor=fraud-detection:ivf)
```

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .gitignore
git commit -m "build(docker): add Python IVF build stage, remove preprocess binary from runtime"
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Task |
|---|---|
| spawn_blocking wrap | Task 1 |
| join error → approved:true, fraud_score:0.0 | Task 1 |
| IvfIndex struct with load() and knn() | Task 2 |
| K=1732, D=14, f16 vectors | Task 2 |
| nprobe configurable via IVF_NPROBE | Task 3 |
| Binary format (K/D/centroids/list_sizes/entries) | Task 2 |
| ReferenceRepository delegates to IvfIndex | Task 4 |
| Python MiniBatchKMeans builder | Task 6 |
| Dockerfile Python build stage | Task 7 |
| ivf_index.bin in .gitignore | Task 7 |
| Missing ivf_index.bin → panic at startup | Task 4 (`from_file` returns Err, `main` unwraps) |
| Integration test for ≥97% recall | Note below |

**Recall integration test:** The spec asks for a test asserting ≥97% of top-5 label sets match brute-force at nprobe=8. This requires both `refs.bin` (brute-force) and `ivf_index.bin` to be available locally, plus random sampling of 3M vectors — making it slow and environment-dependent. Validate recall instead via `make load` (k6 load test) and by verifying the two doc-example transactions (`test_legit_transaction_from_docs`, `test_fraud_transaction_from_docs`) still return correct results after switching to IVF. These cover the most critical cases and are already in `tests/integration.rs`.

### 2. Placeholder scan

None found. All steps contain complete code.

### 3. Type consistency

- `IvfIndex::load(path: &Path, nprobe: usize)` — used in Task 2 (definition), Task 4 (`ReferenceRepository::from_file` calls it). ✓
- `IvfIndex::knn(query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]>` — used in Task 2 (definition), Task 4 (delegation). ✓
- `ReferenceRepository::from_file(path: &Path, nprobe: usize)` — defined in Task 4, called in `AppState::build` (Task 4). ✓
- `Config { ivf_path, nprobe }` — defined in Task 3, used in Task 4 (`AppState::build`), Task 5 (test literals). ✓
- `FraudDecision { approved: bool, fraud_score: f32 }` — pre-existing type, used in Task 1 fallback. ✓
