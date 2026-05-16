# IVF Latency Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push remote p99 from 83ms to ~5-15ms by reducing KNN work per request, adding SIMD centroid distance, and eliminating spawn_blocking thread handoff latency.

**Architecture:** Increase IVF clusters K=1732→3000 to shrink per-cluster size, drop nprobe_fast=5→3 probes so fast-path scans 3000 vectors instead of 8650. Add AVX2 SIMD to centroid distance to offset the doubled centroid scan count. Remove `spawn_blocking` since KNN is now sub-1ms and safe on a Tokio worker thread. Rebalance CPU from nginx (0.10→0.05) to each API (0.45→0.475).

**Tech Stack:** Rust, Axum, Tokio, std::arch AVX2 intrinsics, Python/sklearn for IVF rebuild, Docker Compose

---

## File Map

| File | Change |
|------|--------|
| `src/config.rs` | Add `nprobe_fast: usize` field, read from `IVF_NPROBE_FAST` env (default 3) |
| `src/repository/ivf.rs` | (1) `load()` accepts `nprobe_fast` param instead of hardcoding 5; (2) SIMD `centroid_sq_dist_simd`; (3) `centroid_sq_dist` dispatches to SIMD |
| `src/repository/reference.rs` | `from_file()` accepts `nprobe_fast`, passes to `IvfIndex::load` |
| `src/lib.rs` | Pass `config.nprobe_fast` to `from_file()`; update test Config literal |
| `src/usecase/score_fraud.rs` | Update test helper `make_repo()` to pass `nprobe_fast=3` |
| `tests/integration.rs` | Update `Config` struct literal to include `nprobe_fast: 3` |
| `src/web/handlers.rs` | Remove `spawn_blocking` + `timeout`; call `state.use_case.execute(&tx)` directly |
| `src/main.rs` | `worker_threads` 2→4; remove `.max_blocking_threads(2)` |
| `docker-compose.yml` | nginx CPU `"0.10"`→`"0.05"`; api1+api2 CPU `"0.45"`→`"0.475"` |
| `tools/build_ivf.py` | `K = 1732` → `K = 3000` |

---

### Task 1: nprobe_fast Config Wiring

Thread `nprobe_fast` from env var → Config → IvfIndex::load, replacing the hardcoded value of 5. This touches five files atomically since all callers of `IvfIndex::load` and `ReferenceRepository::from_file` must be updated simultaneously.

**Files:**
- Modify: `src/config.rs`
- Modify: `src/repository/ivf.rs:95-101` (load constructor, test helpers)
- Modify: `src/repository/reference.rs:10-12`
- Modify: `src/lib.rs:26` and `src/lib.rs:55-59` (test)
- Modify: `src/usecase/score_fraud.rs:57` (test helper)
- Modify: `tests/integration.rs:8-14` (Config literal)

- [ ] **Step 1: Write a failing test for nprobe_fast in Config**

Add to `src/config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_nprobe_fast_default() {
        std::env::remove_var("IVF_NPROBE_FAST");
        let config = Config::from_env();
        assert_eq!(config.nprobe_fast, 3);
    }

    #[test]
    fn test_config_nprobe_fast_from_env() {
        std::env::set_var("IVF_NPROBE_FAST", "5");
        let config = Config::from_env();
        assert_eq!(config.nprobe_fast, 5);
        std::env::remove_var("IVF_NPROBE_FAST");
    }
}
```

- [ ] **Step 2: Run test to confirm compile failure**

```bash
cargo test config::tests 2>&1 | head -30
```
Expected: compile error — `Config` has no field `nprobe_fast`.

- [ ] **Step 3: Add `nprobe_fast` to Config**

Replace the entire `src/config.rs`:

```rust
use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub ivf_path: PathBuf,
    pub mcc_path: PathBuf,
    pub norm_path: PathBuf,
    pub nprobe_fast: usize,
    pub nprobe_slow: usize,
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
                std::env::var("IVF_PATH").unwrap_or_else(|_| "resources/ivf_index.bin".to_string()),
            ),
            mcc_path: PathBuf::from(
                std::env::var("MCC_PATH").unwrap_or_else(|_| "resources/mcc_risk.json".to_string()),
            ),
            norm_path: PathBuf::from(
                std::env::var("NORM_PATH")
                    .unwrap_or_else(|_| "resources/normalization.json".to_string()),
            ),
            nprobe_fast: std::env::var("IVF_NPROBE_FAST")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .expect("IVF_NPROBE_FAST must be a valid integer"),
            nprobe_slow: std::env::var("IVF_NPROBE")
                .unwrap_or_else(|_| "24".to_string())
                .parse()
                .expect("IVF_NPROBE must be a valid integer"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_nprobe_fast_default() {
        std::env::remove_var("IVF_NPROBE_FAST");
        let config = Config::from_env();
        assert_eq!(config.nprobe_fast, 3);
    }

    #[test]
    fn test_config_nprobe_fast_from_env() {
        std::env::set_var("IVF_NPROBE_FAST", "5");
        let config = Config::from_env();
        assert_eq!(config.nprobe_fast, 5);
        std::env::remove_var("IVF_NPROBE_FAST");
    }
}
```

- [ ] **Step 4: Update IvfIndex::load to accept nprobe_fast**

In `src/repository/ivf.rs`, change `load` to accept `nprobe_fast: usize` and remove hardcoded 5. Replace the `load` signature and its `Ok(Self { ... })` block:

```rust
pub fn load(path: &Path, nprobe_fast: usize, nprobe_slow: usize) -> std::io::Result<Self> {
```

And the return at line 95-101:
```rust
Ok(Self {
    k,
    nprobe_fast,
    nprobe_slow,
    centroids,
    lists,
})
```

- [ ] **Step 5: Update IvfIndex test helpers in ivf.rs**

Every call to `IvfIndex::load` in `src/repository/ivf.rs` tests currently has the form `IvfIndex::load(&path, N)`. Change all of them to `IvfIndex::load(&path, 5, N)` (nprobe_fast=5 keeps existing test semantics).

Calls to update (search with `IvfIndex::load` in the file):
- `test_load_parses_header`: `IvfIndex::load(&path, 5, 1)`
- `test_knn_query_near_legit_cluster`: `IvfIndex::load(&path, 5, 1)`
- `test_knn_query_near_fraud_cluster`: `IvfIndex::load(&path, 5, 1)`
- `test_knn_nprobe_2_returns_from_both_clusters`: `IvfIndex::load(&path, 5, 2)`
- `test_load_rejects_truncated_file`: `IvfIndex::load(&path, 5, 1)`
- `test_knn_nprobe_clamped_to_k`: `IvfIndex::load(&path, 5, 999)`
- `test_knn_mixed_labels_ordered_by_distance`: `IvfIndex::load(&path, 5, 2)`
- `test_knn_explicit_nprobe_param`: `IvfIndex::load(&path, 5, 24)`
- `test_knn_adaptive_unambiguous_legit_uses_stage1`: `IvfIndex::load(&path, 5, 24)`
- `test_knn_adaptive_unambiguous_fraud_uses_stage1`: `IvfIndex::load(&path, 5, 24)`
- `test_knn_adaptive_ambiguous_triggers_stage2`: `IvfIndex::load(&path, 5, 6)`

- [ ] **Step 6: Update ReferenceRepository::from_file**

Replace `src/repository/reference.rs`:

```rust
use super::ivf::IvfIndex;
use smallvec::SmallVec;
use std::path::Path;

pub struct ReferenceRepository {
    ivf: IvfIndex,
}

impl ReferenceRepository {
    pub fn from_file(path: &Path, nprobe_fast: usize, nprobe_slow: usize) -> std::io::Result<Self> {
        let ivf = IvfIndex::load(path, nprobe_fast, nprobe_slow)?;
        Ok(Self { ivf })
    }

    pub fn knn_adaptive(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        self.ivf.knn_adaptive(query, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tiny_repo_ivf(name: &str) -> std::path::PathBuf {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&14u32.to_le_bytes());
        for _ in 0..14 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..14 {
            buf.extend_from_slice(&10.0f32.to_le_bytes());
        }
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        for _ in 0..3 {
            for _ in 0..14 {
                buf.extend_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
            }
            buf.push(0u8);
        }
        for _ in 0..3 {
            for _ in 0..14 {
                buf.extend_from_slice(&half::f16::from_f32(10.0).to_le_bytes());
            }
            buf.push(1u8);
        }
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, buf).unwrap();
        path
    }

    #[test]
    fn test_knn_adaptive_legit_query() {
        let path = write_tiny_repo_ivf("repo_adapt_legit.bin");
        let repo = ReferenceRepository::from_file(&path, 5, 24).unwrap();
        let query = [0.0f32; 14];
        let labels = repo.knn_adaptive(&query, 5);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().filter(|&&l| l == 1).count() <= 2);
        std::fs::remove_file(&path).ok();
    }
}
```

- [ ] **Step 7: Update lib.rs AppState::build and test**

In `src/lib.rs`, change:
```rust
let repository = ReferenceRepository::from_file(&config.ivf_path, config.nprobe_slow)?;
```
To:
```rust
let repository = ReferenceRepository::from_file(&config.ivf_path, config.nprobe_fast, config.nprobe_slow)?;
```

Also add validation for nprobe_fast. After the `nprobe_slow == 0` check, add:
```rust
if config.nprobe_fast == 0 {
    return Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "IVF_NPROBE_FAST must be >= 1",
    ));
}
```

Update the test Config literal in `src/lib.rs` to add `nprobe_fast: 3`:
```rust
let config = Config {
    port: 3000,
    ivf_path: PathBuf::from("resources/ivf_index.bin"),
    mcc_path: PathBuf::from("resources/mcc_risk.json"),
    norm_path: PathBuf::from("resources/normalization.json"),
    nprobe_fast: 3,
    nprobe_slow: 24,
};
```

- [ ] **Step 8: Update score_fraud.rs test helper**

In `src/usecase/score_fraud.rs`, the `make_repo()` test helper calls `ReferenceRepository::from_file(&path, 24)`. Change to `ReferenceRepository::from_file(&path, 3, 24)`.

- [ ] **Step 9: Update integration test Config literal**

In `tests/integration.rs`, change:
```rust
let config = Config {
    port: 3000,
    ivf_path: PathBuf::from("resources/ivf_index.bin"),
    mcc_path: PathBuf::from("resources/mcc_risk.json"),
    norm_path: PathBuf::from("resources/normalization.json"),
    nprobe_fast: 3,
    nprobe_slow: 8,
};
```

- [ ] **Step 10: Run unit tests**

```bash
cargo test --lib 2>&1 | tail -20
```
Expected: all tests pass, no compile errors.

- [ ] **Step 11: Commit**

```bash
git add src/config.rs src/repository/ivf.rs src/repository/reference.rs src/lib.rs src/usecase/score_fraud.rs tests/integration.rs
git commit -m "feat(config): add IVF_NPROBE_FAST env var, thread nprobe_fast through IvfIndex"
```

---

### Task 2: SIMD Centroid Distance

Replace scalar `centroid_sq_dist` with AVX2 dispatch, same pattern as existing `vec_sq_dist_simd`. Centroid scan doubles with K=3000; SIMD reduces cost ~4x to net positive.

**Files:**
- Modify: `src/repository/ivf.rs`

- [ ] **Step 1: Write failing test for centroid_sq_dist correctness**

Add to the `#[cfg(test)]` block in `src/repository/ivf.rs`:

```rust
#[test]
fn test_centroid_sq_dist_correctness() {
    let mut q16 = [0.0f32; 16];
    q16[..14].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]);
    let centroid = [0.5f32; 14];

    let expected: f32 = (0..14usize)
        .map(|i| { let d = q16[i] - centroid[i]; d * d })
        .sum();

    let result = centroid_sq_dist(&q16, &centroid);
    assert!(
        (result - expected).abs() < 1e-3,
        "centroid_sq_dist diverges: got {result}, expected {expected}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails (centroid_sq_dist has wrong signature)**

```bash
cargo test test_centroid_sq_dist_correctness 2>&1 | head -20
```
Expected: compile error — `centroid_sq_dist` currently takes `(&[f32; 14], &[f32; 14])`, not `(&[f32; 16], &[f32; 14])`.

- [ ] **Step 3: Replace centroid_sq_dist with SIMD dispatch version**

In `src/repository/ivf.rs`, replace the `centroid_sq_dist` function and add the SIMD variant. Insert before the existing `vec_sq_dist_simd` function:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn centroid_sq_dist_simd(query16: &[f32; 16], centroid16: &[f32; 16]) -> f32 {
    use std::arch::x86_64::*;

    let q0 = _mm256_loadu_ps(query16.as_ptr());
    let c0 = _mm256_loadu_ps(centroid16.as_ptr());
    let diff0 = _mm256_sub_ps(q0, c0);
    let sq0 = _mm256_mul_ps(diff0, diff0);

    let q1 = _mm256_loadu_ps(query16.as_ptr().add(8));
    let c1 = _mm256_loadu_ps(centroid16.as_ptr().add(8));
    let diff1 = _mm256_sub_ps(q1, c1);
    let sq1 = _mm256_mul_ps(diff1, diff1);

    let sum = _mm256_add_ps(sq0, sq1);
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    _mm_cvtss_f32(sum1)
}

fn centroid_sq_dist(query16: &[f32; 16], centroid: &[f32; 14]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut c16 = [0.0f32; 16];
            c16[..14].copy_from_slice(centroid);
            return unsafe { centroid_sq_dist_simd(query16, &c16) };
        }
    }
    let mut sum = 0.0f32;
    for i in 0..14 {
        let d = query16[i] - centroid[i];
        sum += d * d;
    }
    sum
}
```

Remove the old scalar `centroid_sq_dist` function entirely.

- [ ] **Step 4: Update knn() call site to pass q16 instead of query**

In `IvfIndex::knn`, the centroid distance map currently calls `centroid_sq_dist(query, c)`. Change it to use `q16` (already in scope):

```rust
buf.extend(
    self.centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (centroid_sq_dist(&q16, c), i)),
);
```

- [ ] **Step 5: Run tests**

```bash
cargo test --lib 2>&1 | tail -20
```
Expected: all tests pass including `test_centroid_sq_dist_correctness`.

- [ ] **Step 6: Verify clippy clean**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -20
```
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/repository/ivf.rs
git commit -m "perf(ivf): AVX2 SIMD centroid distance, O(K) centroid scan 4x faster"
```

---

### Task 3: Remove spawn_blocking from Handler

KNN with nprobe_fast=3, K=3000 scans ~3000 SIMD ops ≈ 100-200μs — safe on a Tokio worker thread. Removing `spawn_blocking` eliminates the thread handoff (~100-200μs overhead) and blocking pool queue. Removing the timeout eliminates false negatives from timed-out detection.

**Files:**
- Modify: `src/web/handlers.rs`

- [ ] **Step 1: Write test asserting handler does not require spawn_blocking**

The integration tests in `tests/integration.rs` already cover handler behavior. Before changing, run them to confirm current baseline:

```bash
cargo test --test integration 2>&1 | tail -20
```
Expected: all integration tests pass (requires `resources/ivf_index.bin` to exist; skip if not available).

- [ ] **Step 2: Replace handler implementation**

Replace the entire `src/web/handlers.rs`:

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
use std::sync::Arc;

pub async fn ready_handler() -> &'static str {
    "ok"
}

pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> impl axum::response::IntoResponse {
    let tx = into_transaction(req);
    let decision = state.use_case.execute(&tx);
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
```

Note: `std::time::Duration` import removed (no longer needed). `FraudDecision` import kept since it's used in the old code — **remove it** from the new version since `execute` returns `FraudDecision` but it's not explicitly named in the new handler.

Wait — `FraudDecision` is NOT needed in the imports in the new version because the handler just calls `execute()` and accesses `.approved` and `.fraud_score` fields without naming the type. Remove it from the import block as shown above.

- [ ] **Step 3: Run clippy to catch unused imports**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -20
```
Expected: no warnings. If `FraudDecision` is flagged as unused import, remove it.

- [ ] **Step 4: Run all tests**

```bash
cargo test 2>&1 | tail -20
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/web/handlers.rs
git commit -m "perf(handler): remove spawn_blocking, call execute() inline on worker thread"
```

---

### Task 4: Tokio Runtime Tuning

With no `spawn_blocking` in the hot path, `max_blocking_threads` constraint is irrelevant. 4 worker threads better saturates the 0.475 CPU allocation under concurrent load (scheduler quantum gives each thread ~119ms of CPU time per 500ms period).

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Update Tokio runtime builder**

In `src/main.rs`, replace:
```rust
tokio::runtime::Builder::new_multi_thread()
    .worker_threads(2)
    .max_blocking_threads(2)
    .enable_all()
    .build()
```

With:
```rust
tokio::runtime::Builder::new_multi_thread()
    .worker_threads(4)
    .enable_all()
    .build()
```

- [ ] **Step 2: Run build to verify**

```bash
cargo build --release 2>&1 | tail -10
```
Expected: compiles cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "perf(runtime): worker_threads 2->4, remove max_blocking_threads"
```

---

### Task 5: Docker CPU Rebalance

Redistribute 0.10 CPU from nginx to API containers. Total remains exactly 1.0 CPU. nginx only proxies Unix socket traffic — 0.05 CPU is sufficient.

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Update CPU limits**

In `docker-compose.yml`, make these three changes:

nginx deploy resources:
```yaml
    deploy:
      resources:
        limits:
          cpus: "0.05"
          memory: "10MB"
```

api1 deploy resources:
```yaml
    deploy:
      resources:
        limits:
          cpus: "0.475"
          memory: "170MB"
```

api2 deploy resources:
```yaml
    deploy:
      resources:
        limits:
          cpus: "0.475"
          memory: "170MB"
```

- [ ] **Step 2: Verify CPU total**

```bash
grep -A3 "cpus:" docker-compose.yml
```
Expected output shows: `"0.05"`, `"0.475"`, `"0.475"`. Sum = 1.0 ✓

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "perf(deploy): nginx 0.10->0.05 CPU, api1+api2 0.45->0.475 CPU each"
```

---

### Task 6: IVF Rebuild with K=3000

K=3000 produces tighter clusters (~1000 vectors/cluster vs ~1730). nprobe_fast=3 then scans 3×1000=3000 vectors vs the old 5×1730=8650 — 2.9x fewer comparisons on the fast path. nprobe_slow=12 (down from 24 in the env var if desired, but the existing `IVF_NPROBE=24` remains safe — still correct, just slightly more work on the slow path).

**Files:**
- Modify: `tools/build_ivf.py`

- [ ] **Step 1: Update K in build_ivf.py**

In `tools/build_ivf.py`, change line 30:
```python
K = 3000
```

Also update the expected output comment at the end of the file from `"expected ~87 MB"` to `"expected ~87 MB"` (size is approximately the same since K change only affects centroids, not entries; centroid overhead is negligible).

- [ ] **Step 2: Rebuild the IVF index**

This takes 3-8 minutes:
```bash
make ivf
```
Or equivalently: `uv run tools/build_ivf.py`

Expected output:
```
Loading resources/references.json.gz...
Loaded N vectors, D=14
Clustering K=3000, batch_size=50000, n_init=3...
Building inverted lists...
Writing resources/ivf_index.bin...
Done. resources/ivf_index.bin = XX.X MB (expected ~87 MB)
```

No `ERROR:` lines. No crash.

- [ ] **Step 3: Verify index loads**

```bash
cargo test --test integration -- test_ready_endpoint 2>&1
```
Expected: passes (AppState::build loads the new index without error).

- [ ] **Step 4: Run regression tests against new index**

```bash
cargo test --test integration 2>&1 | tail -20
```
Expected: all integration tests pass. Specifically:
- `test_legit_transaction_from_docs` → `approved: true`, `fraud_score: 0.0`
- `test_fraud_transaction_from_docs` → `approved: false`, `fraud_score: 1.0`

If these fail, nprobe_fast=3 may be insufficient for the tighter clusters. Increase `IVF_NPROBE_FAST` to 4 or 5 and re-test before shipping.

- [ ] **Step 5: Run smoke test**

Start the service locally and run:
```bash
make dev &
sleep 2
make smoke
```
Expected: all 5 smoke requests return 200 with valid JSON.

- [ ] **Step 6: Commit**

```bash
git add tools/build_ivf.py resources/ivf_index.bin
git commit -m "perf(ivf): rebuild K=3000, tighter clusters reduce fast-path scan 3x"
```

---

## Verification Checklist

After all tasks complete, run the full suite:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
make up
make smoke
```

Then run the load test to measure local p99:
```bash
make load
```

Expected local p99 < 1ms (was 0.35ms before; K=3000 with nprobe_fast=3 may be slightly faster or similar). FP and FN counts should match or improve vs. baseline (FP=1, FN=2 @ K=1732/nprobe=5).

Remote submission: `make submission` after all tests pass and docker smoke confirms correct behavior.
