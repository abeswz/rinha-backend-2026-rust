# Adaptive Nprobe + spawn_blocking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix -6000 score from production test by adding spawn_blocking, adaptive two-stage nprobe, mimalloc, warmup, and nginx tuning.

**Architecture:** Three orthogonal changes — (1) move IVF search off Tokio worker via spawn_blocking + thread tuning, (2) two-stage nprobe (fast=5, slow=24) that only runs deep search when first stage returns ambiguous vote count, (3) mimalloc + warmup for allocation and cache warmth. No structural changes to the service graph.

**Tech Stack:** Rust, Tokio, Axum, half::f16, smallvec, mimalloc 0.1, nginx

---

## File Map

| File | Change |
|------|--------|
| `src/repository/ivf.rs` | Remove `nprobe` field; add `nprobe_fast`/`nprobe_slow`; `knn(nprobe)` param; add `knn_adaptive` |
| `src/repository/reference.rs` | Add `knn_adaptive` delegation |
| `src/usecase/score_fraud.rs` | Call `knn_adaptive` instead of `knn` |
| `src/config.rs` | Rename `nprobe` → `nprobe_slow`; default 24 |
| `src/lib.rs` | Use `config.nprobe_slow`; add warmup loop |
| `src/web/handlers.rs` | Replace `timeout` with `spawn_blocking` |
| `src/main.rs` | `worker_threads(2)`, `max_blocking_threads(8)`, mimalloc |
| `Cargo.toml` | Add `mimalloc = { version = "0.1", default-features = false }` |
| `nginx.conf` | `worker_connections 2048`, `keepalive 64`, `proxy_read_timeout 1800ms` |

---

## Task 1: Refactor IvfIndex — Remove nprobe from struct, make knn take nprobe param

**Files:**
- Modify: `src/repository/ivf.rs`

- [ ] **Step 1: Write the failing test for parameterized knn**

Add to `#[cfg(test)]` in `src/repository/ivf.rs`:

```rust
#[test]
fn test_knn_explicit_nprobe_param() {
    let path = write_tiny_ivf("test_ivf_explicit_nprobe.bin");
    let idx = IvfIndex::load(&path, 24).unwrap();
    let query = [0.0f32; 14];
    let labels = idx.knn(&query, 3, 1);
    assert_eq!(labels.len(), 3);
    assert!(labels.iter().all(|&l| l == 0));
    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test test_knn_explicit_nprobe_param 2>&1 | tail -20
```

Expected: compile error — `knn` takes 2 args, not 3.

- [ ] **Step 3: Refactor IvfIndex struct and load/knn signatures**

In `src/repository/ivf.rs`, replace the struct and `load` and `knn`:

```rust
pub struct IvfIndex {
    k: usize,
    nprobe_fast: usize,
    nprobe_slow: usize,
    centroids: Vec<[f32; 14]>,
    lists: Vec<Vec<([f16; 16], u8)>>,
}
```

Change `load` signature from `load(path: &Path, nprobe: usize)` to `load(path: &Path, nprobe_slow: usize)`:

```rust
pub fn load(path: &Path, nprobe_slow: usize) -> std::io::Result<Self> {
    // ... (all existing parsing logic unchanged) ...
    Ok(Self {
        k,
        nprobe_fast: 5,
        nprobe_slow,
        centroids,
        lists,
    })
}
```

Change `knn` to accept `nprobe` as a parameter instead of reading `self.nprobe`:

```rust
pub fn knn(&self, query: &[f32; 14], k: usize, nprobe: usize) -> SmallVec<[u8; 5]> {
    let nprobe = nprobe.min(self.k);
    // rest of body unchanged — replace `self.nprobe` with local `nprobe`
```

- [ ] **Step 4: Fix all existing tests that call knn — add the nprobe argument**

In `src/repository/ivf.rs`, update every call inside `#[cfg(test)]`:

```rust
// test_knn_query_near_legit_cluster
let labels = idx.knn(&query, 3, 1);

// test_knn_query_near_fraud_cluster
let labels = idx.knn(&query, 3, 1);

// test_knn_nprobe_2_returns_from_both_clusters
let labels = idx.knn(&query, 3, 2);

// test_knn_nprobe_clamped_to_k
let labels = idx.knn(&query, 3, 999);

// test_knn_mixed_labels_ordered_by_distance
let labels = idx.knn(&query, 5, 2);
```

- [ ] **Step 5: Run all tests**

```bash
cargo test --lib 2>&1 | tail -30
```

Expected: all tests pass (including new `test_knn_explicit_nprobe_param`).

- [ ] **Step 6: Fix compilation of reference.rs which calls knn**

In `src/repository/reference.rs`, `knn` now needs a third arg. Temporarily pass `self.ivf.nprobe_slow` — this is a bridge until Task 3 replaces it with `knn_adaptive`:

```rust
pub fn knn(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
    self.ivf.knn(query, k, self.ivf.nprobe_slow)
}
```

But `nprobe_slow` is private. Add a getter or make it pub(crate):

In `src/repository/ivf.rs`, change the field:

```rust
pub(crate) nprobe_slow: usize,
```

- [ ] **Step 7: Run cargo build**

```bash
cargo build 2>&1 | grep -E "^error"
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/repository/ivf.rs src/repository/reference.rs
git commit -m "refactor(ivf): remove nprobe from struct, pass as knn param"
```

---

## Task 2: Add knn_adaptive to IvfIndex (TDD with staged fixture)

**Files:**
- Modify: `src/repository/ivf.rs`

- [ ] **Step 1: Write staged fixture helper**

Add to `#[cfg(test)]` block in `src/repository/ivf.rs`:

```rust
/// 6-cluster fixture designed to trigger Stage 2.
///
/// Centroids (all-equal 14-dim vectors):
///   C0=[1.0;14]  C1=[2.0;14]  C2=[3.0;14]
///   C3=[4.0;14]  C4=[5.0;14]  C5=[6.0;14]
///
/// Entries near query [2.5;14]:
///   C2: [3.0;14](legit), [2.4;14](fraud)
///   C3: [4.0;14](legit), [2.6;14](fraud)
///   C5: [2.45;14](fraud), [2.50;14](fraud), [2.55;14](fraud)
///
/// With nprobe_fast=5: probes C0-C4 → returns 2 fraud (ambiguous).
/// With nprobe_slow=6: also probes C5 → returns 5 fraud (decisive).
fn make_staged_ivf_bytes() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();

    // header: k=6, d=14
    buf.extend_from_slice(&6u32.to_le_bytes());
    buf.extend_from_slice(&14u32.to_le_bytes());

    // centroids: 6 × [v;14] for v in 1..=6
    for v in 1u32..=6 {
        for _ in 0..14 {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }

    // list sizes: [3, 2, 2, 2, 2, 3]
    for &sz in &[3u32, 2, 2, 2, 2, 3] {
        buf.extend_from_slice(&sz.to_le_bytes());
    }

    fn push_entry(buf: &mut Vec<u8>, val: f32, label: u8) {
        let v = f16::from_f32(val);
        for _ in 0..14 {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.push(label);
    }

    // C0 (centroid=[1;14]): 3 legit
    push_entry(&mut buf, 1.0, 0);
    push_entry(&mut buf, 1.1, 0);
    push_entry(&mut buf, 1.2, 0);

    // C1 (centroid=[2;14]): 2 legit
    push_entry(&mut buf, 2.0, 0);
    push_entry(&mut buf, 2.1, 0);

    // C2 (centroid=[3;14]): 1 legit + 1 fraud
    push_entry(&mut buf, 3.0, 0);
    push_entry(&mut buf, 2.4, 1);

    // C3 (centroid=[4;14]): 1 legit + 1 fraud
    push_entry(&mut buf, 4.0, 0);
    push_entry(&mut buf, 2.6, 1);

    // C4 (centroid=[5;14]): 2 legit
    push_entry(&mut buf, 5.0, 0);
    push_entry(&mut buf, 5.1, 0);

    // C5 (centroid=[6;14]): 3 fraud entries near [2.5;14]
    // Straggler entries — assigned to far cluster at train time
    push_entry(&mut buf, 2.45, 1);
    push_entry(&mut buf, 2.50, 1);
    push_entry(&mut buf, 2.55, 1);

    buf
}

fn write_staged_ivf(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let data = make_staged_ivf_bytes();
    std::fs::write(&path, data).unwrap();
    path
}
```

- [ ] **Step 2: Write failing tests for knn_adaptive**

Add to `#[cfg(test)]` block:

```rust
#[test]
fn test_knn_adaptive_unambiguous_legit_uses_stage1() {
    let path = write_tiny_ivf("test_adapt_legit.bin");
    let idx = IvfIndex::load(&path, 24).unwrap();
    // query near legit cluster → 0 fraud votes → Stage 1 returns immediately
    let labels = idx.knn_adaptive(&[0.0f32; 14], 5);
    assert_eq!(labels.len(), 5);
    assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 0);
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_knn_adaptive_unambiguous_fraud_uses_stage1() {
    let path = write_tiny_ivf("test_adapt_fraud.bin");
    let idx = IvfIndex::load(&path, 24).unwrap();
    // query near fraud cluster → 5 fraud votes → Stage 1 returns immediately
    let labels = idx.knn_adaptive(&[10.0f32; 14], 5);
    assert_eq!(labels.len(), 5);
    assert!(labels.iter().filter(|&&l| l == 1).count() >= 4);
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_knn_adaptive_ambiguous_triggers_stage2() {
    let path = write_staged_ivf("test_adapt_staged.bin");
    // nprobe_slow=6 so Stage 2 probes C5 (which has straggler fraud entries)
    let idx = IvfIndex::load(&path, 6).unwrap();
    let query = [2.5f32; 14];

    // Stage 1 (nprobe=5) → 2 fraud (ambiguous) → triggers Stage 2
    // Stage 2 (nprobe=6) finds C5's straggler fraud entries → 5 fraud
    let labels = idx.knn_adaptive(&query, 5);
    assert_eq!(labels.len(), 5);
    let fraud_count = labels.iter().filter(|&&l| l == 1).count();
    assert!(fraud_count >= 4, "Stage 2 should find straggler fraud entries, got {fraud_count} fraud");

    // Verify Stage 1 alone would have returned only 2 fraud
    let stage1_labels = idx.knn(&query, 5, 5);
    let stage1_fraud = stage1_labels.iter().filter(|&&l| l == 1).count();
    assert_eq!(stage1_fraud, 2, "Stage 1 should be ambiguous (2 fraud), got {stage1_fraud}");

    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cargo test test_knn_adaptive 2>&1 | tail -20
```

Expected: compile error — `knn_adaptive` not defined.

- [ ] **Step 4: Implement knn_adaptive**

Add after the `knn` method in `IvfIndex` impl:

```rust
pub fn knn_adaptive(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
    let stage1 = self.knn(query, k, self.nprobe_fast);
    let fraud_votes = stage1.iter().filter(|&&l| l == 1).count();
    // Unambiguous: 0-1 fraud votes (clear legit) or k-1..k fraud votes (clear fraud)
    if fraud_votes <= 1 || fraud_votes >= k.saturating_sub(1) {
        return stage1;
    }
    // Ambiguous: run full slow-path search
    self.knn(query, k, self.nprobe_slow)
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test test_knn_adaptive 2>&1 | tail -30
```

Expected: all 3 `test_knn_adaptive_*` tests pass.

- [ ] **Step 6: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/repository/ivf.rs
git commit -m "feat(ivf): add two-stage knn_adaptive with nprobe_fast=5"
```

---

## Task 3: Expose knn_adaptive in ReferenceRepository

**Files:**
- Modify: `src/repository/reference.rs`

- [ ] **Step 1: Write failing test**

Add to `src/repository/reference.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tiny_repo_ivf(name: &str) -> std::path::PathBuf {
        // Reuse same binary structure as ivf tests (k=2, d=14, 3+3 entries)
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&14u32.to_le_bytes());
        for _ in 0..14 { buf.extend_from_slice(&0.0f32.to_le_bytes()); }
        for _ in 0..14 { buf.extend_from_slice(&10.0f32.to_le_bytes()); }
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        for _ in 0..3 {
            for _ in 0..14 { buf.extend_from_slice(&half::f16::from_f32(0.1).to_le_bytes()); }
            buf.push(0u8);
        }
        for _ in 0..3 {
            for _ in 0..14 { buf.extend_from_slice(&half::f16::from_f32(10.0).to_le_bytes()); }
            buf.push(1u8);
        }
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, buf).unwrap();
        path
    }

    #[test]
    fn test_knn_adaptive_legit_query() {
        let path = write_tiny_repo_ivf("repo_adapt_legit.bin");
        let repo = ReferenceRepository::from_file(&path, 24).unwrap();
        let query = [0.0f32; 14];
        let labels = repo.knn_adaptive(&query, 5);
        assert_eq!(labels.len(), 5);
        assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 0);
        std::fs::remove_file(&path).ok();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p fraud-detection test_knn_adaptive_legit_query 2>&1 | tail -15
```

Expected: compile error — `knn_adaptive` not on `ReferenceRepository`.

- [ ] **Step 3: Add knn_adaptive to ReferenceRepository**

Replace the full content of `src/repository/reference.rs`:

```rust
use super::ivf::IvfIndex;
use smallvec::SmallVec;
use std::path::Path;

pub struct ReferenceRepository {
    ivf: IvfIndex,
}

impl ReferenceRepository {
    pub fn from_file(path: &Path, nprobe_slow: usize) -> std::io::Result<Self> {
        let ivf = IvfIndex::load(path, nprobe_slow)?;
        Ok(Self { ivf })
    }

    pub fn knn_adaptive(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
        self.ivf.knn_adaptive(query, k)
    }
}
```

Note: the old `knn` method is removed — it was only used by `score_fraud.rs` which will be updated in Task 4.

- [ ] **Step 4: Run tests**

```bash
cargo test 2>&1 | tail -20
```

Expected: all pass. If `score_fraud.rs` fails to compile because it calls `repo.knn`, that's expected — fix it in Task 4.

- [ ] **Step 5: Commit**

```bash
git add src/repository/reference.rs
git commit -m "feat(repository): expose knn_adaptive, remove old knn delegation"
```

---

## Task 4: ScoreFraudUseCase calls knn_adaptive

**Files:**
- Modify: `src/usecase/score_fraud.rs`

- [ ] **Step 1: Write failing test**

Add to `src/usecase/score_fraud.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::repository::reference::ReferenceRepository;
    use crate::service::vectorizer::{MccRiskMap, NormalizationConstants, Vectorizer};
    use crate::domain::transaction::{Customer, LastTransaction, Merchant, Terminal, Transaction};
    use chrono::Utc;

    fn make_repo() -> ReferenceRepository {
        // Tiny 2-cluster fixture: k=2, 3 legit + 3 fraud
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&14u32.to_le_bytes());
        for _ in 0..14 { buf.extend_from_slice(&0.0f32.to_le_bytes()); }
        for _ in 0..14 { buf.extend_from_slice(&10.0f32.to_le_bytes()); }
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        for _ in 0..3 {
            for _ in 0..14 { buf.extend_from_slice(&half::f16::from_f32(0.1).to_le_bytes()); }
            buf.push(0u8);
        }
        for _ in 0..3 {
            for _ in 0..14 { buf.extend_from_slice(&half::f16::from_f32(10.0).to_le_bytes()); }
            buf.push(1u8);
        }
        let path = std::env::temp_dir().join("usecase_test_ivf.bin");
        std::fs::write(&path, &buf).unwrap();
        ReferenceRepository::from_file(&path, 24).unwrap()
    }

    fn make_tx(amount: f64) -> Transaction {
        Transaction {
            id: "test-tx-001".to_string(),
            amount,
            installments: 1,
            requested_at: Utc::now(),
            customer: Customer { avg_amount: 100.0, tx_count_24h: 1, known_merchants: 1 },
            merchant: Merchant { id: "m1".to_string(), mcc: 5411, avg_amount: 100.0 },
            terminal: Terminal { is_online: true, card_present: true, km_from_home: 0.5 },
            last_transaction: None,
        }
    }

    #[test]
    fn test_execute_returns_fraud_decision() {
        let repo = make_repo();
        let norm = NormalizationConstants::default();
        let mcc_risk = MccRiskMap::default();
        let use_case = ScoreFraudUseCase {
            vectorizer: Vectorizer::new(norm, mcc_risk),
            repository: repo,
        };
        let tx = make_tx(100.0);
        let decision = use_case.execute(&tx);
        assert!(decision.fraud_score >= 0.0 && decision.fraud_score <= 1.0);
    }
}
```

- [ ] **Step 2: Run test to verify current state**

```bash
cargo test test_execute_returns_fraud_decision 2>&1 | tail -20
```

Expected: compile error if `repo.knn` is gone from reference.rs (Task 3 removed it). This confirms the dependency.

- [ ] **Step 3: Update execute to call knn_adaptive**

Replace `src/usecase/score_fraud.rs`:

```rust
use crate::domain::{fraud::FraudDecision, transaction::Transaction};
use crate::repository::reference::ReferenceRepository;
use crate::service::vectorizer::Vectorizer;

pub struct ScoreFraudUseCase {
    pub vectorizer: Vectorizer,
    pub repository: ReferenceRepository,
}

impl ScoreFraudUseCase {
    pub fn execute(&self, tx: &Transaction) -> FraudDecision {
        let vector = self.vectorizer.vectorize(tx);
        let labels = self.repository.knn_adaptive(&vector.0, 5);
        let fraud_count = labels.iter().filter(|&&l| l == 1).count();
        let fraud_score = fraud_count as f32 / 5.0;
        FraudDecision {
            approved: fraud_score < 0.6,
            fraud_score,
        }
    }
}
```

- [ ] **Step 4: Check what NormalizationConstants::default and MccRiskMap::default look like**

```bash
grep -n "impl.*Default\|fn default" src/service/vectorizer.rs
```

If they don't implement `Default`, update the test's `make_repo` to load from test fixtures or use `from_file` with a temp file, OR add `#[derive(Default)]` to those types if they're simple. Check the vectorizer to decide. If `Default` impls don't exist, replace the test with one that doesn't need a full `Vectorizer` — just verify the use_case compiles and delegates correctly.

- [ ] **Step 5: Run all tests**

```bash
cargo test 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/usecase/score_fraud.rs
git commit -m "feat(usecase): use knn_adaptive for two-stage fraud scoring"
```

---

## Task 5: Config — rename nprobe to nprobe_slow, default 24

**Files:**
- Modify: `src/config.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Update Config struct**

Replace `src/config.rs`:

```rust
use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub ivf_path: PathBuf,
    pub mcc_path: PathBuf,
    pub norm_path: PathBuf,
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
            nprobe_slow: std::env::var("IVF_NPROBE")
                .unwrap_or_else(|_| "24".to_string())
                .parse()
                .expect("IVF_NPROBE must be a valid integer"),
        }
    }
}
```

- [ ] **Step 2: Update AppState::build in src/lib.rs**

Change the two references from `config.nprobe` to `config.nprobe_slow`:

```rust
if config.nprobe_slow == 0 {
    return Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "IVF_NPROBE must be >= 1",
    ));
}
let repository = ReferenceRepository::from_file(&config.ivf_path, config.nprobe_slow)?;
```

- [ ] **Step 3: Run cargo build**

```bash
cargo build 2>&1 | grep "^error"
```

Expected: no errors.

- [ ] **Step 4: Run tests**

```bash
cargo test 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/lib.rs
git commit -m "fix(config): rename nprobe to nprobe_slow, set IVF_NPROBE default to 24"
```

---

## Task 6: AppState warmup

**Files:**
- Modify: `src/lib.rs`

Warmup primes CPU branch predictors and L2/L3 caches before test traffic arrives. 500 dummy KNN queries run after the index is loaded.

- [ ] **Step 1: Write failing test**

Add to `src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    #[test]
    fn test_appstate_build_warmup_does_not_panic() {
        // Warmup should complete without panic when index is loaded
        // This test verifies build() succeeds in isolation (not testing warmup count)
        // Full warmup is exercised in integration tests with the real binary
        let config = Config {
            port: 3000,
            ivf_path: PathBuf::from("resources/ivf_index.bin"),
            mcc_path: PathBuf::from("resources/mcc_risk.json"),
            norm_path: PathBuf::from("resources/normalization.json"),
            nprobe_slow: 24,
        };
        // Only run if resources exist (CI/CD may not have them)
        if config.ivf_path.exists() {
            let state = AppState::build(&config);
            assert!(state.is_ok(), "AppState::build should succeed: {:?}", state.err());
        }
    }
}
```

- [ ] **Step 2: Run test to baseline**

```bash
cargo test test_appstate_build_warmup 2>&1 | tail -15
```

Expected: pass (skips if resources not present) or compiles correctly.

- [ ] **Step 3: Add warmup to AppState::build in src/lib.rs**

After building the `state`, add the warmup loop before returning `Ok(state)`:

```rust
pub fn build(config: &Config) -> std::io::Result<Self> {
    if config.nprobe_slow == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "IVF_NPROBE must be >= 1",
        ));
    }
    let repository = ReferenceRepository::from_file(&config.ivf_path, config.nprobe_slow)?;
    let norm = NormalizationConstants::from_file(&config.norm_path)?;
    let mcc_risk = MccRiskMap::from_file(&config.mcc_path)?;
    let vectorizer = Vectorizer::new(norm, mcc_risk);
    let state = Self {
        use_case: ScoreFraudUseCase { vectorizer, repository },
    };
    // Prime CPU branch predictors and L2/L3 caches before serving traffic
    let warmup_query = [0.0f32; 14];
    for _ in 0..500 {
        state.use_case.repository.knn_adaptive(&warmup_query, 5);
    }
    Ok(state)
}
```

- [ ] **Step 4: Run build and tests**

```bash
cargo build 2>&1 | grep "^error" && cargo test 2>&1 | tail -20
```

Expected: no errors, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "perf(appstate): warm up knn cache with 500 dummy queries on startup"
```

---

## Task 7: Handler — replace timeout with spawn_blocking

**Files:**
- Modify: `src/web/handlers.rs`

The current code wraps `execute()` in `timeout(1500ms, ...)`. Since `execute()` is a synchronous CPU computation (IVF search), the timeout future drops without actually cancelling the work — the blocking thread runs to completion anyway, leaking CPU. Replace with `spawn_blocking`.

- [ ] **Step 1: Write a test verifying handler responds**

Check if `axum-test` integration tests exist:

```bash
find . -name "*.rs" | xargs grep -l "axum_test\|axum-test" 2>/dev/null
```

If integration tests exist, check them and add a test that posts a valid transaction and expects 200 OK. If no integration test file exists, add one at `tests/api.rs`:

```rust
// tests/api.rs
use axum_test::TestServer;
use fraud_detection::{config::Config, web::router::build_router, AppState};
use std::sync::Arc;

fn test_server() -> Option<TestServer> {
    let config = Config {
        port: 3000,
        ivf_path: "resources/ivf_index.bin".into(),
        mcc_path: "resources/mcc_risk.json".into(),
        norm_path: "resources/normalization.json".into(),
        nprobe_slow: 24,
    };
    if !config.ivf_path.exists() {
        return None; // skip when resources absent
    }
    let state = Arc::new(AppState::build(&config).unwrap());
    Some(TestServer::new(build_router(state)).unwrap())
}

#[tokio::test]
async fn test_fraud_score_handler_returns_200() {
    let Some(server) = test_server() else { return };
    let body = serde_json::json!({
        "id": "tx-001",
        "transaction": { "amount": 100.0, "installments": 1, "requested_at": "2024-01-01T00:00:00Z" },
        "customer": { "avg_amount": 100.0, "tx_count_24h": 1, "known_merchants": 1 },
        "merchant": { "id": "m1", "mcc": 5411, "avg_amount": 100.0 },
        "terminal": { "is_online": true, "card_present": true, "km_from_home": 0.5 },
        "last_transaction": null
    });
    let resp = server.post("/score").json(&body).await;
    resp.assert_status_ok();
    let json: serde_json::Value = resp.json();
    assert!(json["approved"].is_boolean());
    assert!(json["fraud_score"].is_number());
}
```

- [ ] **Step 2: Run test to baseline**

```bash
cargo test test_fraud_score_handler_returns_200 2>&1 | tail -15
```

Expected: pass or skip (if resources absent).

- [ ] **Step 3: Replace timeout with spawn_blocking in handlers.rs**

Replace the full content of `src/web/handlers.rs`:

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
    let decision = tokio::task::spawn_blocking(move || state.use_case.execute(&tx))
        .await
        .unwrap_or(FraudDecision {
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

Note: `spawn_blocking` requires the closure to be `'static`. `state` is `Arc<AppState>` which is `Send + Sync + 'static`. `tx` is owned. Both are moved into the closure — this compiles.

- [ ] **Step 4: Run build and tests**

```bash
cargo build 2>&1 | grep "^error" && cargo test 2>&1 | tail -20
```

Expected: no errors, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/web/handlers.rs
git commit -m "fix(handler): replace timeout with spawn_blocking for IVF search"
```

---

## Task 8: main.rs — thread tuning + mimalloc

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/main.rs`

- [ ] **Step 1: Add mimalloc to Cargo.toml**

In the `[dependencies]` section of `Cargo.toml`, add:

```toml
mimalloc = { version = "0.1", default-features = false }
```

- [ ] **Step 2: Update main.rs**

Replace full content of `src/main.rs`:

```rust
use fraud_detection::{config::Config, web::router::build_router, AppState};
use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let config = Config::from_env();
    let state = Arc::new(AppState::build(&config).expect("failed to initialize AppState"));
    let addr = format!("0.0.0.0:{}", config.port);
    let router = build_router(state);

    tracing::info!("listening on {addr}");

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(8)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("failed to bind listener");
            axum::serve(listener, router).await.expect("server error");
        });
}
```

Rationale for thread counts:
- `worker_threads(2)`: 2 async threads handle accept/parse/serialize
- `max_blocking_threads(8)`: up to 8 threads run concurrent IVF searches; with 0.475 CPU/instance and ~0.5ms/IVF call, OS scheduler throttles to actual CPU budget

- [ ] **Step 3: Run cargo build**

```bash
cargo build --release 2>&1 | grep "^error"
```

Expected: no errors. mimalloc will be compiled and linked.

- [ ] **Step 4: Run tests**

```bash
cargo test 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/main.rs
git commit -m "perf(runtime): worker_threads=2, max_blocking_threads=8, mimalloc allocator"
```

---

## Task 9: nginx.conf tuning

**Files:**
- Modify: `nginx.conf`

- [ ] **Step 1: Update nginx.conf**

Replace the full content of `nginx.conf`:

```nginx
worker_processes 1;

events {
    worker_connections 2048;
}

http {
    upstream api_backends {
        least_conn;
        server api1:3000;
        server api2:3000;
        keepalive 64;
    }

    server {
        listen 9999;

        location / {
            proxy_pass http://api_backends;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 1800ms;
            proxy_connect_timeout 1s;
        }
    }
}
```

Changes from baseline:
- `worker_connections 1024` → `2048`: headroom for burst connections
- `keepalive 32` → `64`: more persistent connections to backends
- `proxy_read_timeout 2s` → `1800ms`: gives 200ms margin before test's 2001ms timeout

- [ ] **Step 2: Verify nginx config parses (if nginx is installed)**

```bash
nginx -t -c $(pwd)/nginx.conf 2>&1 || echo "nginx not installed locally — skip"
```

- [ ] **Step 3: Commit**

```bash
git add nginx.conf
git commit -m "perf(nginx): worker_connections=2048, keepalive=64, read_timeout=1800ms"
```

---

## Task 10: Final verification

- [ ] **Step 1: Full build**

```bash
cargo build --release 2>&1 | grep -E "^error|^warning.*unused"
```

Expected: no errors, no unused warnings.

- [ ] **Step 2: Clippy**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -20
```

Expected: clean.

- [ ] **Step 3: Format**

```bash
cargo fmt --check 2>&1
```

Expected: no diff.

- [ ] **Step 4: Full test suite**

```bash
cargo test 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 5: Smoke-check the staged fixture test explicitly**

```bash
cargo test test_knn_adaptive_ambiguous_triggers_stage2 -- --nocapture 2>&1
```

Expected: PASSED, Stage 2 found ≥4 fraud entries.

---

## Self-Review Against Spec

| Spec requirement | Task |
|---|---|
| `execute()` in `spawn_blocking` | Task 7 |
| Remove `timeout` wrapper | Task 7 |
| `worker_threads(2)`, `max_blocking_threads(8)` | Task 8 |
| `nprobe_fast=5` hardcoded in struct | Task 1 (load sets `nprobe_fast=5`) |
| `nprobe_slow` from `IVF_NPROBE` env (default 24) | Task 5 |
| `knn_adaptive` two-stage logic | Task 2 |
| Stage 2 re-runs centroid sort (no partial reuse) | Task 2 (knn called twice) |
| `ScoreFraudUseCase::execute` calls `knn_adaptive` | Task 4 |
| mimalloc global allocator | Task 8 |
| 500 dummy warmup queries | Task 6 |
| `nginx: worker_connections 2048` | Task 9 |
| `nginx: keepalive 64` | Task 9 |
| `nginx: proxy_read_timeout 1800ms` | Task 9 |
