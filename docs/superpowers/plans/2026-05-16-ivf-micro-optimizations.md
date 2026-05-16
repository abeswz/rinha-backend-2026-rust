# IVF Micro-Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore spawn_blocking to fix the p99=2001ms regression, then eliminate per-request centroid padding overhead and realloc tail latency.

**Architecture:** Four independent changes applied in order: (1) spawn_blocking critical fix, (2) load-time centroid pre-padding to eliminate hot-loop stack allocs, (3) CENTROID_BUF capacity bump to eliminate realloc, (4) nprobe_fast=2 env var in docker-compose to cut CPU per fast-path call.

**Tech Stack:** Rust, Tokio (spawn_blocking), AVX2 SIMD, docker-compose

---

## File Map

| File | Change |
|------|--------|
| `src/web/handlers.rs` | Wrap `use_case.execute` in `spawn_blocking` |
| `src/repository/ivf.rs` | `centroids: Vec<[f32;16]>`, pre-pad at load, fix `centroid_sq_dist` sig, bump `CENTROID_BUF` to 3072 |
| `docker-compose.yml` | Add `IVF_NPROBE_FAST=2` to api1 and api2 |

---

## Task 1: Restore spawn_blocking (CRITICAL — fix p99 regression)

**Files:**
- Modify: `src/web/handlers.rs:18`

**Background:** Without `spawn_blocking`, Tokio worker threads run CPU-bound KNN directly. Under Docker cgroup CPU throttling (0.475 CPU quota), the kernel throttles a worker thread mid-KNN. The Tokio reactor cannot schedule new tasks. Requests queue, hit the 2001ms k6 timeout. The fix is transparent to callers — same input/output, behavior moves to Tokio's blocking thread pool.

**Test strategy:** The behavior change is not observable from the caller's perspective. The existing integration tests in `tests/integration.rs` serve as the regression guard: they verify that legit and fraud transactions still return correct responses after the refactor.

- [ ] **Step 1: Establish test baseline — run tests before changing anything**

```bash
cargo test 2>&1 | tail -5
```

Expected: all tests pass (34 tests). If any fail, stop and investigate before continuing.

- [ ] **Step 2: Implement spawn_blocking in handler**

Current `src/web/handlers.rs:13-23`:
```rust
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
```

Replace with:
```rust
pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> impl axum::response::IntoResponse {
    let tx = into_transaction(req);
    let decision = tokio::task::spawn_blocking(move || state.use_case.execute(&tx))
        .await
        .expect("KNN task panicked");
    Json(FraudScoreResponse {
        approved: decision.approved,
        fraud_score: decision.fraud_score,
    })
}
```

- [ ] **Step 3: Verify compilation**

```bash
cargo build 2>&1 | grep -E "^error"
```

Expected: no output (clean build).

- [ ] **Step 4: Run all tests**

```bash
cargo test 2>&1 | tail -5
```

Expected: all 34 tests pass.

- [ ] **Step 5: Clippy**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | grep -E "^error"
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/web/handlers.rs
git commit -m "fix(handler): restore spawn_blocking for CPU-bound KNN

Removes spawn_blocking caused Tokio workers to block under Docker cgroup
CPU throttling, queuing requests until k6 timeout (p99=2001ms, 7.47% errors).
KNN now runs on Tokio blocking thread pool; Tokio reactor stays free for I/O."
```

---

## Task 2: Pre-pad centroids to [f32;16] at load time

**Files:**
- Modify: `src/repository/ivf.rs`

**Background:** `IvfIndex.centroids` is currently `Vec<[f32;14]>`. Every call to `centroid_sq_dist` in the KNN hot loop creates a `[f32;16]` on the stack and copies 14 elements before calling the AVX2 SIMD path. With K=3000 centroids per request, this is 3000 × (stack alloc + 56-byte copy). Pre-padding at load time eliminates this entirely — the SIMD function receives a `&[f32;16]` directly.

**Key invariant:** On-disk binary format is unchanged (14×f32 per centroid). Padding to 16 dims happens during `load()` only.

- [ ] **Step 1: Write the updated test for `test_centroid_sq_dist_correctness` — make it compile-fail first by changing the arg type while the function still takes `&[f32;14]`**

In `src/repository/ivf.rs` test module, find `test_centroid_sq_dist_correctness` (line 524). Change:

```rust
let centroid = [0.5f32; 14];
```

to:

```rust
let mut centroid = [0.0f32; 16];
centroid[..14].fill(0.5);
```

The `expected` computation block stays unchanged (it uses `0..14` range, and dims 14-15 are 0.0 in both `q16` and `centroid`, so they contribute 0 to the sum).

Run:
```bash
cargo test test_centroid_sq_dist_correctness 2>&1 | head -20
```

Expected: **compile error** — `centroid_sq_dist` still has signature `(&[f32;16], &[f32;14])`, but you're now passing `&[f32;16]`. This confirms the test will validate the new signature.

- [ ] **Step 2: Change `IvfIndex.centroids` field type**

In `src/repository/ivf.rs`, line 14:
```rust
centroids: Vec<[f32; 14]>,
```
Change to:
```rust
centroids: Vec<[f32; 16]>,
```

- [ ] **Step 3: Update `load()` centroid parsing to pad to 16 dims**

In `src/repository/ivf.rs`, lines 48-56 (the centroid parsing loop):

Current:
```rust
let mut centroids = Vec::with_capacity(k);
for _ in 0..k {
    let mut c = [0.0f32; 14];
    for elem in &mut c {
        *elem = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
    }
    centroids.push(c);
}
```

Replace with:
```rust
let mut centroids = Vec::with_capacity(k);
for _ in 0..k {
    let mut c = [0.0f32; 16];
    for elem in &mut c[..14] {
        *elem = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
    }
    centroids.push(c);
}
```

Dims 14 and 15 default to `0.0f32` (zero-initialized). On-disk format is still 14×f32; we only read 14 elements.

- [ ] **Step 4: Update `centroid_sq_dist` signature and remove internal copy**

Current `centroid_sq_dist` (lines 191-206):
```rust
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

Replace with:
```rust
fn centroid_sq_dist(query16: &[f32; 16], centroid16: &[f32; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { centroid_sq_dist_simd(query16, centroid16) };
        }
    }
    let mut sum = 0.0f32;
    for i in 0..14 {
        let d = query16[i] - centroid16[i];
        sum += d * d;
    }
    sum
}
```

The scalar fallback iterates 0..14 only. Dims 14-15 are 0.0 in both query (padded in `knn()`) and centroid (padded at load time), so they contribute 0 and can safely be skipped.

- [ ] **Step 5: Run all tests**

```bash
cargo test 2>&1 | tail -5
```

Expected: all 34 tests pass, including `test_centroid_sq_dist_correctness`.

- [ ] **Step 6: Clippy**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | grep -E "^error"
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add src/repository/ivf.rs
git commit -m "perf(ivf): pre-pad centroids to [f32;16] at load time

Eliminates 3000 x (stack alloc + 56-byte copy) per request. centroid_sq_dist
now accepts &[f32;16] directly and calls AVX2 SIMD without internal temp buffer.
On-disk binary format unchanged: padding happens during load() only."
```

---

## Task 3: Bump CENTROID_BUF capacity 2048→3072

**Files:**
- Modify: `src/repository/ivf.rs:7`

**Background:** `CENTROID_BUF` is a thread-local `Vec<(f32, usize)>` used in `knn()` to collect all centroid distances before partial select. With K=3000 centroids, the current capacity of 2048 triggers a realloc on the first request processed by each blocking thread. Setting capacity to 3072 (next power-of-1024 above K=3000) ensures no realloc ever occurs.

- [ ] **Step 1: Change CENTROID_BUF capacity**

In `src/repository/ivf.rs`, line 7:
```rust
static CENTROID_BUF: RefCell<Vec<(f32, usize)>> = RefCell::new(Vec::with_capacity(2048));
```

Change to:
```rust
static CENTROID_BUF: RefCell<Vec<(f32, usize)>> = RefCell::new(Vec::with_capacity(3072));
```

- [ ] **Step 2: Run all tests**

```bash
cargo test 2>&1 | tail -5
```

Expected: all 34 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/repository/ivf.rs
git commit -m "perf(ivf): bump CENTROID_BUF capacity 2048->3072

K=3000 centroids triggered a realloc on first request per blocking thread.
3072 > K so Vec never reallocates."
```

---

## Task 4: Set IVF_NPROBE_FAST=2 in docker-compose

**Files:**
- Modify: `docker-compose.yml`

**Background:** Default `IVF_NPROBE_FAST=3` scans 3 clusters × ~1000 vectors = ~3000 comparisons on the fast path. Setting to 2 cuts this to ~2000 comparisons — ~33% less blocking-thread CPU per fast-path call. The trade-off is possible FN increase. Accept FN ≤ 20 in load test; if exceeded, revert `IVF_NPROBE_FAST=2→3` in this file (no recompile).

- [ ] **Step 1: Add IVF_NPROBE_FAST=2 to api1 environment**

In `docker-compose.yml`, the `api1` service `environment` block (lines 28-30):

Current:
```yaml
    environment:
      - PORT=3000
      - IVF_NPROBE=24
      - SOCKET_PATH=/run/sock/api1.sock
```

Replace with:
```yaml
    environment:
      - PORT=3000
      - IVF_NPROBE=24
      - IVF_NPROBE_FAST=2
      - SOCKET_PATH=/run/sock/api1.sock
```

- [ ] **Step 2: Add IVF_NPROBE_FAST=2 to api2 environment**

In `docker-compose.yml`, the `api2` service `environment` block (lines 43-45):

Current:
```yaml
    environment:
      - PORT=3000
      - IVF_NPROBE=24
      - SOCKET_PATH=/run/sock/api2.sock
```

Replace with:
```yaml
    environment:
      - PORT=3000
      - IVF_NPROBE=24
      - IVF_NPROBE_FAST=2
      - SOCKET_PATH=/run/sock/api2.sock
```

- [ ] **Step 3: Run all tests (code unchanged — verify nothing broken)**

```bash
cargo test 2>&1 | tail -5
```

Expected: all 34 tests pass.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "perf(config): set IVF_NPROBE_FAST=2 in docker-compose

Cuts fast-path comparisons from ~3000 to ~2000 (-33% CPU per request).
Revert to 3 if load test shows FN > 20 (no recompile needed)."
```

---

## Final Verification

- [ ] **Run full test suite and clippy**

```bash
cargo test && cargo clippy --all-targets --all-features -- -D warnings
```

Expected: all 34 tests pass, zero clippy warnings.

- [ ] **Run local load test**

```bash
make load
```

Compare p99 and FP/FN vs pre-regression baseline (FP=5, FN=9). Accept FN ≤ 20. If FN > 20, revert `IVF_NPROBE_FAST=2` to `3` in `docker-compose.yml`.

---

## Rollback Reference

| Change | Rollback |
|--------|---------|
| spawn_blocking | Non-negotiable — do not revert |
| Pre-padded centroids | `git revert <commit>` |
| CENTROID_BUF capacity | `git revert <commit>` |
| nprobe_fast=2 | Change `IVF_NPROBE_FAST=2` → `IVF_NPROBE_FAST=3` in docker-compose, no recompile |
