# Design: Adaptive Nprobe + Blocking Fix

**Date:** 2026-05-16  
**Context:** First production test scored -6000 (both p99 and detection cuts triggered)

---

## Problem Statement

Test results:
- `p99 = 2001.99ms` → cut triggered (-3000)
- `failure_rate = 17.94%` → detection cut triggered (-3000)
- `http_errors = 5408` (~10% of responses)
- `final_score = -6000` (floor)

Root causes:
1. `execute()` called directly in async handler without `spawn_blocking` — blocks Tokio's single worker thread
2. `worker_threads(1)` — only one thread processes all requests sequentially
3. Fixed `nprobe=4` misses neighbors at cluster boundaries for ambiguous cases

IVF data is already in memory; the problem is CPU blocking, not I/O.

---

## Architecture

No structural changes. Same: nginx → api1/api2 → in-memory IVF index.

Three focused changes:

### Change 1: spawn_blocking + thread tuning

**handlers.rs** — move `execute()` into blocking thread pool:

```rust
pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> impl axum::response::IntoResponse {
    let tx = into_transaction(req);
    let decision = tokio::task::spawn_blocking(move || {
        state.use_case.execute(&tx)
    })
    .await
    .unwrap_or(FraudDecision { approved: true, fraud_score: 0.0 });

    Json(FraudScoreResponse {
        approved: decision.approved,
        fraud_score: decision.fraud_score,
    })
}
```

Remove the existing `timeout(1500ms, ...)` wrapper — it's redundant and misleading with `spawn_blocking` (the blocking thread runs to completion regardless of future drop).

**main.rs** — tuned thread counts:

```rust
tokio::runtime::Builder::new_multi_thread()
    .worker_threads(2)
    .max_blocking_threads(8)
    .enable_all()
```

Rationale: 2 async threads handle accept/parse/serialize; up to 8 blocking threads run concurrent IVF searches. With 0.475 CPU per instance and ~0.5ms per IVF call, capacity ≈ 950 req/s per instance, 1900 combined. Test sends 900 total → ~2× headroom.

### Change 2: Adaptive two-stage nprobe

**Observation from jairoblatt reference:** Most transactions are unambiguous. With k=5 neighbors, votes split 0-1 or 4-5 for fraud in the majority of cases. Only 2-3 votes (ambiguous) need a deeper search.

**IvfIndex changes:**

- Remove `nprobe` from struct; pass as parameter to `knn()`
- Add `knn_adaptive()` implementing two-stage search:

```
Stage 1: nprobe = 5
  Count fraud votes from 5 nearest clusters.
  If votes ∈ {0, 1, 4, 5}: unambiguous → return immediately.
  If votes ∈ {2, 3}: ambiguous → continue to Stage 2.

Stage 2: nprobe = 24
  Re-run knn over 24 nearest clusters.
  Return definitive result.
```

`ScoreFraudUseCase::execute` calls `knn_adaptive` instead of `knn`.

Config: `IVF_NPROBE` env var controls Stage 2 slow-path probe count (default 24). Stage 1 fast-path is hardcoded at 5.

Stage 2 re-runs the centroid sort with the full slow nprobe count — it does not reuse Stage 1's partial result. Centroid sort is O(K) ≈ 1732 ops and cheap; re-running is correct and simpler than patching the partial result.

`IvfIndex` stores both: `nprobe_fast: usize` (always 5) and `nprobe_slow: usize` (from env). `knn_adaptive` uses both. Standalone `knn(nprobe)` method kept for tests.

### Change 3: mimalloc + warmup

**mimalloc** replaces the system allocator. Under concurrent allocation patterns from the blocking thread pool, mimalloc reduces fragmentation and lock contention.

```toml
[dependencies]
mimalloc = { version = "0.1", default-features = false }
```

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**Warmup:** 500 dummy KNN queries in `AppState::build` after index loads. Primes CPU branch predictors and L2/L3 caches before test traffic arrives.

**nginx.conf minor tuning:**
- `worker_connections 2048` (up from 1024)
- `keepalive 64` (up from 32)
- `proxy_read_timeout 1800ms` (aligned with test's 2001ms timeout, gives 200ms margin)

---

## Expected Outcome

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| p99 | 2001.99ms (cut) | <20ms |
| failure_rate | 17.94% | <2% |
| http_errors | 5408 | ~0 |
| final_score | -6000 | >4000 |

---

## Files Changed

| File | Change |
|------|--------|
| `src/web/handlers.rs` | spawn_blocking, remove timeout wrapper |
| `src/main.rs` | worker_threads=2, max_blocking_threads=8, mimalloc |
| `src/repository/ivf.rs` | nprobe per-call, adaptive two-stage knn |
| `src/usecase/score_fraud.rs` | call knn_adaptive |
| `Cargo.toml` | add mimalloc dep |
| `nginx.conf` | worker_connections, keepalive, proxy_read_timeout |

---

## Constraints

- Memory budget: 170MB per instance — no new data structures
- CPU budget: 0.475 CPU per instance — thread pool bounded by OS scheduler
- No external dependencies beyond mimalloc
- Stay on Tokio + Axum stack
- Network mode: bridge (no UDS between nginx and API)
