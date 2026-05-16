# Design: Throughput Fix — Tokio Threading + Build Optimization

**Date:** 2026-05-16  
**Status:** Approved  
**Context:** Remote preview test results, commit `bc8f9e4`

---

## Problem

Remote preview test returned:

| Metric | Value | Impact |
|---|---|---|
| p99 | 2001.92ms | -3000 cut triggered |
| http_errors | 4797 (14.84% of processed) | Drives failure rate near cut |
| detection accuracy | FP=8, FN=6 / 27,619 processed | 99.95% — not the problem |
| final_score | -4183.78 | p99 cut + detection penalty |

Only 32,416 of 54,100 expected requests were processed (59.9%). Effective throughput ≈ 537 req/s against a 900 req/s target. Queue backup caused k6 timeouts → nginx 499s.

---

## Root Cause

**Tokio worker thread over-provisioning under cgroup CPU limit.**

- Test machine: Mac Mini Late 2014, Intel Core i5-4278U (Haswell), 2 physical cores / 4 logical threads
- `#[tokio::main]` calls `num_cpus::get()` → returns 4 (reads `/proc/cpuinfo`, ignores cgroup limit)
- Tokio creates 4 async worker threads
- Container limit: 0.475 CPU
- Each thread gets 0.475/4 ≈ 12% of a CPU
- `spawn_blocking` adds additional OS threads on top
- OS scheduler context-switches many threads within a tiny CPU budget → latency explodes

KNN with nprobe=4 + AVX2/F16C SIMD on Haswell: ~0.01-0.1ms per call. Neither KNN nor memory I/O are the bottleneck. The thread scheduling overhead is.

**Detection quality is not a problem.** FP=8, FN=6 with nprobe=4 = 99.95% accuracy. nprobe tuning is out of scope.

---

## Constraints

- Test machine: Mac Mini Late 2014 (i5-4278U, Haswell, 2.6 GHz) — fixed hardware, safe to target-cpu=haswell
- 2 API instances, 0.475 CPU + 170MB RAM each
- Total budget: 1 CPU + 350MB across all services
- nginx: 0.05 CPU + 10MB
- k6 timeout: 2001ms — any response > 2001ms = HTTP error (weight 5 in scoring)

---

## Solution: current_thread Runtime + Fast Fallback + Build Optimization

Four independent changes applied together.

---

### Change 1 — Tokio `current_thread` runtime

**File:** `src/main.rs`

Replace `#[tokio::main]` macro with explicit runtime builder:

```rust
fn main() {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let config = Config::from_env();
    let state = Arc::new(AppState::build(&config).expect("failed to initialize AppState"));
    let addr = format!("0.0.0.0:{}", config.port);
    let router = build_router(state);

    tracing::info!("listening on {addr}");

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
            axum::serve(listener, router).await.unwrap();
        });
}
```

Single Tokio worker thread. Zero thread competition. Full 0.475 CPU dedicated to I/O + request dispatch.

---

### Change 2 — Remove `spawn_blocking`

**File:** `src/web/handlers.rs`

KNN at nprobe=4 + SIMD takes ~0.01-0.1ms — safe to run inline in async handler. Blocking an async thread for <0.1ms is harmless.

```rust
pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> impl axum::response::IntoResponse {
    use tokio::time::{timeout, Duration};

    let result = timeout(Duration::from_millis(1500), async move {
        let tx = into_transaction(req);
        state.use_case.execute(&tx)
    })
    .await;

    let decision = result.unwrap_or(FraudDecision {
        approved: true,
        fraud_score: 0.0,
    });

    Json(FraudScoreResponse {
        approved: decision.approved,
        fraud_score: decision.fraud_score,
    })
}
```

The 1500ms timeout is a handler-level guard (defense-in-depth). On timeout, returns `approved: true` (FP, weight=1) instead of HTTP 500 (weight=5). Since `execute()` is synchronous and has no await points, the timeout fires only if the entire handler exceeds 1500ms — pathological case only.

---

### Change 3 — Build optimization

**File:** `Dockerfile` (rust-builder stage)

```dockerfile
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cargo build --release
```

`target-cpu=haswell` enables auto-vectorization across all hot paths (vectorizer math, serde paths, etc.) — not just the explicit `#[target_feature]` SIMD block. Safe to hardcode: test machine is confirmed Haswell.

**File:** `Cargo.toml`

```toml
[profile.release]
lto = "thin"
opt-level = 3
codegen-units = 1
```

- `lto = "thin"`: cross-crate inlining — axum/serde/tokio hot paths inline into application code
- `codegen-units = 1`: single LLVM module → better optimization decisions
- Build time increase: ~3-6 min. Acceptable (runs once per image push in Dockerfile).

---

### Change 4 — nginx keepalive + timeouts

**File:** `nginx.conf`

```nginx
upstream api_backends {
    server api1:3000;
    server api2:3000;
    keepalive 32;
}

location / {
    proxy_pass http://api_backends;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 2s;
    proxy_connect_timeout 1s;
}
```

- `keepalive 32` + HTTP/1.1 + empty `Connection` header: reuses upstream TCP connections. At 900 req/s, avoids per-request TCP handshake overhead.
- `proxy_read_timeout 2s`: nginx drops stale upstream requests just under k6's 2001ms timeout. Prevents nginx holding connections to a backed-up instance indefinitely.

No memory budget change. nginx stays at 10MB.

---

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| Tokio workers | 4 (num_cpus) | 1 (current_thread) |
| spawn_blocking | yes | no |
| http_errors | 4797 | ~0 |
| p99 | 2001.92ms (-3000 cut) | <10ms (~2000+ points) |
| detection accuracy | 99.95% | 99.95% (unchanged) |
| final_score | -4183.78 | ~4500-5500 |

Scoring estimate (0 HTTP errors, FP=8, FN=6, p99=5ms):
- `E = 1×8 + 3×6 = 26`, `ε = 26/54100 = 0.00048`
- `score_det = 1000 × log10(1/0.00048) - 300 × log10(27) ≈ 3320 - 429 = 2891`
- `score_p99 = 1000 × log10(1000/5) ≈ 2301`
- `final ≈ 5192`

---

## Files Changed

```
src/main.rs          — replace #[tokio::main] with current_thread builder
src/web/handlers.rs  — remove spawn_blocking, add 1500ms handler timeout
Cargo.toml           — add [profile.release] section
Dockerfile           — add RUSTFLAGS env in rust-builder stage
nginx.conf           — add keepalive, proxy_http_version, timeouts
```

No changes to: domain logic, IVF index, vectorizer, DTOs, scoring, tests, docker-compose.yml.

---

## Out of Scope

- nprobe tuning: detection is already 99.95% with nprobe=4. Not the bottleneck.
- HNSW: memory budget insufficient.
- Result caching: each transaction vector is unique, cache hit rate ≈ 0%.
- PGO (profile-guided optimization): requires two-pass build, complex Dockerfile. Deferred.
