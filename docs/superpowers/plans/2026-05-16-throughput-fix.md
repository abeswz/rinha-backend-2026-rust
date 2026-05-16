# Throughput Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate Tokio thread over-provisioning and connection overhead to reduce p99 from 2001ms to <10ms and HTTP errors from 4797 to ~0.

**Architecture:** Four independent changes — (1) single-threaded Tokio runtime to stop thread competition within 0.475 CPU budget, (2) inline KNN execution to remove spawn_blocking overhead, (3) release build hardening with LTO + Haswell SIMD, (4) nginx keepalive to eliminate per-request TCP handshake cost.

**Tech Stack:** Rust/Axum/Tokio, nginx

---

## File Map

| File | Change |
|---|---|
| `Cargo.toml` | Add `[profile.release]` with lto/opt-level/codegen-units |
| `Dockerfile` | Add `ENV RUSTFLAGS="-C target-cpu=haswell"` before build |
| `src/main.rs` | Replace `#[tokio::main]` with explicit `current_thread` builder |
| `src/web/handlers.rs` | Remove `spawn_blocking`, inline execution, add 1500ms timeout fallback |
| `nginx.conf` | Add keepalive 32, HTTP/1.1, proxy timeouts |

No changes to: domain logic, IVF index, vectorizer, DTOs, scoring, tests, `docker-compose.yml`.

---

### Task 1: Add release profile to Cargo.toml

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Read current Cargo.toml**

Confirm the file ends after `[profile.profiling]` with no `[profile.release]` section.

Current tail of `Cargo.toml`:
```toml
[profile.profiling]
inherits = "release"
debug = 1
```

- [ ] **Step 2: Add `[profile.release]` section**

Append after `[profile.profiling]`:

```toml
[profile.release]
lto = "thin"
opt-level = 3
codegen-units = 1
```

Full resulting `Cargo.toml` (additions only — append to existing file):
```toml
[profile.release]
lto = "thin"
opt-level = 3
codegen-units = 1
```

- [ ] **Step 3: Verify compilation still succeeds**

Run:
```bash
cargo check --release
```
Expected: no errors, no warnings.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml
git commit -m "perf(build): add release profile with lto=thin, codegen-units=1"
```

---

### Task 2: Add RUSTFLAGS to Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Read current rust-builder stage**

Current stage (lines 9–18):
```dockerfile
FROM rust:1.82-slim AS rust-builder
RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY bin/ bin/
COPY resources/ resources/
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
RUN cargo build --release
```

- [ ] **Step 2: Add RUSTFLAGS before cargo build**

Insert `ENV RUSTFLAGS="-C target-cpu=haswell"` immediately before `RUN cargo build --release`. Result:

```dockerfile
FROM rust:1.82-slim AS rust-builder
RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY bin/ bin/
COPY resources/ resources/
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cargo build --release
```

- [ ] **Step 3: Verify Dockerfile syntax**

Run:
```bash
docker build --dry-run . 2>/dev/null || echo "docker not available — skip"
```

If Docker unavailable, visually confirm no syntax errors (each `RUN`/`COPY`/`ENV` on its own line, no trailing backslash issues).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "perf(build): enable haswell SIMD target via RUSTFLAGS"
```

---

### Task 3: Switch to current_thread runtime in main.rs

**Context:** `#[tokio::main]` calls `num_cpus::get()` → returns 4 (reads `/proc/cpuinfo`, ignores cgroup). Container limit is 0.475 CPU. 4 threads × 0.475/4 ≈ 12% CPU each → scheduler thrash → latency explodes. `current_thread` runtime uses 1 worker thread and dedicates the full 0.475 CPU budget to it.

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Verify existing integration tests pass before touching anything**

Run:
```bash
cargo test --test integration
```
Expected: all tests pass. If any fail, stop — do not proceed until resolved.

- [ ] **Step 2: Replace runtime in main.rs**

Current `src/main.rs`:
```rust
use fraud_detection::{config::Config, web::router::build_router, AppState};
use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let config = Config::from_env();
    let state = Arc::new(AppState::build(&config).expect("failed to initialize AppState"));
    let addr = format!("0.0.0.0:{}", config.port);
    let router = build_router(state);

    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
```

Replace entirely with:
```rust
use fraud_detection::{config::Config, web::router::build_router, AppState};
use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

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

- [ ] **Step 3: Verify it compiles**

Run:
```bash
cargo check
```
Expected: no errors.

- [ ] **Step 4: Run integration tests to confirm runtime change doesn't break routing**

Run:
```bash
cargo test --test integration
```
Expected: all tests pass. `axum_test::TestServer` creates its own runtime internally, so the `main()` runtime change doesn't affect test infrastructure.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "perf(runtime): switch to current_thread tokio runtime"
```

---

### Task 4: Remove spawn_blocking, inline KNN, add handler timeout

**Context:** KNN with nprobe=4 + SIMD takes 0.01–0.1ms — safe to run inline in async. Current `spawn_blocking` adds OS thread creation overhead and loses CPU time to thread context switching within the 0.475 CPU budget. The 1500ms timeout is a defense-in-depth guard: on timeout, return `approved: true` (FP penalty = 1) instead of letting the handler hang and produce an HTTP error (penalty = 5). Since `execute()` is sync with no await points, the timeout fires only in pathological cases.

**Files:**
- Modify: `src/web/handlers.rs`

- [ ] **Step 1: Verify current handler tests pass**

Run:
```bash
cargo test --test integration
```
Expected: all tests pass.

- [ ] **Step 2: Replace handler implementation**

Current `fraud_score_handler` (lines 17–38 in `src/web/handlers.rs`):
```rust
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
    .unwrap_or_else(|e| {
        tracing::error!(err = ?e, "spawn_blocking join failed; falling back to approved");
        FraudDecision {
            approved: true,
            fraud_score: 0.0,
        }
    });
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

- [ ] **Step 3: Remove unused imports**

After the change, `Arc::clone` is no longer used in this file. Check the import at line 11:
```rust
use std::sync::Arc;
```
`Arc` is still used as the `State<Arc<AppState>>` extractor type — keep it. Run clippy to confirm no dead imports:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```
Expected: no warnings.

- [ ] **Step 4: Run integration tests to verify handler behavior unchanged**

Run:
```bash
cargo test --test integration
```
Expected: all 5 tests pass:
- `test_ready_endpoint` — PASS
- `test_legit_transaction_from_docs` — PASS (approved=true, fraud_score=0.0)
- `test_fraud_transaction_from_docs` — PASS (approved=false, fraud_score=1.0)
- `test_null_last_transaction_no_panic` — PASS
- `test_malformed_missing_field` — PASS (422)
- `test_malformed_invalid_timestamp` — PASS (422)

- [ ] **Step 5: Commit**

```bash
git add src/web/handlers.rs
git commit -m "perf(handler): remove spawn_blocking, inline KNN, add 1500ms timeout fallback"
```

---

### Task 5: Update nginx.conf with keepalive and timeouts

**Context:** At 900 req/s, a new TCP handshake per request wastes ~1ms each. `keepalive 32` reuses upstream connections. `proxy_read_timeout 2s` drops stale upstream requests just before k6's 2001ms hard cut, preventing nginx from holding connections to a backed-up instance.

**Files:**
- Modify: `nginx.conf`

- [ ] **Step 1: Read current nginx.conf**

Current content:
```nginx
worker_processes 1;

events {
    worker_connections 1024;
}

http {
    upstream api_backends {
        server api1:3000;
        server api2:3000;
    }

    server {
        listen 9999;

        location / {
            proxy_pass http://api_backends;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

- [ ] **Step 2: Replace nginx.conf**

```nginx
worker_processes 1;

events {
    worker_connections 1024;
}

http {
    upstream api_backends {
        server api1:3000;
        server api2:3000;
        keepalive 32;
    }

    server {
        listen 9999;

        location / {
            proxy_pass http://api_backends;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 2s;
            proxy_connect_timeout 1s;
        }
    }
}
```

Key additions:
- `keepalive 32` — pool of 32 keepalive connections to each upstream
- `proxy_http_version 1.1` + `proxy_set_header Connection ""` — required for keepalive to work (HTTP/1.0 closes connections by default)
- `proxy_read_timeout 2s` — drop upstream connections that don't respond within 2s (just under k6's 2001ms)
- `proxy_connect_timeout 1s` — fail fast on TCP connect to unhealthy upstream

- [ ] **Step 3: Verify nginx config syntax (if nginx available locally)**

Run:
```bash
nginx -t -c $(pwd)/nginx.conf 2>/dev/null || echo "nginx not local — visual review only"
```

If nginx unavailable locally, visually confirm:
- `keepalive` is inside `upstream {}` block
- `proxy_http_version` and `proxy_set_header Connection ""` are inside `location {}` block
- No missing semicolons or braces

- [ ] **Step 4: Commit**

```bash
git add nginx.conf
git commit -m "perf(nginx): add keepalive 32, HTTP/1.1 upstream, proxy timeouts"
```

---

### Task 6: Final verification

- [ ] **Step 1: Full test suite**

Run:
```bash
cargo test
```
Expected: all tests pass with 0 failures.

- [ ] **Step 2: Clippy clean**

Run:
```bash
cargo clippy --all-targets --all-features -- -D warnings
```
Expected: 0 warnings.

- [ ] **Step 3: Format check**

Run:
```bash
cargo fmt --check
```
Expected: 0 diff (or run `cargo fmt` then commit).

- [ ] **Step 4: Release build**

Run:
```bash
cargo build --release 2>&1 | tail -5
```
Expected: compiles successfully. Build will take 3–6 minutes due to `lto = "thin"` + `codegen-units = 1`.

---

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| Tokio workers | 4 (num_cpus) | 1 (current_thread) |
| spawn_blocking | yes | no |
| http_errors | 4797 | ~0 |
| p99 | 2001.92ms (-3000 pts) | <10ms (~+2000 pts) |
| detection accuracy | 99.95% | 99.95% (unchanged) |
| final_score | -4183.78 | ~5000–5200 |
