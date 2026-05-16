# Throughput Fix v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate p99=2001ms and reduce HTTP errors to ~0 by fixing runtime concurrency, removing the biggest per-request heap allocation, and improving nginx load distribution.

**Architecture:** Three independent code changes — (1) multi-thread runtime with 2 workers so I/O and KNN don't share a single event loop thread, (2) replace `HashSet<String>` with `Vec<String>` for `known_merchants` to eliminate 3–10 heap allocations per request, (3) `least_conn` in nginx so the faster backend absorbs more load rather than round-robin stacking on a slow one. Plus a forced clean Docker rebuild to guarantee the binary matches the source.

**Tech Stack:** Rust/Axum/Tokio, nginx

**Root cause confirmed by analysis:** `new_current_thread()` serializes all requests on one event loop thread. At 450 req/s per instance, the single thread can't drain the queue fast enough → backlog → nginx `proxy_read_timeout 2s` → HTTP errors → p99 cut. The IVF index has zero locks; multi-thread is safe with no structural changes.

**Docker rebuild note:** This environment cannot run `docker build` steps that invoke `RUN` (veth/networking limitation). After all code changes are committed and pushed, the user must run `make publish && make submission` on a machine with full Docker support (local dev machine or CI).

---

## File Map

| File | Change |
|---|---|
| `src/main.rs` | `new_current_thread()` → `new_multi_thread().worker_threads(2)` |
| `src/domain/transaction.rs` | `Customer.known_merchants: HashSet<String>` → `Vec<String>` |
| `src/service/vectorizer.rs` | Update unit test fixtures: `HashSet::from([...])` → `vec![...]` |
| `src/web/handlers.rs` | Remove `HashSet` import and `.collect::<HashSet<_>>()` |
| `nginx.conf` | Add `least_conn;` to upstream block |

No changes to: IVF index, scoring logic, DTOs, integration tests, regression tests, docker-compose.yml, Dockerfile, Cargo.toml.

---

### Task 1: Switch to multi-thread runtime with 2 workers

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Read current src/main.rs**

Confirm it uses `Builder::new_current_thread()`. Current file:
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
        .expect("failed to build tokio runtime")
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr).await.expect("failed to bind listener");
            axum::serve(listener, router).await.expect("server error");
        });
}
```

- [ ] **Step 2: Replace runtime builder**

Replace `tokio::runtime::Builder::new_current_thread()` with `new_multi_thread().worker_threads(2)`. Full resulting file:

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

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr).await.expect("failed to bind listener");
            axum::serve(listener, router).await.expect("server error");
        });
}
```

Why `worker_threads(2)` not default: default calls `num_cpus::get()` → 4 threads on the host, thrashing 0.475 CPU. 2 workers: one handles TCP accept/read/write while the other processes KNN. Work-stealing balances automatically. The IVF index is `Arc`-shared, read-only — zero locking, fully thread-safe.

- [ ] **Step 3: Verify compilation**

```bash
cargo check
```
Expected: no errors.

- [ ] **Step 4: Run integration tests**

```bash
cargo test --test integration
```
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "perf(runtime): switch to multi_thread with worker_threads(2)"
```

---

### Task 2: Replace HashSet<String> with Vec<String> in hot path

**Context:** `known_merchants` is built every request and used for a single `contains()` call. `HashSet` allocates a backing hash table + hashes every string. `Vec<String>` with linear `.contains()` is faster for N < ~15 (cache-local, no hashing). Eliminates the largest per-request heap allocation.

**Files:**
- Modify: `src/domain/transaction.rs`
- Modify: `src/service/vectorizer.rs` (unit test fixtures only)
- Modify: `src/web/handlers.rs` (remove collect and import)

- [ ] **Step 1: Update Customer struct in transaction.rs**

Current `src/domain/transaction.rs`:
```rust
use chrono::{DateTime, Utc};
use std::collections::HashSet;

pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: HashSet<String>,
}
```

Replace entirely with:
```rust
use chrono::{DateTime, Utc};

pub struct Transaction {
    pub id: String,
    pub amount: f32,
    pub installments: u32,
    pub requested_at: DateTime<Utc>,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

pub struct LastTransaction {
    pub timestamp: DateTime<Utc>,
    pub km_from_current: f32,
}
```

Changes: removed `use std::collections::HashSet;`, changed `known_merchants: HashSet<String>` → `Vec<String>`.

- [ ] **Step 2: Update vectorizer.rs unit test fixtures**

In `src/service/vectorizer.rs`, find the two test fixture functions (`legit_tx` and `fraud_tx`). They use `HashSet::from([...])`. Replace with `vec![...]`.

Find in `legit_tx`:
```rust
known_merchants: HashSet::from(["MERC-003".to_string(), "MERC-016".to_string()]),
```
Replace with:
```rust
known_merchants: vec!["MERC-003".to_string(), "MERC-016".to_string()],
```

Find in `fraud_tx`:
```rust
known_merchants: HashSet::from([
    "MERC-008".to_string(),
    "MERC-007".to_string(),
    "MERC-005".to_string(),
]),
```
Replace with:
```rust
known_merchants: vec![
    "MERC-008".to_string(),
    "MERC-007".to_string(),
    "MERC-005".to_string(),
],
```

Also remove the `use std::collections::HashSet;` import from vectorizer.rs if it's no longer used anywhere else in that file. Run `cargo check` — if unused import warning appears, remove it.

- [ ] **Step 3: Update handlers.rs**

In `src/web/handlers.rs`, `into_transaction()` currently does:
```rust
known_merchants: req
    .customer
    .known_merchants
    .into_iter()
    .collect::<HashSet<_>>(),
```

Replace with:
```rust
known_merchants: req.customer.known_merchants,
```

`req.customer.known_merchants` is already `Vec<String>` (from the DTO deserialization). No collect needed.

Also remove `use std::collections::HashSet;` from the imports at the top of handlers.rs (line 10).

- [ ] **Step 4: Verify compilation and no warnings**

```bash
cargo clippy --all-targets --all-features -- -D warnings
```
Expected: 0 warnings, 0 errors.

- [ ] **Step 5: Run all tests**

```bash
cargo test
```
Expected: all tests pass. The `Vec::contains()` method works identically to `HashSet::contains()` for this use case — vectorizer behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/domain/transaction.rs src/service/vectorizer.rs src/web/handlers.rs
git commit -m "perf(alloc): replace HashSet<String> with Vec<String> for known_merchants"
```

---

### Task 3: Add least_conn to nginx upstream

**Context:** Round-robin sends request N+1 to the backend that just received request N, regardless of whether it has finished. `least_conn` sends to the backend with fewest active connections — so if api1 is under load and api2 is free, api2 absorbs the burst.

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

- [ ] **Step 2: Add least_conn**

Add `least_conn;` as the first directive inside `upstream api_backends {}`:

```nginx
worker_processes 1;

events {
    worker_connections 1024;
}

http {
    upstream api_backends {
        least_conn;
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

- [ ] **Step 3: Commit**

```bash
git add nginx.conf
git commit -m "perf(nginx): add least_conn load balancing"
```

---

### Task 4: Push and update submission

**Context:** This environment cannot run `docker build` commands that invoke `RUN` (veth/networking limitation). The Docker image must be built on a machine with full Docker support. The submission branch must be updated to point to the new image tag.

- [ ] **Step 1: Verify all tests pass**

```bash
cargo test && cargo clippy --all-targets --all-features -- -D warnings && cargo fmt --check
```
Expected: all pass, 0 warnings, clean format.

- [ ] **Step 2: Push main branch**

```bash
git push origin main
```

- [ ] **Step 3: Build and push image (run on machine with Docker)**

On a machine with full Docker support (not this environment):
```bash
git pull origin main
make publish
```

`make publish` runs:
```bash
docker build -t ghcr.io/abeswz/fraud-detection-rinha-backend-2026:$(git rev-parse --short HEAD) .
docker push ghcr.io/abeswz/fraud-detection-rinha-backend-2026:$(git rev-parse --short HEAD)
```

Expected: image built fresh (`--no-cache` not needed since src/ changed → layer cache invalidated) and pushed to GHCR.

- [ ] **Step 4: Update submission branch**

On the same machine:
```bash
make submission
```

Expected: submission branch force-updated with docker-compose.yml pointing to new image tag.

---

## Expected Outcome

| Metric | Before v2 | After v2 |
|---|---|---|
| Tokio workers | 1 (current_thread) | 2 (multi_thread) |
| known_merchants allocation | HashSet (hash table + N hashes) | Vec (contiguous array) |
| nginx balancing | round-robin | least_conn |
| http_errors | 3072 | ~0 |
| p99 | 2001ms (cut) | <100ms |
| final_score | -3832 | >+2000 |
