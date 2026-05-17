# sonic-rs JSON Throughput + spawn_blocking Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace serde_json request parsing with sonic-rs (AVX2 SIMD, 2–4× faster) and eliminate spawn_blocking scheduling overhead (~50–200μs/req) by running KNN inline on async workers.

**Architecture:** New `SonicJson<T>` axum extractor wraps `sonic_rs::from_slice`, returning HTTP 422 on failure — identical behavior to axum's `Json<T>`. Handler calls `use_case.execute()` synchronously, removing the `tokio::time::timeout` + `spawn_blocking` wrapping. `TOKIO_MAX_BLOCKING_THREADS` drops to 1 (only startup I/O remains).

**Tech Stack:** Rust, axum 0.8, sonic-rs 0.3, tokio, serde

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `Cargo.toml` | Modify | Add sonic-rs dependency |
| `src/web/extractors.rs` | Create | `SonicJson<T>` + `SonicJsonRejection` types + unit tests |
| `src/web/mod.rs` | Modify | Expose `extractors` module |
| `src/web/handlers.rs` | Modify | Use SonicJson, remove spawn_blocking + timeout + stale tests |
| `src/main.rs` | Modify | `TOKIO_MAX_BLOCKING_THREADS` 2→1, update comment + assert |

---

## Task 1: Add sonic-rs dependency + declare extractors module

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/web/mod.rs`

- [ ] **Step 1: Add sonic-rs to Cargo.toml**

In `[dependencies]`, add after the `serde_json` line:

```toml
sonic-rs = "0.3"
```

Full diff context:
```toml
serde_json = "1"
sonic-rs = "0.3"
```

- [ ] **Step 2: Add extractors module to `src/web/mod.rs`**

Current content:
```rust
pub mod dto;
pub mod handlers;
pub mod router;
```

Replace with:
```rust
pub mod dto;
pub mod extractors;
pub mod handlers;
pub mod router;
```

- [ ] **Step 3: Verify sonic-rs downloads and compiles**

```bash
cargo check 2>&1 | head -20
```

Expected: compile error about missing `src/web/extractors.rs` file (module declared but not found). This is correct — we add it next.

---

## Task 2: Write failing extractor tests (TDD)

**Files:**
- Create: `src/web/extractors.rs`

- [ ] **Step 1: Create extractors.rs with tests only (no implementation)**

Create `src/web/extractors.rs` with this content:

```rust
use axum::{
    body::Bytes,
    extract::{FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;

pub struct SonicJson<T>(pub T);

pub struct SonicJsonRejection;

impl IntoResponse for SonicJsonRejection {
    fn into_response(self) -> Response {
        todo!()
    }
}

impl<T, S> FromRequest<S> for SonicJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = SonicJsonRejection;

    async fn from_request(_req: Request, _state: &S) -> Result<Self, Self::Rejection> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejection_is_422() {
        let resp = SonicJsonRejection.into_response();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn test_sonic_parses_valid_json() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Simple {
            x: u32,
        }
        let bytes = b"{\"x\": 42}";
        let result: Result<Simple, _> = sonic_rs::from_slice(bytes);
        assert_eq!(result.unwrap(), Simple { x: 42 });
    }

    #[test]
    fn test_sonic_rejects_malformed_json() {
        #[derive(serde::Deserialize)]
        struct Simple {
            x: u32,
        }
        let result: Result<Simple, _> = sonic_rs::from_slice(b"not json");
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Run the tests to confirm they panic (todo!)**

```bash
cargo test web::extractors 2>&1 | tail -20
```

Expected: tests fail/panic with `not yet implemented`. The two sonic-rs tests (`test_sonic_parses_valid_json`, `test_sonic_rejects_malformed_json`) will pass immediately since they test `sonic_rs::from_slice` directly. `test_rejection_is_422` will panic at `todo!()`.

---

## Task 3: Implement SonicJson extractor

**Files:**
- Modify: `src/web/extractors.rs`

- [ ] **Step 1: Implement `SonicJsonRejection::into_response` and `FromRequest`**

Replace the `todo!()` stubs. Full final content of `src/web/extractors.rs`:

```rust
use axum::{
    body::Bytes,
    extract::{FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;

pub struct SonicJson<T>(pub T);

pub struct SonicJsonRejection;

impl IntoResponse for SonicJsonRejection {
    fn into_response(self) -> Response {
        StatusCode::UNPROCESSABLE_ENTITY.into_response()
    }
}

impl<T, S> FromRequest<S> for SonicJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = SonicJsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(|_| SonicJsonRejection)?;
        sonic_rs::from_slice(&bytes)
            .map(SonicJson)
            .map_err(|_| SonicJsonRejection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejection_is_422() {
        let resp = SonicJsonRejection.into_response();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn test_sonic_parses_valid_json() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Simple {
            x: u32,
        }
        let bytes = b"{\"x\": 42}";
        let result: Result<Simple, _> = sonic_rs::from_slice(bytes);
        assert_eq!(result.unwrap(), Simple { x: 42 });
    }

    #[test]
    fn test_sonic_rejects_malformed_json() {
        #[derive(serde::Deserialize)]
        struct Simple {
            x: u32,
        }
        let result: Result<Simple, _> = sonic_rs::from_slice(b"not json");
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Run extractor tests to verify all pass**

```bash
cargo test web::extractors 2>&1
```

Expected output:
```
test web::extractors::tests::test_rejection_is_422 ... ok
test web::extractors::tests::test_sonic_parses_valid_json ... ok
test web::extractors::tests::test_sonic_rejects_malformed_json ... ok
```

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock src/web/mod.rs src/web/extractors.rs
git commit -m "feat: add SonicJson extractor backed by sonic-rs SIMD parser"
```

---

## Task 4: Update handler — replace Json with SonicJson, remove spawn_blocking

**Files:**
- Modify: `src/web/handlers.rs`

- [ ] **Step 1: Rewrite `src/web/handlers.rs`**

Replace the entire file content with:

```rust
use crate::AppState;
use crate::{
    domain::transaction::{Customer, LastTransaction, Merchant, Terminal, Transaction},
    web::dto::TransactionRequest,
    web::extractors::SonicJson,
};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
};
use std::sync::Arc;

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
    SonicJson(req): SonicJson<TransactionRequest>,
) -> impl IntoResponse {
    let tx = into_transaction(req);
    let decision = state.use_case.execute(&tx);

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

    #[test]
    fn test_static_bodies_all_valid_json() {
        for (i, body) in STATIC_BODIES.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(body)
                .unwrap_or_else(|_| panic!("body[{i}] is not valid JSON: {body}"));
            assert!(v.get("approved").is_some(), "body[{i}] missing 'approved'");
            assert!(v.get("fraud_score").is_some(), "body[{i}] missing 'fraud_score'");
        }
    }
}
```

**What changed vs before:**
- Removed imports: `axum::Json`, `crate::domain::fraud::FraudDecision`, `std::time::Duration`
- Added import: `crate::web::extractors::SonicJson`
- `fraud_score_handler`: replaced `Json(req): Json<TransactionRequest>` with `SonicJson(req): SonicJson<TransactionRequest>`
- `fraud_score_handler`: replaced `tokio::time::timeout(... spawn_blocking(...))` with direct `state.use_case.execute(&tx)`
- Removed tests that tested the spawn_blocking/timeout mechanism (now-deleted code): `test_timeout_fallback_is_approved_true`, `test_fast_execution_returns_actual_decision`

- [ ] **Step 2: Run all tests**

```bash
cargo test 2>&1
```

Expected: all tests pass. Integration tests in `tests/integration.rs` already cover 422 on malformed/missing fields — these continue to pass because `SonicJson` returns the same HTTP 422 status.

- [ ] **Step 3: Run clippy**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1
```

Expected: no warnings or errors.

- [ ] **Step 4: Commit**

```bash
git add src/web/handlers.rs
git commit -m "perf: replace Json extractor with SonicJson, remove spawn_blocking"
```

---

## Task 5: Update main.rs — reduce blocking thread pool to 1

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Update `TOKIO_MAX_BLOCKING_THREADS` constant and its comment**

In `src/main.rs`, locate these lines (around line 9–13):

```rust
// Under 0.475 CPU cgroup (competition constraint): more threads = cgroup throttling = reactor starvation.
// These constants caused p99=2001ms + 1986 HTTP errors when WORKER=4 + unbounded blocking pool.
// Under 0.475 CPU cgroup per instance: WORKER=4 + unbounded blocking caused p99=2001ms + 1986 HTTP errors.
// Little's Law at peak: ~450 req/s × 5ms KNN ≈ 2.25 concurrent → 2 blocking threads is the right bound.
const TOKIO_WORKER_THREADS: usize = 2;
const TOKIO_MAX_BLOCKING_THREADS: usize = 2;
```

Replace with:

```rust
// Under 0.475 CPU cgroup (competition constraint): more threads = cgroup throttling = reactor starvation.
// These constants caused p99=2001ms + 1986 HTTP errors when WORKER=4 + unbounded blocking pool.
// Under 0.475 CPU cgroup per instance: WORKER=4 + unbounded blocking caused p99=2001ms + 1986 HTTP errors.
// KNN runs inline on async workers (no spawn_blocking). Only startup I/O (AppState::build) uses blocking threads.
const TOKIO_WORKER_THREADS: usize = 2;
const TOKIO_MAX_BLOCKING_THREADS: usize = 1;
```

- [ ] **Step 2: Update the test assert in `src/main.rs`**

Locate the test (around line 60–68):

```rust
#[test]
fn test_tokio_runtime_config_bounded_for_cgroup_throttling() {
    // 0.475 CPU cgroup per instance: each extra thread steals from reactor.
    // WORKER=4 + unbounded blocking → p99=2001ms + 1986 HTTP errors in competition.
    assert_eq!(TOKIO_WORKER_THREADS, 2, "worker_threads must be 2 on 0.475 CPU cgroup");
    assert_eq!(
        TOKIO_MAX_BLOCKING_THREADS, 2,
        "max_blocking_threads must be 2: unbounded blocking pool starves Tokio reactor under cgroup throttling"
    );
}
```

Replace with:

```rust
#[test]
fn test_tokio_runtime_config_bounded_for_cgroup_throttling() {
    // 0.475 CPU cgroup per instance: each extra thread steals from reactor.
    // WORKER=4 + unbounded blocking → p99=2001ms + 1986 HTTP errors in competition.
    assert_eq!(TOKIO_WORKER_THREADS, 2, "worker_threads must be 2 on 0.475 CPU cgroup");
    assert_eq!(
        TOKIO_MAX_BLOCKING_THREADS, 1,
        "max_blocking_threads must be 1: KNN runs inline, only startup I/O needs a blocking thread"
    );
}
```

- [ ] **Step 3: Run all tests**

```bash
cargo test 2>&1
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "perf: reduce blocking thread pool to 1, KNN now runs inline"
```

---

## Self-Review Checklist

- [x] **sonic-rs = "0.3"** in Cargo.toml → Task 1
- [x] **`src/web/extractors.rs`** with `SonicJson<T>: FromRequest` backed by `sonic_rs::from_slice` → Task 3
- [x] **HTTP 422 on parse failure** (identical to axum `Json<T>`) → `SonicJsonRejection` returns `UNPROCESSABLE_ENTITY` → Task 3
- [x] **`src/web/mod.rs`** exposes `extractors` module → Task 1
- [x] **`handlers.rs`** uses `SonicJson`, drops spawn_blocking + timeout → Task 4
- [x] **Timeout fallback removed** (no longer reachable) → Task 4
- [x] **`TOKIO_MAX_BLOCKING_THREADS` 2→1** → Task 5
- [x] **Tests for extractor**: rejection 422, valid parse, malformed parse → Task 3
- [x] **Stale handler tests removed**: `test_timeout_fallback_is_approved_true`, `test_fast_execution_returns_actual_decision` — these test the spawn_blocking mechanism that is deleted → Task 4
- [x] **`test_static_bodies_all_valid_json` kept** — tests STATIC_BODIES content, independent of dispatch mechanism
- [x] **Integration tests unchanged** — they hit HTTP endpoints and verify 422/200 behavior; SonicJson returns same status codes as axum's Json
- [x] **No placeholder steps** — all code shown in full
- [x] **Type consistency** — `SonicJson<T>` named consistently across extractors.rs and handlers.rs
