## Objective

Build a minimal, high-performance fraud detection API in Rust.

The service:
1. receives transaction payloads
2. processes fraud comparison vectors
3. computes fraud risk
4. returns structured JSON responses

Primary goals:
- low latency
- low memory usage
- predictable performance
- maintainability

---

## Stack

- Axum
- Tokio
- Reqwest
- Serde
- dotenvy
- tracing

Do not replace the stack unless explicitly requested.

---

## Engineering Priorities

Priority order:
1. correctness
2. simplicity
3. performance
4. maintainability
5. extensibility

---

## Memory Constraints

Target runtime:
- 1 CPU
- 350MB RAM

Optimize for:
- low allocations
- reduced cloning
- bounded memory growth
- async I/O efficiency

---

## Fraud Engine Rules

Fraud logic must:
- be deterministic
- avoid randomness
- avoid hidden mutable global state
- support reproducible tests

Core logic must be independent from HTTP transport.

---

## API Guidelines

Use:
- typed request DTOs
- typed response DTOs
- structured errors
- HTTP status consistency

Avoid:
- generic string errors
- dynamic maps
- untyped JSON processing

---

## Testing Expectations

Every core feature requires:
- unit tests
- integration tests
- regression tests

Important:
Tests must validate real fraud scenarios.

Examples:
- duplicated transaction patterns
- abnormal velocity
- suspicious value spikes
- malformed requests
- invalid timestamps

---

## Performance Guidelines

Prefer:
- iterators
- slices
- borrowing
- preallocated buffers when justified

Avoid:
- unnecessary Arc usage
- unnecessary Mutex usage
- blocking calls inside async
- excessive cloning

---

## Logging

Use `tracing`.

Requirements:
- structured logs
- request IDs
- error context

Never log:
- secrets
- full sensitive payloads

---

## Deliverables

All generated code must:
- compile
- pass tests
- be formatted
- pass clippy
- avoid warnings

---

## Commands

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```
