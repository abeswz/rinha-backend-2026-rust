# sonic-rs JSON Throughput + spawn_blocking Removal

**Date:** 2026-05-17
**Goal:** Reduce per-request CPU overhead at 0.45 vCPU by replacing serde_json with sonic-rs for request parsing and eliminating spawn_blocking scheduling overhead.

---

## Context

Local bench (throttled via `--compatibility`): p99 = 0.31ms, score ~5800.
Remote (competition VM, 0.45 vCPU): p99 = 88ms, score ~3873.
Ratio: ~284x.

Root cause: load ramps to 900 req/s (450/instance at peak). At 0.45 vCPU the full request pipeline saturates CPU. Two identified costs:

1. **serde_json request parsing**: complex nested JSON with `DateTime<Utc>`, `Vec<String>`, optional fields. Estimated 0.5–1ms/req on competition VM.
2. **spawn_blocking scheduling overhead**: each request dispatches KNN to a blocking thread pool, incurring ~50–200μs of thread scheduling cost under CPU cgroup throttle. At 450 req/s this wastes ~20% of the available CPU budget.

Response path already eliminated via `STATIC_BODIES`. This spec addresses the request path.

---

## Changes

### 1. sonic-rs Custom Extractor

**New file:** `src/web/extractors.rs`

Implements `SonicJson<T>: axum::extract::FromRequest` backed by `sonic_rs::from_slice`. Reads request body as raw bytes, parses with sonic-rs SIMD JSON parser (AVX2-accelerated, confirmed available on competition server). Returns HTTP 422 on parse failure — identical error behavior to axum's `Json<T>`.

**Why sonic-rs over simd-json:**
- AVX2 confirmed on competition server (binary uses `RUSTFLAGS="+avx2"` and runs without SIGILL)
- 2–4× faster than serde_json vs simd-json's 1.5–3×
- Works with existing `serde::Deserialize` derive macros — no DTO changes needed

**Cargo.toml:** add `sonic-rs = "0.3"`

**`src/web/handlers.rs`:** replace `Json(req): Json<TransactionRequest>` with `SonicJson(req): SonicJson<TransactionRequest>`.

**DTOs (`src/web/dto.rs`):** unchanged. `#[derive(Deserialize)]` is compatible with sonic-rs.

### 2. Remove spawn_blocking

**Current flow:**
```
async worker → tokio::time::timeout → spawn_blocking → blocking thread → KNN → async worker
```

**New flow:**
```
async worker → KNN (inline, sync) → response
```

KNN is deterministic O(K × D), completes in <1ms locally and estimated <10ms on competition VM under throttle. Running inline on the async worker is safe — it does not block I/O (no syscalls, no network), and the compute window is short enough that it does not starve the Tokio reactor.

Removes the `tokio::time::timeout` wrapper (no longer meaningful without spawn_blocking; KNN cannot hang).

The timeout fallback (`FraudDecision { approved: true, fraud_count: 0 }`) is no longer reachable — remove it from the handler.

**`src/main.rs`:**
- `TOKIO_MAX_BLOCKING_THREADS`: 2 → 1 (only startup I/O remains; index load on `AppState::build`)
- `TOKIO_WORKER_THREADS`: unchanged at 2
- Update asserts and comments to reflect new rationale

**Tradeoff:** if KNN ever exceeds ~20ms on a worker thread (would require a bug or regression), it delays accept/parse on that worker. Acceptable: KNN is bounded, deterministic, and regression-protected by existing tests.

---

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` | Add `sonic-rs = "0.3"` |
| `src/web/extractors.rs` | New — `SonicJson<T>` axum extractor |
| `src/web/handlers.rs` | Use `SonicJson`, remove spawn_blocking + timeout |
| `src/web/mod.rs` | Expose `extractors` module |
| `src/main.rs` | `TOKIO_MAX_BLOCKING_THREADS` 2→1, update comments |

---

## Bench Calibration Methodology

Local and remote benchmarks will not produce matching absolute numbers due to hardware differences (user machine ~4GHz vs competition VM ~2GHz + hypervisor overhead). Use **relative improvement** as the signal.

**Current calibration point (code before this spec):**
- Local p99: 0.31ms
- Remote p99: 88ms
- Ratio: 284×

**Rule:** if local p99 drops by X%, expect remote p99 to drop by approximately X%. Do not try to predict remote ms from local ms directly.

**Recalibration required when:** the dominant bottleneck changes (e.g., after this spec eliminates JSON parsing as the bottleneck, the ratio will shift because SIMD on the local machine outperforms the competition VM more aggressively for KNN than for JSON).

Track ratio after each significant submit to keep the calibration current.

---

## Testing

- All existing unit, integration, and regression tests must pass unchanged (no DTO changes, KNN behavior unchanged)
- Add test to `src/web/extractors.rs`: valid JSON parses correctly, malformed JSON returns 422
- `cargo clippy --all-targets --all-features -- -D warnings`: no errors
- `make bench` after changes: local p99 expected to drop from 0.31ms (baseline is already fast locally; the gain will be visible at throttled bench or remote)

---

## Expected Outcome

- Request parsing: ~2–4× faster (sonic-rs vs serde_json)
- Scheduling overhead: eliminated (~50–200μs/req saved)
- Combined: meaningfully reduces CPU saturation at 450 req/s peak
- Remote p99: expected to drop from 88ms, magnitude to be confirmed via submit
- Score: p99_score component (currently 1052/3000) expected to improve toward 3000 (max)
