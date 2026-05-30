# Clean Redesign — Fraud Detection

**Date**: 2026-05-30  
**Goal**: p99 < 1ms remote, score 6000 (or close). Clean, minimal implementation.

---

## Context

Best remote result to date: `a586fa8`, nginx, pure IVF, 0 FP/FN → **84ms p99, score 4071**.

Key regressions identified:
- Model fast-path (m2cgen, 1.65MB) added → **98ms p99** (worse). Cause: I-cache pollution.
- Custom Rust LB replaced nginx → **146ms p99**. Cause: new Unix socket per request, no keepalive.
- `spawn_blocking(max=2)` serializes IVF: only 2 concurrent IVF computations. Under burst, queueing dominates p99.

---

## Architecture

```
k6 (port 9999)
    │ TCP
HAProxy (mode tcp, roundrobin)
    │ Unix socket
    ├── /run/sock/api1.sock → api1 (0.45 CPU, 172MB)
    └── /run/sock/api2.sock → api2 (0.45 CPU, 172MB)
```

**HAProxy in TCP mode**: no HTTP parsing, raw byte forwarding, Unix socket upstream. Lower overhead than nginx HTTP mode. Per-connection balancing (fine for stateless API).

**API instances**: 2 (same binary, different `SOCK` env vars).

---

## Runtime Configuration

```rust
tokio::runtime::Builder::new_multi_thread()
    .worker_threads(4)
    .thread_stack_size(512 * 1024)  // 512KB vs 2MB default
    .enable_all()
    .build()
```

- **4 workers**: 4 concurrent inline IVF computations. Under 250 VU burst, new requests accepted by non-blocked workers. No blocking pool needed.
- **512KB stacks**: IVF stack usage <5KB. Saves 6MB vs default 2MB stacks.
- **No `max_blocking_threads`**: Blocking pool removed. IVF runs inline.

---

## Request Hot Path

```
accept (async) → spawn(serve_connection)
    → read HTTP (async, keep-alive loop)
    → parse_headers → detect route
    → read body
    → json::parse → vector::vectorize
    → knn::knn5_ivf (inline, no spawn_blocking)
    → write static &[u8] response
```

**Zero allocations** in the hot path:
- JSON parser: stack-only (arrays on stack, no Vec)
- IVF: thread-local DISTS buffer (pre-allocated), stack vars only
- Response: static `&[u8]` slices

---

## IVF (knn.rs) Changes

Remove from hot path:
- `Instant::now()` / elapsed timing (was called on every request)
- Metrics counters (removed module)

Remove dead code:
- `centroid_dists_scalar` (x86_64 always uses AVX2)
- `probe_scalar` and `scan_blocks_scalar` (non-x86_64 paths, remote is x86_64)

Keep:
- Adaptive probe: NPROBE=5 fast → NPROBE=24 full for ambiguous (fraud_count==2||3). Helps 0 FP/FN on dataset 797.
- AVX2+FMA centroid distance, AVX2 block scan, prefetch hints.

---

## JSON Parser (json.rs)

Merge `parse_full` and `parse_positional` into a single `parse` function.

**Why two existed**: `parse_positional` skips brace navigation (faster), `parse_full` falls back to it. In practice, both parsers are positional (rely on field order), not key-name based. The "fallback" never adds safety on malformed input; both fail together.

Single implementation: use `parse_positional` logic (the faster one), drop `parse_full`. Removes ~250 duplicate lines.

---

## Files Deleted

| File | Reason |
|---|---|
| `src/fraud/model.rs` | Proven to hurt p99 via I-cache; not needed |
| `src/fraud/model_gen.rs` | 1.65MB LightGBM tree; I-cache killer |
| `src/metrics.rs` | Debug infra, not required by problem spec |
| `bin/lb.rs` | Replaced by HAProxy |

---

## Files Modified

| File | Change summary |
|---|---|
| `src/main.rs` | worker_threads=4, stack=512KB, remove model::init() and model::warmup() calls |
| `src/fraud/mod.rs` | Remove `pub mod model` |
| `src/fraud/knn.rs` | Remove Instant, metrics, scalar fallbacks (~80 lines) |
| `src/fraud/json.rs` | Merge to single parse(), remove parse_full (~250 lines) |
| `src/net/http.rs` | Remove model path, remove Route::Metrics, remove metrics counters |
| `docker-compose.yml` | lb service: `haproxy:alpine`, mount `haproxy.cfg` |

---

## New File: haproxy.cfg

```haproxy
global
    maxconn 500
    tune.bufsize 4096
    log /dev/null local0

defaults
    mode tcp
    timeout connect 50ms
    timeout client 30s
    timeout server 30s

frontend ft
    bind *:9999
    default_backend api

backend api
    balance roundrobin
    server api1 unix@/run/sock/api1.sock
    server api2 unix@/run/sock/api2.sock
```

---

## Memory Budget (per API instance)

| Component | Size |
|---|---|
| Binary (code + 30.8MB gzip index) | ~36MB |
| Decompressed IVF index | ~87MB |
| 4 threads × 512KB stacks | 2MB |
| Thread-local DISTS (4 × 16KB) | 64KB |
| Tokio runtime + misc | ~5MB |
| **Total estimated** | **~130MB** |
| Limit | 172MB |
| **Headroom** | **~42MB** ✓ |

---

## Docker Compose Resource Allocation

| Service | CPU | Memory |
|---|---|---|
| haproxy | 0.05 | 6MB |
| api1 | 0.45 | 172MB |
| api2 | 0.45 | 172MB |
| **Total** | **0.95** | **350MB** |

---

## Expected Impact

| Metric | Before (a586fa8) | After |
|---|---|---|
| Model overhead | ~14ms extra p99 | Removed |
| spawn_blocking limit | 2 concurrent IVF | 4 concurrent inline |
| LB | nginx (HTTP) | HAProxy (TCP) |
| Dead code | ~400 lines | Removed |
| Remote p99 estimate | 84ms | 10–30ms (optimistic: sub-1ms) |

The sub-0.9ms target is achievable if the remote hardware allows per-request service time < 0.22ms. This requires validation by remote test. The design removes all known overhead sources; remaining latency is hardware-bound.

---

## Testing Plan

1. `cargo test` — all existing tests must pass
2. `make bench` — local p99 should stay ≤ 0.21ms, score = 6000
3. Remote submission — target improvement from 84ms
4. If p99 > 10ms remotely: investigate with `GET /metrics` → removed, add back if needed
