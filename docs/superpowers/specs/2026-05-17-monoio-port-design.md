# Design: monoio io_uring port for maximum p99 / score

**Date:** 2026-05-17  
**Goal:** Replace axum/tokio HTTP stack with monoio + io_uring, hand-rolled HTTP/JSON parsers, embedded IVF1 index — targeting lowest possible p99 and highest competition score.

---

## Motivation

Current stack (axum + tokio + sonic-rs + chrono + HashMap MCC) achieves p99=0.29ms, score=5819. Reference project using monoio achieves significantly lower p99. Key overhead sources identified:

- axum/tokio reactor on 0.475 CPU cgroup adds scheduling jitter
- serde deserialization builds heap-allocated structs per request
- chrono ISO parsing overhead
- `HashMap<String, f32>` MCC lookup (hash + string alloc)
- `lto = "thin"`, missing `panic = "abort"`, `overflow-checks` enabled
- `Vec<f32>` centroids/blocks: no 32-byte alignment guarantee
- Runtime `is_x86_feature_detected!()` on every KNN call

---

## Architecture

```
nginx → Unix socket
           ↓
  monoio io_uring (1 thread per instance, 0.45 CPU cgroup)
           ↓
  net::http::serve_connection (per connection, no spawn overhead)
    read() → 8KB rx_buf (stack)
    HTTP parse (memchr, zero alloc)
    fraud/json::parse(body) → Payload (stack struct)
    fraud/vector::vectorize(&payload) → [f32; 14] (stack)
    fraud/knn::knn5_ivf(&q, dataset()) → u8
    net::response::http_body_for(count) → &'static [u8]
    iovec push → writev (batch multiple responses per flush)
```

Zero heap allocations on the hot path.

---

## File Structure

```
src/
  main.rs              - monoio runtime bootstrap + UnixListener

  net/
    mod.rs
    http.rs            - connection loop, HTTP parser, writev batching
    response.rs        - 6 prebuilt static HTTP response buffers

  fraud/
    mod.rs
    json.rs            - positional JSON parser → Payload
    vector.rs          - vectorize(&Payload) → [f32; 14]
    knn.rs             - AVX2/FMA knn5 + warmup
    data.rs            - Dataset (OnceLock, embedded IVF1 gzip, AVec)

  env.rs               - SOCK env var reader

bin/
  build_index.rs       - kmeans++ K=4096, writes data/index.bin.gz (IVF1)
```

---

## Component Specs

### `main.rs`
- Call `data::init()` and `knn::warmup()` before binding
- monoio `RuntimeBuilder::<IoUringDriver>` with `entries=1024`
- `UnixListener` on `env::sock_path()`
- `loop { accept → monoio::spawn(http::serve_connection(stream)) }`
- No TCP listener, no FD passing (nginx handles UDS directly)

### `net/http.rs`
- `RX_CAP = 8192` rx buffer per connection (stack-allocated boxed slice)
- `find_header_end`: `memchr::memmem::find(buf, b"\r\n\r\n")`
- `parse_content_length`: `memchr::memchr2(b'c', b'C', ...)` scan, case-insensitive
- Route detection by byte prefix: `POST /fraud-score`, `GET /ready`
- Batch up to `MAX_IOVECS=16` iovec responses before `writev`
- `writev` loop handles partial writes

### `net/response.rs`
- 6 `&'static [u8]` complete HTTP/1.1 200 responses with JSON body
- `fn http_body_for(fraud_count: u8) -> &'static [u8]` indexes into array
- One `RESP_READY`, one `RESP_NOT_FOUND`, one `RESP_BAD_REQ` static buffer

### `fraud/json.rs`
- `Payload` struct: all fields on stack (~80 bytes)
- Positional parsing — advances through fixed field order, no field-name matching
- `memchr::memchr` / `memchr2` for `"` and `:` navigation
- `is_unknown_merchant` computed inline via slice comparison (no Vec alloc)
- ISO datetime → `(y: u16, mo: u8, d: u8, h: u8, min: u8)` tuple, manual digit extraction
- `minutes_between`: manual days-since-epoch arithmetic (no chrono)
- Returns `Option<Payload>` — `None` on malformed input falls back to `RESP_FRAUD_FALLBACK`

### `fraud/vector.rs`
- `round4(x: f32) -> f32 { (x * 10000.0).round() * 0.0001 }` applied to all float features
- MCC lookup: `match mcc: u32 { 5411 => 0.15, 5812 => 0.30, ... _ => 0.50 }` — no HashMap, no alloc
- Returns `[f32; 14]` on stack

### `fraud/knn.rs`
- `#[target_feature(enable = "avx2,fma")]` on all SIMD functions — compile-time, no runtime cpuid
- `FAST_NPROBE: usize = 5`, `FULL_NPROBE: usize = 24`
- Adaptive: run fast probe → if `fraud_count == 2 || fraud_count == 3` → run full probe
- `compute_centroid_dists`: column-major layout, 16-wide unrolled AVX2 loop
- `top_n_from_dists<N>`: AVX2 mask-based top-N selection
- `scan_blocks`: FMA distance accumulation, partial early-exit after 8 dims, prefetch 8 blocks ahead
- `warmup()`: 500 pseudo-random queries before serving traffic

### `fraud/data.rs`
- `static INDEX_GZ: &[u8] = include_bytes!("../../data/index.bin.gz")`
- `static DATASET: OnceLock<Dataset>`
- `Dataset.centroids`: `AVec<f32, ConstAlign<32>>` — 32-byte aligned for SIMD
- `Dataset.blocks`: `AVec<i16, ConstAlign<32>>` — 32-byte aligned
- IVF1 format: magic `b"IVF1"`, n, k, d=14, centroids, offsets, labels, blocks

### `bin/build_index.rs`
- Reads `resources/references.json.gz`
- Applies same `vectorize` + `round4` as runtime (must match exactly)
- kmeans++ init (sample 50k), 25 Lloyd iterations
- K=4096 centroids, column-major centroid layout
- Writes `data/index.bin.gz` (IVF1, gzip best compression)
- Parallel assignment via `std::thread::scope`

---

## Build Configuration

### `Cargo.toml`

**Dependencies:**
```toml
[dependencies]
monoio      = { version = "0.2", default-features = false, features = ["iouring", "legacy", "macros"] }
memchr      = "2"
mimalloc    = { version = "0.1", default-features = false }
aligned-vec = "0.6"
flate2      = { version = "1", default-features = false, features = ["rust_backend"] }
libc        = "0.2"
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
```

Remove: `axum`, `tokio`, `sonic-rs`, `chrono`, `smallvec`, `tracing`, `tracing-subscriber`, `tower-http`, `dotenvy`, `half`

**Release profile:**
```toml
[profile.release]
opt-level        = 3
lto              = "fat"
codegen-units    = 1
strip            = true
panic            = "abort"
debug            = 0
overflow-checks  = false
incremental      = false

[profile.release.package."*"]
opt-level = 3
```

---

## Dockerfile (multi-stage, index embedded)

```dockerfile
FROM rust:latest AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin build_index
RUN ./target/release/build_index
RUN cargo build --release --bin fraud-detection

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/fraud-detection /fraud-detection
CMD ["/fraud-detection"]
```

The `build_index` step writes `data/index.bin.gz`. The second `cargo build` embeds it via `include_bytes!`. Index is never read from disk at runtime.

---

## `docker-compose.yml` changes

For each API service:
```yaml
environment:
  - SOCK=/run/sock/api1.sock   # replaces PORT + SOCKET_PATH
security_opt:
  - seccomp:unconfined          # required: io_uring syscalls blocked by default Docker seccomp
ulimits:
  nofile:
    soft: 65535
    hard: 65535
```

Remove `PORT` env var — no TCP listener in new implementation.

**Readiness check:** nginx startup command currently does `wget http://api1:3000/ready` — fails with no TCP listener. Replace with socket existence check:
```sh
until [ -S /run/sock/api1.sock ] && [ -S /run/sock/api2.sock ]; do sleep 1; done
```
`data::init()` + `knn::warmup()` run before `UnixListener::bind`, so socket presence guarantees server is ready.

**JSON field order** verified against `resources/example-payloads.json` — matches reference parser order exactly. Positional parsing is safe.

---

## What Gets Deleted

- `src/domain/`
- `src/service/`
- `src/repository/`
- `src/usecase/`
- `src/web/`
- `src/config.rs`
- `src/error.rs`
- `src/lib.rs`
- `bin/preprocess.rs`

---

## Tests

Targeted unit tests only — no integration framework needed:

| File | What to test |
|------|-------------|
| `fraud/json.rs` | Parse known payloads from `resources/example-payloads.json`, assert field values |
| `fraud/vector.rs` | Known input → expected `[f32; 14]` output (with round4) |
| `fraud/knn.rs` | Smoke: warmup + query returns u8 in 0..=5 |
| `net/response.rs` | All 6 bodies are valid JSON with `approved` + `fraud_score` |

No mocks. No axum-test. No once_cell.
