## Objective

High-performance fraud detection API in Rust. Minimal allocations, sub-millisecond p99.

> **Manutenção obrigatória:** Após cada teste remoto ou mudança arquitetural relevante, atualizar `PROGRESS.md`.
> Esse arquivo é o recap rápido do estado do projeto — mantê-lo atualizado evita ter que reconstruir contexto do zero.

---

## Stack

- **tokio** (raw, minimal features: rt, rt-multi-thread, macros, net, io-util)
- **mimalloc** — global allocator
- **memchr** — fast byte scanning in HTTP parser
- **aligned-vec, flate2** — IVF index loading
- **serde, serde_json** — request deserialization only
- **libc** — low-level OS calls

No Axum. No Reqwest. No tracing. No dotenvy.

---

## Architecture

**Transport**: Unix socket → custom HTTP/1.1 parser → keep-alive connections.
No HTTP framework. Responses are pre-baked static `&[u8]` slices (6 levels).

**Fraud pipeline** (per request):
1. Deserialize JSON → build 14-dim f32 feature vector
2. IVF KNN search (K=4096 centroids, NPROBE=5 fast / 24 full)
3. Count fraud labels among top-5 neighbors → return static response

**Model** (`fraud/model.rs`): m2cgen-generated inline Rust (LightGBM, <50µs).
Currently compiled in but **fast-path disabled** — all requests route to IVF to avoid false positives.

**Binaries**:
- `src/main.rs` — HTTP server (2 worker threads, 2 blocking threads)
- `bin/lb.rs` — custom load balancer
- `bin/build_index.rs` — IVF index builder

**ML tools** (Python, run offline):
- `tools/train_model.py` — LightGBM training + m2cgen export
- `tools/build_ivf.py` — builds `resources/ivf_index.bin`

---

## Engineering Priorities

1. correctness
2. simplicity
3. performance
4. maintainability

---

## Constraints

- 1 CPU, 350MB RAM
- No unnecessary allocations, clones, Arc, or Mutex
- No blocking calls inside async
- Fraud logic independent from HTTP transport
- Deterministic — no randomness, no hidden global mutable state

---

## Release Profile

LTO fat, codegen-units=1, strip, panic=abort, overflow-checks=false.

---

## Testing

Every core feature needs unit + integration + regression tests.
Tests must cover real fraud scenarios: duplicate patterns, velocity spikes, malformed requests, invalid timestamps.

---

## Commands

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```
