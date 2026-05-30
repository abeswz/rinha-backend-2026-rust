# Clean Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all dead code and overhead (model, metrics, custom LB, spawn_blocking) and replace with inline IVF + 4 tokio workers + HAProxy TCP mode.

**Architecture:** Unix socket → HAProxy TCP → 2 API instances. Each instance: 4 tokio workers, inline IVF (no spawn_blocking), 512KB stacks. Hot path: zero allocations, static response slices.

**Tech Stack:** Rust/Tokio (custom HTTP), AVX2+FMA IVF, HAProxy alpine, mimalloc.

---

## File Map

| Action | File | Change |
|--------|------|--------|
| Modify | `src/fraud/knn.rs` | Remove Instant, metrics refs, scalar dead code |
| Modify | `src/net/http.rs` | Inline IVF, remove model path + Metrics route + metrics refs |
| Modify | `src/fraud/json.rs` | Delete parse_full, rename parse_positional → parse |
| Modify | `src/fraud/mod.rs` | Remove `pub mod model` |
| Modify | `src/main.rs` | worker_threads=4, stack=512KB, remove mod metrics + model init |
| Modify | `Cargo.toml` | Remove `[[bin]] lb`, remove serde/serde_json deps |
| Modify | `Dockerfile` | Remove lb binary from build + copy |
| Modify | `docker-compose.yml` | lb service → haproxy:alpine |
| Delete | `src/fraud/model.rs` | Proven I-cache killer |
| Delete | `src/fraud/model_gen.rs` | 1.65MB LightGBM tree |
| Delete | `src/metrics.rs` | Debug infra only |
| Delete | `bin/lb.rs` | Replaced by HAProxy |
| Create | `haproxy.cfg` | TCP mode, Unix socket upstreams |

---

### Task 1: Strip knn.rs — remove Instant, metrics, scalar dead code

**Files:**
- Modify: `src/fraud/knn.rs`

- [ ] **Step 1: Verify test currently passes**

```bash
cargo test fraud::knn -- --nocapture 2>&1 | tail -20
```
Expected: all knn tests PASS.

- [ ] **Step 2: Delete the `fast_probe_counter_increments` test**

In `src/fraud/knn.rs`, delete this entire test (lines ~345–354):

```rust
    #[test]
    fn fast_probe_counter_increments() {
        use crate::metrics;
        use std::sync::atomic::Ordering::Relaxed;

        data::init();
        let before = metrics::FAST_PROBE_COUNT.load(Relaxed);
        let _ = knn5_ivf(&[0.0f32; 14], data::dataset());
        let after = metrics::FAST_PROBE_COUNT.load(Relaxed);
        assert!(after > before, "FAST_PROBE_COUNT must increase after knn5_ivf call");
    }
```

- [ ] **Step 3: Rewrite `knn5_ivf` to remove Instant and metrics**

Replace the current `knn5_ivf` function body (lines 12–34) with:

```rust
pub fn knn5_ivf(q: &[f32; 14], ds: &Dataset) -> u8 {
    let fast = probe(q, ds, FAST_NPROBE);
    let fraud_count = count_fraud(fast);
    if fraud_count == 2 || fraud_count == 3 {
        let full = probe(q, ds, FULL_NPROBE);
        count_fraud(full) as u8
    } else {
        fraud_count as u8
    }
}
```

- [ ] **Step 4: Delete `centroid_dists_scalar` (lines ~93–111)**

Remove the entire function:

```rust
#[allow(dead_code)]
fn centroid_dists_scalar(q: &[f32; 14], centroids: *const f32, dists: *mut f32) {
    for i in 0..K {
        unsafe {
            *dists.add(i) = 0.0f32;
        }
    }
    // ...
}
```

- [ ] **Step 5: Delete `probe_scalar` and `scan_blocks_scalar`**

Remove both functions gated with `#[cfg(not(target_arch = "x86_64"))]`:
- `fn probe_scalar(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5]` (lines ~151–157)
- `fn scan_blocks_scalar(q: &[f32; 14], ds: &Dataset, probed: &[u16]) -> [u8; 5]` (lines ~159–206)

- [ ] **Step 6: Update `probe()` non-x86_64 branch to use `unimplemented!`**

Replace the current `probe()` function:

```rust
fn probe(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        probe_avx2(q, ds, nprobe)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (q, ds, nprobe);
        unimplemented!("requires x86_64")
    }
}
```

- [ ] **Step 7: Run tests**

```bash
cargo test fraud::knn -- --nocapture 2>&1 | tail -20
```
Expected: `count_fraud_correct`, `top_n_centroids_fast_smallest_first`, `smoke_warmup_and_query`, `smoke_fraud_heavy_query` — all PASS. `fast_probe_counter_increments` gone.

- [ ] **Step 8: Commit**

```bash
git add src/fraud/knn.rs
git commit -m "perf(knn): remove Instant, metrics, scalar dead code from hot path"
```

---

### Task 2: Rewrite http.rs — inline IVF, remove model + metrics + Metrics route

**Files:**
- Modify: `src/net/http.rs`

- [ ] **Step 1: Delete model-related and metrics-related tests from `src/net/http.rs`**

Remove these four test functions entirely:
- `model_classifies_all_ones_as_fraud`
- `model_fast_path_legit_returns_approved_true`
- `detect_route_metrics`
- `metrics_body_contains_all_keys`

- [ ] **Step 2: Remove `Route::Metrics` variant and update `detect_route`**

Replace the `Route` enum:

```rust
#[derive(Debug, PartialEq)]
pub enum Route {
    FraudScore,
    Ready,
    NotFound,
}
```

Replace `detect_route`:

```rust
pub fn detect_route(first_line: &[u8]) -> Route {
    if first_line.starts_with(b"POST /fraud-score") {
        Route::FraudScore
    } else if first_line.starts_with(b"GET /ready") {
        Route::Ready
    } else {
        Route::NotFound
    }
}
```

- [ ] **Step 3: Delete `build_metrics_body` function**

Remove the entire function:

```rust
pub fn build_metrics_body() -> String {
    let s = crate::metrics::snapshot();
    format!(...)
}
```

- [ ] **Step 4: Rewrite the `FraudScore` arm and remove `Metrics` arm in `serve_connection`**

Replace the entire `match route { ... }` block inside `serve_connection` with:

```rust
match route {
    Route::Ready => {
        tx_buf.extend_from_slice(RESP_READY);
        consumed += header_end;
    }
    Route::NotFound => {
        tx_buf.extend_from_slice(RESP_NOT_FOUND);
        consumed += header_end;
    }
    Route::FraudScore => {
        let cl = match parse_content_length(header_bytes) {
            Some(n) => n,
            None => {
                tx_buf.extend_from_slice(RESP_BAD_REQ);
                consumed += header_end;
                continue;
            }
        };
        let body_end = consumed + header_end + cl;
        if header_end + cl > RX_CAP {
            tx_buf.extend_from_slice(RESP_BAD_REQ);
            consumed += header_end;
            continue;
        }
        if body_end > rx_buf.len() {
            break; // need more data
        }
        let body = &rx_buf[consumed + header_end..body_end];
        let resp = match json::parse(body) {
            Some(payload) => {
                let vec = vector::vectorize(&payload);
                http_body_for(knn::knn5_ivf(&vec, ds))
            }
            None => RESP_BAD_REQ,
        };
        tx_buf.extend_from_slice(resp);
        consumed = body_end;
    }
}
```

- [ ] **Step 5: Clean up removed imports at top of http.rs**

The `use` statement at the top should become:

```rust
use crate::fraud::{data, json, knn, vector};
use crate::net::response::{http_body_for, RESP_BAD_REQ, RESP_NOT_FOUND, RESP_READY};
use memchr::memmem;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
```

- [ ] **Step 6: Run tests**

```bash
cargo test net::http -- --nocapture 2>&1 | tail -20
```
Expected: `find_header_end_locates_crlfcrlf`, `find_header_end_returns_none_when_absent`, `parse_content_length_*`, `route_detection` — all PASS.

Note: `route_detection` test currently checks `detect_route(b"GET /unknown HTTP/1.1") == Route::NotFound`. It does NOT check `Route::Metrics` (that was in `detect_route_metrics` which we deleted). Verify the remaining route_detection test still compiles:

```rust
#[test]
fn route_detection() {
    assert_eq!(detect_route(b"POST /fraud-score HTTP/1.1"), Route::FraudScore);
    assert_eq!(detect_route(b"GET /ready HTTP/1.1"), Route::Ready);
    assert_eq!(detect_route(b"GET /unknown HTTP/1.1"), Route::NotFound);
}
```

- [ ] **Step 7: Commit**

```bash
git add src/net/http.rs
git commit -m "perf(http): inline IVF, remove model fast-path and metrics route"
```

---

### Task 3: Merge json.rs — single parse() function

**Files:**
- Modify: `src/fraud/json.rs`

- [ ] **Step 1: Delete tests that reference parse_full or parse_positional directly**

Remove these three test functions from `src/fraud/json.rs`:
- `parse_falls_back_to_full_when_positional_fails`
- `parse_positional_matches_full_for_legit`
- `parse_positional_matches_full_for_tx_with_last`

- [ ] **Step 2: Run tests to confirm remaining tests pass before refactor**

```bash
cargo test fraud::json -- --nocapture 2>&1 | tail -20
```
Expected: `parse_legit_no_last_tx`, `parse_tx_with_last_transaction`, `parse_unknown_merchant`, `parse_returns_none_on_garbage` — all PASS (parse_full and parse_positional still exist at this point).

- [ ] **Step 3: Delete `parse_full` and replace `parse` with the positional logic**

The goal: `parse()` becomes what `parse_positional()` currently is. `parse_full()` and the old `parse_positional()` are deleted.

Replace the three functions (current `parse`, `parse_full`, `parse_positional`) with a single `pub fn parse` that contains exactly the body of the current `parse_positional`. The signature stays identical:

```rust
pub fn parse(buf: &[u8]) -> Option<Payload> {
    let mut pos = memchr(b'{', buf)?;
    pos += 1;

    // "id": skip value
    skip_to_value(buf, &mut pos)?;
    read_string(buf, &mut pos)?;

    // "transaction" key, then "amount" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    skip_to_value(buf, &mut pos)?;
    let installments: u8 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "requested_at"
    skip_to_value(buf, &mut pos)?;
    if buf.get(pos) != Some(&b'"') {
        return None;
    }
    let dt_start = pos + 1;
    let (y, mo, d, hour, _min_ignored) = parse_iso(buf, dt_start)?;
    let weekday = date_weekday(y, mo, d);
    let cur_time = (y, mo, d, hour, parse_digits2(buf, dt_start + 14)?);
    pos = dt_start + memchr(b'"', &buf[dt_start..])? + 1;

    // "customer" key, then "avg_amount" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let customer_avg_amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    skip_to_value(buf, &mut pos)?;
    let tx_count_24h: u8 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "known_merchants": array
    skip_to_value(buf, &mut pos)?;
    pos += memchr(b'[', &buf[pos..])? + 1;

    const MAX_KNOWN: usize = 32;
    const MAX_ID_LEN: usize = 16;
    let mut known_buf = [[0u8; MAX_ID_LEN]; MAX_KNOWN];
    let mut known_lens = [0u8; MAX_KNOWN];
    let mut known_count: usize = 0;

    loop {
        while pos < buf.len() && matches!(buf[pos], b' ' | b'\n' | b'\r' | b'\t' | b',') {
            pos += 1;
        }
        if pos >= buf.len() {
            return None;
        }
        if buf[pos] == b']' {
            pos += 1;
            break;
        }
        if buf[pos] == b'"' {
            let s = read_string(buf, &mut pos)?;
            if known_count < MAX_KNOWN {
                let len = s.len().min(MAX_ID_LEN);
                known_buf[known_count][..len].copy_from_slice(&s[..len]);
                known_lens[known_count] = len as u8;
                known_count += 1;
            }
        } else {
            pos += 1;
        }
    }

    // "merchant" key, then "id" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let merch_id = read_string(buf, &mut pos)?;
    let is_unknown_merchant = !(0..known_count).any(|i| {
        let len = known_lens[i] as usize;
        merch_id.len() == len && merch_id == &known_buf[i][..len]
    });

    skip_to_value(buf, &mut pos)?;
    let mcc_str = read_string(buf, &mut pos)?;
    let mcc: u32 = std::str::from_utf8(mcc_str).ok()?.parse().ok()?;

    skip_to_value(buf, &mut pos)?;
    let merchant_avg_amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "terminal" key, then "is_online" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let tok = read_token(buf, &mut pos)?;
    let is_online = tok == b"true";

    skip_to_value(buf, &mut pos)?;
    let tok = read_token(buf, &mut pos)?;
    let card_present = tok == b"true";

    skip_to_value(buf, &mut pos)?;
    let km_from_home: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "last_transaction": null | { ... }
    skip_to_value(buf, &mut pos)?;

    while pos < buf.len() && matches!(buf[pos], b' ' | b'\n' | b'\r' | b'\t') {
        pos += 1;
    }

    let (has_last_tx, minutes_since_last, km_from_current) =
        if buf.get(pos..pos + 4) == Some(b"null") {
            (false, 0.0f32, 0.0f32)
        } else if buf.get(pos) == Some(&b'{') {
            pos += 1;

            skip_to_value(buf, &mut pos)?;
            if buf.get(pos) != Some(&b'"') {
                return None;
            }
            let ts_start = pos + 1;
            let prev_time = (
                parse_digits4(buf, ts_start)?,
                parse_digits2(buf, ts_start + 5)?,
                parse_digits2(buf, ts_start + 8)?,
                parse_digits2(buf, ts_start + 11)?,
                parse_digits2(buf, ts_start + 14)?,
            );
            pos = ts_start + memchr(b'"', &buf[ts_start..])? + 1;

            skip_to_value(buf, &mut pos)?;
            let km_cur: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
                .ok()?
                .parse()
                .ok()?;

            let mins = minutes_between(cur_time, prev_time);
            (true, mins, km_cur)
        } else {
            return None;
        };

    Some(Payload {
        amount,
        installments,
        hour,
        weekday,
        customer_avg_amount,
        tx_count_24h,
        is_unknown_merchant,
        mcc,
        merchant_avg_amount,
        is_online,
        card_present,
        km_from_home,
        has_last_tx,
        minutes_since_last,
        km_from_current,
    })
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test fraud::json -- --nocapture 2>&1 | tail -20
```
Expected: `parse_legit_no_last_tx`, `parse_tx_with_last_transaction`, `parse_unknown_merchant`, `parse_returns_none_on_garbage` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/json.rs
git commit -m "refactor(json): merge parse_full into parse, remove 250-line duplicate"
```

---

### Task 4: Delete model files + update fraud/mod.rs

**Files:**
- Delete: `src/fraud/model.rs`
- Delete: `src/fraud/model_gen.rs`
- Modify: `src/fraud/mod.rs`

- [ ] **Step 1: Remove `pub mod model` from `src/fraud/mod.rs`**

Replace the file content with:

```rust
pub mod data;
pub mod json;
pub mod knn;
pub mod vector;
```

- [ ] **Step 2: Delete model files**

```bash
rm src/fraud/model.rs src/fraud/model_gen.rs
```

- [ ] **Step 3: Run tests**

```bash
cargo test 2>&1 | tail -20
```
Expected: all tests PASS. `model.rs` and `model_gen.rs` are gone; no references remain (http.rs was already cleaned in Task 2).

- [ ] **Step 4: Commit**

```bash
git add src/fraud/mod.rs
git rm src/fraud/model.rs src/fraud/model_gen.rs
git commit -m "feat: delete model.rs and model_gen.rs (I-cache killer, proven to hurt p99)"
```

---

### Task 5: Delete metrics.rs + update main.rs

**Files:**
- Delete: `src/metrics.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Rewrite `src/main.rs`**

Replace the entire file:

```rust
mod env;
mod fraud;
mod net;

use tokio::net::UnixListener;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(512 * 1024)
        .enable_all()
        .build()
        .expect("failed to build runtime")
        .block_on(run());
}

async fn run() {
    fraud::data::init();

    let sock_path = env::sock_path();
    let _ = std::fs::remove_file(&sock_path);

    let listener = UnixListener::bind(&sock_path).expect("failed to bind unix socket");

    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o777))
        .expect("failed to set socket permissions");

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                tokio::spawn(net::http::serve_connection(stream));
            }
            Err(e) => {
                eprintln!("accept error: {e}");
                break;
            }
        }
    }
}
```

Changes vs before: `mod metrics` removed, `fraud::model::init()` removed, `fraud::knn::warmup()` removed, `worker_threads(2)` → `worker_threads(4)`, `.max_blocking_threads(2)` removed, `.thread_stack_size(512 * 1024)` added.

- [ ] **Step 2: Delete `src/metrics.rs`**

```bash
rm src/metrics.rs
```

- [ ] **Step 3: Run tests**

```bash
cargo test 2>&1 | tail -30
```
Expected: all tests PASS. No `mod metrics` references remain anywhere.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git rm src/metrics.rs
git commit -m "perf(runtime): worker_threads=4, stack=512KB, remove blocking pool and warmup"
```

---

### Task 6: Docker infra — haproxy.cfg, docker-compose.yml, Dockerfile, Cargo.toml, delete lb.rs

**Files:**
- Create: `haproxy.cfg`
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Modify: `Cargo.toml`
- Delete: `bin/lb.rs`

- [ ] **Step 1: Create `haproxy.cfg` in project root**

```
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

- [ ] **Step 2: Rewrite `docker-compose.yml`**

Replace the entire file:

```yaml
volumes:
  sock:

services:
  lb:
    image: haproxy:alpine
    ports:
      - "9999:9999"
    volumes:
      - sock:/run/sock
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    networks:
      - fraud-net
    deploy:
      resources:
        limits:
          cpus: "0.05"
          memory: "6MB"

  api1:
    build: .
    environment:
      - SOCK=/run/sock/api1.sock
    volumes:
      - sock:/run/sock
    networks:
      - fraud-net
    ulimits:
      nofile:
        soft: 65535
        hard: 65535
    deploy:
      resources:
        limits:
          cpus: "0.45"
          memory: "172MB"

  api2:
    build: .
    environment:
      - SOCK=/run/sock/api2.sock
    volumes:
      - sock:/run/sock
    networks:
      - fraud-net
    ulimits:
      nofile:
        soft: 65535
        hard: 65535
    deploy:
      resources:
        limits:
          cpus: "0.45"
          memory: "172MB"

networks:
  fraud-net:
    driver: bridge
```

- [ ] **Step 3: Update `Dockerfile` — remove lb binary**

Replace the entire file:

```dockerfile
FROM rust:latest AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin fraud-detection

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/fraud-detection /fraud-detection
CMD ["/fraud-detection"]
```

- [ ] **Step 4: Update `Cargo.toml` — remove `[[bin]] lb` and unused serde deps**

Remove this stanza:

```toml
[[bin]]
name = "lb"
path = "bin/lb.rs"
```

Remove these two dependency lines:

```toml
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
```

The remaining `[dependencies]` section should be:

```toml
[dependencies]
tokio       = { version = "1", default-features = false, features = ["rt", "rt-multi-thread", "macros", "net", "io-util"] }
memchr      = "2"
mimalloc    = { version = "0.1", default-features = false }
aligned-vec = "0.6"
flate2      = { version = "1", default-features = false, features = ["rust_backend"] }
libc        = "0.2"
```

- [ ] **Step 5: Delete `bin/lb.rs`**

```bash
rm bin/lb.rs
```

- [ ] **Step 6: Run `cargo test` to verify nothing broke**

```bash
cargo test 2>&1 | tail -20
```
Expected: all tests PASS.

- [ ] **Step 7: Run `cargo build --release` and verify binary size**

```bash
cargo build --release --bin fraud-detection 2>&1 | tail -5
ls -lh target/release/fraud-detection
```
Expected: build succeeds, no lb binary required.

- [ ] **Step 8: Commit**

```bash
git add haproxy.cfg docker-compose.yml Dockerfile Cargo.toml Cargo.lock
git rm bin/lb.rs
git commit -m "feat(infra): replace custom LB with haproxy:alpine TCP mode, remove serde deps"
```

---

### Task 7: Final integration verify

- [ ] **Step 1: Run full test suite**

```bash
cargo test 2>&1
```
Expected: all tests PASS, zero failures.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets -- -D warnings 2>&1 | tail -20
```
Expected: no warnings.

- [ ] **Step 3: Run release build**

```bash
cargo build --release 2>&1 | tail -5
```
Expected: compiles clean.

- [ ] **Step 4: Run local bench (if available)**

```bash
make bench 2>&1 | tail -20
```
Expected: local p99 ≤ 0.21ms, score 6000, 0 FP/FN.

- [ ] **Step 5: Update PROGRESS.md with implementation notes**

Add an entry documenting:
- What was changed (model deleted, metrics deleted, lb deleted, HAProxy, 4 workers, 512KB stacks)
- Local bench result
- Expected remote improvement

- [ ] **Step 6: Push to remote for submission**

```bash
git push
```
Then trigger remote test run per normal submission process.
