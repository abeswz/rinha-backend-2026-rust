# Design: Monoio → Tokio Migration

**Date:** 2026-05-17
**Goal:** Replace monoio/io_uring runtime with Tokio current-thread to pass Docker default seccomp profile and qualify for competition submission.

---

## Problem

Monoio requires `security_opt: seccomp:unconfined` because Docker's default seccomp profile blocks io_uring syscalls (`io_uring_setup`, `io_uring_enter`, `io_uring_register`). The competition environment rejects containers with this override — submission is rejected before execution.

---

## Approach

Surgical Tokio swap. Replace only the monoio API surface (`main.rs`, `net/http.rs`). All fraud logic, hand-rolled HTTP parser, static response buffers, SIMD KNN, and build pipeline are unchanged.

Additionally harden NGINX: disable access logging (eliminates disk write per request), remove proxy headers the API never reads.

---

## Architecture

```
internet → NGINX (unix socket, least_conn, keepalive 64)
                ↓                    ↓
         api1.sock            api2.sock
              ↓                    ↓
    tokio current-thread    tokio current-thread
    UnixListener             UnixListener
         ↓
    net::http::serve_connection (per connection)
      AsyncReadExt::read_buf → rx_buf (Vec<u8>, cap 8192)
      find_header_end / detect_route / parse_content_length
      fraud::{json, vector, knn}
      http_body_for(count) → &'static [u8]
      AsyncWriteExt::write_all → flush
```

Tokio `current_thread` flavor eliminates work-stealing, scheduler contention, and cross-thread wakeup cost under the 0.45 CPU cgroup.

---

## File Change Map

| File | Action | Notes |
|------|--------|-------|
| `Cargo.toml` | Modify | `monoio` → `tokio` (rt, macros, net, io-util) |
| `src/main.rs` | Rewrite | `#[tokio::main(flavor = "current_thread")]`, `tokio::net::UnixListener`, `tokio::spawn` |
| `src/net/http.rs` | Modify | `tokio::net::UnixStream`, `AsyncReadExt`/`AsyncWriteExt`, drop owned-buffer pattern |
| `docker-compose.yml` | Modify | Remove `security_opt: seccomp:unconfined` from api1 and api2 |
| `nginx.conf` | Modify | `access_log off`, remove unused `proxy_set_header Host` and `proxy_set_header X-Real-IP` |
| `Dockerfile` | No change | Already correct; no `RUSTFLAGS` native needed (KNN uses explicit `#[target_feature]`) |

**Unchanged:** `src/fraud/`, `src/env.rs`, `src/net/response.rs`, `src/net/mod.rs`, `bin/build_index.rs`

---

## Key API Swap

### Runtime (`main.rs`)

```rust
// before
monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
    .with_entries(1024)
    .build()
    .expect("failed to build monoio runtime")
    .block_on(async { ... });

// after
#[tokio::main(flavor = "current_thread")]
async fn main() { ... }
```

### Listener (`main.rs`)

```rust
// before
use monoio::net::{ListenerOpts, UnixListener};
let opts = ListenerOpts::new().reuse_port(false).reuse_addr(false);
let listener = UnixListener::bind_with_config(&sock_path, &opts)?;

// after
use tokio::net::UnixListener;
let listener = UnixListener::bind(&sock_path)?;
```

### Accept loop (`main.rs`)

```rust
// before
monoio::spawn(net::http::serve_connection(stream));

// after
tokio::spawn(net::http::serve_connection(stream));
```

### Read (`net/http.rs`)

```rust
// before (monoio owned-buffer model)
let read_buf = Vec::with_capacity(RX_CAP);
let (res, buf) = stream.read(read_buf).await;
match res {
    Ok(0) => break,
    Err(_) => break,
    Ok(n) => rx_buf.extend_from_slice(&buf[..n]),
}

// after (tokio readiness model)
use tokio::io::AsyncReadExt;
let n = stream.read_buf(&mut rx_buf).await?;
if n == 0 { break; }
```

### Write (`net/http.rs`)

```rust
// before (monoio owned-buffer write_all)
let out = std::mem::take(&mut tx_buf);
let (res, mut out) = stream.write_all(out).await;
out.clear();
tx_buf = out;
if res.is_err() { break; }

// after
use tokio::io::AsyncWriteExt;
stream.write_all(&tx_buf).await?;
tx_buf.clear();
```

---

## Cargo.toml Changes

```toml
# remove
monoio = { version = "0.2", default-features = false, features = ["iouring", "legacy", "macros"] }

# add
tokio = { version = "1", default-features = false, features = ["rt", "macros", "net", "io-util"] }
```

All other dependencies (`memchr`, `mimalloc`, `aligned-vec`, `flate2`, `libc`, `serde`, `serde_json`) stay.

---

## docker-compose.yml Changes

Remove from both `api1` and `api2`:
```yaml
security_opt:
  - seccomp:unconfined
```

Keep `ulimits.nofile` (65535), CPU/memory limits, volume mounts.

---

## nginx.conf Changes

```nginx
worker_processes 1;

events {
    worker_connections 2048;
}

access_log off;          # add: eliminates disk write per request

http {
    upstream api_backends {
        least_conn;
        server unix:/run/sock/api1.sock;
        server unix:/run/sock/api2.sock;
        keepalive 64;
    }

    server {
        listen 9999;

        location / {
            proxy_pass http://api_backends;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            # removed: proxy_set_header Host $host;        (API never reads)
            # removed: proxy_set_header X-Real-IP $remote_addr;  (API never reads)
            proxy_read_timeout 1800ms;
            proxy_connect_timeout 1s;
        }
    }
}
```

---

## Tests

All existing tests pass without modification. They operate on pure functions (`find_header_end`, `parse_content_length`, `detect_route`, `http_body_for`, fraud logic) — none touch monoio. No new tests required; the runtime swap has no observable behavior change at the unit level.

Verify:
```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --bin fraud-detection
docker compose build
docker compose up
curl http://localhost:9999/ready
```

---

## What Does NOT Change

- Hot path allocations: zero
- JSON parser: hand-rolled positional (`fraud/json.rs`), no serde in hot path
- Response buffers: 9 static `&[u8]` slices
- KNN: AVX2/FMA with compile-time `#[target_feature]`, no runtime cpuid
- IVF1 index: embedded via `include_bytes!`, decoded once at startup
- Socket permissions (0o777 chmod after bind)
- Mimalloc global allocator
- Release profile: `lto=fat`, `codegen-units=1`, `panic=abort`, `opt-level=3`, `strip=true`
