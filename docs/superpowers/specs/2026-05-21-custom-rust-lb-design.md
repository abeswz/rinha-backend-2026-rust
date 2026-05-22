# Custom Rust TCP Proxy LB

**Date:** 2026-05-21
**Goal:** Replace nginx with a minimal Rust TCP proxy to eliminate CFS CPU throttling on the competition server (Mac Mini Late 2014, Intel Core i5-4278U, 2.6GHz Haswell).

**Baseline:** nginx `cpus: 0.10` → p99 = 366ms on competition → score = 3435.45 (score_p99=436, score_det=3000).

**Target:** LB `cpus: 0.05` custom Rust proxy → p99 ≤ 5ms on competition → score ≥ 5300.

---

## Root Cause

nginx at `cpus: 0.10` on competition hardware runs under Linux CFS with a 100ms period and 10ms quota. Under the competition load (~900 req/s), nginx exhausts its 10ms CPU quota and is throttled for up to 90ms before the next period. This directly causes p99 spikes of ~90-360ms, not IVF computation (which is ~10–100μs).

Evidence: detection_score = 3000 (perfect accuracy). Score loss is entirely in score_p99.

nginx adds unnecessary overhead: HTTP module parsing, config evaluation, access log infrastructure, event loop indirection — none needed for a pure-forward proxy.

---

## Solution

Replace nginx with a ~50-line Rust binary (`lb`) that:

1. Listens on TCP `0.0.0.0:9999`
2. Accepts client connections
3. Alternates assignments to `api1.sock` / `api2.sock` via `AtomicUsize % 2` (round-robin, connection-level)
4. Copies bytes bidirectionally with `tokio::io::copy` — no byte inspection

No HTTP parsing. No headers. No logging. No timeouts. Zero business logic.

### Why connection-level round-robin is sufficient

The competition test (k6) opens N virtual user connections. Each VU gets assigned to api1 or api2 alternately. With any reasonable number of VUs (≥ 10), load distributes evenly. Request-level round-robin would require HTTP boundary detection, adding parsing overhead that defeats the purpose.

### Why TCP_NODELAY matters

Without `TCP_NODELAY`, Nagle's algorithm buffers small writes. Responses from the API (pre-computed static bytes, ≤ 150 bytes) would be delayed by up to 40ms waiting for the buffer to fill. Setting `TCP_NODELAY` on the client-side socket eliminates this.

---

## Implementation

### Cargo.toml addition

```toml
[[bin]]
name = "lb"
path = "bin/lb.rs"
```

Dependencies required: `tokio` (already present with `net`, `io-util`, `rt-multi-thread`, `macros`).

### bin/lb.rs

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::{TcpListener, UnixStream};
use tokio::io;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

const SOCK1: &str = "/run/sock/api1.sock";
const SOCK2: &str = "/run/sock/api2.sock";

fn main() {
    // Block (before starting async runtime) until both API sockets exist.
    // std::thread::sleep avoids requiring the `tokio/time` feature.
    loop {
        if std::path::Path::new(SOCK1).exists() && std::path::Path::new(SOCK2).exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("build runtime")
        .block_on(serve());
}

async fn serve() {
    let listener = TcpListener::bind("0.0.0.0:9999").await.expect("bind 9999");
    loop {
        match listener.accept().await {
            Ok((stream, _)) => { tokio::spawn(forward(stream)); }
            Err(_) => {}
        }
    }
}

async fn forward(mut client: tokio::net::TcpStream) {
    let _ = client.set_nodelay(true);
    let idx = COUNTER.fetch_add(1, Ordering::Relaxed) % 2;
    let sock_path = if idx == 0 { SOCK1 } else { SOCK2 };

    let Ok(mut backend) = UnixStream::connect(sock_path).await else { return };

    let (mut cr, mut cw) = client.into_split();
    let (mut br, mut bw) = backend.into_split();

    let _ = tokio::join!(
        io::copy(&mut cr, &mut bw),
        io::copy(&mut br, &mut cw),
    );
}
```

**Runtime choice:** `current_thread` + `enable_io()` only — the LB is pure IO, zero CPU computation. Single-threaded async avoids thread creation overhead and stays well within 0.05 CPU. No `time` feature needed.

**Wait loop:** blocking `std::thread::sleep` before the async runtime starts. Avoids needing `tokio/time` feature and keeps Cargo.toml unchanged. LB accepts connections only after both API Unix sockets exist.

**Error handling:** silent drop on connect failure. Under the competition test, APIs start before significant load arrives (warmup phase). Mid-test API failures return a dropped connection, which k6 counts as an HTTP error — but this is the same behavior nginx exhibits.

### Dockerfile

Both `fraud-detection` and `lb` binaries are built by `cargo build --release`. The existing multi-stage Dockerfile must copy both binaries in the final stage:

```dockerfile
# In the final stage, copy both binaries:
COPY --from=builder /app/target/release/fraud-detection /fraud-detection
COPY --from=builder /app/target/release/lb /lb
```

**Submission image:** The submission branch docker-compose.yml must reference a pre-built public image (not `build: .`). After building, push:

```bash
docker build --platform linux/amd64 -t ghcr.io/<user>/rinha-backend:latest .
docker push ghcr.io/<user>/rinha-backend:latest
```

Then in submission branch docker-compose.yml, replace `build: .` with `image: ghcr.io/<user>/rinha-backend:latest` for both `lb` and `api1`/`api2` services.

### docker-compose.yml

Full replacement:

```yaml
volumes:
  sock:

services:
  lb:
    build: .
    command: ["/lb"]
    ports:
      - "9999:9999"
    volumes:
      - sock:/run/sock
    networks:
      - fraud-net
    deploy:
      resources:
        limits:
          cpus: "0.05"
          memory: "5MB"

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
          cpus: "0.475"
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
          cpus: "0.475"
          memory: "172MB"

networks:
  fraud-net:
    driver: bridge
```

**Resource totals:** 0.05 + 0.475 + 0.475 = **1.00 CPU**. 5 + 172 + 172 = **349 MB** (within 350MB limit).

**Removed:** nginx.conf (no longer needed). `depends_on` removed — LB polls sockets directly.

---

## CPU Budget Comparison

| Service | Before | After | Delta |
|---------|--------|-------|-------|
| LB | 0.10 (nginx) | 0.05 (Rust) | −0.05 |
| api1 | 0.45 | 0.475 | +0.025 |
| api2 | 0.45 | 0.475 | +0.025 |

---

## Expected Score Impact

| Metric | Before | After (est.) |
|--------|--------|--------------|
| p99 (competition) | 366ms | 1–5ms |
| score_p99 | 436 | 2300–3000 |
| score_det | 3000 | 3000 |
| **final_score** | **3435** | **5300–6000** |

---

## Testing

### Unit / compile

```bash
cargo build --release
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

### Local integration

```bash
docker compose up --build -d
# wait for ready
curl -s http://localhost:9999/ready
# run local k6 test
```

### Verify round-robin

```bash
# Two requests should alternate between api1 and api2
curl -s http://localhost:9999/ready
curl -s http://localhost:9999/ready
# check logs or add temporary logging to confirm alternation
```

### Verify no business logic in LB

LB binary must not import any fraud detection code. `cargo build --bin lb` must succeed without `fraud` module.

---

## Constraints Satisfied

| Rule | Status |
|------|--------|
| ≥ 1 LB + 2 API instances | ✓ lb + api1 + api2 |
| Round-robin, no business logic | ✓ AtomicUsize % 2, byte pipe only |
| Total ≤ 1 CPU | ✓ 1.00 CPU |
| Total ≤ 350 MB | ✓ 349 MB |
| linux/amd64 compatible | ✓ Rust target x86_64-unknown-linux-gnu |
| bridge network | ✓ |
| port 9999 | ✓ |
| Public image | ✓ build + push to ghcr.io/user/rinha-backend; submission branch uses `image:` not `build:` |
