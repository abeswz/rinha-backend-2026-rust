FROM rust:1.92-slim AS builder

WORKDIR /build

COPY fraud-detection/.claude/worktrees/rust-port-go-strategy/Cargo.toml \
     fraud-detection/.claude/worktrees/rust-port-go-strategy/Cargo.lock ./
COPY fraud-detection/.claude/worktrees/rust-port-go-strategy/src/ ./src/
COPY fraud-detection/.claude/worktrees/rust-port-go-strategy/bin/ ./bin/

RUN cargo build --release --bin fraud-api --bin lb

FROM gcr.io/distroless/cc-debian12

COPY gopher-fraud-detection/index/index_p0.bin  /app/index/
COPY gopher-fraud-detection/index/index_p1.bin  /app/index/
COPY gopher-fraud-detection/index/index_p2.bin  /app/index/
COPY gopher-fraud-detection/index/index_p3.bin  /app/index/
COPY gopher-fraud-detection/index/index_p4.bin  /app/index/
COPY gopher-fraud-detection/index/index_p5.bin  /app/index/
COPY gopher-fraud-detection/index/index_p6.bin  /app/index/
COPY gopher-fraud-detection/index/index_p7.bin  /app/index/
COPY gopher-fraud-detection/index/index_p8.bin  /app/index/
COPY gopher-fraud-detection/index/index_p9.bin  /app/index/
COPY gopher-fraud-detection/index/index_p10.bin /app/index/
COPY gopher-fraud-detection/index/index_p11.bin /app/index/
COPY --from=builder /build/target/release/fraud-api /fraud-api
COPY --from=builder /build/target/release/lb        /lb

WORKDIR /app
