FROM rust:1.92-slim AS builder

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY bin/ ./bin/

RUN cargo build --release --bin fraud-api --bin lb

FROM gcr.io/distroless/cc-debian12

COPY index/ /app/index/
COPY --from=builder /build/target/release/fraud-api /fraud-api
COPY --from=builder /build/target/release/lb        /lb

WORKDIR /app
