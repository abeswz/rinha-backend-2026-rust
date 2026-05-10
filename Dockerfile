FROM rust:1.80-slim as builder

RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY bin/ bin/
COPY resources/ resources/

RUN cargo build --release
RUN cargo run --bin preprocess

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/fraud-detection ./
COPY --from=builder /app/resources/refs.bin resources/refs.bin
COPY --from=builder /app/resources/mcc_risk.json resources/mcc_risk.json
COPY --from=builder /app/resources/normalization.json resources/normalization.json

EXPOSE 3000
CMD ["./fraud-detection"]
