FROM rust:latest AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin fraud-detection --bin lb

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/fraud-detection /fraud-detection
COPY --from=builder /app/target/release/lb /lb
CMD ["/fraud-detection"]
