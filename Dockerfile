FROM rust:latest AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin fraud-api --bin lb

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/fraud-api /fraud-api
COPY --from=builder /app/target/release/lb /lb
CMD ["/fraud-api"]
