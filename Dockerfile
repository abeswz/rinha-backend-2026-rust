FROM rust:latest AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin build_index
RUN ./target/release/build_index
RUN cargo build --release --bin fraud-detection

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/fraud-detection /fraud-detection
CMD ["/fraud-detection"]
