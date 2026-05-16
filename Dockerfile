# Stage 1: Build IVF index (Python)
FROM python:3.12-slim AS ivf-builder
WORKDIR /build
COPY resources/references.json.gz resources/
COPY tools/ tools/
RUN pip install --no-cache-dir -r tools/requirements.txt && \
    python tools/build_ivf.py

# Stage 2: Build Rust binary
FROM rust:1.82-slim AS rust-builder
RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY bin/ bin/
COPY resources/ resources/
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cargo build --release

# Stage 3: Runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=rust-builder /app/target/release/fraud-detection ./
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
COPY --from=rust-builder /app/resources/mcc_risk.json resources/mcc_risk.json
COPY --from=rust-builder /app/resources/normalization.json resources/normalization.json
EXPOSE 3000
CMD ["./fraud-detection"]
