FROM python:3.12-slim AS ivf-builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /build
COPY resources/references.json.gz resources/
COPY tools/ tools/
RUN uv pip install --system --no-cache -r tools/requirements.txt && \
    python tools/build_ivf.py

FROM rust:1.83-slim AS rust-builder
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src bin && \
    printf '' > src/lib.rs && \
    printf 'fn main() {}' > src/main.rs && \
    printf 'fn main() {}' > bin/preprocess.rs && \
    cargo build --release --bin fraud-detection && \
    rm -rf src bin

COPY src/ src/
COPY bin/ bin/
COPY resources/mcc_risk.json resources/mcc_risk.json
COPY resources/normalization.json resources/normalization.json
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
ENV RUSTFLAGS="-C target-feature=+avx2,+fma,+f16c,+bmi2,+popcnt -C strip=symbols"
RUN cargo build --release --bin fraud-detection

FROM gcr.io/distroless/cc-debian12
WORKDIR /app
COPY --from=rust-builder /app/target/release/fraud-detection ./
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/ivf_index.bin
COPY --from=rust-builder /app/resources/mcc_risk.json resources/mcc_risk.json
COPY --from=rust-builder /app/resources/normalization.json resources/normalization.json
EXPOSE 3000
ENTRYPOINT ["./fraud-detection"]
