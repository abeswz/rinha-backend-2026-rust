#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
Build IVF2 index from resources/references.json.gz.
Output: resources/ivf_index.bin

IVF2 binary format (all little-endian):
  [4B]        magic: IVF2
  [4B u32]    n   — total vectors (before padding)
  [4B u32]    k   — number of centroids
  [4B u32]    d   — dimensions (14)
  [d*k*4B]    centroids f32, column-major: centroids[d_idx * k + ci]
  [(k+1)*4B]  block_offsets u32 — offsets[ci]..offsets[ci+1] = block range (unit: 8-vec block)
  [total_blocks*8 B] labels u8 — padding slots = 0
  [total_blocks*d*8*2B] blocks i16 — blocks[(block_idx*d+dim)*8+slot], padding = i16::MAX

Quantization: i16 = round(f32 * 10_000). Range [-3.2768, 3.2767] maps exactly.
Sentinel -1.0 (null last_transaction) → -10000. Fits in i16.
"""

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

INPUT = Path("resources/references.json.gz")
OUTPUT = Path("resources/ivf_index.bin")
K = 4096
D = 14
BATCH_SIZE = 50_000
N_INIT = 3
RANDOM_STATE = 42
SCALE = 10_000


def quantize(arr: np.ndarray) -> np.ndarray:
    """f32 → i16 via round(x * 10000). Clips to i16 range."""
    return np.clip(np.round(arr * SCALE), -32768, 32767).astype(np.int16)


def _write_ivf2(vectors: np.ndarray, labels: np.ndarray, k: int, output_path: str) -> None:
    """Fit KMeans, build IVF2 binary, write to output_path."""
    n, d = vectors.shape
    assert d == D

    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=BATCH_SIZE,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    assignments = km.fit_predict(vectors)
    centroids = km.cluster_centers_.astype(np.float32)  # shape: (k, d)

    # Group vector indices by cluster
    cluster_vecs: list[list[int]] = [[] for _ in range(k)]
    for i, ci in enumerate(assignments):
        cluster_vecs[ci].append(i)

    # Compute block offsets (unit: 8-vector blocks)
    block_offsets = np.zeros(k + 1, dtype=np.uint32)
    for ci in range(k):
        n_blocks = (len(cluster_vecs[ci]) + 7) // 8
        block_offsets[ci + 1] = block_offsets[ci] + n_blocks

    total_blocks = int(block_offsets[k])
    padded_n = total_blocks * 8

    out_labels = np.zeros(padded_n, dtype=np.uint8)
    out_blocks = np.full(total_blocks * d * 8, fill_value=32767, dtype=np.int16)

    for ci in range(k):
        block_start = int(block_offsets[ci])
        vecs_ci = cluster_vecs[ci]
        n_blocks = int(block_offsets[ci + 1]) - block_start

        for bk in range(n_blocks):
            block_idx = block_start + bk
            label_base = block_idx * 8
            block_base = block_idx * d * 8

            for slot in range(8):
                vi_pos = bk * 8 + slot
                if vi_pos >= len(vecs_ci):
                    break
                vi = vecs_ci[vi_pos]
                out_labels[label_base + slot] = labels[vi]
                for dim in range(d):
                    out_blocks[block_base + dim * 8 + slot] = quantize(vectors[vi, dim : dim + 1])[0]

    # centroids: column-major [d * k]
    centroids_t = centroids.T.copy()  # shape (d, k), row = dim

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"IVF2")
        f.write(struct.pack("<III", n, k, d))
        f.write(centroids_t.astype(np.float32).tobytes())
        f.write(block_offsets.tobytes())
        f.write(out_labels.tobytes())
        f.write(out_blocks.tobytes())


def _test_ivf2_roundtrip():
    import tempfile
    import os

    n_test, k_test = 80, 4
    rng = np.random.default_rng(42)
    vecs = rng.uniform(-1, 1, (n_test, D)).astype(np.float32)
    lbls = rng.integers(0, 2, n_test).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = f.name

    try:
        _write_ivf2(vecs, lbls, k_test, tmp)
        with open(tmp, "rb") as f:
            magic = f.read(4)
            assert magic == b"IVF2", f"bad magic: {magic}"
            n_out = int.from_bytes(f.read(4), "little")
            k_out = int.from_bytes(f.read(4), "little")
            d_out = int.from_bytes(f.read(4), "little")
            assert n_out == n_test, f"n mismatch: {n_out} != {n_test}"
            assert k_out == k_test, f"k mismatch: {k_out} != {k_test}"
            assert d_out == D, f"d mismatch: {d_out} != {D}"
        print("PASS: IVF2 roundtrip test")
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test_ivf2_roundtrip()
        sys.exit(0)

    print(f"Loading {INPUT}...", flush=True)
    if not INPUT.exists():
        sys.exit(f"ERROR: {INPUT} not found. Run from the fraud-detection directory.")

    with gzip.open(INPUT, "rt", encoding="utf-8") as f:
        refs = json.load(f)

    vectors = np.array([r["vector"] for r in refs], dtype=np.float32)
    labels = np.array([1 if r["label"] == "fraud" else 0 for r in refs], dtype=np.uint8)
    N, d_check = vectors.shape
    print(f"Loaded {N} vectors, D={d_check}", flush=True)

    if d_check != D:
        sys.exit(f"ERROR: expected D={D}, got {d_check}")

    print(f"Building IVF2 index K={K}...", flush=True)
    _write_ivf2(vectors, labels, K, str(OUTPUT))

    size_mb = OUTPUT.stat().st_size / 1024**2
    print(f"Done. {OUTPUT} = {size_mb:.1f} MB", flush=True)
