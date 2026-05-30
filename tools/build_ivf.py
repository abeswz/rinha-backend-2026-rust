#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "faiss-cpu"]
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
  [total_blocks*d*8*1B] blocks i8 — blocks[(block_idx*d+dim)*8+slot], padding = i8::MAX

Quantization: i8 = round(f32 * 100). All features in [-1.0, 1.0] → i8 [-100, 100].
Sentinel -1.0 (null last_transaction) → -100. Fits in i8.
"""

import gzip
import json
import struct
import sys
from pathlib import Path

import faiss
import numpy as np

INPUT = Path("resources/references.json.gz")
OUTPUT = Path("resources/ivf_index.bin")
K = 4096
D = 14
RANDOM_STATE = 42
SCALE = 100


def quantize(arr: np.ndarray) -> np.ndarray:
    """f32 → i8 via round(x * 100). All features in [-1.0, 1.0] → [-100, 100]."""
    return np.clip(np.round(arr * SCALE), -128, 127).astype(np.int8)


def _write_ivf2(
    vectors: np.ndarray,
    labels: np.ndarray,
    k: int,
    output_path: str,
) -> None:
    """Fit KMeans, build IVF2 binary, write to output_path."""
    n, d = vectors.shape
    assert d == D

    print("Training KMeans...", flush=True)

    km = faiss.Kmeans(
        d,
        k,
        niter=20,
        nredo=1,
        verbose=True,
        seed=RANDOM_STATE,
    )

    km.train(vectors)

    centroids = km.centroids

    print("Assigning vectors to centroids...", flush=True)

    _, asn = km.index.search(vectors, 1)
    assignments = asn.ravel()

    print("Quantizing vectors...", flush=True)

    qvectors = quantize(vectors)

    print("Sorting by cluster...", flush=True)

    order = np.argsort(assignments, kind="stable")
    sorted_clusters = assignments[order]

    counts = np.bincount(assignments, minlength=k)

    cluster_offsets = np.zeros(k + 1, dtype=np.int64)
    cluster_offsets[1:] = np.cumsum(counts)

    block_offsets = np.zeros(k + 1, dtype=np.uint32)

    for ci in range(k):
        n_vecs = int(counts[ci])
        block_offsets[ci + 1] = block_offsets[ci] + ((n_vecs + 7) // 8)

    total_blocks = int(block_offsets[-1])

    out_labels = np.zeros(total_blocks * 8, dtype=np.uint8)

    out_blocks = np.full(
        total_blocks * d * 8,
        fill_value=127,
        dtype=np.int8,
    )

    print(
        f"Packing {total_blocks:,} blocks...",
        flush=True,
    )

    for ci in range(k):
        start = int(cluster_offsets[ci])
        end = int(cluster_offsets[ci + 1])

        if start == end:
            continue

        idx = order[start:end]

        cluster_labels = labels[idx]
        cluster_vectors = qvectors[idx]

        n_vecs = len(idx)
        n_blocks = (n_vecs + 7) // 8

        padded_size = n_blocks * 8
        pad = padded_size - n_vecs

        if pad:
            cluster_labels = np.pad(
                cluster_labels,
                (0, pad),
                mode="constant",
                constant_values=0,
            )

            cluster_vectors = np.pad(
                cluster_vectors,
                ((0, pad), (0, 0)),
                mode="constant",
                constant_values=127,
            )

        labels_2d = cluster_labels.reshape(
            n_blocks,
            8,
        )

        blocks_3d = cluster_vectors.reshape(n_blocks, 8, d).transpose(0, 2, 1)

        block_start = int(block_offsets[ci])

        out_labels[block_start * 8 : (block_start + n_blocks) * 8] = labels_2d.ravel()

        out_blocks[block_start * d * 8 : (block_start + n_blocks) * d * 8] = (
            blocks_3d.ravel()
        )

    print("Writing file...", flush=True)

    centroids_t = centroids.T.copy()

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "wb") as f:
        f.write(b"IVF2")
        f.write(struct.pack("<III", n, k, d))
        f.write(centroids_t.astype(np.float32).tobytes())
        f.write(block_offsets.tobytes())
        f.write(out_labels.tobytes())
        f.write(out_blocks.tobytes())


def _test_ivf2_roundtrip():
    import os
    import tempfile

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
