#!/usr/bin/env python3
"""
Build IVF index from resources/references.json.gz.
Output: resources/ivf_index.bin

Binary format (all little-endian):
  [4B u32] K  — number of clusters
  [4B u32] D  — dimensions (14)
  [K * D * 4B f32] centroids — row-major
  [K * 4B u32] list_sizes
  [for each cluster i: list_sizes[i] * (D * 2B f16 + 1B u8)]
    — vector (f16 x 14) followed by label (u8: 0=legit, 1=fraud)
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
K = 1732
BATCH_SIZE = 50_000
N_INIT = 3
RANDOM_STATE = 42

print(f"Loading {INPUT}...", flush=True)
if not INPUT.exists():
    sys.exit(f"ERROR: {INPUT} not found. Run from the fraud-detection directory.")

with gzip.open(INPUT, "rt", encoding="utf-8") as f:
    refs = json.load(f)

vectors = np.array([r["vector"] for r in refs], dtype=np.float32)
labels = np.array([1 if r["label"] == "fraud" else 0 for r in refs], dtype=np.uint8)
N, D = vectors.shape
print(f"Loaded {N} vectors, D={D}", flush=True)

if D != 14:
    sys.exit(f"ERROR: expected D=14, got {D}")

print(
    f"Clustering K={K}, batch_size={BATCH_SIZE}, n_init={N_INIT}...",
    flush=True,
)
km = MiniBatchKMeans(
    n_clusters=K,
    batch_size=BATCH_SIZE,
    n_init=N_INIT,
    random_state=RANDOM_STATE,
    verbose=0,
)
assignments = km.fit_predict(vectors)
centroids = km.cluster_centers_.astype(np.float32)  # shape: (K, 14)

print("Building inverted lists...", flush=True)
lists: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(K)]
for i in range(N):
    lists[assignments[i]].append((vectors[i], int(labels[i])))

empty = sum(1 for lst in lists if len(lst) == 0)
if empty:
    print(f"WARNING: {empty} empty clusters", flush=True)

print(f"Writing {OUTPUT}...", flush=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "wb") as f:
    # Header
    f.write(struct.pack("<II", K, D))
    # Centroids: K * 14 * 4B f32, row-major
    for centroid in centroids:
        f.write(centroid.astype(np.dtype('<f4')).tobytes())
    # List sizes: K * 4B u32
    for lst in lists:
        f.write(struct.pack("<I", len(lst)))
    # Entries: for each cluster, each entry is 14 * 2B f16 + 1B u8
    for lst in lists:
        for vec, label in lst:
            f.write(vec.astype(np.dtype('<f2')).tobytes())
            f.write(struct.pack("B", label))

size_mb = OUTPUT.stat().st_size / 1024**2
print(f"Done. {OUTPUT} = {size_mb:.1f} MB (expected ~87 MB)", flush=True)
