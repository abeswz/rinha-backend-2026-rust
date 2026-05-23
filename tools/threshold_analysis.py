#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "onnxruntime>=1.18"]
# ///
"""
Exp 3: Fast-path safety zone analysis.

Counts fraud-labeled vectors in references.json.gz with model score in (0.20, 0.25].
If count == 0: raising LOW from 0.20 to 0.25 introduces no FNs — safe for Exp 4b.
If count > 0: do NOT raise LOW.

Usage: uv run tools/threshold_analysis.py
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt

ROOT = Path(__file__).parent.parent
REFS_PATH = ROOT / "resources" / "references.json.gz"
MODEL_PATH = ROOT / "resources" / "model.onnx"

CURRENT_LOW = 0.20
NEW_LOW = 0.25
D = 14


def load_references() -> tuple[np.ndarray, np.ndarray]:
    print("Loading references.json.gz...", file=sys.stderr)
    vectors, labels = [], []
    with gzip.open(REFS_PATH, "rt") as f:
        content = f.read().strip()
        records = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.splitlines() if l.strip()]
    for r in records:
        vectors.append(r["vector"])
        labels.append(1 if r["label"] == "fraud" else 0)
    print(f"  {len(records)} records loaded", file=sys.stderr)
    return np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.int32)


def score_all(sess: rt.InferenceSession, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(X), batch):
        result = sess.run(None, {name: X[i:i+batch]})
        out.append(result[1][:, 1])   # P(fraud) from probability output
    return np.concatenate(out)


def main() -> None:
    sess = rt.InferenceSession(str(MODEL_PATH))
    X, y = load_references()

    print("Scoring all vectors...", file=sys.stderr)
    probs = score_all(sess, X)

    in_zone = (probs > CURRENT_LOW) & (probs <= NEW_LOW)
    fraud_in_zone = in_zone & (y == 1)
    legit_in_zone = in_zone & (y == 0)

    print(f"\n=== Threshold Analysis: LOW {CURRENT_LOW} → {NEW_LOW} ===")
    print(f"Vectors in zone ({CURRENT_LOW}, {NEW_LOW}]: {int(in_zone.sum())}")
    print(f"  Legit in zone:  {int(legit_in_zone.sum())}")
    print(f"  Fraud in zone:  {int(fraud_in_zone.sum())}")
    print()

    if int(fraud_in_zone.sum()) == 0:
        print(f"SAFE: Raising LOW to {NEW_LOW} introduces 0 FNs on the reference dataset.")
        print("Decision gate passed — proceed to Exp 4b.")
    else:
        fraud_scores = probs[fraud_in_zone]
        print(f"UNSAFE: {int(fraud_in_zone.sum())} fraud vector(s) would be fast-pathed as Legit.")
        print("Do NOT raise LOW.")
        print(f"Fraud scores in zone: min={fraud_scores.min():.4f}  max={fraud_scores.max():.4f}  mean={fraud_scores.mean():.4f}")


if __name__ == "__main__":
    main()
