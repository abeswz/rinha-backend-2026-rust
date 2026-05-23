#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""
Exp 2: IVF recall benchmark.

Compares brute-force top-5 KNN vs IVF (nprobe=5 fast, nprobe=24 full)
for each payload in test/test-data.json.

Usage: uv run tools/eval_recall.py
"""

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
INDEX_PATH = ROOT / "resources" / "ivf_index.bin"
REFS_PATH = ROOT / "resources" / "references.json.gz"
TEST_PATH = ROOT / "test" / "test-data.json"

K_NEIGHBORS = 5
FAST_NPROBE = 5
FULL_NPROBE = 24
SCALE = 10_000

# ── vectorization (mirrors vector.rs exactly) ──────────────────────────────

MCC_RISK = {
    5411: 0.15, 5812: 0.30, 5912: 0.20, 5944: 0.45,
    7801: 0.80, 7802: 0.75, 7995: 0.85, 4511: 0.35,
    5311: 0.25, 5999: 0.50,
}

def _round4(x: float) -> float:
    return round(x * 10000) / 10000

def _mcc_risk(mcc: int) -> float:
    return MCC_RISK.get(mcc, 0.50)

def _days_since_epoch(y: int, mo: int, d: int) -> int:
    if mo <= 2:
        y -= 1
        mo += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (mo + 1)) + d + b - 1524

def _minutes_between(cur: tuple, prev: tuple) -> float:
    dc = _days_since_epoch(cur[0], cur[1], cur[2])
    dp = _days_since_epoch(prev[0], prev[1], prev[2])
    return float((dc - dp) * 1440 + (cur[3] * 60 + cur[4]) - (prev[3] * 60 + prev[4]))

def _date_weekday(y: int, mo: int, d: int) -> int:
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    if mo < 3:
        y -= 1
    return ((y + y // 4 - y // 100 + y // 400 + t[mo - 1] + d) % 7 + 6) % 7

def vectorize(req: dict) -> np.ndarray:
    tx = req["transaction"]
    cust = req["customer"]
    merch = req["merchant"]
    term = req["terminal"]
    last = req.get("last_transaction")

    dt = tx["requested_at"]
    y, mo, d = int(dt[0:4]), int(dt[5:7]), int(dt[8:10])
    hour, minute = int(dt[11:13]), int(dt[14:16])
    weekday = _date_weekday(y, mo, d)
    cur_time = (y, mo, d, hour, minute)

    amount = float(tx["amount"])
    installments = int(tx["installments"])
    cust_avg = float(cust["avg_amount"])
    tx_count = int(cust["tx_count_24h"])
    known = set(cust["known_merchants"])
    merch_id = merch["id"]
    mcc = int(merch["mcc"])
    merch_avg = float(merch["avg_amount"])
    is_online = bool(term["is_online"])
    card_present = bool(term["card_present"])
    km_home = float(term["km_from_home"])
    is_unknown = merch_id not in known

    if last is not None:
        ts = last["timestamp"]
        prev_time = (int(ts[0:4]), int(ts[5:7]), int(ts[8:10]), int(ts[11:13]), int(ts[14:16]))
        mins_norm = _round4(min(max(_minutes_between(cur_time, prev_time) / 1440.0, 0.0), 1.0))
        km_cur_norm = _round4(min(max(float(last["km_from_current"]) / 1000.0, 0.0), 1.0))
    else:
        mins_norm = -1.0
        km_cur_norm = -1.0

    ratio = (amount / cust_avg) / 10.0 if cust_avg > 0 else 0.0

    return np.array([
        _round4(min(max(amount / 10_000.0, 0.0), 1.0)),
        _round4(min(max(installments / 12.0, 0.0), 1.0)),
        _round4(min(max(ratio, 0.0), 1.0)),
        _round4(hour / 23.0),
        _round4(weekday / 6.0),
        mins_norm,
        km_cur_norm,
        _round4(min(max(km_home / 1000.0, 0.0), 1.0)),
        _round4(min(max(tx_count / 20.0, 0.0), 1.0)),
        1.0 if is_online else 0.0,
        1.0 if card_present else 0.0,
        1.0 if is_unknown else 0.0,
        _mcc_risk(mcc),
        _round4(min(max(merch_avg / 10_000.0, 0.0), 1.0)),
    ], dtype=np.float32)

# ── IVF index loader ───────────────────────────────────────────────────────

def load_ivf_index(path: Path) -> dict:
    data = path.read_bytes()
    off = 0
    assert data[off:off+4] == b"IVF2", f"Bad magic: {data[off:off+4]!r}"
    off += 4
    n, k, d = struct.unpack_from("<III", data, off)
    off += 12
    centroids = np.frombuffer(data, dtype="<f4", count=d * k, offset=off).reshape(d, k).copy()
    off += d * k * 4
    block_offsets = np.frombuffer(data, dtype="<u4", count=k + 1, offset=off).copy()
    off += (k + 1) * 4
    total_blocks = int(block_offsets[-1])
    labels = np.frombuffer(data, dtype=np.uint8, count=total_blocks * 8, offset=off).copy()
    off += total_blocks * 8
    blocks = np.frombuffer(data, dtype="<i2", count=total_blocks * d * 8, offset=off).reshape(total_blocks, d, 8).copy()
    print(f"  IVF: n={n}, k={k}, d={d}, total_blocks={total_blocks}", file=sys.stderr)
    return {"n": n, "k": k, "d": d, "centroids": centroids, "block_offsets": block_offsets,
            "labels": labels, "blocks": blocks}

# ── brute-force top-5 ─────────────────────────────────────────────────────

def brute_force_knn5(q: np.ndarray, all_vecs: np.ndarray, all_labels: np.ndarray) -> np.ndarray:
    diffs = all_vecs - q
    dists = (diffs * diffs).sum(axis=1)
    idx = np.argpartition(dists, K_NEIGHBORS)[:K_NEIGHBORS]
    idx = idx[np.argsort(dists[idx])]
    return all_labels[idx]

# ── IVF top-5 ─────────────────────────────────────────────────────────────

def ivf_knn5(q: np.ndarray, idx: dict, nprobe: int) -> np.ndarray:
    # centroid distances
    diffs = idx["centroids"] - q[:, None]   # (d, k)
    c_dists = (diffs * diffs).sum(axis=0)   # (k,)
    top_ci = np.argpartition(c_dists, nprobe)[:nprobe]

    candidates = []  # list of (dist_f32, label_u8)
    for ci in top_ci:
        bs, be = int(idx["block_offsets"][ci]), int(idx["block_offsets"][ci + 1])
        if bs == be:
            continue
        # blocks[bs:be] shape (blocks, d, 8); dequantize
        blk = idx["blocks"][bs:be].astype(np.float32) * (1.0 / SCALE)
        # transpose to (blocks, 8, d) then flatten to (blocks*8, d)
        vecs = blk.transpose(0, 2, 1).reshape(-1, 14)
        lbl = idx["labels"][bs * 8:be * 8]
        diffs2 = vecs - q
        dists2 = (diffs2 * diffs2).sum(axis=1)
        for dist, label in zip(dists2, lbl):
            candidates.append((float(dist), int(label)))

    candidates.sort(key=lambda x: x[0])
    top5 = np.array([c[1] for c in candidates[:K_NEIGHBORS]], dtype=np.uint8)
    if len(top5) < K_NEIGHBORS:
        top5 = np.pad(top5, (0, K_NEIGHBORS - len(top5)))
    return top5

# ── fraud decision: ≥3 fraud in top-5 ─────────────────────────────────────

def is_fraud(labels: np.ndarray) -> bool:
    return int(labels.sum()) >= 3

# ── reference loader ───────────────────────────────────────────────────────

def load_references() -> tuple[np.ndarray, np.ndarray]:
    print("Loading references.json.gz...", file=sys.stderr)
    vectors, labels = [], []
    with gzip.open(REFS_PATH, "rt") as f:
        content = f.read().strip()
        records = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.splitlines() if l.strip()]
    for r in records:
        vectors.append(r["vector"])
        labels.append(1 if r["label"] == "fraud" else 0)
    print(f"  {len(vectors)} reference vectors", file=sys.stderr)
    return np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.uint8)

# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ivf = load_ivf_index(INDEX_PATH)
    all_vecs, all_labels = load_references()

    print("Loading test-data.json...", file=sys.stderr)
    test_entries = json.loads(TEST_PATH.read_text())["entries"]
    print(f"  {len(test_entries)} test entries", file=sys.stderr)

    total = fast_agree = full_agree = fast_flips = full_flips = 0
    flip_details = []

    for entry in test_entries:
        req = entry["request"]
        q = vectorize(req)

        bf = brute_force_knn5(q, all_vecs, all_labels)
        fast = ivf_knn5(q, ivf, FAST_NPROBE)
        full = ivf_knn5(q, ivf, FULL_NPROBE)

        total += 1
        if np.array_equal(fast, bf):
            fast_agree += 1
        if np.array_equal(full, bf):
            full_agree += 1

        if is_fraud(fast) != is_fraud(bf):
            fast_flips += 1
            flip_details.append({
                "id": req["id"],
                "bf": is_fraud(bf), "fast": is_fraud(fast), "full": is_fraud(full),
                "bf_n": int(bf.sum()), "fast_n": int(fast.sum()), "full_n": int(full.sum()),
            })
        if is_fraud(full) != is_fraud(bf):
            full_flips += 1

    print(f"\n=== Recall Results (N={total}) ===")
    print(f"nprobe={FAST_NPROBE} exact label match: {fast_agree}/{total} ({fast_agree/total*100:.1f}%)")
    print(f"nprobe={FULL_NPROBE} exact label match: {full_agree}/{total} ({full_agree/total*100:.1f}%)")
    print(f"nprobe={FAST_NPROBE} decision flips vs brute-force: {fast_flips}")
    print(f"nprobe={FULL_NPROBE} decision flips vs brute-force: {full_flips}")

    if flip_details:
        print(f"\n=== Fast-probe decision flips ({len(flip_details)} cases) ===")
        for d in flip_details[:20]:
            print(f"  {d['id']}: bf={d['bf']} fast={d['fast']} full={d['full']}  "
                  f"(bf_count={d['bf_n']} fast_count={d['fast_n']} full_count={d['full_n']})")


if __name__ == "__main__":
    main()
