#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "lightgbm>=4.3",
#   "scikit-learn>=1.4",
#   "skl2onnx>=1.16",
#   "onnxmltools>=1.12",
#   "numpy>=1.26",
#   "onnxruntime>=1.18",
# ]
# ///
"""Train LightGBM fraud classifier, export to resources/model.onnx."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import onnxmltools.convert.lightgbm.shape_calculators.Classifier as _lgb_clf_module
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm as lgb_converter
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxmlFloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType

# onnxmltools shape calculator checks input types against its own FloatTensorType,
# but convert_sklearn passes skl2onnx's FloatTensorType. Patch the check to accept both.
_orig_type_check = _lgb_clf_module.check_input_and_output_types


def _patched_type_check(operator, good_input_types=None, good_output_types=None):
    if good_input_types is not None:
        extended = list(good_input_types)
        if OnnxmlFloatTensorType in extended and FloatTensorType not in extended:
            extended.append(FloatTensorType)
        good_input_types = extended
    return _orig_type_check(operator, good_input_types=good_input_types, good_output_types=good_output_types)


_lgb_clf_module.check_input_and_output_types = _patched_type_check

from onnxmltools.convert.lightgbm.shape_calculators.Classifier import (  # noqa: E402
    calculate_lightgbm_classifier_output_shapes,
)
import onnxruntime as rt

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "resources" / "references.json.gz"
OUTPUT = ROOT / "resources" / "model.onnx"
D = 14


def load_data() -> tuple[np.ndarray, np.ndarray]:
    print("Loading data...", file=sys.stderr)
    vectors, labels = [], []
    with gzip.open(INPUT, "rt") as f:
        content = f.read().strip()
        # Handle both array JSON and newline-delimited JSON
        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
    for r in records:
        vectors.append(r["vector"])
        labels.append(1 if r["label"] == "fraud" else 0)
    print(f"Loaded {len(vectors)} records", file=sys.stderr)
    X = np.array(vectors, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    assert X.shape[1] == D, f"Expected {D} dims, got {X.shape[1]}"
    return X, y


def train(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        num_leaves=63,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    print(f"Val AUC-ROC: {auc:.6f}", file=sys.stderr)
    assert auc >= 0.99, f"AUC {auc:.4f} < 0.99 — model too weak"

    # Validate threshold behaviour on validation set
    fp = ((probs >= 0.95) & (y_val == 0)).sum()
    fn = ((probs <= 0.20) & (y_val == 1)).sum()
    uncertain = ((probs > 0.20) & (probs < 0.95)).sum()
    edge_pct = uncertain / len(y_val) * 100
    print(f"FP (p>=0.95, label=0): {fp}", file=sys.stderr)
    print(f"FN (p<=0.20, label=1): {fn}", file=sys.stderr)
    print(f"Edge case rate: {edge_pct:.1f}%", file=sys.stderr)
    assert fp == 0, f"FP count {fp} > 0 at p>=0.95"
    assert fn == 0, f"FN count {fn} > 0 at p<=0.20"
    assert edge_pct <= 15.0, f"Edge rate {edge_pct:.1f}% > 15%"
    return model


def export_onnx(model: lgb.LGBMClassifier) -> None:
    update_registered_converter(
        lgb.LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_lightgbm_classifier_output_shapes,
        lgb_converter,
        options={"nocl": [True, False], "zipmap": [True, False]},
    )
    initial_type = [("float_input", FloatTensorType([None, D]))]
    options = {lgb.LGBMClassifier: {"zipmap": False}}
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        options=options,
        target_opset={"": 12, "ai.onnx.ml": 3},
    )
    OUTPUT.write_bytes(onx.SerializeToString())
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)", file=sys.stderr)


def validate_onnx() -> None:
    sess = rt.InferenceSession(str(OUTPUT))
    input_name = sess.get_inputs()[0].name
    # clearly fraudulent vector (all high-risk values)
    fraud_vec = np.array([[1.0] * D], dtype=np.float32)
    out = sess.run(None, {input_name: fraud_vec})
    p_fraud = out[1][0][1]
    assert p_fraud >= 0.70, f"Fraud test vector got p_fraud={p_fraud:.4f} < 0.70"
    # clearly legit vector (all zeros)
    legit_vec = np.zeros((1, D), dtype=np.float32)
    out = sess.run(None, {input_name: legit_vec})
    p_fraud_legit = out[1][0][1]
    assert p_fraud_legit <= 0.30, f"Legit test vector got p_fraud={p_fraud_legit:.4f} > 0.30"
    print("ONNX validation passed", file=sys.stderr)


if __name__ == "__main__":
    X, y = load_data()
    model = train(X, y)
    export_onnx(model)
    validate_onnx()
    print("Done.", file=sys.stderr)
