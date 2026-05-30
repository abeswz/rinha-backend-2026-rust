# LightGBM Hybrid Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the IVF full probe (nprobe=24) with a cost-aware LightGBM model for ambiguous cases (fraud_count=2 or 3), reducing E=FP+3×FN and improving detection_score from 1477 toward 2500+.

**Architecture:** IVF fast probe (nprobe=8) handles clear cases (count=0,1,4,5). For count=2 or 3, a LightGBM model trained with class_weight={0:1,1:3} decides using threshold=0.25 (Bayes-optimal for asymmetric cost). Full probe is eliminated from hot path.

**Tech Stack:** Python/LightGBM/m2cgen (training), Rust/AVX2 (runtime). No new Rust dependencies.

**Spec:** `docs/superpowers/specs/2026-05-30-lightgbm-hybrid-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/train_model.py` | Modify | Add class_weight, replace threshold assertions with E-score assertion |
| `src/fraud/model_gen.rs` | Create (generated) | m2cgen Rust export of trained LightGBM — do not edit manually |
| `src/fraud/mod.rs` | Modify | Expose `model_gen` as public submodule |
| `src/fraud/knn.rs` | Modify | Call LightGBM for count=2,3 instead of full probe |

---

## Task 1: Update train_model.py for cost-aware training

**Files:**
- Modify: `tools/train_model.py`

- [ ] **Step 1: Add `class_weight` to LGBMClassifier constructor**

In `tools/train_model.py`, find the `train()` function. Replace the model constructor (lines ~82-90):

```python
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        num_leaves=63,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        class_weight={0: 1, 1: 3},
    )
```

- [ ] **Step 2: Replace threshold assertions with E-score validation**

In `tools/train_model.py`, inside `train()`, replace the validation block (lines ~97-107):

```python
    # Before:
    # fp = ((probs >= 0.95) & (y_val == 0)).sum()
    # fn = ((probs <= 0.20) & (y_val == 1)).sum()
    # uncertain = ((probs > 0.20) & (probs < 0.95)).sum()
    # edge_pct = uncertain / len(y_val) * 100
    # print(f"FP (p>=0.95, label=0): {fp}", file=sys.stderr)
    # print(f"FN (p<=0.20, label=1): {fn}", file=sys.stderr)
    # print(f"Edge case rate: {edge_pct:.1f}%", file=sys.stderr)
    # assert fp == 0, f"FP count {fp} > 0 at p>=0.95"
    # assert fn == 0, f"FN count {fn} > 0 at p<=0.20"
    # assert edge_pct <= 15.0, f"Edge rate {edge_pct:.1f}% > 15%"

    # After:
    pred_fraud = probs >= 0.25
    fp_count = int(((pred_fraud) & (y_val == 0)).sum())
    fn_count = int(((~pred_fraud) & (y_val == 1)).sum())
    E_val = fp_count + 3 * fn_count
    print(f"Val E at t=0.25 (FP+3×FN): {E_val}  (FP={fp_count}, FN={fn_count})", file=sys.stderr)
    assert auc >= 0.99, f"AUC {auc:.4f} < 0.99"
    assert E_val < 200, f"E={E_val} >= 200 on val set — model too weak"
```

- [ ] **Step 3: Verify script is syntactically valid**

```bash
uv run python -c "import ast; ast.parse(open('tools/train_model.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tools/train_model.py
git commit -m "feat(train): cost-aware LightGBM with class_weight 1:3

Add class_weight={0:1, 1:3} to encode E=FP+3×FN scoring formula.
Replace hard threshold assertions (p<=0.20, p>=0.95) with E-score
assertion at optimal threshold=0.25."
```

---

## Task 2: Retrain model and generate model_gen.rs

**Files:**
- Create: `src/fraud/model_gen.rs` (generated)
- Update: `resources/model.onnx` (generated)

- [ ] **Step 1: Run training**

```bash
uv run tools/train_model.py
```

Expected output (approximate):
```
Loaded 3000000 records
Val AUC-ROC: 0.99xxxx
Val E at t=0.25 (FP+3×FN): NNN  (FP=XX, FN=YY)
Wrote resources/model.onnx (XXX KB)
ONNX validation passed
Wrote src/fraud/model_gen.rs (XXX KB, NNNN branches)
Done.
```

Gate: E_val printed must be < 200. If assertion fires, training failed — do not proceed.

- [ ] **Step 2: Verify model_gen.rs was created with expected exports**

```bash
grep "pub fn score\|pub fn sigmoid" src/fraud/model_gen.rs
```

Expected:
```
pub fn score(input: &[f64]) -> f64 {
pub fn sigmoid(x: f64) -> f64 {
```

Both functions must be present and public. If either is missing, check `export_rust()` in `train_model.py`.

- [ ] **Step 3: Commit generated file**

```bash
git add src/fraud/model_gen.rs resources/model.onnx
git commit -m "feat(model): retrain LightGBM with class_weight 1:3

Generated model_gen.rs via m2cgen. E_val=<INSERT_ACTUAL_E> at t=0.25
on validation set."
```

---

## Task 3: Expose model_gen module in Rust

**Files:**
- Modify: `src/fraud/mod.rs`

- [ ] **Step 1: Write a test that will fail until model_gen is accessible**

In `src/fraud/knn.rs`, inside the existing `#[cfg(test)] mod tests` block, add:

```rust
#[test]
fn model_gen_sanity() {
    // All-ones = high-risk features → P(fraud) should be > 0.5
    let fraud_q = [1.0f64; 14];
    let p_fraud = crate::fraud::model_gen::sigmoid(
        crate::fraud::model_gen::score(&fraud_q)
    );
    assert!(p_fraud > 0.5, "all-ones features got P(fraud)={p_fraud:.4f}, expected > 0.5");

    // All-zeros = low-risk features → P(fraud) should be < 0.5
    let legit_q = [0.0f64; 14];
    let p_legit = crate::fraud::model_gen::sigmoid(
        crate::fraud::model_gen::score(&legit_q)
    );
    assert!(p_legit < 0.5, "all-zeros features got P(fraud)={p_legit:.4f}, expected < 0.5");
}
```

- [ ] **Step 2: Run test to verify it fails (model_gen not exposed yet)**

```bash
cargo test model_gen_sanity 2>&1 | tail -5
```

Expected: compile error `error[E0433]: failed to resolve: could not find 'model_gen'`

- [ ] **Step 3: Add model_gen to mod.rs**

In `src/fraud/mod.rs`, add one line:

```rust
pub mod data;
pub mod json;
pub mod knn;
pub mod model_gen;
pub mod vector;
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test model_gen_sanity -- --nocapture
```

Expected:
```
test fraud::knn::tests::model_gen_sanity ... ok
```

- [ ] **Step 5: Commit**

```bash
git add src/fraud/mod.rs src/fraud/knn.rs
git commit -m "feat(model): expose model_gen module and add sanity test"
```

---

## Task 4: Replace full probe with LightGBM in knn5_ivf

**Files:**
- Modify: `src/fraud/knn.rs`

- [ ] **Step 1: Write a failing test for the hybrid behavior**

In `src/fraud/knn.rs`, inside the `#[cfg(test)] mod tests` block, add:

```rust
#[test]
fn lgbm_decides_ambiguous_cases() {
    data::init();
    let ds = data::dataset();

    // Run enough different queries to hit count=2 or count=3 from fast probe.
    // If LightGBM path is wired, result must still be 0..=5.
    // This is a regression + integration test: it panics if LightGBM panics.
    let test_queries: &[[f32; 14]] = &[
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 0.5, 0.5],
        [0.3, 0.1, 0.3, 0.6, 0.4, 0.2, 0.1, 0.3, 0.4, 0.0, 1.0, 0.0, 0.3, 0.1],
        [0.8, 0.0, 0.8, 0.3, 0.7, 0.5, 0.5, 0.8, 0.3, 1.0, 1.0, 1.0, 0.8, 0.0],
    ];
    for q in test_queries {
        let result = knn5_ivf(q, ds);
        assert!(result <= 5, "knn5_ivf returned {result} (must be 0..=5)");
        // Result must be 2 or 3 only (the LightGBM branch returns 2 or 3),
        // OR 0,1,4,5 from the fast-path. Either is valid.
    }
}
```

- [ ] **Step 2: Run test to verify it currently passes (baseline — not yet using LightGBM)**

```bash
cargo test lgbm_decides_ambiguous_cases -- --nocapture
```

Expected: PASS (current code still works, test is a regression guard)

- [ ] **Step 3: Implement LightGBM hybrid in knn5_ivf**

In `src/fraud/knn.rs`, replace the `knn5_ivf` function (lines 12-21):

```rust
pub fn knn5_ivf(q: &[f32; 14], ds: &Dataset) -> u8 {
    let fast = probe(q, ds, FAST_NPROBE);
    let fraud_count = count_fraud(fast);
    if fraud_count == 2 || fraud_count == 3 {
        let q_f64: [f64; 14] = std::array::from_fn(|i| q[i] as f64);
        let p_fraud = super::model_gen::sigmoid(super::model_gen::score(&q_f64));
        if p_fraud >= 0.25 { 3 } else { 2 }
    } else {
        fraud_count as u8
    }
}
```

- [ ] **Step 4: Run all tests**

```bash
cargo test 2>&1 | tail -10
```

Expected:
```
test result: ok. N passed; 0 failed; 0 ignored
```

All existing smoke tests (`smoke_zero_query`, `smoke_fraud_heavy_query`, `count_fraud_correct`, `top_n_centroids_fast_smallest_first`) must pass.

- [ ] **Step 5: Verify clippy is clean**

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | grep "^error" | head -10
```

Expected: no output (no errors)

- [ ] **Step 6: Commit**

```bash
git add src/fraud/knn.rs
git commit -m "feat(knn): replace full probe with LightGBM for ambiguous cases

For fraud_count=2 or 3 from fast probe (nprobe=8), call LightGBM
with threshold=0.25 instead of running full probe (nprobe=24).
Threshold 0.25 is Bayes-optimal for E=FP+3×FN scoring formula."
```

---

## Task 5: End-to-end validation and benchmark

**Files:** none changed — validation only

- [ ] **Step 1: Measure recall vs brute-force**

```bash
uv run tools/eval_recall.py 2>&1
```

Expected (approximate):
```
=== Recall Results (N=54100) ===
nprobe=8 decision flips vs brute-force: < 150
```

Note: eval_recall compares IVF vs brute-force, NOT LightGBM vs brute-force. The flip count may be similar to pre-LightGBM — that is expected. The LightGBM improvement shows up in `make bench` against ground truth labels.

- [ ] **Step 2: Run local benchmark**

```bash
make bench
```

Inspect `test/results.json` after bench completes:

```bash
python3 -c "
import json
r = json.load(open('test/results.json'))
s = r['scoring']
print(f'FP={s[\"breakdown\"][\"false_positive_detections\"]}')
print(f'FN={s[\"breakdown\"][\"false_negative_detections\"]}')
print(f'E={s[\"weighted_errors_E\"]}')
print(f'detection_score={s[\"detection_score\"][\"value\"]}')
print(f'final_score={s[\"final_score\"]}')
print(f'p99={r[\"p99\"]}')
"
```

**Gate — must pass before submission:**
- `detection_score > 2000` (was 1477)
- `p99_score = 3000` (must not regress — p99 ≤ 0.99ms)
- `final_score > 5000`

- [ ] **Step 3: If gate passes — submit**

```bash
make submission
```

Then open new remote test issue.

- [ ] **Step 4: If gate fails — diagnose**

If `detection_score` did not improve:
- Check `FP` and `FN` counts. If both increased vs baseline (FP=76, FN=81), the LightGBM model is too weak or threshold is wrong.
- Re-run `uv run tools/train_model.py` and check E_val output. If E_val ≥ 200, model assertion caught it.
- Try lowering threshold from 0.25 to 0.20: change `p_fraud >= 0.25` to `p_fraud >= 0.20` in `knn.rs`.

If `p99_score` regressed (p99 > 1ms):
- LightGBM inference is too slow for the ambiguous case rate under load.
- Profile: add timing around the LightGBM call, measure µs per call.
- Consider reverting to full probe for p99 safety.
