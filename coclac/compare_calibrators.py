#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
So sánh 3 calibrator trên cùng claim_dataset.json:

1) Logistic (CoClaC-gốc): logistic regression trên feature φ(c).
2) Latent-truth (E2): Gaussian generative trên belief vector 5D.
3) LogicConstrainedCalibrator (E1): MLP + logic loss.

Yêu cầu:
  - coclac.py có các class:
      - OpenAILLMClient
      - ClaimExample
      - LatticeGenerator
      - BeliefElicitor
      - FeatureBuilder
      - LatentTruthCalibrator   (E2)
  - logic_calibrator.py có:
      - LogicConstrainedCalibrator  (E1)
  - claim_dataset.json đã tồn tại.

Chạy:
  python compare_calibrators.py
"""

import json
import math
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import math
from coclac import (
    OpenAILLMClient,
    ClaimExample,
    LatticeGenerator,
    BeliefElicitor,
    FeatureBuilder,
    LatentTruthCalibrator,
)
from logic_calibrator import LogicConstrainedCalibrator


# ----------------- Utils: load claim_dataset.json -----------------


def load_claim_dataset(path: str) -> List[ClaimExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[ClaimExample] = []
    for item in data:
        ex = ClaimExample(
            question=item["question"],
            answer=item["answer"],
            claim_text=item["claim_text"],
            label=int(item["label"]),
        )
        examples.append(ex)
    return examples


# ----------------- Metrics: Brier, ECE, accuracy-coverage -----------------


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def compute_ece(
    y_true: np.ndarray, y_pred: np.ndarray, num_bins: int = 10
) -> Tuple[float, dict]:
    """
    Expected Calibration Error with equal-width bins on [0, 1].
    Also return bin stats for inspection.
    """
    assert y_true.shape == y_pred.shape
    n = len(y_true)
    if n == 0:
        return math.nan, {}

    # clamp predictions
    p = np.clip(y_pred, 0.0, 1.0)

    # Bins: [0,1/num_bins), [1/num_bins, 2/num_bins), ..., [ (num_bins-1)/num_bins, 1 ]
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(p, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    ece = 0.0
    bin_stats = {
        "bin_edges": bin_edges.tolist(),
        "bins": [],  # each: {"bin_idx", "count", "avg_conf", "avg_acc"}
    }

    for b in range(num_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            bin_stats["bins"].append(
                {
                    "bin_idx": b,
                    "count": 0,
                    "avg_conf": None,
                    "avg_acc": None,
                }
            )
            continue

        conf_avg = float(p[mask].mean())
        acc_avg = float(y_true[mask].mean())
        weight = count / n
        ece += weight * abs(conf_avg - acc_avg)

        bin_stats["bins"].append(
            {
                "bin_idx": b,
                "count": count,
                "avg_conf": conf_avg,
                "avg_acc": acc_avg,
            }
        )

    return float(ece), bin_stats


def compute_accuracy_coverage_curve(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: List[float]
) -> List[dict]:
    """
    For each threshold tau, keep predictions with p >= tau.
    Return list of dicts: {"threshold", "coverage", "accuracy", "count"}
    """
    assert y_true.shape == y_pred.shape
    n = len(y_true)
    results = []

    for tau in thresholds:
        mask = y_pred >= tau
        count = int(mask.sum())
        if count == 0:
            coverage = 0.0
            accuracy = None
        else:
            coverage = count / n
            accuracy = float(y_true[mask].mean())

        results.append(
            {
                "threshold": float(tau),
                "coverage": coverage,
                "accuracy": accuracy,
                "count": count,
            }
        )

    return results


# ----------------- Build features + beliefs -----------------


def build_feats_and_beliefs(
    claim_examples: List[ClaimExample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each claim, build:
      - features φ(c) ∈ R^12
      - belief vector v(c) ∈ R^5  [p_c, p_para, p_weak, p_str, p_neg]
      - label y ∈ {0,1}

    Return:
      X_feats: (N, 12)
      X_beliefs: (N, 5)
      y: (N,)
    """
    llm = OpenAILLMClient()
    lattice_gen = LatticeGenerator(llm)
    belief_elicitor = BeliefElicitor(llm)
    feat_builder = FeatureBuilder()

    X_feats_list = []
    X_beliefs_list = []
    y_list = []

    print("Building features + beliefs for all claims...")
    for ex in tqdm(claim_examples, desc="Claims", unit="claim"):
        lattice = lattice_gen.build_lattice(ex.claim_text)
        beliefs = belief_elicitor.elicit_beliefs(lattice)
        feats = feat_builder.build_features(beliefs)

        belief_vec = np.array(
            [
                beliefs.p_original,
                beliefs.p_paraphrase,
                beliefs.p_weakening,
                beliefs.p_strengthening,
                beliefs.p_negation,
            ],
            dtype=np.float32,
        )

        X_feats_list.append(feats)
        X_beliefs_list.append(belief_vec)
        y_list.append(ex.label)

    X_feats = np.stack(X_feats_list, axis=0)
    X_beliefs = np.stack(X_beliefs_list, axis=0)
    y = np.array(y_list, dtype=np.float32)

    return X_feats, X_beliefs, y


# ----------------- Main comparison -----------------


def main():
    claim_path = "claim_dataset.json"
    claims: List[ClaimExample] = load_claim_dataset(claim_path)
    print(f"Loaded {len(claims)} claim examples from {claim_path}")

    if len(claims) == 0:
        print("No claim examples found. Aborting.")
        return

    # Shuffle & split
    random.seed(42)
    random.shuffle(claims)

    train_frac = 0.7
    split_idx = int(len(claims) * train_frac)
    train_claims = claims[:split_idx]
    test_claims = claims[split_idx:]

    print(f"Train size: {len(train_claims)}, Test size: {len(test_claims)}")

    # Build features & beliefs
    X_train_feats, X_train_beliefs, y_train = build_feats_and_beliefs(train_claims)
    X_test_feats, X_test_beliefs, y_test = build_feats_and_beliefs(test_claims)

    # Baseline constant
    p_mean = float(y_test.mean())
    brier_const = float(((p_mean - y_test) ** 2).mean())
    print(f"\nBaseline constant (p={p_mean:.3f}) Brier: {brier_const:.4f}")

    # ---------- 1) Logistic calibrator ----------
    print("\n[1] Training Logistic calibrator (CoClaC-gốc) ...")
    logreg = LogisticRegression(
        max_iter=100,
        solver="lbfgs",
    )
    logreg.fit(X_train_feats, y_train)
    y_pred_log = logreg.predict_proba(X_test_feats)[:, 1]

    brier_log = compute_brier(y_test, y_pred_log)
    ece_log, _ = compute_ece(y_test, y_pred_log, num_bins=10)
    acc_log = float(((y_pred_log >= 0.5).astype(int) == y_test).mean())

    # ---------- 2) Latent-truth calibrator (E2) ----------
    print("\n[2] Training Latent-truth calibrator (E2) ...")
    lt_calib = LatentTruthCalibrator()
    lt_calib.fit(X_train_beliefs, y_train)
    y_pred_lt = lt_calib.predict_proba(X_test_beliefs)

    brier_lt = compute_brier(y_test, y_pred_lt)
    ece_lt, _ = compute_ece(y_test, y_pred_lt, num_bins=10)
    acc_lt = float(((y_pred_lt >= 0.5).astype(int) == y_test).mean())

    # ---------- 3) LogicConstrainedCalibrator (E1) ----------
    print("\n[3] Training LogicConstrainedCalibrator (E1) ...")
    logic_calib = LogicConstrainedCalibrator(
        alpha_neg=1.0,
        alpha_weak=0.5,
        alpha_str=0.5,
        w_bce=1.0,   # có thể giữ default, nhưng ghi rõ cho dễ đọc
        w_logic=1.0,
        w_reg=0.1,
    )

    logic_calib.fit(
        V=X_train_beliefs,   # dùng belief vector 5-D
        y=y_train,
        batch_size=64,
        lr=1e-3,
        epochs=20,
    )

    y_pred_logic = logic_calib.predict_proba(X_test_beliefs)

    brier_logic = compute_brier(y_test, y_pred_logic)
    ece_logic, _ = compute_ece(y_test, y_pred_logic, num_bins=10)
    acc_logic = float(((y_pred_logic >= 0.5).astype(int) == y_test).mean())


    # ---------- Summary ----------
    print("\n=== Comparison of Calibrators (claim-level) ===")
    print(f"Test size: {len(y_test)}")
    print(f"Baseline constant Brier: {brier_const:.4f}")
    print("\nMethod\t\tBrier\t\tECE\t\tAcc@0.5")
    print(f"Logistic\t{brier_log:.4f}\t\t{ece_log:.4f}\t\t{acc_log:.4f}")
    print(f"LatentTruth\t{brier_lt:.4f}\t\t{ece_lt:.4f}\t\t{acc_lt:.4f}")
    print(f"LogicConstr\t{brier_logic:.4f}\t\t{ece_logic:.4f}\t\t{acc_logic:.4f}")

    # (Optional) accuracy-coverage curves cho từng method
    thresholds = [round(t, 2) for t in np.linspace(0.0, 1.0, 21)]

    print("\n--- Accuracy–coverage (sampled) for each method ---")
    for name, y_pred in [
        ("Logistic", y_pred_log),
        ("LatentTruth", y_pred_lt),
        ("LogicConstr", y_pred_logic),
    ]:
        print(f"\n[{name}]")
        acc_cov = compute_accuracy_coverage_curve(y_test, y_pred, thresholds)
        for item in acc_cov[::4]:
            tau = item["threshold"]
            cov = item["coverage"]
            acc = item["accuracy"]
            cnt = item["count"]
            acc_str = "None" if acc is None else f"{acc:.3f}"
            print(f"  tau={tau:.2f} | coverage={cov:.3f} | accuracy={acc_str} | n={cnt}")


if __name__ == "__main__":
    main()
