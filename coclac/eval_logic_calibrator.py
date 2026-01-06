#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from coclac import (
    OpenAILLMClient,
    ClaimExample,
    LatticeGenerator,
    BeliefElicitor,
    FeatureBuilder,
)
from logic_calibrator import LogicConstrainedCalibrator


# ---------- Load claim_dataset.json ----------

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


# ---------- Metrics (Brier, ECE, acc-coverage) ----------

def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def compute_ece(
    y_true: np.ndarray, y_pred: np.ndarray, num_bins: int = 10
) -> Tuple[float, dict]:
    assert y_true.shape == y_pred.shape
    n = len(y_true)
    if n == 0:
        return math.nan, {}

    p = np.clip(y_pred, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(p, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    ece = 0.0
    bin_stats = {"bin_edges": bin_edges.tolist(), "bins": []}

    for b in range(num_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            bin_stats["bins"].append(
                {"bin_idx": b, "count": 0, "avg_conf": None, "avg_acc": None}
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


# ---------- Build features + beliefs for all claims ----------

def build_feats_and_beliefs(
    claim_examples: List[ClaimExample]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      X_feats: (N, 12)  from FeatureBuilder
      X_beliefs: (N, 5) [p_c, p_para, p_weak, p_str, p_neg]
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


# ---------- Main ----------

def main():
    claim_path = "claim_dataset.json"
    claim_examples = load_claim_dataset(claim_path)
    print(f"Loaded {len(claim_examples)} claim examples from {claim_path}")

    if len(claim_examples) == 0:
        print("No claim examples found. Aborting.")
        return

    # Shuffle + split
    random.seed(42)
    random.shuffle(claim_examples)
    train_frac = 0.7
    split_idx = int(len(claim_examples) * train_frac)

    train_claims = claim_examples[:split_idx]
    test_claims = claim_examples[split_idx:]

    print(f"Train size: {len(train_claims)}, Test size: {len(test_claims)}")

    # Build feats/beliefs
    X_train_feats, X_train_beliefs, y_train = build_feats_and_beliefs(train_claims)
    X_test_feats, X_test_beliefs, y_test = build_feats_and_beliefs(test_claims)

    # Train logic-constrained calibrator
    calib = LogicConstrainedCalibrator(
        alpha_neg=1.0,
        alpha_weak=0.5,
        alpha_str=0.5,
    )

    print("Fitting LogicConstrainedCalibrator on train set...")
    calib.fit(
        X_feats=X_train_feats,
        y=y_train,
        X_beliefs=X_train_beliefs,
        batch_size=64,
        lr=1e-3,
        epochs=20,
    )

    # Predict on test
    print("Predicting on test set...")
    y_pred = calib.predict_proba(X_test_feats)

    # Metrics
    brier = compute_brier(y_test, y_pred)
    ece, _ = compute_ece(y_test, y_pred, num_bins=10)

    preds_bin = (y_pred >= 0.5).astype(np.int32)
    overall_acc = float((preds_bin == y_test).mean())

    # Baseline constant Brier
    p_mean = float(y_test.mean())
    brier_const = float(((p_mean - y_test) ** 2).mean())

    print("\n=== LogicConstrainedCalibrator results (claim-level) ===")
    print(f"Brier score: {brier:.4f}")
    print(f"ECE (10 bins): {ece:.4f}")
    print(f"Overall accuracy (threshold 0.5): {overall_acc:.4f}")
    print(f"Baseline constant (p={p_mean:.3f}) Brier: {brier_const:.4f}")

    thresholds = [round(t, 2) for t in np.linspace(0.0, 1.0, 21)]
    acc_cov_curve = compute_accuracy_coverage_curve(y_test, y_pred, thresholds)

    print("\nAccuracy–coverage curve (sampled points):")
    for item in acc_cov_curve[::4]:
        tau = item["threshold"]
        cov = item["coverage"]
        acc = item["accuracy"]
        cnt = item["count"]
        acc_str = "None" if acc is None else f"{acc:.3f}"
        print(f"  tau={tau:.2f} | coverage={cov:.3f} | accuracy={acc_str} | n={cnt}")


if __name__ == "__main__":
    main()
