# eval_coclac.py
import json
import math
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import joblib  # NEW: for saving calibrators

from coclac import (
    OpenAILLMClient,   # subclass LLMClient using OpenAI API
    ClaimExample,
    CoClaCPipeline,
)


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


# ----------------- Main evaluation -----------------


def main():
    # 1. Load claim dataset
    claim_path = "claim_dataset.json"
    claims: List[ClaimExample] = load_claim_dataset(claim_path)
    print(f"Loaded {len(claims)} claim examples from {claim_path}")

    if len(claims) == 0:
        print("No claim examples found. Aborting.")
        return

    # 2. Shuffle & split train/test
    random.seed(42)
    random.shuffle(claims)

    train_frac = 0.7
    split_idx = int(len(claims) * train_frac)

    # 70% train, 30% test
    train_claims = claims[:split_idx]
    test_claims = claims[split_idx:]

    print(f"Train size: {len(train_claims)}, Test size: {len(test_claims)}")

    # 3. Init LLM client & CoClaC
    llm = OpenAILLMClient()  # model lấy từ env OPENAI_MODEL (hoặc default)
    coclac = CoClaCPipeline(llm)

    # 4. Train calibrator on train set
    print("Fitting CoClaC calibrator on train set...")
    coclac.fit_calibrator(train_claims)

    # 4.1. Save trained calibrator(s) for later inference
    # Logistic calibrator (current default)
    if getattr(coclac, "calibrator", None) is not None:
        joblib.dump(coclac.calibrator, "coclac_logistic_calibrator.joblib")
        print("Saved logistic calibrator to coclac_logistic_calibrator.joblib")

    # If you later add latent-truth calibrator in CoClaCPipeline, this will save it too
    if getattr(coclac, "latent_calibrator", None) is not None:
        joblib.dump(coclac.latent_calibrator, "coclac_latent_calibrator.joblib")
        print("Saved latent-truth calibrator to coclac_latent_calibrator.joblib")

    # 5. Predict on test set (this will call LLM for each claim to build lattice + beliefs)
    y_true_list = []
    y_pred_list = []

    print("Predicting on test set...")
    for ex in tqdm(test_claims, desc="Claims", unit="claim"):
        p = coclac.predict_claim_confidence(ex.claim_text)
        y_pred_list.append(p)
        y_true_list.append(ex.label)

    y_true = np.array(y_true_list, dtype=np.float32)
    y_pred = np.array(y_pred_list, dtype=np.float32)

    # 6. Compute metrics
    brier = compute_brier(y_true, y_pred)
    ece, bin_stats = compute_ece(y_true, y_pred, num_bins=10)

    thresholds = [round(t, 2) for t in np.linspace(0.0, 1.0, 21)]  # 0.00, 0.05, ..., 1.00
    acc_cov_curve = compute_accuracy_coverage_curve(y_true, y_pred, thresholds)

    # Accuracy at threshold 0.5 (standard)
    preds_bin = (y_pred >= 0.5).astype(np.int32)
    overall_acc = float((preds_bin == y_true).mean())

    print("\n=== Evaluation results (claim-level) ===")
    print(f"Brier score: {brier:.4f}")
    print(f"ECE (10 bins): {ece:.4f}")
    print(f"Overall accuracy (threshold 0.5): {overall_acc:.4f}")

    # 7. Print a few points on accuracy-coverage curve
    print("\nAccuracy–coverage curve (sampled points):")
    for item in acc_cov_curve[::4]:  # every 4th threshold for brevity
        tau = item["threshold"]
        cov = item["coverage"]
        acc = item["accuracy"]
        cnt = item["count"]
        acc_str = "None" if acc is None else f"{acc:.3f}"
        print(f"  tau={tau:.2f} | coverage={cov:.3f} | accuracy={acc_str} | n={cnt}")

    # 8. Save detailed metrics to JSON for plotting later
    out = {
        "brier": brier,
        "ece": ece,
        "ece_bin_stats": bin_stats,
        "accuracy_coverage_curve": acc_cov_curve,
        "overall_accuracy": overall_acc,
        "n_test": len(test_claims),
    }

    out_path = "coclac_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed evaluation results to {out_path}")


if __name__ == "__main__":
    main()
