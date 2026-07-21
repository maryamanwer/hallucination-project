"""
Phase 4 - Early-Detection Classifier.

Trains a lightweight classifier that looks only at the first `k_early`
sentences of a generation and predicts whether the REMAINDER of the
generation will contain at least one hallucination. If autocorrelation
is confirmed (Phase 3), early hallucination signal should be predictive
of downstream hallucination risk - this is the practical payoff the
project aims to test.

Feature set (all computable from text + labels only, i.e. no logprobs
required - see the note in `extract_features` for what to add when you
have real token-level access via HF `output_scores=True`):
    - hallucination_rate_early: fraction of first k sentences hallucinated
    - any_halluc_early: whether any of the first k sentences hallucinated
    - last_label_early: label of the k-th sentence (recency feature)
    - run_length_at_k: length of the current run (streak) at position k
    - mean_sentence_length_early: mean token count of first k sentences
    - lexical_diversity_early: type-token ratio of first k sentences

When you move to real LLM inference (generate_hf.py with
output_scores=True), add:
    - mean_token_entropy_early: mean predictive entropy over the first
      k sentences' tokens (strong hallucination-risk signal in the
      literature; requires logits, not available from text alone)
    - mean_top1_prob_early: mean top-1 token probability
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _run_length_at(labels: List[int], k: int) -> int:
    if k == 0:
        return 0
    val = labels[k - 1]
    length = 0
    for label in reversed(labels[:k]):
        if label == val:
            length += 1
        else:
            break
    return length


def extract_features(record: dict, k_early: int, label_key: str = "labels") -> Dict:
    labels = record[label_key]
    sentences = record["sentences"]
    k = min(k_early, len(labels))
    early_labels = labels[:k]
    early_sentences = sentences[:k]

    words = " ".join(early_sentences).split()
    ttr = len(set(w.lower() for w in words)) / len(words) if words else 0.0

    return {
        "item_id": record.get("item_id"),
        "model": record.get("model"),
        "task_type": record.get("task_type"),
        "temperature": record.get("temperature"),
        "hallucination_rate_early": float(np.mean(early_labels)) if early_labels else 0.0,
        "any_halluc_early": int(any(early_labels)),
        "last_label_early": early_labels[-1] if early_labels else 0,
        "run_length_at_k": _run_length_at(labels, k),
        "mean_sentence_length_early": float(np.mean([len(s.split()) for s in early_sentences]))
                                       if early_sentences else 0.0,
        "lexical_diversity_early": ttr,
        # target: does the REMAINDER (after position k) contain a hallucination?
        "target_remainder_has_halluc": int(any(labels[k:])) if len(labels) > k else 0,
        "has_remainder": int(len(labels) > k),
    }


def build_feature_table(records: List[dict], k_early: int = 3,
                         label_key: str = "labels") -> pd.DataFrame:
    rows = [extract_features(r, k_early, label_key) for r in records]
    df = pd.DataFrame(rows)
    # only sequences with a non-trivial remainder are meaningful examples
    return df[df["has_remainder"] == 1].reset_index(drop=True)


FEATURE_COLS = [
    "hallucination_rate_early", "any_halluc_early", "last_label_early",
    "run_length_at_k", "mean_sentence_length_early", "lexical_diversity_early",
]


def train_early_detector(df: pd.DataFrame, test_size: float = 0.25,
                          random_state: int = 0) -> Tuple[LogisticRegression, StandardScaler, Dict]:
    """Train logistic regression early-detection classifier; return the
    fitted model, fitted scaler, and held-out evaluation metrics."""
    X = df[FEATURE_COLS].values
    y = df["target_remainder_has_halluc"].values

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Target has only one class in this data - need both remainder-"
            "has-hallucination and remainder-clean examples to train/evaluate."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    metrics = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else float("nan"),
        "baseline_majority_class_accuracy": float(max(np.mean(y_test), 1 - np.mean(y_test))),
        "feature_coefficients": dict(zip(FEATURE_COLS, clf.coef_[0].tolist())),
    }
    return clf, scaler, metrics


def predict_risk(clf: LogisticRegression, scaler: StandardScaler,
                  features: Dict) -> float:
    x = np.array([[features[c] for c in FEATURE_COLS]])
    return float(clf.predict_proba(scaler.transform(x))[0, 1])
