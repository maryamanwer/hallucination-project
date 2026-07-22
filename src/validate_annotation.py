
"""
Phase 2 - Manual verification of automated hallucination labels.
Computes Cohen's kappa between your manual judgments and the automated
scorer, as required by the methodology's "manual verification on a sample".
"""

import argparse
import json
import random

from sklearn.metrics import cohen_kappa_score

from src.annotate import load_jsonl


def sample_sentences_for_review(records, n_sample, seed=0):
    random.seed(seed)
    pool = []
    for rec in records:
        for sent, label, score in zip(rec["sentences"], rec["labels"], rec["scores"]):
            pool.append({
                "item_id": rec["item_id"],
                "sentence": sent,
                "facts": rec.get("facts", []),
                "auto_label": label,
                "score": score,
            })
    random.shuffle(pool)
    return pool[:n_sample]


def run_interactive_review(records, n_sample=30, out_path="data/manual_review.jsonl"):
    sample = sample_sentences_for_review(records, n_sample)
    print(f"Reviewing {len(sample)} sentences. For each, enter:")
    print("  1 = hallucinated (unsupported by / contradicts the facts)")
    print("  0 = faithful (supported by the facts)")
    print("  s = skip (uncertain / not enough context)\n")

    reviewed = []
    for i, s in enumerate(sample):
        print(f"[{i+1}/{len(sample)}] item={s['item_id']}")
        print(f"  Facts: {s['facts']}")
        print(f"  Sentence: {s['sentence']}")
        print(f"  (automated label: {s['auto_label']}, score={s['score']:.3f})")
        while True:
            resp = input("  Your label (0/1/s): ").strip().lower()
            if resp in ("0", "1", "s"):
                break
            print("  Please enter 0, 1, or s.")
        if resp != "s":
            s["manual_label"] = int(resp)
            reviewed.append(s)
        print()

    with open(out_path, "w", encoding="utf-8") as f:
        for r in reviewed:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(reviewed)} manually-reviewed sentences -> {out_path}")
    return reviewed


def compute_agreement(reviewed):
    auto = [r["auto_label"] for r in reviewed]
    manual = [r["manual_label"] for r in reviewed]
    agreement_rate = sum(a == m for a, m in zip(auto, manual)) / len(reviewed)
    kappa = cohen_kappa_score(auto, manual) if len(set(auto + manual)) > 1 else float("nan")
    return {
        "n_reviewed": len(reviewed),
        "raw_agreement_rate": agreement_rate,
        "cohens_kappa": kappa,
        "interpretation": (
            "almost perfect" if kappa >= 0.8 else
            "substantial" if kappa >= 0.6 else
            "moderate" if kappa >= 0.4 else
            "fair" if kappa >= 0.2 else
            "slight/poor"
        ) if kappa == kappa else "undefined (no label variance in sample)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/annotated_corpus.jsonl")
    parser.add_argument("--n_sample", type=int, default=30)
    parser.add_argument("--out_path", default="data/manual_review.jsonl")
    args = parser.parse_args()

    records = load_jsonl(args.in_path)
    reviewed = run_interactive_review(records, args.n_sample, args.out_path)
    if reviewed:
        result = compute_agreement(reviewed)
        print("\n--- Agreement result (report this in your methodology) ---")
        print(json.dumps(result, indent=2))
