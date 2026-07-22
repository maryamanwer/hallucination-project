%%writefile /content/hallucination_project/src/coverage_report.py
"""
Coverage report - proves (with numbers, not just code) that all four task
categories in the project proposal were actually generated, annotated, and
analysed, not just theoretically supported by the framework.
"""

import argparse
import pandas as pd

from src.annotate import load_jsonl
from src.dataset import load_dataset

TASK_TYPES = ["biographical_qa", "scientific_summarisation",
              "multi_hop_reasoning", "long_form_generation"]


def build_coverage_report(records, dataset_items):
    df = pd.DataFrame(records)
    items_by_type = {}
    for item in dataset_items:
        items_by_type.setdefault(item["task_type"], set()).add(item["id"])

    rows = []
    for task_type in TASK_TYPES:
        sub = df[df["task_type"] == task_type] if "task_type" in df.columns else df.iloc[0:0]
        n_items_defined = len(items_by_type.get(task_type, set()))
        n_items_covered = sub["item_id"].nunique() if len(sub) else 0
        rows.append({
            "task_type": task_type,
            "n_items_defined_in_dataset": n_items_defined,
            "n_items_with_generations": n_items_covered,
            "n_generations": len(sub),
            "n_models_used": sub["model"].nunique() if len(sub) else 0,
            "n_temperatures_used": sub["temperature"].nunique() if len(sub) else 0,
            "mean_sentence_count": float(sub["sentences"].apply(len).mean()) if len(sub) else 0.0,
            "mean_hallucination_rate": float(
                sub["labels"].apply(lambda l: sum(l) / len(l) if l else 0.0).mean()
            ) if len(sub) else 0.0,
            "fully_covered": bool(n_items_covered >= n_items_defined and n_items_defined > 0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/annotated_corpus.jsonl")
    parser.add_argument("--out_path", default="results_real/coverage_report")
    args = parser.parse_args()

    records = load_jsonl(args.in_path)
    dataset_items = load_dataset()
    report = build_coverage_report(records, dataset_items)
    report.to_csv(args.out_path + ".csv", index=False)
    report.to_json(args.out_path + ".json", orient="records", indent=2)
    print(report.to_string(index=False))
    all_covered = report["fully_covered"].all()
    print("\n" + ("✓ ALL FOUR TASK CATEGORIES EVALUATED" if all_covered
                   else "✗ NOT ALL TASK CATEGORIES ARE COVERED YET"))
    print(f"Saved -> {args.out_path}.csv / .json")


if __name__ == "__main__":
    main()
