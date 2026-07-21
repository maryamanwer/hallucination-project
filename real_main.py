"""
Runs Phases 3 and 4 on a REAL annotated corpus (output of generate_api.py
/ generate_hf.py + annotate.py), instead of the synthetic demo data used
by main.py.

Usage:
    python real_main.py --in_path data/annotated_corpus.jsonl \
        --out_dir results_real

With a small pilot corpus (a handful of items x a couple of models x a
couple of temperatures, ~20-50 generations), expect:
    - autocorrelation.py output (Durbin-Watson / Ljung-Box / ACF / bursts)
      to run fine on any number of sequences, though the hypothesis
      verdict will have low statistical power until you have more data.
    - classifier.py / intervention.py to possibly fail or produce
      unreliable metrics if your corpus is too small or too one-sided
      (e.g. all sequences have a remainder hallucination, or none do).
      This script catches that and tells you plainly rather than
      producing misleading numbers.

To get a corpus large enough for reliable Phase 3/4 results, use
--n_samples on generate_api.py to repeat generations per item (adds
variance across the same prompt/temperature), and sweep across all
TEMPERATURES and both model scales in src/config.py.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import autocorrelation as ac
from src import classifier as clf_mod
from src import intervention as iv
from src.annotate import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/annotated_corpus.jsonl")
    parser.add_argument("--out_dir", default="results_real")
    parser.add_argument("--k_early", type=int, default=2,
                         help="How many early sentences the Phase-4 "
                              "classifier looks at. Keep this small (2-3) "
                              "for short pilot generations.")
    parser.add_argument("--risk_threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_jsonl(args.in_path)
    print(f"Loaded {len(records)} annotated generations from {args.in_path}")

    if len(records) < 10:
        print("WARNING: fewer than 10 generations - Phase 3/4 results below "
              "are a pilot sanity-check only, not dissertation-grade "
              "evidence. Generate more (--n_samples, more temperatures/"
              "models) before drawing conclusions.")

    # ---- Phase 3: autocorrelation analysis ----
    print("\n--- Phase 3: autocorrelation analysis ---")
    df_stats = ac.analyse_corpus(records, label_key="labels", max_lag=3, min_length=4)
    df_stats.to_csv(os.path.join(args.out_dir, "sequence_stats.csv"), index=False)
    print(f"Analysed {len(df_stats)} sequences "
          f"({df_stats['durbin_watson'].notna().sum()} long enough for DW/Ljung-Box)")

    if df_stats["model"].nunique() > 0:
        group_summary = ac.summarise_by_group(df_stats, ["model", "temperature"])
        group_summary.to_csv(os.path.join(args.out_dir, "group_summary.csv"), index=False)
        print(group_summary.to_string(index=False))

    verdict = ac.hypothesis_verdict(df_stats)
    with open(os.path.join(args.out_dir, "hypothesis_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print("\nVerdict:", verdict["verdict"])
    print(json.dumps(verdict, indent=2))

    has_acf = df_stats["acf_values"].apply(len).sum() > 0
    if has_acf:
        plt.figure(figsize=(6, 4))
        for model in sorted(df_stats["model"].dropna().unique()):
            sub = df_stats[(df_stats["model"] == model) & (df_stats["acf_values"].apply(len) > 0)]
            if len(sub) == 0:
                continue
            max_lag = max(len(v) for v in sub["acf_values"])
            lag_means = [
                np.mean([v[lag] for v in sub["acf_values"] if len(v) > lag])
                for lag in range(max_lag)
            ]
            plt.plot(range(1, max_lag + 1), lag_means, marker="o", label=str(model))
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.xlabel("Lag (sentences)")
        plt.ylabel("Mean autocorrelation")
        plt.title("Hallucination-label ACF decay (real corpus)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "acf_by_model.png"), dpi=150)
        plt.close()
        print(f"Saved ACF plot -> {args.out_dir}/acf_by_model.png")
    else:
        print("Skipped ACF plot: no sequence had enough sentences for a lag-1 ACF value.")

    print("\n--- Phase 4: early-detection classifier ---")
    try:
        feat_df = clf_mod.build_feature_table(records, k_early=args.k_early, label_key="labels")
        print(f"{len(feat_df)} examples with a non-trivial remainder after "
              f"the first {args.k_early} sentences")
        trained_clf, scaler, clf_metrics = clf_mod.train_early_detector(feat_df)
        with open(os.path.join(args.out_dir, "classifier_metrics.json"), "w") as f:
            json.dump(clf_metrics, f, indent=2)
        print(json.dumps(clf_metrics, indent=2))
    except (ValueError, KeyError) as e:
        print(f"Skipped classifier training: {e}")
        print("This usually means your corpus is too small, or every "
              "sequence has the same outcome. Generate more data with "
              "varied temperatures/models to fix this.")
        trained_clf = scaler = None

    if trained_clf is not None:
        print("\n--- Phase 4: intervention experiment ---")
        print("Classifier trained successfully. To run the real restart "
              "intervention, implement a regen_oracle in intervention.py "
              "that calls generate_api.call_api() on the truncated prompt, "
              "then call iv.run_intervention_experiment(...) with it.")

    print(f"\nAll available results written to {args.out_dir}/")


if __name__ == "__main__":
    main()
