"""
End-to-end pipeline runner (demo mode: uses the synthetic Markov-chain
corpus from src/simulate.py so this runs anywhere with no GPU/API access).

Run:
    python main.py

Swap `USE_REAL_ANNOTATOR` / feed in `data/raw_generations.jsonl` from
generate_hf.py once you're on the HPC cluster - see README.md for the
exact swap points.

Outputs (all written to results/):
    demo_corpus.jsonl              - raw simulated generations
    annotated_corpus.jsonl         - + predicted labels (lexical scorer)
    sequence_stats.csv             - per-generation DW / Ljung-Box / ACF
    group_summary.csv              - aggregated by model x temperature
    hypothesis_verdict.json        - headline Phase-3 result
    classifier_metrics.json        - Phase-4 held-out evaluation
    intervention_summary.json      - Phase-4 intervention experiment
    acf_by_model.png               - plot: mean ACF decay per model
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import autocorrelation as ac
from src import classifier as clf_mod
from src import intervention as iv
from src.annotate import annotate_corpus, LexicalOverlapScorer, save_jsonl as save_annotated_jsonl
from src.simulate import build_demo_corpus, save_jsonl as save_raw_jsonl

RESULTS_DIR = "results"
LABEL_KEY_FOR_STATS = "true_labels"  # use simulator ground truth for the
                                     # headline demo stats; switch to
                                     # "labels" to evaluate the annotator
                                     # itself against that ground truth.


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Phase 2 (demo stand-in): build synthetic corpus ----
    print("Building demo corpus (synthetic Markov-chain generations)...")
    corpus = build_demo_corpus(n_samples_per_config=10, length=20)
    save_raw_jsonl(corpus, os.path.join(RESULTS_DIR, "demo_corpus.jsonl"))
    print(f"  {len(corpus)} generations across "
          f"{len(set(r['model'] for r in corpus))} models x "
          f"{len(set(r['temperature'] for r in corpus))} temperatures x "
          f"{len(set(r['task_type'] for r in corpus))} task types")

    # ---- Phase 2: annotate with the lexical-overlap scorer ----
    # (facts already attached by the simulator; annotate_generation reads
    # record["facts"] directly)
    from src.annotate import annotate_generation
    print("Annotating sentences with LexicalOverlapScorer...")
    scorer = LexicalOverlapScorer()
    annotated = [annotate_generation(r, scorer=scorer) for r in corpus]
    save_annotated_jsonl(annotated, os.path.join(RESULTS_DIR, "annotated_corpus.jsonl"))

    # quick sanity check: how well does the cheap lexical scorer recover
    # the simulator's known ground-truth labels?
    agreement = np.mean([
        int(a == b)
        for rec in annotated
        for a, b in zip(rec["labels"], rec["true_labels"])
    ])
    print(f"  Lexical-scorer vs ground-truth label agreement: {agreement:.2%}")

    # ---- Phase 3: autocorrelation analysis ----
    print("Running Durbin-Watson / Ljung-Box / ACF analysis "
          f"(using '{LABEL_KEY_FOR_STATS}')...")
    df_stats = ac.analyse_corpus(annotated, label_key=LABEL_KEY_FOR_STATS, max_lag=3)
    df_stats.to_csv(os.path.join(RESULTS_DIR, "sequence_stats.csv"), index=False)

    group_summary = ac.summarise_by_group(df_stats, ["model", "temperature"])
    group_summary.to_csv(os.path.join(RESULTS_DIR, "group_summary.csv"), index=False)

    verdict = ac.hypothesis_verdict(df_stats)
    with open(os.path.join(RESULTS_DIR, "hypothesis_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print("  Verdict:", verdict["verdict"])
    print(f"  {verdict['pct_significant_ljungbox_at_alpha']:.1%} of sequences "
          f"significant at alpha={verdict['alpha']}; "
          f"mean Durbin-Watson={verdict['mean_durbin_watson']:.3f}")

    # ---- plot: mean ACF decay per model ----
    plt.figure(figsize=(6, 4))
    for model in sorted(df_stats["model"].unique()):
        sub = df_stats[(df_stats["model"] == model) & (df_stats["acf_values"].apply(len) > 0)]
        if len(sub) == 0:
            continue
        max_lag = max(len(v) for v in sub["acf_values"])
        lag_means = []
        for lag in range(max_lag):
            vals = [v[lag] for v in sub["acf_values"] if len(v) > lag]
            lag_means.append(np.mean(vals) if vals else np.nan)
        plt.plot(range(1, max_lag + 1), lag_means, marker="o", label=model)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Lag (sentences)")
    plt.ylabel("Mean autocorrelation")
    plt.title("Hallucination-label ACF decay by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "acf_by_model.png"), dpi=150)
    plt.close()

    # ---- Phase 4: early-detection classifier ----
    print("Training early-detection classifier...")
    feat_df = clf_mod.build_feature_table(annotated, k_early=3, label_key=LABEL_KEY_FOR_STATS)
    trained_clf, scaler, clf_metrics = clf_mod.train_early_detector(feat_df)
    with open(os.path.join(RESULTS_DIR, "classifier_metrics.json"), "w") as f:
        json.dump(clf_metrics, f, indent=2)
    print(f"  Held-out accuracy={clf_metrics['accuracy']:.3f} "
          f"(majority-class baseline={clf_metrics['baseline_majority_class_accuracy']:.3f}), "
          f"F1={clf_metrics['f1']:.3f}, ROC-AUC={clf_metrics['roc_auc']:.3f}")

    # ---- Phase 4: intervention experiment ----
    print("Running restart-intervention experiment...")
    iv_df = iv.run_intervention_experiment(
        annotated, trained_clf, scaler, k_early=3, risk_threshold=0.5,
        label_key=LABEL_KEY_FOR_STATS,
    )
    iv_df.to_csv(os.path.join(RESULTS_DIR, "intervention_results.csv"), index=False)
    iv_summary = iv.summarise_intervention(iv_df)
    with open(os.path.join(RESULTS_DIR, "intervention_summary.json"), "w") as f:
        json.dump(iv_summary, f, indent=2)
    print(f"  Among {iv_summary['n_intervened']} intervened generations: "
          f"hallucination rate {iv_summary['mean_halluc_rate_baseline_among_intervened']:.3f} "
          f"-> {iv_summary['mean_halluc_rate_after_among_intervened']:.3f} after restart")

    print(f"\nAll results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
