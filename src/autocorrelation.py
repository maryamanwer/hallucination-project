"""
Phase 3 - Autocorrelation Analysis.

Formally tests whether hallucination label sequences (binary time series,
1=hallucinated, 0=faithful, indexed by sentence position within a single
generation) are temporally autocorrelated, and characterises the
clustering structure (burst length, inter-burst interval).

Three complementary tests, as specified in the project definition:

1. Durbin-Watson statistic - tests first-order autocorrelation of the
   label sequence treated as residuals of its own mean. DW ~ 2 means no
   autocorrelation; DW < 2 means positive autocorrelation (clustering);
   DW > 2 means negative autocorrelation (alternating).

2. Ljung-Box test - joint test for autocorrelation up to lag h (captures
   structure beyond just lag 1). Small p-value -> reject the null of no
   autocorrelation.

3. Lag-k autocorrelation function (ACF) - the actual correlation between
   the sequence and its lag-k shifted version, for k = 1..K, showing how
   far the "memory" of a hallucination extends.

Sequences shorter than `min_length` are skipped (Durbin-Watson and
Ljung-Box are not meaningful/stable on very short series).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


@dataclass
class SequenceStats:
    item_id: str
    model: str
    temperature: float
    task_type: str
    length: int
    durbin_watson: Optional[float]
    ljung_box_stat: Optional[float]
    ljung_box_pvalue: Optional[float]
    acf_values: List[float] = field(default_factory=list)
    burst_lengths: List[int] = field(default_factory=list)
    inter_burst_intervals: List[int] = field(default_factory=list)
    mean_label: float = 0.0


def run_length_encode(labels: List[int]) -> List[Tuple[int, int]]:
    """Return list of (value, run_length) for consecutive runs."""
    if not labels:
        return []
    runs = []
    current_val, current_len = labels[0], 1
    for v in labels[1:]:
        if v == current_val:
            current_len += 1
        else:
            runs.append((current_val, current_len))
            current_val, current_len = v, 1
    runs.append((current_val, current_len))
    return runs


def burst_statistics(labels: List[int]) -> Tuple[List[int], List[int]]:
    """Extract burst lengths (runs of 1s) and inter-burst intervals
    (runs of 0s between bursts) from a label sequence."""
    runs = run_length_encode(labels)
    burst_lengths = [length for val, length in runs if val == 1]
    inter_burst = [length for val, length in runs if val == 0]
    return burst_lengths, inter_burst


def analyse_sequence(labels: List[int], max_lag: int = 3,
                      min_length: int = 5) -> Dict:
    """Run DW, Ljung-Box, and ACF on a single binary label sequence.

    Returns a dict of results; DW/Ljung-Box entries are None if the
    sequence is too short or has zero variance (all 0s or all 1s), since
    autocorrelation is undefined for a constant series.
    """
    n = len(labels)
    arr = np.asarray(labels, dtype=float)
    result = {
        "length": n,
        "mean_label": float(arr.mean()) if n else 0.0,
        "durbin_watson": None,
        "ljung_box_stat": None,
        "ljung_box_pvalue": None,
        "acf_values": [],
    }
    if n < min_length or arr.std() == 0:
        return result

    # Durbin-Watson on the (mean-centred) series, standard usage for a
    # single autocorrelation diagnostic.
    residuals = arr - arr.mean()
    result["durbin_watson"] = float(durbin_watson(residuals))

    # Ljung-Box up to max_lag (or n-2, whichever is smaller, since the
    # test requires lag < n).
    lag = min(max_lag, n - 2)
    if lag >= 1:
        lb = acorr_ljungbox(arr, lags=[lag], return_df=True)
        result["ljung_box_stat"] = float(lb["lb_stat"].iloc[0])
        result["ljung_box_pvalue"] = float(lb["lb_pvalue"].iloc[0])

    # Lag-k ACF, k=1..max_lag
    nlags = min(max_lag, n - 1)
    if nlags >= 1:
        acf_vals = acf(arr, nlags=nlags, fft=False)
        result["acf_values"] = [float(v) for v in acf_vals[1:]]  # drop lag-0 (=1.0)

    return result


def analyse_corpus(annotated_records: List[dict], label_key: str = "labels",
                    max_lag: int = 3, min_length: int = 5) -> pd.DataFrame:
    """Run per-generation autocorrelation analysis across an annotated
    corpus (output of annotate.annotate_corpus / simulate.build_demo_corpus).

    `label_key` lets you switch between the annotator's predicted labels
    ("labels") and, for the demo corpus, the simulator's ground-truth
    labels ("true_labels") to validate the annotator itself.
    """
    rows = []
    for rec in annotated_records:
        labels = rec.get(label_key)
        if labels is None:
            continue
        stats = analyse_sequence(labels, max_lag=max_lag, min_length=min_length)
        burst_lengths, inter_burst = burst_statistics(labels)
        rows.append({
            "item_id": rec.get("item_id"),
            "model": rec.get("model"),
            "temperature": rec.get("temperature"),
            "task_type": rec.get("task_type"),
            **stats,
            "mean_burst_length": float(np.mean(burst_lengths)) if burst_lengths else 0.0,
            "n_bursts": len(burst_lengths),
            "mean_inter_burst_interval": float(np.mean(inter_burst)) if inter_burst else 0.0,
        })
    return pd.DataFrame(rows)


def summarise_by_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate autocorrelation statistics by model / task_type /
    temperature (Phase 3: 'does autocorrelation magnitude vary by task
    type, model scale, or prompt structure?').
    """
    agg = df.groupby(group_cols).agg(
        n_sequences=("length", "count"),
        mean_durbin_watson=("durbin_watson", "mean"),
        pct_significant_ljungbox=("ljung_box_pvalue", lambda s: float((s < 0.05).mean())),
        mean_acf_lag1=("acf_values", lambda s: float(np.mean([v[0] for v in s if len(v) > 0]))
                        if any(len(v) > 0 for v in s) else np.nan),
        mean_burst_length=("mean_burst_length", "mean"),
        mean_hallucination_rate=("mean_label", "mean"),
    ).reset_index()
    return agg


def hypothesis_verdict(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """Aggregate, corpus-level verdict on the central hypothesis: are
    hallucination events temporally autocorrelated?

    Reports the proportion of individual sequences with a statistically
    significant Ljung-Box test, and the mean Durbin-Watson statistic
    (should be meaningfully < 2 under positive autocorrelation / bursting).
    """
    valid = df.dropna(subset=["ljung_box_pvalue"])
    n_valid = len(valid)
    if n_valid == 0:
        return {"verdict": "insufficient data", "n_valid_sequences": 0}

    pct_significant = float((valid["ljung_box_pvalue"] < alpha).mean())
    mean_dw = float(valid["durbin_watson"].mean())

    if pct_significant > 0.5 and mean_dw < 1.8:
        verdict = ("REJECT independence null: hallucinations show significant "
                   "positive temporal autocorrelation (bursting) in the "
                   "majority of sequences.")
    elif pct_significant > 0.2:
        verdict = ("PARTIAL evidence of autocorrelation: a substantial minority "
                   "of sequences show significant clustering, but it is not "
                   "the dominant pattern across the corpus.")
    else:
        verdict = ("FAIL TO REJECT independence null: little evidence of "
                   "systematic temporal autocorrelation in this corpus.")

    return {
        "verdict": verdict,
        "n_valid_sequences": n_valid,
        "pct_significant_ljungbox_at_alpha": pct_significant,
        "alpha": alpha,
        "mean_durbin_watson": mean_dw,
    }

