"""
Phase 4 - Intervention Experiment.

Once the early-detection classifier fires (predicted risk above
`risk_threshold`), the proposed intervention is: truncate the generation
at the last faithful sentence and regenerate from there, rather than
letting the model continue autoregressively into a hallucination burst.

Evaluating this properly requires actually re-querying the LLM
(generate_hf.py) at the truncation point - that must happen on your
GPU/HPC session, since it needs a live model. What this module gives you
is the FULL EVALUATION HARNESS: given (a) a classifier, (b) a corpus of
annotated sequences, and (c) a "regeneration oracle" (a function that,
given a truncated prompt, returns a fresh continuation + its labels),
it runs the intervention-vs-no-intervention comparison and reports the
standard factuality/task-completion metrics.

For the demo (no live model available here), `simulated_regeneration_oracle`
draws a fresh continuation from the SAME Markov chain used to build the
demo corpus - i.e. it assumes regenerating "resets" the hallucination
streak, which is the intervention's core hypothesis. Replace this oracle
with a real call to generate_hf.py's `generate_for_item` for the
dissertation experiment.
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.simulate import markov_label_sequence, SYNTHETIC_MODEL_CONFIGS


RegenOracle = Callable[[dict, int], List[int]]  # (record, truncate_at) -> new_labels_after_truncation


def simulated_regeneration_oracle(record: dict, truncate_at: int) -> List[int]:
    """Demo oracle: regenerate the remainder of the sequence assuming the
    restart resets the Markov chain to the faithful state (state 0),
    using the same (p01, p11) as the original synthetic model/temperature.
    """
    model = record["model"]
    cfg = SYNTHETIC_MODEL_CONFIGS[model]
    remaining_len = len(record["true_labels"]) - truncate_at
    if remaining_len <= 0:
        return []
    return markov_label_sequence(remaining_len, p01=cfg["p01"], p11=cfg["p11"], start_state=0)


def run_intervention_experiment(records: List[dict], clf, scaler,
                                 k_early: int = 3, risk_threshold: float = 0.5,
                                 regen_oracle: RegenOracle = simulated_regeneration_oracle,
                                 label_key: str = "true_labels") -> pd.DataFrame:
    """For each record: compute early-window risk; if above threshold,
    simulate a restart from the last faithful sentence at/after k_early
    and splice in the oracle's fresh continuation; otherwise keep the
    original (no-intervention) continuation. Compare final hallucination
    rate of intervened vs non-intervened outputs.
    """
    from src.classifier import extract_features, FEATURE_COLS

    rows = []
    for rec in records:
        labels = rec[label_key]
        if len(labels) <= k_early:
            continue

        feats = extract_features(rec, k_early, label_key=label_key)
        x = np.array([[feats[c] for c in FEATURE_COLS]])
        risk = float(clf.predict_proba(scaler.transform(x))[0, 1])
        intervene = risk >= risk_threshold

        baseline_labels = labels  # no-intervention: original continuation stands
        if intervene:
            # find last faithful sentence at/after k_early to restart from
            restart_at = k_early
            for i in range(k_early, 0, -1):
                if labels[i - 1] == 0:
                    restart_at = i
                    break
            new_tail = regen_oracle(rec, restart_at)
            intervened_labels = labels[:restart_at] + new_tail
        else:
            intervened_labels = labels

        rows.append({
            "item_id": rec.get("item_id"),
            "model": rec.get("model"),
            "temperature": rec.get("temperature"),
            "predicted_risk": risk,
            "intervened": intervene,
            "baseline_halluc_rate": float(np.mean(baseline_labels)),
            "intervened_halluc_rate": float(np.mean(intervened_labels)),
            "baseline_any_halluc": int(any(baseline_labels)),
            "intervened_any_halluc": int(any(intervened_labels)),
        })

    return pd.DataFrame(rows)


def summarise_intervention(df: pd.DataFrame) -> Dict:
    intervened = df[df["intervened"]]
    not_intervened = df[~df["intervened"]]

    def _rate(sub, col):
        return float(sub[col].mean()) if len(sub) else float("nan")

    return {
        "n_total": len(df),
        "n_intervened": len(intervened),
        "n_not_intervened": len(not_intervened),
        "mean_halluc_rate_baseline_all": _rate(df, "baseline_halluc_rate"),
        "mean_halluc_rate_after_intervention_all": _rate(df, "intervened_halluc_rate"),
        "mean_halluc_rate_baseline_among_intervened": _rate(intervened, "baseline_halluc_rate"),
        "mean_halluc_rate_after_among_intervened": _rate(intervened, "intervened_halluc_rate"),
        "pct_any_halluc_baseline_among_intervened": _rate(intervened, "baseline_any_halluc"),
        "pct_any_halluc_after_among_intervened": _rate(intervened, "intervened_any_halluc"),
    }
