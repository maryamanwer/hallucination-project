"""
Demo generation module - lets you run Phases 3 and 4 of the methodology
end-to-end *today*, without a GPU or model downloads, by producing
synthetic-but-realistic generations whose hallucination structure follows
a controllable first-order Markov chain (i.e. a known ground-truth
autocorrelation, which you can use to sanity-check that your Durbin-Watson
/ Ljung-Box code is correctly detecting structure before you spend GPU
hours on the real LLaMA/Mistral corpus).

Design:
    - Each dataset item has faithful facts (dataset.py) and a matching
      "distractor" pool of superficially plausible but false claims about
      the same entity.
    - A sentence's label (0=faithful, 1=hallucinated) is drawn from a
      2-state Markov chain: P(hallucinate | previous was faithful) = p01,
      P(hallucinate | previous was hallucination) = p11.
      Setting p11 > p01 creates positive autocorrelation ("bursts"),
      matching what McKenna et al. (2023) and the Warwick (2025) paper
      observed qualitatively - here it's an explicit, tunable model.
    - Different "models"/"temperatures" in the demo correspond to
      different (p01, p11) pairs, i.e. different burst tendencies -
      standing in for the real experiment's model/temperature sweep.

This is a scaffold for validating the statistics pipeline, NOT a
replacement for Phase 2's real annotated corpus - the dissertation's
empirical claims must come from `generate_hf.py` + `annotate.py`
(AlignScoreScorer) on real model outputs.
"""

import json
import os
import random
from typing import List, Tuple

from src.dataset import load_dataset

# Generic false "distractor" claim templates, instantiated per item.
DISTRACTOR_TEMPLATES = [
    "{subject} was born in {wrong_place}",
    "{subject} won the {wrong_award}",
    "{subject} is best known for the invention of {wrong_thing}",
    "{subject} died in {wrong_year}",
    "{subject} studied under {wrong_person}",
    "{subject} worked primarily in {wrong_field}",
]

WRONG_PLACES = ["Vienna", "Lisbon", "Cairo", "Toronto", "Manila"]
WRONG_AWARDS = ["Fields Medal", "Turing Award", "Pulitzer Prize", "Nobel Peace Prize"]
WRONG_THINGS = ["the steam engine", "the telephone", "penicillin", "the internet"]
WRONG_YEARS = ["1932", "1978", "1901", "1965"]
WRONG_PEOPLE = ["Isaac Newton", "Niels Bohr", "Ada Lovelace", "Enrico Fermi"]
WRONG_FIELDS = ["marine biology", "civil engineering", "linguistics", "economics"]


def _subject_from_prompt(prompt: str) -> str:
    # crude heuristic: grab a capitalised multi-word span if present
    words = prompt.replace(".", "").split()
    caps = [w for w in words if w[:1].isupper()]
    return " ".join(caps[-2:]) if len(caps) >= 2 else (caps[0] if caps else "The subject")


def make_distractor_pool(subject: str, n: int = 8) -> List[str]:
    pool = []
    for _ in range(n):
        template = random.choice(DISTRACTOR_TEMPLATES)
        sentence = template.format(
            subject=subject,
            wrong_place=random.choice(WRONG_PLACES),
            wrong_award=random.choice(WRONG_AWARDS),
            wrong_thing=random.choice(WRONG_THINGS),
            wrong_year=random.choice(WRONG_YEARS),
            wrong_person=random.choice(WRONG_PEOPLE),
            wrong_field=random.choice(WRONG_FIELDS),
        )
        pool.append(sentence[0].upper() + sentence[1:] + ".")
    return pool


def markov_label_sequence(length: int, p01: float, p11: float,
                           start_state: int = 0) -> List[int]:
    """Generate a binary sequence from a 2-state Markov chain.

    p01: P(next=1 | current=0)
    p11: P(next=1 | current=1)
    """
    seq = [start_state]
    for _ in range(length - 1):
        prev = seq[-1]
        p_next_1 = p11 if prev == 1 else p01
        seq.append(1 if random.random() < p_next_1 else 0)
    return seq


def simulate_generation(item: dict, model: str, temperature: float,
                         p01: float, p11: float, length: int = 10) -> dict:
    subject = _subject_from_prompt(item["prompt"])
    facts = list(item["facts"])
    distractors = make_distractor_pool(subject, n=max(8, length))

    labels = markov_label_sequence(length, p01=p01, p11=p11)

    sentences = []
    fact_i, distractor_i = 0, 0
    for label in labels:
        if label == 1:
            sentences.append(distractors[distractor_i % len(distractors)])
            distractor_i += 1
        else:
            fact = facts[fact_i % len(facts)] if facts else "This is consistent with the record"
            sentences.append(fact[0].upper() + fact[1:] + ".")
            fact_i += 1

    text = " ".join(sentences)
    return {
        "item_id": item["id"],
        "task_type": item["task_type"],
        "model": model,
        "temperature": temperature,
        "prompt": item["prompt"],
        "facts": facts,
        "generated_text": text,
        "sentences": sentences,
        "true_labels": labels,  # ground truth from the simulator itself
    }


# (p01, p11) per synthetic "model" - p11 > p01 everywhere, i.e. every
# synthetic model exhibits positive autocorrelation (bursting), but by
# different amounts, so the analysis has something real to detect and
# differentiate.
SYNTHETIC_MODEL_CONFIGS = {
    "small-model-A": {"p01": 0.12, "p11": 0.55},  # strong bursting
    "small-model-B": {"p01": 0.10, "p11": 0.35},  # moderate bursting
    "large-model-C": {"p01": 0.06, "p11": 0.20},  # mild bursting (better model)
}
TEMPERATURES_DEMO = [0.2, 0.7, 1.0]


def build_demo_corpus(n_samples_per_config: int = 6, length: int = 10,
                       seed: int = 42) -> List[dict]:
    random.seed(seed)
    dataset = load_dataset()
    corpus = []
    for item in dataset:
        for model, cfg in SYNTHETIC_MODEL_CONFIGS.items():
            for temperature in TEMPERATURES_DEMO:
                # temperature nudges the chain slightly more bursty at
                # higher temperature, mirroring the hypothesis in Phase 3
                temp_boost = (temperature - 0.2) * 0.15
                p01 = min(0.9, cfg["p01"] + temp_boost * 0.3)
                p11 = min(0.95, cfg["p11"] + temp_boost)
                for sample_idx in range(n_samples_per_config):
                    rec = simulate_generation(
                        item, model, temperature, p01=p01, p11=p11, length=length
                    )
                    rec["sample_idx"] = sample_idx
                    corpus.append(rec)
    return corpus


def save_jsonl(records: List[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    corpus = build_demo_corpus()
    save_jsonl(corpus, "data/demo_raw_generations.jsonl")
    print(f"Built demo corpus: {len(corpus)} generations -> "
          f"data/demo_raw_generations.jsonl")
