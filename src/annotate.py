"""
Phase 2 - Sentence-level hallucination annotation.

Turns raw generated text into a binary label sequence:
    1 = hallucinated sentence (unsupported by / contradicts ground truth)
    0 = faithful sentence

Two scorers are provided:

1. `LexicalOverlapScorer` (fallback, fully offline, no model download) -
   a TF-IDF cosine-similarity proxy against the item's atomic ground-truth
   facts. Useful for a quick sanity-check of the statistical pipeline
   without any GPU or internet access, but NOT the scorer used for
   reported dissertation results.

2. `AlignScoreScorer` - the real scorer used for the dissertation.
   Requires a downloaded checkpoint from the HuggingFace Hub:
       pip install alignscore
       python -m spacy download en_core_web_sm
       curl -L -o ckpts/AlignScore-large.ckpt https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt
   This is now the default when the script is run standalone (see
   __main__ below) - it silently falls back to LexicalOverlapScorer only
   if AlignScore/its checkpoint isn't available, and prints which one
   ran so it's always clear which scorer produced a given results file.

Swap the scorer used in `annotate_generation` / `annotate_corpus` by
changing the `scorer` argument - nothing else in the pipeline needs to
change, since both scorers implement `.score(sentence, facts) -> float`.
"""

import json
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text: str) -> List[str]:
    """Simple, dependency-free sentence splitter.

    Good enough for model-generated prose. Swap for `nltk.sent_tokenize`
    or `spacy` if you need to handle abbreviations / edge cases more
    carefully for the dissertation corpus.
    """
    text = text.strip().replace("\n", " ")
    # split on '.', '!', '?' followed by a space and a capital letter,
    # while keeping the punctuation attached to the sentence.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


class LexicalOverlapScorer:
    """TF-IDF cosine similarity between a sentence and the best-matching
    ground-truth fact. Below `threshold` similarity to *every* fact ->
    treated as unsupported (hallucinated).
    """

    def __init__(self, threshold: float = 0.18):
        self.threshold = threshold

    def score(self, sentence: str, facts: List[str]) -> float:
        """Return max cosine similarity of `sentence` to any fact in `facts`."""
        if not facts:
            return 0.0
        corpus = facts + [sentence]
        vectorizer = TfidfVectorizer(stop_words="english").fit(corpus)
        vecs = vectorizer.transform(corpus)
        fact_vecs, sent_vec = vecs[:-1], vecs[-1]
        sims = cosine_similarity(sent_vec, fact_vecs)[0]
        return float(sims.max())

    def label(self, sentence: str, facts: List[str]) -> int:
        return int(self.score(sentence, facts) < self.threshold)


class AlignScoreScorer:
    """Real factual-consistency scorer (NLI-style) for production use.

    Requires: pip install alignscore; and a downloaded checkpoint.
    Not runnable in a no-GPU / no-internet sandbox - included so the
    swap is a one-line change once you're on the HPC cluster or a Colab
    GPU runtime.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda:0",
                 threshold: float = 0.5):
        from alignscore import AlignScore  # noqa: F401  (heavy import)
        self.model = AlignScore(
            model="roberta-large",
            batch_size=16,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )
        self.threshold = threshold

    def score(self, sentence: str, facts: List[str]) -> float:
        context = " ".join(facts)
        return float(self.model.score(contexts=[context], claims=[sentence])[0])

    def label(self, sentence: str, facts: List[str]) -> int:
        return int(self.score(sentence, facts) < self.threshold)


def annotate_generation(record: dict, scorer=None) -> dict:
    """Annotate a single generation record (see generate_hf.py schema).

    Returns the record augmented with:
        "sentences": list[str]
        "labels": list[int]   (1 = hallucinated, 0 = faithful)
        "scores": list[float] (raw scorer output, for threshold analysis)
    """
    scorer = scorer or LexicalOverlapScorer()
    sentences = split_sentences(record["generated_text"])
    facts = record.get("facts", [])
    scores = [scorer.score(s, facts) for s in sentences]
    labels = [int(sc < scorer.threshold) for sc in scores]
    out = dict(record)
    out["sentences"] = sentences
    out["scores"] = scores
    out["labels"] = labels
    return out


def annotate_corpus(records: List[dict], facts_by_id: dict, scorer=None) -> List[dict]:
    """Annotate a list of raw generation records, attaching each item's
    ground-truth facts by item_id before scoring."""
    annotated = []
    for r in records:
        r = dict(r)
        r["facts"] = facts_by_id.get(r["item_id"], [])
        annotated.append(annotate_generation(r, scorer=scorer))
    return annotated


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    import argparse
    from src.dataset import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/raw_generations.jsonl")
    parser.add_argument("--out_path", default="data/annotated_corpus.jsonl")
    parser.add_argument("--ckpt_path", default="ckpts/AlignScore-large.ckpt",
                         help="Path to AlignScore checkpoint. If missing or "
                              "unloadable, falls back to LexicalOverlapScorer.")
    args = parser.parse_args()

    facts_by_id = {item["id"]: item["facts"] for item in load_dataset()}
    raw = load_jsonl(args.in_path)

    try:
        scorer = AlignScoreScorer(ckpt_path=args.ckpt_path, device="cuda:0")
        print(f"Using AlignScoreScorer (ckpt: {args.ckpt_path})")
    except Exception as e:
        print(f"Falling back to LexicalOverlapScorer (AlignScore unavailable: {e})")
        scorer = LexicalOverlapScorer()

    annotated = annotate_corpus(raw, facts_by_id, scorer=scorer)
    save_jsonl(annotated, args.out_path)
    print(f"Annotated {len(annotated)} generations -> {args.out_path}")
