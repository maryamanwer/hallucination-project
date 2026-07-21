# Hallucination Autocorrelation — Codebase

Implements the methodology in your project definition: testing whether
hallucination events within a single LLM generation are temporally
autocorrelated (Phases 1–4), with a working demo you can run right now
and clear swap-in points for the real experiment on your HPC/Colab GPU.

## Why a "demo mode" exists

This was built in a sandbox with **no GPU and no access to the
HuggingFace Hub**, so it can't download LLaMA/Mistral weights or run
real inference. Rather than hand you untested code, everything except
model download + inference is fully implemented and verified to run
end-to-end, using a synthetic corpus generator (`src/simulate.py`) that
produces realistic text with a **known, controllable** ground-truth
autocorrelation structure (a 2-state Markov chain). This lets you:

1. Confirm the statistics pipeline actually detects autocorrelation
   when it's really there, before spending GPU hours on the real corpus.
2. See every output file (CSVs, JSON verdicts, plots) the real pipeline
   will produce, so you know what to expect.

**The demo numbers are not your dissertation's results.** They're a
scaffold. Section "Running on real models" below is the part that
produces your actual empirical findings.

## Quickstart (runs anywhere, ~10 seconds)

```bash
pip install -r requirements.txt
python main.py
```

Output goes to `results/`:
| File | What it is |
|---|---|
| `demo_corpus.jsonl` | raw synthetic generations |
| `annotated_corpus.jsonl` | + sentence-level hallucination labels |
| `sequence_stats.csv` | per-generation Durbin-Watson / Ljung-Box / ACF |
| `group_summary.csv` | aggregated by model × temperature |
| `hypothesis_verdict.json` | headline Phase-3 result |
| `classifier_metrics.json` | Phase-4 held-out classifier evaluation |
| `intervention_results.csv`, `intervention_summary.json` | Phase-4 restart-intervention experiment |
| `acf_by_model.png` | mean ACF decay per model, lags 1–3 |

## Module map

```
src/config.py          — model registry, temperatures, task types (EDIT THIS)
src/dataset.py          — Phase 2: seed items with atomic ground-truth facts (EXTEND THIS)
src/generate_hf.py       — Phase 2: REAL LLM inference via HF Transformers (RUN ON HPC/GPU)
src/simulate.py          — demo-only: synthetic Markov-chain corpus generator
src/annotate.py          — Phase 2: sentence splitting + factuality scoring
                              - LexicalOverlapScorer: offline, no GPU, works now
                              - AlignScoreScorer: real scorer, needs GPU + download
src/autocorrelation.py  — Phase 3: Durbin-Watson, Ljung-Box, lag-k ACF, burst stats
src/classifier.py       — Phase 4: early-detection logistic-regression classifier
src/intervention.py     — Phase 4: restart-intervention experiment harness
main.py                 — orchestrates the whole pipeline (demo mode)
```

## Running on real models (what you'll do on Apocrita / Colab Pro)

1. **Extend the dataset.** Add real items to `src/dataset.py` (or write
   a loader that pulls from Wikipedia / HotpotQA / MuSiQue / arXiv
   abstracts) until you have enough per task type. Keep `facts` as
   short, atomic, independently-checkable claims — that's what makes
   annotation tractable.

2. **Generate real outputs**, on a GPU node:
   ```bash
   pip install torch transformers accelerate bitsandbytes
   python -m src.generate_hf --model llama3-8b --temperature 0.7 \
       --out data/raw_generations.jsonl
   # repeat per model in MODEL_REGISTRY x per temperature in TEMPERATURES
   ```
   For 70B models, `generate_hf.py` already uses `device_map="auto"`;
   add `load_in_4bit=True` to `AutoModelForCausalLM.from_pretrained(...)`
   if you don't have multi-GPU.

3. **Annotate with the real scorer**:
   ```bash
   pip install alignscore && python -m spacy download en_core_web_sm
   # download an AlignScore checkpoint per https://github.com/yuh-zha/AlignScore
   ```
   Then in `annotate.py`'s `__main__` block (or a new script), swap:
   ```python
   scorer = AlignScoreScorer(ckpt_path="/path/to/checkpoint.ckpt")
   annotated = annotate_corpus(raw, facts_by_id, scorer=scorer)
   ```
   Manually verify a sample (per Phase 2 of the methodology) before
   trusting the automated labels at scale.

4. **Run the same Phase 3/4 code** (`autocorrelation.py`,
   `classifier.py`, `intervention.py`) directly on the real annotated
   corpus — nothing in those modules needs to change, since they only
   depend on the `sentences` / `labels` / `model` / `temperature` /
   `task_type` schema, which `annotate_corpus` produces identically
   whether the labels came from the lexical scorer or AlignScore.
   Just point `main.py` (or a copy of it) at your real
   `annotated_corpus.jsonl` instead of calling `build_demo_corpus()`.

5. **For the real intervention experiment**, replace
   `simulated_regeneration_oracle` in `intervention.py` with a function
   that calls `generate_hf.generate_for_item` on the truncated prompt
   (prompt + faithful sentences so far) to get a genuine fresh
   continuation, then re-annotates it with the same scorer.

## Interpreting `hypothesis_verdict.json`

- `mean_durbin_watson` well below 2 → positive autocorrelation
  (hallucinations cluster rather than occurring independently).
- `pct_significant_ljungbox_at_alpha` → proportion of individual
  generations where the null of "no autocorrelation up to lag 3" is
  rejected at α=0.05. Report this alongside the DW mean; a few highly
  autocorrelated long sequences can pull the mean without most
  sequences being individually significant, so both numbers matter for
  your write-up.
- `acf_by_model.png` shows how far the "memory" of a hallucination
  extends (lag 1 vs lag 2 vs lag 3) — this is your direct answer to
  "how far does autocorrelation decay?" from the project aims.

## Known simplifications to flag in your dissertation

- `LexicalOverlapScorer` is a placeholder, not a substitute for
  FActScore/AlignScore — TF-IDF overlap will miss paraphrase-level
  hallucinations. Only use its numbers for pipeline validation.
- The simple sentence splitter (`split_sentences`) doesn't handle
  abbreviations (e.g. "Dr.", "U.S.") — swap for `nltk.sent_tokenize` if
  your corpus has many.
- The classifier's feature set is text-only. Once you have real
  logits from `generate_hf.py` (`output_scores=True` in the `pipeline`
  call), add token-entropy and top-1-probability features — these are
  the strongest hallucination-risk predictors in the literature and are
  currently missing from `classifier.py`'s `FEATURE_COLS`.
