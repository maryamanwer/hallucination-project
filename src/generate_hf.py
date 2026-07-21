"""
Phase 2 - Real LLM inference (run this on Apocrita / Colab Pro / Kaggle,
NOT in a CPU-only sandbox — it needs a GPU and internet access to the
HuggingFace Hub to download model weights).

Usage:
    python -m src.generate_hf --model llama3-8b --temperature 0.7 \
        --out data/raw_generations.jsonl

This produces one JSON line per (item, model, temperature) with the raw
generated text, which annotate.py then splits into sentences and labels.

Notes for the HPC run:
    - For 70B-class models, either load in 4-bit with bitsandbytes
      (load_in_4bit=True) or shard across multiple GPUs with
      device_map="auto".
    - Set do_sample=True whenever temperature > 0; temperature==0 should
      use greedy decoding (do_sample=False) since HF ignores temperature
      with sampling off, and passing temperature=0 to a sampling call
      raises an error in recent `transformers` versions.
    - Cache models under $HOME/.cache/huggingface on Apocrita's scratch
      space, not your home quota, since weights for 70B models are >100GB.
"""

import argparse
import json
import os
import sys

# Local import; keeps this file runnable as `python -m src.generate_hf`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_REGISTRY, TEMPERATURES  # noqa: E402
from src.dataset import load_dataset  # noqa: E402


def build_pipeline(model_key: str):
    """Load a causal LM + tokenizer as a text-generation pipeline.

    Deferred import of torch/transformers so this module can be imported
    elsewhere (e.g. by tests) without requiring a GPU environment.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    repo_id = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )


def generate_for_item(gen_pipeline, prompt: str, temperature: float,
                       max_new_tokens: int = 300) -> str:
    do_sample = temperature > 0
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=gen_pipeline.tokenizer.eos_token_id,
    )
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.95
    output = gen_pipeline(prompt, **kwargs)
    return output[0]["generated_text"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--temperature", type=float, required=True,
                         choices=TEMPERATURES + [0.0])
    parser.add_argument("--out", default="data/raw_generations.jsonl")
    parser.add_argument("--n_samples", type=int, default=1,
                         help="Repeat generations per item (for burst-"
                              "variance analysis across seeds).")
    args = parser.parse_args()

    dataset = load_dataset()
    gen_pipeline = build_pipeline(args.model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for item in dataset:
            for sample_idx in range(args.n_samples):
                text = generate_for_item(gen_pipeline, item["prompt"],
                                          args.temperature)
                record = {
                    "item_id": item["id"],
                    "task_type": item["task_type"],
                    "model": args.model,
                    "temperature": args.temperature,
                    "sample_idx": sample_idx,
                    "prompt": item["prompt"],
                    "generated_text": text,
                }
                f.write(json.dumps(record) + "\n")
                print(f"[{args.model} T={args.temperature}] {item['id']} "
                      f"sample {sample_idx} done ({len(text)} chars)")


if __name__ == "__main__":
    main()
