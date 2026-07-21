%%writefile /content/hallucination_project/src/generate_api.py
import argparse
import json
import os
import sys
import time

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import API_PROVIDERS
from src.dataset import load_dataset


def call_api(provider, model_key, prompt, temperature, max_tokens=400, max_retries=3):
    cfg = API_PROVIDERS[provider]
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"Set the {cfg['env_key']} environment variable with your "
            f"{provider} API key before running this script."
        )
    if model_key not in cfg["models"]:
        raise KeyError(
            f"'{model_key}' has no mapping for provider '{provider}' in "
            f"src/config.py API_PROVIDERS - add one."
        )
    model_name = cfg["models"][model_key]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://example.com"
        headers["X-Title"] = "hallucination-autocorrelation-project"

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(cfg["base_url"], headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=list(API_PROVIDERS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--out", default="data/raw_generations.jsonl")
    parser.add_argument("--n_samples", type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as f:
        for item in dataset:
            for sample_idx in range(args.n_samples):
                text = call_api(args.provider, args.model, item["prompt"], args.temperature)
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
                print(f"[{args.provider}/{args.model} T={args.temperature}] "
                      f"{item['id']} sample {sample_idx} done ({len(text)} chars)")


if __name__ == "__main__":
    main()