import argparse
import json
import os
import sys
import time
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import API_PROVIDERS
from src.dataset import load_dataset


def call_api(provider, model_key, prompt, temperature, max_tokens=400, max_retries=6):
    cfg = API_PROVIDERS[provider]
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise RuntimeError(f"Set the {cfg['env_key']} environment variable with your {provider} API key.")
    if model_key not in cfg["models"]:
        raise KeyError(f"'{model_key}' has no mapping for provider '{provider}' in src/config.py API_PROVIDERS.")
    model_name = cfg["models"][model_key]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": max_tokens}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://example.com"
        headers["X-Title"] = "hallucination-autocorrelation-project"

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(cfg["base_url"], headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = min(60, 5 * (attempt + 1))
                print(f"  rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                last_err = "429 rate limited"
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_err}")


def load_completed_keys(out_path):
    completed = set()
    if not os.path.exists(out_path):
        return completed
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            completed.add((r["item_id"], r["model"], r["temperature"], r["sample_idx"]))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=list(API_PROVIDERS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--out", default="data/raw_generations.jsonl")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--sleep_between", type=float, default=1.2,
                         help="Seconds to wait between successful calls (raise this for models "
                              "with stricter per-minute limits, e.g. llama3-70b).")
    args = parser.parse_args()

    dataset = load_dataset()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    completed = load_completed_keys(args.out)
    if completed:
        print(f"  Resume mode: {len(completed)} combos already done, skipping those.")

    n_ok, n_failed = 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        for item in dataset:
            for sample_idx in range(args.n_samples):
                key = (item["id"], args.model, args.temperature, sample_idx)
                if key in completed:
                    continue
                try:
                    text = call_api(args.provider, args.model, item["prompt"], args.temperature)
                except RuntimeError as e:
                    print(f"  SKIPPING {item['id']} sample {sample_idx}: {e}")
                    n_failed += 1
                    time.sleep(3)
                    continue
                record = {
                    "item_id": item["id"], "task_type": item["task_type"],
                    "model": args.model, "temperature": args.temperature,
                    "sample_idx": sample_idx, "prompt": item["prompt"],
                    "generated_text": text,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                n_ok += 1
                print(f"[{args.provider}/{args.model} T={args.temperature}] "
                      f"{item['id']} sample {sample_idx} done ({len(text)} chars)")
                time.sleep(args.sleep_between)

    print(f"\nDone: {n_ok} succeeded, {n_failed} skipped after retries.")
    if n_failed > 0:
        print("Re-run the same command to top up skipped items (resume mode will skip completed ones).")


if __name__ == "__main__":
    main()
