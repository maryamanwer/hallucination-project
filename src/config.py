"""
Global configuration for the hallucination-autocorrelation project.

Edit MODEL_REGISTRY and TASK_TYPES to match what you actually run on the
HPC cluster (Apocrita) or Colab. Nothing else in the codebase needs to
change if you keep the same field names.
"""

# ---------------------------------------------------------------------------
# Model families / scales you plan to compare (Phase 2 of the methodology).
# key -> HuggingFace repo id. Swap these for whatever you actually have
# quota/GPU for. 8B-class models run on a single A100; 70B needs multi-GPU
# or 4-bit quantisation (bitsandbytes / GGUF).
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "llama3-8b":  "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "falcon-7b":  "tiiuae/falcon-7b-instruct",
}

# Decoding temperatures to sweep, per the methodology (Phase 3: does
# autocorrelation magnitude vary with temperature?)
TEMPERATURES = [0.2, 0.7, 1.0]

# Task categories from the project definition.
TASK_TYPES = [
    "biographical_qa",
    "scientific_summarisation",
    "multi_hop_reasoning",
    "long_form_generation",
]

# Hallucination label convention used everywhere downstream:
# 1 = hallucinated sentence, 0 = faithful sentence.
HALLUC_LABEL = 1
FAITHFUL_LABEL = 0

RESULTS_DIR = "results"
DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Hosted OpenAI-compatible API providers (src/generate_api.py) - use these
# instead of MODEL_REGISTRY when you don't have/want local GPU access.
#
# NOTE: hosted providers retire/rename model IDs often. If a --model
# argument 404s or 400s, check the provider's current model list and
# update the mapping below - everything in generate_api.py is
# provider-agnostic and needs no other changes.
#   Groq's live list:       https://console.groq.com/docs/models
#   Together's live list:   https://docs.together.ai/docs/serverless-models
#   OpenRouter's live list:  https://openrouter.ai/models
# ---------------------------------------------------------------------------
API_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "env_key": "GROQ_API_KEY",
        "models": {
            "llama3-8b": "llama-3.1-8b-instant",
            "llama3-70b": "llama-3.3-70b-versatile",
            "mistral-7b": "openai/gpt-oss-20b",
        },
    },
    "together": {
        "base_url": "https://api.together.xyz/v1/chat/completions",
        "env_key": "TOGETHER_API_KEY",
        "models": {
            "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
            "falcon-7b": "tiiuae/falcon-7b-instruct",
        },
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "env_key": "OPENROUTER_API_KEY",
        "models": {
            "llama3-8b": "meta-llama/llama-3-8b-instruct",
            "llama3-70b": "meta-llama/llama-3-70b-instruct",
            "mistral-7b": "mistralai/mistral-7b-instruct",
            "falcon-7b": "tiiuae/falcon-7b-instruct",
        },
    },
}
