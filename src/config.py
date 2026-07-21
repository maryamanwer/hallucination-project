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
API_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "env_key": "GROQ_API_KEY",
        "models": {
            "llama3-8b": "llama3-8b-8192",
            "llama3-70b": "llama3-70b-8192",
            "mistral-7b": "mixtral-8x7b-32768",
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
