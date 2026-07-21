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
