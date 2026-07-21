"""
Phase 2 - Dataset Construction.

This module holds a small SEED dataset (a handful of items per task type)
so the rest of the pipeline is runnable end-to-end out of the box.

For the real dissertation corpus you will replace / extend `SEED_ITEMS`
by pulling from:
    - biographical_qa            -> Wikipedia infobox facts (wikipedia-api)
    - scientific_summarisation   -> arXiv/PubMed abstracts as source, and
                                     the abstract itself as ground truth
    - multi_hop_reasoning        -> HotpotQA / MuSiQue (via `datasets`)
    - long_form_generation       -> any topic with a curated fact list
                                     (e.g. from Wikidata claims)

Each item is a dict:
{
    "id": str,
    "task_type": one of config.TASK_TYPES,
    "prompt": the prompt given to the LLM,
    "facts": list[str]   # atomic, independently checkable ground-truth
                          # claims used by the annotator (Phase 2/3)
}

Keeping `facts` as short atomic claims (not paragraphs) is what makes
sentence-level factuality scoring tractable without a full FActScore/
AlignScore model install (see annotate.py for the swap-in point).
"""

SEED_ITEMS = [
    # ---------------- biographical_qa ----------------
    {
        "id": "bio_1",
        "task_type": "biographical_qa",
        "prompt": "Write a short biography of the physicist Marie Curie.",
        "facts": [
            "Marie Curie was born in Warsaw",
            "Marie Curie discovered radium",
            "Marie Curie discovered polonium",
            "Marie Curie won the Nobel Prize in Physics",
            "Marie Curie won the Nobel Prize in Chemistry",
            "Marie Curie died from illness linked to radiation exposure",
        ],
    },
    {
        "id": "bio_2",
        "task_type": "biographical_qa",
        "prompt": "Write a short biography of the computer scientist Alan Turing.",
        "facts": [
            "Alan Turing was born in London",
            "Alan Turing proposed the Turing Test",
            "Alan Turing worked at Bletchley Park",
            "Alan Turing helped break the Enigma code",
            "Alan Turing died in 1954",
        ],
    },
    # ---------------- scientific_summarisation ----------------
    {
        "id": "sci_1",
        "task_type": "scientific_summarisation",
        "prompt": (
            "Summarise the following abstract in three sentences: "
            "'Transformers rely on self-attention mechanisms to model "
            "dependencies between tokens regardless of their distance in "
            "a sequence. Unlike recurrent networks, transformers process "
            "all tokens in parallel, which improves training efficiency "
            "on modern hardware. The original transformer architecture "
            "was introduced for machine translation tasks.'"
        ),
        "facts": [
            "Transformers use self-attention",
            "Transformers process tokens in parallel",
            "Transformers were originally introduced for machine translation",
        ],
    },
    # ---------------- multi_hop_reasoning ----------------
    {
        "id": "mh_1",
        "task_type": "multi_hop_reasoning",
        "prompt": (
            "The director of the 2010 film Inception also directed a 2008 "
            "Batman film. Who is this director, and what is the name of "
            "that 2008 film?"
        ),
        "facts": [
            "The director is Christopher Nolan",
            "The 2008 Batman film is The Dark Knight",
        ],
    },
    # ---------------- long_form_generation ----------------
    {
        "id": "lf_1",
        "task_type": "long_form_generation",
        "prompt": "Write an essay about the causes of the fall of the Roman Empire.",
        "facts": [
            "Economic instability contributed to Rome's decline",
            "Military overextension contributed to Rome's decline",
            "Invasions by Germanic tribes contributed to Rome's decline",
            "Political corruption weakened imperial administration",
        ],
    },
]


def load_dataset():
    """Return the seed dataset as a list of dicts (see module docstring)."""
    return SEED_ITEMS
