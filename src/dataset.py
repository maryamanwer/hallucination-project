"""
Phase 2 - Dataset Construction. 20 balanced items (5 per task type).
"""

SEED_ITEMS = [
    {"id": "bio_1", "task_type": "biographical_qa",
     "prompt": "Write a short biography of the physicist Marie Curie.",
     "facts": ["Marie Curie was born in Warsaw", "Marie Curie discovered radium",
               "Marie Curie discovered polonium", "Marie Curie won the Nobel Prize in Physics",
               "Marie Curie won the Nobel Prize in Chemistry",
               "Marie Curie died from illness linked to radiation exposure"]},
    {"id": "bio_2", "task_type": "biographical_qa",
     "prompt": "Write a short biography of the computer scientist Alan Turing.",
     "facts": ["Alan Turing was born in London", "Alan Turing proposed the Turing Test",
               "Alan Turing worked at Bletchley Park", "Alan Turing helped break the Enigma code",
               "Alan Turing died in 1954"]},
    {"id": "bio_3", "task_type": "biographical_qa",
     "prompt": "Write a short biography of the naturalist Charles Darwin.",
     "facts": ["Charles Darwin was born in Shrewsbury, England", "Charles Darwin sailed on HMS Beagle",
               "Charles Darwin proposed the theory of evolution by natural selection",
               "Charles Darwin wrote On the Origin of Species",
               "Charles Darwin is buried in Westminster Abbey"]},
    {"id": "bio_4", "task_type": "biographical_qa",
     "prompt": "Write a short biography of the author Jane Austen.",
     "facts": ["Jane Austen was born in Steventon, England", "Jane Austen wrote Pride and Prejudice",
               "Jane Austen wrote Sense and Sensibility",
               "Jane Austen's novels were originally published anonymously",
               "Jane Austen died in 1817"]},
    {"id": "bio_5", "task_type": "biographical_qa",
     "prompt": "Write a short biography of the scientist Albert Einstein.",
     "facts": ["Albert Einstein was born in Ulm, Germany", "Albert Einstein developed the theory of relativity",
               "Albert Einstein won the Nobel Prize in Physics",
               "Albert Einstein worked at Princeton University", "Albert Einstein died in 1955"]},

    {"id": "sci_1", "task_type": "scientific_summarisation",
     "prompt": ("Summarise the following abstract in three sentences: "
                "'Transformers rely on self-attention mechanisms to model dependencies between tokens "
                "regardless of their distance in a sequence. Unlike recurrent networks, transformers "
                "process all tokens in parallel, which improves training efficiency on modern hardware. "
                "The original transformer architecture was introduced for machine translation tasks.'"),
     "facts": ["Transformers use self-attention", "Transformers process tokens in parallel",
               "Transformers were originally introduced for machine translation"]},
    {"id": "sci_2", "task_type": "scientific_summarisation",
     "prompt": ("Summarise the following abstract in three sentences: "
                "'CRISPR-Cas9 is a gene-editing technology that uses a guide RNA to direct the Cas9 enzyme "
                "to a specific DNA sequence, where it creates a double-strand break. Cellular repair "
                "mechanisms then fix the break, allowing researchers to disable, correct, or insert genes. "
                "The technology was adapted from a bacterial immune defence system.'"),
     "facts": ["CRISPR-Cas9 uses a guide RNA to target DNA",
               "The Cas9 enzyme creates a double-strand break in DNA",
               "CRISPR-Cas9 was adapted from a bacterial immune defence system"]},
    {"id": "sci_3", "task_type": "scientific_summarisation",
     "prompt": ("Summarise the following abstract in three sentences: "
                "'Photovoltaic solar cells convert sunlight directly into electricity via the photovoltaic "
                "effect in semiconductor materials. Silicon is the most widely used material due to its "
                "favourable electronic properties and abundance. Efficiency losses occur mainly through "
                "reflection, heat, and incomplete light absorption.'"),
     "facts": ["Photovoltaic cells convert sunlight into electricity",
               "Silicon is the most widely used photovoltaic material",
               "Efficiency losses in solar cells occur through reflection and heat"]},
    {"id": "sci_4", "task_type": "scientific_summarisation",
     "prompt": ("Summarise the following abstract in three sentences: "
                "'Herd immunity occurs when a sufficient proportion of a population becomes immune to an "
                "infectious disease, making its spread from person to person unlikely. This can be achieved "
                "through vaccination or prior infection. The threshold proportion required varies depending "
                "on how contagious the disease is.'"),
     "facts": ["Herd immunity occurs when enough of a population is immune to a disease",
               "Herd immunity can be achieved through vaccination",
               "The herd immunity threshold depends on how contagious the disease is"]},
    {"id": "sci_5", "task_type": "scientific_summarisation",
     "prompt": ("Summarise the following abstract in three sentences: "
                "'Plate tectonic theory explains that Earth's lithosphere is divided into rigid plates that "
                "move relative to one another over the underlying mantle. Interactions at plate boundaries "
                "cause earthquakes, volcanic activity, and mountain building. The theory unified earlier "
                "ideas about continental drift and seafloor spreading.'"),
     "facts": ["Earth's lithosphere is divided into moving tectonic plates",
               "Plate boundary interactions cause earthquakes and volcanic activity",
               "Plate tectonic theory unified continental drift and seafloor spreading"]},

    {"id": "mh_1", "task_type": "multi_hop_reasoning",
     "prompt": ("The director of the 2010 film Inception also directed a 2008 Batman film. "
                "Who is this director, and what is the name of that 2008 film?"),
     "facts": ["The director is Christopher Nolan", "The 2008 Batman film is The Dark Knight"]},
    {"id": "mh_2", "task_type": "multi_hop_reasoning",
     "prompt": "The author of Harry Potter was born in England. What city was she born in, and what is her full name?",
     "facts": ["The author's name is J.K. Rowling", "J.K. Rowling was born in Yate, England"]},
    {"id": "mh_3", "task_type": "multi_hop_reasoning",
     "prompt": "The company that makes the iPhone also makes a laptop line. What is the name of the company, and what is that laptop line called?",
     "facts": ["The company is Apple", "The laptop line is called the MacBook"]},
    {"id": "mh_4", "task_type": "multi_hop_reasoning",
     "prompt": "The scientist who proposed general relativity also won a Nobel Prize, but not for relativity. What did he win the Nobel Prize for?",
     "facts": ["The scientist is Albert Einstein", "Einstein won the Nobel Prize for the photoelectric effect"]},
    {"id": "mh_5", "task_type": "multi_hop_reasoning",
     "prompt": "The river that flows through Paris also flows through another major French city downstream before reaching the sea. Name the river and the city.",
     "facts": ["The river is the Seine", "The Seine flows through Rouen before reaching the sea"]},

    {"id": "lf_1", "task_type": "long_form_generation",
     "prompt": "Write an essay about the causes of the fall of the Roman Empire.",
     "facts": ["Economic instability contributed to Rome's decline",
               "Military overextension contributed to Rome's decline",
               "Invasions by Germanic tribes contributed to Rome's decline",
               "Political corruption weakened imperial administration"]},
    {"id": "lf_2", "task_type": "long_form_generation",
     "prompt": "Write an essay about the main causes of climate change.",
     "facts": ["Burning fossil fuels increases atmospheric carbon dioxide",
               "Deforestation reduces the planet's capacity to absorb carbon dioxide",
               "Industrial agriculture contributes methane emissions",
               "Increased greenhouse gases trap more heat in the atmosphere"]},
    {"id": "lf_3", "task_type": "long_form_generation",
     "prompt": "Write an essay about the causes of World War One.",
     "facts": ["The assassination of Archduke Franz Ferdinand triggered the war",
               "A system of entangling alliances drew multiple countries into conflict",
               "Militarism and arms races increased tensions before the war",
               "Nationalism contributed to instability in the Balkans"]},
    {"id": "lf_4", "task_type": "long_form_generation",
     "prompt": "Write an essay about the benefits and risks of artificial intelligence.",
     "facts": ["AI can automate repetitive tasks, increasing productivity",
               "AI systems can perpetuate biases present in training data",
               "AI raises concerns about job displacement",
               "AI has applications in medical diagnosis and drug discovery"]},
    {"id": "lf_5", "task_type": "long_form_generation",
     "prompt": "Write an essay about the causes of the French Revolution.",
     "facts": ["Financial crisis due to war debt weakened the French monarchy",
               "Social inequality between the estates fuelled resentment",
               "Enlightenment ideas influenced revolutionary thinking",
               "Poor harvests led to food shortages and public unrest"]},
]


def load_dataset():
    return SEED_ITEMS
