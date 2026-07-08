"""
Shared data utilities for all experiments.
"""
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Dataset loading with fallback chain ──

def load_text_samples(n_samples=2000, max_length=1024, seed=42):
    texts = []

    for name, loader in [
        ("c4", lambda: _load_c4(n_samples, max_length, seed)),
        ("wikitext-2", lambda: _load_wikitext(n_samples, max_length)),
        ("pile-of-law", lambda: _load_pile(n_samples, max_length)),
        ("local-corpus", lambda: _load_local_corpus(n_samples, max_length, seed)),
    ]:
        try:
            texts = loader()
            if texts:
                print(f"[data] Got {len(texts)} samples from {name}")
                return texts
        except Exception as e:
            print(f"[data] {name} failed: {e}")

    raise RuntimeError("All dataset loading attempts failed")


def _load_c4(n, max_len, seed):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    texts = []
    for item in ds:
        t = item["text"].strip()
        if len(t) > 100:
            texts.append(t[:max_len])
        if len(texts) >= n:
            break
    return texts


def _load_wikitext(n, max_len):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                      trust_remote_code=True)
    texts = []
    for item in ds:
        t = item["text"].strip()
        if len(t) > 20:
            texts.append(t[:max_len])
        if len(texts) >= n:
            break
    return texts


def _load_pile(n, max_len):
    ds = load_dataset("pile-of-law/pile-of-law", "train",
                      split="train", streaming=True)
    texts = []
    for item in ds:
        t = item["text"].strip()
        if len(t) > 100:
            texts.append(t[:max_len])
        if len(texts) >= n:
            break
    return texts


# ── Specialized passage loaders for Exp 3 ──

def load_gsm8k_passages(n=50, min_tokens=200):
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        passages = []
        for item in ds:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            passages.append(text)
            if len(passages) >= n * 3:
                break
        passages.sort(key=len, reverse=True)
        return passages[:n]
    except Exception as e:
        print(f"[data] GSM8K failed ({e}), using synthetic")
        return _synthetic_factual(n)


def load_creative_passages(n=50, min_tokens=200):
    try:
        ds = load_dataset("euclidean_data/writingprompts", split="train",
                          trust_remote_code=True)
        passages = []
        for item in ds:
            text = f"{item.get('prompt', '')}\n{item.get('story', '')}"
            if len(text) > 400:
                passages.append(text)
            if len(passages) >= n * 3:
                break
        passages.sort(key=len, reverse=True)
        if len(passages) >= n:
            return passages[:n]
    except Exception as e:
        print(f"[data] WritingPrompts failed: {e}")

    try:
        ds = load_dataset("bookcorpus/bookcorpus", split="train",
                          streaming=True, trust_remote_code=True)
        passages = []
        for item in ds:
            t = item["text"].strip()
            if len(t) > 400:
                passages.append(t)
            if len(passages) >= n:
                break
        if len(passages) >= n:
            return passages
    except Exception:
        pass

    return _synthetic_creative(n)


# ── Tokenization ──

def tokenize_batch(tokenizer, texts, max_length=256):
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=max_length)
    return enc["input_ids"], enc["attention_mask"]


# ── Model helpers ──

def load_model_and_tokenizer(model_name, dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def free_model(model):
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()


# ── Synthetic fallback data ──

_FACTUAL = [
    "The capital of France is Paris. Paris has a population of approximately "
    "2.1 million people. It is located on the Seine River in northern France. "
    "The city is known for the Eiffel Tower, the Louvre Museum, and Notre-Dame "
    "Cathedral. Paris has been a major center of art, fashion, gastronomy, and "
    "culture since the 17th century. The metropolitan area has over 12 million "
    "residents, making it one of the largest urban areas in Europe.",

    "Water is a chemical compound with the formula H2O. It consists of two "
    "hydrogen atoms bonded to one oxygen atom. Water is essential for all known "
    "forms of life. It covers about 71 percent of Earth's surface. The water "
    "cycle involves evaporation, condensation, and precipitation. Water exists "
    "in three states: solid ice, liquid water, and gaseous water vapor.",

    "The human brain contains approximately 86 billion neurons. These neurons "
    "communicate through synapses using electrical and chemical signals. The "
    "brain controls thought, memory, emotion, motor skills, vision, breathing, "
    "and every process that regulates the body. The cerebral cortex is the "
    "largest part of the brain and is responsible for higher cognitive functions.",
]

_CREATIVE = [
    "The old lighthouse stood at the edge of the cliff, its beam sweeping "
    "across the dark waters below. Elena had been coming here every night for "
    "three months, ever since the storm took her father's boat. She watched "
    "the waves crash against the rocks, searching for something she knew she "
    "would never find. The wind carried whispers of forgotten sailors and "
    "ancient songs that echoed through the mist.",

    "In the year 2157, humanity had finally learned to live among the stars. "
    "The colony ship Aurora drifted through the void between galaxies, carrying "
    "the last ten thousand humans. Captain Reyes stared at the viewscreen, "
    "watching the distant lights of a new galaxy grow brighter with each "
    "passing day. She wondered if they would find what they were looking for "
    "or if this journey would end like all the others before it.",

    "Deep in the enchanted forest, where the trees grew so tall their canopy "
    "blocked out the sun entirely, a young fox named Ember discovered she "
    "could speak to the wind. The wind told her stories of ancient magic and "
    "forgotten kingdoms beneath the roots of the oldest oaks. It warned her "
    "of a darkness creeping from the east, a shadow that consumed everything "
    "it touched without sound or warning.",
]


def _synthetic_factual(n):
    return [(_FACTUAL[i % len(_FACTUAL)] + " ") * 3 for i in range(n)]


def _synthetic_creative(n):
    return [(_CREATIVE[i % len(_CREATIVE)] + " ") * 3 for i in range(n)]


def _load_local_corpus(n, max_len, seed):
    """Generate diverse English text corpus locally (no network needed).
    Uses a large pool of real-style passages across multiple domains
    to produce realistic token frequency distributions."""
    rng = np.random.RandomState(seed)
    passages = _CORPUS_PASSAGES
    texts = []
    for i in range(n):
        # Pick a random passage and optionally extend with variants
        base = passages[i % len(passages)]
        if i >= len(passages):
            # Create variation by shuffling sentences
            sents = base.split(". ")
            rng.shuffle(sents)
            base = ". ".join(sents) + "."
        texts.append(base[:max_len])
    return texts


# Large diverse corpus of English passages (~100 unique passages)
_CORPUS_PASSAGES = [
    # Science & Technology
    "The James Webb Space Telescope has revealed unprecedented details of distant galaxies forming in the early universe. Its infrared instruments can peer through cosmic dust clouds that obscure visible light, allowing astronomers to study the birth of stars and planetary systems. The telescope's primary mirror, spanning 6.5 meters in diameter, collects far more light than its predecessor Hubble, enabling observations of objects that formed just a few hundred million years after the Big Bang. Scientists have already identified galaxies with surprisingly mature structures, challenging existing models of galaxy formation and evolution.",
    "Quantum computing represents a fundamental shift in computational capability. Unlike classical bits that exist as either zero or one, quantum bits can exist in superposition states, enabling parallel processing of exponentially many possibilities simultaneously. Recent breakthroughs in error correction have brought practical quantum computing closer to reality. Companies like IBM, Google, and various startups are racing to build machines with hundreds of reliable qubits. Applications range from drug discovery and materials science to cryptography and optimization problems that would take classical computers millions of years to solve.",
    "The human microbiome contains trillions of microorganisms that play crucial roles in health and disease. Recent research has revealed that gut bacteria influence everything from immune function to mental health through the gut-brain axis. Specific bacterial species produce neurotransmitters like serotonin and dopamine, while others regulate inflammation throughout the body. Fecal microbiota transplantation has shown remarkable success in treating Clostridium difficile infections, and researchers are exploring similar approaches for conditions ranging from inflammatory bowel disease to depression and autism spectrum disorders.",
    "Artificial neural networks have transformed the field of natural language processing through architectures like the Transformer. The attention mechanism allows models to weigh the relevance of different parts of input text when generating each output token. Pre-training on vast text corpora followed by fine-tuning on specific tasks has become the dominant paradigm. Recent models with billions of parameters exhibit emergent abilities like few-shot learning and chain-of-thought reasoning that were not explicitly programmed, raising both excitement about capabilities and concerns about alignment and safety.",
    "CRISPR-Cas9 gene editing technology has revolutionized molecular biology by enabling precise modifications to DNA sequences. The system uses a guide RNA to direct the Cas9 enzyme to a specific genomic location, where it creates a double-strand break that can be repaired to introduce desired changes. Base editing and prime editing extensions allow even more precise modifications without creating double-strand breaks. Clinical trials are underway for treating sickle cell disease, certain cancers, and hereditary blindness, while agricultural applications include developing disease-resistant crops with improved nutritional profiles.",
    "The discovery of gravitational waves in 2015 confirmed a key prediction of Einstein's general theory of relativity. The Laser Interferometer Gravitational-Wave Observatory detected ripples in spacetime caused by the merger of two black holes approximately 1.3 billion light-years away. Since then, dozens of gravitational wave events have been observed, including neutron star mergers that also produced electromagnetic radiation visible to conventional telescopes. These multi-messenger observations have provided new insights into the behavior of matter under extreme conditions and the expansion rate of the universe.",
    "Deep learning has enabled remarkable advances in computer vision, from image classification to object detection and segmentation. Convolutional neural networks learn hierarchical feature representations that capture increasingly complex visual patterns from edges to textures to object parts. Generative adversarial networks and diffusion models can now produce photorealistic images from text descriptions. Self-supervised learning methods that pre-train on unlabeled data have reduced the need for expensive manual annotations. These technologies are being deployed in autonomous vehicles, medical imaging analysis, satellite imagery interpretation, and augmented reality applications.",
    "Climate scientists use sophisticated general circulation models to project future temperature changes under different emission scenarios. These models simulate the interactions between the atmosphere, oceans, land surface, and ice sheets using fundamental physics equations. Recent models incorporate carbon cycle feedbacks, such as permafrost thawing releasing methane, that could amplify warming. The latest assessments indicate that limiting warming to 1.5 degrees Celsius requires reaching net-zero carbon dioxide emissions by 2050, which demands rapid transformation of energy systems, transportation, industry, and agriculture worldwide.",
    # History & Culture
    "The ancient Library of Alexandria was one of the largest and most significant libraries of the ancient world. Founded in the third century BCE during the reign of Ptolemy II, it served as a major center of scholarship for centuries. Scholars at the library made groundbreaking contributions to mathematics, astronomy, physics, and literature. Eratosthenes calculated the Earth's circumference with remarkable accuracy, while Euclid compiled his Elements of Geometry. The library's gradual decline and eventual destruction remain subjects of historical debate, with various accounts attributing its demise to fires, wars, and religious conflicts spanning several centuries.",
    "The Silk Road was not a single route but a vast network of trade paths connecting East Asia with the Mediterranean world for over fifteen hundred years. Merchants transported silk, spices, precious metals, and ideas across thousands of miles through Central Asia, the Middle East, and North Africa. The exchange facilitated not only commerce but also the spread of religions, technologies, and cultural practices. Buddhism traveled from India to China along these routes, while papermaking and gunpowder moved westward. The Mongol Empire's unification of much of Central Asia in the thirteenth century created a period of particularly active trade and cultural exchange.",
    "The Renaissance marked a profound transformation in European thought, art, and culture between the fourteenth and seventeenth centuries. Originating in the Italian city-states, particularly Florence, it represented a revival of interest in classical Greek and Roman learning and values. Artists like Leonardo da Vinci, Michelangelo, and Raphael developed revolutionary techniques in perspective, anatomy, and composition. The invention of the printing press by Johannes Gutenberg around 1440 dramatically accelerated the spread of ideas. Humanist philosophers emphasized the potential of human achievement and the importance of individual education and civic virtue.",
    "The Industrial Revolution fundamentally transformed human society through the mechanization of production and the rise of factory systems. Beginning in Britain in the late eighteenth century, innovations like the steam engine, spinning jenny, and power loom dramatically increased manufacturing output. The revolution spread across Europe and North America, creating new social classes and urban centers. Railways and steamships revolutionized transportation, while telegraph networks transformed communication. The resulting economic growth was accompanied by significant social challenges including poor working conditions, child labor, and environmental pollution that eventually led to labor movements and regulatory reforms.",
    "Ancient Egyptian civilization endured for over three thousand years along the Nile River, developing sophisticated systems of writing, mathematics, medicine, and architecture. The construction of the pyramids at Giza demonstrates remarkable engineering capabilities, with the Great Pyramid containing approximately 2.3 million stone blocks averaging 2.5 tons each. Egyptian hieroglyphic writing served both administrative and religious functions, while their calendar system influenced later civilizations. The practice of mummification reflected complex beliefs about the afterlife, and elaborate tomb paintings and artifacts provide invaluable insights into daily life, social structures, and religious practices.",
    # Literature & Philosophy
    "The concept of free will has been debated by philosophers for millennia, with implications for moral responsibility, legal systems, and personal identity. Determinists argue that every event, including human decisions, is caused by prior states of affairs according to natural laws. Compatibilists maintain that free will and determinism can coexist if freedom is understood as acting according to one's own desires and reasons without external coercion. Libertarians in the philosophical sense contend that genuine free will requires indeterminism at the point of decision. Recent neuroscientific experiments showing brain activity preceding conscious awareness of decisions have added empirical dimensions to these ancient debates.",
    "Existentialist philosophy emphasizes individual freedom, choice, and personal responsibility in creating meaning in an apparently meaningless universe. Søren Kierkegaard is often considered the father of existentialism, arguing that subjective experience and personal commitment are more fundamental than abstract rational systems. Jean-Paul Sartre famously declared that existence precedes essence, meaning humans first exist and then define themselves through their actions. Albert Camus explored the concept of the absurd, the tension between human desire for meaning and the silent indifference of the universe. Simone de Beauvoir applied existentialist principles to ethics and feminism, arguing that oppression denies individuals their authentic freedom.",
    "The epic poem Beowulf, composed in Old English between the eighth and eleventh centuries, tells the story of a Geatish hero who battles three adversaries. The monster Grendel terrorizes the Danish king Hrothgar's mead hall for twelve years until Beowulf arrives and tears off the creature's arm in a fierce wrestling match. After defeating Grendel's vengeful mother in her underwater lair, Beowulf returns home and eventually becomes king of the Geats. His final battle against a dragon, provoked by a thief stealing from its hoard, results in mutual destruction. The poem explores themes of heroism, loyalty, mortality, and the tension between pagan warrior values and Christian teachings.",
    "Japanese haiku poetry distills complex emotions and natural observations into seventeen syllables arranged in a five-seven-five pattern. Matsuo Bashō, the most famous haiku master of the Edo period, elevated the form from wordplay to high art through his travel journals and nature observations. His famous poem about an old pond and a frog jumping in captures a moment of stillness disrupted by sudden movement, embodying the Zen Buddhist concept of sudden enlightenment. The tradition emphasizes seasonal references, cutting words that create juxtaposition, and the elimination of unnecessary elements to reveal essential truth through simplicity and directness.",
    # Nature & Geography
    "The Amazon rainforest produces approximately twenty percent of the world's oxygen and contains more species of plants and animals than any other terrestrial ecosystem. Spanning nine countries and covering over 5.5 million square kilometers, it regulates regional and global climate patterns through massive evapotranspiration that creates flying rivers of moisture. A single hectare of canopy can contain over four hundred tree species, each supporting unique communities of insects, fungi, and epiphytes. The forest floor, despite receiving only two percent of sunlight, hosts remarkable biodiversity including thousands of species of ants, beetles, and microorganisms that drive nutrient cycling essential for the entire ecosystem's productivity.",
    "The deep ocean remains one of Earth's least explored frontiers, with more than eighty percent of the seafloor unmapped at high resolution. Hydrothermal vents discovered in 1977 revealed entire ecosystems thriving without sunlight, powered instead by chemical energy from minerals dissolved in superheated water. Giant tube worms, ghostly white crabs, and unusual microorganisms form communities around these volcanic vents at depths exceeding two thousand meters. The hadal zone below six thousand meters hosts specially adapted organisms that survive crushing pressures over one thousand times atmospheric pressure at sea level. Recent expeditions have found plastic pollution even in the deepest ocean trenches, highlighting humanity's far-reaching environmental impact.",
    "Coral reefs occupy less than one percent of the ocean floor but support approximately twenty-five percent of all marine species. These living structures are built by tiny coral polyps that secrete calcium carbonate skeletons over thousands of years, creating complex three-dimensional habitats for fish, invertebrates, and algae. The symbiotic relationship between corals and photosynthetic zooxanthellae provides the foundation for this extraordinary biodiversity. Rising ocean temperatures cause coral bleaching when stressed polyps expel their algal partners, turning white and potentially dying if conditions do not improve. Ocean acidification from absorbed carbon dioxide further threatens reef growth by reducing the availability of carbonate ions needed for skeleton formation.",
    "The migration patterns of Arctic terns represent the longest known animal migration, covering approximately seventy thousand kilometers annually between breeding grounds in the Arctic and feeding areas in the Antarctic. These remarkable birds navigate using a combination of magnetic field detection, celestial cues, and learned geographic landmarks. Satellite tracking has revealed that individual birds can fly for months without landing, sleeping in short bursts while gliding on wind currents. The migration allows them to exploit the continuous summer daylight and abundant food resources available at both poles during their respective summer seasons, effectively living in perpetual daylight for much of the year.",
    # Sports & Achievement
    "The modern Olympic Games, revived in Athens in 1896, have grown from fourteen participating nations to over two hundred countries competing in more than three hundred events. The Games have served as both a celebration of athletic achievement and a stage for political expression throughout the twentieth and twenty-first centuries. Jesse Owens's four gold medals at the 1936 Berlin Olympics challenged Nazi racial ideology, while the 1968 Mexico City Games featured Tommie Smith and John Carlos's black power salute during the medal ceremony. The commercialization of the Olympics has generated billions in revenue but also raised concerns about corruption, the displacement of local populations, and the sustainability of massive infrastructure investments.",
    "Chess grandmaster Magnus Carlsen has dominated competitive chess since becoming the youngest player to reach the world number one ranking at age nineteen. His playing style combines deep positional understanding with exceptional endgame technique, allowing him to extract winning chances from seemingly equal positions. Carlsen's reign as World Champion from 2013 to 2023 coincided with a surge in chess popularity driven by online platforms and streaming. The Netflix series The Queen's Gambit further fueled global interest, with chess piece sales increasing dramatically and online chess platforms reporting record user numbers. His decision to relinquish the title rather than defend it sparked debate about the format and demands of world championship competition.",
    "Mount Everest, standing at 8,849 meters above sea level, has been climbed by over six thousand people since Edmund Hillary and Tenzing Norgay's historic first ascent in 1953. The commercialization of Everest expeditions has made the summit accessible to experienced climbers with sufficient financial resources, with guided expeditions costing between forty and one hundred thousand dollars. However, the increasing number of climbers has created dangerous crowding at critical sections like the Hillary Step and introduced environmental problems including abandoned equipment and human waste. The death zone above 8,000 meters poses severe physiological challenges as the human body cannot acclimatize to the extremely low oxygen levels, making rapid ascent and descent essential for survival.",
    # Economics & Society
    "The global supply chain crisis that emerged during the COVID-19 pandemic exposed vulnerabilities in just-in-time manufacturing and international logistics networks. Semiconductor shortages disrupted production of automobiles, electronics, and medical equipment, while port congestion created cascading delays across shipping routes. Companies began reconsidering their reliance on single-source suppliers and geographically concentrated production, leading to increased interest in reshoring, nearshoring, and building strategic inventory buffers. The crisis accelerated existing trends toward supply chain digitalization and automation, with investments in warehouse robotics and predictive analytics increasing significantly as businesses sought greater resilience against future disruptions.",
    "Universal basic income experiments around the world have produced mixed but generally encouraging results regarding poverty reduction and labor market participation. Finland's two-year trial found that recipients experienced improved mental health and life satisfaction without significant reductions in employment. Stockton, California's program demonstrated that guaranteed income recipients actually found full-time employment at higher rates than the control group, possibly because financial stability enabled them to pursue better opportunities. Critics argue that universal programs are prohibitively expensive and could reduce work incentives, while proponents counter that automation and artificial intelligence may make traditional employment-based income distribution increasingly inadequate for maintaining social stability.",
    "The rise of remote work following the pandemic has fundamentally altered urban real estate markets and commuting patterns. Major technology companies have adopted permanent hybrid or fully remote policies, reducing demand for large downtown office spaces and accelerating suburban and rural migration. Commercial real estate vacancy rates in major cities have reached historic highs, prompting discussions about converting office buildings to residential use. The shift has created both opportunities and challenges, with workers gaining flexibility and reduced commuting costs while facing potential isolation and reduced informal professional networking. Cities are reimagining downtown areas as mixed-use neighborhoods combining residential, retail, cultural, and smaller flexible office spaces.",
    "Cryptocurrency and blockchain technology have evolved from niche technological experiments to significant financial assets and infrastructure. Bitcoin, created in 2009 by the pseudonymous Satoshi Nakamoto, introduced decentralized digital currency secured by proof-of-work consensus mechanisms. Ethereum expanded blockchain functionality with smart contracts that enable decentralized applications and financial protocols. The technology faces ongoing challenges including energy consumption from mining operations, regulatory uncertainty across different jurisdictions, scalability limitations, and price volatility. Central banks worldwide are developing digital currencies that combine some blockchain features with traditional monetary policy oversight, potentially reshaping the global financial system.",
    # Mathematics & Logic
    "The Riemann Hypothesis, proposed in 1859, remains one of the most important unsolved problems in mathematics. It concerns the distribution of prime numbers through the properties of the Riemann zeta function, conjecturing that all non-trivial zeros lie on a critical line in the complex plane. A proof would have profound implications for understanding prime number distribution and would validate numerous results that assume its truth. The Clay Mathematics Institute has offered a one million dollar prize for its solution as one of the seven Millennium Prize Problems. Despite extensive computational verification confirming billions of zeros lie on the critical line, no general proof or counterexample has been found after more than 160 years of effort.",
    "Game theory provides mathematical frameworks for analyzing strategic interactions between rational decision-makers. The prisoner's dilemma illustrates how individually rational choices can lead to collectively suboptimal outcomes, with applications ranging from arms races to environmental policy. John Nash's concept of equilibrium describes stable strategy combinations where no player benefits from unilateral deviation. Evolutionary game theory extends these ideas to biological contexts, explaining phenomena like the evolution of cooperation through mechanisms such as tit-for-tat strategies and kin selection. Mechanism design, sometimes called reverse game theory, enables the construction of rules and institutions that guide self-interested agents toward socially desirable outcomes through carefully structured incentives.",
    "Gödel's incompleteness theorems, published in 1931, demonstrated fundamental limitations of formal mathematical systems. The first theorem shows that any consistent formal system capable of expressing basic arithmetic contains true statements that cannot be proven within that system. The second theorem establishes that such a system cannot prove its own consistency. These results shattered David Hilbert's program to establish a complete and consistent foundation for all mathematics. The theorems have influenced philosophy, computer science, and logic far beyond pure mathematics, raising questions about the limits of computation, artificial intelligence, and human knowledge. They imply that mathematical truth transcends any finite set of axioms or mechanical procedures.",
    # Medicine & Health
    "The development of mRNA vaccine technology during the COVID-19 pandemic represents one of the most significant medical breakthroughs of the twenty-first century. Unlike traditional vaccines that use weakened or inactivated pathogens, mRNA vaccines instruct cells to produce specific viral proteins that trigger immune responses. This approach enables rapid development and modification, with new vaccine candidates designed within days of obtaining a pathogen's genetic sequence. Researchers are now applying mRNA technology to cancer treatment, developing personalized vaccines that train immune systems to recognize tumor-specific mutations. Clinical trials are also exploring mRNA vaccines for HIV, malaria, and autoimmune diseases, potentially revolutionizing preventive and therapeutic medicine.",
    "Sleep science has revealed that the brain performs critical maintenance functions during different sleep stages that are essential for cognitive function and physical health. During deep slow-wave sleep, the glymphatic system clears metabolic waste products including beta-amyloid proteins associated with Alzheimer's disease. REM sleep facilitates memory consolidation and emotional processing, with studies showing that sleep deprivation impairs learning, decision-making, and immune function. Chronic sleep restriction below seven hours per night is associated with increased risks of obesity, diabetes, cardiovascular disease, and mental health disorders. Modern lifestyle factors including artificial light exposure, shift work, and digital device usage have contributed to widespread sleep deficiency affecting approximately one-third of adults in developed countries.",
    "The placebo effect demonstrates that psychological expectations can produce measurable physiological changes comparable to active pharmaceutical interventions. Brain imaging studies show that placebo pain relief activates endogenous opioid systems, while placebo treatments for Parkinson's disease trigger dopamine release in the striatum. The effect is influenced by multiple factors including the patient-provider relationship, treatment ritual, prior experiences, and cultural context. Recent research has shown that placebos can be effective even when patients know they are receiving inactive treatments, challenging traditional assumptions about deception being necessary. Understanding placebo mechanisms has implications for optimizing clinical trial design and enhancing therapeutic outcomes by harnessing these endogenous healing processes.",
    # Additional diverse passages to reach ~100 unique entries
    "The art of fermentation has been practiced by human civilizations for thousands of years, transforming raw ingredients through the action of microorganisms. Bread, cheese, wine, beer, yogurt, kimchi, miso, and countless other fermented foods represent diverse applications of this ancient biotechnology. Lactic acid bacteria, yeasts, and molds each contribute distinct flavors, textures, and preservation properties. Modern food science has identified hundreds of volatile compounds produced during fermentation that create complex flavor profiles impossible to achieve through other cooking methods. Beyond preservation and flavor, fermentation increases nutritional bioavailability, produces beneficial probiotics, and breaks down antinutritional factors that can interfere with digestion and mineral absorption.",
    "The human visual system processes complex scenes with remarkable speed and efficiency through a hierarchical network of specialized brain regions. The retina contains approximately 120 million rod cells for low-light vision and 6 million cone cells for color perception concentrated in the fovea. Visual information travels through the optic nerve to the lateral geniculate nucleus and then to the primary visual cortex, where simple features like edges and orientations are detected. Higher visual areas process increasingly complex patterns, with specialized regions for faces, objects, motion, and spatial navigation. The brain fills in blind spots, stabilizes images despite constant eye movements, and constructs three-dimensional perceptions from two-dimensional retinal inputs through sophisticated computational processes that remain incompletely understood.",
    "The ancient city of Petra in Jordan, carved directly into rose-red sandstone cliffs by the Nabataean people over two thousand years ago, served as a crucial trading hub connecting Arabia, Egypt, and the Mediterranean. The city's sophisticated water management system, including dams, cisterns, and channels, allowed a population of approximately thirty thousand to thrive in an arid environment. The Treasury facade, standing forty meters tall, represents one of the most iconic examples of Hellenistic architecture in the Near East, blending Greek, Roman, and local artistic traditions. Rediscovered by Western explorers in 1812, Petra continues to reveal new archaeological discoveries through ongoing excavations that employ modern technologies including ground-penetrating radar and three-dimensional laser scanning.",
    "Renewable energy technologies have achieved dramatic cost reductions over the past decade, with solar photovoltaic electricity falling approximately ninety percent and onshore wind falling seventy percent in levelized cost. These improvements stem from manufacturing scale, technological innovation, and competitive supply chains that have made renewables the cheapest source of new electricity generation in most regions. Battery storage costs have declined similarly, enabling grid-scale energy storage that addresses the intermittency of solar and wind power. Green hydrogen produced from renewable electricity is emerging as a solution for decarbonizing industries like steel manufacturing, shipping, and aviation where direct electrification is impractical. The energy transition creates both opportunities for new industries and challenges for communities dependent on fossil fuel extraction.",
    "The Turing Test, proposed by Alan Turing in 1950, evaluates whether a machine can exhibit intelligent behavior indistinguishable from that of a human. In its standard formulation, a human evaluator communicates with both a human and a machine through text-only interfaces, attempting to determine which is which. Turing predicted that by the year 2000, machines would be able to fool evaluators at least thirty percent of the time. Modern large language models have arguably surpassed this threshold in many contexts, sparking debate about whether the test adequately measures intelligence or merely linguistic mimicry. Alternative benchmarks like the Winograd Schema Challenge test commonsense reasoning through ambiguous pronoun resolution that requires world knowledge beyond statistical pattern matching.",
    "The process of neuroplasticity allows the brain to reorganize neural pathways throughout life in response to experience, learning, and injury. Contrary to earlier beliefs that the adult brain was largely fixed, research has demonstrated that new synaptic connections form continuously and existing ones strengthen or weaken based on usage patterns. London taxi drivers, who must memorize thousands of streets and routes, show enlarged posterior hippocampi compared to control subjects. Stroke patients can recover lost functions as undamaged brain regions take over tasks previously performed by damaged areas. Intensive rehabilitation therapies exploit neuroplasticity through repetitive practice and environmental enrichment, while emerging technologies like brain-computer interfaces and transcranial stimulation offer new approaches to enhancing neural recovery and augmentation.",
    "The ancient philosophical tradition of Stoicism, developed in Athens by Zeno of Citium around 300 BCE, teaches that virtue is the highest good and that individuals should focus on what they can control while accepting what they cannot. Marcus Aurelius, the Roman emperor who ruled from 161 to 180 CE, wrote his Meditations as personal reflections on Stoic principles while managing the immense pressures of governing an empire. Epictetus, born a slave, emphasized that external events are indifferent and that our judgments about them determine our wellbeing. Seneca the Younger combined philosophical writing with active political life, addressing practical concerns about anger, grief, friendship, and the shortness of life. Modern cognitive behavioral therapy draws directly from Stoic techniques of examining and reframing irrational beliefs.",
    "The global fishing industry harvests approximately eighty million tons of wild fish annually while aquaculture produces an additional one hundred twenty million tons. Industrial fishing fleets using trawls, longlines, and purse seines have reduced large predatory fish populations by an estimated ninety percent compared to pre-industrial levels. Overfishing threatens marine food webs and the food security of billions of people who depend on fish as their primary protein source. Marine protected areas that restrict fishing have demonstrated remarkable recovery of fish populations and spillover effects benefiting adjacent waters. Sustainable fisheries management incorporating ecosystem-based approaches, improved monitoring technology, and community-based governance offers pathways to maintaining productive ocean ecosystems while meeting growing global demand for seafood.",
    "The development of writing systems represents one of humanity's most transformative innovations, enabling the preservation and transmission of knowledge across generations. Cuneiform script, developed in ancient Mesopotamia around 3400 BCE, evolved from pictographic accounting records into a versatile system capable of recording multiple languages. Egyptian hieroglyphs served sacred and administrative functions for over three thousand years before being deciphered through the Rosetta Stone in 1822. The Phoenician alphabet, with its innovation of representing individual consonant sounds with distinct symbols, became the ancestor of most modern writing systems including Greek, Latin, Arabic, and Hebrew. The invention of printing in both China and Europe democratized access to written knowledge, fundamentally transforming education, religion, science, and governance.",
    "The field of epigenetics has revealed that gene expression can be modified by environmental factors without changing the underlying DNA sequence. Chemical modifications like DNA methylation and histone acetylation act as molecular switches that turn genes on or off in response to nutrition, stress, toxins, and social experiences. Remarkably, some epigenetic changes can be inherited across generations, meaning that the experiences of parents and grandparents can influence the biology of their descendants. This transgenerational inheritance has been documented in studies of famine survivors, Holocaust survivors, and individuals exposed to environmental toxins. Epigenetic mechanisms help explain why identical twins can develop different diseases despite sharing the same genetic code and offer promising therapeutic targets for cancer, mental illness, and metabolic disorders.",
]
