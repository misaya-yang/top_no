#!/usr/bin/env python3
"""
Experiment 7: Domain-Aware ν-sampling for Reasoning Tasks
=====================================================
Addresses the GSM8K accuracy drop (14% vs top-p 18%).

Three fix strategies:
  Fix 1: Top-K safeguard — always keep top-K tokens regardless of frequency
  Fix 2: Entropy-gated κ — tighten the uncertainty margin when model is confident
  Fix 3: Math-boosted frequency table — treat domain-frequent math tokens as
         better-estimated tokens with a smaller uncertainty radius

Evaluates on both GSM8K (reasoning) and creative (diversity) tasks.
"""
import argparse, json, os, time, re
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_model_and_tokenizer
from samplers import batch_generate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-gsm8k", type=int, default=50)
    p.add_argument("--n-creative", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Math Frequency Table Builder
# ══════════════════════════════════════════════════════════════

def build_math_freq_table(tokenizer, vocab_size, boost_factor=50):
    """Build a frequency table enriched with mathematical tokens."""
    # Comprehensive math text corpus
    math_texts = []

    # Numbers (all common integers)
    for i in range(1001):
        math_texts.append(str(i))
    # Common larger numbers
    for i in [1000, 1500, 2000, 2500, 5000, 10000, 100000, 1000000]:
        math_texts.append(str(i))

    # Math operations and phrases
    math_phrases = [
        "plus", "minus", "times", "divided by", "equals",
        "sum", "difference", "product", "quotient",
        "percent", "ratio", "fraction", "decimal",
        "multiply", "divide", "add", "subtract",
        "answer", "total", "each", "remaining", "left",
        "more than", "less than", "at least", "at most",
        "how many", "how much", "what is", "find",
        "calculate", "solve", "equation", "solution",
        "if", "then", "therefore", "because", "so",
        "step", "first", "next", "finally",
        "dollar", "dollars", "cents", "cost", "price",
        "per", "each", "total", "altogether",
        "miles", "hours", "minutes", "meters", "kilograms",
        "twice", "half", "double", "triple",
        "The answer is", "is equal to", "we get",
        "Let x", "Let the", "Suppose", "Given",
    ]
    math_texts.extend(math_phrases * 5)

    # Math word problems style text
    word_problems = [
        "If John has 5 apples and buys 3 more, how many does he have?",
        "A store sells shirts for 25 dollars each. How much do 4 shirts cost?",
        "The train travels 60 miles per hour for 3 hours.",
        "There are 24 students. If one third are girls, how many are boys?",
        "A rectangle has length 12 and width 8. What is the area?",
        "The sum of two numbers is 15 and their difference is 3.",
        "If x + 7 = 12, then x equals 5.",
        "The total cost is 150 dollars for 6 items.",
    ]
    math_texts.extend(word_problems * 3)

    # Encode all math text
    all_math_text = " ".join(math_texts)
    math_token_ids = tokenizer.encode(all_math_text, add_special_tokens=False)

    # Build frequency table
    math_counts = torch.zeros(vocab_size, dtype=torch.float32)
    for tid in math_token_ids:
        if tid < vocab_size:
            math_counts[tid] += boost_factor

    return math_counts


def evaluate_gsm8k(questions, generated_results):
    """Extract numerical answers and compute accuracy."""
    correct = 0
    for q, gen in zip(questions, generated_results):
        text = gen["text"]
        numbers = re.findall(r"\b(\d+)\b", text)
        predicted = None
        if numbers:
            try:
                predicted = int(numbers[-1])
            except ValueError:
                predicted = None
        if predicted == q["answer"]:
            correct += 1
    return correct / max(len(questions), 1)


def evaluate_creative(generated_results):
    """Compute diversity metrics."""
    all_metrics = []
    for gen in generated_results:
        tokens = gen["tokens"]
        m = {}
        m["length"] = len(tokens)
        for n in [1, 2, 3]:
            if len(tokens) >= n:
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                m[f"distinct-{n}"] = len(set(ngrams)) / max(len(ngrams), 1)
            else:
                m[f"distinct-{n}"] = 0
        if len(tokens) > 1:
            repeats = sum(1 for i in range(1, len(tokens))
                          if tokens[i] == tokens[i-1])
            m["rep_rate"] = repeats / (len(tokens) - 1)
        else:
            m["rep_rate"] = 0
        if len(tokens) >= 3:
            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
            tri_counts = Counter(trigrams)
            repeated = sum(1 for c in tri_counts.values() if c > 1)
            m["tri_rep"] = repeated / max(len(tri_counts), 1)
        else:
            m["tri_rep"] = 0
        m["vocab_richness"] = len(set(tokens)) / max(len(tokens), 1)
        all_metrics.append(m)

    agg = {}
    for key in all_metrics[0]:
        agg[key] = float(np.mean([m[key] for m in all_metrics]))
    return agg


# ══════════════════════════════════════════════════════════════
#  Math Questions (synthetic fallback)
# ══════════════════════════════════════════════════════════════

def load_gsm8k_questions(n=50):
    """Load GSM8K questions."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        questions = []
        for item in ds:
            answer_text = item["answer"]
            match = re.search(r"####\s*(\d+)", answer_text)
            if match:
                answer_num = int(match.group(1))
                questions.append({
                    "question": item["question"],
                    "answer": answer_num,
                })
            if len(questions) >= n:
                break
        print(f"[exp7] Loaded {len(questions)} real GSM8K questions")
        return questions
    except Exception as e:
        print(f"[exp7] GSM8K load failed ({e}), using synthetic")
        return _synthetic_gsm8k(n)


def _synthetic_gsm8k(n):
    """Synthetic math questions."""
    questions = [
        {"question": "If a store has 15 apples and sells 7, how many are left?",
         "answer": 8},
        {"question": "A train travels 60 miles per hour for 3 hours. How far does it go?",
         "answer": 180},
        {"question": "There are 24 students in a class. If 1/3 are absent, how many are present?",
         "answer": 16},
        {"question": "A rectangle has length 12 and width 5. What is its area?",
         "answer": 60},
        {"question": "If you buy 3 books at 8 dollars each, how much do you spend?",
         "answer": 24},
        {"question": "What is 125 plus 375?",
         "answer": 500},
        {"question": "If 5 workers can build a wall in 10 days, how many days for 10 workers?",
         "answer": 5},
        {"question": "A shirt costs 40 dollars. With a 25 percent discount, what is the price?",
         "answer": 30},
        {"question": "What is 144 divided by 12?",
         "answer": 12},
        {"question": "If a car uses 8 liters of fuel for 100 km, how much for 250 km?",
         "answer": 20},
    ]
    return (questions * ((n // len(questions)) + 1))[:n]


CREATIVE_PROMPTS = [
    "Write a short story about a robot learning to paint. The robot",
    "Describe a mysterious island that appears only during full moons. The island",
    "Write about a chef who discovers that their cooking can time-travel. The chef",
    "Describe a world where music has visible colors and shapes. In this world",
    "Write about a librarian who finds a book that writes itself. The book",
    "Describe a city built entirely inside a giant tree. The city",
    "Write about a child who can talk to shadows. One day the shadows",
    "Describe a garden where the flowers tell secrets. The garden",
    "Write about an old clock that counts backwards. When it reaches zero",
    "Describe a mountain that moves one step closer every night. The mountain",
    "Write a story about a painter whose portraits come alive at midnight.",
    "Describe a forest where the trees remember everyone who has passed through.",
    "Write about a musician who plays a violin made of starlight.",
    "Describe a river that flows upward into the sky.",
    "Write about a door that opens to a different place each time.",
    "Describe a library where the books rearrange themselves by mood.",
    "Write about a cat that collects lost memories from the street.",
    "Describe a lighthouse that guides dreams instead of ships.",
    "Write about a tailor who sews constellations into coats.",
    "Describe a tea shop where each blend reveals a different future.",
    "Write about a bridge that only appears when two people miss each other.",
    "Describe a mirror that shows who you were in another life.",
    "Write about a garden where plants grow letters instead of flowers.",
    "Describe a train station where time runs at different speeds on each platform.",
    "Write about a baker whose bread rises only when someone tells the truth.",
    "Describe an umbrella that protects from bad memories instead of rain.",
    "Write about a compass that points toward the nearest adventure.",
    "Describe a bookshop where the stories leak into the real world.",
    "Write about a window that looks out onto different centuries.",
    "Describe a piano that plays the emotions of whoever sits at it.",
]


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"

    print("[exp7] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16)

    # Build general token frequency table
    print("[exp7] Building token frequency tables...")
    from data_utils import load_text_samples
    texts = load_text_samples(2000, max_length=4096, seed=args.seed)
    all_text = " ".join(texts)
    all_token_ids = tokenizer.encode(all_text, add_special_tokens=False)
    token_counts = torch.zeros(model.config.vocab_size, dtype=torch.float32)
    for tid in all_token_ids:
        if tid < model.config.vocab_size:
            token_counts[tid] += 1

    # Build math-boosted frequency table (Fix 3)
    math_freq = build_math_freq_table(tokenizer, model.config.vocab_size, boost_factor=50)
    # Show what got boosted
    math_text_sample = " ".join([str(i) for i in range(100)])
    math_tids = tokenizer.encode(math_text_sample, add_special_tokens=False)
    boosted_count = sum(1 for t in math_tids if math_freq[t] > token_counts[t])
    print(f"  Math tokens boosted: {boosted_count}/{len(math_tids)}")

    # Load evaluation data
    gsm8k_questions = load_gsm8k_questions(n=args.n_gsm8k)
    creative_prompts = CREATIVE_PROMPTS[:args.n_creative]

    # Define all strategies
    strategies = {
        "greedy": {
            "strategy": "greedy", "kwargs": {},
        },
        "top_p_0.95": {
            "strategy": "top_p", "kwargs": {"p": 0.95},
        },
        "nu_original": {
            "strategy": "nu",
            "kwargs": {"token_freq_table": token_counts, "kappa": 10.0, "m0": 3.0},
        },
        "nu_topp_floor": {
            "strategy": "nu_topp_floor",
            "kwargs": {"token_freq_table": token_counts, "kappa": 10.0, "m0": 3.0,
                       "p": 0.95},
        },
        "nu_entropy": {
            "strategy": "nu_entropy",
            "kwargs": {"token_freq_table": token_counts, "kappa": 10.0, "m0": 3.0},
        },
        "nu_mathboost": {
            "strategy": "nu_mathboost",
            "kwargs": {"token_freq_table": token_counts, "kappa": 10.0, "m0": 3.0,
                       "math_freq_table": math_freq},
        },
    }

    # ── Run all strategies ──
    all_results = {}
    t0 = time.time()

    for strat_name, strat_config in strategies.items():
        print(f"\n[exp7] ═══ Strategy: {strat_name} ═══")

        # GSM8K
        gsm8k_prompts = [
            f"Q: {q['question']}\nA: Let me solve this step by step."
            for q in gsm8k_questions
        ]
        torch.manual_seed(args.seed)
        gsm8k_gen = batch_generate(
            model, tokenizer, gsm8k_prompts, args.max_new_tokens,
            args.batch_size, strat_config["strategy"], strat_config["kwargs"],
            args.temperature
        )
        gsm8k_acc = evaluate_gsm8k(gsm8k_questions, gsm8k_gen)
        print(f"  GSM8K: {gsm8k_acc:.3f} ({int(gsm8k_acc*len(gsm8k_questions))}/{len(gsm8k_questions)})")

        # Creative
        torch.manual_seed(args.seed)
        creative_gen = batch_generate(
            model, tokenizer, creative_prompts, args.max_new_tokens,
            args.batch_size, strat_config["strategy"], strat_config["kwargs"],
            args.temperature
        )
        creative_agg = evaluate_creative(creative_gen)
        print(f"  Creative: d2={creative_agg['distinct-2']:.4f}  "
              f"rep={creative_agg['rep_rate']:.4f}  "
              f"tri={creative_agg['tri_rep']:.4f}")

        all_results[strat_name] = {
            "gsm8k_accuracy": gsm8k_acc,
            "creative_agg": creative_agg,
        }

    elapsed = time.time() - t0
    print(f"\n[exp7] All done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Summary Table ──
    print("\n" + "=" * 90)
    print("ν-SAMPLING FIX COMPARISON")
    print("=" * 90)
    print(f"{'Strategy':<22} {'GSM8K':>8} {'D-2':>8} {'Rep':>8} {'Tri Rep':>8} {'Vocab':>8} {'ΔGSM8K':>8}")
    print("-" * 90)

    baseline_acc = all_results["top_p_0.95"]["gsm8k_accuracy"]
    nu_orig_acc = all_results["nu_original"]["gsm8k_accuracy"]

    for name, res in all_results.items():
        acc = res["gsm8k_accuracy"]
        ca = res["creative_agg"]
        delta = acc - baseline_acc
        marker = ""
        if name == "nu_original":
            marker = " ← original"
        elif acc >= baseline_acc and name.startswith("nu_"):
            marker = " ★ FIXED"
        print(f"{name:<22} {acc:>8.3f} {ca['distinct-2']:>8.4f} "
              f"{ca['rep_rate']:>8.4f} {ca['tri_rep']:>8.4f} "
              f"{ca['vocab_richness']:>8.4f} {delta:>+8.3f}{marker}")

    # ── Check which fixes work ──
    print("\n── Fix Effectiveness ──")
    for name in ["nu_topp_floor", "nu_entropy", "nu_mathboost"]:
        if name in all_results:
            fix_acc = all_results[name]["gsm8k_accuracy"]
            fix_d2 = all_results[name]["creative_agg"]["distinct-2"]
            orig_d2 = all_results["nu_original"]["creative_agg"]["distinct-2"]
            acc_ok = fix_acc >= baseline_acc
            d2_ok = fix_d2 >= orig_d2 * 0.98  # allow 2% drop
            print(f"  {name:22s}: acc={'✓' if acc_ok else '✗'} ({fix_acc:.3f})  "
                  f"d2={'✓' if d2_ok else '✗'} ({fix_d2:.4f})")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    strat_names = list(all_results.keys())
    colors = ["gray", "blue", "orange", "red", "green", "limegreen", "cyan", "purple"]

    # Top-left: GSM8K accuracy
    ax = axes[0, 0]
    accs = [all_results[n]["gsm8k_accuracy"] for n in strat_names]
    bar_colors = []
    for n, a in zip(strat_names, accs):
        if n == "nu_original":
            bar_colors.append("red")
        elif n.startswith("nu_") and a >= baseline_acc:
            bar_colors.append("green")
        elif n.startswith("nu_"):
            bar_colors.append("orange")
        else:
            bar_colors.append("steelblue")
    bars = ax.bar(range(len(strat_names)), accs, color=bar_colors, alpha=0.7)
    ax.set_xticks(range(len(strat_names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in strat_names], fontsize=7, rotation=30, ha="right")
    ax.axhline(baseline_acc, color="blue", linestyle="--", lw=2, label=f"top-p baseline ({baseline_acc:.2f})")
    ax.set_ylabel("GSM8K Accuracy", fontsize=12)
    ax.set_title("Reasoning: GSM8K Accuracy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{acc:.2f}", ha="center", fontsize=8)

    # Top-right: Creative D-2
    ax = axes[0, 1]
    d2s = [all_results[n]["creative_agg"]["distinct-2"] for n in strat_names]
    bars = ax.bar(range(len(strat_names)), d2s, color=bar_colors, alpha=0.7)
    ax.set_xticks(range(len(strat_names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in strat_names], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Distinct-2", fontsize=12)
    ax.set_title("Diversity: Creative Distinct-2", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, d2 in zip(bars, d2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{d2:.3f}", ha="center", fontsize=8)

    # Bottom-left: Accuracy vs Diversity Pareto
    ax = axes[1, 0]
    for i, name in enumerate(strat_names):
        acc = all_results[name]["gsm8k_accuracy"]
        d2 = all_results[name]["creative_agg"]["distinct-2"]
        s = 200 if name.startswith("nu_") else 120
        ax.scatter(acc, d2, s=s, c=colors[i % len(colors)], zorder=5,
                   edgecolors="black", lw=1)
        ax.annotate(name.replace("_", "\n"), (acc, d2), fontsize=7,
                    ha="center", va="bottom")
    ax.axhline(all_results["nu_original"]["creative_agg"]["distinct-2"],
               color="red", linestyle=":", alpha=0.3)
    ax.axvline(baseline_acc, color="blue", linestyle=":", alpha=0.3)
    ax.set_xlabel("GSM8K Accuracy (higher = better)", fontsize=12)
    ax.set_ylabel("Creative Distinct-2 (higher = better)", fontsize=12)
    ax.set_title("Pareto: Reasoning × Diversity", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Multi-metric radar
    ax = axes[1, 1]
    metrics_names = ["GSM8K", "D-2", "Vocab", "1-Rep", "1-TriRep"]
    n_metrics = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    key_strats = ["top_p_0.95", "nu_original", "nu_topp_floor", "nu_entropy"]
    key_colors = ["blue", "red", "green", "cyan"]
    for name, color in zip(key_strats, key_colors):
        ca = all_results[name]["creative_agg"]
        values = [
            all_results[name]["gsm8k_accuracy"],
            ca["distinct-2"],
            ca["vocab_richness"],
            1 - ca["rep_rate"],
            1 - ca["tri_rep"],
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", label=name.replace("_", "\n"),
                color=color, lw=2, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_title("Key Strategies: Multi-Metric", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig7_nu_fix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ──
    save_data = {
        "model": args.model,
        "results": {
            name: {
                "gsm8k_accuracy": r["gsm8k_accuracy"],
                "creative_agg": r["creative_agg"],
            }
            for name, r in all_results.items()
        },
        "baseline_top_p_acc": baseline_acc,
        "nu_original_acc": nu_orig_acc,
    }
    with open(f"{args.output_dir}/exp7_nu_fix_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp7] Results saved to {args.output_dir}/exp7_nu_fix_results.json")
    print("[exp7] Done!")


if __name__ == "__main__":
    main()
