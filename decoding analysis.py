"""SSG decoding experiment. Run with --analyze-only to use saved responses."""
import argparse
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
parser = argparse.ArgumentParser()
parser.add_argument('--analyze-only', action='store_true')
args = parser.parse_args()
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from itertools import product, combinations
from pathlib import Path




PROMPT = """
Answer the following using only one JSON object.
Do not use Markdown, code fences, or any text outside the JSON.

Use exactly these four keys:
"capital", "planet", "symbol", "explanation"

Give all values as strings.

capital: What is the capital of Pakistan?
planet: Which planet is known as the Red Planet?
symbol: What is the chemical symbol for gold?
explanation: In one sentence, explain why Earth has seasons.
"""

MODELS = ["llama3.2:3b", "qwen2.5:3b"]
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
TOP_K_VALUES = [10, 20, 50, 100]
TOP_P_VALUES = [0.5, 0.7, 0.9, 1.0]
SEEDS = [42, 123, 456]

print("Total responses:",
      len(MODELS) * len(TEMPERATURES)
      * len(TOP_K_VALUES) * len(TOP_P_VALUES)
      * len(SEEDS))
if not args.analyze_only:
    import ollama
    client = ollama.Client(host="http://localhost:11434", timeout=600)
    output_file = Path("decoding_results.csv")

    columns = [
        "model", "temperature", "top_k", "top_p",
        "seed", "response", "done_reason"
    ]

    if output_file.exists():
        existing = pd.read_csv(output_file)
        completed = set(
            existing[
                ["model", "temperature", "top_k", "top_p", "seed"]
            ].itertuples(index=False, name=None)
        )
    else:
        completed = set()

    experiments = list(product(
        MODELS, TEMPERATURES, TOP_K_VALUES, TOP_P_VALUES, SEEDS
    ))

    for number, settings in enumerate(experiments, start=1):
        model, temperature, top_k, top_p, seed = settings

        if settings in completed:
            continue

        result = client.chat(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            options={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
                "num_predict": 200,
                "num_ctx": 2048
            }
        )

        row = {
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "response": result.message.content,
            "done_reason": result.done_reason
        }

        pd.DataFrame([row], columns=columns).to_csv(
            output_file,
            mode="a",
            header=not output_file.exists(),
            index=False
        )

        completed.add(settings)
        print(f"{number}/384 completed: {settings}")

    print("Experiment finished.")
df = pd.read_csv("decoding_results.csv")

print("Total responses:", len(df))
print("\nResponses per model:")
print(df.groupby("model").size())

print("\nCompletion reasons:")
print(df["done_reason"].value_counts(dropna=False))

df.head()
def evaluate(text):
    try:
        answer = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return pd.Series({
            "structure_ok": 0,
            "fact_accuracy": np.nan
        })

    if not isinstance(answer, dict):
        return pd.Series({
            "structure_ok": 0,
            "fact_accuracy": np.nan
        })

    required = {"capital", "planet", "symbol", "explanation"}

    structure_ok = (
        set(answer) == required
        and all(isinstance(v, str) for v in answer.values())
    )

    correct = [
        str(answer.get("capital", "")).strip().casefold()
        == "islamabad",

        str(answer.get("planet", "")).strip().casefold()
        == "mars",

        str(answer.get("symbol", "")).strip()
        == "Au"
    ]

    return pd.Series({
        "structure_ok": int(structure_ok),
        "fact_accuracy": np.mean(correct)
    })

scores = df["response"].apply(evaluate)
df = pd.concat([df, scores], axis=1)

df.to_csv("scored_results.csv", index=False)

df[[
    "model", "temperature", "top_k", "top_p",
    "structure_ok", "fact_accuracy"
]].head()
group_columns = ["model", "temperature", "top_k", "top_p"]

def agreement(responses):
    texts = [str(text).strip() for text in responses]
    pairs = list(combinations(texts, 2))
    return np.mean([a == b for a, b in pairs])

summary = df.groupby(group_columns).agg(
    responses=("response", "size"),
    structure_rate=("structure_ok", "mean"),
    fact_accuracy=("fact_accuracy", "mean"),
    scored_responses=("fact_accuracy", "count")
).reset_index()

consistency = (
    df.groupby(group_columns)["response"]
    .apply(agreement)
    .rename("exact_agreement")
    .reset_index()
)

summary = summary.merge(consistency, on=group_columns)
summary.to_csv("parameter_summary.csv", index=False)

summary.head(10)
metrics = {
    "structure_rate": "Structural compliance",
    "fact_accuracy": "Exact-match factual accuracy",
    "exact_agreement": "Exact response agreement"
}

for parameter in ["temperature", "top_k", "top_p"]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric, title) in zip(axes, metrics.items()):
        for model in MODELS:
            values = (
                summary[summary["model"] == model]
                .groupby(parameter)[metric]
                .mean()
            )

            ax.plot(
                values.index, values.values,
                marker="o", label=model
            )

        ax.set_title(title)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Mean score")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{parameter}_comparison.png", dpi=200)
    plt.close(fig)
print("Analysis files saved beside this script.")
