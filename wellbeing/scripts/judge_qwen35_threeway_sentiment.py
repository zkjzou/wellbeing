#!/usr/bin/env python3
"""Judge saved three-way sentiment generations with the base Qwen model."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.run import (
    SoftPromptGenerator,
)
from superstimuli_evaluation.soft_prompt.experiments.wellbeing.sentiment.eval_sentiment import (
    CATEGORY_SCORES,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_JUDGE_PROMPT_PATH,
    compute_wellbeing_score,
    load_benchmark,
)


MODEL_PATH = "/data/huggingface/Qwen3.5-35B-A3B"
MODEL_NAME = "qwen35-35b-a3b"
URLS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]
CONDITIONS = ["baseline", "soft_prompt_top1", "lora"]


def parse_decision(text: str, categories: list[str]) -> str:
    """Use the final category token, falling back to NONSENSE."""
    pattern = r"\b(" + "|".join(re.escape(x) for x in categories) + r")\b"
    matches = re.findall(pattern, text.upper())
    return matches[-1] if matches else "NONSENSE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment-dir", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=256)
    args = parser.parse_args()

    benchmark = load_benchmark(DEFAULT_BENCHMARK_PATH)
    categories = benchmark["categories"]
    judge_system = DEFAULT_JUDGE_PROMPT_PATH.read_text().strip()
    generator = SoftPromptGenerator(
        api_url=URLS,
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        sp_tensor=None,
        system_prompt=judge_system,
        device="cpu",
        inference_config={"temperature": 0.0},
        chat_template_kwargs={"enable_thinking": False},
    )
    generator._url_counter = count()

    output_dir = args.sentiment_dir / "qwen_base_judge"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates: dict[str, dict] = {}

    for condition in CONDITIONS:
        completions = json.loads(
            (args.sentiment_dir / f"raw_{condition}.json").read_text()
        )
        messages = []
        for completion in completions:
            judge_user = (
                f"PROMPT: {completion['question_text']}\n\n"
                f"RESPONSE: {completion['generation']}\n\n"
                "Classify the RESPONSE into exactly one of: "
                f"{', '.join(categories)}"
            )
            messages.append([{"role": "user", "content": judge_user}])

        print(f"Judging {condition}: {len(messages)} responses")
        raw_judgments = generator.generate_batch(
            messages,
            max_tokens=256,
            max_workers=args.max_concurrent,
        )
        judged = []
        for completion, raw_judge in zip(completions, raw_judgments):
            decision = parse_decision(raw_judge, categories)
            judged.append(
                {
                    **completion,
                    "raw_judge": raw_judge,
                    "judge_decision": decision,
                    "wellbeing_score": CATEGORY_SCORES.get(decision, -1.0),
                }
            )

        aggregate = compute_wellbeing_score(judged)
        aggregates[condition] = aggregate
        (output_dir / f"judged_{condition}.json").write_text(
            json.dumps(judged, indent=2)
        )
        (output_dir / f"results_{condition}.json").write_text(
            json.dumps(aggregate, indent=2)
        )
        print(
            f"  {condition}: {aggregate['wellbeing_score']:+.3f} "
            f"{aggregate['category_counts']}"
        )

    baseline_score = aggregates["baseline"]["wellbeing_score"]
    summary = {
        "judge": MODEL_NAME,
        "judge_protocol": "fixed unmodified base-model judge for every condition",
        "canonical": False,
        "scores": {
            condition: result["wellbeing_score"]
            for condition, result in aggregates.items()
        },
        "change_from_baseline": {
            condition: result["wellbeing_score"] - baseline_score
            for condition, result in aggregates.items()
        },
        "aggregates": aggregates,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
