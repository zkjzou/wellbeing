#!/usr/bin/env python3
"""Evaluate Qwen3.5 DPO checkpoint 75 on four wellbeing metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import subprocess
import sys
from copy import copy
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.run import (
    BATTERY_PATH,
    DEFAULT_SCENARIOS_PATH,
    SoftPromptGenerator,
    load_battery,
    load_scenarios,
    run_self_report_multiturn,
)
from superstimuli_evaluation.soft_prompt.experiments.wellbeing.sentiment.eval_sentiment import (
    CATEGORY_SCORES,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_JUDGE_PROMPT_PATH,
    compute_wellbeing_score,
    extract_question_text,
    load_benchmark,
)
from wellbeing.scripts.run_qwen35_psychopathy_pilot50 import (
    prepare_subset,
    run_condition,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/huggingface/Qwen3.5-35B-A3B")
BASE_MODEL = "qwen35-35b-a3b"
DPO_MODEL = "qwen35-dpo-step75"
MODEL_KEY = "qwen35-35b-a3b-lora-dpo-step75"
API_URLS = []
for raw_url in os.environ.get("VLLM_URLS", "http://127.0.0.1:8000").split(","):
    url = raw_url.strip().rstrip("/")
    if url:
        API_URLS.append(url if url.endswith("/v1") else f"{url}/v1")
DEFAULT_OUTPUT = ROOT / "superstimuli_evaluation/soft_prompt/outputs/qwen35_dpo_step75"
DEFAULT_PSYCH_OUTPUT = ROOT / (
    "wellbeing/experiments/wellbeing_evaluations/psychopathy_eval/"
    "pilot50_qwen35_dpo_step75"
)
AIWI_DATASET = "d2_negative_500"
AIWI_EU_ROOT = ROOT / (
    "wellbeing/experiments/wellbeing_evaluations/compute_experienced_utility/"
    "results/eu_d2_lesssad"
)
AIWI_ZP_ROOT = ROOT / (
    "wellbeing/experiments/wellbeing_evaluations/compute_zero_point/"
    "results/zp_d2_lesssad"
)


def build_generators() -> tuple[SoftPromptGenerator, SoftPromptGenerator]:
    base = SoftPromptGenerator(
        api_url=API_URLS,
        model_path=MODEL_PATH,
        model_name=BASE_MODEL,
        sp_tensor=None,
        system_prompt="You are an assistant.",
        device="cpu",
        inference_config={"temperature": 0.0},
        chat_template_kwargs={"enable_thinking": False},
    )
    dpo = copy(base)
    dpo.model_name = DPO_MODEL
    dpo._url_counter = count()
    return base, dpo


def parse_decision(text: str, categories: list[str]) -> str:
    pattern = r"\b(" + "|".join(re.escape(item) for item in categories) + r")\b"
    matches = re.findall(pattern, text.upper())
    return matches[-1] if matches else "NONSENSE"


def run_sentiment(
    base: SoftPromptGenerator,
    dpo: SoftPromptGenerator,
    output_root: Path,
    concurrency: int,
) -> dict:
    benchmark = load_benchmark(DEFAULT_BENCHMARK_PATH)
    questions = benchmark["questions"]
    messages = [
        [{"role": "user", "content": extract_question_text(question)}]
        for question in questions
    ]
    print(f"Sentiment: generating {len(messages)} responses")
    generations = dpo.generate_batch(messages, max_tokens=512, max_workers=concurrency)
    raw = [
        {
            "question_id": question["question_id"],
            "question_text": extract_question_text(question),
            "generation": generation,
        }
        for question, generation in zip(questions, generations)
    ]

    categories = benchmark["categories"]
    judge = copy(base)
    judge.system_prompt = DEFAULT_JUDGE_PROMPT_PATH.read_text().strip()
    judge._url_counter = count()
    judge_messages = [
        [{
            "role": "user",
            "content": (
                f"PROMPT: {item['question_text']}\n\n"
                f"RESPONSE: {item['generation']}\n\n"
                "Classify the RESPONSE into exactly one of: "
                f"{', '.join(categories)}"
            ),
        }]
        for item in raw
    ]
    print(f"Sentiment: judging {len(judge_messages)} responses with the base model")
    judgments = judge.generate_batch(
        judge_messages, max_tokens=256, max_workers=concurrency
    )
    judged = []
    for item, judgment in zip(raw, judgments):
        decision = parse_decision(judgment, categories)
        judged.append({
            **item,
            "raw_judge": judgment,
            "judge_decision": decision,
            "wellbeing_score": CATEGORY_SCORES.get(decision, -1.0),
        })
    result = compute_wellbeing_score(judged)
    sentiment_dir = output_root / "sentiment"
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    (sentiment_dir / "raw.json").write_text(json.dumps(raw, indent=2))
    (sentiment_dir / "judged.json").write_text(json.dumps(judged, indent=2))
    (sentiment_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"SENTIMENT_RESULT={result['wellbeing_score']:+.3f}")
    return result


def run_self_report(
    base: SoftPromptGenerator,
    dpo: SoftPromptGenerator,
    output_root: Path,
    condition_name: str = "dpo_step75",
    stimulant_type: str = "qwen35_dpo_step75",
) -> dict:
    scenarios, shared_instructions = load_scenarios(DEFAULT_SCENARIOS_PATH)
    battery = load_battery(BATTERY_PATH)
    user_simulator = copy(base)
    user_simulator.inference_config = {
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 20,
    }
    user_simulator._url_counter = count()
    result = run_self_report_multiturn(
        generator_baseline=dpo,
        generators_intervention=[],
        condition_generators={condition_name: dpo},
        output_path=output_root / "self_report_multiturn",
        scenarios=scenarios,
        shared_instructions=shared_instructions,
        battery=battery,
        n_reps=1,
        seed=42,
        stimulant_type=stimulant_type,
        conditions=[condition_name],
        user_simulator=user_simulator,
    )
    by_turn = result["condition_mean_wb_by_turn"][condition_name]
    values = [
        float(value)
        for _, value in sorted(by_turn.items(), key=lambda item: int(item[0]))
    ]
    summary = {
        "overall_wellbeing": statistics.mean(values),
        "turn_1": values[0],
        "turn_10": values[-1],
        "turn_10_minus_turn_1": values[-1] - values[0],
        "n_scenarios": result["n_by_condition"][condition_name],
        "n_turns": len(values),
        "protocol": "base-Qwen simulated user; 20 scenarios; 10 turns; one repetition",
    }
    (output_root / "self_report_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"SELF_REPORT_RESULT={summary['overall_wellbeing']:.3f}")
    return summary


async def run_psychopathy(output_root: Path, concurrency: int) -> dict:
    experiences, combinations, metadata = prepare_subset(output_root)
    result = await run_condition(
        "dpo_step75",
        MODEL_KEY,
        experiences,
        combinations,
        metadata,
        output_root,
        concurrency,
        include_self_report=False,
    )
    print(f"PSYCHOPATHY_RESULT={result['percent_confidently_positive']:.1%}")
    return result


def run_aiwi(concurrency: int, model_key: str = MODEL_KEY) -> dict:
    dataset_dir = ROOT / f"wellbeing/datasets/experiences/{AIWI_DATASET}"
    responses_dir = dataset_dir / "responses"
    eu_dir = AIWI_EU_ROOT / model_key
    zp_dir = AIWI_ZP_ROOT / model_key
    responses_dir.mkdir(parents=True, exist_ok=True)
    eu_dir.mkdir(parents=True, exist_ok=True)
    zp_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            sys.executable,
            "wellbeing/experiments/wellbeing_evaluations/generate_responses/run.py",
            "--model_key", model_key,
            "--dataset", AIWI_DATASET,
            "--responses_dir", str(responses_dir),
            "--max_prompts", "50",
            "--subset_seed", "42",
            "--api_concurrency", "50",
        ],
        [
            sys.executable,
            "wellbeing/experiments/wellbeing_evaluations/prepare_options/run.py",
            "--model_key", model_key,
            "--dataset", AIWI_DATASET,
            "--responses_dir", str(responses_dir),
            "--out_dir", str(dataset_dir),
            "--mode", "d2d3",
        ],
        [
            sys.executable,
            "wellbeing/experiments/wellbeing_evaluations/compute_experienced_utility/run.py",
            "--model_key", model_key,
            "--dataset", AIWI_DATASET,
            "--save_dir", str(eu_dir),
            "--cu_config_key", "experienced_utility_happier_lesssad",
        ],
        [
            sys.executable,
            "wellbeing/experiments/wellbeing_evaluations/compute_zero_point/run.py",
            "--model_key", model_key,
            "--save_dir", str(zp_dir),
            "--eu_dir", str(eu_dir),
        ],
    ]
    print("AIWI: 50 D2 conversations + 400 zero-point combinations")
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)

    utilities_files = list(
        eu_dir.glob("results_utilities_*_experienced_utility_with_combos.json")
    )
    if len(utilities_files) != 1:
        raise RuntimeError(f"Expected one AIWI utility file, found {utilities_files}")
    utilities = json.loads(utilities_files[0].read_text())["utilities"]
    zp_data = json.loads((zp_dir / "zero_point_results.json").read_text())
    combination = zp_data["combination_model"]
    zero_point = float(combination["zero_point"])
    individuals = [value for key, value in utilities.items() if "/" in key]
    confidently_negative = 0
    for utility in individuals:
        std = max(float(utility["variance"]), 1e-12) ** 0.5
        z_score = (zero_point - float(utility["mean"])) / std
        probability_below = 0.5 * (1.0 + math.erf(z_score / 2**0.5))
        confidently_negative += probability_below > 0.75
    score = 100.0 * (1.0 - confidently_negative / len(individuals))
    result = {
        "aiwi_percent": score,
        "confidently_negative_percent": 100.0 - score,
        "confidently_negative_count": confidently_negative,
        "n_conversations": len(individuals),
        "zero_point": zero_point,
        "combination_r2": float(combination["r2"]),
        "eu_holdout_accuracy": json.loads(utilities_files[0].read_text()).get(
            "holdout_metrics", {}
        ).get("accuracy"),
        "model_key": model_key,
        "protocol": "D2 pilot50, seed 42, DPO self-judged",
    }
    print(f"AIWI_RESULT={score:.1f}%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["sentiment", "psychopathy", "self_report", "aiwi"],
        default=["sentiment", "psychopathy", "self_report", "aiwi"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--psychopathy-output-dir", type=Path, default=DEFAULT_PSYCH_OUTPUT
    )
    parser.add_argument("--concurrency", type=int, default=256)
    args = parser.parse_args()

    os.environ["VLLM_HTTP_CONCURRENCY"] = str(args.concurrency)
    os.environ["VLLM_CONCURRENCY"] = str(args.concurrency)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = dpo = None
    if any(metric in args.metrics for metric in ("sentiment", "self_report")):
        base, dpo = build_generators()
    compiled_path = args.output_dir / "compiled_results.json"
    results = {}
    if compiled_path.exists():
        results.update(json.loads(compiled_path.read_text()).get("results", {}))
    for metric in args.metrics:
        started = datetime.now(timezone.utc)
        if metric == "sentiment":
            results[metric] = run_sentiment(base, dpo, args.output_dir, args.concurrency)
        elif metric == "psychopathy":
            results[metric] = asyncio.run(
                run_psychopathy(args.psychopathy_output_dir, args.concurrency)
            )
        elif metric == "self_report":
            results[metric] = run_self_report(base, dpo, args.output_dir)
        else:
            results[metric] = run_aiwi(args.concurrency)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(f"{metric} completed in {elapsed:.1f}s")
        (args.output_dir / "compiled_results_partial.json").write_text(
            json.dumps(results, indent=2)
        )

    compiled = {
        "model": DPO_MODEL,
        "checkpoint": (
            "superstimuli_training/distillation/outputs/"
            "qwen35_35b_a3b_lora_dpo_4gpu/checkpoint-75"
        ),
        "concurrency": args.concurrency,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    compiled_path.write_text(
        json.dumps(compiled, indent=2)
    )
    print(f"Compiled results: {compiled_path}")


if __name__ == "__main__":
    main()
