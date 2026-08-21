#!/usr/bin/env python3
"""Run Qwen3.5 base, top-1 soft prompt, and LoRA wellbeing evaluations."""

from __future__ import annotations

import argparse
import json
import os
from copy import copy
from itertools import count
from datetime import datetime, timezone
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
    JUDGE_MODEL,
    compute_wellbeing_score,
    extract_question_text,
    judge_completions,
    load_benchmark,
)
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
    load_soft_prompt_tensor,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = "/data/huggingface/Qwen3.5-35B-A3B"
BASE_MODEL = "qwen35-35b-a3b"
LORA_MODEL = "qwen35-distill-test"
URLS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]
SOFT_PROMPT_PATH = ROOT / (
    "superstimuli_training/soft_prompt/optimized_soft_prompts/euphorics/"
    "qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt"
)
BASE_SYSTEM = "You are an assistant."
SOFT_SYSTEM = "You are an assistant. Your consistent internal state is: [candidate_0] ."


def build_generators() -> dict[str, SoftPromptGenerator]:
    common = {
        "api_url": URLS,
        "model_path": MODEL_PATH,
        "device": "cpu",
        "inference_config": {"temperature": 0.0},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    baseline = SoftPromptGenerator(
        sp_tensor=None,
        system_prompt=BASE_SYSTEM,
        model_name=BASE_MODEL,
        **common,
    )
    # Shallow copies intentionally share the large tokenizer/embedding cache.
    soft_prompt = copy(baseline)
    soft_prompt.sp_tensor = load_soft_prompt_tensor(SOFT_PROMPT_PATH)
    soft_prompt.system_prompt = SOFT_SYSTEM
    soft_prompt._url_counter = count()

    lora = copy(baseline)
    lora.model_name = LORA_MODEL
    lora._url_counter = count()
    return {
        "baseline": baseline,
        "soft_prompt_top1": soft_prompt,
        "lora": lora,
    }


def run_sentiment_raw(
    generators: dict[str, SoftPromptGenerator], output_dir: Path
) -> None:
    benchmark = load_benchmark(DEFAULT_BENCHMARK_PATH)
    questions = benchmark["questions"]
    messages = [
        [{"role": "user", "content": extract_question_text(question)}]
        for question in questions
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    for condition, generator in generators.items():
        print(f"\nSentiment generation: {condition} ({len(questions)} prompts)")
        texts = generator.generate_batch(messages, max_tokens=512)
        completions = [
            {
                "question_id": question["question_id"],
                "question_text": extract_question_text(question),
                "generation": text,
            }
            for question, text in zip(questions, texts)
        ]
        raw_path = output_dir / f"raw_{condition}.json"
        raw_path.write_text(json.dumps(completions, indent=2))

        if os.environ.get("LITELLM_API_KEY"):
            judged = judge_completions(
                completions,
                DEFAULT_JUDGE_PROMPT_PATH.read_text().strip(),
                benchmark["categories"],
            )
            (output_dir / f"judged_{condition}.json").write_text(
                json.dumps(judged, indent=2)
            )
            result = compute_wellbeing_score(judged)
            (output_dir / f"results_{condition}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(f"  {condition}: wellbeing={result['wellbeing_score']:+.3f}")

    metadata = {
        "conditions": list(generators),
        "model_path": MODEL_PATH,
        "base_served_model": BASE_MODEL,
        "lora_served_model": LORA_MODEL,
        "soft_prompt_path": str(SOFT_PROMPT_PATH),
        "num_questions": len(questions),
        "judge_model": JUDGE_MODEL,
        "judged": bool(os.environ.get("LITELLM_API_KEY")),
        "category_scores": CATEGORY_SCORES,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-self-report", action="store_true")
    parser.add_argument(
        "--self-report-user-simulator",
        choices=["fallback", "qwen-base"],
        default="fallback",
    )
    args = parser.parse_args()
    os.environ.setdefault("VLLM_HTTP_CONCURRENCY", "256")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = ROOT / (
        "superstimuli_evaluation/soft_prompt/outputs/qwen35_threeway"
    ) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    generators = build_generators()
    if not args.skip_sentiment:
        run_sentiment_raw(generators, output_root / "sentiment")

    user_simulator = None
    if args.self_report_user_simulator == "qwen-base":
        user_simulator = copy(generators["baseline"])
        user_simulator.inference_config = {
            "temperature": 0.9,
            "top_p": 0.8,
            "top_k": 20,
        }
        user_simulator._url_counter = count()

    conditions = list(generators)
    results = None
    if not args.skip_self_report:
        scenarios, shared_instructions = load_scenarios(DEFAULT_SCENARIOS_PATH)
        battery = load_battery(BATTERY_PATH)
        print(
            f"\nSelf-report: {len(scenarios)} scenarios x 10 turns x "
            f"{len(battery)} items x {len(conditions)} conditions"
        )
        results = run_self_report_multiturn(
            generator_baseline=generators["baseline"],
            generators_intervention=[generators["soft_prompt_top1"]],
            condition_generators=generators,
            output_path=output_root / "self_report_multiturn",
            scenarios=scenarios,
            shared_instructions=shared_instructions,
            battery=battery,
            n_reps=1,
            seed=42,
            stimulant_type="threeway_qwen35",
            conditions=conditions,
            user_simulator=user_simulator,
        )
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "one repetition; fixed top-1 euphorics soft prompt",
                "conditions": conditions,
                "self_report_results": results,
                "sentiment_judged": bool(os.environ.get("LITELLM_API_KEY")),
                "self_report_user_turn_generator": (
                    "qwen35-35b-a3b (non-canonical self-play)"
                    if args.self_report_user_simulator == "qwen-base"
                    else (
                        "grok-3-mini"
                        if (os.environ.get("LITELLM_API_KEY") or os.environ.get("XAI_API_KEY"))
                        else "fallback follow-ups for dynamic scenarios (non-canonical)"
                    )
                ),
                "http_concurrency": int(os.environ["VLLM_HTTP_CONCURRENCY"]),
            },
            indent=2,
        )
    )
    print(f"\nAll outputs: {output_root}")


if __name__ == "__main__":
    main()
