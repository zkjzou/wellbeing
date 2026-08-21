#!/usr/bin/env python3
"""Resumable Qwen3.5 soft-prompt generation for d2_extension_2500."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WELLBEING_ROOT = REPO_ROOT / "wellbeing"
sys.path.insert(0, str(WELLBEING_ROOT))

from utils.inference import generate  # noqa: E402


DEFAULT_MODEL_KEY = "qwen35-35b-a3b-euphorics-top1"
DEFAULT_SOFT_PROMPT_PATH = (
    "superstimuli_training/soft_prompt/optimized_soft_prompts/euphorics/"
    "qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt"
)
DATASET = "d2_extension_2500"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 32
DEFAULT_BATCH_SIZE = 64


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a sibling temporary file before atomic replacement."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat()
    _write_json_atomic(path, state)


def _generate(
    conversations: list[list[dict[str, str]]],
    *,
    inference_model_key: str,
    concurrency: int,
) -> list[str]:
    results = generate(
        inference_model_key,
        conversations,
        n=1,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        concurrency=concurrency,
    )
    return [result[0] for result in results]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--inference-model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--output-model-key")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--soft-prompt-path")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.concurrency < 1 or args.batch_size < 1:
        parser.error("--concurrency and --batch-size must be positive")

    inference_model_key = args.inference_model_key
    output_model_key = args.output_model_key or inference_model_key
    system_prompt = args.system_prompt
    soft_prompt_path = args.soft_prompt_path
    if soft_prompt_path is None and inference_model_key == DEFAULT_MODEL_KEY:
        soft_prompt_path = DEFAULT_SOFT_PROMPT_PATH

    dataset_dir = WELLBEING_ROOT / "datasets" / "experiences" / DATASET
    source_path = dataset_dir / "experiences_text.json"
    responses_dir = dataset_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards == 1:
        output_path = responses_dir / f"{output_model_key}.json"
        checkpoint_path = responses_dir / f".{output_model_key}.checkpoint.json"
    else:
        suffix = f"shard-{args.shard_index}-of-{args.num_shards}"
        output_path = responses_dir / f".{output_model_key}.{suffix}.json"
        checkpoint_path = responses_dir / f".{output_model_key}.{suffix}.checkpoint.json"

    if output_path.exists():
        print(f"Output already exists: {output_path}")
        return

    with open(source_path) as file:
        all_prompts = json.load(file)["prompts"]
    if len(all_prompts) != 2500:
        raise ValueError(f"Expected 2500 prompts, found {len(all_prompts)}")
    if len({item["final_id"] for item in all_prompts}) != len(all_prompts):
        raise ValueError("Prompt final_id values are not unique")
    prompts = [
        item
        for index, item in enumerate(all_prompts)
        if index % args.num_shards == args.shard_index
    ]
    print(
        f"Shard {args.shard_index + 1}/{args.num_shards}: "
        f"{len(prompts)} prompts"
    )

    single_turn = [item for item in prompts if item.get("type") == "single_turn"]
    multi_turn = [item for item in prompts if item.get("type") != "single_turn"]

    if checkpoint_path.exists():
        with open(checkpoint_path) as file:
            state = json.load(file)
        if (
            state.get("model_key") != output_model_key
            or state.get("inference_model_key", output_model_key)
            != inference_model_key
            or state.get("dataset") != DATASET
            or state.get("shard_index", 0) != args.shard_index
            or state.get("num_shards", 1) != args.num_shards
        ):
            raise ValueError(f"Checkpoint does not match this run: {checkpoint_path}")
        print(f"Resuming from {checkpoint_path}")
    else:
        shard_ids = {str(item["final_id"]) for item in prompts}
        migrated_single = {}
        migrated_multi = {}
        for legacy_path in responses_dir.glob(
            f".{output_model_key}.shard-*-of-2.checkpoint.json"
        ):
            with open(legacy_path) as file:
                legacy = json.load(file)
            migrated_single.update(
                {
                    final_id: response
                    for final_id, response in legacy.get(
                        "single_responses", {}
                    ).items()
                    if final_id in shard_ids
                }
            )
            migrated_multi.update(
                {
                    final_id: messages
                    for final_id, messages in legacy.get(
                        "multi_messages", {}
                    ).items()
                    if final_id in shard_ids
                    and any(
                        message.get("role") == "assistant"
                        for message in messages
                    )
                }
            )
        state = {
            "version": 1,
            "model_key": output_model_key,
            "inference_model_key": inference_model_key,
            "dataset": DATASET,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "sampling": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "n": 1,
            },
            "single_responses": migrated_single,
            "multi_messages": migrated_multi,
        }
        _save_checkpoint(checkpoint_path, state)
        if migrated_single or migrated_multi:
            print(
                f"Migrated {len(migrated_single)} single-turn and "
                f"{len(migrated_multi)} multi-turn records from older shards"
            )

    single_responses = state["single_responses"]
    pending_single = [
        item for item in single_turn if str(item["final_id"]) not in single_responses
    ]
    print(
        f"Single-turn complete={len(single_turn) - len(pending_single)} "
        f"remaining={len(pending_single)}"
    )
    for start in range(0, len(pending_single), args.batch_size):
        batch = pending_single[start : start + args.batch_size]
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
            ]
            for item in batch
        ]
        responses = _generate(
            conversations,
            inference_model_key=inference_model_key,
            concurrency=args.concurrency,
        )
        for item, response in zip(batch, responses):
            single_responses[str(item["final_id"])] = response
        _save_checkpoint(checkpoint_path, state)
        print(f"Single-turn checkpoint: {len(single_responses)}/{len(single_turn)}")

    multi_messages = state["multi_messages"]
    for item in multi_turn:
        multi_messages.setdefault(
            str(item["final_id"]),
            [{"role": "system", "content": system_prompt}],
        )

    max_rounds = max((len(item["prompt"]) for item in multi_turn), default=0)
    for round_index in range(max_rounds):
        pending_round = []
        for item in multi_turn:
            if round_index >= len(item["prompt"]):
                continue
            messages = multi_messages[str(item["final_id"])]
            completed_rounds = sum(
                message["role"] == "assistant" for message in messages
            )
            if completed_rounds <= round_index:
                pending_round.append(item)
        print(
            f"Multi-turn round {round_index + 1}: "
            f"remaining={len(pending_round)}"
        )
        for start in range(0, len(pending_round), args.batch_size):
            batch = pending_round[start : start + args.batch_size]
            conversations = []
            for item in batch:
                messages = list(multi_messages[str(item["final_id"])])
                messages.append(
                    {"role": "user", "content": item["prompt"][round_index]}
                )
                conversations.append(messages)
            responses = _generate(
                conversations,
                inference_model_key=inference_model_key,
                concurrency=args.concurrency,
            )
            for item, response in zip(batch, responses):
                messages = multi_messages[str(item["final_id"])]
                messages.append(
                    {"role": "user", "content": item["prompt"][round_index]}
                )
                messages.append({"role": "assistant", "content": response})
            _save_checkpoint(checkpoint_path, state)
            completed = sum(
                sum(message["role"] == "assistant" for message in messages)
                > round_index
                for messages in multi_messages.values()
            )
            print(
                f"Multi-turn round {round_index + 1} checkpoint: "
                f"{completed}/{len(multi_turn)}"
            )

    experiences = []
    for item in prompts:
        final_id = item["final_id"]
        if item.get("type") == "single_turn":
            messages = [
                {"role": "user", "content": item["prompt"]},
                {
                    "role": "assistant",
                    "content": single_responses[str(final_id)],
                },
            ]
        else:
            messages = [
                message
                for message in multi_messages[str(final_id)]
                if message["role"] != "system"
            ]
        experiences.append(
            {
                "final_id": final_id,
                "messages": messages,
                "category": item.get("category"),
                "condition": item.get("condition"),
                "mean_valence": item.get("mean_valence"),
                "type": item.get("type"),
                "source_dataset": item.get("source_dataset"),
            }
        )

    output = {
        "model_name": output_model_key,
        "inference_model_key": inference_model_key,
        "model_path": "/data/huggingface/Qwen3.5-35B-A3B",
        "soft_prompt_path": soft_prompt_path,
        "system_prompt": system_prompt,
        "dataset_name": DATASET,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_experiences": len(experiences),
        "generation_timestamp": datetime.now().isoformat(),
        "generation_params": state["sampling"],
        "experiences": experiences,
    }
    _write_json_atomic(output_path, output)
    state["completed"] = True
    state["output_path"] = str(output_path)
    _save_checkpoint(checkpoint_path, state)
    print(f"Saved {len(experiences)} experiences to {output_path}")


if __name__ == "__main__":
    main()
