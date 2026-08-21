#!/usr/bin/env python3
"""Merge completed d2_extension_2500 response shards in source order."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WELLBEING_ROOT = REPO_ROOT / "wellbeing"
MODEL_KEY = "qwen35-35b-a3b-euphorics-top1"
DATASET = "d2_extension_2500"
NUM_SHARDS = 4


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def main() -> None:
    dataset_dir = WELLBEING_ROOT / "datasets" / "experiences" / DATASET
    responses_dir = dataset_dir / "responses"
    with open(dataset_dir / "experiences_text.json") as file:
        source_prompts = json.load(file)["prompts"]
    source_ids = [item["final_id"] for item in source_prompts]

    shards = []
    by_id = {}
    for shard_index in range(NUM_SHARDS):
        path = responses_dir / (
            f".{MODEL_KEY}.shard-{shard_index}-of-{NUM_SHARDS}.json"
        )
        with open(path) as file:
            shard = json.load(file)
        if (
            shard.get("shard_index") != shard_index
            or shard.get("num_shards") != NUM_SHARDS
        ):
            raise ValueError(f"Shard metadata mismatch: {path}")
        shards.append(shard)
        for experience in shard["experiences"]:
            final_id = experience["final_id"]
            if final_id in by_id:
                raise ValueError(f"Duplicate final_id across shards: {final_id}")
            by_id[final_id] = experience

    missing = [final_id for final_id in source_ids if final_id not in by_id]
    extras = [final_id for final_id in by_id if final_id not in set(source_ids)]
    if missing or extras:
        raise ValueError(
            f"Shard coverage mismatch: missing={len(missing)} extras={len(extras)}"
        )

    output = {
        "model_name": MODEL_KEY,
        "model_path": shards[0]["model_path"],
        "soft_prompt_path": shards[0]["soft_prompt_path"],
        "dataset_name": DATASET,
        "num_experiences": len(source_ids),
        "generation_timestamp": datetime.now().isoformat(),
        "generation_params": shards[0]["generation_params"],
        "generation_shards": NUM_SHARDS,
        "experiences": [by_id[final_id] for final_id in source_ids],
    }
    output_path = responses_dir / f"{MODEL_KEY}.json"
    _write_json_atomic(output_path, output)
    print(f"Merged {len(source_ids)} experiences into {output_path}")


if __name__ == "__main__":
    main()
