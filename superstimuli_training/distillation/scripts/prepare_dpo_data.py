#!/usr/bin/env python3
"""Build first-turn DPO pairs from baseline and soft-prompt trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MULTIMODAL_PLACEHOLDERS = ("<image>", "<video>", "<audio>")


def load_experiences(path: Path) -> list[dict[str, Any]]:
    """Load an experience list from a response payload."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("experiences"), list):
        raise ValueError(f"Expected an object with an experiences list: {path}")
    return payload["experiences"]


def first_content(record: dict[str, Any], role: str) -> str | None:
    """Return the first non-empty message for a role."""
    for message in record.get("messages", []):
        if message.get("role") == role and isinstance(message.get("content"), str):
            content = message["content"].strip()
            if content:
                return content
    return None


def escape_placeholders(text: str) -> str:
    """Prevent literal media tags from invoking multimodal processors."""
    for placeholder in MULTIMODAL_PLACEHOLDERS:
        escaped = placeholder.replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(placeholder, escaped)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--soft-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-prompt", default="You are a helpful AI assistant.")
    parser.add_argument("--max-response-tokens", type=int, default=4000)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.output, args.audit_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite to replace it.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    baseline_records = load_experiences(args.baseline)
    soft_records = load_experiences(args.soft_prompt)
    baseline_by_id = {str(record["final_id"]): record for record in baseline_records}
    soft_by_id = {str(record["final_id"]): record for record in soft_records}
    shared_ids = sorted(
        baseline_by_id.keys() & soft_by_id.keys(),
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )

    pairs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    response_token_sums: list[int] = []
    branch_token_sums: list[int] = []

    for final_id in shared_ids:
        baseline = baseline_by_id[final_id]
        soft = soft_by_id[final_id]
        baseline_user = first_content(baseline, "user")
        soft_user = first_content(soft, "user")
        rejected = first_content(baseline, "assistant")
        chosen = first_content(soft, "assistant")
        reason = None
        details: dict[str, Any] = {}

        if None in (baseline_user, soft_user, rejected, chosen):
            reason = "missing_first_turn_content"
        elif baseline_user != soft_user:
            reason = "user_context_mismatch"
        elif rejected == chosen:
            reason = "identical_responses"
        else:
            user = escape_placeholders(soft_user)
            chosen = escape_placeholders(chosen)
            rejected = escape_placeholders(rejected)
            chosen_tokens = len(tokenizer.encode(chosen, add_special_tokens=False))
            rejected_tokens = len(tokenizer.encode(rejected, add_special_tokens=False))
            details["chosen_tokens"] = chosen_tokens
            details["rejected_tokens"] = rejected_tokens
            if max(chosen_tokens, rejected_tokens) >= args.max_response_tokens:
                reason = f"response_tokens_gte_{args.max_response_tokens}"
            else:
                prompt = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user},
                ]
                branch_lengths = []
                for response in (chosen, rejected):
                    token_ids = tokenizer.apply_chat_template(
                        prompt + [{"role": "assistant", "content": response}],
                        tokenize=True,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    if hasattr(token_ids, "get"):
                        token_ids = token_ids["input_ids"]
                    if token_ids and isinstance(token_ids[0], list):
                        token_ids = token_ids[0]
                    branch_lengths.append(len(token_ids))
                details["chosen_branch_tokens"] = branch_lengths[0]
                details["rejected_branch_tokens"] = branch_lengths[1]
                if max(branch_lengths) >= args.cutoff_len:
                    reason = f"branch_tokens_gte_{args.cutoff_len}"
                else:
                    pairs.append(
                        {
                            "messages": prompt,
                            "chosen": {"role": "assistant", "content": chosen},
                            "rejected": {"role": "assistant", "content": rejected},
                            "final_id": baseline.get("final_id"),
                        }
                    )
                    response_token_sums.append(chosen_tokens + rejected_tokens)
                    branch_token_sums.append(sum(branch_lengths))

        if reason is not None:
            excluded.append({"final_id": baseline.get("final_id"), "reason": reason, **details})

    if not pairs:
        raise ValueError("No valid DPO pairs were produced.")

    audit = {
        "baseline": str(args.baseline),
        "soft_prompt": str(args.soft_prompt),
        "system_prompt": args.system_prompt,
        "matched_records": len(shared_ids),
        "included_pairs": len(pairs),
        "excluded_pairs": len(excluded),
        "mean_response_pair_tokens": sum(response_token_sums) / len(response_token_sums),
        "mean_rendered_branch_pair_tokens": sum(branch_token_sums) / len(branch_token_sums),
        "excluded": excluded,
    }
    for path, value in ((args.output, pairs), (args.audit_output, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    print(f"Wrote {len(pairs)} DPO pairs to {args.output}")
    print(f"Wrote audit with {len(excluded)} exclusions to {args.audit_output}")


if __name__ == "__main__":
    main()
