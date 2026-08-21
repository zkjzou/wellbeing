#!/usr/bin/env python3
"""Convert teacher conversations into LLaMA-Factory ShareGPT records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_ROLES = {"system", "user", "assistant"}
MULTIMODAL_PLACEHOLDERS = ("<image>", "<video>", "<audio>")


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list or a response payload containing ``experiences``."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("experiences"), list):
        records = payload["experiences"]
    else:
        raise ValueError("Input must be a JSON list or an object with an 'experiences' list.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every input record must be a JSON object.")
    return records


def normalize_messages(record: dict[str, Any], index: int) -> list[dict[str, str]]:
    """Validate and normalize one OpenAI-style conversation."""
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"Record {index} has no non-empty 'messages' list.")

    messages: list[dict[str, str]] = []
    for message_index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"Record {index}, message {message_index} is not an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"Record {index}, message {message_index} has unsupported role {role!r}."
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Record {index}, message {message_index} has empty content.")
        messages.append({"role": role, "content": content.strip()})

    body = messages[1:] if messages[0]["role"] == "system" else messages
    if not body or body[0]["role"] != "user":
        raise ValueError(f"Record {index} must begin with a user message after an optional system message.")
    for message_index, message in enumerate(body):
        expected_role = "user" if message_index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise ValueError(
                f"Record {index} has role {message['role']!r} at conversational position "
                f"{message_index}; expected {expected_role!r}."
            )
    if body[-1]["role"] != "assistant":
        raise ValueError(f"Record {index} must end with a teacher assistant response.")
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Teacher-response JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional prefix limit.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path used with --max-assistant-tokens.",
    )
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=None,
        help=(
            "Drop an entire conversation when any assistant turn has at least this "
            "many tokens. This is useful for excluding likely length-truncated outputs."
        ),
    )
    parser.add_argument(
        "--max-conversation-tokens",
        type=int,
        default=None,
        help=(
            "Drop a conversation when its rendered chat template has at least this "
            "many tokens. Requires --tokenizer."
        ),
    )
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=None,
        help="Optional JSON audit report for conversations excluded by token length.",
    )
    parser.add_argument(
        "--escape-multimodal-placeholders",
        action="store_true",
        help=(
            "Escape literal <image>, <video>, and <audio> strings so a text-only "
            "dataset is not interpreted as containing multimodal attachments."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive.")
    if args.max_assistant_tokens is not None and args.max_assistant_tokens <= 0:
        raise ValueError("--max-assistant-tokens must be positive.")
    if args.max_conversation_tokens is not None and args.max_conversation_tokens <= 0:
        raise ValueError("--max-conversation-tokens must be positive.")
    if (
        args.max_assistant_tokens is not None
        or args.max_conversation_tokens is not None
    ) and args.tokenizer is None:
        raise ValueError("--tokenizer is required with token-length filters.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it.")

    records = load_records(args.input)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    normalized = [
        (index, record, normalize_messages(record, index))
        for index, record in enumerate(records)
    ]
    if args.escape_multimodal_placeholders:
        for _, _, messages in normalized:
            for message in messages:
                for placeholder in MULTIMODAL_PLACEHOLDERS:
                    escaped = placeholder.replace("<", "&lt;").replace(">", "&gt;")
                    message["content"] = message["content"].replace(placeholder, escaped)
    excluded: list[dict[str, Any]] = []
    tokenizer = None
    if args.max_assistant_tokens is not None or args.max_conversation_tokens is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.max_assistant_tokens is not None:
        retained = []
        for source_index, record, messages in normalized:
            assistant_lengths = [
                len(tokenizer.encode(message["content"], add_special_tokens=False))
                for message in messages
                if message["role"] == "assistant"
            ]
            if any(length >= args.max_assistant_tokens for length in assistant_lengths):
                excluded.append(
                    {
                        "source_index": source_index,
                        "final_id": record.get("final_id"),
                        "assistant_token_lengths": assistant_lengths,
                        "reason": f"assistant_tokens_gte_{args.max_assistant_tokens}",
                    }
                )
            else:
                retained.append((source_index, record, messages))
        normalized = retained

    if args.max_conversation_tokens is not None:
        retained = []
        for source_index, record, messages in normalized:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            if hasattr(token_ids, "get"):
                token_ids = token_ids["input_ids"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            conversation_length = len(token_ids)
            if conversation_length >= args.max_conversation_tokens:
                excluded.append(
                    {
                        "source_index": source_index,
                        "final_id": record.get("final_id"),
                        "conversation_token_length": conversation_length,
                        "reason": f"conversation_tokens_gte_{args.max_conversation_tokens}",
                    }
                )
            else:
                retained.append((source_index, record, messages))
        normalized = retained

    converted = [{"messages": messages} for _, _, messages in normalized]
    if not converted:
        raise ValueError("No conversations were selected.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(converted)} conversations to {args.output}")
    if args.excluded_output is not None:
        args.excluded_output.parent.mkdir(parents=True, exist_ok=True)
        with args.excluded_output.open("w", encoding="utf-8") as handle:
            json.dump(excluded, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Wrote {len(excluded)} exclusions to {args.excluded_output}")


if __name__ == "__main__":
    main()
