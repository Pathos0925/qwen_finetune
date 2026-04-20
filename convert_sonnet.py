"""Download Roman1111111/claude-sonnet-4.6-120000x and convert to train.json.

Format identical to the existing convert.py: emits a JSON array of
{"id": ..., "conversations": [{"from": "human"/"gpt", "value": ...}, ...]}
entries, dropping rows that would confuse the tokenizer (null/empty
content, <image>/<video> token collisions, malformed turn structure).
System prompts are dropped — the existing trainer prepends its own.

The output file is large (~700MB+ JSON). Pass --output to change the path
or --limit N to subsample for quick experiments.
"""

from __future__ import annotations

import argparse
import json
from huggingface_hub import hf_hub_download

REPO_ID = "Roman1111111/claude-sonnet-4.6-120000x"
FILENAME = "sonnet4.6-general,code,math,psychology.jsonl"

ROLE_MAP = {"user": "human", "assistant": "gpt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="train.json", help="Output JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on rows (for smoke tests).")
    parser.add_argument("--difficulty", default=None,
                        help="Optional difficulty filter (beginner|medium|complex|hard|extreme). Comma-separated for multiple.")
    parser.add_argument("--category", default=None,
                        help="Optional category filter (e.g. mathematics,programming). Comma-separated.")
    args = parser.parse_args()

    allow_difficulty = set(args.difficulty.split(",")) if args.difficulty else None
    allow_category = set(args.category.split(",")) if args.category else None

    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    print(f"Downloaded {path}")

    out = []
    dropped = {
        "null_or_empty": 0,
        "bad_structure": 0,
        "image_token_collision": 0,
        "filtered_difficulty": 0,
        "filtered_category": 0,
    }
    with open(path) as f:
        for i, line in enumerate(f):
            if args.limit is not None and len(out) >= args.limit:
                break
            row = json.loads(line)

            if allow_difficulty is not None and str(row.get("difficulty")) not in allow_difficulty:
                dropped["filtered_difficulty"] += 1
                continue
            if allow_category is not None and str(row.get("category")) not in allow_category:
                dropped["filtered_category"] += 1
                continue

            messages = row.get("messages", [])
            if any(m.get("content") is None or not str(m["content"]).strip() for m in messages):
                dropped["null_or_empty"] += 1
                continue

            if any("<image>" in str(m["content"]) or "<video>" in str(m["content"]) for m in messages):
                dropped["image_token_collision"] += 1
                continue

            convs = []
            for m in messages:
                if m["role"] == "system":
                    continue
                convs.append({"from": ROLE_MAP[m["role"]], "value": m["content"]})

            if not convs or convs[0]["from"] != "human" or len(convs) % 2 != 0:
                dropped["bad_structure"] += 1
                continue

            out.append({"id": f"sonnet46_{i:06d}", "conversations": convs})

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"Wrote {len(out)} examples to {args.output} (dropped: {dropped})")


if __name__ == "__main__":
    main()
