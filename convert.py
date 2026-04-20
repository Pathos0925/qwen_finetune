import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Roman1111111/claude-opus-4.6-10000x",
    filename="opus46_final.jsonl",
    repo_type="dataset",
)

ROLE_MAP = {"user": "human", "assistant": "gpt"}
out = []
dropped = {"null_or_empty": 0, "bad_structure": 0, "image_token_collision": 0}
with open(path) as f:
    for i, line in enumerate(f):
        row = json.loads(line)

        if any(m["content"] is None or not str(m["content"]).strip() for m in row["messages"]):
            dropped["null_or_empty"] += 1
            continue

        if any("<image>" in m["content"] or "<video>" in m["content"] for m in row["messages"]):
            dropped["image_token_collision"] += 1
            continue

        convs = []
        for m in row["messages"]:
            if m["role"] == "system":
                continue
            convs.append({"from": ROLE_MAP[m["role"]], "value": m["content"]})
        if not convs or convs[0]["from"] != "human" or len(convs) % 2 != 0:
            dropped["bad_structure"] += 1
            continue
        out.append({"id": f"opus46_{i:06d}", "conversations": convs})

with open("train.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(out)} examples to train.json (dropped: {dropped})")