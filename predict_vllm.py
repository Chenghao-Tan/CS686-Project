import json
import pickle

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

SEED = 42
path = "./new/enwiki/ds-shuffledchoices.jsonl"
outpath = "./new/pred.pkl"
model = "./Meta-Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = """
You are given:
• A short reference passage, wrapped by <ref>...</ref>.
• A question to test your knowledge.
• Four answer choices labeled A/B/C/D.

Task:
Decide which option best answers the question.
**Output only the single capital letter (A, B, C, or D). Do not output anything else.**
"""


prompts: list[dict[str, str]] = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line:
            continue
        data = json.loads(line)
        # prompts.append({"index": data["index"], "prompt": data["ref"] + data["input"]})
        prompts.append(
            {
                "index": data["index"],
                "prompt": data["ref"]
                + "\n"
                + data["input"]
                + "\n"
                + "\n".join(data["options"])
                + "\n",
            }
        )


llm = LLM(model=model, gpu_memory_utilization=0.9)


class LogitsSpy:
    def __init__(self):
        self.logits = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        self.logits.append(logits.tolist())
        return logits


# Greedy Decoding
for x in tqdm(prompts):
    logits_spy = LogitsSpy()
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        seed=SEED,
        stop_token_ids=[128001, 128008, 128009],
        max_tokens=256,
        logits_processors=[logits_spy],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": x["prompt"]},
    ]
    outs = llm.chat(messages, sampling, use_tqdm=False)  # type:ignore

    for i, out in enumerate(outs):
        pred = out.outputs[0]
        logits = logits_spy.logits
        prompts[i]["prediction"] = pred.text.strip()
        prompts[i]["logits"] = logits  # type:ignore


"""
with open(outpath, "w", encoding="utf-8") as f:
    for data in prompts:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
"""
with open(outpath, "wb") as f:
    pickle.dump(prompts, f, protocol=pickle.HIGHEST_PROTOCOL)
