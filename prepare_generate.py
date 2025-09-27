import json

from tqdm import tqdm


path = "./new/enwiki"
SYSTEM_PROMPT = """You are given a short Wikipedia abstract delimited by <ref></ref>.
1. Devise ONE question that someone could answer **only if** they've read that abstract.
2. Provide the **correct answer** in one sentence.
Return JSON strictly in this form:
{
  "question": "……",
  "answer":   "……"
}"""

INPUT_TEMPLATE = "<ref>{abstract}</ref>"


def from_2024(ex):
    try:
        return int(ex["date_created"][:4]) == 2024 and "abstract" in ex and "name" in ex
    except:
        return False


buffer = []
pbar = tqdm(total=20000)
for i in range(38):
    filename = path + "/" + "enwiki_namespace_0_" + f"{i}" + ".jsonl"
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            try:
                data = json.loads(line)
                if from_2024(data):
                    buffer.append(data)
                else:
                    continue
            except:
                continue
            pbar.update(1)
            if len(buffer) >= 20000:
                break
    if len(buffer) >= 20000:
        break

abstract = []
for i, data in enumerate(buffer):
    abstract.append(
        {
            "name": data["name"],
            "index": f"{i}",
            "instruction": SYSTEM_PROMPT,
            "input": INPUT_TEMPLATE.format(abstract=data["abstract"]),
        }
    )

with open(path + "/" + "input.jsonl", "w", encoding="utf-8") as f:
    for obj in abstract:
        line = json.dumps(obj, ensure_ascii=False)
        f.write(line + "\n")
