import json

from tqdm import tqdm


with open("./new/enwiki/merged.json", "r", encoding="utf-8") as f:
    old = json.load(f)
    with open("./new/enwiki/cleaned.jsonl", "w", encoding="utf-8") as f2:
        for data in tqdm(old):
            if "redirect" in data["input"].lower():
                continue
            elif "abstract" in data["output"].lower():
                continue
            else:
                try:
                    raw = data["output"]
                    left = raw.index("{")
                    right = raw.rindex("}") + 1
                    data["output"] = raw[left:right]
                    t = json.loads(data["output"])
                    t["question"]
                    t["answer"]
                except:
                    continue
                else:
                    newline = json.dumps(data, ensure_ascii=False)
                    f2.write(newline + "\n")

with open("./new/enwiki/cleaned.jsonl", "r", encoding="utf-8") as f:
    with open("./new/enwiki/ds.jsonl", "w", encoding="utf-8") as f2:
        for line in tqdm(f):
            if not line:
                continue
            data = json.loads(line)

            index = data["index"]
            name = data["name"]
            abstract = data["input"]
            pair = json.loads(data["output"])
            question = pair["question"]
            answer = pair["answer"]
            data["instruction"] = question
            new = {
                "index": index,
                "name": name,
                "ref": abstract,
                "input": question,
                "output": answer,
            }
            newline = json.dumps(new, ensure_ascii=False)
            f2.write(newline + "\n")
