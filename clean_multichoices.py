import json

from tqdm import tqdm


bads = []
with open("./new/enwiki/merged-multichoices.json", "r", encoding="utf-8") as f:
    old = json.load(f)
    with open("./new/enwiki/cleaned-multichoices.jsonl", "w", encoding="utf-8") as f2:
        for data in tqdm(old):
            if False:
                continue
            else:
                try:
                    raw = data["output"]
                    left = raw.index("{")
                    right = raw.rindex("}") + 1
                    data["output"] = raw[left:right]
                    t = json.loads(data["output"])
                    if not len(t["options"]) == 4:
                        raise
                    for option in t["options"]:
                        if option.strip().lower()[0] not in ["a", "b", "c", "d"]:
                            raise
                    if t["correct_choice"].strip().lower() not in ["a", "b", "c", "d"]:
                        raise
                except:
                    bads.append(data["index"])
                    continue
                else:
                    newline = json.dumps(data, ensure_ascii=False)
                    f2.write(newline + "\n")
print(bads)

abstract = {}
with open("./new/enwiki/ds.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f):
        if not line:
            continue
        data = json.loads(line)
        abstract[data["index"]] = data["ref"]
with open("./new/enwiki/cleaned-multichoices.jsonl", "r", encoding="utf-8") as f:
    with open("./new/enwiki/ds-multichoices.jsonl", "w", encoding="utf-8") as f2:
        for line in tqdm(f):
            if not line:
                continue
            data = json.loads(line)

            index = data["index"]
            name = data["name"]
            pair = json.loads(data["input"])
            question = pair["question"]
            answer = pair["answer"]
            multichoices = json.loads(data["output"])
            options = multichoices["options"]
            correct_choice = multichoices["correct_choice"]
            new = {
                "index": index,
                "name": name,
                "ref": abstract[index],
                "input": question,
                "options": options,
                "output": correct_choice,
                "answer": answer,
            }
            newline = json.dumps(new, ensure_ascii=False)
            f2.write(newline + "\n")
