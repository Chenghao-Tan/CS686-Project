import json
import random

from tqdm import tqdm


path = "./new/enwiki/ds-shuffledchoices-eval.jsonl"
nsample = int(input("input n: "))


ds = []
with open(path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        if not line:
            continue
        else:
            ds.append(json.loads(line))
random.shuffle(ds)
samples = random.sample(ds, k=nsample)
for data in tqdm(samples):
    print(json.dumps(data, ensure_ascii=False, indent=4))
    input()
