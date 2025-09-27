import json
import re

import pandas as pd


path = ""
outpath = ""


def judge(pred1: str, pred2: str, gt: str) -> bool:
    """
    pred1 is the one out of output_text, need to be cleaned
    pred2 is the one out of output_ids, do not need to be cleaned
    """
    result2 = pred2 == gt
    if len(pred1) > 20:
        if pred1[0:2] not in ["A ", "B ", "C ", "D ", "A.", "B.", "C.", "D."]:
            return result2
        else:
            pred1 = pred1[0]
    pred1 = re.sub(r"I\s+|(?i:assistant)", "", pred1)
    pred1 = re.sub(r"[\r\n\t]+", "", pred1)
    pred1 = re.sub(r"[^ABCD]", "", pred1)
    if len(pred1) > 1:
        pred1 = pred1[0]
    result1 = pred1 == gt
    if result1 == result2:
        return result1
    else:
        return False


choice_mapping: dict[int, str] = {32: "A", 33: "B", 34: "C", 35: "D"}


rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line:
            continue

        data = json.loads(line)
        index = data["index"]
        gt = data["gt"].upper()
        prediction = data["prediction"]
        eu = data["eu"]
        choices = data["choices"]
        Pchoices = data["Pchoices"]
        top = data["top"]
        stats = data["stats"]

        seq_idx = next(
            (
                i
                for i, (output_id, _, _) in enumerate(top)
                if int(output_id) in choice_mapping
            ),
            0,  # use the first token by default
        )
        choice_idx = ord(gt) - ord("A")
        correct = judge(
            prediction,
            choice_mapping[
                (
                    int(top[seq_idx][0])
                    if int(top[seq_idx][0]) in choice_mapping
                    else max(enumerate(Pchoices[seq_idx]), key=lambda x: x[1])[0] + 32
                )
            ],
            gt,
        )
        eu = eu[seq_idx]
        logit = choices[seq_idx][choice_idx]
        logit_top = top[seq_idx][1]
        prob = Pchoices[seq_idx][choice_idx]
        prob_top = top[seq_idx][2]
        mean, std, median, q25, q75 = stats[seq_idx]

        rows.append(
            {
                "index": index,
                "correct": correct,
                "eu": eu,
                "logit": logit,
                "prob": prob,
                "logit_top": logit_top,
                "prob_top": prob_top,
                "mean": mean,
                "std": std,
                "median": median,
                "q25": q25,
                "q75": q75,
            }
        )
df = pd.DataFrame(rows)
df.to_csv(outpath, index=False, encoding="utf-8")
print(df.head())
