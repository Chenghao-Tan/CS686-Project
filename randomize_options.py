from copy import deepcopy
import json
import random
import string

from tqdm import tqdm


def shuffle_multichoices(item: dict) -> dict:
    options = {x[0].strip().upper(): x[1:] for x in item["options"]}
    correct_choice = item["output"].strip().upper()
    correct_answer = options[correct_choice]

    shuffled_options = list(options.values())
    random.shuffle(shuffled_options)

    A2Z = list(string.ascii_uppercase)
    new_options = []
    new_correct_choice = None
    for i, text in enumerate(shuffled_options):
        letter = A2Z[i]
        new_options.append(letter + text)

        if text == correct_answer:
            new_correct_choice = letter

    new = deepcopy(item)
    new["options"] = new_options
    new["output"] = new_correct_choice
    return new


with open("./new/enwiki/ds-multichoices.jsonl", "r", encoding="utf-8") as f:
    with open("./new/enwiki/ds-shuffledchoices.jsonl", "w", encoding="utf-8") as f2:
        for line in tqdm(f):
            if not line:
                continue
            item = json.loads(line)
            shuffled = shuffle_multichoices(item)
            newline = json.dumps(shuffled, ensure_ascii=False)
            f2.write(newline + "\n")
