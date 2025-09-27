import json


SYSTEM_PROMPT = """
You are an assessment-item writer.

You will receive a single-object JSON containing:
{
  "question": "<original question text>",
  "answer":   "<the single correct answer>"
}

Augment the answer into a four-option single-correct item and return **only** a new JSON object with **exactly** these keys:

{
  "options":  ["A. <choice 1>", "B. <choice 2>", "C. <choice 3>", "D. <choice 4>"],
  "correct_choice": <the sole letter of the correct choice in the array (A/B/C/D)>
}

Rules for constructing the `options` array
1. Include the given correct answer once; generate **three unique, plausible distractors** that:
   • match the type/length/tone of the correct answer
   • are factually incorrect for the question
   • are not giveaways like “None of the above”
2. Randomly shuffle the four choices so the correct answer's position varies.
3. Keep each option concise (< 30 characters).
4. Do **not** repeat the correct answer, and do not output explanations.

Output must be valid JSON (no markdown fences, headings, or extra keys).
Return nothing except that JSON object.
"""


with open("./new/enwiki/cleaned.jsonl", "r", encoding="utf-8") as f:
    with open("./new/enwiki/input-multichoices.jsonl", "w", encoding="utf-8") as f2:
        for line in f:
            if not line:
                continue
            data = json.loads(line)

            new = {
                "name": data["name"],
                "index": data["index"],
                "instruction": SYSTEM_PROMPT,
                "input": data["output"],
            }

            newline = json.dumps(new, ensure_ascii=False)
            f2.write(newline + "\n")
