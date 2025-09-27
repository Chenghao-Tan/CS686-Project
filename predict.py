import json

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
path = "./new/enwiki/ds-shuffledchoices-eval.jsonl"
outpath = "./new/pred_ref_notrain.jsonl"
modelpath = "./Meta-Llama-3.1-8B-Instruct"
REF = True
OPT = True


torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    modelpath, trust_remote_code=True, torch_dtype=torch.bfloat16
)
model.eval()


SYSTEM_PROMPT_REF = """
You are given:
• A short reference passage, wrapped by <ref>...</ref>.
• A question to test your knowledge.
• Four answer choices labeled A/B/C/D.

Task:
Decide which option best answers the question.
**Output only the single capital letter (A, B, C, or D). Do not output anything else.**
"""
SYSTEM_PROMPT_NOREF = """
You are given:
• A question to test your knowledge.
• Four answer choices labeled A/B/C/D.

Task:
Decide which option best answers the question.
**Output only the single capital letter (A, B, C, or D). Do not output anything else.**
"""

choices_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
eos_id = tokenizer.eos_token_id

max_new_tokens = 1
prompts: list[dict[str, str]] = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line:
            continue
        data = json.loads(line)

        messages = []
        gt = ""
        if OPT:
            max_new_tokens = 8
            messages.append(
                {"role": "system", "content": SYSTEM_PROMPT_REF}
                if REF
                else {"role": "system", "content": SYSTEM_PROMPT_NOREF}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        ((data["ref"] + "\n") if REF else "")
                        + (data["input"] + "\n")
                        + ("\n".join(data["options"]) + "\n")
                    ),
                }
            )
            gt = data["output"]
        else:
            max_new_tokens = 256
            messages.append(
                {
                    "role": "user",
                    "content": (
                        ((data["ref"] + "\n") if REF else "") + (data["input"] + "\n")
                    ),
                }
            )
            gt = data["answer"]
        prompts.append(
            {
                "index": data["index"],
                "gt": gt,
                "input_text": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            }
        )


class PromptsDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


ds = PromptsDataset(prompts)
dataloader = DataLoader(
    ds,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)


def compute_eu(logits: torch.Tensor, k: int = 64):
    # logits: batch,length,vocab
    with torch.no_grad():
        topk_logits, _ = torch.topk(logits, k, dim=-1)
        alpha = torch.relu(topk_logits)
        sum_alpha = alpha.sum(dim=-1)
        return k / (k + sum_alpha)


accelerator = Accelerator()
model, dataloader = accelerator.prepare(model, dataloader)
torch.set_float32_matmul_precision("high")  # For A100

results = {
    "index": [],
    "gt": [],
    "prediction": [],
    "eu": [],
    "choices": [],
    "Pchoices": [],
    "top": [],
    "stats": [],
}
for x in tqdm(dataloader):
    with torch.no_grad():
        index = x["index"]
        gt = x["gt"]
        input_text = x["input_text"]
        accelerator.print()

        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(
            model.device
        )
        outputs = accelerator.unwrap_model(model).generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            top_p=1.0,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
        )
        output_ids = outputs.sequences[:, inputs.input_ids.shape[-1] :]
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output_lens = []
        for seq in output_ids.tolist():
            if eos_id in seq:
                true_len = seq.index(eos_id) + 1
            else:
                true_len = len(seq)
            output_lens.append(true_len)
        logits = torch.stack(outputs.logits, dim=1)
        eu = compute_eu(logits, k=64).tolist()
        choices = torch.stack([logits[:, :, x] for x in choices_ids], dim=-1).tolist()
        probs = torch.softmax(torch.stack(outputs.scores, dim=1), dim=-1)
        Pchoices = torch.stack([probs[:, :, x] for x in choices_ids], dim=-1).tolist()
        top_v, top_i = logits.max(dim=-1)
        top_p, _ = probs.max(dim=-1)
        top = torch.stack([top_i, top_v, top_p], dim=-1).tolist()
        stats = torch.stack(
            [
                logits.mean(dim=-1),
                logits.std(correction=0, dim=-1),
                logits.median(dim=-1)[0],
                logits.quantile(0.25, dim=-1),
                logits.quantile(0.75, dim=-1),
            ],
            dim=-1,
        ).tolist()
        for i, n in enumerate(output_lens):
            eu[i] = eu[i][:n]
            choices[i] = choices[i][:n]
            Pchoices[i] = Pchoices[i][:n]
            top[i] = top[i][:n]
            stats[i] = stats[i][:n]

        results["index"].extend(accelerator.gather_for_metrics(index))
        results["gt"].extend(accelerator.gather_for_metrics(gt))
        results["prediction"].extend(accelerator.gather_for_metrics(output_text))
        results["eu"].extend(accelerator.gather_for_metrics(eu))
        results["choices"].extend(accelerator.gather_for_metrics(choices))
        results["Pchoices"].extend(accelerator.gather_for_metrics(Pchoices))
        results["top"].extend(accelerator.gather_for_metrics(top))
        results["stats"].extend(accelerator.gather_for_metrics(stats))

if accelerator.is_main_process:
    with open(outpath, "w", encoding="utf-8") as f:
        n = len(results["index"])
        for i in tqdm(range(n)):
            data = {
                "index": results["index"][i],
                "gt": results["gt"][i],
                "prediction": results["prediction"][i],
                "eu": results["eu"][i],
                "choices": results["choices"][i],
                "Pchoices": results["Pchoices"][i],
                "top": results["top"][i],
                "stats": results["stats"][i],
            }
            newline = json.dumps(data, ensure_ascii=False)
            f.write(newline + "\n")
