import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import build_baseline_messages, build_train_messages, load_jsonl
from scoring import canonicalize, summarize

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
END_OF_TURN = "<|eot_id|>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="LoRA adapter path (omit to test the base model)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--name", required=True, help="name for the results files")
    args = p.parse_args()

    labels = json.loads(Path("data/processed/labels.json").read_text(encoding="utf-8"))
    val = load_jsonl("data/processed/val.jsonl")
    val = random.Random(0).sample(val, args.n)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"  # generation needs left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = [tokenizer.convert_tokens_to_ids(END_OF_TURN), tokenizer.eos_token_id]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": 0}
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def build_prompt(ex):
        if args.adapter:
            messages = build_train_messages(ex["text"], "")[:-1]  # short prompt
        else:
            messages = build_baseline_messages(ex["text"], labels)  # 77-label prompt
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    results = []
    prompt_tokens = []
    start = time.time()
    for i in range(0, len(val), args.batch_size):
        batch = val[i : i + args.batch_size]
        prompts = [build_prompt(ex) for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=24,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        replies = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for ex, reply, mask in zip(batch, replies, enc["attention_mask"]):
            prompt_tokens.append(int(mask.sum()))
            results.append({
                "text": ex["text"],
                "label": ex["label"],
                "reply": reply,
                "pred": canonicalize(reply, labels),
            })
        print(f"{len(results)}/{len(val)} done")
    elapsed = time.time() - start

    summary = summarize(results, labels)
    summary["name"] = args.name
    summary["batch_size"] = args.batch_size
    summary["avg_prompt_tokens"] = round(sum(prompt_tokens) / len(prompt_tokens), 1)
    summary["seconds_per_example"] = round(elapsed / len(results), 3)

    Path("results").mkdir(exist_ok=True)
    with open(f"results/eval_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(f"results/eval_{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()