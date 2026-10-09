import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import build_train_messages, load_jsonl

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
END_OF_TURN = "<|eot_id|>"


def encode_example(ex, tokenizer, max_len=256):
    """Turn one example into token ids, with loss only on the label tokens."""
    messages = build_train_messages(ex["text"], ex["label"])
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(ex["label"] + END_OF_TURN, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    """Pads every example in a batch to the same length."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), longest), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_train", type=int, default=0, help="limit training examples (0 = all)")
    p.add_argument("--max_steps", type=int, default=-1, help="stop after this many steps (-1 = use epochs)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--output_dir", default="adapters/run1")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    assert tokenizer.convert_tokens_to_ids(END_OF_TURN) is not None, "end-of-turn token missing"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    train = load_jsonl("data/processed/train.jsonl")
    val = load_jsonl("data/processed/val.jsonl")
    rng = random.Random(0)
    rng.shuffle(train)
    if args.max_train > 0:
        train = train[: args.max_train]
    val = rng.sample(val, 200)

    train_ds = [encode_example(ex, tokenizer) for ex in train]
    val_ds = [encode_example(ex, tokenizer) for ex in val]
    print(f"train examples: {len(train_ds)}  val examples: {len(val_ds)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(pad_id),
    )
    trainer.train()
    final_eval = trainer.evaluate()
    print("final validation loss:", final_eval["eval_loss"])

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    Path("results").mkdir(exist_ok=True)
    log_path = Path("results") / f"train_log_{Path(args.output_dir).name}.json"
    log_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"adapter saved to {args.output_dir}, log saved to {log_path}")


if __name__ == "__main__":
    main()