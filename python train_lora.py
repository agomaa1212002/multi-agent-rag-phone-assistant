import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # fits 16GB well
DATA_PATH = "C:/LLM_PROJECTR/data/train.jsonl"
OUT_DIR = "C:/LLM_PROJECTR/lora_out"

def format_chat(example):
    # example["messages"] is a list of dicts with role/content
    msgs = example["messages"]
    text = ""
    for m in msgs:
        role = m["role"]
        content = m["content"]
        text += f"<|{role}|>\n{content}\n"
    text += "<|assistant|>\n"
    return {"text": text}

def main():
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.map(format_chat, remove_columns=ds.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    cfg = SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        max_seq_length=768,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        peft_config=lora,
        args=cfg,
        dataset_text_field="text"
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Saved LoRA adapter to:", OUT_DIR)

if __name__ == "__main__":
    main()
