import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "C:/LLM_PROJECTR/lora_out"

def generate(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()

def main():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype="auto")

    tuned = PeftModel.from_pretrained(base, ADAPTER)

    messages = [
        {"role": "system", "content": "You are a phone product assistant."},
        {"role": "user", "content": """Phone Data:
Brand: Honor
Model: 400 Pro
OS: Android v15
Chipset: Snapdragon 8 Gen 3
RAM: 12GB
Storage: 512GB
Battery: Li-Ion 5300mAh
Camera: 200+12MP
Refresh Rate: 120 Hz

Question: Compare Honor 400 vs Honor 400 Pro"""}
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("\n=== BASE ===\n")
    print(generate(base, tok, prompt))

    print("\n=== TUNED (LoRA) ===\n")
    print(generate(tuned, tok, prompt))

if __name__ == "__main__":
    main()
