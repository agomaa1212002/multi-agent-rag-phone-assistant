import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = r"C:\LLM_PROJECTR\lora_out"

SYSTEM_PROMPT = (
    "You are a phone product assistant. "
    "Answer ONLY using the provided phone data. "
  
    "Be concise, factual, and user-friendly."
)


def generate(model, tok, messages, temperature=0.7, top_p=0.9):
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()


def main():
    print("Loading model... (this may take a moment)")

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        device_map="auto",
        torch_dtype="auto"
    )

    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    print("✅ LoRA Qwen model loaded")
    print("Type your prompt. Type 'exit' to quit.\n")

    while True:
        user_input = input("USER > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        answer = generate(model, tok, messages)
        print("\nASSISTANT >", answer, "\n")


if __name__ == "__main__":
    main()
