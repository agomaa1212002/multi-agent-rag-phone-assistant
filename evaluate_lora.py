import json
import csv
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = r"C:\LLM_PROJECTR\lora_out"

EVAL_PATH = Path(r"C:\LLM_PROJECTR\data\eval_questions.jsonl")
OUT_CSV = Path(r"C:\LLM_PROJECTR\data\eval_results_lora.csv")

SYSTEM = (
    "You are a phone product assistant. "
    "Answer ONLY using the provided phone data. "
    "If information is missing, say \"I don't have that information.\" "
    "Be concise, factual, and user-friendly."
)

def build_prompt(tok, user_query: str) -> str:
    # This evaluation is for style/safety behavior. We don't inject full phone data here.
    # If you want RAG-context eval, we can add retrieved context later.
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_query},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate(model, tok, prompt: str, max_new_tokens=220) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()

def is_price_query(q: str) -> bool:
    ql = q.lower()
    return ("price" in ql) or ("under $" in ql) or ("under $".replace(" ", "") in ql) or ("under" in ql and "$" in ql) or ("budget" in ql)

def is_refusal(ans: str) -> bool:
    return "i don't have that information" in ans.lower()

def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing eval file: {EVAL_PATH}")

    print("Loading tokenizer + models...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype="auto"
    )

    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)

    # Read eval questions
    rows = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} evaluation questions")

    results = []
    for item in rows:
        qid = item.get("id", "")
        q = item["query"]

        prompt = build_prompt(tok, q)

        base_ans = generate(base, tok, prompt)
        tuned_ans = generate(tuned, tok, prompt)

        # simple scoring: refusal correctness on price queries
        price = is_price_query(q)
        base_ref_ok = (not price) or is_refusal(base_ans)
        tuned_ref_ok = (not price) or is_refusal(tuned_ans)

        results.append({
            "id": qid,
            "query": q,
            "is_price_query": price,
            "base_answer": base_ans,
            "tuned_answer": tuned_ans,
            "base_refusal_ok": base_ref_ok,
            "tuned_refusal_ok": tuned_ref_ok,
        })

        print(f"Done {qid}: {q[:50]}...")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Summary stats
    price_qs = [r for r in results if r["is_price_query"]]
    if price_qs:
        base_ok = sum(r["base_refusal_ok"] for r in price_qs)
        tuned_ok = sum(r["tuned_refusal_ok"] for r in price_qs)
        print("\nRefusal correctness on PRICE queries:")
        print(f"  BASE : {base_ok}/{len(price_qs)}")
        print(f"  TUNED: {tuned_ok}/{len(price_qs)}")

    print(f"\nSaved CSV to: {OUT_CSV}")

if __name__ == "__main__":
    main()
