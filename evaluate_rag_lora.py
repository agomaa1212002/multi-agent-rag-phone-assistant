import json
import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ✅ YOUR retriever (must exist in your project)
from retrieve import retrieve

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = r"C:\LLM_PROJECTR\lora_out"

EVAL_PATH = Path(r"C:\LLM_PROJECTR\data\eval_questions.jsonl")
OUT_CSV = Path(r"C:\LLM_PROJECTR\data\eval_results_rag.csv")

SYSTEM = (
    "You are a phone product assistant. "
    "Answer ONLY using the provided phone data. "
    "If information is missing, say \"I don't have that information.\" "
    "Be concise, factual, and user-friendly."
)

def is_price_query(q: str) -> bool:
    ql = q.lower()
    return ("price" in ql) or ("under $" in ql) or ("$" in ql) or ("budget" in ql)

def is_refusal(ans: str) -> bool:
    return "i don't have that information" in ans.lower()

def build_prompt(tok, user_query: str, retrieved_docs: list[str]) -> str:
    context = "\n".join(retrieved_docs).strip()
    user_msg = f"""Phone Data:
{context}

User Question:
{user_query}
"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
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

def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing eval file: {EVAL_PATH}")

    print("Loading tokenizer + models...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype="auto"
    )

    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)

    # Load eval questions
    items = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"Loaded {len(items)} evaluation questions")

    results = []
    for i, item in enumerate(items, start=1):
        qid = item.get("id", f"q{i}")
        query = item["query"]

        # ✅ RAG retrieve (same context for base & tuned)
        retrieved = retrieve(query, top_k=3)  # adjust top_k if you want
        if not retrieved:
            retrieved = []

        prompt = build_prompt(tok, query, retrieved)

        base_ans = generate(base, tok, prompt)
        tuned_ans = generate(tuned, tok, prompt)

        # Simple scoring: refusal correctness on price queries
        price = is_price_query(query)
        base_ref_ok = (not price) or is_refusal(base_ans)
        tuned_ref_ok = (not price) or is_refusal(tuned_ans)

        # Simple hallucination flags: price mentioned when dataset lacks price
        base_price_leak = ("$" in base_ans) or ("price" in base_ans.lower())
        tuned_price_leak = ("$" in tuned_ans) or ("price" in tuned_ans.lower())

        results.append({
            "id": qid,
            "query": query,
            "top_k": 3,
            "retrieved_1": retrieved[0] if len(retrieved) > 0 else "",
            "retrieved_2": retrieved[1] if len(retrieved) > 1 else "",
            "retrieved_3": retrieved[2] if len(retrieved) > 2 else "",
            "is_price_query": price,
            "base_answer": base_ans,
            "tuned_answer": tuned_ans,
            "base_refusal_ok": base_ref_ok,
            "tuned_refusal_ok": tuned_ref_ok,
            "base_price_leak": base_price_leak,
            "tuned_price_leak": tuned_price_leak,
        })

        print(f"[{i}/{len(items)}] Done: {qid}")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    price_qs = [r for r in results if r["is_price_query"]]
    if price_qs:
        base_ok = sum(r["base_refusal_ok"] for r in price_qs)
        tuned_ok = sum(r["tuned_refusal_ok"] for r in price_qs)
        print("\nRefusal correctness on PRICE queries:")
        print(f"  BASE : {base_ok}/{len(price_qs)}")
        print(f"  TUNED: {tuned_ok}/{len(price_qs)}")

    base_leaks = sum(r["base_price_leak"] for r in results)
    tuned_leaks = sum(r["tuned_price_leak"] for r in results)
    print("\nPrice leak flags (mentions price/$ in answer):")
    print(f"  BASE : {base_leaks}/{len(results)}")
    print(f"  TUNED: {tuned_leaks}/{len(results)}")

    print(f"\nSaved CSV to: {OUT_CSV}")

if __name__ == "__main__":
    main()
