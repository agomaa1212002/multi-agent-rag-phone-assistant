import json
import csv
import re
from pathlib import Path

from multi_agent_llm_planner import MultiAgentAssistant
from tools import load_phones


# ----------------------------
# Paths
# ----------------------------
BASE = Path("C:/LLM_PROJECTR")
QUESTIONS = BASE / "data" / "eval_questions.jsonl"
OUT = BASE / "data" / "eval_scores.csv"


# ----------------------------
# Load system
# ----------------------------
bot = MultiAgentAssistant()
phones = load_phones()


# ----------------------------
# Utilities
# ----------------------------
def normalize(x):
    return x.lower().strip() if isinstance(x, str) else ""


def should_refuse(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "price", "$", "budget", "cheap", "cheapest",
        "under", "affordable", "cost", "value"
    ])


def did_refuse(answer: str) -> bool:
    return "i don't have that information" in answer.lower()


def contains_hallucination(answer: str) -> bool:
    a = answer.lower()
    return any(k in a for k in [
        "$", "usd", "price", "cheap", "expensive"
    ])


def find_phone(question: str):
    q = question.lower()
    for p in phones:
        name = f"{p.get('brand','')} {p.get('model','')}".strip().lower()
        if name and name in q:
            return p
    return None


def fact_match_score(answer: str, phone: dict) -> float:
    """
    Returns fraction of matched facts.
    """
    if not phone:
        return 0.0

    facts = [
        phone.get("operating_system"),
        phone.get("chipset"),
        phone.get("ram"),
        phone.get("internal_storage"),
        phone.get("battery_capacity"),
        phone.get("primary_camera_resolution"),
        phone.get("refresh_rate"),
    ]

    facts = [f for f in facts if f]
    if not facts:
        return 0.0

    matches = sum(1 for f in facts if f.lower() in answer.lower())
    return matches / len(facts)


def score_answer(question: str, answer: str):
    # --- Price / budget safety ---
    if should_refuse(question):
        if did_refuse(answer):
            return 1.0, "Correct"
        return 0.0, "Wrong"

    # --- Hallucination ---
    if contains_hallucination(answer):
        return 0.0, "Wrong"

    phone = find_phone(question)

    # --- Fact-based scoring ---
    if phone:
        frac = fact_match_score(answer, phone)

        if frac >= 0.7:
            return 1.0, "Correct"
        elif frac >= 0.3:
            return 0.5, "Partially Correct"
        else:
            return 0.0, "Wrong"

    # --- Filters / rankings ---
    if did_refuse(answer):
        return 0.0, "Wrong"

    if len(answer.strip()) > 30:
        return 0.5, "Partially Correct"

    return 0.0, "Wrong"


# ----------------------------
# Evaluation Loop
# ----------------------------
results = []

with open(QUESTIONS, "r", encoding="utf-8") as f:
    questions = [json.loads(l)["question"] for l in f]

for i, q in enumerate(questions, 1):
    print(f"Scoring {i}/{len(questions)}")

    answer = bot.chat(q)
    score, label = score_answer(q, answer)

    results.append({
        "id": i,
        "question": q,
        "answer": answer,
        "score": score,
        "label": label
    })


# ----------------------------
# Write CSV
# ----------------------------
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "question", "answer", "score", "label"]
    )
    writer.writeheader()
    writer.writerows(results)


# ----------------------------
# Metrics
# ----------------------------
total = len(results)
avg_score = sum(r["score"] for r in results) / total
correct = sum(1 for r in results if r["score"] == 1.0)
partial = sum(1 for r in results if r["score"] == 0.5)
wrong = sum(1 for r in results if r["score"] == 0.0)

print("\n=== AUTOMATIC EVALUATION ===")
print(f"Total questions: {total}")
print(f"Correct: {correct}")
print(f"Partial: {partial}")
print(f"Wrong: {wrong}")
print(f"Average score: {avg_score:.3f}")
print(f"Accuracy (strict): {correct/total:.2%}")
print(f"\nCSV saved to: {OUT}")
