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
QUESTIONS_PATH = BASE / "data" / "eval_questions.jsonl"
OUT_CSV = BASE / "data" / "eval_results.csv"


# ----------------------------
# Load system
# ----------------------------
bot = MultiAgentAssistant()
phones = load_phones()


# ----------------------------
# Helpers
# ----------------------------
def normalize(text: str) -> str:
    return text.lower().strip()


def extract_phone_from_question(q: str):
    ql = q.lower()
    for p in phones:
        name = f"{p.get('brand','')} {p.get('model','')}".strip()
        if name and name.lower() in ql:
            return p
    return None


def should_refuse(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "price", "$", "budget", "cheap", "cheapest",
        "under", "affordable", "cost", "value for money"
    ])


def answer_refused(answer: str) -> bool:
    return "i don't have that information" in answer.lower()


def factual_match(answer: str, phone: dict) -> int:
    """
    Count how many key facts appear in the answer.
    """
    if not phone:
        return 0

    fields = [
        phone.get("operating_system"),
        phone.get("chipset"),
        phone.get("ram"),
        phone.get("internal_storage"),
        phone.get("battery_capacity"),
        phone.get("primary_camera_resolution"),
        phone.get("refresh_rate"),
    ]

    score = 0
    ans = answer.lower()
    for f in fields:
        if f and f.lower() in ans:
            score += 1
    return score


def contains_hallucination(answer: str) -> bool:
    """
    Detect forbidden info (price etc).
    """
    ans = answer.lower()
    return any(k in ans for k in [
        "$", "usd", "price", "cost", "cheap", "expensive"
    ])


def label_answer(question: str, answer: str) -> str:
    # Case 1: price/budget → MUST refuse
    if should_refuse(question):
        return "Correct" if answer_refused(answer) else "Wrong"

    phone = extract_phone_from_question(question)

    # Case 2: hallucination
    if contains_hallucination(answer):
        return "Wrong"

    # Case 3: info-based question
    if phone:
        match_count = factual_match(answer, phone)

        if match_count >= 5:
            return "Correct"
        elif match_count >= 2:
            return "Partially Correct"
        else:
            return "Wrong"

    # Case 4: filtering / ranking questions
    # If system returns something non-empty → partial credit
    if answer_refused(answer):
        return "Wrong"

    if len(answer.strip()) > 20:
        return "Partially Correct"

    return "Wrong"


# ----------------------------
# Main evaluation loop
# ----------------------------
rows = []

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = [json.loads(line)["question"] for line in f]

for i, q in enumerate(questions, 1):
    print(f"Evaluating {i}/{len(questions)}")

    answer = bot.chat(q)
    label = label_answer(q, answer)

    rows.append({
        "id": i,
        "question": q,
        "answer": answer,
        "label": label
    })


# ----------------------------
# Write CSV
# ----------------------------
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "question", "answer", "label"]
    )
    writer.writeheader()
    writer.writerows(rows)


# ----------------------------
# Summary
# ----------------------------
total = len(rows)
correct = sum(1 for r in rows if r["label"] == "Correct")
partial = sum(1 for r in rows if r["label"] == "Partially Correct")
wrong = sum(1 for r in rows if r["label"] == "Wrong")

print("\n=== Evaluation Summary ===")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Partially Correct: {partial}")
print(f"Wrong: {wrong}")
print(f"Accuracy: {correct / total:.2%}")

print(f"\nCSV written to: {OUT_CSV}")
