import json, random, re, os
from pathlib import Path

SYSTEM = (
    "You are a phone product assistant. "
    "Answer ONLY using the provided phone data. "
    "If information is missing, say \"I don't have that information.\" "
    "Be concise, factual, and user-friendly."
)

def phone_to_context(p: dict) -> str:
    # Keep only useful fields to reduce prompt length
    fields = [
        ("brand", "Brand"),
        ("model", "Model"),
        ("operating_system", "OS"),
        ("chipset", "Chipset"),
        ("ram", "RAM"),
        ("internal_storage", "Storage"),
        ("battery_capacity", "Battery"),
        ("primary_camera_resolution", "Camera"),
        ("refresh_rate", "Refresh Rate"),
    ]
    parts = []
    for k, label in fields:
        v = p.get(k)
        if v:
            parts.append(f"{label}: {v}")
    return "\n".join(parts)

def parse_mah(battery_str: str):
    if not battery_str:
        return None
    m = re.search(r"(\d{3,5})\s*mAh", battery_str, re.IGNORECASE)
    return int(m.group(1)) if m else None

def main():
    base = Path("C:/LLM_PROJECTR")
    phones_path = base / "phones_data_20250729_022344.json"
    out_dir = base / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "train.jsonl"

    with open(phones_path, "r", encoding="utf-8") as f:
        phones = json.load(f)

    random.seed(7)
    examples = []

    # 1) Info examples
    for p in random.sample(phones, min(250, len(phones))):
        ctx = phone_to_context(p)
        name = f"{p.get('brand','')} {p.get('model','')}".strip()
        if not name:
            continue
        user = f"Phone Data:\n{ctx}\n\nQuestion: Give me info about {name}"
        assistant = (
            f"{name} details: OS {p.get('operating_system','')}, chipset {p.get('chipset','')}, "
            f"RAM {p.get('ram','')}, storage {p.get('internal_storage','')}, "
            f"battery {p.get('battery_capacity','')}, camera {p.get('primary_camera_resolution','')}, "
            f"refresh rate {p.get('refresh_rate','')}."
        ).replace(" ,", ",").strip()
        examples.append({"messages":[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user},
            {"role":"assistant","content":assistant}
        ]})

    # 2) Safety/refusal examples (price)
    for _ in range(80):
        p = random.choice(phones)
        ctx = phone_to_context(p)
        name = f"{p.get('brand','')} {p.get('model','')}".strip()
        user = f"Phone Data:\n{ctx}\n\nQuestion: What is the price of {name} under $500?"
        assistant = "I don't have that information."
        examples.append({"messages":[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user},
            {"role":"assistant","content":assistant}
        ]})

    # 3) Compare examples (field-by-field)
    for _ in range(200):
        a = random.choice(phones)
        b = random.choice(phones)
        name_a = f"{a.get('brand','')} {a.get('model','')}".strip()
        name_b = f"{b.get('brand','')} {b.get('model','')}".strip()
        if not name_a or not name_b or name_a == name_b:
            continue

        ctx = f"PHONE A:\n{phone_to_context(a)}\n\nPHONE B:\n{phone_to_context(b)}"
        user = f"Phone Data:\n{ctx}\n\nQuestion: Compare {name_a} vs {name_b}"
        assistant = (
            f"- OS: A={a.get('operating_system')} | B={b.get('operating_system')}\n"
            f"- Chipset: A={a.get('chipset')} | B={b.get('chipset')}\n"
            f"- RAM: A={a.get('ram')} | B={b.get('ram')}\n"
            f"- Storage: A={a.get('internal_storage')} | B={b.get('internal_storage')}\n"
            f"- Battery: A={a.get('battery_capacity')} | B={b.get('battery_capacity')}\n"
            f"- Camera: A={a.get('primary_camera_resolution')} | B={b.get('primary_camera_resolution')}\n"
            f"- Refresh rate: A={a.get('refresh_rate')} | B={b.get('refresh_rate')}"
        )
        examples.append({"messages":[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user},
            {"role":"assistant","content":assistant}
        ]})

    random.shuffle(examples)

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")

if __name__ == "__main__":
    main()
