import re
from retrieve import retrieve
from tools import load_phones, filter_phones, top_battery, compare_phones

SYSTEM_PROMPT = """
You are a phone product assistant.
Answer ONLY using the provided phone data.
If information is missing, say "I don't have that information."
Be concise and factual.
""".strip()

phones = load_phones()

def _extract_int_after_keywords(text: str, keywords):
    """
    Extract an integer that appears after any keyword.
    Example: "battery at least 5000" -> 5000
    """
    t = text.lower()
    for kw in keywords:
        if kw in t:
            # search for first number in the full query
            m = re.search(r"(\d{3,5})", t)
            if m:
                return int(m.group(1))
    return None

def _extract_gb(text: str, keywords):
    """
    Extract RAM/storage constraints like "8gb", "12 GB", "256gb"
    """
    t = text.lower()
    for kw in keywords:
        if kw in t:
            m = re.search(r"(\d+(?:\.\d+)?)\s*gb", t)
            if m:
                return float(m.group(1))
    return None

def _extract_brand(query: str):
    # very simple brand detection from dataset brands (safe)
    q = query.lower()
    brands = sorted({(p.get("brand") or "").strip() for p in phones if p.get("brand")}, key=len, reverse=True)
    for b in brands:
        if b and b.lower() in q:
            return b
    return None

def _format_phone_brief(p):
    return (
        f"{p.get('brand','')} {p.get('model','')}".strip()
        + f" | OS: {p.get('operating_system')}"
        + f" | RAM: {p.get('ram')}"
        + f" | Storage: {p.get('internal_storage')}"
        + f" | Battery: {p.get('battery_capacity')}"
        + f" | Camera: {p.get('primary_camera_resolution')}"
    )

def answer(query: str) -> str:
    q = query.strip()
    ql = q.lower()

    # ---------- TOOL: Compare ----------
    # Example: "compare Honor 400 vs Honor 400 Pro"
    if "compare" in ql or "vs" in ql:
        # naive split for names: use "vs" or "compare"
        if "vs" in ql:
            parts = re.split(r"\bvs\b", q, flags=re.IGNORECASE)
        else:
            # "compare A and B"
            parts = re.split(r"\bcompare\b|\band\b", q, flags=re.IGNORECASE)

        parts = [p.strip(" :-") for p in parts if p.strip(" :-")]
        if len(parts) >= 2:
            name_a = parts[0]
            name_b = parts[1]
            cmp = compare_phones(phones, name_a, name_b)
            if "error" in cmp:
                return "I don't have that information."

            a = cmp["A"]
            b = cmp["B"]
            return (
                f"{SYSTEM_PROMPT}\n\n"
                f"Comparison (from available data):\n"
                f"- A: {a['brand']} {a['model']} | OS: {a['operating_system']} | Chipset: {a['chipset']} | "
                f"RAM: {a['ram']} | Storage: {a['internal_storage']} | Battery: {a['battery_capacity']} | "
                f"Camera: {a['primary_camera_resolution']} | Refresh: {a['refresh_rate']}\n"
                f"- B: {b['brand']} {b['model']} | OS: {b['operating_system']} | Chipset: {b['chipset']} | "
                f"RAM: {b['ram']} | Storage: {b['internal_storage']} | Battery: {b['battery_capacity']} | "
                f"Camera: {b['primary_camera_resolution']} | Refresh: {b['refresh_rate']}\n"
            )

        return "I don't have that information."

    # ---------- TOOL: Best / Top battery ----------
    if ("best battery" in ql) or ("top battery" in ql) or ("highest battery" in ql):
        brand = _extract_brand(q)
        os_contains = "android" if "android" in ql else None

        results = top_battery(phones, brand=brand, os_contains=os_contains, top_k=5)
        if not results:
            return "I don't have that information."

        lines = [f"- {name}: {mah} mAh" for name, mah in results]
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Top battery phones (from available data):\n"
            + "\n".join(lines)
        )

    # ---------- TOOL: Filter constraints ----------
    # Examples:
    # "android phone with 8gb ram and 256gb storage"
    # "phone with battery 6000mah"
    wants_filter = any(k in ql for k in ["under", "less than", "more than", "at least", "min", "maximum", ">= ", "<= "]) \
                  or ("ram" in ql) or ("storage" in ql) or ("battery" in ql)

    if wants_filter:
        brand = _extract_brand(q)
        os_contains = "android" if "android" in ql else None

        min_ram_gb = _extract_gb(q, ["ram"])
        min_storage_gb = _extract_gb(q, ["storage", "rom"])

        min_battery_mah = _extract_int_after_keywords(q, ["battery", "mah"])

        filtered = filter_phones(
            phones,
            brand=brand,
            os_contains=os_contains,
            min_ram_gb=min_ram_gb,
            min_storage_gb=min_storage_gb,
            min_battery_mah=min_battery_mah,
            limit=8
        )

        # Special case: user mentions price but we don't have price
        if "$" in q or "price" in ql or "under" in ql:
            # dataset has no price → refuse safely
            # (unless your dataset later adds price)
            if all("price" not in (p or {}) for p in phones):
                return "I don't have that information."

        if not filtered:
            return "I don't have that information."

        # Use RAG to provide extra details about top few results (Tools + RAG)
        top_names = [f"{p.get('brand','')} {p.get('model','')}".strip() for p in filtered[:3]]
        rag_context = []
        for name in top_names:
            rag_context.extend(retrieve(f"info about {name}", top_k=1))

        rag_block = "\n".join(rag_context).strip()

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Filtered results (from available data):\n"
            + "\n".join([_format_phone_brief(p) for p in filtered])
            + ("\n\nExtra details (RAG):\n" + rag_block if rag_block else "")
        )

    # ---------- Default: RAG info / general question ----------
    retrieved_docs = retrieve(q, top_k=3)
    if not retrieved_docs:
        return "I don't have that information."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Answer based on available data:\n"
        + "\n".join([f"- {d}" for d in retrieved_docs])
    )


if __name__ == "__main__":
    # Try a few tests
    print(answer("give me info about the Oppo Yoyo"))
    print("\n" + "="*60 + "\n")
    print(answer("compare Honor 400 vs Honor 400 Pro"))
    print("\n" + "="*60 + "\n")
    print(answer("best battery android phone"))
    print("\n" + "="*60 + "\n")
    print(answer("android phone with 12gb ram and 256gb storage and battery 6000mah"))
