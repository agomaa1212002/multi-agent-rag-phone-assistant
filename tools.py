import json
import re
from typing import Any, Dict, List, Optional, Tuple
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "phones_data_20250729_022344.json")

def load_phones(path: str = DATA_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_int(text: Optional[str]) -> Optional[int]:
    """
    Extract first integer from a string like '12GB', '512MB', 'Li-Ion 7200mAh', '120 Hz'.
    Returns None if not found.
    """
    if not text:
        return None
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else None

def _mah(battery_capacity: Optional[str]) -> Optional[int]:
    """
    Extract mAh from battery_capacity like 'Li-Ion 7200mAh'
    """
    if not battery_capacity:
        return None
    m = re.search(r"(\d+)\s*mAh", str(battery_capacity), re.IGNORECASE)
    if m:
        return int(m.group(1))
    # fallback: any integer
    return _to_int(battery_capacity)

def _gb(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    s = str(value).strip().upper()

    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None

    num = float(m.group(1))

    if "TB" in s:
        return num * 1024.0
    if "MB" in s:
        return num / 1024.0
    if "GB" in s:
        return num

    # fallback: assume GB
    return num


def filter_phones(
    phones: List[Dict[str, Any]],
    brand: Optional[str] = None,
    os_contains: Optional[str] = None,
    min_ram_gb: Optional[float] = None,
    min_storage_gb: Optional[float] = None,
    min_battery_mah: Optional[int] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Tool: Filter phones by structured constraints.
    """
    results = []
    for p in phones:
        b = (p.get("brand") or "").lower()
        os = (p.get("operating_system") or "").lower()

        ram_gb = _gb(p.get("ram"))
        storage_gb = _gb(p.get("internal_storage"))
        batt = _mah(p.get("battery_capacity"))

        if brand and brand.lower() not in b:
            continue
        if os_contains and os_contains.lower() not in os:
            continue
        if min_ram_gb is not None and (ram_gb is None or ram_gb < min_ram_gb):
            continue
        if min_storage_gb is not None and (storage_gb is None or storage_gb < min_storage_gb):
            continue
        if min_battery_mah is not None and (batt is None or batt < min_battery_mah):
            continue

        results.append(p)

    return results[:limit]

def top_battery(
    phones: List[Dict[str, Any]],
    brand: Optional[str] = None,
    os_contains: Optional[str] = None,
    top_k: int = 5
) -> List[Tuple[str, int]]:
    """
    Tool: Return top_k phones by battery (mAh).
    """
    scored = []
    for p in phones:
        b = (p.get("brand") or "").lower()
        os = (p.get("operating_system") or "").lower()
        if brand and brand.lower() not in b:
            continue
        if os_contains and os_contains.lower() not in os:
            continue

        batt = _mah(p.get("battery_capacity"))
        if batt is None:
            continue

        name = f"{p.get('brand', '')} {p.get('model', '')}".strip()
        scored.append((name, batt))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def compare_phones(
    phones: List[Dict[str, Any]],
    name_a: str,
    name_b: str
) -> Dict[str, Dict[str, Any]]:
    """
    Tool: Compare two phones by name (best-effort substring match).
    """
    def find_one(name: str) -> Optional[Dict[str, Any]]:
        n = name.lower().strip()

        candidates = []
        for p in phones:
            full = f"{p.get('brand','')} {p.get('model','')}".lower().strip()
            if not full:
                continue

            score = 0
            if full == n:
                score = 100
            elif full.startswith(n):
                score = 80
            elif n in full:
                score = 60

            if score > 0:
                # longer full name often means more specific model
                score += min(len(full) / 100.0, 1.0)
                candidates.append((score, p))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


    a = find_one(name_a)
    b = find_one(name_b)

    if a is None or b is None:
        return {"error": "I don't have that information."}

    def pick(p: Dict[str, Any]) -> Dict[str, Any]:
     return {
        "brand": p.get("brand"),
        "model": p.get("model"),
        "operating_system": p.get("operating_system"),
        "display_type": p.get("display_type"),          
        "screen_size": p.get("screen_size"),            
        "resolution": p.get("resolution"),
        "chipset": p.get("chipset"),
        "ram": p.get("ram"),
        "internal_storage": p.get("internal_storage"),
        "battery_capacity": p.get("battery_capacity"),
        "quick_charging": p.get("quick_charging"),     
        "primary_camera_resolution": p.get("primary_camera_resolution"),
        "refresh_rate": p.get("refresh_rate"),
    }

    return {
        "A": pick(a),
        "B": pick(b)
    }
