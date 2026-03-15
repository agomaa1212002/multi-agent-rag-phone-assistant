from tools import load_phones, filter_phones, top_battery, compare_phones

phones = load_phones()

print("\n=== Top battery phones (Android only) ===")
for name, mah in top_battery(phones, os_contains="android", top_k=5):
    print(f"- {name}: {mah} mAh")

print("\n=== Filter: Android phones with >= 8GB RAM and >= 256GB storage ===")
filtered = filter_phones(
    phones,
    os_contains="android",
    min_ram_gb=8,
    min_storage_gb=256,
    limit=5
)
for p in filtered:
    print(f"- {p.get('brand')} {p.get('model')} | RAM: {p.get('ram')} | Storage: {p.get('internal_storage')} | Battery: {p.get('battery_capacity')}")

print("\n=== Compare two phones ===")
cmp = compare_phones(phones, "Honor 400 Pro", "Honor 400")
print(cmp)
