"""
Scale cross-layer SOM→E and VIP→E AMPA conductances by 0.5.
Only affects entries where source layer != target layer.
"""

import csv
import copy

CSV_PATH = "config/conductances_AMPA2_alpha_v2.csv"
SCALE = 0.5

LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]

def get_layer(pop_name):
    for layer in LAYERS:
        if pop_name.endswith(layer):
            return layer
    return None

def get_type(pop_name):
    return pop_name.split("_")[0]

# Read
with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    rows = list(reader)

header = rows[0]
col_names = header[1:]  # target populations

changes = []

for i in range(1, len(rows)):
    row = rows[i]
    if not row or not row[0].strip():
        continue
    src = row[0].strip()
    src_type = get_type(src)
    src_layer = get_layer(src)

    if src_type not in ("SOM", "VIP"):
        continue

    for j in range(1, len(row)):
        tgt = col_names[j - 1].strip()
        tgt_type = get_type(tgt)
        tgt_layer = get_layer(tgt)

        if tgt_type != "E":
            continue
        if src_layer == tgt_layer:
            continue

        old_val = float(row[j])
        if old_val == 0.0:
            continue
        new_val = round(old_val * SCALE, 6)
        changes.append((src, tgt, old_val, new_val))
        row[j] = str(new_val)

# Write
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

# Summary
print(f"{'Source':<12} {'Target':<10} {'Old':>10} {'New':>10}")
print("-" * 45)
for src, tgt, old, new in changes:
    print(f"{src:<12} {tgt:<10} {old:>10.6f} {new:>10.6f}")
print(f"\nTotal changes: {len(changes)}")
