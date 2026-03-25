"""Scale cross-layer E→E conductances by 0.5 in AMPA and NMDA CSV files."""

import csv
import re
import os

SCALE = 0.5
LAYERS = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
FILES = [
    'config/conductances_AMPA2_alpha_v2.csv',
    'config/conductances_NMDA2_alpha_v2.csv',
]

BASE = os.path.dirname(os.path.abspath(__file__))


def get_layer(name):
    """Extract layer from population name like E_L23."""
    for l in LAYERS:
        if name.endswith(l):
            return l
    return None


def process_file(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    # First column is source label
    col_names = header[1:]

    changes = []

    for i in range(1, len(rows)):
        row = rows[i]
        if not row or not row[0].strip():
            continue
        src = row[0].strip()
        # Only E source rows
        if not src.startswith('E_'):
            continue
        src_layer = get_layer(src)
        if src_layer is None:
            continue

        for j in range(1, len(row)):
            tgt = col_names[j - 1].strip()
            # Only E target columns
            if not tgt.startswith('E_'):
                continue
            tgt_layer = get_layer(tgt)
            if tgt_layer is None:
                continue
            # Cross-layer only
            if src_layer == tgt_layer:
                continue

            old_val = float(row[j])
            if old_val == 0.0:
                continue
            new_val = round(old_val * SCALE, 10)
            changes.append((src, tgt, old_val, new_val))
            row[j] = str(new_val)

    # Write back
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return changes


for fpath in FILES:
    full = os.path.join(BASE, fpath)
    print(f"\n{'='*60}")
    print(f"File: {fpath}")
    print(f"{'='*60}")
    changes = process_file(full)
    if not changes:
        print("  No changes.")
    else:
        for src, tgt, old, new in changes:
            print(f"  {src} -> {tgt}: {old} -> {new}")
        print(f"  Total: {len(changes)} entries scaled by {SCALE}")
