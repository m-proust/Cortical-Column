"""Scale cross-layer conductances by 0.5 for specific connection types."""
import csv
import os

SCALE = 0.5
LAYERS = ['L23', 'L4AB', 'L4C', 'L5', 'L6']

# Connection types to scale: (source_type, target_type)
CONN_TYPES = [
    ('E', 'SOM'),
    ('E', 'VIP'),
    ('SOM', 'VIP'),
    ('VIP', 'SOM'),
]

def get_pop_info(name):
    """Extract (type, layer) from population name like 'E_L23' or 'SOM_L4AB'."""
    for layer in LAYERS:
        if name.endswith('_' + layer):
            pop_type = name[:-(len(layer) + 1)]
            return pop_type, layer
    return None, None

def process_file(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    col_names = header[1:]  # target populations

    changes = []

    for i in range(1, len(rows)):
        src_name = rows[i][0]
        src_type, src_layer = get_pop_info(src_name)
        if src_type is None:
            continue

        for j in range(1, len(rows[i])):
            tgt_name = col_names[j - 1]
            tgt_type, tgt_layer = get_pop_info(tgt_name)
            if tgt_type is None:
                continue

            # Check if this is a cross-layer connection of the right type
            if src_layer == tgt_layer:
                continue

            if (src_type, tgt_type) not in CONN_TYPES:
                continue

            old_val = float(rows[i][j])
            if old_val == 0.0:
                continue

            new_val = old_val * SCALE
            changes.append((src_name, tgt_name, old_val, new_val))
            rows[i][j] = str(new_val)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return changes

files = [
    '/Users/mathildeproust/Desktop/Cortical-Column/config/conductances_AMPA2_alpha_v2.csv',
    '/Users/mathildeproust/Desktop/Cortical-Column/config/conductances_NMDA2_alpha_v2.csv',
]

for filepath in files:
    fname = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")
    print(f"{'='*60}")
    changes = process_file(filepath)
    if not changes:
        print("  No changes made.")
    else:
        print(f"  {len(changes)} entries scaled by {SCALE}:\n")
        for src, tgt, old, new in changes:
            print(f"  {src:>10s} -> {tgt:<10s}:  {old:.6g}  ->  {new:.6g}")

print("\nDone.")
