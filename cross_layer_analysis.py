#!/usr/bin/env python3
"""
Cross-layer synchronization analysis for cortical column model.
Computes effective excitatory/inhibitory input budgets for each E population,
comparing within-layer vs cross-layer contributions.
"""

import pandas as pd
import numpy as np

# --- Load data ---
ampa = pd.read_csv('config/conductances_AMPA2_alpha_v2.csv', index_col=0)
nmda = pd.read_csv('config/conductances_NMDA2_alpha_v2.csv', index_col=0)
prob = pd.read_csv('config/connection_probabilities2.csv', index_col=0)

# Fix column name if needed
ampa.index = ampa.index.str.strip()
nmda.index = nmda.index.str.strip()
prob.index = prob.index.str.strip()
ampa.columns = [c.strip() for c in ampa.columns]
nmda.columns = [c.strip() for c in nmda.columns]
prob.columns = [c.strip() for c in prob.columns]

# Population sizes
neuron_counts = {
    'E_L23': 3520, 'PV_L23': 260, 'SOM_L23': 410, 'VIP_L23': 263,
    'E_L4AB': 2720, 'PV_L4AB': 340, 'SOM_L4AB': 210, 'VIP_L4AB': 130,
    'E_L4C': 3192, 'PV_L4C': 320, 'SOM_L4C': 200, 'VIP_L4C': 88,
    'E_L5': 1600, 'PV_L5': 220, 'SOM_L5': 110, 'VIP_L5': 70,
    'E_L6': 2040, 'PV_L6': 195, 'SOM_L6': 110, 'VIP_L6': 70,
}

layers = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
pop_types = ['E', 'PV', 'SOM', 'VIP']
e_pops = [f'E_{l}' for l in layers]
all_pops = [f'{t}_{l}' for l in layers for t in pop_types]

def get_layer(pop):
    return '_'.join(pop.split('_')[1:])

def get_type(pop):
    return pop.split('_')[0]

def is_excitatory(pop):
    return get_type(pop) == 'E'

def is_inhibitory(pop):
    return get_type(pop) in ('PV', 'SOM', 'VIP')

def same_layer(pop1, pop2):
    return get_layer(pop1) == get_layer(pop2)

# --- Compute effective input for each target E population ---
print("=" * 120)
print("CROSS-LAYER SYNCHRONIZATION ANALYSIS")
print("=" * 120)
print()
print("Effective input = (AMPA_g + NMDA_g) * prob * N_source  for excitatory sources")
print("Effective input = AMPA_g * prob * N_source              for inhibitory sources (GABA via AMPA column)")
print("Units: nS (conductance * probability * count = effective total conductance input)")
print()

results = {}

for target in e_pops:
    target_layer = get_layer(target)

    cross_exc = 0.0
    cross_inh = 0.0
    within_exc = 0.0
    within_inh = 0.0

    cross_exc_detail = []
    cross_inh_detail = []
    within_exc_detail = []
    within_inh_detail = []

    for source in all_pops:
        if source not in ampa.index or target not in ampa.columns:
            continue

        a = ampa.loc[source, target]
        n = nmda.loc[source, target] if source in nmda.index and target in nmda.columns else 0.0
        p = prob.loc[source, target] if source in prob.index and target in prob.columns else 0.0
        N = neuron_counts.get(source, 0)

        if is_excitatory(source):
            eff = (a + n) * p * N
            if same_layer(source, target):
                within_exc += eff
                within_exc_detail.append((source, a, n, p, N, eff))
            else:
                cross_exc += eff
                cross_exc_detail.append((source, a, n, p, N, eff))
        elif is_inhibitory(source):
            # Inhibitory: AMPA column has GABA conductances
            eff = a * p * N
            if same_layer(source, target):
                within_inh += eff
                within_inh_detail.append((source, a, p, N, eff))
            else:
                cross_inh += eff
                cross_inh_detail.append((source, a, p, N, eff))

    results[target] = {
        'cross_exc': cross_exc,
        'cross_inh': cross_inh,
        'within_exc': within_exc,
        'within_inh': within_inh,
        'cross_exc_detail': cross_exc_detail,
        'cross_inh_detail': cross_inh_detail,
        'within_exc_detail': within_exc_detail,
        'within_inh_detail': within_inh_detail,
    }

# --- Table 1: Summary per target E population ---
print("=" * 120)
print("TABLE 1: EFFECTIVE INPUT BUDGET PER TARGET E POPULATION")
print("=" * 120)
print(f"{'Target':<10} {'Cross-Exc':>12} {'Cross-Inh':>12} {'Within-Exc':>12} {'Within-Inh':>12} {'Cross E/I':>10} {'Within E/I':>10} {'Cross/Within E':>15} {'Cross-E / Within-I':>18}")
print("-" * 120)

for target in e_pops:
    r = results[target]
    ce = r['cross_exc']
    ci = r['cross_inh']
    we = r['within_exc']
    wi = r['within_inh']
    cross_ei = ce / ci if ci > 0 else float('inf')
    within_ei = we / wi if wi > 0 else float('inf')
    cross_within = ce / we if we > 0 else float('inf')
    cross_e_within_i = ce / wi if wi > 0 else float('inf')
    print(f"{target:<10} {ce:>12.1f} {ci:>12.1f} {we:>12.1f} {wi:>12.1f} {cross_ei:>10.2f} {within_ei:>10.2f} {cross_within:>15.2f} {cross_e_within_i:>18.2f}")

# --- Table 2: Top cross-layer E->E connections ---
print()
print("=" * 120)
print("TABLE 2: TOP 20 CROSS-LAYER E->E CONNECTIONS (by effective excitation)")
print("=" * 120)
print(f"{'Rank':<6} {'Source->Target':<25} {'AMPA_g':>8} {'NMDA_g':>8} {'Prob':>8} {'N_src':>7} {'Eff_Input':>12} {'% of target cross-E':>20}")
print("-" * 120)

all_cross_ee = []
for target in e_pops:
    for (src, a, n, p, N, eff) in results[target]['cross_exc_detail']:
        all_cross_ee.append((src, target, a, n, p, N, eff))

all_cross_ee.sort(key=lambda x: x[6], reverse=True)

for i, (src, tgt, a, n, p, N, eff) in enumerate(all_cross_ee[:20]):
    pct = 100 * eff / results[tgt]['cross_exc'] if results[tgt]['cross_exc'] > 0 else 0
    print(f"{i+1:<6} {src+'->'+tgt:<25} {a:>8.4f} {n:>8.4f} {p:>8.4f} {N:>7} {eff:>12.1f} {pct:>19.1f}%")

# --- Table 3: Detailed breakdown per target ---
print()
print("=" * 120)
print("TABLE 3: DETAILED CROSS-LAYER EXCITATORY INPUTS TO EACH E POPULATION")
print("=" * 120)

for target in e_pops:
    r = results[target]
    print(f"\n--- {target} ---")
    print(f"  Within-layer E: {r['within_exc']:.1f}  |  Within-layer I: {r['within_inh']:.1f}  |  Within E/I: {r['within_exc']/r['within_inh']:.2f}" if r['within_inh'] > 0 else f"  Within-layer E: {r['within_exc']:.1f}  |  Within-layer I: {r['within_inh']:.1f}")
    print(f"  Cross-layer  E: {r['cross_exc']:.1f}  |  Cross-layer  I: {r['cross_inh']:.1f}  |  Cross  E/I: {r['cross_exc']/r['cross_inh']:.2f}" if r['cross_inh'] > 0 else f"  Cross-layer  E: {r['cross_exc']:.1f}  |  Cross-layer  I: {r['cross_inh']:.1f}")

    print(f"\n  Cross-layer excitatory sources:")
    print(f"    {'Source':<12} {'AMPA_g':>8} {'NMDA_g':>8} {'Prob':>8} {'N_src':>7} {'Effective':>12}")
    for (src, a, n, p, N, eff) in sorted(r['cross_exc_detail'], key=lambda x: x[5], reverse=True):
        print(f"    {src:<12} {a:>8.4f} {n:>8.4f} {p:>8.4f} {N:>7} {eff:>12.1f}")

    print(f"\n  Cross-layer inhibitory sources:")
    print(f"    {'Source':<12} {'GABA_g':>8} {'Prob':>8} {'N_src':>7} {'Effective':>12}")
    for (src, a, p, N, eff) in sorted(r['cross_inh_detail'], key=lambda x: x[4], reverse=True):
        if eff > 0.1:
            print(f"    {src:<12} {a:>8.4f} {p:>8.4f} {N:>7} {eff:>12.1f}")

# --- Table 4: Cross-layer excitation as fraction of total inhibition ---
print()
print("=" * 120)
print("TABLE 4: CROSS-LAYER EXCITATION vs TOTAL (WITHIN+CROSS) INHIBITION")
print("=" * 120)
print(f"{'Target':<10} {'Cross-Exc':>12} {'Total-Inh':>12} {'Ratio':>10} {'Unmatched Exc':>15}")
print("-" * 80)

for target in e_pops:
    r = results[target]
    total_inh = r['within_inh'] + r['cross_inh']
    ratio = r['cross_exc'] / total_inh if total_inh > 0 else float('inf')
    unmatched = r['cross_exc'] - r['cross_inh']
    print(f"{target:<10} {r['cross_exc']:>12.1f} {total_inh:>12.1f} {ratio:>10.2f} {unmatched:>15.1f}")

# --- Table 5: Bidirectional E-E loop analysis ---
print()
print("=" * 120)
print("TABLE 5: BIDIRECTIONAL E-E CROSS-LAYER LOOPS (potential synchronization drivers)")
print("=" * 120)
print(f"{'Loop':<30} {'A->B eff':>12} {'B->A eff':>12} {'Geometric mean':>15} {'Loop strength':>15}")
print("-" * 100)

loops = []
for i, l1 in enumerate(layers):
    for l2 in layers[i+1:]:
        src1 = f'E_{l1}'
        src2 = f'E_{l2}'

        # A->B
        a1 = ampa.loc[src1, src2]
        n1 = nmda.loc[src1, src2]
        p1 = prob.loc[src1, src2]
        N1 = neuron_counts[src1]
        eff_ab = (a1 + n1) * p1 * N1

        # B->A
        a2 = ampa.loc[src2, src1]
        n2 = nmda.loc[src2, src1]
        p2 = prob.loc[src2, src1]
        N2 = neuron_counts[src2]
        eff_ba = (a2 + n2) * p2 * N2

        geom = np.sqrt(eff_ab * eff_ba) if eff_ab > 0 and eff_ba > 0 else 0
        loop_strength = eff_ab + eff_ba
        loops.append((f'{src1} <-> {src2}', eff_ab, eff_ba, geom, loop_strength))

loops.sort(key=lambda x: x[4], reverse=True)
for (name, ab, ba, geom, total) in loops:
    print(f"{name:<30} {ab:>12.1f} {ba:>12.1f} {geom:>15.1f} {total:>15.1f}")

# --- Table 6: Cross-layer inhibitory connections (are there any significant ones?) ---
print()
print("=" * 120)
print("TABLE 6: ALL CROSS-LAYER INHIBITORY->E CONNECTIONS (effective > 1.0)")
print("=" * 120)
print(f"{'Source->Target':<25} {'GABA_g':>8} {'Prob':>8} {'N_src':>7} {'Effective':>12}")
print("-" * 70)

all_cross_ie = []
for target in e_pops:
    for (src, a, p, N, eff) in results[target]['cross_inh_detail']:
        if eff > 1.0:
            all_cross_ie.append((src, target, a, p, N, eff))

all_cross_ie.sort(key=lambda x: x[5], reverse=True)
for (src, tgt, a, p, N, eff) in all_cross_ie:
    print(f"{src+'->'+tgt:<25} {a:>8.4f} {p:>8.4f} {N:>7} {eff:>12.1f}")

# --- DIAGNOSIS ---
print()
print("=" * 120)
print("DIAGNOSIS: WHY CROSS-LAYER CONNECTIONS CAUSE SYNCHRONIZATION")
print("=" * 120)
print()

# Compute key metrics
for target in e_pops:
    r = results[target]
    total_inh = r['within_inh'] + r['cross_inh']
    cross_unmatched = r['cross_exc'] - r['cross_inh']
    pct_unmatched = 100 * cross_unmatched / total_inh if total_inh > 0 else float('inf')
    print(f"  {target}: Cross-layer excitation unmatched by cross-layer inhibition = {cross_unmatched:.1f}")
    print(f"           This unmatched excitation = {pct_unmatched:.1f}% of total inhibition")
    print()

print()
print("KEY FINDINGS:")
print("-" * 80)

# Find the worst offenders
worst_targets = sorted(e_pops, key=lambda t: results[t]['cross_exc'] / (results[t]['within_inh'] + results[t]['cross_inh']) if (results[t]['within_inh'] + results[t]['cross_inh']) > 0 else 0, reverse=True)

for t in worst_targets:
    r = results[t]
    total_inh = r['within_inh'] + r['cross_inh']
    print(f"  {t}: cross_exc/total_inh = {r['cross_exc']/total_inh:.2f}" if total_inh > 0 else f"  {t}: no inhibition")

print()
print("TOP 5 PROBLEMATIC CONNECTIONS (candidates for reduction):")
print("-" * 80)
for i, (src, tgt, a, n, p, N, eff) in enumerate(all_cross_ee[:5]):
    r = results[tgt]
    total_inh = r['within_inh'] + r['cross_inh']
    pct_of_inh = 100 * eff / total_inh if total_inh > 0 else float('inf')
    print(f"  {i+1}. {src}->{tgt}: eff={eff:.1f} ({pct_of_inh:.1f}% of target's total inhibition)")
    print(f"     AMPA={a:.4f}, NMDA={n:.4f}, prob={p:.4f}, N={N}")
    # Suggest reduction
    if total_inh > 0:
        # To make this connection contribute < 10% of total inhibition
        target_eff = 0.10 * total_inh
        if eff > target_eff:
            scale = target_eff / eff
            print(f"     RECOMMENDATION: Scale AMPA to {a*scale:.4f} and NMDA to {n*scale:.4f} (x{scale:.2f})")
    print()

# Feedback loop analysis
print()
print("FEEDBACK LOOP RISK:")
print("-" * 80)
print("The strongest bidirectional E-E loops can sustain reverberating excitation.")
print("When layer A excites layer B AND layer B excites layer A, any activity")
print("fluctuation gets amplified across layers, driving synchronization.")
print()
for (name, ab, ba, geom, total) in loops[:5]:
    print(f"  {name}: total loop strength = {total:.1f}, geometric mean = {geom:.1f}")

print()
print()
print("SUMMARY OF SYNCHRONIZATION MECHANISM:")
print("=" * 80)

total_cross_exc_all = sum(results[t]['cross_exc'] for t in e_pops)
total_cross_inh_all = sum(results[t]['cross_inh'] for t in e_pops)
total_within_inh_all = sum(results[t]['within_inh'] for t in e_pops)

print(f"  Total cross-layer excitation across all E pops: {total_cross_exc_all:.1f}")
print(f"  Total cross-layer inhibition across all E pops: {total_cross_inh_all:.1f}")
print(f"  Ratio: {total_cross_exc_all/total_cross_inh_all:.2f}x" if total_cross_inh_all > 0 else "  No cross-layer inhibition!")
print(f"  Total within-layer inhibition across all E pops: {total_within_inh_all:.1f}")
print(f"  Cross-layer exc as % of within-layer inh: {100*total_cross_exc_all/total_within_inh_all:.1f}%")
print()
print("  Cross-layer connections are overwhelmingly excitatory (E->E).")
print("  Cross-layer inhibition is sparse and weak by comparison.")
print("  Local inhibitory circuits cannot counterbalance the cross-layer excitation")
print("  because they only respond AFTER the excitatory volley has already propagated.")
print("  This creates a synchronizing chain: E_A -> E_B -> E_A (positive feedback loop).")
