# Pre-registered test: is alpha set by the SOM->PV->E disinhibitory product? (written 2026-10-02, before any run)

Candidate found post hoc (8 indices tried on 93 rate-matched isolated-layer factorial conditions, 5 seeds):
D = (PV->E per-cell K*g) x (SOM->PV per-cell K*g), AMPA/GABA tables.
Pooled R2 0.64; L6-L23 offset at equal D +0.26 dB (other indices ~+1 dB).
Fit: alpha_dB = -4.89 * log10(D) + 15.43   (data range log10 D 2.19 - 3.10)
Base values: L23 D 1270 (log 3.10, pred 0.26, measured 0.1); L6 D 227 (log 2.36, pred 3.92, measured 5.0)

Conditions (isolated, rate-matched, 10 seeds, E-rate alpha above 1/f), predictions in dB:
| condition | how D changes | D factor | predicted alpha |
|---|---|---|---|
| L23 SOM->PV x0.5     | SP only | 0.5    | 1.73 |
| L23 SOM->PV x0.25    | SP only | 0.25   | 3.20 |
| L23 SOM->PV x0.125   | SP only | 0.125  | 4.67 |
| L23 SOM->PV x0.0625  | SP only | 0.0625 | 6.14 (extrapolation) |
| L23 PV->E x0.5       | PE only | 0.5    | 1.73 |
| L23 PV->E x0.25      | PE only | 0.25   | 3.20 |
| L6 SOM->PV x0.25     | SP only | 0.25   | 6.86 (extrapolation) |
| L6 SOM->PV x0.5      | SP only | 0.5    | 5.39 (extrapolation) |
| L6 SOM->PV x2        | SP only | 2      | 2.45 |
| L6 SOM->PV x4        | SP only | 4      | 0.97 |

Pass criteria (decided now):
1. Product, not either factor: L23 SP x0.5 vs PE x0.5, and SP x0.25 vs PE x0.25, each differ by < 1 dB.
2. One curve: every condition within +-1.5 dB of its prediction (L6 base itself is off by +1.1).
3. Monotonic in D within each layer.

## Outcome (scored 2026-10-02, 10 seeds each) — FAILED as a single-scalar mechanism
| condition | measured [95% CI] | predicted | diff |
|---|---|---|---|
| L23 SP x0.5 | 1.24 [1.07, 1.43] | 1.73 | -0.49 |
| L23 SP x0.25 | 1.55 [1.18, 1.92] | 3.20 | -1.65 |
| L23 SP x0.125 | 2.28 [1.95, 2.60] | 4.67 | -2.39 |
| L23 SP x0.0625 | 2.16 [1.93, 2.41] | 6.14 | -3.98 |
| L23 PE x0.5 | 2.04 [1.76, 2.32] | 1.73 | +0.31 |
| L23 PE x0.25 | 3.73 [3.51, 3.98] | 3.20 | +0.53 |
| L6 SP x0.25 | 5.24 [4.97, 5.48] | 6.86 | -1.62 |
| L6 SP x0.5 | 5.05 [4.84, 5.29] | 5.39 | -0.34 |
| L6 SP x2 | 3.68 [3.36, 3.99] | 2.45 | +1.23 |
| L6 SP x4 | 1.32 [0.96, 1.66] | 0.97 | +0.35 |
1. Product not factor: x0.5 pass (SP-PE -0.80 [-1.14,-0.45]); x0.25 FAIL (-2.18 [-2.62,-1.75]).
2. Within +-1.5 dB: 6/10 -> FAIL (lowering SOM->PV saturates at ~2 dB in L23, ~5 dB in L6).
3. Monotonic: pass up to saturation.
Conclusion: D = PE x SP is not the control variable. PV->E is the stronger single knob; SOM->PV only
works in the raising direction (removes L6 alpha at x4) and saturates when lowered.
