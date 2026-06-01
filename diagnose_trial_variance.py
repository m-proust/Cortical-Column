"""Quantitative synchrony / firing-irregularity diagnostic for one trial.

Reports per (layer, population):
  - mean firing rate (baseline and stim)
  - CV of inter-spike intervals (averaged across neurons): <0.4 clock-like,
    0.4-0.7 borderline, >0.7 irregular (Poisson-like ~1.0)
  - synchrony index: variance(population PSTH) / mean(population PSTH).
    Low values (~1) = asynchronous, high values = synchronous bursting.
"""

import os
import argparse
import numpy as np


def _isi_cv(spike_times, n_neurons):
    cvs = []
    for i in range(n_neurons):
        sp = spike_times[i]
        if len(sp) < 2:
            continue
        isi = np.diff(sp)
        if isi.mean() <= 0:
            continue
        cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else float('nan')


def _by_neuron(times_ms, idx, n_neurons):
    out = [[] for _ in range(n_neurons)]
    for t, k in zip(times_ms, idx):
        if 0 <= k < n_neurons:
            out[int(k)].append(float(t))
    return [np.asarray(s) for s in out]


def _psth(times_ms, t_lo, t_hi, bin_ms=2.0):
    bins = np.arange(t_lo, t_hi + bin_ms, bin_ms)
    h, _ = np.histogram(times_ms, bins=bins)
    return h.astype(float)


def diagnose(fname):
    data = np.load(fname, allow_pickle=True)
    spike_data = data['spike_data'].item()
    stim = float(data['stim_onset_ms'])
    t_end = float(data['time_array_ms'][-1])

    baseline_win = (max(0.0, stim - 1500.0), stim)
    stim_win = (stim + 500.0, min(t_end, stim + 2000.0))

    print(f"\n=== {os.path.basename(fname)} ===")
    print(f"baseline window: {baseline_win[0]:.0f}-{baseline_win[1]:.0f} ms   "
          f"stim window: {stim_win[0]:.0f}-{stim_win[1]:.0f} ms")
    print(f"{'layer':>6} {'pop':>5}  {'rate_b':>7} {'rate_s':>7}  "
          f"{'cv_b':>5} {'cv_s':>5}  {'sync_b':>7} {'sync_s':>7}  verdict")

    for layer, pops in spike_data.items():
        for pop_key, payload in pops.items():
            pop = pop_key.replace('_spikes', '')
            payload = payload.item() if hasattr(payload, 'item') else payload
            times = np.asarray(payload['times_ms'])
            idx = np.asarray(payload['spike_indices'])
            n_neurons = int(idx.max()) + 1 if len(idx) else 1

            mask_b = (times >= baseline_win[0]) & (times < baseline_win[1])
            mask_s = (times >= stim_win[0]) & (times < stim_win[1])
            dur_b = (baseline_win[1] - baseline_win[0]) / 1000.0
            dur_s = (stim_win[1] - stim_win[0]) / 1000.0

            rate_b = mask_b.sum() / (n_neurons * dur_b) if dur_b > 0 else 0.0
            rate_s = mask_s.sum() / (n_neurons * dur_s) if dur_s > 0 else 0.0

            per_neuron_b = _by_neuron(times[mask_b], idx[mask_b], n_neurons)
            per_neuron_s = _by_neuron(times[mask_s], idx[mask_s], n_neurons)
            cv_b = _isi_cv(per_neuron_b, n_neurons)
            cv_s = _isi_cv(per_neuron_s, n_neurons)

            psth_b = _psth(times[mask_b], baseline_win[0], baseline_win[1])
            psth_s = _psth(times[mask_s], stim_win[0], stim_win[1])
            sync_b = (psth_b.var() / psth_b.mean()) if psth_b.mean() > 0 else float('nan')
            sync_s = (psth_s.var() / psth_s.mean()) if psth_s.mean() > 0 else float('nan')

            verdict = []
            if cv_s < 0.4 or cv_b < 0.4:
                verdict.append('CLOCK-LIKE')
            if rate_s > 100:
                verdict.append('HIGH-RATE')
            if sync_s > 50 or sync_b > 50:
                verdict.append('OVER-SYNC')
            if not verdict:
                verdict.append('ok')

            print(f"{layer:>6} {pop:>5}  {rate_b:>7.1f} {rate_s:>7.1f}  "
                  f"{cv_b:>5.2f} {cv_s:>5.2f}  {sync_b:>7.1f} {sync_s:>7.1f}  "
                  f"{','.join(verdict)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str,
                        default='results/trials_19_05_4')
    parser.add_argument('--trials', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()

    for idx in args.trials:
        fname = os.path.join(args.base_path, f"trial_{idx:03d}.npz")
        if not os.path.exists(fname):
            print(f"skip: {fname} not found")
            continue
        diagnose(fname)
