"""Side-by-side PSD comparison for ONE trial: main.py's view vs lesion view.

For each bipolar channel we plot:
    LEFT  -- main.py replica: Welch on the EARLY window (last `pre_ms` ms of
             the baseline phase, i.e. just before the would-be stimulus),
             nperseg same as main.py, single segment, hann window.
             Peak marker = np.argmax(PSD)  (what main.py annotates).
    RIGHT -- lesion-analysis replica: multitaper PSD on the LATE window
             (the last `late_ms` ms of the trial). Peak markers =
             scipy.signal.find_peaks with prominence on log10(PSD).

We also print a per-channel table comparing the two methods so we can see
where they disagree.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, detrend

from laminar_power_change import multitaper_psd
from lesion_significance2 import BANDS


def main(npz_path, out_path, pre_ms=1000, late_ms=2000, fmax=100,
         min_prom=0.05):
    d = np.load(npz_path, allow_pickle=True)
    bip = d['bipolar_matrix']            # (n_ch, n_samples)
    t = d['time_array_ms']               # ms
    depths = d['channel_depths']
    labels = d['channel_labels']
    baseline_ms = int(d['baseline_ms'])
    n_ch = bip.shape[0]
    fs = 1000.0 / float(np.mean(np.diff(t)))  # Hz
    print(f'fs = {fs:.1f} Hz, baseline_ms = {baseline_ms}, '
          f'trial length = {t[-1]:.1f} ms')

    # main.py window: from (baseline_ms - pre_ms) to baseline_ms
    pre_lo = baseline_ms - pre_ms
    pre_hi = baseline_ms
    mask_pre = (t >= pre_lo) & (t < pre_hi)
    pre_n = int(mask_pre.sum())

    # multitaper window: last late_ms of trial
    t_end = float(t[-1])
    mask_late = (t >= t_end - late_ms) & (t <= t_end)
    late_n = int(mask_late.sum())

    # Welch nperseg matches the formula in src/visualization.py:451
    #   nperseg = 100 * min(1024, len(lfp_pre) // 4)
    nperseg = 100 * min(1024, pre_n // 4)
    nperseg = min(nperseg, pre_n)  # Welch would silently truncate; be explicit
    print(f'pre window n={pre_n} samples, Welch nperseg={nperseg} '
          f'(one segment, no averaging if nperseg==pre_n)')
    print(f'late window n={late_n} samples, multitaper NW=2')

    # set up figure: 1 row per channel, 2 columns
    fig, axes = plt.subplots(n_ch, 2, figsize=(13, 2.0 * n_ch),
                             sharex=False)
    if n_ch == 1:
        axes = axes[None, :]

    table = []
    for ch in range(n_ch):
        lfp_pre = bip[ch, mask_pre]
        lfp_late = bip[ch, mask_late]

        # --- main.py replica: Welch, single window, no detrend ---
        f_w, psd_w = welch(lfp_pre, fs=fs, nperseg=nperseg, window='hann')
        m_w = f_w <= fmax
        f_w, psd_w = f_w[m_w], psd_w[m_w]
        argmax_idx = int(np.argmax(psd_w))
        argmax_f = float(f_w[argmax_idx])

        # find_peaks on the SAME welch spectrum for an apples-to-apples view
        y_w = np.log10(np.maximum(psd_w, 1e-20))
        idx_pk_w, props_w = find_peaks(y_w, prominence=min_prom)
        pk_w = []
        for j, i in enumerate(idx_pk_w):
            pk_w.append({'f': float(f_w[i]), 'p': float(props_w['prominences'][j])})
        pk_w.sort(key=lambda x: -x['p'])

        # --- multitaper on late window, with detrend ---
        seg = detrend(lfp_late.astype(np.float64))
        nfft = 2 ** int(np.ceil(np.log2(len(seg))))
        f_m, psd_m = multitaper_psd(seg, fs=fs, NW=2, nfft=nfft)
        m_m = f_m <= fmax
        f_m, psd_m = f_m[m_m], psd_m[m_m]
        argmax_m = float(f_m[int(np.argmax(psd_m))])

        y_m = np.log10(np.maximum(psd_m, 1e-20))
        idx_pk_m, props_m = find_peaks(y_m, prominence=min_prom)
        pk_m = []
        for j, i in enumerate(idx_pk_m):
            pk_m.append({'f': float(f_m[i]), 'p': float(props_m['prominences'][j])})
        pk_m.sort(key=lambda x: -x['p'])

        # dominant of find_peaks (highest log-power among >= 2 Hz)
        def _dominant(f_arr, y_arr, peaks):
            cands = [p for p in peaks if p['f'] >= 2.0]
            if not cands:
                return None
            # pick by highest power (same rule as control_peak_freqs.py)
            best_f = None; best_y = -np.inf
            for p in cands:
                i = np.argmin(np.abs(f_arr - p['f']))
                if y_arr[i] > best_y:
                    best_y = y_arr[i]
                    best_f = p['f']
            return best_f

        dom_w = _dominant(f_w, y_w, pk_w)
        dom_m = _dominant(f_m, y_m, pk_m)

        # ---------- plot left (main.py replica) ----------
        ax = axes[ch, 0]
        ax.semilogy(f_w, psd_w, 'b-', lw=1.4)
        ax.axvline(argmax_f, color='k', linestyle=':',
                   label=f'argmax {argmax_f:.1f} Hz')
        if dom_w is not None and abs(dom_w - argmax_f) > 0.5:
            ax.axvline(dom_w, color='C2', linestyle='--',
                       label=f'find_peaks dominant {dom_w:.1f} Hz')
        # mark all find_peaks
        for p in pk_w[:4]:
            i = np.argmin(np.abs(f_w - p['f']))
            ax.plot(p['f'], psd_w[i], 'o', color='C1', ms=5,
                    mec='k', mew=0.4)
        ax.set_xlim(0, fmax)
        ax.set_ylabel(f'{labels[ch]}\nz={depths[ch]:+.2f}', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc='upper right')
        ax.grid(True, which='both', alpha=0.25)
        if ch == 0:
            ax.set_title(f"main.py replica: Welch on EARLY window "
                         f"({pre_lo}–{pre_hi} ms), nperseg={nperseg}",
                         fontsize=10)

        # ---------- plot right (multitaper, late) ----------
        ax = axes[ch, 1]
        ax.semilogy(f_m, psd_m, 'r-', lw=1.4)
        ax.axvline(argmax_m, color='k', linestyle=':',
                   label=f'argmax {argmax_m:.1f} Hz')
        if dom_m is not None and abs(dom_m - argmax_m) > 0.5:
            ax.axvline(dom_m, color='C2', linestyle='--',
                       label=f'find_peaks dominant {dom_m:.1f} Hz')
        for p in pk_m[:4]:
            i = np.argmin(np.abs(f_m - p['f']))
            ax.plot(p['f'], psd_m[i], 'o', color='C1', ms=5,
                    mec='k', mew=0.4)
        ax.set_xlim(0, fmax)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc='upper right')
        ax.grid(True, which='both', alpha=0.25)
        if ch == 0:
            ax.set_title(f"lesion replica: multitaper on LATE window "
                         f"(last {late_ms} ms), NW=2",
                         fontsize=10)

        # ---------- band lines ----------
        for axc in (axes[ch, 0], axes[ch, 1]):
            for name, (lo, hi) in BANDS.items():
                axc.axvspan(lo, hi, color='gray', alpha=0.04)

        # ---------- collect table row ----------
        table.append({
            'ch': str(labels[ch]),
            'z': float(depths[ch]),
            'welch_argmax': argmax_f,
            'welch_dom_findpeaks': dom_w if dom_w is not None else float('nan'),
            'welch_top_proms': ','.join(f"{p['f']:.1f}Hz({p['p']:.2f})"
                                         for p in pk_w[:3]),
            'mt_argmax': argmax_m,
            'mt_dom_findpeaks': dom_m if dom_m is not None else float('nan'),
            'mt_top_proms': ','.join(f"{p['f']:.1f}Hz({p['p']:.2f})"
                                      for p in pk_m[:3]),
        })

    axes[-1, 0].set_xlabel('Frequency (Hz)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')
    fig.suptitle(
        f'PSD method comparison -- {os.path.basename(npz_path)}  '
        f'(net seed={int(d["network_seed"])}, trial {int(d["trial_id"])})',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')

    # ---- text table ----
    print()
    print(f'{"ch":>10} {"z":>6} | {"W argmax":>9} {"W find_p":>9}  '
          f'| {"MT argmax":>10} {"MT find_p":>10}')
    print('-' * 88)
    for r in table:
        print(f'{r["ch"]:>10} {r["z"]:+6.2f} | '
              f'{r["welch_argmax"]:>9.2f} {r["welch_dom_findpeaks"]:>9.2f}  | '
              f'{r["mt_argmax"]:>10.2f} {r["mt_dom_findpeaks"]:>10.2f}')

    print('\nTop find_peaks prominences (freq(Hz)(prom)) per channel:')
    for r in table:
        print(f'{r["ch"]:>10}  Welch: {r["welch_top_proms"]}'
              f'   |   MT: {r["mt_top_proms"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', default='results/control_2026-06-03/trial_000.npz')
    parser.add_argument('--out', default='figures/compare_psd_methods.png')
    parser.add_argument('--pre-ms', type=int, default=1000,
                        help='Width of the main.py "pre-stim" window.')
    parser.add_argument('--late-ms', type=int, default=2000,
                        help='Width of the lesion-analysis "late" window.')
    parser.add_argument('--fmax', type=float, default=100)
    parser.add_argument('--min-prom', type=float, default=0.05)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    main(args.trial, args.out, pre_ms=args.pre_ms, late_ms=args.late_ms,
         fmax=args.fmax, min_prom=args.min_prom)
