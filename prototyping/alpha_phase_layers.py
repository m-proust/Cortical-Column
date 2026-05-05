import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert


LAYER_Z_RANGES = {
    'L23':  (0.45, 1.10),
    'L4AB': (0.14, 0.45),
    'L4C':  (-0.14, 0.14),
    'L5':   (-0.34, -0.14),
    'L6':   (-0.62, -0.34),
}
LAYER_ORDER = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
LAYER_COLORS = {
    'L23':  '#1f77b4',
    'L4AB': '#2ca02c',
    'L4C':  '#9467bd',
    'L5':   '#ff7f0e',
    'L6':   '#d62728',
}

COMPARTMENTS = {
    'supra':    ['L23'],
    'granular': ['L4AB', 'L4C'],
    'infra':    ['L5', 'L6'],
}
COMPARTMENT_ORDER = ['supra', 'granular', 'infra']


def assign_layer(z):
    for layer, (lo, hi) in LAYER_Z_RANGES.items():
        if lo <= z < hi:
            return layer
    return None


def layer_average(bipolar_matrix, channel_depths):
    out = {}
    for layer in LAYER_ORDER:
        lo, hi = LAYER_Z_RANGES[layer]
        idx = np.where((channel_depths >= lo) & (channel_depths < hi))[0]
        if len(idx) == 0:
            continue
        out[layer] = np.mean(bipolar_matrix[idx], axis=0)
    return out


def compartment_average(bipolar_matrix, channel_depths):
    """One LFP signal per compartment by pooling all channels whose depth falls
    in any of that compartment's constituent layer z-ranges, then averaging."""
    out = {}
    for comp in COMPARTMENT_ORDER:
        idx_all = []
        for layer in COMPARTMENTS[comp]:
            lo, hi = LAYER_Z_RANGES[layer]
            idx = np.where((channel_depths >= lo) & (channel_depths < hi))[0]
            idx_all.extend(idx.tolist())
        if not idx_all:
            continue
        out[comp] = np.mean(bipolar_matrix[np.array(idx_all)], axis=0)
    return out


def bandpass(sig, fs, band, order=4):
    nyq = fs / 2.0
    sos = butter(order, [band[0] / nyq, band[1] / nyq],
                 btype='bandpass', output='sos')
    return sosfiltfilt(sos, sig)


def load_trial(fpath):
    d = np.load(fpath, allow_pickle=True)
    return {k: d[k] for k in d.files}


def plv(phase_a, phase_b):
    return np.abs(np.mean(np.exp(1j * (phase_a - phase_b))))


def compute_plv_matrix(trial_files, alpha_band, window, window_kind='baseline',
                       pool_fn=None, group_order=None):
    """Returns (layers, plv_mat, lag_mat_rad, plv_std, lag_std_rad).

    window is (t_lo, t_hi). Either bound may be None: when window_kind is
    'baseline', None for t_hi defaults to stim_onset; when 'stimulus', None for
    t_lo defaults to stim_onset and None for t_hi defaults to end of trial.

    lag_mat_rad[i,j] = circular mean of (phase_i - phase_j); positive value
    means layer i leads layer j (i's phase is ahead).

    plv_std is the across-trial standard deviation of per-trial |z|.
    lag_std_rad is the across-trial circular standard deviation of per-trial
    angle(z), i.e. sqrt(-2 * ln(R)) where R is the resultant length over trials.
    """
    if pool_fn is None:
        pool_fn = layer_average
        group_order = LAYER_ORDER
    elif group_order is None:
        raise ValueError("group_order must be provided when pool_fn is custom")

    layers_present = None
    plv_per_trial = []  # list of (n,n) arrays
    cs_per_trial = []   # list of (n,n) complex arrays (z per trial, |z|<=1)

    for fpath in trial_files:
        trial = load_trial(fpath)
        bipolar = trial['bipolar_matrix']
        depths = trial['channel_depths']
        time_ms = trial['time_array_ms']
        dt = float(time_ms[1] - time_ms[0])
        fs = 1000.0 / dt
        stim = float(trial.get('stim_onset_ms', 2000))

        t_lo, t_hi = window
        if window_kind == 'baseline':
            if t_hi is None:
                t_hi = stim
        elif window_kind == 'stimulus':
            if t_lo is None:
                t_lo = stim
            if t_hi is None:
                t_hi = float(time_ms[-1])
        mask = (time_ms >= t_lo) & (time_ms < t_hi)
        if not np.any(mask):
            continue

        sigs = pool_fn(bipolar, depths)
        if layers_present is None:
            layers_present = [g for g in group_order if g in sigs]
        n = len(layers_present)

        phases = {}
        for g in layers_present:
            seg = sigs[g][mask]
            phases[g] = np.angle(hilbert(bandpass(seg, fs, alpha_band)))

        plv_trial = np.zeros((n, n))
        cs_trial = np.zeros((n, n), dtype=complex)
        for i, la in enumerate(layers_present):
            for j, lb in enumerate(layers_present):
                diff = phases[la] - phases[lb]
                z = np.mean(np.exp(1j * diff))
                plv_trial[i, j] = np.abs(z)
                cs_trial[i, j] = z
        plv_per_trial.append(plv_trial)
        cs_per_trial.append(cs_trial)

    if not plv_per_trial:
        raise RuntimeError("no trials produced PLV data")

    plv_arr = np.stack(plv_per_trial, axis=0)        # (T, n, n)
    cs_arr = np.stack(cs_per_trial, axis=0)          # (T, n, n) complex
    plv_mat = plv_arr.mean(axis=0)
    plv_std = plv_arr.std(axis=0)
    lag_mat = np.angle(cs_arr.mean(axis=0))

    # circular std across trials of per-trial lag angle
    unit = np.exp(1j * np.angle(cs_arr))             # unit complex per trial
    R = np.abs(unit.mean(axis=0))                    # resultant length, in [0,1]
    R_safe = np.clip(R, 1e-12, 1.0)
    lag_std = np.sqrt(-2.0 * np.log(R_safe))         # radians

    return layers_present, plv_mat, lag_mat, plv_std, lag_std


def plot_single_trial(trial, alpha_band, save_path,
                      time_window=None):
    bipolar = trial['bipolar_matrix']
    depths = trial['channel_depths']
    time_ms = trial['time_array_ms']
    dt = float(time_ms[1] - time_ms[0])
    fs = 1000.0 / dt
    stim = float(trial.get('stim_onset_ms', 2000))

    layer_sigs = layer_average(bipolar, depths)
    layers_present = [l for l in LAYER_ORDER if l in layer_sigs]

    if time_window is None:
        t_lo, t_hi = 500.0, stim
    else:
        t_lo, t_hi = time_window
    mask = (time_ms >= t_lo) & (time_ms < t_hi)
    t_seg = time_ms[mask]

    filt = {}
    phase = {}
    for layer in layers_present:
        seg = layer_sigs[layer][mask]
        f = bandpass(seg, fs, alpha_band)
        filt[layer] = f
        phase[layer] = np.angle(hilbert(f))

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                      gridspec_kw={'height_ratios': [1.4, 1]})

    spacing = 0
    amps = [np.max(np.abs(filt[l])) for l in layers_present]
    spacing = 2.2 * max(amps) if amps else 1.0
    for i, layer in enumerate(layers_present):
        offset = (len(layers_present) - 1 - i) * spacing
        ax_a.plot(t_seg, filt[layer] + offset,
                  color=LAYER_COLORS[layer], lw=1.0, label=layer)
        ax_a.text(t_seg[0] - (t_seg[-1] - t_seg[0]) * 0.012, offset,
                  layer, color=LAYER_COLORS[layer],
                  fontsize=10, fontweight='bold',
                  va='center', ha='right')
    if stim >= t_lo and stim <= t_hi:
        ax_a.axvline(stim, color='k', ls='--', alpha=0.4, lw=1)
    ax_a.set_ylabel('alpha-filtered bipolar LFP\n(stacked, a.u.)')
    ax_a.set_title(f'Layer alpha [{alpha_band[0]}-{alpha_band[1]} Hz] '
                   f'(trial {int(trial.get("trial_id", -1))})',
                   fontsize=11, fontweight='bold')
    ax_a.set_yticks([])

    for layer in layers_present:
        ax_b.plot(t_seg, phase[layer], color=LAYER_COLORS[layer],
                  lw=0.9, label=layer, alpha=0.9)
    if stim >= t_lo and stim <= t_hi:
        ax_b.axvline(stim, color='k', ls='--', alpha=0.4, lw=1)
    ax_b.set_xlabel('time (ms)')
    ax_b.set_ylabel('alpha phase (rad)')
    ax_b.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_b.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax_b.legend(loc='upper right', fontsize=8, ncol=len(layers_present))
    ax_b.set_title('Instantaneous alpha phase per layer (hilbert)',
                   fontsize=11, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def plot_plv_and_lag(layers, plv_mat, lag_mat, save_path,
                     window_label='baseline'):
    n = len(layers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    im = ax.imshow(plv_mat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(layers)
    ax.set_yticklabels(layers)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{plv_mat[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if plv_mat[i, j] < 0.6 else 'black',
                    fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label='PLV')
    ax.set_title('Inter-layer alpha PLV', fontsize=11, fontweight='bold')

    ax = axes[1]
    lag_deg = np.degrees(lag_mat)
    vmax = 180
    im = ax.imshow(lag_deg, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(layers)
    ax.set_yticklabels(layers)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{lag_deg[i, j]:+.0f}°',
                    ha='center', va='center',
                    color='black' if abs(lag_deg[i, j]) < 90 else 'white',
                    fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label='phase of (row − col), deg')
    ax.set_title('Inter-layer phase lag',
                 fontsize=11, fontweight='bold')

    fig.suptitle(f'{window_label.capitalize()} alpha coupling between layers (mean)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def plot_lag_masked(layers, plv_mat, lag_mat, save_path, plv_threshold=0.4,
                    window_label='baseline'):
    """Plot phase-lag matrix but blank out cells where PLV < threshold.

    Low-PLV cells are unreliable directional estimates (the resultant vector is
    near zero, so its angle is essentially noise) — masking them out keeps only
    the lags worth interpreting.
    """
    n = len(layers)
    lag_deg = np.degrees(lag_mat)
    mask = (plv_mat < plv_threshold) | (np.eye(n, dtype=bool))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    display = np.where(mask, np.nan, lag_deg)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color='#dddddd')
    im = ax.imshow(display, vmin=-180, vmax=180, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(layers)
    ax.set_yticklabels(layers)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                txt = '—' if i != j else ''
                ax.text(j, i, txt, ha='center', va='center',
                        color='#888888', fontsize=10)
            else:
                v = lag_deg[i, j]
                ax.text(j, i, f'{v:+.0f}°', ha='center', va='center',
                        color='black' if abs(v) < 90 else 'white',
                        fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label='phase of (row − col), deg')
    ax.set_title(f'Inter-layer phase lag for relevant PLV only '
                 f'({window_label})',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def plot_compare_windows(layers, plv_a, lag_a, plv_b, lag_b,
                         save_path, label_a='baseline', label_b='stimulus',
                         plv_threshold=0.4):
    """Side-by-side baseline vs stimulus PLV and (masked) phase lag."""
    n = len(layers)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5),
                             gridspec_kw={'width_ratios': [1, 1, 1]})

    def _plv_panel(ax, mat, title):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(layers); ax.set_yticklabels(layers)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{mat[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if mat[i, j] < 0.6 else 'black',
                        fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return im

    def _lag_panel(ax, plv_mat, lag_mat, title):
        lag_deg = np.degrees(lag_mat)
        mask = (plv_mat < plv_threshold) | (np.eye(n, dtype=bool))
        display = np.where(mask, np.nan, lag_deg)
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad(color='#dddddd')
        im = ax.imshow(display, vmin=-180, vmax=180, cmap=cmap)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(layers); ax.set_yticklabels(layers)
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    txt = '—' if i != j else ''
                    ax.text(j, i, txt, ha='center', va='center',
                            color='#888888', fontsize=9)
                else:
                    v = lag_deg[i, j]
                    ax.text(j, i, f'{v:+.0f}°', ha='center', va='center',
                            color='black' if abs(v) < 90 else 'white',
                            fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return im

    _plv_panel(axes[0, 0], plv_a, f'PLV — {label_a}')
    im_plv_b = _plv_panel(axes[0, 1], plv_b, f'PLV — {label_b}')
    im_dplv = axes[0, 2].imshow(plv_b - plv_a, vmin=-0.4, vmax=0.4, cmap='PuOr_r')
    axes[0, 2].set_xticks(range(n)); axes[0, 2].set_yticks(range(n))
    axes[0, 2].set_xticklabels(layers); axes[0, 2].set_yticklabels(layers)
    delta_plv = plv_b - plv_a
    for i in range(n):
        for j in range(n):
            axes[0, 2].text(j, i, f'{delta_plv[i, j]:+.2f}',
                            ha='center', va='center', fontsize=8,
                            color='black')
    axes[0, 2].set_title(f'ΔPLV ({label_b} − {label_a})',
                         fontsize=10, fontweight='bold')

    fig.colorbar(im_plv_b, ax=axes[0, :2].tolist(), shrink=0.7, label='PLV')
    fig.colorbar(im_dplv, ax=axes[0, 2], shrink=0.7, label='ΔPLV')

    _lag_panel(axes[1, 0], plv_a, lag_a,
               f'lag — {label_a} (PLV ≥ {plv_threshold:.2f})')
    im_lag_b = _lag_panel(axes[1, 1], plv_b, lag_b,
                          f'lag — {label_b} (PLV ≥ {plv_threshold:.2f})')

    # phase shift (b − a), wrapped to [-180,180], masked where either window
    # has insufficient PLV
    diff = np.angle(np.exp(1j * (lag_b - lag_a)))
    diff_deg = np.degrees(diff)
    mask = ((plv_a < plv_threshold) | (plv_b < plv_threshold)
            | np.eye(n, dtype=bool))
    display = np.where(mask, np.nan, diff_deg)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color='#dddddd')
    im_dlag = axes[1, 2].imshow(display, vmin=-180, vmax=180, cmap=cmap)
    axes[1, 2].set_xticks(range(n)); axes[1, 2].set_yticks(range(n))
    axes[1, 2].set_xticklabels(layers); axes[1, 2].set_yticklabels(layers)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                txt = '—' if i != j else ''
                axes[1, 2].text(j, i, txt, ha='center', va='center',
                                color='#888888', fontsize=9)
            else:
                v = diff_deg[i, j]
                axes[1, 2].text(j, i, f'{v:+.0f}°', ha='center', va='center',
                                color='black' if abs(v) < 90 else 'white',
                                fontsize=8)
    axes[1, 2].set_title(f'Δ lag ({label_b} − {label_a}, wrapped)',
                         fontsize=10, fontweight='bold')

    fig.colorbar(im_lag_b, ax=axes[1, :2].tolist(), shrink=0.7,
                 label='lag (deg)')
    fig.colorbar(im_dlag, ax=axes[1, 2], shrink=0.7, label='Δ lag (deg)')

    fig.suptitle(f'{label_a} vs {label_b}: alpha PLV and phase lag',
                 fontsize=13, fontweight='bold')
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def print_matrices(label, layers, plv_mat, plv_std, lag_mat, lag_std):
    print(f'=== {label} ===')
    print('PLV matrix (mean across trials):')
    for i, la in enumerate(layers):
        row = '  '.join(f'{plv_mat[i, j]:.2f}' for j in range(len(layers)))
        print(f"  {la:>4}  {row}")
    print('PLV std across trials:')
    for i, la in enumerate(layers):
        row = '  '.join(f'{plv_std[i, j]:.2f}' for j in range(len(layers)))
        print(f"  {la:>4}  {row}")
    print('phase lag matrix (deg, positive = row leads col):')
    lag_deg = np.degrees(lag_mat)
    for i, la in enumerate(layers):
        row = '  '.join(f'{lag_deg[i, j]:+6.1f}' for j in range(len(layers)))
        print(f"  {la:>4}  {row}")
    print('phase lag circular std across trials (deg):')
    lag_std_deg = np.degrees(lag_std)
    for i, la in enumerate(layers):
        row = '  '.join(f'{lag_std_deg[i, j]:6.1f}' for j in range(len(layers)))
        print(f"  {la:>4}  {row}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial_dir', type=str, required=True)
    p.add_argument('--trial_idx', type=int, default=0,
                   help='index of trial to plot in panels A/B')
    p.add_argument('--alpha_lo', type=float, default=7.0)
    p.add_argument('--alpha_hi', type=float, default=14.0)
    p.add_argument('--warmup_ms', type=float, default=500.0,
                   help='start of baseline window (skip warmup)')
    p.add_argument('--t_lo', type=float, default=None,
                   help='custom start time (ms) for the single-trial plot')
    p.add_argument('--t_hi', type=float, default=None,
                   help='custom end time (ms) for the single-trial plot')
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--plv_threshold', type=float, default=0.4,
                   help='hide phase-lag cells with PLV below this in masked plot')
    args = p.parse_args()

    save_dir = args.save_dir or os.path.join(args.trial_dir, 'alpha_phase_layers')
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.trial_dir, 'trial_*.npz')))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz in {args.trial_dir}")
    print(f"loaded {len(files)} trials")

    alpha_band = (args.alpha_lo, args.alpha_hi)

    idx = max(0, min(args.trial_idx, len(files) - 1))
    trial = load_trial(files[idx])
    if args.t_lo is not None and args.t_hi is not None:
        win = (args.t_lo, args.t_hi)
    else:
        stim = float(trial.get('stim_onset_ms', 2000))
        win = (args.warmup_ms, stim)
    plot_single_trial(
        trial, alpha_band,
        save_path=os.path.join(save_dir, f'layer_alpha_trace_trial{idx:03d}.png'),
        time_window=win,
    )

    layers, plv_b, lag_b, plv_b_std, lag_b_std = compute_plv_matrix(
        files, alpha_band,
        window=(args.warmup_ms, None),
        window_kind='baseline',
    )
    plot_plv_and_lag(layers, plv_b, lag_b,
                     save_path=os.path.join(save_dir, 'inter_layer_plv_baseline.png'),
                     window_label='baseline')
    plot_lag_masked(layers, plv_b, lag_b,
                    save_path=os.path.join(save_dir, 'inter_layer_lag_masked_baseline.png'),
                    plv_threshold=args.plv_threshold,
                    window_label='baseline')

    layers_s, plv_s, lag_s, plv_s_std, lag_s_std = compute_plv_matrix(
        files, alpha_band,
        window=(None, None),
        window_kind='stimulus',
    )
    if layers_s != layers:
        print(f"warning: layers differ between baseline ({layers}) and stimulus ({layers_s})")
    plot_plv_and_lag(layers_s, plv_s, lag_s,
                     save_path=os.path.join(save_dir, 'inter_layer_plv_stimulus.png'),
                     window_label='stimulus')
    plot_lag_masked(layers_s, plv_s, lag_s,
                    save_path=os.path.join(save_dir, 'inter_layer_lag_masked_stimulus.png'),
                    plv_threshold=args.plv_threshold,
                    window_label='stimulus')

    plot_compare_windows(layers, plv_b, lag_b, plv_s, lag_s,
                         save_path=os.path.join(save_dir, 'inter_layer_baseline_vs_stimulus.png'),
                         label_a='baseline', label_b='stimulus',
                         plv_threshold=args.plv_threshold)

    print('layers found:', layers)
    print_matrices('BASELINE', layers, plv_b, plv_b_std, lag_b, lag_b_std)
    print_matrices('STIMULUS', layers_s, plv_s, plv_s_std, lag_s, lag_s_std)

    delta_plv = plv_s - plv_b
    print('=== ΔPLV (stimulus − baseline) ===')
    for i, la in enumerate(layers):
        row = '  '.join(f'{delta_plv[i, j]:+.2f}' for j in range(len(layers)))
        print(f"  {la:>4}  {row}")
    delta_lag = np.degrees(np.angle(np.exp(1j * (lag_s - lag_b))))
    print('=== Δ phase lag (deg, wrapped to [-180,180]) ===')
    for i, la in enumerate(layers):
        row = '  '.join(f'{delta_lag[i, j]:+6.1f}' for j in range(len(layers)))
        print(f"  {la:>4}  {row}")

    # ---- compartment-level (supra / granular / infra) ----
    comp_b, plvc_b, lagc_b, plvc_b_std, lagc_b_std = compute_plv_matrix(
        files, alpha_band,
        window=(args.warmup_ms, None),
        window_kind='baseline',
        pool_fn=compartment_average,
        group_order=COMPARTMENT_ORDER,
    )
    comp_s, plvc_s, lagc_s, plvc_s_std, lagc_s_std = compute_plv_matrix(
        files, alpha_band,
        window=(None, None),
        window_kind='stimulus',
        pool_fn=compartment_average,
        group_order=COMPARTMENT_ORDER,
    )

    plot_compare_windows(comp_b, plvc_b, lagc_b, plvc_s, lagc_s,
                         save_path=os.path.join(save_dir,
                             'compartments_baseline_vs_stimulus.png'),
                         label_a='baseline', label_b='stimulus',
                         plv_threshold=args.plv_threshold)

    print('compartments found:', comp_b)
    print_matrices('COMPARTMENTS — BASELINE',
                   comp_b, plvc_b, plvc_b_std, lagc_b, lagc_b_std)
    print_matrices('COMPARTMENTS — STIMULUS',
                   comp_s, plvc_s, plvc_s_std, lagc_s, lagc_s_std)
    delta_plv_c = plvc_s - plvc_b
    print('=== Compartment ΔPLV (stimulus − baseline) ===')
    for i, c in enumerate(comp_b):
        row = '  '.join(f'{delta_plv_c[i, j]:+.2f}' for j in range(len(comp_b)))
        print(f"  {c:>9}  {row}")
    delta_lag_c = np.degrees(np.angle(np.exp(1j * (lagc_s - lagc_b))))
    print('=== Compartment Δ phase lag (deg, wrapped) ===')
    for i, c in enumerate(comp_b):
        row = '  '.join(f'{delta_lag_c[i, j]:+6.1f}' for j in range(len(comp_b)))
        print(f"  {c:>9}  {row}")


if __name__ == '__main__':
    main()
