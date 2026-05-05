"""Plot LFPs for one trial: kernel method vs synaptic-current method.

Generates 6 figures total:
  - unipolar (raw amplitudes), unipolar (z-scored), unipolar PSD
  - bipolar  (raw amplitudes), bipolar  (z-scored), bipolar  PSD

Kernel = black, synaptic-current = gray.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

KERNEL_COLOR = '#1f6feb'   # deep blue
CURRENT_COLOR = '#e8590c'  # warm orange


def zscore(x, axis=-1):
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def stacked_offsets(x, spacing_factor=1.5):
    sd = x.std()
    step = spacing_factor * (sd if sd > 0 else 1.0)
    return np.arange(x.shape[0]) * step, step


def _slice_window(t, x, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], x[:, m]


def _fs_from_time(t_ms):
    dt_s = np.median(np.diff(t_ms)) * 1e-3
    return 1.0 / dt_s


def _bipolar(x):
    """Diff of consecutive channels: returns (n_ch-1, T)."""
    return np.diff(x, axis=0)


# ---------- generic plot helpers (operate on already-prepared arrays) ----------

def _plot_stacks(t_k, t_c, x_k, x_c, depths, channel_names, stim_on, t0, t1,
                 out_path: Path, title: str, overlap_spacing_sd: float):
    n_ch = x_k.shape[0]
    order = np.argsort(depths)

    off_k, step_k = stacked_offsets(x_k)
    off_c, step_c = stacked_offsets(x_c)
    kz = zscore(x_k, axis=1)
    cz = zscore(x_c, axis=1)
    off_z = np.arange(n_ch) * overlap_spacing_sd

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    ax_k, ax_c, ax_o = axes

    for i, ch in enumerate(order):
        ax_k.plot(t_k, x_k[ch] + off_k[i], color=KERNEL_COLOR, lw=0.6)
        ax_c.plot(t_c, x_c[ch] + off_c[i], color=CURRENT_COLOR, lw=0.6)
        ax_o.plot(t_k, kz[ch] + off_z[i], color=KERNEL_COLOR, lw=0.6,
                  label='kernel' if i == 0 else None)
        ax_o.plot(t_c, cz[ch] + off_z[i], color=CURRENT_COLOR, lw=0.6, alpha=0.9,
                  label='synaptic current' if i == 0 else None)

    labels = [f'{channel_names[ch]} (z={depths[ch]:.2f})' for ch in order]
    ax_k.set_yticks(off_k); ax_k.set_yticklabels(labels, fontsize=8)
    ax_k.set_title(f'Kernel (raw, step≈{step_k:.2g})')
    ax_k.set_ylabel('Channel (superficial → deep)')
    ax_c.set_yticks(off_c); ax_c.set_yticklabels(labels, fontsize=8)
    ax_c.set_title(f'Synaptic current (raw, step≈{step_c:.2g})')
    ax_o.set_yticks(off_z); ax_o.set_yticklabels(labels, fontsize=8)
    ax_o.set_title('Overlapped (z-scored)')
    ax_o.legend(loc='upper right', fontsize=9)

    for ax in axes:
        if t0 <= stim_on <= t1:
            ax.axvline(stim_on, color='k', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(t0, t1)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def _plot_normalized(t_k, t_c, x_k, x_c, depths, channel_names, stim_on, t0, t1,
                     out_path: Path, title: str, spacing_sd: float):
    n_ch = x_k.shape[0]
    order = np.argsort(depths)
    kz = zscore(x_k, axis=1)
    cz = zscore(x_c, axis=1)
    offsets = np.arange(n_ch) * spacing_sd

    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    ax_k, ax_c, ax_o = axes

    for i, ch in enumerate(order):
        ax_k.plot(t_k, kz[ch] + offsets[i], color=KERNEL_COLOR, lw=0.6)
        ax_c.plot(t_c, cz[ch] + offsets[i], color=CURRENT_COLOR, lw=0.6)
        ax_o.plot(t_k, kz[ch] + offsets[i], color=KERNEL_COLOR, lw=0.6,
                  label='kernel' if i == 0 else None)
        ax_o.plot(t_c, cz[ch] + offsets[i], color=CURRENT_COLOR, lw=0.6, alpha=0.9,
                  label='synaptic current' if i == 0 else None)

    labels = [f'{channel_names[ch]} (z={depths[ch]:.2f})' for ch in order]
    ax_k.set_yticks(offsets); ax_k.set_yticklabels(labels, fontsize=8)
    ax_k.set_ylabel('Channel (superficial → deep)')
    ax_o.legend(loc='upper right', fontsize=9)

    for ax, sub in zip(axes, ['Kernel (z)', 'Synaptic current (z)', 'Overlapped (z)']):
        if t0 <= stim_on <= t1:
            ax.axvline(stim_on, color='k', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel('Time (ms)')
        ax.set_title(sub)
        ax.set_xlim(t0, t1)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def _plot_psd(t_k, t_c, x_k, x_c, depths, channel_names, t0, t1,
              out_path: Path, title: str, fmin: float, fmax: float | None):
    fs_k = _fs_from_time(t_k)
    fs_c = _fs_from_time(t_c)
    n_ch = x_k.shape[0]
    order = np.argsort(depths)

    win_s = min(1.0, (t1 - t0) / 2000.0)
    nper_k = min(max(256, int(win_s * fs_k)), x_k.shape[1])
    nper_c = min(max(64, int(win_s * fs_c)), x_c.shape[1])

    # PSD computed on per-channel z-scored signals so amplitudes are comparable across methods
    x_k_n = zscore(x_k, axis=1)
    x_c_n = zscore(x_c, axis=1)
    f_k, P_k = welch(x_k_n, fs=fs_k, nperseg=nper_k, axis=1)
    f_c, P_c = welch(x_c_n, fs=fs_c, nperseg=nper_c, axis=1)
    if fmax is None:
        fmax = min(f_k[-1], f_c[-1])

    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, ch in enumerate(order):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        mk = (f_k >= fmin) & (f_k <= fmax)
        mc = (f_c >= fmin) & (f_c <= fmax)
        ax.loglog(f_k[mk], P_k[ch, mk], color=KERNEL_COLOR, lw=1.0,
                  label='kernel' if idx == 0 else None)
        ax.loglog(f_c[mc], P_c[ch, mc], color=CURRENT_COLOR, lw=1.0,
                  label='synaptic current' if idx == 0 else None)
        ax.set_title(f'{channel_names[ch]} (z={depths[ch]:.2f})', fontsize=9)
        ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)

    for idx in range(n_ch, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel('PSD')
    for c in range(ncols):
        axes[-1, c].set_xlabel('Frequency (Hz)')
    axes[0, 0].legend(loc='lower left', fontsize=8)

    # Common x-axis so the 1/f slope is visually comparable across electrodes
    for ax in axes.ravel():
        ax.set_xlim(fmin, fmax)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


# ---------- main entry ----------

def run_trial(trial_path: Path, trials_dir: Path, t0: float, t1: float,
              spacing_sd: float, fmin: float, fmax: float | None):
    d = np.load(trial_path, allow_pickle=True)

    t_kern_full = np.asarray(d['time_array_ms'], dtype=float)
    t_curr_full = np.asarray(d['time_current_ms'], dtype=float)
    lfp_kern = np.asarray(d['lfp_matrix'], dtype=float)          # (16, T_kern)
    lfp_curr = np.asarray(d['lfp_current_matrix'], dtype=float)  # (16, T_curr)
    stim_on = float(d['stim_onset_ms'])
    trial_id = int(d['trial_id'])

    t_kern, lfp_kern = _slice_window(t_kern_full, lfp_kern, t0, t1)
    t_curr, lfp_curr = _slice_window(t_curr_full, lfp_curr, t0, t1)

    elec_pos = np.asarray(d['electrode_positions'], dtype=float)  # (16, 3)
    uni_depths = elec_pos[:, 2]
    uni_names = [f'ch{i}' for i in range(uni_depths.size)]

    # Bipolar: diff between consecutive electrodes -> midpoint depth
    bip_kern = _bipolar(lfp_kern)
    bip_curr = _bipolar(lfp_curr)
    bip_depths = (uni_depths[:-1] + uni_depths[1:]) / 2
    bip_names = [f'ch{i+1}-ch{i}' for i in range(bip_depths.size)]

    suffix = f'_{int(t0)}-{int(t1)}ms'
    base = f'trial_{trial_id:03d}{suffix}'

    # ----- unipolar -----
    _plot_stacks(t_kern, t_curr, lfp_kern, lfp_curr, uni_depths, uni_names,
                 stim_on, t0, t1,
                 trials_dir / f'lfp_compare_{base}.png',
                 f'Trial {trial_id}: unipolar LFP — kernel vs synaptic current ({t0:.0f}–{t1:.0f} ms)',
                 spacing_sd)
    _plot_normalized(t_kern, t_curr, lfp_kern, lfp_curr, uni_depths, uni_names,
                     stim_on, t0, t1,
                     trials_dir / f'lfp_compare_normalized_{base}.png',
                     f'Trial {trial_id}: unipolar LFP — both normalized ({t0:.0f}–{t1:.0f} ms)',
                     spacing_sd)
    _plot_psd(t_kern, t_curr, lfp_kern, lfp_curr, uni_depths, uni_names,
              t0, t1,
              trials_dir / f'lfp_psd_{base}.png',
              f'Trial {trial_id}: unipolar PSD ({t0:.0f}–{t1:.0f} ms)',
              fmin, fmax)

    # ----- bipolar -----
    _plot_stacks(t_kern, t_curr, bip_kern, bip_curr, bip_depths, bip_names,
                 stim_on, t0, t1,
                 trials_dir / f'lfp_bipolar_compare_{base}.png',
                 f'Trial {trial_id}: bipolar LFP — kernel vs synaptic current ({t0:.0f}–{t1:.0f} ms)',
                 spacing_sd)
    _plot_normalized(t_kern, t_curr, bip_kern, bip_curr, bip_depths, bip_names,
                     stim_on, t0, t1,
                     trials_dir / f'lfp_bipolar_compare_normalized_{base}.png',
                     f'Trial {trial_id}: bipolar LFP — both normalized ({t0:.0f}–{t1:.0f} ms)',
                     spacing_sd)
    _plot_psd(t_kern, t_curr, bip_kern, bip_curr, bip_depths, bip_names,
              t0, t1,
              trials_dir / f'lfp_bipolar_psd_{base}.png',
              f'Trial {trial_id}: bipolar PSD ({t0:.0f}–{t1:.0f} ms)',
              fmin, fmax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials-dir', default='results/trials_26_04')
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--spacing-sd', type=float, default=4.0)
    p.add_argument('--t0', type=float, default=500.0)
    p.add_argument('--t1', type=float, default=2000.0)
    p.add_argument('--fmin', type=float, default=1.0)
    p.add_argument('--fmax', type=float, default=None)
    args = p.parse_args()

    trials_dir = Path(args.trials_dir)
    trial_path = trials_dir / f'trial_{args.trial:03d}.npz'
    run_trial(trial_path, trials_dir, args.t0, args.t1, args.spacing_sd, args.fmin, args.fmax)


if __name__ == '__main__':
    main()
