
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.signal import detrend, butter, sosfiltfilt
from scipy.signal.windows import dpss
from scipy.ndimage import zoom, gaussian_filter1d


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 110,
})


LAYER_Z = [('L23',   0.45,  1.10),
           ('L4AB',  0.14,  0.45),
           ('L4C',  -0.14,  0.14),
           ('L5',   -0.34, -0.14),
           ('L6',   -0.62, -0.34)]


# ---------------------------------------------------------------------------
# I/O & PSD helpers
# ---------------------------------------------------------------------------
def load_trials(base_path, n_trials):
    all_trials = []
    for trial_idx in range(n_trials):
        fname = f"{base_path}/trial_{trial_idx:03d}.npz"
        data = np.load(fname, allow_pickle=True)
        trial_data = {
            'time': data['time_array_ms'],
            'bipolar_lfp': data['bipolar_matrix'],
            'lfp_matrix': data['lfp_matrix'],
            'baseline_ms': float(data['baseline_ms']),
            'stim_onset_ms': float(data['stim_onset_ms']),
            'channel_labels': data['channel_labels'],
            'channel_depths': data['channel_depths'],
            'electrode_positions': data['electrode_positions'],
        }
        if 'lfp_current_matrix' in data.files:
            trial_data['lfp_current_matrix'] = data['lfp_current_matrix']
            trial_data['time_current_ms'] = data['time_current_ms']
        all_trials.append(trial_data)
    return all_trials


def highpass(x, fs, cutoff_hz=1.0, order=4):
    """Zero-phase Butterworth high-pass."""
    sos = butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def remove_aperiodic(f, psd, fit_range=(2, 80),
                     peak_drop_iters=2, peak_threshold=2.0):
    """Subtract a 1/f^chi aperiodic component from a PSD.

    Fits log10(psd) = b - chi * log10(f) over `fit_range`, iteratively
    excluding bins whose log-residual exceeds `peak_threshold` * residual
    std (these are oscillatory peaks that shouldn't bias the fit).

    Returns the residual in log10 units (i.e. log10(psd) - aperiodic_fit).
    Positive values are bumps above the 1/f trend (genuine oscillations);
    zero ≈ broadband activity at the trend.

    Works on a 1-D PSD; for multi-D arrays apply along the last axis.
    """
    psd = np.asarray(psd, dtype=float)
    if psd.ndim > 1:
        out = np.empty_like(psd)
        for idx in np.ndindex(psd.shape[:-1]):
            out[idx] = remove_aperiodic(f, psd[idx], fit_range=fit_range,
                                        peak_drop_iters=peak_drop_iters,
                                        peak_threshold=peak_threshold)
        return out

    mask = (f >= fit_range[0]) & (f <= fit_range[1]) & (psd > 0)
    if mask.sum() < 5:
        return np.zeros_like(psd)

    log_f = np.log10(f[mask])
    log_p = np.log10(psd[mask])

    keep = np.ones_like(log_f, dtype=bool)
    for _ in range(peak_drop_iters + 1):
        coefs = np.polyfit(log_f[keep], log_p[keep], 1)
        resid = log_p - np.polyval(coefs, log_f)
        thr = peak_threshold * np.std(resid[keep])
        new_keep = resid < thr      # only drop positive bumps (peaks)
        if new_keep.sum() < 5 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    log_psd = np.log10(np.where(psd > 0, psd, np.nan))
    fit_full = np.polyval(coefs, np.log10(np.where(f > 0, f, np.nan)))
    return log_psd - fit_full       # log10 residual (oscillatory power)


def multitaper_psd(data, fs, NW=2, nfft=None):
    data = data - np.mean(data)
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(data))))
    K = int(2 * NW - 1)
    tapers = dpss(len(data), NW, K)
    psds = []
    for taper in tapers:
        f, p = signal.periodogram(data * taper, fs=fs, nfft=nfft,
                                  scaling='density')
        psds.append(p)
    return f, np.mean(psds, axis=0)


def _time_vec(trial, lfp_key):
    if lfp_key in ('lfp_current_matrix', 'bipolar_lfp_current'):
        return np.asarray(trial['time_current_ms'])
    return np.asarray(trial['time'])


def get_channel_z(trial, lfp_key='bipolar_lfp'):
    """z-coordinate (mm, pia positive) of each channel.

    Bipolar: midpoint of consecutive contacts. Monopolar: contact z.
    """
    pos = np.asarray(trial['electrode_positions'])
    z = pos[:, 2]
    n_ch = trial[lfp_key].shape[0]
    if n_ch == len(z) - 1:
        return (z[:-1] + z[1:]) / 2.0
    return z[:n_ch]


def assign_layer(z):
    for name, lo, hi in LAYER_Z:
        if lo <= z < hi:
            return name
    return 'OUT'


def add_bipolar_current(all_trials):
    """Compute bipolar (consecutive-channel diff) of `lfp_current_matrix`
    in-place on each trial as `bipolar_lfp_current`. No-op if the
    monopolar synaptic-current LFP is missing or the bipolar already
    exists. Returns `all_trials` for chaining.
    """
    for trial in all_trials:
        if 'lfp_current_matrix' not in trial:
            continue
        if 'bipolar_lfp_current' in trial:
            continue
        trial['bipolar_lfp_current'] = np.diff(trial['lfp_current_matrix'],
                                               axis=0)
    return all_trials


def compute_psd_cube(all_trials, lfp_key='bipolar_lfp',
                     pre_window_ms=500, post_window_ms=500,
                     post_start_ms=200, freq_range=(1, 120),
                     do_detrend=True, hp_cutoff_hz=1.0,
                     keep_in_layer_only=True,
                     remove_aperiodic_fit=False,
                     aperiodic_fit_range=(2, 80)):
    """Return f (F,), channel_z (C,), psd_pre, psd_post (T, C, F), fs.

    Channels are deep→superficial (index 0 = deepest). Each channel is
    high-pass filtered (zero-phase) at hp_cutoff_hz before windowing — set
    to None/0 to disable.

    If keep_in_layer_only is True, channels whose z falls outside every
    layer in LAYER_Z (i.e. above pia or in white matter) are dropped.

    If remove_aperiodic_fit is True, a 1/f^chi background is fit per
    (trial, channel, condition) on log-log PSD over `aperiodic_fit_range`
    and subtracted. The returned arrays are then in **log10-power
    residual** units (not raw power) — i.e. positive values are bumps
    above the 1/f trend. Downstream %-change becomes meaningless in this
    mode; use `psd_post - psd_pre` (a difference of log10 residuals,
    which is a log-power ratio in dB/10).
    """
    t0 = _time_vec(all_trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))
    n_channels = all_trials[0][lfp_key].shape[0]
    channel_z = get_channel_z(all_trials[0], lfp_key)

    if keep_in_layer_only:
        keep = np.array([assign_layer(z) != 'OUT' for z in channel_z])
        if not keep.any():
            raise RuntimeError('No channels fall inside any defined layer.')
        channel_z = channel_z[keep]
        keep_idx = np.where(keep)[0]
    else:
        keep_idx = np.arange(n_channels)

    psd_pre_all, psd_post_all, f_out = [], [], None
    for trial in all_trials:
        time = _time_vec(trial, lfp_key)
        stim = trial['stim_onset_ms']
        pre_mask = (time >= stim - pre_window_ms) & (time < stim)
        post_mask = ((time >= stim + post_start_ms) &
                     (time < stim + post_start_ms + post_window_ms))
        psd_pre_ch, psd_post_ch = [], []
        for ch in keep_idx:
            full = trial[lfp_key][ch].astype(float)
            if hp_cutoff_hz:
                full = highpass(full, fs, cutoff_hz=hp_cutoff_hz)
            pre = full[pre_mask].copy()
            post = full[post_mask].copy()
            if len(pre) == 0 or len(post) == 0:
                continue
            if do_detrend:
                pre, post = detrend(pre), detrend(post)
            nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
            f, p_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
            _, p_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)
            psd_pre_ch.append(p_pre)
            psd_post_ch.append(p_post)
            f_out = f
        psd_pre_all.append(psd_pre_ch)
        psd_post_all.append(psd_post_ch)

    psd_pre = np.array(psd_pre_all)
    psd_post = np.array(psd_post_all)
    fmask = (f_out >= freq_range[0]) & (f_out <= freq_range[1])
    f_crop = f_out[fmask]
    psd_pre = psd_pre[..., fmask]
    psd_post = psd_post[..., fmask]

    if remove_aperiodic_fit:
        psd_pre = remove_aperiodic(f_crop, psd_pre,
                                   fit_range=aperiodic_fit_range)
        psd_post = remove_aperiodic(f_crop, psd_post,
                                    fit_range=aperiodic_fit_range)

    return f_crop, channel_z, psd_pre, psd_post, fs


def _annotate_layers(ax, color='white', fontsize=8):
    """Layer boundary lines + names on a y-axis where y = z (mm, pia=top)."""
    ymin, ymax = ax.get_ylim()
    drawn_z = set()
    for name, lo, hi in LAYER_Z:
        for z in (lo, hi):
            if z not in drawn_z and ymin <= z <= ymax:
                ax.axhline(z, color=color, lw=0.4, alpha=0.6)
                drawn_z.add(z)
        zc = (lo + hi) / 2
        if ymin <= zc <= ymax:
            xlim = ax.get_xlim()
            x_text = xlim[0] * 1.05 if ax.get_xscale() == 'log' else \
                     xlim[0] + 0.02 * (xlim[1] - xlim[0])
            ax.text(x_text, zc, name, color=color,
                    fontsize=fontsize, va='center', alpha=0.9)


# ---------------------------------------------------------------------------
# 3D surface
# ---------------------------------------------------------------------------
def plot_3d_surface(f, channel_z, psd_pre, psd_post, title='',
                    upsample_depth=8, smooth_freq=4,
                    clip_percentile=99, vmin=None, vmax=None):
    """Smooth 3D surface of % power change.

    Bipolar gives ~15 channels along depth, which makes the surface look
    blocky. We cubic-spline upsample along depth (factor `upsample_depth`)
    and apply a small box smooth along freq (`smooth_freq` bins).

    Color scale is asymmetric and data-driven: vmin/vmax default to the
    `clip_percentile` of the negative/positive tail (clips extreme outliers
    so colors aren't dominated by a few cells). Pass vmin/vmax explicitly
    to override. The diverging colormap is still centered at 0.
    """
    pct = (psd_post.mean(0) - psd_pre.mean(0)) / (psd_pre.mean(0) + 1e-12) * 100

    pct_up = zoom(pct, (upsample_depth, 1), order=3)
    z_up = np.linspace(channel_z.min(), channel_z.max(), pct_up.shape[0])
    if smooth_freq and smooth_freq > 1:
        kernel = np.ones(smooth_freq) / smooth_freq
        pct_up = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode='same'), 1, pct_up)

    if vmin is None:
        neg = pct_up[pct_up < 0]
        vmin = np.percentile(neg, 100 - clip_percentile) if neg.size else -1.0
        vmin = min(vmin, -1.0)   # avoid degenerate range when nothing decreases
    if vmax is None:
        pos = pct_up[pct_up > 0]
        vmax = np.percentile(pos, clip_percentile) if pos.size else 1.0
        vmax = max(vmax, 1.0)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    F, Z = np.meshgrid(f, z_up)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    surf = ax.plot_surface(F, Z, pct_up, cmap='RdBu_r', norm=norm,
                           edgecolor='none', alpha=0.95, antialiased=True,
                           rcount=80, ccount=80)
    ax.contour(F, Z, pct_up, zdir='z', offset=np.nanmin(pct_up) - 20,
               cmap='RdBu_r', norm=norm, levels=12)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Depth z (mm)')
    ax.set_zlabel('% change')
    ax.set_title(f'3D surface of % power change — {title} '
                 f'[{vmin:+.0f}, {vmax:+.0f}]%')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='% change')
    ax.view_init(elev=25, azim=-60)
    return fig


# ---------------------------------------------------------------------------
# Per-layer % change spectra
# ---------------------------------------------------------------------------
def plot_layer_pct_change(f, channel_z, psd_pre, psd_post, title='',
                          show_ci=True, smooth_sigma=2.5):
    """One axes: x=frequency, y=% change, one line per cortical layer.

    Channels falling inside a layer (per LAYER_Z) are averaged. Lines are
    Gaussian-smoothed along frequency (sigma in freq bins). Colors run
    superficial → deep along a custom blue → violet → pink gradient.
    """
    layer_order = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
    cmap = LinearSegmentedColormap.from_list(
        'blue_violet_pink',
        ['#2563eb',   # deep blue (superficial = L23)
         '#7c3aed',   # violet
         '#c026d3',   # magenta
         '#ec4899',   # pink (deep = L6)
         ])

    by_layer = {n: [] for n in layer_order}
    for i, z in enumerate(channel_z):
        lab = assign_layer(z)
        if lab in by_layer:
            by_layer[lab].append(i)
    present = [n for n in layer_order if by_layer[n]]
    colors = {n: cmap(i / max(len(present) - 1, 1))
              for i, n in enumerate(present)}

    def smooth(v):
        if smooth_sigma and smooth_sigma > 0:
            return gaussian_filter1d(v, smooth_sigma, mode='nearest')
        return v

    fig, ax = plt.subplots(figsize=(10, 6))

    for name in present:
        ch_idx = by_layer[name]
        pre_layer = psd_pre[:, ch_idx, :].mean(1)
        post_layer = psd_post[:, ch_idx, :].mean(1)
        pct = (post_layer - pre_layer) / (pre_layer + 1e-12) * 100

        mu = smooth(pct.mean(0))
        ax.plot(f, mu, color=colors[name], lw=2.4,
                label=f'{name}  (n={len(ch_idx)})')

        if show_ci:
            sem = smooth(pct.std(0) / np.sqrt(pct.shape[0]))
            ax.fill_between(f, mu - sem, mu + sem,
                            color=colors[name], alpha=0.15, lw=0)

    ax.axhline(0, color='k', lw=0.6, alpha=0.6)
    for x in (10, 40):
        ax.axvline(x, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power change (%)')
    ax.set_title(f'Per-layer % power change — {title}')
    ax.legend(loc='best', frameon=True, title='Layer (super → deep)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Publication summary panel
# ---------------------------------------------------------------------------
def plot_summary_panel(f, channel_z, psd_pre, psd_post,
                       bands=(('alpha', 8, 13), ('gamma', 30, 80)),
                       title=''):
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.4, 1, 1],
                  height_ratios=[1, 1], hspace=0.4, wspace=0.4)
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[0, 2], sharey=ax_b1)
    ax_sp1 = fig.add_subplot(gs[1, 1])
    ax_sp2 = fig.add_subplot(gs[1, 2], sharey=ax_sp1)

    pct = (psd_post.mean(0) - psd_pre.mean(0)) / (psd_pre.mean(0) + 1e-12) * 100
    F, Z = np.meshgrid(f, channel_z)
    cf = ax_heat.contourf(F, Z, pct, levels=21, cmap='RdBu_r',
                          norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100),
                          extend='both')
    ax_heat.contour(F, Z, pct, levels=[0], colors='k', linewidths=0.5)
    ax_heat.set_xlabel('Frequency (Hz)')
    ax_heat.set_ylabel('Cortical depth z (mm, pia = top)')
    ax_heat.set_title('% power change')
    plt.colorbar(cf, ax=ax_heat, fraction=0.04, pad=0.02, label='%')
    _annotate_layers(ax_heat, color='k')

    def band_pow(psd, lo, hi):
        m = (f >= lo) & (f <= hi)
        return np.trapz(psd[..., m], f[m], axis=-1)

    for ax, (name, lo, hi) in zip([ax_b1, ax_b2], bands):
        for arr, color, lab in [(band_pow(psd_pre, lo, hi), '#3a86ff', 'base'),
                                (band_pow(psd_post, lo, hi), '#fb5607', 'stim')]:
            mu = arr.mean(0)
            sem = arr.std(0) / np.sqrt(arr.shape[0])
            ax.fill_betweenx(channel_z, mu - sem, mu + sem,
                             color=color, alpha=0.2)
            ax.plot(mu, channel_z, color=color, lw=2, label=lab)
        ax.set_xscale('log')
        ax.set_title(f'{name} ({lo}–{hi} Hz)')
        ax.set_xlabel('Power')
        _annotate_layers(ax, color='gray')
    ax_b1.set_ylabel('Depth z (mm)')
    ax_b1.legend(fontsize=8)

    band_effects = {n: band_pow(psd_post, lo, hi).mean(0) -
                    band_pow(psd_pre, lo, hi).mean(0)
                    for n, lo, hi in bands}
    ch_decrease = int(np.argmin(band_effects[bands[0][0]]))
    ch_increase = int(np.argmax(band_effects[bands[1][0]]))
    pairs = [
        (ax_sp1, ch_decrease,
         f'strongest {bands[0][0]} ↓  {assign_layer(channel_z[ch_decrease])} '
         f'z={channel_z[ch_decrease]:+.2f}'),
        (ax_sp2, ch_increase,
         f'strongest {bands[1][0]} ↑  {assign_layer(channel_z[ch_increase])} '
         f'z={channel_z[ch_increase]:+.2f}'),
    ]
    for ax, ch, lab in pairs:
        pre_db = 10 * np.log10(psd_pre[:, ch, :] + 1e-10)
        post_db = 10 * np.log10(psd_post[:, ch, :] + 1e-10)
        for arr, color, llab in [(pre_db, '#3a86ff', 'baseline'),
                                 (post_db, '#fb5607', 'stim')]:
            mu = arr.mean(0)
            sem = arr.std(0) / np.sqrt(arr.shape[0])
            ax.fill_between(f, mu - sem, mu + sem, color=color, alpha=0.2)
            ax.plot(f, mu, color=color, lw=1.8, label=llab)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(lab, fontsize=10)
        for x in (10, 40):
            ax.axvline(x, color='gray', lw=0.4, ls='--')
    ax_sp1.set_ylabel('Power (dB)')
    ax_sp1.legend(fontsize=8)
    fig.suptitle(title, fontsize=14, y=1.0)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    base_path = 'results/trials2_pop_sweep/L6_PV'
    n_trials = 10
    title = 'L6 PV'

    all_trials = load_trials(base_path, n_trials)
    add_bipolar_current(all_trials)   # adds 'bipolar_lfp_current' if available
    print(f'Loaded {len(all_trials)} trials')

    common_kwargs = dict(pre_window_ms=1000, post_window_ms=1000,
                         post_start_ms=500, freq_range=(1, 120))

    lfp_keys = [('bipolar_lfp', 'kernel-method bipolar')]
    if 'bipolar_lfp_current' in all_trials[0]:
        lfp_keys.append(('bipolar_lfp_current', 'synaptic-current bipolar'))
    else:
        print('No lfp_current_matrix in these trials; '
              'skipping synaptic-current LFP.')

    for lfp_key, label in lfp_keys:
        print(f'\n=== {label} ({lfp_key}) ===')
        f, channel_z, psd_pre, psd_post, fs = compute_psd_cube(
            all_trials, lfp_key=lfp_key, **common_kwargs)

        print('Channels (deep → superficial):')
        for i, z in enumerate(channel_z):
            print(f'  ch{i:2d}  z={z:+.3f}  {assign_layer(z)}')

        run_title = f'{title} — {label}'
        plot_3d_surface(f, channel_z, psd_pre, psd_post, title=run_title)
        plot_layer_pct_change(f, channel_z, psd_pre, psd_post, title=run_title)
        plot_summary_panel(f, channel_z, psd_pre, psd_post, title=run_title)

    plt.show()
