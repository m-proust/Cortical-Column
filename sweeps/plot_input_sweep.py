import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

LAYER_DEPTH_RANGES = {
    'L23':  (0.45, 1.30),
    'L4AB': (0.14, 0.45),
    'L4C':  (-0.14, 0.14),
    'L5':   (-0.49, -0.14),
    'L6':   (-0.94, -0.49),
}

FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
}

BAND_COLORS = {
    'delta': '#e41a1c',
    'theta': '#377eb8',
    'alpha': '#4daf4a',
    'beta': '#984ea3',
    'gamma': '#ff7f00',
}


def add_frequency_band_lines(ax, fmax, show_labels=True):

    for band_name, (fmin_band, fmax_band) in FREQUENCY_BANDS.items():
        if fmin_band > fmax:
            continue

        color = BAND_COLORS[band_name]

        if fmin_band > 0 and fmin_band <= fmax:
            ax.axhline(y=fmin_band, color=color, linestyle='--', linewidth=0.8, alpha=0.7)

        if show_labels:
            visible_fmax = min(fmax_band, fmax)
            if visible_fmax > fmin_band:
                label_y = (fmin_band + visible_fmax) / 2
                ax.text(
                    1.02, label_y, band_name.capitalize(),
                    transform=ax.get_yaxis_transform(),
                    fontsize=6, color=color, fontweight='bold',
                    va='center', ha='left'
                )


def load_sweep_results(filepath):
    data = np.load(filepath, allow_pickle=True)

    # Handle both old format (single target) and new format (multiple targets)
    if 'target_layers' in data:
        # New format with multiple targets
        target_layers = data['target_layers']
        target_pops = data['target_pops']
        weight_scales = data['weight_scales']
        # Build targets dict and string for display
        targets = {
            (str(layer), str(pop)): float(scale)
            for layer, pop, scale in zip(target_layers, target_pops, weight_scales)
        }
        targets_str = ", ".join([
            f"{pop}_{layer}(x{scale})" for (layer, pop), scale in targets.items()
        ])
    else:
        # Old format (single target) - backward compatibility
        targets = {(str(data['target_layer']), str(data['target_pop'])): 1.0}
        targets_str = f"{data['target_pop']}_{data['target_layer']}"

    return {
        'targets': targets,
        'targets_str': targets_str,
        'rate_values': data['rate_values'],
        'frequencies': data['frequencies'],
        'psd_stack': data['psd_stack'],
        'electrode_positions': data['electrode_positions'],
    }


def get_excluded_electrode_indices(electrode_positions, exclude_layers):
    """Return set of electrode indices whose z-depth falls within any excluded layer."""
    excluded = set()
    for i, pos in enumerate(electrode_positions):
        z = pos[2]
        for layer in exclude_layers:
            layer = layer.upper()
            if layer not in LAYER_DEPTH_RANGES:
                raise ValueError(f"Unknown layer '{layer}'. Valid layers: {list(LAYER_DEPTH_RANGES.keys())}")
            zmin, zmax = LAYER_DEPTH_RANGES[layer]
            if zmin <= z <= zmax:
                excluded.add(i)
                break
    return excluded


def get_electrode_labels(electrode_positions):
    depths = [pos[2] for pos in electrode_positions]
    labels = []
    for i, z in enumerate(depths):
        labels.append(f"Elec {i+1}\n(z={z:.2f}mm)")
    return labels, depths


def plot_sweep_heatmaps(
    filepath,
    fmax=100,
    fmin=None,
    log_scale=False,
    cmap='viridis',
    save_path=None,
    figsize=(16, 14),
    show_bands=True,
    exclude_layers=None,
):
    """
    Plot power spectrum heatmaps for each electrode.

    Parameters
    ----------
    log_scale : bool
        If False (default), use linear color scaling like the paper.
        If True, use logarithmic color scaling.
    """
    data = load_sweep_results(filepath)
    targets_str = data['targets_str']
    rate_values = data['rate_values']
    frequencies = data['frequencies']
    psd_stack = data['psd_stack']
    electrode_positions = data['electrode_positions']

    freq_mask = frequencies <= fmax
    if fmin is not None:
        freq_mask &= frequencies >= fmin
    frequencies = frequencies[freq_mask]
    psd_stack = psd_stack[:, :, freq_mask]

    electrode_labels, depths = get_electrode_labels(electrode_positions)

    excluded = get_excluded_electrode_indices(electrode_positions, exclude_layers or [])
    depth_order = [i for i in np.argsort(depths) if i not in excluded]
    n_electrodes = len(depth_order)

    n_cols = 4
    n_rows = max(1, (n_electrodes + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 4))
    axes = np.array(axes).flatten()

    psd_included = psd_stack[:, list(depth_order), :]
    vmin = psd_included.min()
    vmax = psd_included.max()

    if log_scale:
        vmin = max(vmin, 1e-10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for plot_idx, elec_idx in enumerate(depth_order):
        ax = axes[plot_idx]

        psd_elec = psd_stack[:, elec_idx, :].T

        if log_scale:
            psd_elec = np.maximum(psd_elec, 1e-10)

        im = ax.imshow(
            psd_elec,
            aspect='auto',
            origin='lower',
            extent=[rate_values[0], rate_values[-1], frequencies[0], frequencies[-1]],
            cmap=cmap,
            norm=norm if log_scale else None,
            vmin=None if log_scale else vmin,
            vmax=None if log_scale else vmax,
        )

        z = depths[elec_idx]
        ax.set_title(f"Electrode {elec_idx+1} (z={z:.2f}mm)", fontsize=10)

        last_row_start = (n_electrodes - 1) // n_cols * n_cols
        if plot_idx >= last_row_start:
            ax.set_xlabel("Input rate (Hz)", fontsize=9)
        if plot_idx % n_cols == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=9)

        ax.tick_params(labelsize=8)

        if show_bands:
            show_labels = (plot_idx % n_cols == n_cols - 1) or (plot_idx == n_electrodes - 1)
            add_frequency_band_lines(ax, fmax, show_labels=show_labels)

    for idx in range(n_electrodes, len(axes)):
        axes[idx].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power (a.u.)', fontsize=11)

    fig.suptitle(
        f"Power spectra vs input rate - Stimulating {targets_str}",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    return fig


def plot_sweep_summary(
    filepath,
    freq_band=(30, 80),
    save_path=None,
    figsize=(12, 8),
):

    data = load_sweep_results(filepath)
    targets_str = data['targets_str']
    rate_values = data['rate_values']
    frequencies = data['frequencies']
    psd_stack = data['psd_stack']
    electrode_positions = data['electrode_positions']

    freq_mask = (frequencies >= freq_band[0]) & (frequencies <= freq_band[1])

    band_power = psd_stack[:, :, freq_mask].mean(axis=2)  

    n_electrodes = band_power.shape[1]
    depths = [pos[2] for pos in electrode_positions]
    depth_order = np.argsort(depths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, n_electrodes))

    for i, elec_idx in enumerate(depth_order):
        z = depths[elec_idx]
        ax1.plot(rate_values, band_power[:, elec_idx],
                 color=colors[i], label=f"z={z:.2f}mm", linewidth=1.5)

    ax1.set_xlabel("Input rate (Hz)", fontsize=11)
    ax1.set_ylabel(f"Mean power ({freq_band[0]}-{freq_band[1]} Hz)", fontsize=11)
    ax1.set_title("Band power vs input rate", fontsize=12)
    ax1.legend(fontsize=7, ncol=2, loc='upper left')
    ax1.grid(True, alpha=0.3)

    im = ax2.imshow(
        band_power.T[depth_order, :],
        aspect='auto',
        origin='lower',
        extent=[rate_values[0], rate_values[-1], 0, n_electrodes],
        cmap='hot',
    )
    ax2.set_xlabel("Input rate (Hz)", fontsize=11)
    ax2.set_ylabel("Electrode (by depth)", fontsize=11)
    ax2.set_title(f"Band power ({freq_band[0]}-{freq_band[1]} Hz)", fontsize=12)

    yticks = np.arange(0.5, n_electrodes, 1)
    ylabels = [f"{depths[depth_order[i]]:.2f}" for i in range(n_electrodes)]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=7)
    ax2.set_ylabel("Depth (mm)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Power (a.u.)', fontsize=10)

    fig.suptitle(
        f"Summary: Stimulating {targets_str}",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    return fig


def plot_sweep_difference(
    filepath,
    fmax=100,
    fmin=None,
    baseline_idx=0,
    cmap='RdBu_r',
    save_path=None,
    figsize=(16, 14),
    show_bands=True,
    exclude_layers=None,
):

    data = load_sweep_results(filepath)
    targets_str = data['targets_str']
    rate_values = data['rate_values']
    frequencies = data['frequencies']
    psd_stack = data['psd_stack']
    electrode_positions = data['electrode_positions']

    freq_mask = frequencies <= fmax
    if fmin is not None:
        freq_mask &= frequencies >= fmin
    frequencies = frequencies[freq_mask]
    psd_stack = psd_stack[:, :, freq_mask]

    baseline_psd = psd_stack[baseline_idx, :, :]
    psd_diff = psd_stack - baseline_psd[np.newaxis, :, :]
    electrode_labels, depths = get_electrode_labels(electrode_positions)

    excluded = get_excluded_electrode_indices(electrode_positions, exclude_layers or [])
    depth_order = [i for i in np.argsort(depths) if i not in excluded]
    n_electrodes = len(depth_order)

    n_cols = 4
    n_rows = max(1, (n_electrodes + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 4))
    axes = np.array(axes).flatten()

    psd_diff_included = psd_diff[:, list(depth_order), :]
    max_abs = np.abs(psd_diff_included).max()
    vmin, vmax = -max_abs, max_abs

    for plot_idx, elec_idx in enumerate(depth_order):
        ax = axes[plot_idx]

        psd_diff_elec = psd_diff[:, elec_idx, :].T

        im = ax.imshow(
            psd_diff_elec,
            aspect='auto',
            origin='lower',
            extent=[rate_values[0], rate_values[-1], frequencies[0], frequencies[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        z = depths[elec_idx]
        ax.set_title(f"Electrode {elec_idx+1} (z={z:.2f}mm)", fontsize=10)

        last_row_start = (n_electrodes - 1) // n_cols * n_cols
        if plot_idx >= last_row_start:
            ax.set_xlabel("Input rate (Hz)", fontsize=9)
        if plot_idx % n_cols == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=9)

        ax.tick_params(labelsize=8)

        if show_bands:
            show_labels = (plot_idx % n_cols == n_cols - 1) or (plot_idx == n_electrodes - 1)
            add_frequency_band_lines(ax, fmax, show_labels=show_labels)

    for idx in range(n_electrodes, len(axes)):
        axes[idx].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power difference (a.u.)', fontsize=11)

    baseline_rate = rate_values[baseline_idx]
    fig.suptitle(
        f"Power spectra difference from baseline ({baseline_rate:.1f} Hz) - "
        f"Stimulating {targets_str}",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot input sweep results")
    parser.add_argument("filepath", type=str, help="Path to sweep results .npz file")
    parser.add_argument("--fmax", type=float, default=100, help="Max frequency to display")
    parser.add_argument("--high-pass", type=float, default=None, metavar="FREQ",
                        help="High-pass filter: only plot frequencies >= FREQ Hz (e.g., --high-pass 7)")
    parser.add_argument("--log", action="store_true", help="Use logarithmic (not linear) color scale")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap")
    parser.add_argument("--save", type=str, default=None, help="Save path for figure")
    parser.add_argument("--summary", action="store_true", help="Also plot summary figure")
    parser.add_argument("--band", type=float, nargs=2, default=[30, 80], help="Frequency band for summary")
    parser.add_argument("--diff", action="store_true", help="Also plot difference from baseline")
    parser.add_argument("--baseline-idx", type=int, default=0, help="Index of baseline rate for difference plot")
    parser.add_argument("--no-bands", action="store_true", help="Hide frequency band annotations")
    parser.add_argument(
        "--exclude", type=str, nargs="+", default=[], metavar="LAYER",
        help="Exclude electrodes from these layers (e.g. --exclude L23 L4AB). "
             f"Valid layers: {list(LAYER_DEPTH_RANGES.keys())}"
    )

    args = parser.parse_args()

    plot_sweep_heatmaps(
        args.filepath,
        fmax=args.fmax,
        fmin=args.high_pass,
        log_scale=args.log,
        cmap=args.cmap,
        save_path=args.save,
        show_bands=not args.no_bands,
        exclude_layers=args.exclude,
    )

    if args.summary:
        summary_save = None
        if args.save:
            base, ext = os.path.splitext(args.save)
            summary_save = f"{base}_summary{ext}"

        plot_sweep_summary(
            args.filepath,
            freq_band=tuple(args.band),
            save_path=summary_save,
        )

    if args.diff:
        diff_save = None
        if args.save:
            base, ext = os.path.splitext(args.save)
            diff_save = f"{base}_diff{ext}"

        plot_sweep_difference(
            args.filepath,
            fmax=args.fmax,
            fmin=args.high_pass,
            baseline_idx=args.baseline_idx,
            save_path=diff_save,
            show_bands=not args.no_bands,
            exclude_layers=args.exclude,
        )
