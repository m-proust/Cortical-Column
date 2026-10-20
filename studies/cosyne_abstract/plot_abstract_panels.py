"""Cosyne abstract: draw the model panels of Figs. 2 and 3 with matplotlib, from the tables written by
abstract_data.py. A quick way to look at the results without LaTeX; the abstract itself draws the same
tables with pgfplots (~/Desktop/cosyne_abstract/cosyne_abstract.tex), with the same colours and axis ranges.

    ~/Desktop/venv/bin/python studies/cosyne_abstract/abstract_data.py          # tables -> figures/cosyne_abstract/data
    ~/Desktop/venv/bin/python studies/cosyne_abstract/plot_abstract_panels.py   # -> figures/cosyne_abstract/fig2_model.png, fig3_model.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D = os.path.join(ROOT, 'figures', 'cosyne_abstract', 'data')
OUT = os.path.join(ROOT, 'figures', 'cosyne_abstract')
rA, rB, rC, rD, rE, GREY, BAND = '#002255', '#2F6DB5', '#8FBCE6', '#FFB27A', '#FF6D1E', '#8C8C8C', '#EBEBEB'
SOM, PV = '#1F4E96', '#C0392B'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif',
                     'legend.frameon': False, 'legend.fontsize': 6.5})


def tab(name):
    with open(os.path.join(D, name)) as fh:
        head = fh.readline().split()
    a = np.loadtxt(os.path.join(D, name), skiprows=1)
    return {h: a[:, i] for i, h in enumerate(head)}


def spectrum_axes(ax, fmax, title, ylabel='E-rate power (dB)'):
    ax.axvspan(8, 13, color=BAND, lw=0, zorder=0)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set(xscale='log', xlim=(1.5, fmax), xlabel='Frequency (Hz)', ylabel=ylabel, title=title)
    ticks = [t for t in (2, 5, 10, 20, 40, 80) if t <= fmax]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()


def fig2():
    fig, ax = plt.subplots(1, 5, figsize=(12, 2.6), layout='constrained')
    t = tab('iso_lfp.dat')
    for k, c, lab, lw in [('L23', rA, 'L2/3', 1), ('L4AB', rB, 'L4AB', 1), ('L4C', rC, 'L4C', 1), ('L5', rD, 'L5', 1.6),
                          ('L6', rE, 'L6', 1.6)]:
        ax[0].plot(t['f'], t[k], color=c, lw=lw, label=lab)
    spectrum_axes(ax[0], 80, 'A  Isolated layers', 'LFP power (dB above 1/f)'); ax[0].set_ylim(-5, 15); ax[0].legend()
    t = tab('reduced.dat')
    for k, c, lab in [('ESOM', SOM, 'E + SOM'), ('ESOMPV', PV, 'E + SOM + PV'), ('EPV', GREY, 'E + PV')]:
        ax[1].plot(t['f'], t[k], color=c, label=lab)
    spectrum_axes(ax[1], 40, 'B  Reduced L6 circuits'); ax[1].set_ylim(-16, 48); ax[1].legend()
    t = tab('transplant.dat')
    ax[2].plot(t['f'], t['alone'], color=GREY, label='alone')
    ax[2].plot(t['f'], t['swap'], color=rE, label='+ L6 PV→E, SOM→PV')
    spectrum_axes(ax[2], 40, 'C  Layer 2/3'); ax[2].set_ylim(-8, 13); ax[2].legend()
    t = tab('finite.dat')
    ax[3].plot(t['N'], t['invN'], '--', color=GREY, lw=0.8, label='∝ 1/N')
    for k, c in zip(['s0', 's2', 's4', 's11', 's20'], [rA, rB, rC, rD, rE]):
        ax[3].plot(t['N'], t[k], 'o-', color=c, ms=2.5, label=k[1:] + ' % shared')
    ax[3].set(xscale='log', yscale='log', xlim=(800, 20000), ylim=(4e-4, 0.2), xlabel='Layer-6 E cells',
              ylabel='α power (Hz²/Hz)', title='D  Shared input'); ax[3].legend(fontsize=5.5)
    t = tab('freq_theory.dat')
    for tau, c in [(10, rA), (20, rB), (40, rE)]:
        ax[4].plot(t['f'], t[f'net{tau}'], color=c, label=f'{tau} ms')
        ax[4].plot(t['f'], t[f'th{tau}'], '--', color=c)
    spectrum_axes(ax[4], 40, 'E  SOM→E decay', 'E-rate power (norm.)'); ax[4].set(xlim=(4, 40), ylim=(0, 1.3))
    ax[4].plot([], [], 'k-', label='network'); ax[4].plot([], [], 'k--', label='linear theory'); ax[4].legend(fontsize=5.5)
    fig.savefig(os.path.join(OUT, 'fig2_model.png'), dpi=200); plt.close(fig)


def phase_axes(ax, xlabel, ylabel, title):
    ax.set(xlim=(-180, 180), xticks=[-180, -90, 0, 90, 180], xlabel=xlabel, ylabel=ylabel, title=title)


def fig3():
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.6), layout='constrained')
    for a, name, title in [(ax[0], 'replay_L4C.dat', 'A  L4C driven by L6'), (ax[1], 'replay_L23.dat', 'A  L2/3 driven by L4C')]:
        t = tab(name)
        a.plot(t['f'], t['none'], color=GREY, label='no input')
        a.plot(t['f'], t['poisson'], color=rA, label='random timing')
        a.plot(t['f'], t['replay'], color=rE, label='real spikes')
        spectrum_axes(a, 40, title); a.set_ylim(-13, 9)
    ax[1].legend()
    t = tab('gating_amp.dat')
    for k, c, ls, lab in [('weak', GREY, '--', 'weak α'), ('strong', rE, '-', 'strong α')]:
        ax[2].fill_between(t['deg'], t[k + 'lo'], t[k + 'hi'], color=c, alpha=0.25, lw=0)
        ax[2].plot(t['deg'], t[k], ls, color=c, label=lab)
    phase_axes(ax[2], 'Local α phase (deg)', 'L4C response (%)', 'B  Ongoing α amplitude'); ax[2].set_ylim(-24, 30); ax[2].legend()
    t = tab('gating_fb.dat')
    for k, c, lab in [('off', rA, 'feedback off'), ('on', rE, 'feedback on')]:
        ax[3].fill_between(t['deg'], t[k + 'lo'], t[k + 'hi'], color=c, alpha=0.2, lw=0)
        ax[3].plot(t['deg'], t[k], color=c, label=lab)
    phase_axes(ax[3], 'Layer-6 α phase (deg)', 'L2/3 response (Hz)', 'D  Feedback, layer 2/3'); ax[3].set_ylim(1.05, 1.85)
    ax[3].legend()
    fig.savefig(os.path.join(OUT, 'fig3_model.png'), dpi=200); plt.close(fig)


if __name__ == '__main__':
    fig2(); fig3()
    print('saved', os.path.join(OUT, 'fig2_model.png'), 'and fig3_model.png')
