import json, os, time, warnings; warnings.filterwarnings('ignore')
import numpy as np
from eif_theory import *
ROOT='/Users/mathildeproust/Desktop/Cortical-Column'
d=np.load(os.path.join(fm.OUT, 'analytic', 'Nmeas.npz')); fr,N,Nf=d['fr'],d['N'],d['Nf']
runs=[r for r in fm.load('L6') if 'rate_frozen_A+' in r[0].files]; meta=runs[0][1]
rates=json.load(open(os.path.join(fm.OUT, 'L6_results.json')))['rates_hz']
cells=sample_cells(meta, rates, n=200)
res={}
for j,p in enumerate(meta['pops']):
    t=time.time()
    s0=solve(cells,p,[10.0],rates[p],1.0)
    # calibrate one scalar (noise scale) to the measured RATE only
    lo,hi=0.3,3.0
    for _ in range(18):
        mid=np.sqrt(lo*hi); r=solve(cells,p,[10.0],rates[p],mid)['r0'].mean()
        lo,hi=(mid,hi) if r<rates[p] else (lo,mid)
    sc=np.sqrt(lo*hi); s=solve(cells,p,fr,rates[p],sc)
    c=cells[p]
    chi0=s['chi_eif'].mean(0); fz=adapt(s['chi_eif'],c,fr,s['gtot'],s['tau'],'frozen').mean(0)
    fu=adapt(s['chi_eif'],c,fr,s['gtot'],s['tau'],'full').mean(0)
    mf=(Nf[:,j,0]-Nf[:,j,2])/80; ma=(N[:,j,0]-N[:,j,2])/80
    res[p]=dict(scale=sc, chi0=chi0, frozen=fz, full=fu, mf=mf, ma=ma)
    print(f'\n== {p}: Campbell-only rate {s0["r0"].mean():.2f} Hz (measured {rates[p]:.2f}); noise scale for rate match {sc:.2f}; '
          f'sigma_V {np.median(s["sig"]):.2f}->{np.median(s["sig"])*1:.2f} mV, tau_eff {np.median(s["tau"]):.1f} ms, Vbar {np.median(s["Vbar"]):.1f} (meas {meta["state_stats"][p]["v"][0]*1e3:.1f})  {time.time()-t:.0f}s')
    print('  f  | theory frozen  meas frozen | theory full  meas full   (|chi| Hz/nA, phase deg)')
    for i in [0,2,4,7,9,11,14,19,24,29,34,39]:
        f=lambda z:f'{abs(z)*1e3:6.0f} {np.degrees(np.angle(z)):5.0f}'
        print(f'{fr[i]:4.0f} | {f(fz[i])}  {f(mf[i])} | {f(fu[i])}  {f(ma[i])}')
np.save(os.path.join(fm.OUT, 'analytic', 'theory_res.npy'),res,allow_pickle=True)
