"""Fixed analyses for original-profile feedback replication; network-level summaries."""
from pathlib import Path
import os,json
import numpy as np
from scipy.signal import detrend,welch
os.environ.setdefault('MPLCONFIGDIR',str(Path.home()/'.cache'/'mpl_feedback_protocol'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom,gaussian_filter
from laminar_power_change_snapshot import multitaper_psd
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'saved_trials/cosyne/feedback_protocol'   # data in, metrics.json / RESULTS.md / arrays out
FIG=REPO/'figures/cosyne/feedback_protocol';FIG.mkdir(parents=True,exist_ok=True)
SEEDS=list(range(920101,920107))

def fit(y,phase):
    b=np.linalg.lstsq(np.c_[np.ones(len(y)),np.cos(phase),np.sin(phase)],y,rcond=None)[0]
    if not np.isfinite(b).all():raise RuntimeError('Nonfinite phase fit')
    amp=float(np.hypot(b[1],b[2]))
    return dict(mean=float(b[0]),cos=float(b[1]),sin=float(b[2]),amplitude=amp,
                relative_percent=100*amp/abs(b[0]) if abs(b[0])>.05 else None)

def summary(x):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(81820)
    ci=np.quantile(x[rng.integers(0,len(x),(5000,len(x)))].mean(1),[.025,.975]).tolist() if len(x)>1 else None
    return dict(n=len(x),mean=float(x.mean()),ci95=ci,per_network=x.tolist())

def getpair(kind,seed):
    paths=[OUT/'data'/f'{kind}_{c}_seed{seed}.npz' for c in [0,1]]
    if not all(p.exists() for p in paths):return None
    a,b=[np.load(p) for p in paths]
    assert str(a['wiring_digest'])==str(b['wiring_digest'])
    assert str(a['baseline_digest'])==str(b['baseline_digest'])
    return a,b

def spectral():
    pairs=[(s,getpair('spectrum',s)) for s in SEEDS]
    pairs=[(s,p) for s,p in pairs if p is not None]
    results=dict(seeds=[s for s,p in pairs],signals={})
    if not pairs:return results
    for field,dtfield,label in [('bipolar_kernel_uv','kernel_dt_ms','Kernel LFP'),('bipolar_lfp_uv','lfp_dt_ms','Synaptic-current LFP')]:
        powers=[]
        for seed,pair in pairs:
            arms=[]
            for z in pair:
                dt=float(z[dtfield]);fs=1000/dt
                windows=[]
                for start in [1500,3000]:
                    x=z[field][:,int(round(start/dt)):int(round((start+500)/dt))]
                    ps=[]
                    for trace in x:
                        f,p=multitaper_psd(detrend(trace.astype(float)),fs=fs,NW=2)
                        if not np.isfinite(p).all():raise RuntimeError('Nonfinite spectral estimate')
                        ps.append(p)
                    windows.append(ps)
                arms.append(windows)
            powers.append(arms)
        p=np.asarray(powers) # seed, off/on, pre/post, channel, frequency
        changes=10*np.log10((p[:,:,1]+1e-20)/(p[:,:,0]+1e-20))
        effect=changes[:,1]-changes[:,0]
        pos=pairs[0][1][0]['electrode_positions_mm'][:,2]*1000
        depth=(pos[:-1]+pos[1:])/2;order=np.argsort(depth)[::-1]
        mask=(f>=0)&(f<=120)
        maps=[changes[:,0].mean(0),changes[:,1].mean(0),effect.mean(0)]
        v=max(1,float(np.percentile(np.abs(np.array(maps)[:,:,mask]),98)))
        for smooth,suffix in [(False,'raw'),(True,'smoothed')]:
            fig,axs=plt.subplots(1,3,figsize=(15,6),layout='constrained')
            for ax,m,title in zip(axs,maps,['Feedforward maintained: post/pre','Feedback added: post/pre','Feedback effect: difference of changes']):
                arr=m[order][:,mask]
                if smooth:arr=gaussian_filter(zoom(arr,(12,1),order=3),(.8,1.2))
                im=ax.imshow(arr,origin='upper',aspect='auto',extent=[f[mask][0],f[mask][-1],depth.min()-75,depth.max()+75],cmap='RdBu_r',vmin=-v,vmax=v)
                ax.set(title=title,xlabel='Frequency (Hz)',ylabel='Model z (µm; superficial at top)')
                for edge in [-620,-340,-140,140,360,800]:ax.axhline(edge,color='k',lw=.4,ls='--')
                for hz in [8,13]:ax.axvline(hz,color='k',lw=.6,ls=':')
                fig.colorbar(im,ax=ax,label='Power change (dB)',shrink=.8)
            fig.suptitle(f'{label}: {len(pairs)} paired networks • feedback added at 2000 ms\nPre 1500–2000 ms; post 3000–3500 ms; original multitaper estimator')
            fig.savefig(FIG/f'{field}_{suffix}.png',dpi=160);plt.close(fig)
        band=(f>=8)&(f<=13)
        bp=np.trapezoid(p[...,band],f[band],axis=-1)
        bandchange=10*np.log10((bp[:,:,1]+1e-20)/(bp[:,:,0]+1e-20))
        bandeffect=bandchange[:,1]-bandchange[:,0]
        layers={}
        for L,lo,hi in [('L23',360,800),('L4A',140,360),('L4C',-140,140),('L5',-340,-140),('L6',-620,-340)]:
            idx=(depth>=lo)&(depth<hi)
            layers[L]=dict(feedback_post_pre_db=summary(bandchange[:,1,idx].mean(1)),
                           control_post_pre_db=summary(bandchange[:,0,idx].mean(1)),
                           feedback_specific_db=summary(bandeffect[:,idx].mean(1)))
        results['signals'][field]=layers
        np.savez_compressed(OUT/f'{field}_spectral_arrays.npz',power=p,frequency_hz=f,depth_um=depth,changes_db=changes,effect_db=effect,seeds=results['seeds'])
    return results

def probes():
    rows=[];seeds=[];alpha=[]
    for seed in SEEDS:
        pair=getpair('probe',seed)
        if pair is None:continue
        seeds.append(seed)
        for c,z in enumerate(pair):
            layers=list(z['layers'])
            for L in ['L4C','L23']:
                li=layers.index(L);ci=int(np.flatnonzero(z['condition_targets']==L)[0])
                counts=z['counts'][:,:,li,:15].sum(-1)/(int(z['n_e'][li])*.015)
                response=counts[:,ci]-counts[:,0]
                rows.append(dict(seed=seed,feedback=c,layer=L,source=fit(response,z['phase_L6']),local=fit(response,z[f'phase_{L}']),raw_mean=float(response.mean())))
            f,p=welch(z['rates_hz'][:,0,4000:],fs=1000,nperseg=2000,axis=-1)
            band=(f>=7)&(f<=16)
            alpha.append(dict(seed=seed,feedback=c,layers=layers,rate_alpha=np.trapezoid(p[:,band],f[band],axis=-1).tolist(),rate_mean=z['rates_hz'][:,0,4000:].mean(-1).tolist()))
    result=dict(seeds=seeds,responses=rows,population=alpha,changes={})
    if not seeds:return result
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,L in enumerate(['L4C','L23']):
        result['changes'][L]={}
        for col,key in enumerate(['mean','amplitude','relative_percent']):
            differences=[]
            for seed in seeds:
                vals=[next(r for r in rows if r['seed']==seed and r['feedback']==c and r['layer']==L)['source'][key] for c in [0,1]]
                if any(x is None for x in vals):continue
                differences.append(vals[1]-vals[0]);axs[row,col].plot([0,1],vals,'o-',alpha=.7)
            result['changes'][L][key]=summary(differences)
            axs[row,col].set(xticks=[0,1],xticklabels=['Feedforward','+ feedback'],title=f'{L}: {key}',ylabel='%' if key=='relative_percent' else 'Evoked Hz/cell')
    fig.suptitle(f'25 pA direct probes: {len(seeds)} paired networks; no rate matching\nPhase reference: each condition’s L6 rate; 96 onsets per network; 0–15 ms pulse − sham')
    fig.savefig(FIG/'feedback_excitability.png',dpi=170);plt.close(fig)
    return result

def main():
    s=spectral();p=probes()
    result=dict(spectral=s,probes=p)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    lines=['# Feedforward maintained + feedback: replication','',f"Completed: {len(s['seeds'])}/6 spectral pairs; {len(p['seeds'])}/6 excitability pairs.",'',
        'This is an ongoing, fixed-size replication. Intermediate results are descriptive. Feedforward begins at 0 ms and remains on. Feedback is added at 2000 ms using the original builders. Control receives feedforward throughout. Baseline states/spikes and wiring must match exactly within every pair. No rate matching.','',
        'The spectral map uses the original 500 ms pre window and 500 ms post window beginning 1000 ms after feedback onset, NW=2 DPSS, linear detrending, and mean per-network log power change. Kernel LFP uses the original 0.1 ms grid; the complementary current signal uses 1 ms online samples. Smoothed plots are display-only; network seeds are the replication unit. Kernel convolution may mix activity across the immediate onset boundary.','',
        'The feedback-specific contrast subtracts the control post/pre change from the feedback post/pre change. Alpha integrates available 8–13 Hz bins; this short-window estimator has broad spectral smoothing. Longer population-rate spectra from probe trajectories provide a complementary 7–16 Hz measure.','']
    for signal,ls in s['signals'].items():
        lines += [f'## {signal}: 8–13 Hz feedback-specific change','', '| Layer | Mean change (dB) | Network-bootstrap 95% interval |','|---|---:|---|']
        for L,x in ls.items():
            r=x['feedback_specific_db'];lines.append(f"| {L} | {r['mean']:+.3f} | {r['ci95']} |")
    if p['changes']:
        lines+=['','## Excitability: feedback minus control','']
        for L,x in p['changes'].items():
            for key,r in x.items():lines.append(f"- {L} {key}: {r['mean']:+.4f}; interval {r['ci95']}.")
    lines+=['','Direct current pulses measure incremental excitability during sustained input; they are not additional visual stimuli. Absolute and relative phase modulation are both reported. Positive fitted amplitudes have finite-sample bias. The result cannot by itself establish that alpha mediates feedback suppression; inhibition can change both. No new runs are triggered by a desirable or undesirable effect.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(f"Updated: {len(s['seeds'])} spectral pairs, {len(p['seeds'])} probe pairs",flush=True)

if __name__=='__main__':main()
