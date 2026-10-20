"""Original feedback profile: sustained feedforward, feedback added at 2000 ms.

Branches share state and an explicit Brian seed; baseline continuation is sham.
L4C/L23 each receive 10 ms current pulses at three strengths from the SAME onsets.
"""
from pathlib import Path
import argparse, importlib.util, json, sys, time
HERE=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pilot_base',HERE/'run_pilot.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
import brian2 as b2
from scipy.signal import butter,sosfiltfilt,hilbert

OUT=REPO/'saved_trials/cosyne/feedback_protocol'  # moved from the Codex folder outputs/feedback_protocol
sys.path.insert(0,str(HERE))
from stim_profiles_snapshot import feedback_input, feedforward_input
from lfp_kernel_snapshot import calculate_lfp_kernel_method
LAYERS=base.LAYERS
TARGETS=['L4C','L23']
STATE_VARS=['v','gE_AMPA','gE_NMDA','gPV','gSOM','gVIP']
STATE_UNITS=[b2.mV,b2.nS,b2.nS,b2.nS,b2.nS,b2.nS]


def branch(column,target,dose,onset):
    group=None if target=='sham' else column.layers[target].neuron_groups['E']
    if group is not None: group.I=dose*b2.pA
    column.network.run(10*b2.ms)
    if group is not None: group.I=0*b2.pA
    column.network.run(50*b2.ms)
    counts=np.zeros((len(LAYERS),60),np.int32)
    for li,L in enumerate(LAYERS):
        t=np.asarray(column.layers[L].monitors['E_spikes'].t/b2.ms)
        t=t[np.searchsorted(t,onset):]-onset
        counts[li]=np.histogram(t,np.arange(61))[0]
    return counts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spectral-only',action='store_true')
    ap.add_argument('--seed',type=int,default=920081)
    ap.add_argument('--onsets',type=int,default=12)
    ap.add_argument('--tag',default='calibration')
    ap.add_argument('--doses',default='10,25,50')
    ap.add_argument('--feedback',type=int,choices=[0,1],default=1)
    ap.add_argument('--baseline-only',action='store_true',help='Replay the unpulsed RNG schedule and record exact bipolar current LFP online')
    ap.add_argument('--donor',default=None,help='Unpulsed intact source NPZ for L6 outgoing replay')
    ap.add_argument('--shift-ms',type=float,default=0.,help='Circular shift of all outgoing L6 replay')
    ap.add_argument('--l23-extra-shift-ms',type=float,default=0.)
    args=ap.parse_args()
    doses=[float(v) for v in args.doses.split(',')]
    conditions=[('sham',0.)]+[(L,d) for L in TARGETS for d in doses]
    stem=f'{args.tag}_seed{args.seed}'
    dest=OUT/'data'/f'{stem}.npz'
    if dest.exists():raise FileExistsError(dest)
    statusfile=OUT/'logs'/f'{stem}.json'
    start=time.time()
    status=dict(status='building',**vars(args),started_unix=start)
    base.atomic_json(statusfile,status)
    b2.start_scope();b2.seed(args.seed);b2.defaultclock.dt=base.CONFIG['simulation']['DT']
    col=base.CorticalColumn(column_id=0,config=base.CONFIG)
    for layer in col.layers.values():base.add_heterogeneity_to_layer(layer,base.CONFIG)
    net=col.network
    ampa=base.CONFIG['synapses']['Q']['EXT_AMPA'];nmda=base.CONFIG['synapses']['Q']['EXT_NMDA']
    b2.seed(args.seed+2)  # trials.py trial_id=0 noise-seed convention
    feedforward_objects=feedforward_input(col,ampa,nmda)
    net.add(*feedforward_objects)
    if args.spectral_only:
        args.baseline_only=True
        args.onsets=0
    status.update(vars(args))
    base.atomic_json(statusfile,status)
    for layer in col.layers.values():
        for name,mon in layer.monitors.items():
            if not name.endswith('_spikes'):net.remove(mon)
    # Representative, predetermined neurons; full-population moments at onsets below.
    monitors={}
    for L in ['L6','L4C','L23']:
        g=col.layers[L].neuron_groups['E']
        ids=np.unique(np.linspace(0,len(g)-1,32,dtype=int))
        monitors[L]=b2.StateMonitor(g,STATE_VARS,record=ids,dt=1*b2.ms,name=f'direct_state_{L}')
        net.add(monitors[L])
    digest=base.fingerprint(col,[])
    donor=np.load(args.donor) if args.donor else None
    replay_metadata=[]
    if donor is not None:
        if int(donor['network_seed'])!=args.seed:raise ValueError('Donor must match network seed')
        if str(donor['wiring_digest'])!=digest:raise ValueError('Donor wiring mismatch')
        period=float(donor['rates_hz'].shape[-1])
        for name,original in col.inter_layer_synapses.items():
            src,target,pre,post=name.split('_')
            if src!='L6':continue
            shift=args.shift_ms+(args.l23_extra_shift_ms if target=='L23' else 0.)
            times=np.asarray(donor[f'traj_L6_{pre}_t_ms'],dtype=float)
            ids=np.asarray(donor[f'traj_L6_{pre}_i'],dtype=int)
            dt=float(b2.defaultclock.dt/b2.ms)
            shifted=np.mod(np.round((times+shift)/dt)*dt,period)
            order=np.argsort(shifted,kind='stable')
            gen=b2.SpikeGeneratorGroup(len(col.layers['L6'].neuron_groups[pre]),ids[order],shifted[order]*b2.ms,
                                      sorted=True,name=f'replay_source_{name}')
            syn=b2.Synapses(gen,col.layers[target].neuron_groups[post],on_pre=original.pre.code,
                            name=f'replay_connection_{name}')
            syn.connect(i=np.asarray(original.i[:]),j=np.asarray(original.j[:]))
            syn.delay=original.delay[:]
            original.active=False
            net.add(gen,syn)
            replay_metadata.append(dict(pathway=name,shift_ms=shift,synapses=len(syn),
                                        source_spikes=len(ids),on_pre=original.pre.code))
    rng=np.random.default_rng(args.seed+8821)
    onsets=np.r_[4000.,4000+np.cumsum(rng.integers(75,111,args.onsets-1))] if args.onsets else np.array([])
    end_ms=max(4000.,onsets[-1]+1000.) if len(onsets) else 4000.
    lfp=None
    if args.baseline_only:
        # Same point-source equation and sign convention as tools/lfp_current_method.py.
        # Online summation avoids retaining all-neuron current histories in memory.
        electrode=np.array(base.CONFIG['electrode_positions'])*1e-3
        weighted_groups=[]
        for layer in col.layers.values():
            for g in layer.neuron_groups.values():
                xyz=np.column_stack([np.asarray(getattr(g,v)/b2.meter) for v in ['x','y','z']])
                dist=np.maximum(np.linalg.norm(electrode[:,None,:]-xyz[None,:,:],axis=-1),10e-6)
                weights=2.30/(4*np.pi)*1e6*np.diff(1/dist,axis=0)
                state={v:g.variables[v].get_value() for v in ['v','gE_AMPA','gE_NMDA','gPV','gSOM','gVIP']}
                weighted_groups.append((state,weights,float(g.namespace['Ee']),float(g.namespace['Ei'])))
        lfp=np.full((len(electrode)-1,int(end_ms)),np.nan)
        def record_current(t):
            idx=int(round(float(t/b2.ms)))
            if idx>=lfp.shape[1]:return
            value=np.zeros(lfp.shape[0])
            # Direct arrays avoid Brian subexpression evaluation during an active run.
            # All arrays are SI-valued; -0.062*v/mV is -62*v with v in volts.
            with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
                for state,w,ee,ei in weighted_groups:
                    v=state['v']
                    excitation=(state['gE_AMPA']+state['gE_NMDA']/(1+.28*np.exp(-62*v)))*(ee-v)
                    inhibition=(state['gPV']+state['gSOM']+state['gVIP'])*(v-ei)
                    value+=w@(excitation+inhibition)
            if not np.isfinite(value).all():raise RuntimeError(f'Nonfinite current sample at {idx} ms')
            lfp[:,idx]=value
        recorder=b2.NetworkOperation(record_current,dt=1*b2.ms,when='start',order=1,name='feedback_current_recorder')
        net.add(recorder)
    if donor is not None:
        if not np.array_equal(onsets,donor['onsets_ms']):raise ValueError('Donor onset schedule mismatch')
        if end_ms>period+.01:raise ValueError('Donor trajectory is too short')
    counts=np.zeros((args.onsets,len(conditions),len(LAYERS),60),np.int32)
    moments=np.zeros((args.onsets,3,len(STATE_VARS),2))
    checks=[]
    net.run(2000*b2.ms)
    baseline_digest=base.snapshot_end(col)
    if args.feedback:
        feedback_objects=feedback_input(col,ampa,nmda)
        net.add(*feedback_objects)
    if len(onsets):net.run((onsets[0]-2000)*b2.ms)
    print(f'Built and settled in {time.time()-start:.1f}s',flush=True)
    for k,on in enumerate(onsets):
        now=float(net.t/b2.ms)
        if on>now+1e-6:net.run((on-now)*b2.ms)
        for li,L in enumerate(['L6','L4C','L23']):
            g=col.layers[L].neuron_groups['E']
            for vi,(var,unit) in enumerate(zip(STATE_VARS,STATE_UNITS)):
                x=np.asarray(getattr(g,var)[:]/unit)
                moments[k,li,vi]=[x.mean(),x.std()]
        if args.baseline_only:
            rseed=int(np.random.SeedSequence([args.seed,k,205]).generate_state(1)[0])
            b2.seed(rseed)
            if k in {0,len(onsets)-1}:net.run(60*b2.ms)
            status.update(status='recording baseline currents',completed_onsets=k+1,elapsed_seconds=time.time()-start,wiring_digest=digest)
            base.atomic_json(statusfile,status)
            continue
        net.store('pre')
        rseed=int(np.random.SeedSequence([args.seed,k,205]).generate_state(1)[0])
        def restore():
            net.restore('pre',restore_random_state=False);b2.seed(rseed)
        sham_signature=None
        for ci,(L,dose) in enumerate(conditions):
            restore();counts[k,ci]=branch(col,L,dose,on)
            if ci==0 and k in {0,len(onsets)-1}:sham_signature=base.snapshot_end(col)
        restore()
        if sham_signature is not None:
            replica=branch(col,'sham',0,on)
            passed=base.snapshot_end(col)==sham_signature and np.array_equal(replica,counts[k,0])
            if not passed:raise RuntimeError('60ms sham/continuation mismatch')
            checks.append(dict(onset=int(k),exact_sham_and_continuation=True))
        status.update(status='simulating',completed_onsets=k+1,elapsed_seconds=time.time()-start,
                      wiring_digest=digest,checks=checks)
        base.atomic_json(statusfile,status)
        np.savez_compressed(OUT/'data'/f'{stem}_partial.npz',counts=counts[:k+1],
            onsets_ms=onsets[:k+1],condition_targets=np.array([v[0] for v in conditions]),
            condition_doses_pa=np.array([v[1] for v in conditions]))
        print(f'{stem}: onset {k+1}/{len(onsets)}; {time.time()-start:.1f}s elapsed',flush=True)
    net.run((end_ms-float(net.t/b2.ms))*b2.ms)
    duration=int(end_ms)
    rates=np.zeros((len(LAYERS),4,duration),np.float32)
    extra=dict(baseline_digest=baseline_digest,feedback_onset_ms=2000.,feedforward_enabled=True,stim_seed=args.seed+2)
    if args.spectral_only:
        spikes={L:col.layers[L].monitors for L in LAYERS}
        groups={L:col.layers[L].neuron_groups for L in LAYERS}
        kernel,t=calculate_lfp_kernel_method(spikes,groups,base.CONFIG['layers'],base.CONFIG['electrode_positions'],dt_ms=.1,sim_duration_ms=end_ms)
        extra['bipolar_kernel_uv']=np.diff(kernel,axis=0).astype(np.float32)
        extra['kernel_dt_ms']=.1
    if lfp is not None:
        if not np.isfinite(lfp).all():raise RuntimeError(f'Missing/nonfinite current LFP samples: {np.sum(~np.isfinite(lfp))}; first times {np.flatnonzero(~np.isfinite(lfp).all(0))[:10]}')
        extra['bipolar_lfp_uv']=lfp.astype(np.float32)
        extra['lfp_dt_ms']=1.
        extra['electrode_positions_mm']=np.array(base.CONFIG['electrode_positions'])
    if donor is not None:
        donor_rate=donor['rates_hz'][LAYERS.index('L6'),0]
        analytic=hilbert(sosfiltfilt(butter(4,[7,16],fs=1000,btype='bandpass',output='sos'),donor_rate-donor_rate.mean()))
        extra['phase_source']=np.angle(analytic)[onsets.astype(int)]
        extra['amp_source']=np.abs(analytic)[onsets.astype(int)]
        extra['replay_metadata_json']=json.dumps(replay_metadata)
    for li,L in enumerate(LAYERS):
        for pi,p in enumerate(base.POPS):
            mon=col.layers[L].monitors[p+'_spikes'];t=np.asarray(mon.t/b2.ms)
            n=len(col.layers[L].neuron_groups[p])
            rates[li,pi]=np.bincount(t.astype(int),minlength=duration)[:duration]*1000/n
            # All unpulsed spike identities are retained for subsequent replay experiments.
            extra[f'traj_{L}_{p}_t_ms']=t.astype(np.float32)
            extra[f'traj_{L}_{p}_i']=np.asarray(mon.i).astype(np.int32)
        sig=hilbert(sosfiltfilt(butter(4,[7,16],fs=1000,btype='bandpass',output='sos'),rates[li,0]-rates[li,0].mean()))
        extra[f'phase_{L}']=np.angle(sig)[onsets.astype(int)]
        extra[f'amp_{L}']=np.abs(sig)[onsets.astype(int)]
        if L in monitors:
            extra[f'state_{L}']=np.array([np.asarray(getattr(monitors[L],v)/unit).mean(0)
                        for v,unit in zip(STATE_VARS,STATE_UNITS)],dtype=np.float32)
    np.savez_compressed(dest,**extra,counts=(np.empty((0,),np.int32) if args.baseline_only else counts),onsets_ms=onsets,
        condition_targets=np.array([v[0] for v in conditions]),
        condition_doses_pa=np.array([v[1] for v in conditions]),layers=np.array(LAYERS),
        n_e=np.array([base.CONFIG['layers'][L]['neuron_counts']['E'] for L in LAYERS]),
        rates_hz=rates,state_variables=np.array(STATE_VARS),state_onset_moments=moments,
        state_layers=np.array(['L6','L4C','L23']),network_seed=args.seed,wiring_digest=digest,feedback_enabled=args.feedback,
        metadata_json=json.dumps(dict(args=vars(args),checks=checks,phase_band_hz=[7,16],
            phase_method='retrospective Hilbert of unpulsed E rate',protocol='feedforward maintained from 0 ms; feedback added at 2000 ms; no rate matching',pulse_ms=10,
            response_ms=60,replay=replay_metadata,elapsed_seconds=time.time()-start)))
    status.update(status='complete',output=str(dest),elapsed_seconds=time.time()-start)
    base.atomic_json(statusfile,status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
