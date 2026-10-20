"""Bounded serial replication. All scientific analyses run locally after each pair."""
from pathlib import Path
import os,sys,json,time,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'saved_trials/cosyne/feedback_protocol'
ANALYZE=HERE/'analyze.py'
ENV=os.environ.copy()
ENV.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
STATE=dict(status='starting',completed=[],total_jobs=24,pid=os.getpid(),started_unix=time.time())

def save():
    temp=OUT/'queue_status.tmp';temp.write_text(json.dumps(STATE,indent=2));temp.replace(OUT/'queue_status.json')
    (OUT/'RUN_STATUS.md').write_text(f"# Original-profile feedback replication\n\nStatus: **{STATE['status']}**\n\nStage: {STATE.get('stage','validation')}\n\nCompleted jobs: {len(STATE['completed'])}/24. Six paired networks, spectra first, then excitability. One simulation at a time.\n\nAnalysis and plots update automatically after each completed pair. These local programs do not require model credits. They require the computer to remain running; sleep pauses execution. A failed validation stops this batch. A STOP file in this folder stops the batch after its current job.\n\n{STATE.get('error','')}\n")

def validate(path,spectral):
    z=np.load(path)
    assert bool(z['feedforward_enabled']) and float(z['feedback_onset_ms'])==2000
    assert np.isfinite(z['rates_hz']).all()
    if spectral:
        assert z['counts'].size==0 and z['rates_hz'].shape[-1]==4000
        for key in ['bipolar_lfp_uv','bipolar_kernel_uv']:assert np.isfinite(z[key]).all()
    else:
        checks=json.loads(str(z['metadata_json']))['checks']
        assert len(checks)==2 and all(c['exact_sham_and_continuation'] for c in checks)
        assert z['counts'].shape[0]==96 and np.array_equal(z['condition_doses_pa'],[0,25,25])

def main():
    save()
    try:
        v=json.loads((OUT/'validation.json').read_text())
        if not v['passed']:raise RuntimeError('Smoke validation did not pass')
        for kind in ['spectrum','probe']:
            for seed in range(920101,920107):
                for c in [0,1]:
                    if (OUT/'STOP').exists():raise InterruptedError('STOP requested')
                    tag=f'{kind}_{c}';stem=f'{tag}_seed{seed}';dest=OUT/'data'/f'{stem}.npz'
                    STATE.update(status='running',stage=stem);save()
                    if not dest.exists():
                        cmd=[sys.executable,'-u',str(HERE/'run_feedback.py'),'--tag',tag,'--seed',str(seed),'--feedback',str(c),'--onsets','96','--doses','25']
                        if kind=='spectrum':cmd+=['--spectral-only']
                        with (OUT/'logs'/f'{stem}.log').open('w') as f:
                            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=ENV)
                            STATE['worker_pid']=child.pid;save();code=child.wait()
                        if code:raise RuntimeError(f'{stem} failed with exit code {code}')
                    validate(dest,kind=='spectrum')
                    if kind=='probe':
                        spectral=np.load(OUT/'data'/f'spectrum_{c}_seed{seed}.npz');probe=np.load(dest)
                        assert str(spectral['baseline_digest'])==str(probe['baseline_digest'])
                        assert np.array_equal(spectral['rates_hz'],probe['rates_hz'][:,:,:4000]),'Probe extension changed original unpulsed 4 s trajectory'
                    STATE['completed'].append(stem);save()
                a,b=[np.load(OUT/'data'/f'{kind}_{c}_seed{seed}.npz') for c in [0,1]]
                assert str(a['wiring_digest'])==str(b['wiring_digest'])
                assert str(a['baseline_digest'])==str(b['baseline_digest']),'Off/on pre-feedback states or spikes differ'
                assert np.array_equal(a['rates_hz'][:,:,:2000],b['rates_hz'][:,:,:2000])
                with (OUT/'logs'/'analysis.log').open('a') as f:
                    subprocess.run([sys.executable,str(ANALYZE)],stdout=f,stderr=subprocess.STDOUT,env=ENV,check=True)
        STATE.update(status='complete',stage='All six pairs and analyses complete',finished_unix=time.time());save()
    except InterruptedError as e:
        STATE.update(status='stopped',error=str(e));save()
    except Exception as e:
        STATE.update(status='failed',error=repr(e));save();raise

if __name__=='__main__':main()
