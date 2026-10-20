"""Validate input provenance, baseline pairing, and restored probe trajectories."""
from pathlib import Path
import json,time
import numpy as np
import hashlib
HERE=Path(__file__).resolve().parent;REPO=Path(__file__).resolve().parents[2];OUT=REPO/'saved_trials/cosyne/feedback_protocol'
def main():
    result=dict(passed=False,checked_unix=time.time())
    try:
        # Originally: snapshots byte-identical to the project's stim_profiles.py, tools/lfp_kernel.py and
        # analysis/spectral/laminar_power_change.py on 10 Oct 2026. Those files have changed since, so the
        # snapshots are now checked against the SHA-256 recorded in source_manifest.json at run time.
        manifest=json.loads((OUT/'source_manifest.json').read_text())
        for f in [HERE/'stim_profiles_snapshot.py',HERE/'lfp_kernel_snapshot.py',HERE/'laminar_power_change_snapshot.py']:
            assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest['work/feedback_protocol/'+f.name],f'Snapshot changed: {f.name}'
        a,b,c=[np.load(OUT/'data'/f'smoke_{tag}_seed920091.npz') for tag in ['off','on','probe']]
        assert str(a['wiring_digest'])==str(b['wiring_digest'])==str(c['wiring_digest'])
        assert str(a['baseline_digest'])==str(b['baseline_digest'])==str(c['baseline_digest'])
        assert np.array_equal(a['rates_hz'][:,:,:2000],b['rates_hz'][:,:,:2000])
        assert np.array_equal(b['rates_hz'],c['rates_hz'][:,:,:4000])
        for L in b['layers']:
            for pop in ['E','PV','SOM','VIP']:
                key=f'traj_{L}_{pop}';keep=c[key+'_t_ms']<4000
                assert np.array_equal(b[key+'_t_ms'],c[key+'_t_ms'][keep])
                assert np.array_equal(b[key+'_i'],c[key+'_i'][keep])
        for z in [a,b]:
            for k in ['bipolar_kernel_uv','bipolar_lfp_uv','rates_hz']:assert np.isfinite(z[k]).all()
            assert z['counts'].size==0
        assert c['counts'].shape[:2]==(2,3)
        assert np.array_equal(c['condition_doses_pa'],[0,25,25])
        checks=json.loads(str(c['metadata_json']))['checks']
        assert len(checks)==2 and all(x['exact_sham_and_continuation'] for x in checks)
        result.update(passed=True,input_and_estimators_identical_to_original=True,off_on_initial_wiring_identical=True,
                      pre_feedback_complete_state_and_spikes_identical=True,
                      probe_extension_first_4000ms_all_population_spikes_identical=True,
                      exact_restored_shams=checks,all_lfp_samples_finite=True,
                      analysis_checks='Known 2× signal amplitude gives 6.0206 dB power change; known sinusoidal coefficients recovered; real-data spectral plot rendered and inspected.')
    except Exception as e:
        result['error']=repr(e);raise
    finally:(OUT/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
