"""
Layer class implementation
"""

import brian2 as b2
from brian2 import *
import numpy as np
import cleo


class CorticalLayer:
    
    def __init__(self, name, layer_name, config):
        self.name = name
        self.layer_name = layer_name
        self.config = config
        self.layer_config = config['layers'][layer_name]
        self.syn_model = (self.config['models'].get('synapse_model', 'conductance')).lower()
        self.is_current = (self.syn_model == 'current')
        self.neuron_groups = {}
        self.synapses = {}
        self.monitors = {}
        self.poisson_inputs = {}

        self._create_neuron_groups()
        

        self._set_neuron_parameters()
        self._create_poisson_inputs()
        self._create_internal_connections()
        self._create_monitors()


    def _create_neuron_groups(self):
        models_cfg = self.config.get('models', {})
        common_namespace = dict(models_cfg.get('common_namespace', {}))
        extra_ns = models_cfg.get('namespace', {})
        common_namespace.update(extra_ns)

        eqs_map = self.config['models']['equations']
        threshold = self.config['models']['threshold']
        reset = self.config['models']['reset']
        
        eq_override = self.layer_config.get('equation_override', {})

        for pop_name, n in self.layer_config.get('neuron_counts', {}).items():
            eq_key = eq_override.get(pop_name, pop_name)
            
            t_ref_config = self.config['neurons']['T_REF']
            if isinstance(t_ref_config, dict):
                t_ref = t_ref_config.get(pop_name, t_ref_config.get('E'))
            else:
                t_ref = t_ref_config

            self.neuron_groups[pop_name] = NeuronGroup(
                int(n), eqs_map[eq_key],
                threshold=threshold,
                reset=reset,
                refractory=t_ref,
                namespace=common_namespace
            )
            
            cleo.coords.assign_coords_rand_rect_prism(
                self.neuron_groups[pop_name],
                xlim=self.layer_config['coordinates']['x'],
                ylim=self.layer_config['coordinates']['y'],
                zlim=self.layer_config['coordinates']['z']
            )

    def _set_neuron_parameters(self):
        def set_if_exists(group, attr, value):
            try:
                getattr(group, attr)
            except AttributeError:
                return
            setattr(group, attr, value)

        cfg = self.config
        neurons_cfg = cfg.get('neurons', {})
        intrinsic_global = cfg.get('intrinsic_params', {})
        intrinsic_layer = self.layer_config.get('intrinsic_params', {})
        ic_all = cfg.get('initial_conditions', {})
        vt_default = neurons_cfg.get('VT', -50.0*mV)

        drive = cfg.get('drive', None)
        if drive is not None:
            c = float(drive.get('contrast', 1.0))
            mu_bg = drive.get('mu_bg', {})
            sd_bg = drive.get('sd_bg', {})
            mu_st = drive.get('mu_stim', {})
            sd_st = drive.get('sd_stim', {})
            sg = drive.get('sigma_global', 0*mV)
        eq_override = self.layer_config.get('equation_override', {})
        
        for pop_name, g in self.neuron_groups.items():
            param_key = eq_override.get(pop_name, pop_name)
            # Start with global params, then override with layer-specific
            ip_global = intrinsic_global.get(param_key, intrinsic_global.get(pop_name, {}))
            ip_layer = intrinsic_layer.get(param_key, intrinsic_layer.get(pop_name, {}))
            ip = dict(ip_global, **ip_layer)  # Layer-specific overrides global
       
            set_if_exists(g, 'a', ip.get('a', 0*nS))
            set_if_exists(g, 'b', ip.get('b', 0*pA))
            set_if_exists(g, 'DeltaT', ip.get('DeltaT', 2*mV))
            set_if_exists(g, 'EL', ip.get('EL', -60*mV))
            set_if_exists(g, 'gL', ip.get('gL', 8*nS))
            set_if_exists(g, 'C', ip.get('C', 200*pF))
            set_if_exists(g, 'tauw', ip.get('tauw', 200*ms))

            ic_default = ic_all.get('DEFAULT', {})
            ic_pop = ic_all.get(pop_name, {})
            merged_ic = dict(ic_default, **ic_pop)

            vcut_factor = merged_ic.pop('Vcut_offset_factor', None)
            if vcut_factor is not None and hasattr(g, 'Vcut'):
                dT = ip.get('DeltaT', getattr(g, 'DeltaT', 2*mV))
                set_if_exists(g, 'Vcut', vt_default + vcut_factor * dT)

            for k, v in merged_ic.items():
                set_if_exists(g, k, v)

            if drive is not None:
                mu = mu_bg.get(pop_name, 0*mV) + c * mu_st.get(pop_name, 0*mV)
                sb = sd_bg.get(pop_name, 0*mV)
                ss = sd_st.get(pop_name, 0*mV)
                sigma = sqrt(sb**2 + (c*ss)**2 + sg**2)
                set_if_exists(g, 'mu_drive', mu)
                set_if_exists(g, 'sigma_drive', sigma)

    def _get_delay_params(self, pre_pop, post_pop):
   
        delay_table = {
            ('E', 'E'):     (1.5, 0.5),
            ('E', 'PV'):    (1.2, 0.8),
            ('E', 'SOM'):   (1.5, 0.5),
            ('E', 'VIP'):   (1.5, 0.5),
            ('PV', 'E'):    (0.9, 0.8),
            ('PV', 'PV'):   (1.6, 0.4),
            ('PV', 'SOM'):  (1.2, 0.4),
            ('PV', 'VIP'):  (1.2, 0.4),
            ('SOM', 'E'):   (1.5, 0.5),
            ('SOM', 'PV'):  (1.5, 0.5),
            ('SOM', 'SOM'): (1.5, 0.5),
            ('SOM', 'VIP'): (1.5, 0.5),
            ('VIP', 'E'):   (3.8, 1.0),
            ('VIP', 'PV'):  (3.8, 1.0),
            ('VIP', 'SOM'): (3.8, 1.0),
            ('VIP', 'VIP'): (3.8, 1.0),
        }
    
        mean_ms, std_ms = delay_table.get((pre_pop, post_pop), (1.5, 0.5))
        return mean_ms * ms, std_ms * ms

    def _create_internal_connections(self):
        pmap = self.layer_config.get('connection_prob', {})
        cmap = self.layer_config.get('conductance', {})
        
        for connection, p in pmap.items():
            pre, post = connection.split('_')
            if pre not in self.neuron_groups or post not in self.neuron_groups:
                continue
            
            excitatory = (pre == 'E')
            delay_mean, delay_std = self._get_delay_params(pre, post)
            
            if excitatory:
                #  AMPA and NMDA conductances
                ampa_key = f'{connection}_AMPA'
                nmda_key = f'{connection}_NMDA'
                
                g_ampa = cmap.get(ampa_key, 0.01)
                g_nmda = cmap.get(nmda_key, 0.0)

                on_pre_str = f'gE_AMPA_post += {g_ampa}*nS'
                if g_nmda > 0:
                    on_pre_str += f'\ngE_NMDA_post += {g_nmda}*nS'

                syn = Synapses(
                    self.neuron_groups[pre],
                    self.neuron_groups[post],
                    on_pre=on_pre_str
                )
                syn.connect(p=float(p))
                syn.delay = (f'{delay_mean/ms}*ms + '
                            f'clip(randn()*{delay_std/ms}, '
                            f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
                
                self.synapses[connection] = syn
            else:
                g_inh = cmap.get(connection, 0.02)
                
                target_var = 'g' + pre 
                
                syn = Synapses(
                    self.neuron_groups[pre],
                    self.neuron_groups[post],
                    on_pre=f'{target_var}_post += {g_inh}*nS'
                )
                syn.connect(p=float(p))
                # optional delays
                syn.delay = (f'{delay_mean/ms}*ms + '
                            f'clip(randn()*{delay_std/ms}, '
                            f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
                
                self.synapses[connection] = syn

    def _create_poisson_inputs(self):
        pinputs = self.layer_config.get('poisson_inputs', {})
        
        for pop_name, pconf in pinputs.items():
            actual_pop = pop_name.split('_')[0]
            
            if actual_pop not in self.neuron_groups:
                continue
                
            target_var = pconf.get('target', 'gE_AMPA')
            if self.is_current:
                target_var = {'gE_AMPA': 'sE', 'gi': 'sI'}.get(target_var, target_var)

            if 'N' in pconf and pconf['N'] is not None:
                N = int(pconf['N'])
            else:
                N = float(pconf.get('N', 0.0))

            w_cfg = pconf.get('weight', 'EXT_AMPA')
            weight = self.config['synapses']['Q'].get(w_cfg, 0.5*nS) if isinstance(w_cfg, str) else w_cfg
            rate = pconf.get('rate', self.layer_config.get('input_rate', 0*Hz))

            self.poisson_inputs[pop_name] = PoissonInput(
                self.neuron_groups[actual_pop], target_var, N=N, rate=rate, weight=weight
            )

    def _create_monitors(self):
        for pop_name, group in self.neuron_groups.items():
            candidate_vars = ['v', 'gE', 'gI', 'IsynE', 'IsynE_AMPA', 'IsynE_NMDA',
                               'IsynI', 'IsynIPV', 'IsynISOM', 'IsynIVIP'] if not self.is_current else ['v', 'sE', 'sI']
            vars_to_record = [v for v in candidate_vars if v in group.variables]

            self.monitors[f'{pop_name}_state'] = StateMonitor(group, vars_to_record, record=True, dt=1*ms)
            self.monitors[f'{pop_name}_spikes'] = SpikeMonitor(group, variables='t')
            self.monitors[f'{pop_name}_rate'] = PopulationRateMonitor(group)

    def get_monitor(self, monitor_type, population='E'):
        return self.monitors.get(f'{population}_{monitor_type}')

    def get_neuron_group(self, population='E'):
        return self.neuron_groups.get(population)




