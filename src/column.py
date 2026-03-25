"""
Column class implementation 
"""
import brian2 as b2
from brian2 import *
from .layer import CorticalLayer
from cleo import ephys


class CorticalColumn:

    
    def __init__(self, column_id=0, config=None):
        self.column_id = column_id
        
        if config is None:
            raise ValueError("CorticalColumn requires a config dictionary. Pass CONFIG from config module.")
        
        self.config = config
        self.layer_names = list(config['layers'].keys()) or ['L23', 'L4AB', 'L4C', 'L5', 'L6']
        self.electrode = None
        self.layers = {}
        self.inter_layer_synapses = {}
        
        self._create_layers()
        self._create_inter_layer_connections()
        
        self.network = Network()
        self._assemble_network()
        self._insert_electrode()

    def _insert_electrode(self):
        array_length = 2.25 * b2.mm  # 15 intervals × 150um
        channel_count = 16
        coords = ephys.linear_shank_coords(
            array_length, 
            channel_count=channel_count,
            start_location=(0, 0, -0.9) * b2.mm  
        )
        probe = ephys.Probe(coords, save_history=True)
        self.electrode = probe

    def _assemble_network(self):
        for layer in self.layers.values():
            self.network.add(*layer.neuron_groups.values())
            self.network.add(*layer.synapses.values())
            self.network.add(*layer.poisson_inputs.values())
            self.network.add(*layer.monitors.values())
        
        self.network.add(*self.inter_layer_synapses.values())

    def _create_layers(self):
        for layer_name in self.layer_names:
            if layer_name in self.config['layers']:
                self.layers[layer_name] = CorticalLayer(
                    f"{layer_name}_col{self.column_id}",
                    layer_name,
                    self.config
                )

    def _get_inter_layer_delay_params(self, pre_pop, post_pop,
                                   source_layer, target_layer):
        
        PRIMATE_SCALE = 1.4  # cortical thickness scaling for macaque vs mouse
        
        inter_delay_EE = {
            ('L23', 'L4AB'): 1.6, ('L23', 'L4C'): 1.6,
            ('L23', 'L5'):   1.8,
            ('L23', 'L6'):   2.5,
            ('L4AB', 'L23'): 2.3, ('L4C', 'L23'):  2.3,
            ('L4AB', 'L4C'): 1.3, ('L4C', 'L4AB'): 1.3,
            ('L4AB', 'L5'):  1.5, ('L4C', 'L5'):   1.5,
            ('L4AB', 'L6'):  1.9, ('L4C', 'L6'):   1.9,
            ('L5', 'L23'):   3.0, ('L5', 'L4AB'):  2.5,
            ('L5', 'L4C'):   2.5, ('L5', 'L5'):    1.5,
            ('L5', 'L6'):    1.8,
            ('L6', 'L23'):   3.0,  
            ('L6', 'L4AB'):  2.5, 
            ('L6', 'L4C'):   2.5, 
            ('L6', 'L5'):    3.5,
            ('L6', 'L6'):    1.8,
        }
        
        base_ms = inter_delay_EE.get((source_layer, target_layer), 1.8)
        
        # Adjust by presynaptic cell type (axon speed differences)
        # These ratios come from the intra-layer table:
        # PV axons are ~0.75x the E delay (fastest)
        # SOM axons are ~1.0x the E delay (similar)  
        # VIP axons are ~2.5x the E delay (slowest, thin unmyelinated)
        pre_scale = {
            'E':   1.0,
            'PV':  0.75,
            'SOM': 1.0,
            'VIP': 2.5,
        }.get(pre_pop, 1.0)
        
        mean_ms = base_ms * pre_scale * PRIMATE_SCALE
        std_ms = mean_ms * 0.3  
        
        return mean_ms * ms, std_ms * ms

    def _create_inter_layer_connections(self):
  
        inter_conns = self.config.get('inter_layer_connections', {})
        inter_conds = self.config.get('inter_layer_conductances', {})
        
        for (source_layer, target_layer), conns in inter_conns.items():
            if source_layer not in self.layers or target_layer not in self.layers:
                continue
                
            cond_dict = inter_conds.get((source_layer, target_layer), {})
            
            for conn, prob in conns.items():
                if prob <= 0:
                    continue
                    
                pre, post = conn.split('_')
                excitatory = (pre == 'E')
                
                src_group = self.layers[source_layer].get_neuron_group(pre)
                tgt_group = self.layers[target_layer].get_neuron_group(post)
                
                if src_group is None or tgt_group is None:
                    continue
                
                delay_mean, delay_std = self._get_inter_layer_delay_params(pre, post, source_layer, target_layer)
                delay_expr = (f'{delay_mean/ms}*ms + '
                             f'clip(randn()*{delay_std/ms}, '
                             f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
                
                connection_name = f"{source_layer}_{target_layer}_{conn}"
                
                if excitatory:
                    ampa_key = f'{conn}_AMPA'
                    nmda_key = f'{conn}_NMDA'
                    
                    g_ampa = cond_dict.get(ampa_key, 0.01)
                    g_nmda = cond_dict.get(nmda_key, 0.0)

                    on_pre = f'gE_AMPA_post += {g_ampa}*nS'
                    if g_nmda > 0:
                        on_pre += f'\ngE_NMDA_post += {g_nmda}*nS'

                    syn = Synapses(
                        src_group,
                        tgt_group,
                        on_pre=on_pre
                    )
                    syn.connect(p=float(prob))
                    #optional delays 
                    syn.delay = delay_expr
                    
                    self.inter_layer_synapses[connection_name] = syn
                    
                else:
                    g_inh = cond_dict.get(conn, 0.02)
                    target_var = 'g' + pre  
                    
                    on_pre = f'{target_var}_post += {g_inh}*nS'
                    
                    syn = Synapses(
                        src_group,
                        tgt_group,
                        on_pre=on_pre
                    )
                    syn.connect(p=float(prob))
                    syn.delay = delay_expr
                    
                    self.inter_layer_synapses[connection_name] = syn

    def get_layer(self, layer_name):
        return self.layers.get(layer_name)

    def get_all_monitors(self):
        all_monitors = {}
        for layer_name, layer in self.layers.items():
            all_monitors[layer_name] = layer.monitors
        return all_monitors
    
    def get_all_neuron_groups(self):
        all_groups = {}
        for layer_name, layer in self.layers.items():
            all_groups[layer_name] = layer.neuron_groups
        return all_groups
    
  