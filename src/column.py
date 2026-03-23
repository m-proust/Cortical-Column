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

    def _get_inter_layer_delay_params(self, pre_pop, excitatory):
        if excitatory:
            delay_mean = 1.5*ms
            delay_std = 0.8*ms
        else:
            delay_mean = 1.0*ms
            delay_std = 0.5*ms
        return delay_mean, delay_std

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
                
                delay_mean, delay_std = self._get_inter_layer_delay_params(pre, excitatory)
                delay_expr = (f'{delay_mean/ms}*ms + '
                             f'clip(randn()*{delay_std/ms}, '
                             f'-{delay_std/ms}*0.5, {delay_std/ms}*2)*ms')
                
                connection_name = f"{source_layer}_{target_layer}_{conn}"
                
                if excitatory:
                    ampa_key = f'{conn}_AMPA'
                    nmda_key = f'{conn}_NMDA'
                    
                    g_ampa = cond_dict.get(ampa_key, 0.01)
                    g_nmda = cond_dict.get(nmda_key, 0.005)
                    
                    on_pre = f'''
                    gE_AMPA_post += {g_ampa}*nS
                    gE_NMDA_post += {g_nmda}*nS
                    '''
                    
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
    
  