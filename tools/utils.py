import pandas as pd


def load_connectivity_from_csv(conn_prob_file, cond_ampa_file, cond_nmda_file,
                               layers=None, cell_types=None):

    conn_prob_df = pd.read_csv(conn_prob_file, index_col=0)
    cond_ampa_df = pd.read_csv(cond_ampa_file, index_col=0)
    cond_nmda_df = pd.read_csv(cond_nmda_file, index_col=0)
    if layers is None:
        layers = [ 'L23', 'L4AB', 'L4C', 'L5', 'L6']
    if cell_types is None:
        cell_types = ['E', 'PV', 'SOM', 'VIP']
    l1_cell_types = ['VIP']

    layer_configs = {}
    inter_layer_connections = {}
    inter_layer_conductances = {}

    for layer in layers:
        connection_prob = {}
        conductance = {}
        if layer == 'L1' :
            current_cell_types=l1_cell_types
        else :
            current_cell_types = cell_types

        for src_type in current_cell_types:
            for tgt_type in current_cell_types:
                src_col = f'{src_type}_{layer}'
                tgt_col = f'{tgt_type}_{layer}'

                prob_val = conn_prob_df.loc[src_col, tgt_col]
                if prob_val > 0:
                    conn_key = f'{src_type}_{tgt_type}'
                    connection_prob[conn_key] = prob_val

                if src_type == 'E':
                    ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                    nmda_val = cond_nmda_df.loc[src_col, tgt_col]

                    if ampa_val > 0:
                        cond_key = f'{src_type}_{tgt_type}_AMPA'
                        conductance[cond_key] = ampa_val

                    if nmda_val > 0:
                        cond_key = f'{src_type}_{tgt_type}_NMDA'
                        conductance[cond_key] = nmda_val
                else:
                    ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                    if ampa_val > 0:
                        cond_key = f'{src_type}_{tgt_type}'
                        conductance[cond_key] = ampa_val

        layer_configs[layer] = {
            'connection_prob': connection_prob,
            'conductance': conductance
        }
    

    for src_layer in layers:
        for tgt_layer in layers:
            if src_layer == tgt_layer:
                continue
            
            conn_dict = {}
            cond_dict = {}
            if src_layer == 'L1' :
                src_cell_types=l1_cell_types
            else :
                src_cell_types = cell_types
            if tgt_layer == 'L1':
                tgt_cell_types = l1_cell_types
            else :
                tgt_cell_types = cell_types

            for src_type in src_cell_types:
                for tgt_type in tgt_cell_types:
                    src_col = f'{src_type}_{src_layer}'
                    tgt_col = f'{tgt_type}_{tgt_layer}'

                    prob_val = conn_prob_df.loc[src_col, tgt_col]

                    if prob_val > 0:
                        conn_key = f'{src_type}_{tgt_type}'
                        conn_dict[conn_key] = prob_val

                        if src_type == 'E':
                            ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                            nmda_val = cond_nmda_df.loc[src_col, tgt_col]

                            if ampa_val > 0:
                                cond_dict[f'{src_type}_{tgt_type}_AMPA'] = ampa_val
                            if nmda_val > 0:
                                cond_dict[f'{src_type}_{tgt_type}_NMDA'] = nmda_val
                        else:
                            ampa_val = cond_ampa_df.loc[src_col, tgt_col]
                            if ampa_val > 0:
                                cond_dict[f'{src_type}_{tgt_type}'] = ampa_val

            if conn_dict:
                inter_layer_connections[(src_layer, tgt_layer)] = conn_dict
                inter_layer_conductances[(src_layer, tgt_layer)] = cond_dict
    
    return layer_configs, inter_layer_connections, inter_layer_conductances


