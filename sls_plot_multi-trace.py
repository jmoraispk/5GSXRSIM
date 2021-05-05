# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:28:21 2021

@author: Morais
"""



import numpy as np

# %matplotlib auto
# %matplotlib inline

import plots_functions as plt_func
import utils as ut


"""
Objective: To compare multiple frequencies of multiple traces.

How to do it: ignore the frequency parts, simply plot all traces given and 
              then make the distinction with proper labels.
              
              
NOTES about adding new variables:
    - add at the end of the list;
    - 

"""


# Stats file names from where to import the simulation data
folder_base = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\vsSPEED7'

plots_dir = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Plots\Current Speed Comparisons 1v7' + '\\'

# full list -> IMPORTANT: the order of the variables here will be the order to
#                         access the data in sim_data later
# PS: don't change the 'sp' variable.
vars_to_load = ['sp',                       #  0 
                'buffers',                  #  1 
                'realised_SINR',            #  2 
                'estimated_SINR',           #  3 
                'realised_bitrate_total',   #  4 
                'blocks_with_errors',       #  5 
                'n_transport_blocks',       #  6 
                'experienced_signal_power', #  7 # TODO: got broken after per_prb removal...
                'beams_used',               #  8
                'olla',                     #  9
                'mcs_used',                 # 10
                'real_dl_interference',     # 11
                'est_dl_interference',      # 12
                'channel',                  # 13
                'scheduled_UEs',            # 14
                '']

vars_to_load = ['sp',                       #  0
                'channel',               
                ''] # include an empty thing at the end, just because.

#vars_to_load = [vars_to_load[i] for i in [0, 1, 2, 3]]
vars_to_load = [vars_to_load[i] for i in range(len(vars_to_load)-1)]
    
seeds = [1,2,3]
t = 16
# s = int(ut.get_input_arg(1))
# t = float(ut.get_input_arg(2))

TTI_dur_in_secs = 0.25e-3
x_label = 'Time [TTI]'

for s in seeds:
    

    freq = 0
    files_end = [f'SEED{s}_SPEED-5_FREQ-{freq}_CSIPER-20_APPBIT-100_USERS-None_BW-50_LATBUDGET-10',
                 f'SEED{s}_SPEED-7_FREQ-{freq}_CSIPER-20_APPBIT-100_USERS-None_BW-50_LATBUDGET-10']
    
    
    files = [folder_base + '\\' + f + '\\' for f in files_end]
    
    #run 1
    # if idx == 11: then vars_to_load = ['sp', 'latencies']
    
    # run2 (not from start, the files to load didn't change!)
    # if idx = 9, then 
    
    # Phases of Data analysis:
        # A) setup phase: select folders
        # B) trim phase: select the trimming
        # C) index phase: select index
        #       -> load variables (includes trimming them accordingly to the trim idx)
        #       -> compute variables
        #       -> write results/make plots
    
    # The efficiency rule:
    # if one changes folders (phase A), load new simulation variables;
    # if one changes trimming (phase B), keep all the variables loaded, 
    #                                          but recompute new trimmed variables
    # e.g. at 20h: avg_channel = np.array(channel)[first_tti:last_tti]
    #      at 20h01: change last_tti -> you would not load 'channel' but you would
    #                                   compute the 'avg_channel' again
    # if one changes idx (phase C), see what/if new variables need to be loaded/computed;
    
    # Trick in plots_functions to not compute variables again:
        # if something changed (folder or trimming indices), then erase the values
        # of the computed variables. And they are only computed if they are = None,
        # which only happens at the start of the script or if the variables were erased.
    
    
    trim_ttis = [20, int(4000 * t)]
    sim_data = plt_func.load_sim_results(files, vars_to_load, trim_ttis)
    
    
    trim_secs = np.array(trim_ttis) * TTI_dur_in_secs 
    x_vals = np.arange(trim_secs[0], trim_secs[1], TTI_dur_in_secs)
    
    
    if not ut.isdir(plots_dir):
        ut.makedirs(plots_dir)
    
    # THE INDEX TO USE HERE MUST MATCH THE INDEX IN THE VARS_TO_LOAD LIST!
    
    #var_idxs = [2]
    var_idxs = [1] 
    
    file_idxs = [i for i in range(len(files))]
    
    ue_list = [i for i in range(4)]
    
    t_str = ut.get_time()
    
    
    # Compute average channel across the 5 speeds for the same seed, add to the 
    # data at the end and plot that as well (for 1s, 5s and 32s)
    # -> we want to see if there is some property of the seed that is still present in every speed
    #    (there should be, our standard deviations are smaller across the same seed!)
    # ....var_idxs = [13]
    """
    v = var_idxs[0]
    # such that this doesn't repeat after the first time
    if len(sim_data[v]) == len(files):
        # avg_channel_trace = np.zeros([trim_ttis[-1] - trim_ttis[0], len(ue_list)])
        avg_channel_trace = np.zeros(sim_data[v][0].shape)
        for seed_idx in range(len(files)):
            avg_channel_trace += sim_data[v][seed_idx]
        avg_channel_trace /= seed_idx + 1 # divide by the number of seeds considered
        
        # And append to sim_data
        sim_data[v].append(avg_channel_trace)
    """
    
    """
    avg_SINRs = ut.make_py_list(2, [len(files), len(ue_list)])
    for f in range(len(files)):
        for ue in ue_list:
            avg_SINRs[f][ue] = np.round(np.mean(sim_data[1][f][:,ue]),1)
    
    # mean along axis 1, the ues:
    print(np.mean(avg_SINRs, 1))
    """
    
    for v in var_idxs:
    
        # name_for_file = make_filename(t_str, vars_to_load[v], files, plots_dir)
        # name_for_file = make_filename(t_str, vars_to_load[v], files, 
        #                               sim_data[0][0].plots_dir)
        name_for_file = plots_dir + vars_to_load[var_idxs[0]] + \
                        f'_seed_{s}_speed' + files_end[0][12] + 'vs7'
        
        plt_func. plot_f2(ue_list, x_vals, sim_data, file_idxs, [v], 
                          title=vars_to_load[var_idxs[0]],         
                          y_labels = ['SPEED ' + files_end[0][12], 
                                      'SPEED ' + files_end[1][12]],
                          use_legend=True, ncols=4, filename=name_for_file, 
                          savefig=True)
    
        # avg_1 = []
        # avg_2 = []
        # avg_1.append(sim_data[v][-1]) # do this twice
        # avg_2.append(avg_1)
        # plot_f2(ue_list, x_vals, avg4, [0,1], [0], 
        #         title=vars_to_load[var_idxs[0]],
        #         y_labels = ['Avg. SEED8', 'Avg. SEED3'],
        #         use_legend=True, ncols=4, filename=name_for_file, savefig=True)
