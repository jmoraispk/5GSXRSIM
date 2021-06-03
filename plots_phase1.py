# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:39:35 2020

@author: Morais
"""
import itertools

import numpy as np

# %matplotlib auto
# %matplotlib inline

import utils as ut
import plots_functions as plt_func


"""
There are just 3 ways of expanding this script:
    1- To add a new variable to be loaded (needs trimming as well)
    2- To add a new variable to be computed.
    3- To add a new plot based on the existing variables.

See the example below of a complete case that illustrates the 3 expansions:

    Objective: 1- extract a new variable from the simulation, say measure_x
               2- compute the average of measure_x of the last second
               3- plot the avg_meas_x across time, for every UE.
             
------------------------------------------------------------------------------- 
             
To add a new statistic to be collected:
1- .... add to the end of VARS_TO_LOAD the name of that variable.


( see whether the description of each index is better here or somewhere else, 
 and fetched with a function.)


To add a new plot index:
1- add what variables need to be loaded in get_vars_to_load(). See the 
   meaning of each code in the VARS_TO_LOAD variable in this file.
2- add what variables need to be computed in get_vars_to_compute()
   Note: if there is a variable that can be used for many different plots,
   it needs to be added here. Give it a new name, like "avg_across_traces",
   and add it to the end of VARS_TO_COMPUTE. From now on, sim_data_computed
   will have the results of that variable at the index it has on 
   VARS_TO_COMPUTE.
3- ... add the computation method...... somewhere..
"""

# When doing data analysis, we can avoid repeating many steps. However, 
# when implementing something new in LOADING, TRIMMING OR COMPUTING, the 
# data should be reloaded, retrimmed or recomputed. And note that when
# data is reloaded, it should be retrimmed and recomputed as well, but
# when is only recomputed, the loading and trim processes can be reused.

# When True, all the required variables are always loaded, trimmed or 
# computed, respectively, unconditional of being there already. Put to False 
# the step where implementation is occurring. It's almost always in computation
always_load = True
always_trim = True 
always_compute = True



#-------------------------
stats_folder = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv3\Stats' + '\\'
seeds = [1]
speeds = [1]
csi_periodicities = [20]
app_bitrates= [100]
users = [None]
bandwidths = [50] # MHz
latencies = [10]
freq_idxs = [0]
results_folder = r'Results\Batch X - testing' + '\\'

trim_ttis = [20, 4000 * 1]
TTI_dur_in_secs = 0.25e-3

ttis = np.arange(trim_ttis[0], trim_ttis[1])
x_vals = ttis * TTI_dur_in_secs


# From the simulated UEs, which UEs do we want to plot?
ues = [i for i in range(4)]


#----------------------

multi_trace = 0

combinations = list(itertools.product(speeds, freq_idxs, 
                                      csi_periodicities, app_bitrates,
                                      users, bandwidths, latencies, seeds))


# the variables to be loaded at each position of sim_data_(loaded/trimmed)
# NOTE: ALWAYS ADD VARIABLES AT THE END!!
VARS_NAME_LOAD = ['sp',                       #  0 
                  'buffers',                  #  1 
                  'realised_SINR',            #  2 
                  'estimated_SINR',           #  3 
                  'realised_bitrate_total',   #  4 
                  'blocks_with_errors',       #  5 
                  'n_transport_blocks',       #  6 
                  'beams_used',               #  7
                  'olla',                     #  8
                  'mcs_used',                 #  9
                  'experienced_signal_power', # 10
                  'sig_pow_per_prb',          # 11
                  'real_dl_interference',     # 12
                  'est_dl_interference',      # 13
                  'su_mimo_setting',          # 14
                  'scheduled_UEs',            # 15
                  'channel',                  # 16
                  'channel_per_prb',          # 17
                  '']

# Variable names that can be computed from the loaded and trimmed variables
VARS_NAME_COMPUTE = ['sinr_diff',                         # 0
                     'running_avg_bitrate',               # 1
                     'rolling_avg_bitrate',               # 2
                     'instantaneous_bler',                # 3
                     'running_avg_bler',                  # 4
                     'signal_power_db',                   # 5
                     'signal_power_prb_db',               # 6
                     'real_interference_db',              # 7
                     'est_interference_db',               # 8
                     'beam_formula_simple',               # 9
                     'beam_sum',                          # 10
                     'freq_vec',                          # 11
                     'frames',                            # 12
                     'I_frames',                          # 13
                     'avg_packet_lat',                    # 14
                     'avg_packet_drop_rate',              # 15
                     'avg_pck_lat_per_frame',             # 16
                     'avg_pck_drop_rate_per_frame',       # 17
                     'avg_pck_lat_per_I_frame',           # 18
                     'avg_pck_lat_per_P_frame',           # 19
                     'avg_pck_drop_rate_per_I_frame',     # 20
                     'avg_pck_drop_rate_per_P_frame',     # 21
                     'avg_pck_lat_per_frame_in_gop',      # 22
                     'avg_pck_drop_rate_per_frame_in_gop',# 23
                     'count_ues_scheduled',               # 24
                     'count_ues_bitrate',                 # 25
                     'beam_formula_processed',            # 26
                     'gop_idxs',                          # 27
                     'avg_sinr',                          # 28
                     'avg_sinr_multitrace',               # 29
                     '']

# file_sets has the sets of files to load at any given time.
# e.g. if we want to make a plot for each seed, we just want to load one seed
#      at a time. But if we want to make a plot that is the average of 3 seeds
#      we need to load those 3 seeds to compute the average.
file_sets = []


# Create the file set combinations from the variables given previously

for comb in combinations:

    stats_dir_end = f'SEED{comb[-1]}_SPEED-{comb[0]}_FREQ-{comb[1]}_' + \
                    f'CSIPER-{comb[2]}_APPBIT-{comb[3]}_'+ \
                    f'USERS-{comb[4]}_BW-{comb[5]}_LATBUDGET-{comb[6]}' + '\\'
                    
    print(f'\nDoing for: {stats_dir_end}')
    
    stats_dir = stats_folder + stats_dir_end
    
    # Can't recal what this is for...
    # if use_in_loop:
    #     extra_str = f'_f{comb[1]}_{trim_ttis}s'
    # else:
    #     extra_str = ''
        
    results_filename = results_folder + 'results' # + extra_str
    
    if not ut.isdir(results_folder):
        ut.makedirs(results_folder)    
    
    file_sets.append(stats_dir)

# File Sets is supposed to be a list of sets of files, 
# or a list of lists of files. How the sets are created depends on our
# multi-trace configurations. For single trace, there is only one set of files.
file_sets = [file_sets]

##############################################################################    
    
for file_set in file_sets:
    
    # For the very first run and they don't exist, initialise
    try:
        sim_data_loaded
    except NameError:
        sim_data_loaded = []
        sim_data_trimmed = []
        sim_data_computed = []
        
        # To know when things need to be reloaded or retrimmed
        file_set_temp = ['']
        ttis_temp = np.array([0,0])
    
    # INIT SIM DATA: all sim_data are [trace_idx][variable_idx]
    # This function is where most of the efficiency lies: we set variables to 
    # None when they need to be computed again. And we decide when that's 
    # suppose to happen based on the three variables above and two temp. below
    (sim_data_loaded, sim_data_trimmed, sim_data_computed) = \
        plt_func.init_sim_data(sim_data_loaded, sim_data_trimmed, 
                                    sim_data_computed, VARS_NAME_LOAD, 
                                    VARS_NAME_COMPUTE, ttis, file_set, 
                                    file_set_temp, ttis_temp,
                                    always_load, always_trim, always_compute)
    
    file_set_temp = file_set
    ttis_temp = ttis
    
    # Only sim_data_trimmed is plotted. Some variables of sim_data_loaded 
    # don't need to be trimmed but are copied for convenience (sp and buffers)
    
    # See if the file to use is the last one simulated
    if file_set == ['']: 
        with open("last_stats_folder.txt", 'r') as fh:
            file_set = [fh.readline()]
            

    """
    "X marks the spot" (or spots) where implementation are still needed
    plot_idx:
    0.X  -> Channel Power - Note: computations are computationally demanding!
             Depending on the second decimal place, we compute ther powers in
             two different ways:
    0.1   -> across time (mean power over prbs)                        
X   0.2   -> across time (for all prbs, 1 ue or 4 ues)
X   0.3   -> across prbs (for a given tti)
     
    1     -> Throughput
    1.1   -> 
    1.2   -> 
    
    2     -> SINR (multi-user) estimated vs realised
    2.1   -> SINR (single-user) when there are active transmissions
    2.2   -> SINR (multi-user) when there are active transmissions
    2.3   -> 
    2.4   ->
    
    3     -> Signal power per PRB in Watt (for tti = 3)
    3.1   -> Signal power per PRB in dB (for tti = 3, middle prb as reference)
    3.2   -> Signal power vs Interference power
    3.3   -> Signal power (only) in [dB], equivalent to the beamformed channel
    3.45
    3.5
    3.6
    3.7
    3.8
    
    4.1   -> MCS per user, same axis
    4.2   -> MCS per user, diff axis
    4.3   -> 
    
    5.1   -> Beams per user
    5.2   -> Beams per user filtered: keeps beam value constant instead of
             allowing going to 0 when the UE is not scheduled - better for 
             cases where not all ues are scheduled simultaneously always.
    5.3   -> Beams per user: Same as 5.2 but with one plot per UE.
    5.4
    5.5
    
    7.1   -> BLER: instantaneous
    7.2   -> BLER: running_average
    7.3   -> BLER: instantaneous vs running average
    7.35
X   7.4   -> BLER: instantaneous vs realised bit rate [doublePLOT]
    7.5
    
    9.1   -> OLLA:  instantaneous BLER vs olla parameter (single-user)
                    [when active transmissions] [doublePLOT]
    9.2   -> OLLA:  instantaneous BLER vs olla parameter (multi-user)
                    [when active transmissions] [doublePLOT]
    9.3   -> OLLA: MCS vs olla param [doublePLOT]
    9.4   -> OLLA: instantaneous BLER vs olla param [doublePLOT]
    
    10.1  -> Just average Latency
    10.15
    10.2  -> Just 
    10.25
    10.3
    10.31
    10.4
    10.45
    10.5  -> writes to terminal avg_lat across all frames
    10.55 -> writes to terminal avg_lat across all frames and saves in file
    10.6  -> writes to terminal avg_pdr across all frames
    10.65 -> writes to terminal avg_pdr across all frames and saves in file
    
    11    -> Scheduled UEs: sum of co-scheduled UEs across time
    11.1  -> Scheduled UEs: each UE is 1 when it is scheduled and 0 when not
    11.2  -> Scheduled UEs: each UE is 1 when it is scheduled and 0 when not
                            (all UEs in the same axis)
X   11.3  -> Scheduled UEs: each UE is 1 when it is scheduled and 0 when not
                            (all UEs SUMMED in the same axis)
    11.4  -> UEs with bitrate: each UE. There's a difference between having
             bitrate and being scheduled! The schedule is only updated when
             there's a scheduling update... However, the user can be added 
             to the schedule and get no (useful) bitrate. It will still get
             bits across, but those might have no utility because they have
             transferred before.

    13    -> SU-MIMO setting - number of layers scheduled per UE
    
    14    -> plot packet sequences for each UE. (same axis)
    14.1  -> plot packet sequences for each UE. (separate axis)
    """
    all_idxs_available = [0.1, 1, 1.1, 1.2, 2, 2.1, 2.15, 2.2, 2.3, 2.4, 3, 3.1, 
                          3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.1, 4.2, 4.3, 5.1, 
                          5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 7.4, 7.5, 9.1, 
                          9.2, 9.3, 9.4, 10.1, 10.15, 10.2, 10.25, 10.3, 10.31,
                          10.4, 10.45, 10.5, 10.55, 10.6, 10.65, 11, 11.1, 
                          11.2, 11.3, 13, 14.1, 14.2, 15]
    
    idxs_to_plot = [0.1, 1, 2, 3.45, 3.7, 4.2, 5.4, 7.35, 7.4, 10.45, 14.2]

    idxs_to_plot = all_idxs_available
    
    idxs_to_plot = [15]
    
    # Test save_plot
    save_plots = False
    base_plots_folder = 'Plots\\' 
    
    for i in idxs_to_plot:
        print(f'Plotting {i}')
        
        which_vars_to_load = plt_func.get_vars_to_load(i, VARS_NAME_LOAD)
        
        which_vars_to_compute = \
            plt_func.get_vars_to_compute(i, VARS_NAME_COMPUTE)
        
        # Load data
        plt_func.load_sim_data(file_set, VARS_NAME_LOAD, 
                                    which_vars_to_load, sim_data_loaded)
        
        # Trim data 
        plt_func.trim_sim_data(sim_data_loaded, sim_data_trimmed, 
                                    VARS_NAME_LOAD, which_vars_to_load, 
                                    file_set, trim_ttis)
        
        # Compute additional data: 
            # - some variables might be computed already (check if empty)
            # - first, compute the auxiliar variables that might be needed
            # - second, compute what the actual index needs 
            #   e.g. some average of one of the auxiliar variables across traces)
        
        if multi_trace:
            raise Exception('not ready yet...')
        
        plt_func.compute_sim_data(i, ues, ttis, VARS_NAME_LOAD, 
                                       VARS_NAME_COMPUTE, which_vars_to_compute, 
                                       which_vars_to_load, 
                                       sim_data_trimmed, sim_data_computed,
                                       file_set)
        
        # SEE HOW LONG 2.1 (I THINK) takes to load, trim, compute and plot.
        
        # Plots:
        plt_func.plot_sim_data(i, file_set, ues, ttis, x_vals, 
                                    sim_data_trimmed, sim_data_computed,
                                    results_filename,
                                    base_plots_folder, save_plots)
    
    
    
auto_report = False
if auto_report:
    pass
    # TODO: save and merge pdfs (auto-reporting feature)

