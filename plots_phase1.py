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
csi_periodicities = [5]
app_bitrates= [100]
users = [None]
bandwidths = [50] # MHz
latencies = [10]
freq_idxs = [0]
results_folder = r'Results\Batch X - testing' + '\\'

layer = 0
trim_ttis = [20, int(4000 * 6)]
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
                  'power_per_beam',           # 18
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
                     'power_per_gob_beam',                # 28
                     'x_projection_best_beam',            # 29
                     'y_projection_best_beam',            # 30
                     'beam_switch',                       # 31
                     'xy_projection_all_gob',             # 32
                     'user_pos_for_plot',                 # 33
                     'user_ori_for_plot',                 # 34
                     'individual_beam_gob_details',       # 35
                     'beams_processed',                   # 36
                     'avg_sinr',                          # 37
                     'avg_sinr_multitrace',               # 38
                     '']

# (Loaded) Vars with information per layer
vars_with_layers = [2,3,5,6,7,8,9,10,12,13]

# file_sets has the sets of files to load at any given time.
# e.g. if we want to make a plot for each seed, we just want to load one seed
#      at a time. But if we want to make a plot that is the average of 3 seeds
#      we need to load those 3 seeds to compute the average.
file_sets = []

# Create the file set combinations from the variables given previously

for comb in combinations:

    stats_dir_end = f'SEED{comb[-1]}_SPEED-{comb[0]}_FREQ-{comb[1]}_' + \
                    f'CSIPER-{comb[2]}_APPBIT-{comb[3]}_'+ \
                    f'USERS-{comb[4]}_BW-{comb[5]}_LATBUDGET-{comb[6]}_coph-1' + '\\'
    stats_dir_end = r'SEED1_SPEED-1_FREQ-0_CSIPER-5_APPBIT-100_USERS-None_BW-50_LATBUDGET-10_ROTFACTOR-None_modifiedfinal' + '\\'
    
    print(f'\nDoing for: {stats_dir_end}')
    
    stats_dir = stats_folder + stats_dir_end
    
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
        
    if sim_data_computed == []:
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
                               ttis_temp, file_set_temp, 
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
    plot_idx meanings:

    "X marks the spot" where implementation is still needed
    
    same vs diff plot - all UEs in same plot, or one subplot for each UE
    
    single vs double axis - plot several variables in the same axis or use 
                            left and right axis, to have use different ticks.
                            Note: when the variables have different natures,
                            like bitrate and BLER, it doesn't make sense to 
                            do single axis

    0.1   -> Channel Power across time (mean power over prbs)                        
X   0.2   -> Channel Power across time (for all prbs, 1 ue or 4 ues)
X   0.3   -> Channel Power across prbs (for a given tti)
     
    1     -> Throughput
    1.1   -> Inst. vs Running avg bitrate 
    1.2   -> Inst throughput vs Rolling/moving avg bitrate 
    
    2     -> SINR (multi-user) estimated vs realised
    2.1   -> SINR (single-user) when there are active transmissions
    2.15  -> SINR (multi-user) when there are active transmissions
    2.2   -> SINR vs OLLA (multi-user) when there are active transmissions
    2.3   -> SINR difference (realised - estimated -> negative can mean errors
                                                      positive can mean waste)
    2.4   -> SINR difference vs BLER
    
    3     -> Signal power per PRB in Watt (for tti = 3)
    3.1   -> Signal power per PRB in dB (for tti = 3, middle prb as reference)
    3.2   -> Signal power (only) in [dB], == BEAMFORMED CHANNEL!)
    3.3   -> Signal power vs Interference power (Watt)[single axis]
    3.35  -> Signal power vs Interference power (Watt)[double axis]
    3.4   -> Signal power vs Interference power (dBW) [single axis]
    3.45  -> Signal power vs Interference power (dBW) [double axis]
    3.5   -> Signal power vs Interference power (dBm) [single axis]
    3.55  -> Signal power vs Interference power (dBm) [double axis]
    3.6   -> Estimated vs Realised Interference       [single axis]
    3.65  -> Estimated vs Realised Interference [dB]  [single axis]
    
    4.1   -> MCS per user, same plot
    4.2   -> MCS per user, diff plots
    4.3   -> MCS vs Instantaneous bit rate per user, diff plots
    
    5.1   -> Best beam per user (filtered to prevent back to 0 when not UE is  
                                 not scheduled)
    5.15  -> Beam formula (filtered and smoother - keeps beam value constant 
                           when not scheduled)
    5.2   -> Azi and Elevation [double plot]
    5.3   -> Beam sum (azi + el)
    5.4   -> Beam sum vs SINR
    5.5   -> Beam sum vs BLER
    5.6   -> Beam switch: 1 when there's a change in beams, else 0 (same plot)
    5.65  -> Beam switch: 1 when there's a change in beams, else 0 (diff plot)
    
    7.1   -> BLER instantaneous
    7.2   -> BLER running_average ber
    7.3   -> BLER: instantaneous vs running average [single axis]
    7.35  -> BLER: instantaneous vs running average [double axis]
    7.4   -> BLER instantaneous vs realised bit rate [double axis]
    7.5   -> BLER instantaneous vs realised SINR [double axis]
    
    9.1   -> OLLA:  instantaneous BLER vs olla parameter (single-user)
                    [when active transmissions] [double axis]
    9.2   -> OLLA:  instantaneous BLER vs olla parameter (multi-user)
                    [when active transmissions] [double axis]
    9.3   -> OLLA: MCS vs olla param [double axis]
    9.4   -> OLLA: instantaneous BLER vs olla param [double axis]
    
    10.1  -> Average packet latency of each frame (line plot)
    10.15 -> Average packet latency of each frame (bar plot)
    10.2  -> Average packet drop rate of each frame (line plot)
    10.25 -> Average packet drop rate of each frame (bar plot)
    10.3  -> Average packet latency vs drop rate of each frame (line plot)
             [double axis] 
    10.31 -> Average packet latency vs drop rate of each frame (line plot)
             [double axis] with tick limit control as a demo
    10.4  -> Average packet latency vs drop rate of each frame (line plot) 
             [double axis] with vertical line marking I frames 
    10.45 -> Average packet latency vs drop rate of each frame (bar plot) 
             [double axis] with vertical line marking I frames 
    10.5  -> prints the average packet latency averaged across all frames
    10.55 -> prints average packet latency averaged across all frames 
             + saves to file
    10.6  -> prints the average packet drop rate averaged across all frames
    10.65 -> prints the average packet drop rate averaged across all frames
             + saves to file
    10.7  -> prints all detailed measurement information on:
             -> Average packet latency:
                -> for each frame in the GoP
                -> for each I frame
                -> for each P frame
                -> averaged across all frames and std
             -> Average packet drop rate:
                -> for each frame in the GoP
                -> for each I frame
                -> for each P frame
                -> averaged across all frames and std
    10.8  -> Average packet latency for each frame in the GoP (bar plot) 
    10.9  -> Average packet drop rate for each frame in the GoP (bar plot)
    10.11 -> Average packet latency and drop rate per frame of the GoP 
             [double plot]

    11    -> Scheduled UEs: sum of co-scheduled UEs across time
    11.1  -> Scheduled UEs: each UE is 1 when it is scheduled and 0 when not
    11.2  -> Scheduled UEs: each UE is 1 when it is scheduled and 0 when not,
                            all UEs in the [same plot]                  
    11.3  -> UEs with bitrate: each UE. There's a difference between having
             bitrate and being scheduled! The schedule is only updated when
             there's a scheduling update... However, the user can be added 
             to the schedule and get no (useful) bitrate. It will still get
             bits across, but those might have no utility because they have
             transferred before.
    11.4  -> Scheduled UEs vs signal power (linear)
X    11.5  -> UEs with bitrate vs signal power (linear) --> quite similar to .4

         
    13    -> SU-MIMO setting - number of layers scheduled per UE
    
    14.1  -> Packet sequences for each UE. [same plot]
    14.2  -> Packet sequences for each UE. [diff plot]
    
    15    -> Power of each GoB beam
    
    16    -> Projection of the all chosen beam for each ue [same plot]
    16.1  -> Projection of the all chosen beam for each ue [diff plot]
    16.2  -> Projection of the all beams in GoBs

    # NOTE: the view on the GIF can be changed (top, side and 3D)
    17    -> GIF across time: TRACKS [same plot]
    17.01 -> GIF across time: TRACKS [same plot] + BEAMS (no -3 dB marks)
    17.02 -> GIF across time: TRACKS [same plot] + BEAMS (+constant HPBW)
    17.03 -> GIF across time: TRACKS [same plot] + BEAMS (+correct HPBW)
    17.1  -> GIF across time: Just beams used (just maximum direction)
    17.11 -> GIF across time: Just beams used (with constant HPBW)
    17.12 -> GIF across time: Just beams used (with correct HPBW)
                               (needs to be computed in Matlab and loaded))
    17.2  -> ...
    """
    
    """ 
    Warnings:
        
        - Empty plots: 
            Possibility 1: 10 * log10(0) = -inf -> this does not show in 
                           logarithmic plots. Therefore, try the plot in linear
                           units first to check whether that quantity is 0.
            Possibility 2: You selected tight axis and the data is precisely 
                           at that limit, thus being hidden by the frame
    """
    
    
    # videos, gifs, results printing, etc..
    all_non_plots_available = [10.5, 10.55, 10.6, 10.65, 17, 17.01, 17.02,
                               17.03, 17.11, 17.12, 17.13]
    
    # All that can be saved as figures.
    all_plots_available = [0.1, 1, 1.1, 1.2, 2, 2.1, 2.15, 2.2, 2.3, 2.4, 3, 
                           3.1, 3.2, 3.3, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 4.1, 
                           4.2, 4.3, 5.1, 5.15, 5.2, 5.3, 5.4, 5.5, 5.6, 5.65, 
                           7.1, 7.2, 7.3, 7.35, 7.4, 7.5, 9.1, 9.2, 9.3, 9.4,
                           10.1, 10.15, 10.2, 10.25, 10.3, 10.31, 10.4, 10.45, 
                           10.7, 10.8, 10.9, 10.11, 11, 11.1, 11.2, 11.3, 
                           13, 14.1, 14.2, 15, 16, 16.1, 16.2]

    all_idxs_available = all_plots_available + all_non_plots_available

    idxs_to_plot = [0.1, 1, 2, 3.45, 4.2, 5.4, 7.4, 10.45]

    # idxs_to_plot = all_plots_available
    
    # idxs_to_plot = [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.65, 5.15]
    idxs_to_plot = [17.03]
    # Test save_plot
    save_plots = True
    saveformat = 'pdf' # supported: 'png', 'svg', 'pdf'
    
    base_plots_folder = 'Plots\\' 
    
    for i in idxs_to_plot:
        print(f'Plotting {i}')
        
        # Get which vars need to be loaded and which need to be computed
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
        
        # TODO: multitrace:
        # Compute additional data: 
            # - some variables might be computed already (check if empty)
            # - first, compute the auxiliar variables that might be needed
            # - second, compute what the actual index needs 
            #   e.g. some average of one of the variables across traces require
            #        said variables to be computed already (unless they are 
            #        trimmed vars.) 
        
        
        if multi_trace:
            raise Exception('not ready yet...')
        
        plt_func.compute_sim_data(i, layer, ues, ttis, VARS_NAME_LOAD, 
                                  VARS_NAME_COMPUTE, which_vars_to_compute, 
                                  which_vars_to_load, sim_data_trimmed, 
                                  sim_data_computed, file_set, 
                                  vars_with_layers)
        
        # Plots:
        plt_func.plot_sim_data(i, file_set, layer, ues, ttis, x_vals, 
                               sim_data_trimmed, sim_data_computed,
                               results_filename, base_plots_folder, 
                               save_plots, save_format=saveformat)
    
    
#%% Note: get's unstable with >20 plots. < 10 is the safest.
auto_report = False
if auto_report:
    base = ut.get_cwd() + '\\'    
    folder = r'Plots\SEED1_SPEED-1_FREQ-0_CSIPER-20_APPBIT-100_USERS-None_BW-50_LATBUDGET-10' + '\\'
    files_with_format = ut.get_all_files_of_format(base + folder, '.pdf')
    
    files_with_format = [folder + f for f in files_with_format]
    
    with open('pdf_merge.pdf', 'wb') as fp:
        ut.pdf_cat(files_with_format, fp)

