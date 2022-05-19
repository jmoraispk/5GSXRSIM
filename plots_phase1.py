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
To add a new plot index:
1- If it needs a new variable to be loaded from simulation:
      add what variables need to be loaded in get_vars_to_load(). See the 
      meaning of each code in the VARS_TO_LOAD variable in this file.
2- Add what variables need to be computed in get_vars_to_compute()
      If a new variable needs to be computed, add the index 
      len(VARS_NAME_COMPUTE) to the list VARS_NAME_COMPUTE and name your 
      variable. Then, include that index in get_vars_to_compute().
3- If in step 2 a new computation index was added, now add the computation 
      method in compute_sim_data().
4- Add the plot method in plot_sim_data(): 
      data is either in sim_data_trimmed[f][idx] or sim_data_computed[f][idx],
      where f is the file index (for single-trace plots it is always 0, 
      but for multi-trace plots it refers to the file in that index on the
      file list)
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
stats_folder = r'C:\Users\Srijan\Documents\SXRSIMv3\Stats' + '\\'
seeds_list = [1]
speeds = [3]
csi_periodicities = [5]
app_bitrates= [100]
users = [4]
bandwidths = [400] #50 MHz
latencies = [10]
freq_idxs = [0]
results_folder = r'Results\Batch X - testing' + '\\'
rot_factors_list = [10]
# rot_factors_list = [14]

for seeds in seeds_list:
    for rot_factor in rot_factors_list:
        layer = 0
        trim_ttis = [20, int(4000 * 8)]#[301, 370]
        
        TTI_dur_in_secs = 0.25e-3
        
        ttis = np.arange(trim_ttis[0], trim_ttis[1])
        x_vals = ttis #ttis * TTI_dur_in_secs
        
        
        # From the simulated UEs, which UEs do we want to plot?
        ues = [i for i in range(16)] #[5,9]
        # ues = [0, 5, 10, 15] 
        
        
        #----------------------
        
        multi_trace = 0
        
        combinations = list(itertools.product(speeds, freq_idxs, 
                                              csi_periodicities, app_bitrates,
                                              users, bandwidths, latencies, 
                                              {rot_factor}, {seeds}))
        
        
        # the variables to be loaded at each position of sim_data_(loaded/trimmed)
        # NOTE: ALWAYS ADD VARIABLES AT THE END!!
        VARS_NAME_LOAD = [
            'sp',                       #  0 
            'buffers',                  #  1 
            'realised_SINR',            #  2 [tti] x [ue] x [layer]
            'estimated_SINR',           #  3 [tti] x [ue] x [layer]
            'realised_bitrate',         #  4 [tti] x [ue] x [layer]
            'blocks_with_errors',       #  5 [tti] x [ue] x [layer]
            'n_transport_blocks',       #  6 [tti] x [ue] x [layer]
            'beams_used',               #  7 [tti] x [ue] x [layer] x 2
            'olla',                     #  8 [tti] x [ue]
            'mcs_used',                 #  9 [tti] x [ue] x [layer]
            'experienced_signal_power', # 10 [tti] x [ue] x [layer]
            'sig_pow_per_prb',          # 11 [tti] x [ue] x [layer] x [prb]
            'real_dl_interference',     # 12 [tti] x [ue] x [layer]
            'est_dl_interference',      # 13 [tti] x [ue] x [layer]
            'est_scheduled_layers',     # 14 [tti] x [ue]
            'scheduled_UEs',            # 15 [tti] x [ue]
            'channel',                  # 16 [tti] x [ue]
            'channel_per_prb',          # 17 [tti] x [ue] x [prb]
            'power_per_beam',           # 18 [tti] x [ue] x [layer] x [beam]
            'real_scheduled_layers',    # 19 [tti] x [ue]
            'est_dl_fin_interference',  # 20 [tti] x [ue] x [layer]
            '']
        
        # Variable names that can be computed from the loaded and trimmed variables
        # 1L means it is specific to the layer we specified, it does not have the 
        # information of both layers
        VARS_NAME_COMPUTE = [
            'sinr_diff',                         # 0  [tti] x [ue] x [layer]
            'running_avg_bitrate',               # 1  [tti] x [ue] x [layer]
            'rolling_avg_bitrate',               # 2  [tti] x [ue] x [layer]
            'instantaneous_bler',                # 3  [tti] x [ue] x [layer]
            'running_avg_bler',                  # 4  [tti] x [ue] x [layer]
            'signal_power_db',                   # 5  [tti] x [ue] x [layer]
            'signal_power_prb_db',               # 6  [tti] x [ue] x [layer]
            'real_interference_db',              # 7  [tti] x [ue] x [layer]
            'est_interference_db',               # 8  [tti] x [ue] x [layer]
            'beam_formula_simple',               # 9  [tti] x [ue] x [layer]
            'beam_sum',                          # 10 [tti] x [ue] x [layer]
            'freq_vec',                          # 11 [prb]
            'frames',                            # 12 [frame]
            'I_frames',                          # 13 [frame] x [ue]
            'avg_packet_lat',                    # 14 [frame] x [ue]
            'avg_packet_drop_rate',              # 15 [frame] x [ue]
            'avg_pck_lat_per_frame',             # 16 [ue]
            'avg_pck_drop_rate_per_frame',       # 17 [ue]
            'avg_pck_lat_per_I_frame',           # 18 [ue]
            'avg_pck_lat_per_P_frame',           # 19 [ue]
            'avg_pck_drop_rate_per_I_frame',     # 20 [ue]
            'avg_pck_drop_rate_per_P_frame',     # 21 [ue]
            'avg_pck_lat_per_frame_in_gop',      # 22 [frame in gob] x [ue]
            'avg_pck_drop_rate_per_frame_in_gop',# 23 [frame in gob] x [ue]
            'count_ues_scheduled',               # 24 [tti]
            'count_ues_bitrate',                 # 25 [tti]
            'beam_formula_processed',            # 26 [tti] x [ue] x [layer]
            'gop_idxs',                          # 27 [frame in gob]
            'power_per_gob_beam',                # 28 [tti] x [ue] x [layer] x [beam]
            'x_projection_best_beam',            # 29 [tti] x [ue] x [layer]
            'y_projection_best_beam',            # 30 [tti] x [ue] x [layer]
            'beam_switch',                       # 31 [tti] x [ue] x [layer]
            'xy_projection_all_gob',             # 32 [beam] x 2
            'user_pos_for_plot',                 # 33 
            'user_ori_for_plot',                 # 34
            'individual_beam_gob_details',       # 35
            'beams_processed',                   # 36
            'avg_sinr',                          # 37 [ue] x [layer]
            'avg_sinr_multitrace',               # 38 [ue] x [layer]
            '']
        
        # (Loaded) Vars with information per layer
        vars_with_layers = [2,3,4,5,6,7,9,10,12,13]
        
        # file_sets has the sets of files to load at any given time.
        # e.g. if we want to make a plot for each seed, we just want to load one seed
        #      at a time. But if we want to make a plot that is the average of 3 seeds
        #      we need to load those 3 seeds to compute the average.
        file_sets = []
        
        # Create the file set combinations from the variables given previously
        
        for comb in combinations:
        
            # stats_dir_end = f'SEED{comb[-1]}_SPEED-{comb[0]}_FREQ-{comb[1]}_' + \
            #                 f'CSIPER-{comb[2]}_APPBIT-{comb[3]}_'+ \
            #                 f'USERS-{comb[4]}_BW-{comb[5]}_LATBUDGET-{comb[6]}' + '\\'
                            
            # stats_dir_end = f'SU_SEED{comb[-1]}_FREQ-{comb[1]}_' + \
            #                 f'CSIPER-{comb[2]}_'+ \
            #                 f'USERS-{comb[4]}_ROTFACTOR-{comb[7]}_LAYERS-1_COPH-1_L-1' + '\\'
            
            # stats_dir_end = r'Scenario1_MU_SEED3_FREQ-0_CSIPER-5_USERS-None_ROTFACTOR-None_LAYERS-1_COPH-1_L-1_Adaptive' + '\\'
            stats_dir_end = r'Scenario2_MU_SEED-1_FREQ-0_CSIPER-5_USERS-16_ROTFACTOR-10_LAYERS-1_COPH-1_L-2_2022-05-20_00h24m23s' + '\\'
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
            
            # See if the file to use is the last one simulated
            if file_set == ['']: 
                with open("last_stats_folder.txt", 'r') as fh:
                    file_set = [fh.readline()]
                    
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
            
            # Note: our variables with plotable data are:
            #       --> sim_data_trimmed
            #       --> sim_data_computed
        
                
            """
            plot_idx meanings:
        
            "X marks the spot" where implementation is still needed.
            
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
        X   11.5  -> UEs with bitrate vs signal power (linear) --> quite similar to .4
        
                 
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
            
            18.1 -> (print) Avg. SINR across time per UE
            
            19.1XX -> Plot layer 1 vs layer 2, per UE:
        X        19.12 ->  'realised_SINR',            #  2 
        X        19.13 ->  'estimated_SINR',           #  3 
        X        19.15 ->  'blocks_with_errors',       #  5 
        X        19.16 ->  'n_transport_blocks',       #  6 
        X        19.17 ->  'beams_used',               #  7
        X        19.19 ->  'mcs_used',                 #  9
        X        19.110 -> 'experienced_signal_power', # 10
        X        19.112 -> 'real_dl_interference',     # 12
        X        19.113 -> 'est_dl_interference',      # 13    
            
            19.2 -> ...
            
            20.0XX -> multi-layer plots for loaded variables        (0XX) and
                      multi-layer plots with computed variables too (1XX) 
        X        20.012 ->  'realised_SINR'             #  2
                ....
        X        20.137 ->  'avg_sinr'                  #  37
            """
            
            """ 
            Notes in case something looks weird in the plots:
                - Empty plots: 
                    Possibility 1: 10 * log10(0) = -inf -> this does not show in 
                                   logarithmic plots. Therefore, try the plot in linear
                                   units first to check whether that quantity is 0.
                    Possibility 2: You selected tight axis and the data is precisely 
                                   at that limit, thus being hidden by the frame
            """
            
            
            # videos, gifs, results savig, etc..
            all_non_plots_available = [10.55, 10.65, 17, 17.01, 17.02, 17.03, 17.11, 
                                       17.12, 17.13]
            
            # All that can be saved as figures.
            all_plots_available = [0.1, 1, 1.1, 1.2, 2, 2.1, 2.15, 2.2, 2.3, 2.4, 3, 
                                   3.1, 3.2, 3.3, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 4.1, 
                                   4.2, 4.3, 5.1, 5.15, 5.2, 5.3, 5.4, 5.5, 5.6, 5.65, 
                                   7.1, 7.2, 7.3, 7.35, 7.4, 7.5, 9.1, 9.2, 9.3, 9.4,
                                   10.1, 10.15, 10.2, 10.25, 10.3, 10.31, 10.4, 10.45, 10.65,11.1,  
                                   10.5, 10.6, 10.7, 10.8, 10.9, 10.11, 11, 11.1, 11.2,
                                   11.3, 13, 14.1, 14.2, 15, 16, 16.1, 16.2]
        
            all_idxs_available = all_plots_available + all_non_plots_available
        
            idxs_to_plot = [0.1, 1, 2, 3.45, 4.2, 5.4, 7.4, 10.45, 18.1]
        
            # idxs_to_plot = all_plots_available
            # idxs_to_plot = [i for i in all_plots_available if i >= 0]
            idxs_to_plot = [2, 3.2, 3.3, 3.6]
            idxs_to_plot = [2.3, 3.3, 3.6, 18.1]
            idxs_to_plot = [11.1, 3.2, 18.1 ]
            idxs_to_plot = [3.2]
            idxs_to_plot = [1, 1.1,2, 3.2, 11.4, 3.65, 10.65, 7.1, 7.2]
            # idxs_to_plot = [17.03]
            # idxs_to_plot = [16.1, 16.2, 18.1]
            idxs_to_plot = [1, 1.1, 11.3, 11.1, 10.2, 10.6, 10.45,10.65]
            idxs_to_plot = all_plots_available
            # idxs_to_plot = [0.1, 1, 2, 3.45, 4.2, 5.4, 7.4, 10.45, 18.1]
            idxs_to_plot = [1, 1.1, 10.65, 7.1, 2.3, 2, 11.1]
            # Test save_plot
            save_plots = False
            saveformat = 'pdf' # supported: 'png', 'svg', 'pdf'
            
            base_plots_folder = 'Plots\\' 
            
            plot_params_checked_for_file_set = False
            
            for i in idxs_to_plot:
                print(f'Plotting {i}')
                
                # Get which vars need to be loaded and which need to be computed
                which_vars_to_load = plt_func.get_vars_to_load(i, VARS_NAME_LOAD)
                which_vars_to_trim = [i for i in which_vars_to_load if i != 'sp']
                which_vars_to_compute = \
                    plt_func.get_vars_to_compute(i, VARS_NAME_COMPUTE)
                
                # Load data
                plt_func.load_sim_data(file_set, VARS_NAME_LOAD, 
                                       which_vars_to_load, sim_data_loaded)
                
                if not plot_params_checked_for_file_set:
                    # Make layer and trim indices verifications by accessing the 
                    # Simulation Parameters Object (located at index 0) and checking 
                    # whether the simulation supports the specified layer and trimming
                    plot_params_checked_for_file_set = True
                    
                    for f in range(len(file_set)):
                        sim_layers = sim_data_loaded[f][0].n_layers
                        if layer < 0 or layer >= sim_data_loaded[f][0].n_layers:
                            raise Exception(f'Layer {layer} not supported. The ' \
                                            f'simulation was executed for {sim_layers}')
                        
                        sim_TTIs = sim_data_loaded[f][0].sim_TTIs
                        if trim_ttis[0] < 0 or trim_ttis[1] < 0 or \
                           trim_ttis[0] > sim_TTIs or trim_ttis[1] > sim_TTIs:
                            raise Exception(f'Trim TTIs {trim_ttis} is not supported. ' \
                                            f'The simulation was executed from 0 to ' \
                                            f'{sim_TTIs}.')
                
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
                                          which_vars_to_trim, sim_data_trimmed, 
                                          sim_data_computed, file_set, 
                                          vars_with_layers)
                
                # Plots:
                plt_func.plot_sim_data(i, file_set, layer, ues, ttis, x_vals, 
                                        sim_data_trimmed, sim_data_computed,
                                        results_filename, base_plots_folder, 
                                        save_plots, save_format=saveformat)
            
        
        
        #%% Trick to verify the shape of every variable:
        # 1- comment out the plot_sim_data function above
        # 2- run the cell above with idxs_to_plot = all_plots_available
        # 3- Run this cell afterwards.
        print_vars_shape = True
        if print_vars_shape:
            for i in range(len(VARS_NAME_LOAD)):
                if sim_data_trimmed[0][i] is not None:
                    print(f'{i:>2}: ', end='')
                    try:
                        print(f'{VARS_NAME_LOAD[i]: <25} has shape ' 
                              f'{sim_data_trimmed[0][i].shape}')
                    except AttributeError:
                        print(f"{VARS_NAME_LOAD[i]: <25} does "
                               "not have the attribute 'shape'")
            
            for i in range(len(VARS_NAME_COMPUTE)):
                if sim_data_computed[0][i] is not None:
                    print(f'{i:>2}: {VARS_NAME_COMPUTE[i]: <35} has shape '
                          f'{sim_data_computed[0][i].shape}')
        
        #%% To print an automatic report (all plots in a single PDF)
        #   Note: it gets unstable for > 20 plots. < 10 is the safest.
        auto_report = False
        if auto_report:
            base = ut.get_cwd() + '\\'    
            folder = r'Plots\SEED1_SPEED-1_FREQ-0_CSIPER-20_APPBIT-100_USERS-None_BW-50_LATBUDGET-10' + '\\'
            files_with_format = ut.get_all_files_of_format(base + folder, '.pdf')
            
            files_with_format = [folder + f for f in files_with_format]
            
            with open('pdf_merge.pdf', 'wb') as fp:
                ut.pdf_cat(files_with_format, fp)
        
