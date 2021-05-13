# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:39:35 2020

@author: Morais
"""

import numpy as np

# %matplotlib auto
# %matplotlib inline

import itertools
import utils as ut
import plots_functions as plt_func

# TODO: merge this file with multitrace plot
#       there's a feature there that would be very useful here for obtaining
#       results faster, the selective loading feature. Here, all variables are
#       loaded even if we only want to compute things related with one of them
#       there, for most variables, depending on the plots we want to make, only
#       the necessary variables are loaded from the statistics folder.

"""
Strategy: 
    1- Make sure the loop is printing the directories in the correct order;
    2- Several things must match between plot_looper and sls_plot:
        2.1 - "last_stats_folder" - the folder where the directory to load is 
               written. If we are doing this for multiple stats file 
               simultaneously, the folder needs to be different when we run
               this script or else when the reading time comes, 
    
    
"""
# This variable tells us whether we are running in a loop or independently.
use_in_loop = False


#-------------------------
# 1- Test for Speeds
# stats_folder = r"C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\SPEED_sims_20seeds" + '\\'
# seeds = np.arange(1,21)
# speeds = [1,2,3,4,5]
# csi_periodicities = [1]
# app_bitrates = [75]
# freq_idxs = [1]
# trim_secs = 32
# results_folder = r'Results\Batch 1 - speeds' + '\\'

# 2- Test CSI periodicities
# stats_folder = r"C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\CSI_per_sims_40seeds" + '\\'
# seeds = np.arange(1,41)
# speeds = [3]
# csi_periodicities = [1, 2, 5, 10, 20, 40, 80, 200]
# app_bitrates = [75]
# freq_idxs = [1]
# trim_secs = 32
# results_folder = r'Results\Batch 2 - csi periodicities' + '\\'


# 3- Test Application Bitrates
# stats_folder = r"C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\AppBitrate_sims_20seeds" + '\\'
# seeds = np.arange(1,21)
# speeds = [3]
# csi_periodicities = [1]
# app_bitrates= [25, 50, 75, 100, 125, 150, 175, 200]
# freq_idxs = [0]
# trim_secs = 32
# results_folder = r'Results\Batch 3 - application bit rates' + '\\'


# 4- Test Number of Users and Bandwidths
# stats_folder = r"C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\MU_10seeds_6bws_4nues" + '\\'
# seeds = np.arange(1,11)
# speeds = [3]
# csi_periodicities = [1]
# app_bitrates= [75]
# users = [2,4,6,8]
# bandwidths = [10, 20, 40, 60, 80, 100] # MHz
# freq_idxs = [1]
# trim_secs = 32
# results_folder = r'Results\Batch 4 - number of users and bandwidths' + '\\'


# 5- Test Number of Users and Bandwidths
# stats_folder = r"C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\NewTracks" + '\\'
# seeds = np.arange(5,8)
# speeds = [1, 3, 5]
# csi_periodicities = [20]
# app_bitrates= [100]
# users = [None]
# bandwidths = [40] # MHz
# freq_idxs = [1]
# trim_secs = 16
# results_folder = r'Results\Batch 5 - testing new tracks' + '\\'

# 6- Test Kappas
# stats_folder = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\kappa MLWDF' + '\\'
# seeds = np.arange(1,11)
# speeds = [3]
# csi_periodicities = [20]
# app_bitrates= [100]
# users = [None]
# bandwidths = [40] # MHz
# kappas = [1, 6, 20]
# freq_idxs = [0]
# trim_secs = 16
# results_folder = r'Results\Batch 6 - kappas and schedulers' + '\\'

# 7- Speed test (yes, again!)
# stats_folder = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\LastBatchSpeeds' + '\\'
# seeds = np.arange(1,21)
# speeds = [1, 3, 5]
# csi_periodicities = [20]
# app_bitrates= [100]
# users = [None]
# bandwidths = [40] # MHz
# freq_idxs = [0] ###################################3 try 1
# trim_secs = 16
# results_folder = r'Results\Batch 7 - new tracks speeds' + '\\'

# 8- Latencies (yes, again!)
stats_folder = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\LastBatchLatencies' + '\\'
seeds = [1,2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20]
speeds = [3]
csi_periodicities = [20]
app_bitrates= [100]
users = [None]
bandwidths = [50] # MHz
latencies = [5, 10, 20, 30, 40, 50]
freq_idxs = [0, 1]
trim_secs = 16
results_folder = r'Results\Batch 12 - filtered latencies' + '\\'

# 9- CSIs (last batch)
# stats_folder = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\LastBatchCSIs' + '\\'
# seeds = np.arange(1,21)
# speeds = [3]
# csi_periodicities = [40, 80, 200, 400]
# app_bitrates= [100]
# users = [None]
# bandwidths = [50] # MHz
# latencies = [10]
# freq_idxs = [1]
# trim_secs = 16
# results_folder = r'Results\Batch 9 - CSIs' + '\\'

# 10- Speeds tests with CSI 50 ms
# stats_folder = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\vsSPEED7' + '\\'
# seeds = [1,2,3]
# speeds = [3, 7]
# csi_periodicities = [80]
# app_bitrates= [100]
# users = [None]
# bandwidths = [50] # MHz
# latencies = [10]
# freq_idxs = [0,1]
# trim_secs = 16
# results_folder = r'Results\Batch 11 - vsSPEED7_csi80' + '\\'
#----------------------


if not use_in_loop:
    results_filename = ''
    ut.stop_execution() # No need to continue this cell.


# combinations = list(itertools.product(speeds, freq_idxs, 
#                                       csi_periodicities, app_bitrates, seeds))
    
combinations = list(itertools.product(speeds, freq_idxs, 
                                      csi_periodicities, app_bitrates,
                                      users, bandwidths, latencies, seeds))

# combinations = list(itertools.product(speeds, freq_idxs, 
#                                       csi_periodicities, app_bitrates,
#                                       users, bandwidths, kappas, seeds))

for comb in combinations:

    # stats_dir_end = f'SEED{comb[4]}_SPEED-{comb[0]}_FREQ-{comb[1]}_' + \
    #                 f'CSIPER-{comb[2]}_APPBIT-{comb[3]}' + '\\'
                    
    stats_dir_end = f'SEED{comb[-1]}_SPEED-{comb[0]}_FREQ-{comb[1]}_' + \
                    f'CSIPER-{comb[2]}_APPBIT-{comb[3]}_'+ \
                    f'USERS-{comb[4]}_BW-{comb[5]}_LATBUDGET-{comb[6]}' + '\\'
                    
                    
    # stats_dir_end = f'SEED{comb[-1]}_SPEED-{comb[0]}_FREQ-{comb[1]}_' + \
    #                 f'CSIPER-{comb[2]}_APPBIT-{comb[3]}_'+ \
    #                 f'USERS-{comb[4]}_BW-{comb[5]}_KAPPA-{comb[6]}' + '\\'
                    
    print(f'\nDoing for: {stats_dir_end}')
    
    stats_dir = stats_folder + stats_dir_end
    
    
    if use_in_loop:
        extra_str = f'_f{comb[1]}_{trim_secs}s'
    else:
        extra_str = ''
        
    results_filename = results_folder + 'results' + extra_str
    
    if not ut.isdir(results_folder):
        ut.makedirs(results_folder)    
    
    #%%
    """
    The file, from here onwards, can is for single-trace plots. 
    The plan is:
        1- generate different frequencies at the same time: we have reached a
           point where it takes longer to setup generations and to generate 
           separate builders than to run a less optimal configuration. Therefore,
           we generate all frequencies in the same trace;
        2- simulate different frequencies and different SEEDS separately, thus
           creating different stats files;
        3- It is based on those stats files that we make our analysis, so they 
           need to be named properly in order to differentiate SEEDS, frequencies,
           time divisions, simulation durations, etc... otherwise the traces
           probably should not be compared.
        4- Use this file to analyse one trace extensively;
        5- Use sls_plot_multitrace.py to agglomerate and compute statistics across
           different traces.
    """
    
    
    # Work around to avoid computing the variables multiple times... ignore. 
    stats_dir_temp = ''
    first_tti_temp = 0
    last_tti_temp = 0
    instantaneous_bler, running_avg_bler, beam_formula, avg_lat, drop_rate, \
        frames, n_frames, I_frames, running_avg_bitrate, rolling_avg_bitrate = \
            tuple([None] * 10) 
    
    # Loading variables step
    
    # To get the last stats directory (required if use_in_loop is True)
    open_last = False
    if not use_in_loop:
        if open_last:
            with open("last_stats_folder.txt", 'r') as fh:
                stats_dir = fh.readline()
        else:
            # my_dir = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv2\Stats\NewTracks' + '\\'
            # stats_name = r'SEED6_SPEED-3_FREQ-0_CSIPER-20_APPBIT-100_USERS-None_BW-40'
            
            my_dir = r'C:\Users\Morais\Documents\SXR_Project\SXRSIMv3\Stats' + '\\'
            stats_name = r'SEED3_SPEED-3_FREQ-1_CSIPER-20_APPBIT-100_USERS-None_BW-50_LATBUDGET-10'
            stats_dir = my_dir + stats_name + '\\'
    
    
    sp = ut.load_var_pickle('sp', stats_dir)
    buffers = ut.load_var_pickle('buffers', stats_dir)
    
    # Set to True if we just want to compute the PDR!
    getting_results = use_in_loop # Most cases we want them to have the same value 
    
    if not getting_results:
        realised_SINR = ut.load_var_pickle('realised_SINR', stats_dir)
        estimated_SINR = ut.load_var_pickle('estimated_SINR', stats_dir)
        realised_bitrate_total = ut.load_var_pickle('realised_bitrate_total', stats_dir)
        blocks_with_errors = ut.load_var_pickle('blocks_with_errors', stats_dir)
        n_transport_blocks = ut.load_var_pickle('n_transport_blocks', stats_dir)
        beams_used = ut.load_var_pickle('beams_used', stats_dir)
        olla = ut.load_var_pickle('olla', stats_dir)
        mcs_used = ut.load_var_pickle('mcs_used', stats_dir)
        real_dl_interference = ut.load_var_pickle('real_dl_interference', stats_dir)
        est_dl_interference = ut.load_var_pickle('est_dl_interference', stats_dir)
        scheduled_UEs = ut.load_var_pickle('scheduled_UEs', stats_dir)
        su_mimo_setting = ut.load_var_pickle('su_mimo_setting', stats_dir)
        
        experienced_signal_power = ut.load_var_pickle('experienced_signal_power', stats_dir)
        channel = ut.load_var_pickle('channel', stats_dir)
        if sp.save_per_prb_variables:
            sig_pow_in_prb = ut.load_var_pickle('sig_pow_in_prb', stats_dir)
            channel_per_prb = ut.load_var_pickle('channel_per_prb', stats_dir)
        else:
            signal_power_prb = None
            channel_per_prb = None
    else:
        realised_SINR, estimated_SINR, realised_bitrate_total, \
            blocks_with_errors, n_transport_blocks, beams_used, olla, \
            mcs_used, real_dl_interference, est_dl_interference, \
            scheduled_UEs, su_mimo_setting, experienced_signal_power, \
            channel, signal_power_prb, channel_per_prb = tuple([None] * 16) 
    
    # realised_SINR = ut.load_var_pickle('realised_SINR', stats_dir)
    # estimated_SINR = ut.load_var_pickle('estimated_SINR', stats_dir)
    # Trimming the full trace in time
    trim_sec_or_tti = 'tti'
    
    if trim_sec_or_tti == 'tti':
        # Fill this:
        first_tti = int(4000 * 0.005)
        if use_in_loop:
            last_tti = int(4000 * trim_secs)
        else:
            last_tti = int(4000 * 1)
            
        # automatic:
        first_sec = first_tti * sp.TTI_dur_in_secs
        last_sec = last_tti * sp.TTI_dur_in_secs
    elif trim_sec_or_tti == 'sec':
        # Fill this:
        first_sec = 0.005
        last_sec = 1
        
        # automatic:
        first_tti = int(first_sec / sp.TTI_dur_in_secs)
        last_tti = int(last_sec / sp.TTI_dur_in_secs)
    else:
        raise Exception("Trim on 'secs' (seconds) or 'ttis'")
    
    ttis = np.arange(first_tti, last_tti)
    secs = np.arange(first_sec, last_sec, sp.TTI_dur_in_secs)
    
    
    # Plotting options
    
    x_sec_or_tti = 'sec'
    
    if trim_sec_or_tti == 'tti':
        x_vals = secs
        x_vals_label = 'Time [s]'
        x_vals_save_suffix = '_secs'
    elif trim_sec_or_tti == 'sec':
        x_vals = ttis
        x_vals_label = 'Time [TTI]'
        x_vals_save_suffix = '_ttis'
    else:
        raise Exception("Trim on 'secs' (seconds) or 'ttis'")
    
    
    ues = [i for i in range(sp.n_phy)]
    
    if not getting_results:
        # Convert to np arrays and trim for ploting
        bitrate_realised = np.array(realised_bitrate_total)[first_tti:last_tti]
        sinr_realised = np.array(realised_SINR)[first_tti:last_tti]
        sinr_estimated = np.array(estimated_SINR)[first_tti:last_tti]
        block_errors = np.array(blocks_with_errors)[first_tti:last_tti]
        n_blocks = np.array(n_transport_blocks)[first_tti:last_tti]
        beams = np.array(beams_used)[first_tti:last_tti]
        olla_param = np.array(olla)[first_tti:last_tti]
        mcs = np.array(mcs_used)[first_tti:last_tti]
        dl_interference = np.array(real_dl_interference)[first_tti:last_tti]
        dl_interference_est = np.array(est_dl_interference)[first_tti:last_tti]
        UEs_scheduled = np.array(scheduled_UEs)[first_tti:last_tti]
        su_mimo_layers = np.array(su_mimo_setting)[first_tti:last_tti]
        
        avg_channel = np.array(channel)[first_tti:last_tti]
        signal_power = \
            np.array(experienced_signal_power)[first_tti:last_tti]
        
        if sp.save_per_prb_variables:
            signal_power_prb = np.array(sig_pow_in_prb)[first_tti:last_tti]
            channel_per_prb = np.array(channel_per_prb)[first_tti:last_tti]
        
        # UL tti formula, if first and last tti are multiples of 5.
        # ul_ttis = np.arange(0, last_tti - first_tti, 5) - 1
        # ul_ttis = ul_ttis[1:]
        
        # # Select the downlink ttis only
        # bitrate_realised = np.delete(bitrate_realised, ul_ttis, axis=0)
        # sinr_realised = np.delete(sinr_realised, ul_ttis, axis=0)
        # sinr_estimated = np.delete(sinr_estimated, ul_ttis, axis=0)
        # block_errors = np.delete(block_errors, ul_ttis, axis=0)
        # beams = np.delete(beams, ul_ttis, axis=0)
        # signal_power_prb = np.delete(signal_power_prb, ul_ttis, axis=0)
        # olla_param = np.delete(olla_param, ul_ttis, axis=0)
        # mcs = np.delete(mcs, ul_ttis, axis=0)
        
        
        # Select the layer we want (single-layer plot)
        l_idx = 0
        sinr_realised = sinr_realised[:,:,l_idx]
        sinr_estimated = sinr_estimated[:,:,l_idx]
        block_errors = block_errors[:,:,l_idx]
        n_blocks = n_blocks[:,:,l_idx]
        beams = beams[:,:,l_idx,:]
        mcs = mcs[:,:,l_idx]
        dl_interference = dl_interference[:,:,l_idx]
        dl_interference_est = dl_interference_est[:,:,l_idx]
    
        if sp.save_per_prb_variables:
            signal_power_prb = signal_power_prb[:,:,l_idx]
            channel_per_prb = channel_per_prb[:,:,l_idx]
    else:
        bitrate_realised, sinr_realised, sinr_estimated, block_errors, \
            n_blocks, beams, olla_param, mcs, dl_interference, \
            dl_interference_est, UEs_scheduled, su_mimo_layers, avg_channel, \
            signal_power, signal_power_prb, channel_per_prb = tuple([None] * 16) 
        
    
    # sinr_realised = np.array(realised_SINR)[first_tti:last_tti]
    # sinr_estimated = np.array(estimated_SINR)[first_tti:last_tti]
    # sinr_realised = sinr_realised[:,:,2]
    # sinr_estimated = sinr_estimated[:,:,2]
    """
        plot_idx:
        0.X  -> Channel Power - Note: computations are computationally demanding!
                 Depending on the second decimal place, we compute ther powers in
                 two different ways:
        0.1   -> across time (mean power over prbs)                        
    TODO    0.2   -> across time (for all prbs, 1 ue or 4 ues)
    TODO    0.3   -> across prbs (for a given tti)
         
        1     -> Throughput
        1.1   -> 
        
        2     -> SINR (multi-user) estimated vs realised
        2.1   -> SINR (single-user) when there are active transmissions
        2.2   -> SINR (multi-user) when there are active transmissions
        
        3     -> Signal power per PRB in Watt (for tti = 3)
        3.1   -> Signal power per PRB in dB (for tti = 3, middle prb as reference)
        3.2   -> Signal power vs Interference power
        
        4.1   -> MCS per user, same axis
        4.2   -> MCS per user, diff axis
        
        5.1   -> Beams per user
        5.2   -> Beams per user filtered: keeps beam value constant instead of
                 allowing going to 0 when the UE is not scheduled - better for 
                 cases where not all ues are scheduled simultaneously always.
        5.3   -> Beams per user: Same as 5.2 but with one plot per UE.
        
    TODO    6     -> Beamformed Channel: apply the beams to the channel and see 
                      what the UE is actually experiencing: this should justify 
                      signal power oscillations
        
        7.1   -> BLER: instantaneous
        7.2   -> BLER: running_average
        7.3   -> BLER: instantaneous vs running average
        7.4   -> BLER: instantaneous vs realised bit rate [doublePLOT]  -> NEEDS FIXIN!
        
        9.1   -> OLLA:  instantaneous BLER vs olla parameter (single-user)
                        [when active transmissions] [doublePLOT]
        9.2   -> OLLA:  instantaneous BLER vs olla parameter (multi-user)
                        [when active transmissions] [doublePLOT]
        9.3   -> OLLA: MCS vs olla param [doublePLOT]
        9.4   -> OLLA: instantaneous BLER vs olla param [doublePLOT]
        
        10    -> Latency and Drop Rate: average latency and packet drop rate per frame 
        10.1  -> Just average Latency
        10.2  -> Just 
            
        11    -> Scheduled UEs: sum of co-scheduled UEs across time
        11.1  -> Scheduled UEs: each UE is 1 when it is scheduled and 0 when not
        11.2  -> Scheduled UEs: each UE is 1 when it is scheduled and 0 when not
                                (all UEs in the same axis)
        11.3  -> UEs with bitrate: each UE. There's a difference between having
                 bitrate and being scheduled! The schedule is only updated when
                 there's a scheduling update... However, the user can be added 
                 to the schedule and get no (useful) bitrate. It will still get
                 bits across, but those might have no utility because they have
                 transferred before.
                                
        
    TODO 12    -> deeper dive into the application: latencies and drop rates 
                  based on averages across frames
    
        13    -> SU-MIMO setting - number of layers scheduled per UE
        
        14    -> plot packet sequences for each UE. (same axis)
        14.1  -> plot packet sequences for each UE. (separate axis)
        
        Note: there are idxs that don't plot anything. They just compute and 
        print data and are completely carried out in compute_set functions
        
        
        IMPORTANT: THE MEANING OF THESE INDICES MUST BE CONSISTENT WITH 
                   THE COMPUTATION FUNCTION AND PLOT FUNCTION!
        
    """
    
    """
    Common errors without warnings:
        - Index error - because of trimming indexes bigger than sim duration
    """    
    
    #idxs_to_plot = [1,2,2.1,4.1,7.4,10]
    
    
    idxs_to_plot = [0.1, 1, 2, 3.45, 3.7, 4.2, 5.4, 7.35, 7.4, 10.45, 14.2]
    idxs_to_plot = [0.1, 1, 2, 3.45, 4.2, 5.4, 7.35, 7.4, 10.45]
    # idxs_to_plot = [7.35]
    # idxs_to_plot = [5.4]
    idxs_to_plot = [2, 5.4, 5.6, 5.7, 7.5, 10.9]
    idxs_to_plot = [0.1]
    
    # idxs_to_plot = [10.8]
    for i in idxs_to_plot:
        print(f'Plotting {i}')
        
        # Loads and computes
        
        # HOW WE CHECK IF COMPUTED DATA IS UP-TO-DATE:
        # what makes data up to date is the file and the tti cuts. if the ttis cuts 
        # did not change, neither did the file, then the data computed for that 
        # ocasion is the up-to-date because no other configuration was loaded.
        # And in that case, we don't compute them again.
        
        instantaneous_bler, running_avg_bler, beam_formula, \
            avg_lat, drop_rate, frames, n_frames, I_frames, \
            running_avg_bitrate, rolling_avg_bitrate = \
                plt_func.compute_set_1(i, ues, x_vals, ttis, 
                                       block_errors, n_blocks, beams,
                                       buffers, sp, 
                                       bitrate_realised, stats_dir_temp, 
                                       first_tti_temp, last_tti_temp, stats_dir, 
                                       first_tti, last_tti, instantaneous_bler, 
                                       running_avg_bler, beam_formula, avg_lat, 
                                       drop_rate, frames, n_frames, I_frames,
                                       running_avg_bitrate, rolling_avg_bitrate,
                                       results_filename, use_in_loop)
                
        stats_dir_temp = stats_dir
        first_tti_temp = first_tti
        last_tti_temp = last_tti
        
        # Plots:
        plt_func.plot_set_1(i, False, ues, ttis, x_vals, x_vals_label, sp, buffers,
                            stats_dir, x_vals_save_suffix, bitrate_realised, 
                            signal_power, signal_power_prb, sinr_estimated, 
                            sinr_realised, olla_param, dl_interference, mcs, beams,
                            UEs_scheduled, avg_channel, su_mimo_layers, beam_formula, 
                            instantaneous_bler, running_avg_bler, avg_lat, 
                            drop_rate, frames, I_frames, running_avg_bitrate, 
                            rolling_avg_bitrate, dl_interference_est)
    
