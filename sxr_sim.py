# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:42:34 2020

@author: janeiroja
"""

# Imports of standard Python Libraries
import numpy as np
import itertools
import time
import pandas as pd
import os

# Own code imports
import sls
import utils as ut
import application_traffic as at
import simulation_parameters as sim_par


# Run it in a cell separate for now  
# Different traces for different video qualities/bitrates

"""
init_pcap = not True # No need for init anymore with pre-modified pcap_traces
# def init_pcap_file():
if init_pcap:
#####
    pcap_folder = r"C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\Trace" + '\\'
    pcap_to_simulate = pcap_folder + "trace_APP10.csv"
    # Inputs: 
    tti_dur = 0.00025
    total_ttis = 4000
    burst_param = 0.5
    # Burstiness model (New from Zheng or even more realistic FIFO queue model)
    # ['Joao', 'Zheng', 'Queue']
    burst_model = 'Zheng'
    # burst_model = 'Queue' -> already modified traces
    pcap_file = at.PCAP_File(pcap_to_simulate, tti_dur, total_ttis, burst_param, 
                         burst_model)     
#####

#     return pcap_file
# if init_pcap:    
#     pcap_file = init_pcap_file()
"""

"""
Zheng 
 
TODO: Sub-band scheduling:
      How to define/differentiate the PRBs and frequency samples 
      #PRBs and #Freq defined in sim_par??? 
      => Check mappings in official standards
      => Will variables, calculations etc. be independent for sub-bands???
      => Which are 'global'? E.g. UE buffers, inter-cell-interference, ...
      => Which are separate per sub-band? E.g. SINR, MCS, 'Precoders'(?)
      How to integrate into simulation loop???
      E.g.: 
      for (each TTI)::
          most simulation..
          for (each scheduling-PRBs/sub-band):
              do: simulation.....                   

Ideas:
      - Symbol-level scheduling? 
      - Mini-slots/Multiple slots?
    

"""
# # %%

# parent_folder = r"C:\Users\Morais\Documents\SXR_Project\SXRSIMv3\Matlab\TraceGeneration\CyclicTracks" + '\\'
# parent_folder = r"C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\Matlab\TraceGeneration" + '\\'
parent_folder = os.getcwd() + r"\\Matlab\\TraceGeneration\\"#  + "\\"
#seed = int(ut.get_input_arg(1)) # 1
#speed = int(ut.get_input_arg(2))
seed = 1
speed = 3


# folders_to_simulate = [f"SEED{seed}_SPEED{speed}"]
# folders_to_simulate = ["SEED1_SPEED1_point_centre"]
folders_to_simulate = []
seeds_to_simulate = []
for i in range(1,21):
    folders_to_simulate.append(f"SEED{i}_omni")   
    seeds_to_simulate.append(i)
    # , "Sim_SEED3", "Sim_SEED4"]
folders_to_simulate = [parent_folder + f for f in folders_to_simulate]

print("Seeds to simulate:", seeds_to_simulate)
# raise SystemExit
ticc = time.perf_counter()

freq_idxs = [0]
# csi_periodicities = [4, 8, 20, 40, 80, 200] # in TTIs
csi_periodicities = [5]

# Put to [None] when not looping users, and the user_list is manually set below
# users = [1,2,4,6,8]   
users = [None]
# users = [4]

# application_bitrates = [25, 50, 75, 100, 125, 150, 175, 200] # in Mbps
application_bitrates = [100]
# bandwidths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # MHz
bandwidths = [125] # MHz
# latencies = [10, 20, 30, 40, 50] # ms
# Check whether RAN or E2E-frame latency scheduling is used!!!
latencies = [80]
# E2E_lat = [100]

sim_params = list(itertools.product(folders_to_simulate, freq_idxs,
                                    csi_periodicities, application_bitrates,
                                    users, bandwidths, latencies))

# itertools.product does: 
#   [[1st element of 1st list, ..., 1st of last list], 
#    [1st element of 1st list, ..., 2nd element of last list], 
#    ... ]

# for param in sim_params:
#     print(param)
   
# ut.stop_execution()

ut.get_time()


for param in sim_params:
    # unpack simulation parameters
    sim_folder = param[0]
    freq_idx = param[1]
    csi_periodicity = param[2]
    application_bitrate = param[3]
    users = param[4]
    bw = param[5]
    lat_budget = param[6]
        
    if users != None:
        if users == 1:
            user_list = [0]
        if users == 2:
            user_list = [0, 4]
        elif users == 4:
            user_list = [0, 2, 4, 6]
        elif users == 6:
            user_list = [0, 1, 2, 4, 5, 6]
        elif users == 8:
            user_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            Exception('Not prepared for this number of users...')
    else:
        # when there were only 4 ues
        user_list = [i for i in range(4)]
    
    folder_idx = folders_to_simulate.index(sim_folder)

    # print('------ Setting up simulation parameters  ------')
    # start timer
    t_0 = time.time()
    
    # ----------------------------- SIMULATION --------------------------------
    
    # Initialise the simulation parameters
    sp = sim_par.Simulation_parameters(sim_folder, freq_idx, csi_periodicity,
                                       application_bitrate, user_list, bw, 
                                       lat_budget)
       
    # NOTE: 
        # a) users will subset the generated users;
        # b) bw will use the frequency samples of the generated bandwidht
        #    and consider they were generated for bw instead (expanding them!)
        #    We have proved that for wideband precoding and scheduling, it 
        #    result is exactly the same as dupplicating the samples. 
        
    # print('Done setting Simulation Parameters!')
    
    sim_dur = round(sp.sim_TTIs / 4000, 2)
    
    # Take care of the output
    include_timestamp = False 
    seed_str = folders_to_simulate[folder_idx].split('\\')[-1].split(' ')[0]
    output_stats_folder = '' #SPEED7' + '\\'
    if sp.use_pcap: 
        # if sp.delay_type == 'E2E':
        if (sp.scheduler == 'Frametype' or 
            sp.scheduler == 'Frametype-M-LWDF' or 
            sp.scheduler == 'Frametype-EDD'):
            output_str = \
                f'BW-{bw}_{sp.delay_type}-LAT-{lat_budget}_' + \
                f'LEN-{sim_dur}s_{sp.scheduler}-{sp.frametype_weight}_' + \
                f'Offset-{sp.space_UE_frames}'
        else:
            output_str = f'BW-{bw}_{sp.delay_type}-LAT-{lat_budget}_' + \
                     f'LEN-{sim_dur}s_{sp.scheduler}_' + \
                     f'Offset-{sp.space_UE_frames}'
                     # f'{seed_str}_' + \
                     # f'PCAP-{sp.use_pcap}_' + \
        # elif sp.delay_type == 'RAN':
        #     output_str = f'{seed_str}_' + \
        #                  f'BW-{bw}_{sp.delay_type}-LAT-{lat_budget}_' + \
        #                  f'LEN-{sim_dur}s_{sp.scheduler}_' + \
        #                  f'PCAP-{sp.use_pcap}_' + \
        #                  f'Offset-{sp.space_UE_frames}'
    else:       
        output_str = f'APPBIT-{application_bitrate}_' + \
                     f'BW-{bw}_LAT-{sp.delay_type}-{lat_budget}_' + \
                     f'LEN-{sim_dur}s_{sp.scheduler}_' + \
                     f'Offset-{sp.uniformly_space_UE_I_frames}'
                     # f'UEs-{users}_' + \
                         
                 # f'Burst-{sp.burstiness_model}-{sp.burstiness_param}'
                          # TODO: 
                 # SPEED-{sp.speed_idx} USERS-{users}_'FREQ-{freq_idx}_' + \
    # output_str = output_stats_folder + output_str
    
    # Continue the execution
    # print('Initialising variables...') 
    # print('Using the', sp.scheduler, 'scheduler')    
    # print(f'\nOutput folder: \n{sp.stats_dir}')
    print(f'\nSimulating: {output_str} - {seed_str}')
    # -------------------------------- START --------------------------------
      
    # Setup Application Traffic Model
    user_buffers = []
    # cam_buffers = [] # we assume cameras are wired.
    packet_sequences_DL = [0] * sp.n_phy
    # Compute offsets to space out user I frames.
    pcap_file_ues = [0] * sp.n_phy
    pcap_output_ues = [0] * sp.n_phy
    
    # TODO: How to implement offset with PCAP, if even possible / makes sense?
    if sp.uniformly_space_UE_I_frames:
        # I-frame order: UE 0-1-2-3
        I_frame_offsets = np.linspace(0, sp.GoP / sp.FPS, sp.n_phy + 1)[:-1]
        # Force UE0 to have biggest offset for debugging
        # I_frame_offsets = np.array([0.15, 0.1, 0.05 , 0.0]) 
        # I-frame order: UE 3-2-1-0
        # I_frame_offsets = np.linspace(0.15, 0.0, 4)
    else:
        I_frame_offsets = [0] * sp.n_phy    
        
    
    # TODO: Implement it as option that can be set in sim_par
    # Do this for every function from here onwards that uses the buffer??? 
    if sp.use_pcap:         
        # PCAP input parameters & files
        # pcap_folder = r"C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\Traces" 
        pcap_folder = os.getcwd() + "\\PCAP\Traces" 

        pcap_parameters = "\\trace_APP100_0.6\\" + "SEED1 - 15Q - 50.0% Load\\"
        trace_parameters = pcap_parameters.split('\\')[1] 
        final_trace = f"{trace_parameters}_0.0-17.0s.csv"
        trace_to_simulate = pcap_folder + pcap_parameters + final_trace        
        pcap_to_simulate = pd.read_csv(trace_to_simulate, encoding='utf-16 LE', 
                                       index_col=False)
        print('\nPCAP Trace to simulate:\n', pcap_parameters.strip("\\"))       
        tic = time.perf_counter()        

        for ue in range(0, sp.n_phy): # start at 1 for debug!!!
            # Generate frame sequences
        
            # At start of simulation, create packet sequence with all packets
            # arriving within the first TTI, i.e. the first 0.25 ms       
            tti_duration = sp.TTI_duration.total_seconds()  
            curr_tti = 0 
            # Initialize pcap objects for each UE with modified traces
            # Inputs: 
            pcap_file_ues[ue] = at.PCAP_File(pcap_to_simulate, tti_duration, 
                                    sim_dur, sp.sim_TTIs, sp.burstiness_param, 
                                    sp.burstiness_model, sp.space_UE_frames,
                                    ue, sp.n_phy)
                        
            pcap_output_ues[ue] = np.zeros(pcap_file_ues[ue].total_packets)     
            
            # if ue == 3: raise SystemExit
            # Create packet sequence      
            packet_sequences_DL[ue] = \
                at.gen_pcap_sequence(pcap_file_ues[ue], curr_tti)
        
        toc = time.perf_counter()        
        print(f"Init PCAP time: {toc-tic:0.4f} seconds.")
        
        # Create np.array to save output of all buffers 
        # -> Easier to use further for pcap traces
        # Or/Also: Save outputs as pickle        
        for ue in range(0, sp.n_phy): 
            # From the packet sequences, initialise the Buffers:
            # Buffers for each user, physically located at the BSs        
            user_buffers.append(at.PCAP_Buffer(packet_sequences_DL[ue], 
                                               sp.delay_threshold, 
                                               sp.delay_type, output_str, 
                                               sp.stats_dir, ue, sp.scheduler))
               
    else: 
        for ue in range(0, sp.n_phy):
            frame_sequence_DL = at.gen_frame_sequence(sp.I_size_DL,
                                                  sp.GoP,
                                                  sp.IP_ratio,
                                                  sp.FPS,
                                                  I_frame_offsets[ue])
            if sp.verbose:  
                print('DL frames:')
                frame_sequence_DL.print_frames()
            
            # Create Packet sequences
            packet_sequences_DL[ue] = \
                at.gen_packet_sequence(frame_sequence_DL, 
                                        sp.packet_size, 
                                        sp.burstiness_param,
                                        sp.burstiness_model,
                                        overlap_packets_of_diff_frames=0)
                                       
            if sp.verbose:
                print('DL packets:')
                #packet_sequences_DL.print_packets(first_x_packets=3)
                
                print('DL Packet Sequence')
                packet_sequences_DL[ue].plot_sequence()
                    
            # From the packet sequences, initialise the Buffers:
            # Buffers for each user, physically located at the BSs        
            user_buffers.append(at.Buffer(packet_sequences_DL[ue], 
                                          sp.delay_threshold, sp.delay_type))            

    # Merge user and camera buffers in general variable buffers.
    buffers = user_buffers # + cam_buffers
    # raise SystemExit 
    
    # Note: UEs can be both UL and DL. A better way to call buffers would be
    #       UL and DL. However, for our application, we consider UEs that only
    #       UL and UEs that only DL. Furthermore, the UL is wired. Nonetheless,
    #       if UL is used in the future, do:
    #       UL_buffers = cam_buffers
    #       DL_buffers = user_buffers
    #       And don't forget to complement across the simulator.
    
    
    if sp.n_prb > 1: 
        # Load into Memory the full information bits table necessary for MIESM
        info_bits_table = sls.load_info_bits_table(sp.info_bits_table_path)
    else:
        info_bits_table = None
    
    # Load into Memory the BS precoders
    # Note: there are only precoders in the DL. 
    #        UL is computed implicitly with MR, see find_best_beam in sls.py

    # Dictionary indexed by a tupple of bs and a given angle
    precoders_dict = sls.load_precoders(sp.precoders_paths, sp.vectorize_GoB)
    
    # In the precoders_folder there should be files with the
    # sp.precoder_file_prefix for the correct antennas 
    
    
    # System Level Simulator (SLS) part
    
    # Each UE has a precoder list, each having n_layers of beam_pairs, which
    # are pairs of RX/TX combiners/precoders between the UE and serving BS 
    curr_beam_pairs = {}
    for bs in range(sp.n_bs):
        for ue in range(sp.n_ue):
            for l in range(sp.n_layers):
                curr_beam_pairs[(bs, ue, l)] = sls.Beam_pairs_list()
    
    # initialisations
    curr_time_div = -1
    last_coeff_tti = -1
    coeffs = ''
    
    
    # Per TTI:
    #   Per user:
    #     - estimated SINR [dB]
    #     - realised SINR [dB]
    #     - estimated bits to send (possible to derive MCS from here)
    #     - bits sent
    #     - transport blocks that had transmission errors
    #     - interference (estimation) from other users 
    #     - scheduled resources (wideband, for each UE and TTI)
    
    # About Python lists: they can shrink and expand. The definition below 
    # either creates a python lists with zeros (if there are as many dimensions
    # in the size variable as the stated number of dimensions), or it creates 
    # a list of lists (dim 2) or a list of lists of lists (dim 3) when the size
    # is one unit smaller than the dimension. E.g. active_UEs returns the list
    # of active UEs in a given TTI (i.e. UEs with something to send, i.e. 
    # non-empty buffers)
    
    # The information that is meant to be saved will be stored in lists
    
    # UEs with something to transmit in the given TTI
    active_UEs = ut.make_py_list(2, [sp.sim_TTIs])
    
    # Estimated and Realised Interferences and Signal Powers [Downlink]
    est_dl_interference = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, 
                                              sp.n_layers])
    real_dl_interference = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, 
                                               sp.n_layers])
    
    olla = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    mcs_used = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, sp.n_layers])
    su_mimo_bitrates = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue])
    est_su_mimo_bitrate = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    ue_priority = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    all_delays = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    
    # The UEs with an active link
    scheduled_UEs = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    scheduled_layers = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    su_mimo_setting = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    realised_bits = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, sp.n_layers])
    realised_bitrate_total = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    avg_bitrate = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    blocks_with_errors = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, 
                                             sp.n_layers])
    estimated_SINR = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, sp.n_layers])
    realised_SINR = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, sp.n_layers])
    
    # TODO: Check wasted resources per scheduling tti
    n_prbs_unused = ut.make_py_list(2, [sp.sim_TTIs])

    
    # This tells which direction the beam is pointed, in which polarisation 
    beams_used = ut.make_py_list(4, [sp.sim_TTIs, sp.n_ue, sp.n_layers, 2])
    
    
    if sp.save_per_prb_variables:
        sig_pow_per_prb = ut.make_py_list(4, [sp.sim_TTIs, sp.n_ue, 
                                              sp.n_layers, sp.n_prb])
        channel_per_prb = [] # ut.make_py_list(3, [sp.n_ue, sp.sim_TTIs])
    else:
        sig_pow_per_prb = []
        channel_per_prb = []
    
    
    sp.load_gob_params(precoders_dict)
    if sp.save_power_per_CSI_beam:
        power_per_beam = ut.make_py_list(4, [sp.sim_TTIs, sp.n_ue, sp.n_layers, 
                                             sp.gob_n_beams])
    else:
        power_per_beam = []
    
    channel = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    experienced_signal_power = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    n_transport_blocks = ut.make_py_list(3, [sp.sim_TTIs, sp.n_ue, sp.n_layers])
    
    # TODO: save and plot buffer filling over duration of simulation
    # For 'classic' simulation, those two plots are very similar, and directly 
    # proportional, since all packets have the same size, not the case anymore
    # when using PCAP trace 
    bits_in_buffer = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    packets_in_buffer = ut.make_py_list(2, [sp.sim_TTIs, sp.n_ue])
    
    
    # The schedule is a list of Schedule_entries.
    # The schedule entries have UEs, BSs, beam_pairs, estimated bitrates 
    # and MCS to use in the transmission. The schedule is what is used
    # every tti to assess how many bits got across and are valid for 
    # during scheduling_tti TTIs.
    curr_schedule = {}
    
    # print(f'\nOutput folder: \n{sp.stats_dir}')
    # print(f'\nSimulating: {output_str}\n')
    # print('--------- Starting simulation ---------') 
    
    
    # raise SystemExit     
    
    # Loop for every TTI
    for tti in range(0, sp.sim_TTIs): #  1000): # 
        # Note: tti is the index of the TTI. The time value of the TTI is 
        #       given by tti_timestamp. This is done such that we don't have 
        #       to carry +-1 everywhere we go.
        
        if sp.debug:
            if tti % sp.csi_period == 0:
                print(f"(!!CSI!!) TTI: {tti}")
            else:
                print(f"TTI: {tti}")
        # TTIs    
        if tti % 8000 == 0 or tti == sp.sim_TTIs - 1:
            print(f"TTI: {tti}")
        
        # If necessary, load new set of coefficients
        if tti > last_coeff_tti:
            
            # Copy the coeffs that will be needed in the next batch
            last_coeffs = sls.copy_last_coeffs(coeffs, sp.csi_tti_delay)
                        
            # (Force) Free memory 
            del coeffs
            
            # From current TTI discover which time div we are in
            curr_time_div = sls.get_curr_time_div(tti, sp.time_div_ttis)
            
            # print('Loading batch of coeffs.')
            # curr_coeff_batch = sls.get_curr_time_div(tti, sp.TTIs_per_batch)
            # print(f'Loading batch of coeffs #{curr_coeff_batch} '
            #       f'from time div {curr_time_div}.')
            
            # Save the TTI from which the new set of coefficients may be used
            first_coeff_tti = tti
            
            # Coeffs is a dictionary with 4D arrays: 
            #    - (bs, ue, l) indexes the dictionary
            #    - [ue_ae][bs_ae][prb][tti] indexes the 4D array
            coeffs, last_coeff_tti = \
                sls.load_coeffs(tti,
                                curr_time_div, 
                                sp.time_divisions,
                                sp.time_div_ttis,
                                sp.time_compression_ratio,
                                sp.sim_freq_idx,
                                sp.n_freq,
                                sp.n_bs_gen,
                                sp.n_ue_gen,
                                sp.specify_bss,
                                sp.specify_ues,
                                sp.coeff_file_prefix,  
                                sp.coeff_file_suffix,  
                                sp.n_ue_coeffs,
                                sp.n_bs_coeffs,
                                sp.ae_ue,  
                                sp.ae_bs,  
                                sp.n_prb,
                                sp.TTIs_per_batch)

            # print('Batch loaded.')
            
            # Update channel trace variables such that we can easily relate
            # channel quality with received signal and etc..
            
            sls.update_channel_vars(tti, sp.TTIs_per_batch, sp.n_ue, coeffs,
                                    channel, channel_per_prb, 
                                    sp.save_per_prb_variables)
        
        # Copy information from previous ttis and update some parameters
        tti_timestamp, tti_relative = \
            sls.tti_info_copy_and_update(tti, sp.TTI_duration, first_coeff_tti, 
                                         sp.n_phy, sp.n_layers, 
                                         est_dl_interference, avg_bitrate, 
                                         olla, sp.use_olla, power_per_beam,
                                         sp.save_power_per_CSI_beam)
        
    # Phase 0: slot/TTI identification and Queue update
        # 0- a) Identify the slot type
        slot_type = sls.id_tti(tti, sp.n_slots_in_frame, sp.UL_DL_split)
        
        # 0- b) Filter slot accordingly
        if sp.debug:
            print('Slot: ' + slot_type)
        
        # It is considered that all symbols in the slot are used in the same
        # way, all DL or all UL
        if slot_type == 'UL':
            # if UL TTI, use cam_buffers
            ue_idxs = np.arange(sp.n_phy, sp.n_ue)
            continue  # at the moment, only DL is implemented
            # TODO: ADD functionality for UL (and perhaps only look at UL)
            
        elif slot_type == 'DL':
            # if DL TTI, use users' buffers
            ue_idxs = np.arange(sp.n_phy)
        elif slot_type == 'F':
            # Not implemented further, but the idea is based on both buffers,
            # make a decision of whether DL or UL is best. Transition slots and
            # mini-slots can also be defined here.
            ue_idxs = np.arange(sp.n_ue)
            raise Exception('Not Implemented slot type.')
        else:
            raise Exception('Invalid slot type.')
        
        # TODO: save buffer status before(after?) every queue update 
        if not True:
            for ue in range(sp.n_phy):
                for i in range(len(buffers[ue].bits_left)): 
                    if buffers[ue].bits_left[i] != 0: 
                        # Nr of packets with something left to send
                        packets_in_buffer[tti][ue] +=1 
                # Total bits left to send in buffer (in kByte)
                bits_in_buffer[tti][ue] = sum(buffers[ue].bits_left[:]) / 8000         
        
        # 0- c) Update Queues: Add packets, update delays, drop late packets
        sls.update_queues(ue_idxs, buffers, tti_timestamp, active_UEs, tti, 
                          sp.use_pcap, sp.TTI_dur_in_secs) 
        
        # print("Update queues finished, time:", toc-tic)
        # raise SystemExit
        
        # active UEs are the UEs with non-empty buffers. We are putting those
        # to True, always, because we don't have a robust interference 
        # estimation. This is why the I frames need to be synchronized!
        if sp.always_schedule_every_ue:
            active_UEs[tti] = ue_idxs
    # Phase 1: CSI update
        # 1-a) Update the Latest CSI tti (based on CSI periodicity)
        #  Check if the precoder and other CSI measurements should be updated 
        if tti % sp.csi_period == 0:
            # If an UE was updated before this tti, it needs to be updated again
            last_csi_tti = tti
            
            # The UE will be updated with information from this tti
            # Relative to current time division (Modulo 1000)
            tti_with_csi = sls.get_delayed_relative_tti_csi(tti, tti_relative, 
                                               sp.csi_tti_delay)
            
            # CSI UPDATE: 
                # -interference is updated every csi_period TTIs
                # -precoders are only updated when a user needs, because the 
                # process is more resource consuming (done inside the function)
    
        # ####################### CSI UPDATE ############################    
        # 1- b) Update interference measurements for the DL
            sls.interference_measurements_update(ue_idxs, sp.n_layers, 
                                                 tti, last_csi_tti, 
                                                 sp.csi_tti_delay, 
                                                 est_dl_interference,
                                                 real_dl_interference)
            
        # 1- c) Update precoders
        sls.update_all_precoders(tti, tti_with_csi, active_UEs, sp.n_bs, 
                                 curr_beam_pairs, last_csi_tti, 
                                 precoders_dict, coeffs, last_coeffs, 
                                 sp.n_layers, sp.n_csi_beams, power_per_beam,
                                 sp.save_power_per_CSI_beam, sp.vectorize_GoB)
        
        # From here onwards, we know what precoders are best for each UE, 
        # per layer. This has been verified with LoS simulations, print below
        if sp.debug:
            sls.print_curr_beam_pairs(curr_beam_pairs,
                                      sp.n_bs, sp.n_ue, sp.n_layers, 
                                      'single-layer')
            print('done updating precoders')
            
        # ######################## END OF CSI UPDATE #########################
    
    # Phase 2: Scheduling Update
    
        # If the TTI is not a scheduling information update tti, just copy the 
        # information of the previous tti
        if tti % sp.scheduling_period != 0:
            # Copy UEs that were scheduled previously
            if tti != 0:
                scheduled_UEs[tti] = scheduled_UEs[tti-1]
            
            # Copy the how many layers are decided to be the best for that UE
            for ue in range(sp.n_ue):
                su_mimo_setting[tti][ue] = su_mimo_setting[tti-1][ue]
            
            # And do nothing to the schedules
        else: 
            
            
            # Opposed to what is done with CSI, all scheduling is updated
            # in the scheduling TTI. And it is used until there is another 
            # scheduling TTI. 
            
            # The UE will be updated with information from this tti
            tti_for_scheduling = sls.get_delayed_tti_scheduling(tti,  
                                                     sp.scheduling_tti_delay)
    
                           
            # ####################### SCHEDULING UPDATE ######################
            # 2- a) Which UEs to consider for scheduling?
            # The ones that have something in their buffer: active_UEs
            schedulable_UEs = active_UEs[tti_for_scheduling]
            
            # Given that some UEs are only for UL and others are only for DL
            schedulable_UEs_dl = [ue for ue in schedulable_UEs 
                                  if ue < sp.n_phy]
     
        # Continuation of scheduling step
        if tti % sp.scheduling_period == 0 and len(schedulable_UEs_dl) == 0:
            # Nothing to schedule
            if sp.debug:
                print('No UEs to schedule!')
            curr_schedule['DL'] = []
        elif tti % sp.scheduling_period == 0 and len(schedulable_UEs_dl) != 0:
            
            # 2- b) Choose serving BS per UE
            # The one that has the best precoder to the user.
            # So get the best precoder for each BS, and then pick the best BS.
    
            # with a single bs it is the same for all UEs
            serving_BS_dl = [0 if ue in schedulable_UEs_dl else -1
                             for ue in range(sp.n_phy)]
            
            #-------------------------------            
            
            # 3- Select the best SU-MIMO setting: 1 layer or 2 layers
            # For DL:
                
            sls.su_mimo_choice(tti, tti_for_scheduling, sp.bs_max_pow, 
                               schedulable_UEs_dl, serving_BS_dl, 
                               sp.n_layers, sp.n_prb, 
                               curr_beam_pairs, est_dl_interference, 
                               sp.wideband_noise_power_dl, sp.TTI_duration, 
                               sp.freq_compression_ratio, sp.use_olla, olla, 
                               sp.debug_su_mimo_choice, su_mimo_bitrates, 
                               est_su_mimo_bitrate, su_mimo_setting, 
                               sp.DL_radio_efficiency, sp.bandwidth_multiplier)
            
            if sp.debug:
                print(f"SU-MIMO bitrates: {su_mimo_bitrates[tti][1:sp.n_phy]}")
                
            # -------------------------------

            # 4- Compute UE priorities (Using Scheduler)
            
            curr_priorities = \
                sls.compute_priorities(tti, ue_priority, all_delays, buffers, 
                                       schedulable_UEs_dl, sp.scheduler, 
                                       avg_bitrate, est_su_mimo_bitrate,
                                       ut.get_seconds(sp.delay_threshold), 
                                       sp.scheduler_param_delta, 
                                       sp.scheduler_param_c,
                                       sp.frametype_weight)
            
            if sp.debug:
                print(curr_priorities)
                print(avg_bitrate[tti])
                if tti > 0:
                    print(realised_bitrate_total[tti-1])
                print('Priorities are sorted!')
                
            # -------------------------------
            
            # 5- Select MU-MIMO setting, based on UE priorities
            
            sls.mu_mimo_choice(tti, curr_priorities, curr_schedule, 
                               serving_BS_dl, su_mimo_setting, curr_beam_pairs, 
                               sp.min_beam_distance, scheduled_UEs, 
                               scheduled_layers, sp.debug)
            
            # -------------------------------
            
            # 6- Power Control
            
            sls.power_control(tti, sp.bs_max_pow, scheduled_UEs, 
                              scheduled_layers, curr_schedule)
            
            # -------------------------------
            
            # 7- Update SINRs, expected bitrates and MCS to use
            
            sls.final_mcs_update(tti, curr_schedule, est_dl_interference,
                                 sp.wideband_noise_power_dl, sp.n_prb, 
                                 sp.TTI_dur_in_secs, sp.freq_compression_ratio, 
                                 estimated_SINR, sp.use_olla, olla,
                                 sp.tbs_divisor, sp.DL_radio_efficiency, 
                                 sp.bandwidth_multiplier, scheduled_UEs, 
                                 scheduled_layers)

            
        # ################## END OF SCHEDULING UPDATE ####################
        # print(tti)
        # print('here')
        # Phase 3: TTI Simulation
        sls.tti_simulation(curr_schedule, slot_type, sp.n_prb, sp.debug, 
                           coeffs, tti_relative, 
                           sp.intercell_interference_power_per_prb, 
                           sp.noise_power_per_prb_dl, tti, 
                           real_dl_interference, info_bits_table, buffers, 
                           n_transport_blocks, realised_bits, olla, 
                           sp.use_olla, sp.bler_target, sp.olla_stepsize, 
                           blocks_with_errors, realised_SINR, 
                           sp.TTI_dur_in_secs, sp.time_to_send,	 
                           realised_bitrate_total, beams_used, sig_pow_per_prb, 
                           mcs_used, sp.save_per_prb_variables, 
                           experienced_signal_power)
        
        # TODO: Calculate efficiency of scheduling 
        # if curr_schedule['DL'] != []: 
        #     est_bitrate_tti = curr_schedule['DL'][0].est_bitrate / 1e6
        #     # unused_bitrate = est_bitrate_tti - sum(realised_bitrate_total[tti])
        #     bitrate_per_prb = est_bitrate_tti / 10 
            
        #     non_sched_UEs = np.where(np.array(scheduled_UEs[0]) == 0)
        #     other_buffers_empty = True
        #     for l in range(len(non_sched_UEs[0])): 
        #         other_buffers_empty *= buffers[non_sched_UEs[0][l]].is_empty
            
        #     if other_buffers_empty == False:
        #         n_prbs_unused[tti] = 10 - int(np.ceil(sum(realised_bitrate_total
        #                                       [tti] / bitrate_per_prb)))
        #     else: 
        #         n_prbs_unused[tti] = 0
        # else:
        #     n_prbs_unused[tti] = 0
              
        if sp.debug:
            print(f'----------Done measuring tti {tti} ---------------------')
            for ue in range(sp.n_ue):
                # if schedulable_UEs[tti][ue] == 0:
                #     continue
                print(f'Realised bitrate for ue {ue} in tti {tti} was '
                      f'{realised_bitrate_total[tti][ue]}.'
                      f'Estimated Interference: '
                      f'[{est_dl_interference[tti][ue][0]:.2e}'
                      f', {est_dl_interference[tti][ue][1]:.2e}]; '
                      f'Real Interference: '
                      f'[{real_dl_interference[tti][ue][0]:.2e}, '
                      f'{real_dl_interference[tti][ue][1]:.2e}].')
                  
        # 11- Update end of tti variables        
        sls.update_avg_bitrates(tti, sp.n_ue, realised_bitrate_total, 
                                avg_bitrate, schedulable_UEs_dl)
                      
        # Save buffer status after(?) tti simulation 
        # TODO: for pcap as well 
        if not sp.use_pcap:
            for ue in range(sp.n_phy):
                for i in range(len(buffers[ue].bits_left)): 
                    if buffers[ue].bits_left[i] != 0: 
                        # Nr of packets with something left to send
                        packets_in_buffer[tti][ue] +=1 
                # Total bits left to send in buffer (in kByte)
                bits_in_buffer[tti][ue] = sum(buffers[ue].bits_left[:]) / 8000 
        # ####################################################################
        
    # print('End of tti loop.')    
    # raise SystemExit()
    # One final queue update, in order to account for all the packets that were
    # sent last tti
    for ue in range(sp.n_ue):
        if scheduled_UEs[tti][ue] == 1:
            t = ut.timestamp(s=(tti + 1) * sp.TTI_dur_in_secs)
            if sp.use_pcap: 
                buffers[ue].update_head_of_queue_delay(tti, tti_duration)
                        
                # print(buffers[ue].pdr_info[30:60])                                
            else: 
                buffers[ue].update_head_of_queue_delay(t)    
                
        if sp.use_pcap:
            pcap_output_str = f'{final_trace.split("_")[0]}_' + \
                              f'{final_trace.split("_")[1]}_' + \
                              f'{final_trace.split("_")[2]} - ' + \
                              f'{sim_dur}s_UE{ue}.csv'
            pcap_output_folder = pcap_parameters.split("\\")[2]
            pcap_output_path = sp.stats_dir + "\\PCAP\\" + output_str + \
                               f"\\{seed_str}\\" + \
                               f'\\{pcap_output_folder}\\{trace_parameters}'
                
            buffers[ue].create_pdr_csv(pcap_output_str, pcap_output_path)
            
    # print(f'------ Done simulating for {output_str} ------')
    print(f'Time enlapsed: {round(time.time() - t_0)} secs.')
    
    
    
    
    # Write stats to storage
    write_stats = not 1
    if write_stats:
        
        time_sim_end = ut.get_time()
        
        # Make folder for the stats of this simulation
        
        if include_timestamp:
            sp.stats_path = sp.stats_dir + output_str + f"_{time_sim_end}" + "\\"
        else:
            sp.stats_path = sp.stats_dir + output_str + f"\\{seed_str}\\"
                
        # Write which stats file was produced last, for practical reasons, 
        # e.g.in case we want to analyse plots of it right away.
        with open('last_stats_folder.txt', 'w') as fh:
            fh.write(sp.stats_path)
        
        if not ut.isdir(sp.stats_path):
            ut.makedirs(sp.stats_path)
        else:
            # Overriding! Use a the timestamp to prevent this.
            ut.del_dir(sp.stats_path)
            ut.makedirs(sp.stats_path)
        
        # Write the exact path of the folder where the traces came from
        with open(sp.stats_path + 'parent_generation_folder.txt', 'w') as fh:
            fh.write(sim_folder)
            
        
        # To save some time naming all variables...
        globals_dict = globals()
        
        # Some variables interfere with the ones we actually want to save
        # when they have exactly the same values.
        try:
            del user_buffers
        except:
            print('weird... no user buffers... something is wrong...')
            # A bug we are still trying to catch... almost never happens...
        
        # TODO: use np.save instead. Convert them right here and test.
        # Then simply create them as numpy arrays from the get go.
        
        # Pickle all results 
        ut.save_var_pickle(sp, sp.stats_path, globals_dict)
        ut.save_var_pickle(buffers, sp.stats_path, globals_dict)        
        ut.save_var_pickle(estimated_SINR, sp.stats_path, globals_dict)
        ut.save_var_pickle(realised_SINR, sp.stats_path, globals_dict)
        ut.save_var_pickle(realised_bitrate_total, sp.stats_path, globals_dict)
        # ut.save_var_pickle(n_transport_blocks, sp.stats_path, globals_dict)    
        # ut.save_var_pickle(blocks_with_errors, sp.stats_path, globals_dict)        
        # ut.save_var_pickle(beams_used, sp.stats_path, globals_dict)
        # ut.save_var_pickle(olla, sp.stats_path, globals_dict)
        ut.save_var_pickle(mcs_used, sp.stats_path, globals_dict)
        # ut.save_var_pickle(real_dl_interference, sp.stats_path, globals_dict)
        # ut.save_var_pickle(est_dl_interference, sp.stats_path, globals_dict)
        ut.save_var_pickle(scheduled_UEs, sp.stats_path, globals_dict)
        # ut.save_var_pickle(su_mimo_setting, sp.stats_path, globals_dict)
        ut.save_var_pickle(channel, sp.stats_path, globals_dict)
        # ut.save_var_pickle(experienced_signal_power, sp.stats_path, globals_dict)
        
        # Variables that take the most memory: they are always saved,
        # but when sp.save_per_prb_variables is False, they are None
        # ut.save_var_pickle(sig_pow_per_prb, sp.stats_path, globals_dict)        
        # ut.save_var_pickle(channel_per_prb, sp.stats_path, globals_dict)
        
        # If we are debugging GoBs and we need the power of each CSI beam
        # (is none when sp.save_power_per_CSI_beam is False)
        # ut.save_var_pickle(power_per_beam, sp.stats_path, globals_dict)
        # ut.save_var_pickle(bits_in_buffer, sp.stats_path, globals_dict)
        # ut.save_var_pickle(packets_in_buffer, sp.stats_path, globals_dict)
        # ut.save_var_pickle(active_UEs, sp.stats_path, globals_dict)
        ut.save_var_pickle(ue_priority, sp.stats_path, globals_dict)

        ut.save_var_pickle(n_prbs_unused, sp.stats_path, globals_dict)

        # ut.save_var_pickle(su_mimo_bitrates, sp.stats_path, globals_dict)


tocc = time.perf_counter()
print('End of sxr_sim.')
print(f'Total Time Elapsed: {int((tocc-ticc)/60)} minutes.')