# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:33:50 2020
@author: janeiroja
Parameters concerning System-Level simulation (Application traffic, and more..)
"""

import numpy as np

import scipy.io

import pathlib

import utils as ut


class Simulation_parameters:
    def __init__(self, folder_to_load, freq_idx, csi_periodicity, 
                 application_bitrate, user_list, bw, lat_budget, 
                 n_layers, rot_factor):

        # 1- Init General Variables (General parameters and Vari)
        self.set_simulation_param(freq_idx, csi_periodicity, user_list, 
                                  n_layers, rot_factor)
        
        # 2- Init IO parameters (which folders to write to, and so on)
        self.set_io_param(folder_to_load)
    
        # 3- Init Application Traffic Model parameters
        self.set_application_param(application_bitrate, lat_budget)
        
        # 4- Compute (and load from sim folder) all remaining variables
        self.compute_vars_simulation(bw)

        # 5- Compute the variables required to generate traffic
        self.compute_application_traffic_vars()

    def set_simulation_param(self, freq_idx, csi_periodicity, user_list, 
                             n_layers, rot_factor):
        # ################# Mode 1: General Parameters  #######################
        # Note: everything will be zero-indexed from now on, because now we
        #       are in Python land
        
        
        # All random numbers come from here
        self.numpy_random_seed = 13492
        
        # To print every step of the simulation
        self.verbose = 0
        
        self.debug = 0
        
        # Debug variables
        self.debug_su_mimo_choice = 0
        
        # TTIs to simulate
        self.sim_TTIs = 4000 * 2
        
        # TTIs per batch
        self.TTIs_per_batch = 1000 # min 200
        
        # Will influence TTI duration
        self.simulation_numerology = 2
        
        # Index of the frequency: only the coefficients correspondent to this
        # frequency will be loaded. This index should match the ordering of
        # the channel generation
        self.sim_freq_idx = freq_idx
        
        # Define whether to simulate different values from those generated 
        # in the traces
        self.use_subset_for_simulation = True
        self.specify_bss = [0]
        self.specify_ues = user_list
        
        self.sim_n_ue = len(self.specify_ues)
        self.sim_n_phy = self.sim_n_ue
        self.sim_n_bs = len(self.specify_bss)
        
        # SU: Schedule one user at the time, as many layers as defined in n_layers
        # MU: Scheduled all users (remember the interference problem) at each
        #     tti, if their layers are compatible...
        self.scheduling_method = 'MU'
        
        self.bf_method = 'gob' # 'reciprocity'
        IMPLEMENTED_LAYERS_IMPLICIT_BF = 0
        IMPLEMENTED_LAYERS_EXPLICIT_BF = 2
        
        # Maximum 2 layers per UE
        self.n_layers = n_layers
        
        if self.bf_method == 'gob':
            if self.n_layers > IMPLEMENTED_LAYERS_EXPLICIT_BF:
                raise Exception(f'GoB supports only '
                                f'{IMPLEMENTED_LAYERS_EXPLICIT_BF} layers')
        elif self.bf_method == 'reciprocity':
           if self.n_layers > IMPLEMENTED_LAYERS_IMPLICIT_BF:
                raise Exception(f'GoB supports only '
                                f'{IMPLEMENTED_LAYERS_IMPLICIT_BF} layers')
        else:
            raise Exception('BF method not recognized.')
        
        
        # Maximum total layers, summed over all UEs
        self.max_mu_mimo_layers = 99 # Doesn't do anything yet anyway
        
        
        # The minimum distance allowed between beam indices to co-schedule
        # users with those beams
        # If = 0, co-schedule always, even in the same beam.
        # If = 1, co-schedule if the beams are not be equal.
        # If = 2, don't co-schedule adjacent beams (first diagonal is dist 1.4)
        self.min_beam_distance = 1 # 1 # 2.9 # 4.1
        
        # How many beams should be reported to the BS?
        # If 1, then only the best beam is reported;
        # If 2, the second best is also reported; (etc...)
        # In case we want to test some intelligent way of handling multiple
        # reports
        self.n_csi_beams = 1

        # Rotation Factor (put to None for not applying, namely on a 
        # beam-steering GoB (i.e. the first GoB, non-3GPP))
        self.rot_factor = rot_factor

        # How frequently to update CSI? 
        # CSI: Precoders and Interference measurements
        self.csi_period = csi_periodicity
        
        
        # When there is an update on Channel State Information, from how many
        # TTIs ago is that information? Because the measurement can't be 
        # put to use the same TTI (it requires 1) sending the Reference Signal, 
        # in this case the CSI-RS; 2) let the UE process it; 3) UE sending the
        # measurement back; 4) BS process it.)
        self.csi_tti_delay = 5
        
        
        # The equivalent variables for scheduling: it can happen on a different
        # time scale
        self.scheduling_period = 1
        # For a tti_delay of 1, the last tti is used. 
        # Also know and acknowledgment delay (ACK DELAY)!
        self.scheduling_tti_delay = 0
        
        
        # Maximum TX power per BS 
        self.bs_max_pow = 0.1  # [Watt]
        # There can be 8 independent streams in the air (2 per UE) : 8x8 MIMO
        self.max_layers = 8
        
        # 
        self.bs_max_pow_per_layer = self.bs_max_pow / 10
        
        # A power floor for intercell interferece outside of the antennas
        # used in the room (linear power)
        self.intercell_interference_power_per_prb = 0
        
        
        # Noise related variables
        self.boltz = 1.38e-23
        self.temperature = 290
        
        # Noise figures (increase thermal noise power)
        self.bs_nf = 3  # Noise figure at the BS (for UL) [dB]        
        self.ue_nf = 8  # Noise figure at the UE (for DL) [dB]
        
        # UL/DL split specifics
        self.n_slots_in_frame = 5
        self.UL_DL_split = 0.8
        
        
        # Outter Loop Link Adaptation (OLLA)
        self.use_olla = False
        self.bler_target = 0.1
        self.olla_stepsize = 0.1
        
        
        # Scheduler - ['PF', 'M-LWDF', 'EXP/PF']
        self.scheduler = 'M-LWDF'
        
        # Scheduler parameters
        self.scheduler_param_c = 10
        self.scheduler_param_delta = 0.05
        # These 2 are not used when all traffic is real-time
        # Also, NRT traffic is not implemented.
        # self.epsilon = 1  
        # self.kappa_v2 = 100 (this has nothing to do with beam distances)
        
        # Radio signalling overheads (values according to 3GPP- TS 38.306)
        # For frequency range 1 and FR2, respectively.
        self.UL_radio_overhead = [0.08, 0.10]
        self.DL_radio_overhead = [0.14, 0.18]
        
        
        # For convenience, levels 1, 2, ..., 11. ((PRB number is = level x 25))
        # self.scheduler_granularity_level = 1
        # self.n_schedulable_subbands = 1
        
        # The above definition agrees wit 5G definitions:
        # self.MIN_PRB = 24
        # self.MAX_PRB = 275
        
        # Transport Blocks per scheduling entry
        # The problem with putting all the eggs in the same basket is that 
        # some bad luck can take all those eggs away.
        # Increase the number to distribute the eggs across more baskets.
        # NOTE: this shouldn't be necessary when the 5G compliant 
        # implementation is in place
        # NOTE2: This number will divide the olla step size!
        self.tbs_divisor = 5
        
        # IMPORTANT: should we always co-schedule all UEs?! This makes the 
        #            simulation much simpler.
        # Note: although this is set to True, it doesn't break the minimum
        #       beam distance parameter, which is applied after. 
        #       This parameter actually skips the check of whether users have
        #       packets in the buffer, that's it.
        self.always_schedule_every_ue = True
        
        ##########################################
        # - self.csi_tti_delay = 0
        # - self.always_schedule_every_ue = False
        ##########################################
        
        # TODO: This variable is here because the simulator is not read for
        # not having it. What happens is the following:
            # if we put this to false, then only the UEs with things to send
            # are scheduled. The problem with this is that now we cannot
            # predict how many UEs are going to be co-scheduled, and hence
            # we can't predict the interference, therefore we use the MCS of 
            # the past, which leads to a lot of errors!
            # There are workarounds, but take time. So: postponed for later.
        
        # In case we need per PRB information on the channel and signal power:
        self.save_per_prb_sig_pow = False
        self.save_per_prb_channel = False
        
        # In case we need to know how much power is received given a certain
        # choice of GoB
        self.save_power_per_CSI_beam = False
        
        # Instead of checking which is the best beam through a loop, 
        # create matrices and multiply them. For some sizes of the grid and
        # number of antennas, it may yield better computational performance.
        self.vectorize_GoB = False
        
            
    def set_io_param(self, folder_to_load):
        # #################### IO Files & Names ##############################
        
        self.curr_path = str(pathlib.Path().absolute())
        self.matlab_folder = self.curr_path + '\\Matlab\\'
        
        # Folder where the (partial in mode 3 or full in mode 0) channel 
        # calculations will be stored 

        # Directory where all channel parts will be placed
        self.channel_folder_name = 'Channel_parts\\'
        
        # Each instance should output in the format: {prefix}_{instance ID}
        # {prefix}_part_{partID}_instance_{instanceID}_num_{numerology}
        self.output_preffix = 'ch'
        
        # Define a table of information bits needed for the SLS simulator
        self.info_bits_table_path = (self.curr_path + 
                                     '\\miesm_table.csv')
        
        # Stats folder
        self.stats_dir = self.curr_path + '\\Stats\\'
        
        # Plots folder
        self.plots_dir = self.curr_path + '\\Plots\\'
        
        # Coefficients related:
        self.folder_to_load = folder_to_load
        
        self.folder_to_load = self.folder_to_load + '\\'

        self.coeff_folder = self.folder_to_load + "Channel_parts\\"
        
        # with open(curr_path + 'last_sim_folder.txt') as fp:
        #     self.coeff_folder = fp.read() + self.channel_folder_name
        
        self.coeff_file_prefix = self.coeff_folder + 'fr_part_'
        self.coeff_file_suffix = '_num' + str(self.simulation_numerology)
        
        self.reload_vars_file = self.folder_to_load + 'vars'
        
        
        # Precoders loading files
        self.precoders_folder = self.matlab_folder + 'Precoders\\'
        
        # A precoder for each antenna, for each frequency [freq][bs_idx]
        self.precoders_files = \
            [["precoders_4_4_4_4_pol_3_RI_1_ph_1"], 
             ["1-omni-element"]]
        
        # the case above has a single precoder for each frequency
        # The selected precoder path, with the simulated frequency, is
        # computed below:
        self.precoders_paths = []
        for precoder_file in self.precoders_files[self.sim_freq_idx]:
            self.precoders_paths.append(self.precoders_folder + 
                                        precoder_file)
            
        
    def set_application_param(self, application_bitrate, lat_budget):
        # ##################### Application Parameters ########################
        
        # Space the I frames across the GoP for the existant UEs
        self.uniformly_space_UE_I_frames = False
        # Note: until the TODO in the end of simulation parameters is solved, 
        #       this should be set to False. Otherwise we fall into the 
        #       interference unpredictability problem again.
        
        # Group of Pictures ( Made of: 1 I frame and (GoP-1) P frames )
        self.GoP = 6                     # [6, 9 or 12]
        # Frames per second (we assume Headset and Cameras have the same rate)
        self.FPS = 30
        # P frame size to (divided by) I frame size ratio [on average]
        self.IP_ratio = 0.20
        # P frame std for the gaussian around the IP_ratio [%]
        self.P_std = 0
        
        # Frame Computation Strategy
        # Strategy for computing I frame size 
        # ['AVG_BITRATE_UL', 'AVG_BITRATE_DL', 'BASE_PARAMETERS']
        self.frame_calc_strat = 'AVG_BITRATE_DL'
        
        
        # OPTION 1: AVG_BITRATES_UL
        # Average Bitrate of the Uplink Stream (tip: multiples of GoP size)
        # NOTE: this is may be different than each camera bitrate, if the 
        # camera streams are aggregated before uploaded. (self.ul_aggregation)
        self.avg_bitrate_ul = 10  # [Mpps] 
        
        # OPTION 2: AVG_BITRATES_DL
        # Instead of summing all uplinks into the DL for each user, an
        # average bitrate in the dl can be inserted
        self.avg_bitrate_dl = application_bitrate  # [Mbps]
        
        
        # NOTE: the UL and DL are connected with the number of users:
        #       the DL has always (n_vir + n_phy) times more traffic.
        #       Therefore, in Options 1 and 2, it is possible to compute the
        #       counterpart.
        
        # OPTION 2: BASE PARAMETERS (incomplete but usable)
        # It needs a mapping between headset parameters and DL size
        # At the moment it considers only camera parameters and aggregates the 
        # uplinks into the DL
        
        # If I is computed from app parameters, use the following to derive it:
        # Headset Resolution (we assume the same for all headsets)
        # self.headset_resolution = 1920 * 1080
        
        # Camera resolution - typically 480p or 720p
        # Assumed the same for all cameras
        # 480p - 720 x 480; 720p - 1280 x 720
        self.camera_resolution = 720 * 480      
        # Pixel Format [channels per pixel] - RGB has 3, YUV has 1.5
        self.pixel_format = 3
        # Pixel Depth [bits per channel] - [8, 10, 12, 16] 8 is usual
        self.pixel_depth = 8
        # Bits of resolution for the depth channel   
        self.depth_resolution = 16
        # Tiles Prediction Enable
        self.tile_prediction = False
        # Tiles Prediction Ratio
        self.tiles_prediction_ratio = 0.5
        # Video compression algorithm for I frame size calculation
        self.video_compression_ratio = 1 / 120
        
        
        # Room users are doing VR or AR - matters for only for DL size
        # in VR, the DL for each UE is (PHY + VIR) * UL_AVG_BITRATE
        # in AR, the DL for each UE is (VIR) * UL_AVG_BITRATE
        self.VR_or_AR = 'VR'
        
        # Default Packet size
        self.packet_size = 1500  # bytes
        
        # Packet latency Budget [ms]
        self.packet_lat_budget = lat_budget
        
        # Number of TTIs of slack: discards the packets if the latency budget
        # is less than this many ttis from the latency budget
        self.time_to_send = 2
        
            
    def check_vars_simulation(self):
        """
        Check if variables for coefficient analysis and simulation are correct.
        """
        
        ut.parse_input(self.frame_calc_strat, 
                       ['AVG_BITRATE_UL','AVG_BITRATE_DL', 'BASE_PARAMETERS'])
        ut.parse_input(self.VR_or_AR, ['AR', 'VR'])
        
        
        # if not (1 <= self.scheduler_granularity_level <= 11):
        #     print("Scheduling block level must be between 1 and 11 suc that "
        #           "the limits of MIN 24 PRBs and MAX 275 PRBs defined by "
        #           "3GPP can be respected.")
        #     ut.stop_execution()
        
        ut.parse_input(self.scheduler, ['PF', 'M-LWDF', 'EXP/PF'])

        ut.parse_input(self.scheduling_method, ['SU', 'MU'])
        
                
    def compute_vars_simulation(self, bw):
        
        # Check if variables have values in the correct ranges
        self.check_vars_simulation()
        
        
        # Fix NumPy Seed - from now on, nothing is random.
        np.random.seed(self.numpy_random_seed)
        
        
        # Load Variables from previous simulation
        
        print(f"Loading variables from {self.reload_vars_file}... "
              f"It may generate a warning due to unsupported formats, "
              f"namely Quadriga formats. Ignore it :)")
        vars_dict = scipy.io.loadmat(self.reload_vars_file)
        
        # Extraction of variables
        
        self.matlab_seed = vars_dict['SEED'][0]
        
        self.sim_duration_secs = float(vars_dict['simulation_duration'][0][0])
        self.numerology = vars_dict['numerology'][0]
        self.bandwidth = vars_dict['bandwidth']
        self.freq = vars_dict['f_values'][0][self.sim_freq_idx]
        
        # check if chosen frequency is appropriate to the simulation
        if len(self.bandwidth) <= self.sim_freq_idx:
            print(f"Numerology {self.simulation_numerology} it is not "
                  f"at index {self.sim_freq_idx} of the generated "
                  f"numerologies({self.numerology})")
            ut.stop_execution()
        
        
        # Remaining main parameters
        self.bandwidth = vars_dict['bandwidth'][self.sim_freq_idx][0]
        self.n_prb = vars_dict['n_prb'][self.sim_freq_idx]
        self.n_phy = vars_dict['n_room'][0][0]
        self.n_vir = vars_dict['n_room'][0][1]
        self.n_cam = vars_dict['n_camera_streams'][0][0] * self.n_phy
        self.n_bs = vars_dict['n_tx'][0][0]
        self.n_freq = len(vars_dict['f_values'][0])
        self.time_compression_ratio = vars_dict['time_compression_ratio'][0][0]
        self.freq_compression_ratio = vars_dict['freq_compression_ratio'][0][0]
        
        # Positions
        self.phy_pos = vars_dict['phy_usr_pos']
        self.vir_pos = vars_dict['vir_usr_pos']
        if self.n_cam != 0:
            self.cam_pos = vars_dict['cam_pos']
        self.bs_pos = vars_dict['tx_pos'][0]
        
        # Regarding batches and parallelisation
        self.instances_per_time_div = \
            vars_dict['n_instances_per_time_division'][0][0]
        self.time_divisions = vars_dict['n_time_divisions'][0][0]
        self.n_total_instances = vars_dict['n_total_builders'][0][0]
        
        # Number of Coefficients
        self.n_ue_coeffs = \
            vars_dict['rx_ant_numel'][self.sim_freq_idx].astype(int)
        self.n_bs_coeffs = \
            vars_dict['tx_ant_numel'][self.sim_freq_idx].astype(int)

        # One or two polarisations?
        diff_orthogonal_polarisation = \
            vars_dict['diff_orthogonal_polarisation'][0]
        
        try:
            self.pos_backup = vars_dict['pos_backup']
            self.ori_backup = vars_dict['ori_backup']
            self.initial_pos_backup = vars_dict['initial_pos_backup']
        except KeyError:
            print('WARNING: POSITION AND ORIENTATION WERE NOT BACKED UP FROM '
                  'THE LAST GENERATIONS! Simulation will continue without '
                  'that information.')
        ######### COMPUTATIONS BASED ON THE VALUES LOADED ABOVE #############
        
        if diff_orthogonal_polarisation == 0:
            self.n_polarisations = 1
            # The rest of the simulator is not designed for one polarisation o
            # only.
            # Therefore we will break it here. In a meeting, we need the 2 
            # polarisations to cope with the rapid changes in orientation.
            pass
            # TODO: check whether this is a hard limitation or not.
            # raise Exception('Not enough coefficients! Only supports '
            #                 'single antenna elements, not cross-polarised.')
        else:
            self.n_polarisations = 2
            # This number of polarisations is assumed throughout the simulator. 
            # It leads to 4 possible combinations of polarisations.
            # What this means: we may not even use the variable, but it is 
            # assumed its value is 2. E.g. computing the combinations for 
            # different layers between RX and TX.
            
            
        
        # Lists with the antenna elements indexed by the ue/bs index.
        self.ae_ue = (self.n_ue_coeffs / self.n_polarisations).astype(int)
        self.ae_bs = (self.n_bs_coeffs / self.n_polarisations).astype(int)
        
        # Computation of variables from the extracted ones
        # Simulation duration [s]
        self.sim_duration = ut.timestamp(us = self.sim_duration_secs * 1e6)
        
        # TTI [ms]
        self.TTI_duration = (
            ut.timestamp(0, float(1 / (2**self.simulation_numerology))))
        self.TTI_dur_in_secs = ut.get_seconds(self.TTI_duration)
        
        # This is how frequently we get samples in time domain
        self.update_rate = self.TTI_dur_in_secs * self.time_compression_ratio
        
        # Number of TTIs of the generation (system level simulator)
        self.gen_TTIs = int(self.sim_duration / self.TTI_duration)

        self.n_ue = self.n_phy + self.n_cam
                
        if self.n_cam != 0:
            self.ue_pos = np.concatenate((self.phy_pos, self.cam_pos), axis=1)
        else:
            self.ue_pos = self.phy_pos
            
        self.n_users = self.n_phy + self.n_vir
        
        # For future simplicity:
        self.user_idxs = np.arange(0, self.n_phy)
        self.cam_idxs = np.arange(self.n_phy, self.n_ue)

        
        print(f'Preparing to simulate {self.n_ue} UEs, {self.n_bs} BSs, '
              f' at frequency {self.freq/1e9} GHz.'
              f'Instances per time division: {self.instances_per_time_div}. '
              f'Using {self.time_divisions} time divisions, it results in '
              f'a total of {self.n_total_instances} instances to generate.')
        
        
        
        # Simulation specific variables:
        self.sim_numerology_idx = np.where(self.numerology == 
                                           self.simulation_numerology)
        self.n_prb = self.n_prb[self.sim_numerology_idx][0]
        
        # TODO: we are calling PRB what are actually frequency samples.
        #       we are doing that because in wideband precoding and scheduling
        #       there is no difference.
        
        # In Hertz, the fundamental bandwidths:
        # self.subcarrier_spacing = 15e3 * 2 ** self.simulation_numerology
        # self.prb_bandwidth = self.subcarrier_spacing * 12
        
        # Compute other variables from previous definitions
        
        # Set decent formats to make posterior computations quicker
        self.time_to_send *= self.TTI_duration
        self.packet_lat_budget = ut.timestamp(ms=self.packet_lat_budget)
        
        # Total delay threshold on the packets. Discard after this mark.        
        self.delay_threshold = self.packet_lat_budget - self.time_to_send
        
        # Pass packet size to bits, will be used to compute the number of 
        # packets 
        self.packet_size *= 8 
        
        # The percentage of radio bits that are actually application bits.
        # Currently we only consider this overhead, which probably doesn't 
        # account for guard symbols or symbols of other types (e.g. in DL
        # frame, there are only DL symbols)
        self.DL_radio_efficiency = \
            1 - self.DL_radio_overhead[self.sim_freq_idx]
        self.UL_radio_efficiency = \
            1 - self.UL_radio_overhead[self.sim_freq_idx] 
        
        # ################### Override variables ###########################
        
        # This considers the samples of the generated bandwidth were for a 
        # different bandwidth. If we start doing subband precoding or 
        # scheduling, then this should change. 
        if bw != None:
            print(f'Warning! Overriding bandwidth from {self.bandwidth/1e6} '
                  f'to {bw} MHz!')
            self.bandwidth_multiplier = bw * 1e6 / self.bandwidth
            self.bandwidth = bw * 1e6
        
        # ################### SLS Variables ###########################
        
        # TTIs in each time division.
        self.time_div_ttis = int(self.gen_TTIs / self.time_divisions)
        
        # Actual channel samples per time divisions
        self.sim_samples = int(self.sim_duration_secs / self.update_rate)
        self.time_div_samples = int(self.sim_samples / self.time_divisions)

        # Wideband noise power (linear, [W]) 
        self.wideband_noise_power = (self.boltz * self.temperature * 
                                     self.bandwidth)
        
        self.wideband_noise_power_dl = \
            self.wideband_noise_power * (10 ** (self.ue_nf/10))
        
        # self.wideband_noise_power_ul = \
        #     self.wideband_noise_power * (10 ** self.bs_nf)
            
        self.noise_power_per_prb_dl = \
            self.wideband_noise_power_dl / self.n_prb
            
        # self.noise_power_per_prb_ul = \
        #     self.wideband_noise_power_ul / self.n_prb
        
        
        self.olla_stepsize = self.olla_stepsize / self.tbs_divisor
        
        
        # ################# UE/BS SUBSETTING ##################
        # Allows the simulation of less UEs or BSs than those present in the 
        # channel traces
        
        # Set the variables the trace was generated to:
        self.n_ue_gen = self.n_ue
        self.n_phy_gen = self.n_phy
        self.n_bs_gen = self.n_bs
        
        # The variables n_ue, n_bs and n_phy can now be overwritten
        # depending on the number of users we want to simulate. Done now:
        if self.use_subset_for_simulation:
            self.n_ue = self.sim_n_ue
            self.n_phy = self.sim_n_phy
            self.n_bs = self.sim_n_bs
        else:
            self.specify_bss = [i for i in range(self.n_bs)]
            self.specify_ues = [i for i in range(self.n_ue)]
       
        
       
        ######### for variables that are to be used solely for naming folders#
        
        self.speed_idx = vars_dict['const_mvnt_value'][0][0]
        
        
    def compute_application_traffic_vars(self):
        # ################# Application Parameters ##################
        
        # Interval between consecutive frames [s]
        self.frame_interval = ut.timestamp(1 / self.FPS)
        
        # The application model will consider an aggregated Downlink, i.e.
        # the total traffic in the DL is the sum of all traffic in the UL, not
        # only of the PHY users, but for aggregated over the VIR users too
        # (depending on being AR or VR)
        if self.frame_calc_strat == 'AVG_BITRATE_UL':
            # (I_size + (GOP-1) P_size) / GOP * FPS = avg.bitrate
            # with P_size = I_size * IP_ratio.
            # so: I_size = avg.bitrate / FPS * GOP / (1 + (GOP-1) * IP_ratio)
            self.I_size_UL = (self.avg_bitrate_ul / self.FPS * self.GoP / 
                              (1 + (self.GoP - 1) * self.IP_ratio))
            
        elif self.frame_calc_strat == 'BASE_PARAMETERS':
            # Compute from the base parameters
            I_size_UL_uncompressed = (self.camera_resolution *
                                      (self.pixel_format * 
                                       self.pixel_depth + 
                                       self.depth_resolution)
                                      )
            
            # average frame size with I P framing
            avg_frame_size = (I_size_UL_uncompressed + 
                              I_size_UL_uncompressed * self.IP_ratio * 
                              (self.GoP - 1)) / self.GoP
                              
            
            # The compression ratio obtained by doing I and P framing
            IP_compression_ratio = I_size_UL_uncompressed / avg_frame_size
            
            # Remaining compression ratio after accounting the I & P frames 
            # compression
            RCR = self.video_compression_ratio / IP_compression_ratio
            
            # Here the bitrate is only used 
            self.I_size_UL = I_size_UL_uncompressed * RCR
            
            if self.tile_prediction:
                self.I_size_UL = self.I_size_UL * self.tiles_prediction_ratio
            
            # To finalise, since 2 camera streams will be aggregated in the UL
            if self.ul_aggregation:
                self.I_size_UL *= 2
        elif self.frame_calc_strat == 'AVG_BITRATE_DL':
            self.I_size_DL = (self.avg_bitrate_dl / self.FPS * self.GoP / 
                              (1 + (self.GoP - 1) * self.IP_ratio))
            
            
        # Consider a Naive approach:
        # The DL size will just be a aglomeration of all uplinks, thus, the
        # relation between how much information is there in the UL and in the 
        # DL is a constant away, and said constant depends on VR or AR, because
        # in AR the user doesn't need information about who is present, and in
        # VR the information of all users (all UL) needs to be sent to each DL.
        if self.VR_or_AR == 'VR':
            ul_dl_information_ratio = self.n_vir + self.n_phy
        elif self.VR_or_AR == 'AR':
            ul_dl_information_ratio = self.n_vir

        # Compute the size for the I frame in the opposite direction            
        if self.frame_calc_strat in ['AVG_BITRATE_UL', 'BASE_PARAMETERS']:
            self.I_size_DL = self.I_size_UL * ul_dl_information_ratio
        elif  self.frame_calc_strat in ['AVG_BITRATE_DL']:
            self.I_size_UL = self.I_size_DL / ul_dl_information_ratio
            
            
        # Convert sizes from Mbit to bits since the Transport Block Sizes
        # (chunks of data we'll send at a time) will be computed in bits
        self.I_size_UL = int(self.I_size_UL * 1e6)
        self.I_size_DL = int(self.I_size_DL * 1e6)
        
    
    def load_gob_params(self, precoders_dict):
        
        self.gob_n_beams = precoders_dict[(0, 'n_directions')]
        self.gob_directions = precoders_dict[(0, 'directions')]
        
        
        
