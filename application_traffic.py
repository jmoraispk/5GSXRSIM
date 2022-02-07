# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:43:04 2020

@author: janeiroja
"""

import numpy as np
import copy
import matplotlib.pyplot as plt

import utils as ut
import pandas as pd
import time
import sys
import os

pd.options.mode.chained_assignment = None  # default='warn'
# For memory efficiency, instead of generating the complete amount of packets
# that will go throuhg the bufffer during the simulation duration, we are 
# generating only a GoP size worth of frames, and converting that to packets.

# In this file are the functions to generate Frame Sequences and packetise 
# them in Packet Sequences
# Also, functions to manage the Buffers in UEs uplinking and BSs downlinking


"""
  SEE BELOW!!!:    
  - Should the buffer have a function to sort the packets,
    so that those containing I-frames will be at the front, and the scheduler 
    looks at the number of I-frame packets in the upcoming e.g. 100 packets and 
    and have some kind of multiplier alongside the HOL delay to compute the 
    UE priority?
    E.g. 0 - 9 = 1x, 10 - 19 = 1.1x, 20 - 20 = 1.2x, etc......
    => Depends on size of packets relative to size of TBs!!! 
       (What about dynamic TB sizes???)

-> Add method to sort the buffer firstly by frame-type and then HOL delay
   -> Return #I-frame packets (from the e.g. first 100 packets in buffer)
   -> Use this information in the scheduler    

"""


class PCAP_File:
    def __init__(self, pcap_data, tti_duration, sim_dur, total_ttis, 
                 burst_param, burst_model, space_UE_frames, ue, n_ues):
        # Create Object for the PCAP file which stores information for the 
        # whole pcap trace, e.g. packet sizes, timestamps, indices etc. 
        
        # print("\nStart timing PCAP time")
        # tic = time.perf_counter()

        # Save pcap trace in pandas dataframe
        self.pcap_data = pcap_data 
        
        # For adjusting timestamps for different users (spacing)
        self.ue = ue
        self.n_ues = n_ues
        self.space_UE_frames = space_UE_frames
        self.offset = 0
        
        self.pcap_FPS = 0 
        self.pcap_GoP = 0
        self.pcap_bitrate = 0
        
        # Adjust total packets based on total TTIs simulated
        self.total_packets = 0
        self.sim_dur = sim_dur
        self.tti_duration = tti_duration
        self.total_ttis = total_ttis
        
        # Make some adjustments for easier calculations and efficiency
        self.adjust_pcap_file()
        
        # Adjust all timestamps 
        self.apply_burstiness(burst_param, burst_model)
        
        # Use to modify synchronization of streams between different UEs
        self.ue = ue 
        self.n_ues = n_ues
        
        
        # Create two column array with  
        self.packets_per_tti = np.full((total_ttis, 2), -1, int)
        self.index_packets(self.tti_duration, self.total_ttis)
        
        ## self.pcap_data.to_csv('test_pcap.csv')
        
        # toc = time.perf_counter()        
        # print(f"Init PCAP time: {toc-tic:0.4f} seconds.\n")


    def adjust_pcap_file(self):
        """        
        (Some might not be needed with premodified traces (from queue_sim))
        1. Adjust the types of dataframe to save some memory and execution time
           
        2. Adjust packet sizes to be in bits instead of bytes
        
        3. Calculate and save parameters (FPS, bitrate, etc.)
        
        4. Cut file until total_ttis
        
        """
        
        # Add index column for easier tracking and saving with Frame_info()  
        self.pcap_data["index"] = self.pcap_data.index
        
        # Cut file to match simulation duration / total_ttis
        self.pcap_data = self.pcap_data[self.pcap_data['time'] < \
                                        (self.sim_dur * 1.0)]    
        self.pcap_FPS = (self.pcap_data["frame"].iloc[-1] + 1) / self.sim_dur
        self.pcap_data = self.pcap_data[self.pcap_data['time'] < \
                                        ((self.sim_dur * 1.0) - 0.5)]  
        # Change packet times
        """
        # done using queue_sim
        start_time = self.pcap_data['time'][0]
        self.pcap_data['time'] = self.pcap_data['time'].apply(lambda x: x - 
                                                              start_time)
        Change packet timestamps based on user number -> spacing!
        (alternative or combine with different SEEDs for users)
        """
        
        # Calculate FPS - Bitrate - GoP
        # self.pcap_GoP = self.pcap_data.loc[(self.pcap_data["frametype"] == True)
        #                     & (self.pcap_data["frame"] > 1)].iloc[0]["frame"]
        
        # Between 0 and 1 -> 0 all the same, 1 max timeshift
        param = self.space_UE_frames 
        self.offset = self.ue * (1 / (self.pcap_FPS * self.n_ues)) * param
        self.pcap_data["time"] += self.offset
          
                        
        # Change packet sizes from bytes in pcap file to bits
        self.pcap_data['size'] *= 8
                    
        self.pcap_bitrate = sum(self.pcap_data["size"]) / self.sim_dur
        
        # Cut file/packets based on timestamps until total_ttis
        self.total_packets = self.pcap_data["index"].iloc[-1]
        
        # Add column to collect PDR stats and arrival time at UE
        self.pcap_data["success"] = False 
        self.pcap_data["arr_time"] = 0.000000
               
        
        # Change dtypes
        # self.pcap_data["time"] = self.pcap_data["time"].astype("float32")
        self.pcap_data["size"] = self.pcap_data["size"].astype("int16")
        self.pcap_data["frame"] = self.pcap_data["frame"].astype("int16")
        self.pcap_data["index"] = self.pcap_data["index"].astype("int32")
        

    def apply_burstiness(self, burstiness_par, burstiness_model):
        """
        Adjust all packet timestamps to achieve the desired traffic burstiness.

        Parameters
        ----------
        burstiness_param : FLOAT
            Determines the packet arrival rate in the buffer.
        
        Raises
        ------
        SystemExit
            In case of incorrect/unusable input for burstiness_parameter.

        Returns
        -------
        None.

        """
        if burstiness_par > 1.0 or burstiness_par < 0.0:
            print('Warning: Burstiness parameter has to be between 0.0 and' + \
                  '1.0!\nPlease adjust the parameter in sxr_sim.py!')
            raise SystemExit
        elif burstiness_par == 1:
            # a prevention for division by zero  
            burstiness_par = 1 - 1e-9    
        
        if burstiness_model == 'Queue':
           # print("Using 'queue_sim' for dispersion model.")           
            
           return
        
        elif burstiness_model == 'Zheng':
            print("Using 'Zheng's' dispersion model.")           

            # Step 1: Set all packets belonging to same frame to timestamp set by
            #         the frame rate, i.e. Frame 0 = 0s, Frame 1 = 0.033s etc.        
            time_betw_frames = 1 / self.pcap_FPS
            
            self.pcap_data["time"] = self.pcap_data["frame"] * time_betw_frames
            
            # Step 2: Check how many packets are present for each frame
            for frame in range(self.pcap_data['frame'].iloc[-1] + 1):
                # Save indices of current frame in list
                packets_frame_i = [] # start with empty list
                packets_frame_i.append(self.pcap_data.index[
                    self.pcap_data['frame'] == frame][0].tolist())
                packets_frame_i.append(self.pcap_data.index[
                    self.pcap_data['frame'] == frame][-1].tolist())
                nr_packets = packets_frame_i[1] - packets_frame_i[0] + 1
                
                # Step 3: Depending on set burstiness, equally divide the time 
                # between two frames and add incrementally to each packet by index
                time_betw_packets = (time_betw_frames * (1 - burstiness_par)) \
                                     / nr_packets
             
                for packet in range(packets_frame_i[0], packets_frame_i[1] + 1):
                    self.pcap_data['time'][packet] += (time_betw_packets * 
                        (self.pcap_data['index'][packet] - 
                            self.pcap_data['index'][packets_frame_i[0]]))
                 
        else: 
            print("Burstiness Model unknown, please choose one out of" + 
                  "'Zheng' and 'Queue'.")
                
                
    def index_packets(self, tti_duration, total_ttis):
        """      
        Parameters
        ----------
        tti_duration : float
            Duration of one TTI, set in simulation_parameters.
        total_ttis : int
            Total TTIs of the simulation.
            
        Returns
        -------
        Creates list with index of packets belonging into each timeframe of ttis.

        """
        
        for tti in range(total_ttis): 
            # if tti == total_ttis - 1: print(tti)
            start_time = tti * tti_duration
            end_time = start_time + tti_duration
            
            packet_list = self.pcap_data['index'].loc[(self.pcap_data['time'] < end_time)
                             & (self.pcap_data['time'] >= start_time)].tolist()
            if packet_list != []: 
                self.packets_per_tti[tti][0] = packet_list[0]                                                      
                self.packets_per_tti[tti][1] = packet_list[-1]              
    
                                                      
                      
class PCAP_Packet_Sequence:
    def __init__(self, pcap_file, idx, frames, sizes, types, timestamps):
        
        # Verify?!: 
        # Things like FPS, GoP size etc. should not matter or be worried about
        # since it is all incorporated in the information from timestamps etc.
        
        self.pcap_file = pcap_file # Shortcut to pd dataframe
        
        # Arrays with:
        self.index = idx    
        self.frames = frames # frames each packet belongs to        
        self.sizes = sizes # size of each packet in bytes
                
        # type of parent frame
        # GoP size is 10 -> every frame-number%10=0 packet belongs to I-frame!
        self.frametypes = types 
        
        # Take time stamp of generation? or RTP timestamp?
        self.timestamps = timestamps # timestamp of creation/arrival        
           
                
    def print_pcap_info(self):
        print(f"The current packet sequence consists of" 
              f"{self.total_size} packets with head of queue delay of" 
              f"{self.head_of_queue_lat} ms.") # What unit is the delay???
        return



def gen_pcap_sequence(pcap_file, curr_tti, max_size=0):
    """
    Generate a sequence of packets from pcap file  
    Input: - Object containing stats from PCAP trace
           - Current timestamp (?)
           - Total size of sequence / Nr packets in sequence (?)       
    
    Idea:  
        -> At call, create sequence of packets that will arrive e.g. within the
           next scheduling slot (i.e. 250ms), 
        -> During the simulation at every queue update use this function to 
           create new packets that are then added at the end of the buffer
           (OR: another function that modifies the packets inside buffer)
        
    """      
    # Create lists for packet information    
    index, frames, sizes, frametypes, timestamps = ([] for i in range(5))      
        
    return PCAP_Packet_Sequence(pcap_file, index, frames, sizes, frametypes, 
                                timestamps)



class PCAP_Buffer:
    def __init__(self, pcap_packet_sequence, packet_delay_threshold, 
                 delay_type, file_name, file_folder, ue, scheduler):
        """
        The only functions in this class that should be used outside of the
        class are: 
            - update_queue_time()
            - print_buffer()
            - remove_bits()
            - update_queue_delay() (should be safe as well...)
        """

        self.ue = ue
        self.scheduler = scheduler
        # Track latency at the head of the queue
        self.delay_type = delay_type
        self.head_of_queue_lat = ut.timestamp(0)                
        self.delay_threshold = packet_delay_threshold.total_seconds()
        
        # Empty buffer variable, to signal when the buffer has no packets
        self.is_empty = True
            
        # To know which packet sequence the buffer will be updated with.
        self.pcap_seq = pcap_packet_sequence
        self.pcap_file = self.pcap_seq.pcap_file # copy for good measure
                            
        # Packet size tracker - tracks the bits left to send in each packet
        # Equal to packet sizes!!! -> use for creating TBs/Removing bits
        self.bits_left = [] 
                
                
        """
            XX Number of packets containing I-frames in current buffer 
            Or
            Check packet(s) at head of buffer for frame type 
        """
        self.I_packets = False
        
        
    # def init_pdr_info(self): # , total_ttis):
    #     """
    #     Initialize list for packets within simulated TTIs 
        
    #     2-dim: 
    #         - Packet index 
    #         - Status:
    #             - True = Successfully transmitted
    #             - False = Droppped due to latency budget        

    #     Returns
    #     -------
    #     None.

    #     """
    #     pass
    
    #     return
        
        
    def add_new_packets(self, tti):        
        """
        Update the back of the queue (adds packets).
        It also adds packets that arrive exactly at the same time as the TTI. 
            This function is called before every TTI simulation, the very first
            time it will be "passed", since buffers have been initialized with
            packets arriving in the first TTI duration, after that, new packets
            will come 
        """
        
        idx_0 = self.pcap_file.packets_per_tti[tti][0]
        
        # Check if there are new packets in this tti
        if idx_0 >= 0:
            idx_1 = self.pcap_file.packets_per_tti[tti][1] + 1
            
            self.pcap_seq.index.extend(range(idx_0, idx_1))
            
            self.pcap_seq.frames.extend(self.pcap_file.pcap_data
                                        ['frame'][idx_0:idx_1].tolist()) 
            self.pcap_seq.sizes.extend(self.pcap_file.pcap_data
                                       ['size'][idx_0:idx_1].tolist()) 
            self.pcap_seq.frametypes.extend(self.pcap_file.pcap_data
                                            ['frametype'][idx_0:idx_1].tolist()) 
            self.pcap_seq.timestamps.extend(self.pcap_file.pcap_data
                                            ['time'][idx_0:idx_1].tolist()) 
            
            # Update bit tracker
            self.bits_left = self.pcap_seq.sizes.copy()
            # Buffer is filled
            self.is_empty = False 
            
        return

    
    def update_head_of_queue_delay(self, tti, tti_duration):
        """ 
        Update the latency of the first packet in the buffer        
        Updated method of calculating remaining RAN latency budget
        Add method to calculate E2E frame level latency 
        """
        curr_time = (tti + 1) * tti_duration 
        
        if self.delay_type == 'RAN':
            if self.pcap_seq.index != []: # Check if buffer is empty
                RAN_lat = curr_time - self.pcap_seq.timestamps[0]
                self.head_of_queue_lat = ut.timestamp(RAN_lat)                
            else: 
                self.head_of_queue_lat = ut.timestamp(0)
            return    
        
        elif self.delay_type == 'E2E':
            if self.pcap_seq.index != []:
                # Calculate already experienced E2E delay of head of queue
                # E2E latency starts from time of frame-generation (+ offset)
                packet_gen_time = (self.pcap_seq.frames[0] * (1 / \
                              self.pcap_file.pcap_FPS)) + self.pcap_file.offset
                E2E_lat = curr_time - packet_gen_time                    
                self.head_of_queue_lat = ut.timestamp(E2E_lat)              
            else: 
                self.head_of_queue_lat = ut.timestamp(0)
            return    
        
        else: 
            print("Choose one of the following methods to calculate " + 
                  "packet latencies: 'RAN' or 'E2E'!")
            raise SystemExit
            

    def get_I_packets_info(self):
        """        
        Option a) Look at first xxx packets in the buffer and return the 
                  current number of packets containing an I-frame and use this
                  as weight in the scheduler         
                  -> The exact value of how many packets are looked into might 
                  have to be determined for every TTI, with inputs like packet 
                  size (from sim_par), maximum achievable throughput (or even 
                  better the average achieved TP) per scheduling time slot and 
                  sub-band...
        
        Option b) If there are any I-packets at the head (or first xxx packets)
                  of the buffer, the weight is determined by the delay of the 
                  first I-packet, otherwise by the earliest P-packet
                  
        """         
        # Frametype of parent frame of packet at head of queue
        if not self.is_empty and self.pcap_seq.frametypes != []: 
            self.I_packets = self.pcap_seq.frametypes[0]
            
        else: self.I_packets = False
        
        return
    
    
    # not needed??? 
    """
    # (Set default as unsuccessful / the other way around)
    # -> If using "dumb" PF - all packets will arrive -> add. info in arr time
    def increment_dropped_packet_stats(self, tti, tti_dur_in_sec, idx_dropped):
        
        # time of arrival at UE will be current TTI plus time to send 

        arrival_tti = tti + 1 
        arrival_time = arrival_tti * tti_dur_in_sec        
        
        for idx in idx_dropped:
            self.pcap_file.pcap_data["arr_time"][idx] = 0.0
            self.pcap_file.pcap_data["success"][idx] = False
        
        return
    """
    
    def increment_success_packet_stats(self, tti, tti_dur_in_sec, 
                                       time_to_send, idx_success):
        """
        Save the arrival time of packets at the UE after successful 
        transmissions -> Called after removing bits from Buffer

        Parameters
        ----------
        tti : TYPE
            DESCRIPTION.
        tti_dur_in_sec : TYPE
            DESCRIPTION.
        time_to_send : TYPE
            DESCRIPTION.
        idx_success : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # time of arrival at UE will be current TTI plus time to send 
        time_to_send = time_to_send.total_seconds() 
        arrival_time = tti * tti_dur_in_sec + time_to_send        
                
        for idx in idx_success:
            self.pcap_file.pcap_data["arr_time"][idx] = float(arrival_time)
            self.pcap_file.pcap_data["success"][idx] = True
        
        return
    
        
    def remove_bits(self, tti, tti_dur_in_sec, time_to_send, bits_to_remove, 
                    start_idx=0):
        """
        Removes bits from the buffer that have been succesfully transmitted 
        Note: Update head of queue delay if head is removed 
        is removed.
        
        It starts the packet removal procedure from the beginning of the queue
        if nothing is told in contrary. Else, remove packets starting in a 
        specific part of the queue.
                        
        """
        # Count total bits to remove
        bits_to_remove = bits_to_remove
        
        packet_idx = self.pcap_seq.index.index(start_idx)
        
        idx_empty = []
        
        while bits_to_remove != 0: 
            if self.pcap_seq.sizes[packet_idx] == 0:
                break
            if self.pcap_seq.sizes[packet_idx] <= bits_to_remove: 
                bits_to_remove -= self.pcap_seq.sizes[packet_idx] 
                self.pcap_seq.sizes[packet_idx] = 0 
            else: 
                self.pcap_seq.sizes[packet_idx] -= bits_to_remove 
                
                bits_to_remove = 0
            packet_idx += 1
        
        # Check which packets are empty now  
        for i in range(len(self.pcap_seq.sizes)): 
            if self.pcap_seq.sizes[i] <= 0:
               idx_empty.append(i)    
               
        # delete empty entries   
        if idx_empty != []:
            
            idx_success = self.pcap_seq.index[idx_empty[0]:idx_empty[-1] + 1]
            del self.pcap_seq.index[idx_empty[0]:idx_empty[-1] + 1]
            del self.pcap_seq.frames[idx_empty[0]:idx_empty[-1] + 1]
            del self.pcap_seq.sizes[idx_empty[0]:idx_empty[-1] + 1]
            del self.pcap_seq.frametypes[idx_empty[0]:idx_empty[-1] + 1]
            del self.pcap_seq.timestamps[idx_empty[0]:idx_empty[-1] + 1]
            
            self.increment_success_packet_stats(tti, tti_dur_in_sec, 
                                                time_to_send, idx_success)

        # Update bit tracker
        self.bits_left = self.pcap_seq.sizes.copy()
            
        # Check if buffer is empty after update
        if self.pcap_seq.index == []:
            self.is_empty = True
            
        return
                
    def drop_packet(self):
        
        del self.pcap_seq.index[0]
        del self.pcap_seq.frames[0]
        del self.pcap_seq.sizes[0]
        del self.pcap_seq.frametypes[0]
        del self.pcap_seq.timestamps[0]
                    
        # Update bit tracker
        self.bits_left = self.pcap_seq.sizes.copy()
    
        if self.pcap_seq.index == []:
            self.is_empty = True
    
    def discard_late_packets(self, tti, tti_duration):
        """
        Discards packets until their latencies are within what's achievable.
        Note: the latency of the head will determine if packets are discarded
        or not, therefore, be sure to update it before.
                      
      
        """
        
        curr_time = tti * tti_duration
        index_late_packets = -1 
        
        # Pure proportional fair does not drop any packets at BS
        if self.scheduler == 'PF': return 
        
        # RAN packet latency based
        elif self.delay_type == 'RAN':            
            for index, timestamp in enumerate(self.pcap_seq.timestamps):            
                if curr_time - timestamp > self.delay_threshold: 
                    index_late_packets += 1                   
              
        # E2E frame latency based 
        elif self.delay_type == 'E2E':           
            for index, timestamp in enumerate(self.pcap_seq.timestamps):   
                frame_gen_time = self.pcap_seq.frames[index] * (1 / \
                                     self.pcap_seq.pcap_file.pcap_FPS) + \
                                         self.pcap_seq.pcap_file.offset
                if curr_time - frame_gen_time > self.delay_threshold: 
                    index_late_packets += 1                   
                    
        # Discard packets if necessary
        if index_late_packets >= 0:            
            
            index = index_late_packets + 1
            del self.pcap_seq.index[:index]
            del self.pcap_seq.frames[:index]
            del self.pcap_seq.sizes[:index]
            del self.pcap_seq.frametypes[:index]
            del self.pcap_seq.timestamps[:index]
                        
            # Update bit tracker
            self.bits_left = self.pcap_seq.sizes.copy()
                    
        # Check if buffer is empty after update
        if self.pcap_seq.index == []:
            self.is_empty = True
          
        return
        
        
    def update_queue_time(self, tti, tti_duration):
                
        self.add_new_packets(tti) # Add packets depending on TTI time
        self.update_head_of_queue_delay(tti, tti_duration) # Delay=curr_tti - packet_timestamp!
        self.discard_late_packets(tti, tti_duration) #    
        self.get_I_packets_info()      
       
                
    def gen_transport_blocks(self, bits_left_in_TBs_total, tb_size, tti):
        """
        Returns the transport blocks list tupples with a given size and index.
        The size may be smaller than {tb_size} (namely the last transport block
        or in case the buffer has less bits in it than the transport blocks can 
        carry.
        The index is the packet start index of that transport block.
        
        """
        list_of_transport_blocks = []
                
        num_packets = len(self.bits_left)
                        
        start_idx = 0 # Index of packet at start of a TB
        packet_idx = 0 # Index of packet where bits are deduced
        
        while bits_left_in_TBs_total != 0 and any(self.bits_left): 
                       
            curr_tb_size = 0
            
            while curr_tb_size < tb_size and packet_idx < num_packets:                    
                # Go through all packets in the buffer
                # Add bits from packets in buffer until TB full or no bits left
                if bits_left_in_TBs_total == 0 or not any(self.bits_left):
                    break
                
                bits_left_in_packet = self.bits_left[packet_idx]
                                    
                # More bits in packet than left in TB
                if bits_left_in_packet >= tb_size - curr_tb_size: 
                    bits_to_deduce = tb_size - curr_tb_size
                    
                # Less bits in packet than left in TB
                else: 
                    bits_to_deduce = bits_left_in_packet                

                bits_to_deduce = min(bits_to_deduce, bits_left_in_TBs_total)                             
                
                curr_tb_size += bits_to_deduce
                bits_left_in_TBs_total -= bits_to_deduce
                self.bits_left[packet_idx] -= bits_to_deduce                
                    
                if self.bits_left[packet_idx] <= 0: 
                    packet_idx += 1     
                                    
            if curr_tb_size != 0:    
                list_of_transport_blocks += \
                    [(curr_tb_size, self.pcap_seq.index[start_idx])]
                start_idx = packet_idx
                                
        return list_of_transport_blocks

    
    def create_pdr_csv(self, file_name, file_path):
        """
        Call at the end of the simulation
        
        Save information about dropped & successful packets in csv file to use 
        on pcap trace and replay modified video stream
         
        Returns
        -------
        None.

        """                
    
        full_file_name = os.path.join(file_path, file_name)        
        os.makedirs(file_path, exist_ok=True)

        # Add UE info to output file - create folder for all UE buffers   
        self.pcap_file.pcap_data.to_csv(full_file_name, encoding= 'utf-8') 
                                        # 'utf-16-LE')
        # np.savetxt(full_file_name, self.pdr_info, delimiter=",") #, fmt='%s')
        # if self.ue == 0: 
        # print("Saved csv: ", full_file_name)
        
        return

    
class Frame_Sequence:
    def __init__(self, sizes, types, timestamps, fps):
        
        # Check if sizes are in accordance
        # num_frames = len(sizes)
        # ut.parse_input_lists(sizes, ['int', 'float'], length=num_frames)
        # ut.parse_input_lists(types, ['char'], length=num_frames)
        # ut.parse_input_lists(timestamps, ['datetime.timedelta'], 
        #                      length=num_frames)
        # maybe there's no point making verification if only functions in this
        # module interact with this class.
        
        # Frame Size [bits]
        self.sizes = sizes
        # Type of Frame ['I', 'P']
        self.types = types
        # Timestamp at which the frame arrives to the buffer
        self.timestamps = timestamps
        
        # Frame Indices
        self.frame_idxs = [i for i in range(len(sizes))]
        
        # To facilitate, some stats of the sequence:
        self.GoP = len(sizes)
        
        # GoP_duration = self.timestamps[-1] - self.timestamps[0]
        
        # Since the first frame starts at 0
        # frame_interval = GoP_duration / (self.GoP - 1)
        # self.FPS = round(1 / (frame_interval.seconds + 
        #                       frame_interval.microseconds / 1e6))
        self.FPS = fps
        
        
    def print_frames(self):
        frames_to_print = list(zip(self.types, self.sizes))
        # print(f"GoP of {self.GoP}")
        print('Type  Size [bits]     Timestamp ')
        for frame_idx in self.frame_idxs:
            print(f"{frames_to_print[frame_idx][0]}", end='')
            print("{:>14}".format(frames_to_print[frame_idx][1]), end=' ' * 7)
            print(self.timestamps[frame_idx])
            

class Packet_Sequence:
    def __init__(self, default_packet_size, timestamps, packet_bitrate,
                 parent_frames, frame_sequence, frame_types): 
                 # TODO: frametype   
        # Packet size for all packets in the sequence
        self.packet_size = default_packet_size  # [bits!!!]
        # Packet timestamps
        self.timestamps = timestamps
        # The frame from which the packet was generated (frame indice)        
        self.parent_frames = parent_frames
        # Parent Frame sequence
        self.parent_sequence = frame_sequence
        # print(frame_sequence.types)
        # Frametype of parent frame 
        self.packet_type = frame_types# TODO: Placeholder
        # Bitrate of the packets arriving, when it is constant
        self.packet_bitrate = packet_bitrate
        # Some utilities
        self.num_packets = len(timestamps)
        self.packet_idxs = [i for i in range(self.num_packets)]
        
        # Copy for convenience
        self.GoP = self.parent_sequence.GoP
        self.FPS = self.parent_sequence.FPS
        self.sequence_duration = ut.timestamp(self.GoP / self.FPS)

    
    def print_packets(self, first_x_packets=5):
        """ 
        Prints first_x_packets of each frame, with x=5 by default.
        """
        
        print(f"There are {self.num_packets} packets.")
        print('Packet  Frame     Timestamp ')
        for frame in self.parent_sequence.frame_idxs:
            first_packet_of_frame = self.parent_frames.index(frame)
            for i in range(first_x_packets):
                packet_idx = first_packet_of_frame + i
                print(f"{packet_idx:5}"
                      f"{'':5}"
                      f"{self.parent_frames[packet_idx]:<8}"
                      f"{self.timestamps[packet_idx]}")
                
            
    def plot_sequence(self, save=False, save_name='', cumulative=False, 
                      light=False, alpha=1, bitrate_or_packets='packets',
                      time_resolution=1e-3, cursors=False, ticks=False):
        
        if bitrate_or_packets == 'packets':
            if not light:
                fig, ax = plt.subplots()
            
            up_lim = self.parent_sequence.GoP / self.parent_sequence.FPS
            
            z = np.array([t.microseconds for t in self.timestamps]) * 1e-6
            x = np.arange(0, up_lim, time_resolution)
            # There are -1 bins than points in x.
            
            if light:
                return plt.hist(z, x, cumulative=cumulative, alpha=alpha)
            else:
                bin_values, _, _ = ax.hist(z, x, cumulative=cumulative, 
                                           alpha=alpha)
            
            bin_values = bin_values[bin_values != 0].astype(int)
            
            plt.rcParams.update({'font.size': 17})
            plt.xlabel('Time [s]')
            plt.ylabel('Packets per ms')
            
            if ticks:
                y_ticks = plt.yticks()
                y_ticks = sorted(np.append(y_ticks[0], bin_values))
                plt.yticks(np.unique(y_ticks))
            
            
            if save:
                # plt.savefig('Py Plots\\' + ut.get_time() + '.svg', 
                #             bbox_inches = "tight")
                if save_name == '':
                    name = 'Py Plots\\packet_arrival' + ut.get_time() + '.png'
                    plt.savefig(name, dpi=300, bbox_inches = "tight")
                else:
                    plt.savefig('Py Plots\\' + save_name + '.png',
                                dpi=300, bbox_inches = "tight")
            
            if cursors:
                # datacursor(display='multiple', draggable=True, hover=True)
                plt.show()
        elif bitrate_or_packets == 'bitrate':
            # z = np.array([t.microseconds for t in self.timestamps]) * 1e-6
            
            # fig, ax = plt.subplots()
            # up_lim = self.parent_sequence.GoP / self.parent_sequence.FPS
            # x = np.arange(0, up_lim, time_resolution)
            # # There are -1 bins than points in x.
            
            # bin_values, _, _ = ax.hist(z, x, cumulative=cumulative)
            
            # NOTE: there is a variable called packet_bitrate... which has
            #       basically what we want.
            
            # TODO: PLOT APPLICATION ARRIVAL RATE (I.E ACCUMULATE PACKETS, 
            #                             DIVIDE BY BIN SIZE AND PACKET SIZE)
            
            plt.rcParams.update({'font.size': 17})
            plt.xlabel('Time [s]')
            plt.ylabel('Bit rate [Mbps]')
            
            if save:
                if save_name == '':
                    name = 'Py Plots\\arrival_rate' + ut.get_time() + '.png'
                    plt.savefig(name, dpi=300, bbox_inches = "tight")
                else:
                    plt.savefig('Py Plots\\' + save_name + '.png',
                                dpi=300, bbox_inches = "tight")
        else:
            raise Exception()
            
        
                    
def gen_frame_sequence(I_size, GoP, IP_ratio, FPS, offset=0):
    
    # Generate Frame sequence (a GoP) for UL and for DL
    # Frame sizes depend on the average traffic. Different avg. traffic will be 
    # considered for the uplink. For the downlink, it's only aggregated;
    
    # the frame size has to an integer number of bits
    P_size = int(I_size * IP_ratio)
    
    # We consider no variation in the P frame sizes.
    frame_sizes = [I_size] + [P_size] * (GoP - 1)
    frame_types = ['I'] + ['P'] * (GoP - 1)
    frame_timestamps = [ut.timestamp(0)] + [ut.timestamp(i / FPS)
                                            for i in range(1, GoP)]

    if offset != 0:
        gop_duration = GoP / FPS
        if offset > gop_duration:
            raise Exception('No point delaying it further than a GoP...')
            # We can handle with ease by taking the modulus, but value on this
            # argument is probably an error and should be raised.
        
        # include an offset of the I frame from the beginning of the sequence
        frame_timestamps = [f + ut.timestamp(offset) for f in frame_timestamps]

        frames_to_cycle = 0
        for f in range(GoP):
            if frame_timestamps[f] >= ut.timestamp(gop_duration):
                frame_timestamps[f] -= ut.timestamp(gop_duration)
                # count how many frames to cycle to the beginning
                frames_to_cycle += 1
                
        # Cycle frame types, sizes and timestamps around as well.
        for f in range(frames_to_cycle):
            # put in front and take from the back
            frame_sizes.insert(0,frame_sizes[-1])
            frame_sizes = frame_sizes[:-1]
            
            frame_types.insert(0,frame_types[-1])
            frame_types = frame_types[:-1]
            
            frame_timestamps.insert(0,frame_timestamps[-1])
            frame_timestamps = frame_timestamps[:-1]

    return Frame_Sequence(frame_sizes, frame_types, frame_timestamps, FPS)


def gen_packet_sequence(frame_sequence, packet_size, burstiness_param,
                        burst_model, overlap_packets_of_diff_frames=0): 
    
    """
    Generates a Packet_Sequences from a Frame_Sequence
    the burstiness parameter is used to know how much to space packets
    from a certain frame. 
    if 1, then all packets belonging to a frame are 
    placed practically at the same time instant of that frame generation time. 
    if 0, then it's like having a bottleneck and the packets have a constant 
    flow throughout the GoP.
    
    Note: packet size is in bits
    NOTE2: each packet has a parent frame, and the indices of those two lists
           must be coordinated
           Furthermore, the packet timestamps need to be sorted!
    TODO:  Incorporate info about I or P frame with these lists as well 
           
           
    A final comment about the packet arrival rate's transitions:
    they are abrupt. In a ms there can be no packets, and the next, we can
    have the maximum packet arrival rate. Our assumption is that gradually
    getting to the actual bit arrival rate is less realistic.
    
    """  
    
    if overlap_packets_of_diff_frames:
        print('Warning: I am not sure whether the overlap function is ' + \
              'working as intended after packet cycling. Namely sorting...')
        
    
    # The way we'll create this burstiness is throught a packet 
    # arrival/departure bitrate. High burstiness means a high bitrate, 
    # and vice-versa. Note that we are generating the sequences that will be 
    # appear in the buffers, so this bitrate resembles the packets going 
    # down the stack in the uplink, and the packets arriving at the BS buffers
    # in the downlink
    
    # Firstly, that bitrate computation [average application bitrate]
    frame_bitrate = (sum(frame_sequence.sizes) / 
                     frame_sequence.GoP * 
                     frame_sequence.FPS)
    
    if burstiness_param > 1.0 or burstiness_param < 0.0:
        print('Warning: Burstiness parameter has to be between 0.0 and 1.0!\n' + \
              'Please adjust the parameter in sim_par.py!')
        raise SystemExit
    elif burstiness_param == 1:
        # a prevention for division by zero
        burstiness_param = 1 - 1e-9
        
    # Instantaneous bitrate
    packet_bitrate = frame_bitrate * (1 / (1 - burstiness_param))
    
    # If burtiness_param = 0, Since the size of the I frame is
    # bigger than the P frames, at the same bitrate, the first P frame
    # would start being transmitted with some packets of the I frame still
    # left.
    # So, there are 2 options:
    #   - (overlap_packets_of_diff_frames = 0) Don't overlap the packets and
    #     put the P packets only after the final packet of the I frame;
    #   - (overlap_packets_of_diff_frames = 1) Do overlap. And there will be
    #     a different packet arrival rate in the place where the I frame is;
    
    # NOTE: to get the constant bitrate, the overlap option needs to be 0.
    packet_timestamps = []
    packet_parent_frames = []
    
    # Add frame type information to every packet (used in scheduling)
    packet_type = []
    
    # We need to start in the I frame, because P frames won't overlap among
    # themselves.
    
    # When packets lie outisde of the GoP, we put them back into the beginning. 
    
    # Get I frame idx and create idx list starting on the I frame
    I_frame_idx = frame_sequence.types.index('I')
    frame_idxs = []
    GoP = frame_sequence.GoP
    gop_duration = ut.timestamp(frame_sequence.GoP / frame_sequence.FPS) 
    frame_idxs = [I_frame_idx + i if I_frame_idx + i <= GoP - 1
                  else I_frame_idx + i - GoP for i in range(GoP)]
    
    # We need to count whether we cycled packets on not, to compare regarding
    # overlaps
    for frame_idx in frame_idxs:
        frame_size = frame_sequence.sizes[frame_idx]
        
        if burst_model == 'Joao':            
            # The packets will be equally spaced in time, and that spacing
            # depends on how long it takes to transmit a frame
            time_per_frame = frame_size / packet_bitrate                        
            # All packets are (assumed to be) created equal, i.e. same size.
            # TODO: Explore different options here!
            num_packets = int(np.ceil(frame_size / packet_size))
            first_packet_timestamp = frame_sequence.timestamps[frame_idx]
            packet_interval = ut.timestamp(time_per_frame / num_packets)    
            
        elif burst_model == 'Zheng':            
            # Packets will be equally spaced in time, but for every frame 
            # individually depending on the burstiness parameter
            time_per_frame = 1 / frame_sequence.FPS
            num_packets = int(np.ceil(frame_size / packet_size))
            first_packet_timestamp = frame_sequence.timestamps[frame_idx]
            packet_interval = ut.timestamp((time_per_frame * (1 - \
                                          burstiness_param)) / num_packets)   
                
        else:
            raise Exception("The only available burstiness models are 'Joao'"
                            " and 'Zheng'.")
            
        # only necessary on frames after the I frame
        if not overlap_packets_of_diff_frames and frame_idx != I_frame_idx:                       
            # even if it was at the end of the GoP, it may overlap with
            # packets from the beginning. If it does (the last packet is
            # already on the beginning) then the transformation should
            # happen anyway 
            already_transitioned = packet_timestamps[-1] < \
                frame_sequence.timestamps[frame_idx-1]
                
            if first_packet_timestamp < packet_timestamps[-1]: # TODO
                # check if the previous frame was at the end of the GoP
                frame_idx_in_gop = frame_idxs.index(frame_idx)
                at_the_end_of_GoP = frame_idx < frame_idxs[frame_idx_in_gop-1]
                                    
                if at_the_end_of_GoP and not already_transitioned:
                    pass
                else: 
                    first_packet_timestamp = (packet_timestamps[-1] + 
                                              packet_interval)
                    
            if first_packet_timestamp >= packet_timestamps[-1] and \
                already_transitioned: 
                # Check if transition happened and next frame which starts at 
                # the end of the GoP should instead start after previous frame
                first_packet_timestamp = (packet_timestamps[-1] + 
                                              packet_interval)          
            
        new_packet_times = [first_packet_timestamp + packet_interval * i
                            for i in range(num_packets)]
        
        if new_packet_times[-1] > gop_duration:
            new_packet_times = [pckt_t if pckt_t <= gop_duration else 
                                pckt_t - gop_duration 
                                for pckt_t in new_packet_times]        
        
        packet_timestamps += new_packet_times        
        packet_parent_frames += [frame_idx] * num_packets        
        
    # Sort both timestamps and parent frames
    packet_parent_frames_aux = copy.deepcopy(packet_parent_frames)
    packet_parent_frames = [parent_frame for _, parent_frame in 
                            sorted(zip(packet_timestamps, 
                                       packet_parent_frames_aux))]
    packet_timestamps.sort()
    
    # TODO: Frametype 
    frame_idxs.sort()
    for frame_idx in frame_idxs: 
        frame_size = frame_sequence.sizes[frame_idx]
        num_packets = int(np.ceil(frame_size / packet_size))
        packet_type += [frame_sequence.types[frame_idx]] * num_packets
             
    return Packet_Sequence(packet_size, packet_timestamps, packet_bitrate, 
                           packet_parent_frames, frame_sequence, packet_type)


def get_start_times(n, mode, FPS):       
    pass
       

class Frame_Info():
    # TODO: How to implement this for pcap trace???
    # Before: create frame-sequences and out of those gen_packet_seq
    # Now: Packets are given, gen_frame_seq from that???
    def __init__(self):
        # Number of packets dropped
        self.dropped_packets = 0
        # Average packet latency
        self.avg_lat = ut.timestamp(0)
        # Number of successful packets
        self.successful_packets = 0
    
    def print_info(self):
        print(f'Dropped packets = {self.dropped_packets}\n'
              f'Avg. latency of successful packets = {self.avg_lat}\n'
              f'Successful packets = {self.successful_packets}')


class Buffer:
    def __init__(self, parent_packet_sequence, packet_delay_threshold, 
                 delay_type):
        """
        The only functions in this class that should be used outside of the
        class are: 
            - update_queue_time()
            - print_buffer()
            - remove_bits()
            - update_queue_delay() (should be safe as well...)
        """
        # cursors A and B have absolute timestamps, respectively, of the first
        # and last packet in the queue. We assume that at most GoP can be
        # present in the buffer (which is already worth a few hundreds of ms,
        # so it shouldn't be a problem compared with the packet latency budget)
                
        # Cursor A is responsible for tracking the head of the queue
        self.cursor_a = ut.timestamp(0)
        # Cursor B is responsible for the last packet of the queue
        self.cursor_b = ut.timestamp(us=1)  # ignore, initialisation purposes
        # The packet indices are used to know which packets are in the queue
        # and to track their sizes
        self.cursor_a_idx = 0  # first_packet_idx
        self.cursor_b_idx = 0  # last_packet_idx
        # Ambiguities with plus or minus a packet in the buffers are decided
        # based on the bits_left variable, with the info on the sizes of the
        # packets in the queue.
        
        # Set the latency budget of each packet
        self.packet_delay_threshold = packet_delay_threshold
        
        # Track latency at the head of the queue
        self.head_of_queue_lat = ut.timestamp(0)
        self.delay_type = delay_type # TODO
                
        # Empty buffer variable, to signal when the buffer has no packets
        self.is_empty = True
        
        # self.num_packets_in_buffer = 0
    
        # To know which packet sequence the buffer will be updated with.
        self.parent_packet_seq = parent_packet_sequence
            
        # Period Index - used to know how many periods have passed
        self.periods_past = 0
        # Shortcut for the number of packets the buffer can hold
        self.buffer_size = parent_packet_sequence.num_packets
        
        # Packet size tracker - tracks the bits left to send in each packet
        self.bits_left = np.zeros(self.buffer_size, dtype=int)
        
        # Bits in each packet
        self.default_packet_size = int(parent_packet_sequence.packet_size)
        
        # Create all the information objects for each frame in the 1st GoP
        self.frame_infos = [[Frame_Info()
                             for i in range(self.parent_packet_seq.GoP)]]
                
        """
        TODO:
            Number of packets containing I-frames in current buffer 
            Or
            Check packet(s) at head of buffer for frame type 
        """
        self.I_packets = 0

    
    def get_relative_time(self, t):
        """
        Return the time with respect to the packet sequence.
        Note: Due to comparison reasons, it has to make the 0 into the maximum
        number. So the mapping is:  ]0, 200ms] (for the GoP and FPS considered,
        200ms is the duration of the packet sequence)
        """
        rel_time = t % self.parent_packet_seq.sequence_duration
        
        if rel_time == ut.timestamp(0):
            rel_time = self.parent_packet_seq.sequence_duration
        
        return rel_time
    
    
    def create_new_batch_of_frame_infos(self):
        self.frame_infos += [[Frame_Info()
                              for i in range(self.parent_packet_seq.GoP)]]
    
    
    def add_new_packets(self, tti):
        
        """
        Update the back of the queue (adds packets).
        It also adds packets that arrive exactly at the same time as the TTI. 
       
        """
        # It is assumed that the buffer can only have as many packets as 1 GoP
        # cursor_a can't pass cursor_b (absolute times)
    
        if tti <= self.cursor_b:
            # we've added the packets up to this tti before...
            print('Packets up to this tti were added already.')
            return
        
        # Time relative to the beginning of the period
        relative_time = self.get_relative_time(tti)
        
        if relative_time > self.get_relative_time(self.cursor_b):
            # means it still is the same period, just further ahead
            i = self.cursor_b_idx
        else: 
            # means that the period has ended and that it's starting from
            # the beginning again
            i = 0
            self.cursor_b_idx = 0
            self.periods_past += 1
            self.create_new_batch_of_frame_infos()
        
        new_packets = 0
        
        # while the indice is within limits 
        while i + new_packets < self.buffer_size:
            # if the packet is before the time of the update, add it!
            if (self.parent_packet_seq.timestamps[i + new_packets] <= 
               relative_time): 
                new_packets += 1
            else:
                break
        
        # the final index after the packets are added
        i += new_packets

        # If there was new packets in this tti:        
        if i != self.cursor_b_idx:
            
            if self.is_empty:
                # update the time of arrival of the first packet, for keeping
                # track of the latency
                self.cursor_a = self.get_timestamp(self.cursor_b_idx)
                self.cursor_a_idx = self.cursor_b_idx
                self.is_empty = False
        
            # Update sizes of packets that just arrived
            self.bits_left[self.cursor_b_idx:i] = \
                np.ones(new_packets, dtype=int) * self.default_packet_size
               
            # Increment the cursor
            self.cursor_b_idx = i
            
        self.cursor_b = tti
        
        
    def increment_cursor_a(self):
        self.cursor_a_idx += 1
        if self.cursor_a_idx == self.cursor_b_idx:
            self.is_empty = True
        if self.cursor_a_idx == self.buffer_size:
            self.cursor_a_idx = 0
        if self.cursor_a_idx == self.cursor_b_idx:
            self.is_empty = True # yes, needs to be here again
        if not self.is_empty:
            self.cursor_a = self.get_timestamp(self.cursor_a_idx)
    
    
    def update_head_of_queue_delay(self, tti):
        """ 
        Uses cursor_a to know new latency for the packet in front.
        """
        self.delay_type # TODO
        # Don't do anything if the buffer is empty, or the tti has passed 
        if self.is_empty:
            return
        
        while self.bits_left[self.cursor_a_idx] == 0 and not self.is_empty:
            self.increment_success_packet_stats(self.cursor_a_idx)
            self.increment_cursor_a()
            if self.is_empty:
                break        
        
        if tti >= self.cursor_b:
            # Don't do anything if the input is smaller than the last packet
            # added, since the queue is updated from that already.
            self.head_of_queue_lat = tti - self.cursor_a


    def get_I_packets_info(self):
        """        
        Option a) Look at first xxx packets in the buffer and return the 
                  current number of packets containing an I-frame and use this
                  as weight in the scheduler         
                  -> The exact value of how many packets are looked into might 
                  have to be determined for every TTI, with inputs like packet 
                  size (from sim_par), maximum achievable throughput (or even 
                  better the average achieved TP) per scheduling time slot and 
                  sub-band...
        
        Option b) If there are any I-packets at the head (or first xxx packets)
                  of the buffer, the weight is determined by the delay of the 
                  first I-packet, otherwise by the earliest P-packet
                  
        """         

        self.I_packets = False
        
        # if self.parent_packet_seq.packet_type[self.cursor_a_idx] == 'I' and \
        #     self.bits_left[self.cursor_a_idx] != 0:  
        #         self.num_I_packets = 1    
        if self.cursor_a_idx < self.cursor_b_idx:
            for i in range (self.cursor_a_idx, self.cursor_b_idx + 1):
                if i < self.buffer_size and self.bits_left[i] != 0 and \
                    self.parent_packet_seq.packet_type[i] == 'I': # 
                        self.I_packets = True
                        return
        else:           
            for i in range(self.cursor_b_idx, self.buffer_size):
                if self.bits_left[i] != 0 and \
                    self.parent_packet_seq.packet_type[i] == 'I':
                    self.I_packets = True
                    return
            for i in range(0, self.cursor_a_idx):
                if self.bits_left[i] != 0 and \
                    self.parent_packet_seq.packet_type[i] == 'I':
                    self.I_packets = True
                    return
                
        """
        Like in MAC scheduling paper: 
            for i in range (cursor a, cursor b)
                if bitsleft(i) != 0 and packet_type == 'I'
                    get packet timestamp 
                    I_packets = True
                    return
                
        parent_packet_seq.timestamps[i] - self.head_of_queue_lat 
        """
    
    
    def increment_dropped_packet_stats(self, packet_idx):
        period = self.period_packet_belongs(packet_idx)
        frame_idx = self.parent_packet_seq.parent_frames[packet_idx]
        
        self.frame_infos[period][frame_idx].dropped_packets += 1
            
    def period_packet_belongs(self, packet_idx):
        
        periods = self.periods_past 
        
        if self.cursor_a_idx > self.cursor_b_idx:
            # it means the packets are in transition of periods
            # therefore, from cursor_a_idx to buffer_size, they have
            # 1 less period than the actual number of periods that passed!
            if packet_idx >= self.cursor_a_idx:
                periods -= 1
                
        return periods
    
    
    def increment_success_packet_stats(self, packet_idx):
        
        period = self.period_packet_belongs(packet_idx)
        frame_idx = self.parent_packet_seq.parent_frames[packet_idx]
        lat = self.cursor_b - self.get_timestamp(packet_idx)
                
        # Update the average latency of the packets of that frame
        if self.frame_infos[period][frame_idx].successful_packets != 0:
            self.frame_infos[period][frame_idx].avg_lat = \
                (((self.frame_infos[period][frame_idx].avg_lat * 
                   self.frame_infos[period][frame_idx].successful_packets) + 
                  lat) / 
                 (self.frame_infos[period][frame_idx].successful_packets + 1))
        else:
            self.frame_infos[period][frame_idx].avg_lat = lat
        
        # increments success counter of the frame
        self.frame_infos[period][frame_idx].successful_packets += 1
    
    
    def drop_packet(self):
        """
        Removes the first packet of the queue.
        Updates the drop rate of the frame it belonged to.
        TODO: Maybe add something to control the PDR during the simulation!!!
        """
        
        # If it is called, there must be packets to pop
        self.bits_left[self.cursor_a_idx] = 0    
        self.increment_dropped_packet_stats(self.cursor_a_idx)
        self.increment_cursor_a()
    

    def remove_bits(self, curr_tti, TTI_dur_in_sec, time_to_send, n_bits, 
                    start_idx=-1):        
        """
        Pops n packets out of the queue. 
        Note: the head of queue delay should be updated afterwards if the head
        is removed.
        
        It starts the packet removal procedure from the beginning of the queue
        if nothing is told in contrary. Else, remove packets starting in a 
        specific part of the queue.
        
        Note: the packets are actually only removed from the queue when 
        the queue's head delay is updated.
        """
        
        # Remove from the head
        if start_idx == -1:
            start_idx = self.cursor_a_idx
            
        bits_left_to_sent = n_bits
        idx = start_idx
        
        while bits_left_to_sent != 0:
            if self.bits_left[idx] == 0 and idx == self.cursor_b_idx - 1:
                # self.is_empty = True
                break
            if self.bits_left[idx] <= bits_left_to_sent:
                bits_left_to_sent -= self.bits_left[idx]
                self.bits_left[idx] = 0
            else:
                self.bits_left[idx] -= bits_left_to_sent
                bits_left_to_sent = 0
            idx += 1
            if idx == self.buffer_size:
                idx = 0
                if idx == self.cursor_b_idx:
                    # Empty
                    break
        
        
    def discard_late_packets(self, tti):
        """
        Discards packets until their latencies are within what's achievable.
        Note: the latency of the head will determine if packets are discarded
        or not, therefore, be sure to update it before.
        
        """
        
        while not self.is_empty and (self.head_of_queue_lat > 
                                         (self.packet_delay_threshold)):
            
            # discard packet    
            self.drop_packet()
            
            # update how delayed is the packet in front
            self.update_head_of_queue_delay(tti)
        
        
    def update_queue_time(self, tti):

        self.add_new_packets(tti) # Add packets depending on TTI time
        self.update_head_of_queue_delay(tti) # Delay=curr_tti - packet_timestamp!
        self.discard_late_packets(tti) #    
        self.get_I_packets_info()

        
    def count_non_empty_packets(self):
       
        # Since only newly created packets are added to buffer
        """
        Counts packets that are not empty.
        """
        if self.cursor_a_idx < self.cursor_b_idx:
            # no need for tricks, count packets from a to b
            num_packets = sum(
                (1 for i in range(self.cursor_a_idx, self.cursor_b_idx + 1)
                 if i < self.buffer_size and self.bits_left[i] != 0))
        else:
            num_packets = sum(
                (1 for i in range(self.cursor_b_idx, self.buffer_size)
                 if self.bits_left[i] != 0))
            num_packets += sum(
                (1 for i in range(0, self.cursor_a_idx)
                 if self.bits_left[i] != 0))
            
        return num_packets
    
    
    def cursor_diff(self):
        """
        Returns the difference between the two cursors, taking into account
        they may be in a different order.
        """
        
        if self.cursor_a_idx <= self.cursor_b_idx:
            diff = self.cursor_b_idx - self.cursor_a_idx
        else:
            diff = self.buffer_size - (self.cursor_a_idx - self.cursor_b_idx)
            
        return diff
    
    
    def get_timestamp(self, packet_idx):
        
        """
        Returns the timestamp of a certain packet idx in the queue.
        """
        periods = self.period_packet_belongs(packet_idx)        
        return (self.parent_packet_seq.timestamps[packet_idx] + 
                periods * self.parent_packet_seq.sequence_duration)
    
    def print_buffer(self, first_x_packets=5):
        """
        Prints some general information about the buffer, like its contents
        """
        
        print(f"Cursor A index: {self.cursor_a_idx}, "
              f"Cursor B index: {self.cursor_b_idx}. "
              f"Diff: {self.cursor_diff()}")
        # first_x_packets = self.cursor_diff()
        print(f"Have past {self.periods_past} periods.", end='')
        
        print(f"The buffer is "
              f"{'empty' if self.is_empty else 'NOT empty'}")
        if not self.is_empty:
            print(f"{self.count_non_empty_packets()} "
                  f"packets in the buffer with bits still to send.")
        
            print(f"The head has {self.head_of_queue_lat} of delay.")
            
            print(f"\nThe first {first_x_packets} packets:")
            print("Index   Bits_Left_to_send    Frame    Arrival")
            
            
            for i in range(min(first_x_packets, self.cursor_diff())):
                packet_idx = self.cursor_a_idx + i
                if packet_idx >= self.buffer_size:
                    packet_idx -= self.buffer_size
                print(
                    f"{packet_idx:4}"
                    f"{self.bits_left[packet_idx]:16}"
                    f"{self.parent_packet_seq.parent_frames[packet_idx]:13}"
                    f"{'':6}"
                    f"{self.get_timestamp(packet_idx)}")
   
    
    def print_frame_infos(self, periods=[0]):
        for period in periods:
            print(f"Period {period}.")
            frame_idx = 0
            for frame_info in self.frame_infos[period]:
                print(f"{'':4} Frame {frame_idx}:")
                print(f"{'':8} Successful packets: "
                      f"{frame_info.successful_packets}")
                print(f"{'':8} Dropped packets: {frame_info.dropped_packets}")
                print(f"{'':8} Avg. Latency: {frame_info.avg_lat}")
                frame_idx += 1


    def gen_transport_blocks(self, bits_left_to_put_into_TBs, tb_size, tti):
        """
        Returns the transport blocks list tupples with a given size and index.
        The size may be smaller than {tb_size} (namely the last transport block
        or in case the buffer has less bits in it than the transport blocks can 
        carry.
        The index is the packet start index of that transport block.
        
        """
        
        list_of_transport_blocks = []
        
        buffer_cursor = self.cursor_a_idx
        is_empty = self.is_empty
        bits_left_in_packet = self.bits_left[buffer_cursor]        
        
        while bits_left_to_put_into_TBs != 0 and not is_empty:
            
            # Go to next non-empty packet
            while bits_left_in_packet == 0:
                buffer_cursor += 1
                if buffer_cursor == self.cursor_b_idx:
                    is_empty = True
                    break
                if buffer_cursor == self.buffer_size:
                    buffer_cursor = 0 
                bits_left_in_packet = self.bits_left[buffer_cursor]
            start_packet_idx = buffer_cursor 
            
            curr_tb_size = 0
                        
            # Add packets until the TB is full or the buffer is empty
            while curr_tb_size < tb_size:
                if bits_left_to_put_into_TBs == 0 or is_empty:
                    break
                if bits_left_in_packet == 0:
                    buffer_cursor += 1
                    if buffer_cursor == self.cursor_b_idx:
                        is_empty = True
                        break
                    if buffer_cursor == self.buffer_size:
                        buffer_cursor = 0
                        if buffer_cursor == self.cursor_b_idx:
                            is_empty = True
                            break
                    bits_left_in_packet = self.bits_left[buffer_cursor]
                
                if bits_left_in_packet <= tb_size - curr_tb_size:
                    bits_to_deduce_from_packet = bits_left_in_packet
                else:
                    bits_to_deduce_from_packet = tb_size - curr_tb_size
                
                # Take bits from the packet: either as many as the packet has, 
                # or as many as the TB holds; the smallest of the two.
                bits_to_deduce_from_packet = min(bits_to_deduce_from_packet,
                                                 bits_left_to_put_into_TBs)
                
                curr_tb_size += bits_to_deduce_from_packet
                bits_left_to_put_into_TBs -= bits_to_deduce_from_packet
                bits_left_in_packet -= bits_to_deduce_from_packet
            
            # Add current TB to the TB list
            if curr_tb_size != 0:
                list_of_transport_blocks += [(curr_tb_size, start_packet_idx)]
                
        return list_of_transport_blocks
    