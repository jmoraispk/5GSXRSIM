# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:43:04 2020

@author: janeiroja
"""

import numpy as np
import copy
import matplotlib.pyplot as plt

import utils as ut
# from mpldatacursor import datacursor


# For memory efficiency, instead of generating the complete amount of packets
# that will go throuhg the bufffer during the simulation duration, we are 
# generating only a GoP size worth of frames, and converting that to packets.

# In this file are the functions to generate Frame Sequences and packetise 
# them in Packet Sequences
# Also, functions to manage the Buffers in UEs uplinking and BSs downlinking

"""
TODO Zheng

- Add information about Frame type (I or P) in the IP packets. 
  (Perhaps add separate array with containing 'I/0' or 'P/1')
  Information should be used by the scheduler to calculate UE priority
  Option to save stats about drop rate of packets containing I or P frames
      
  SEE BELOW!!!:    
- When users are scheduled, (multiple) transport blocks at the head of
  their buffer are being sent. When scheduling priority is computed with
  frame-type information, how and where exactly should the frame-type 
  be incorporated???
  - Should a single transport block only contain packets belonging to 
    one certain type of frame? 
  - Following from that, should the buffer have a function to sort the packets,
    so that those containing I-frames will be at the front, and the scheduler 
    looks at the number of I-frame packets in the upcoming e.g. 100 packets and 
    and have some kind of multiplier alongside the HOL delay to compute the 
    UE priority?
    E.g. 0 - 9 = 1x, 10 - 19 = 1.1x, 20 - 20 = 1.2x, etc......
    => Depends on size of packets relative to size of TBs!!! 
       (What about dynamic TB sizes???)

-> Add attribute in buffer containing packet's frame-type
-> Add method to sort the buffer firstly by frame-type and then HOL delay
   -> Return #I-frame packets (from the e.g. first 100 packets in buffer)
   -> Use this information in the scheduler    

"""

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
                        overlap_packets_of_diff_frames=0): # TODO: 'frametype'
    
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
    
    if burstiness_param == 1:
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
    
    # TODO: Add frame type information to every packet (used in scheduling)
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
        
        # The packets will be equally spaced in time, and that spacing
        # depends on how long it takes to transmit a frame
        time_per_frame = frame_size / packet_bitrate
        
        # All packets are (assumed to be) created equal, i.e. same size.
        # TODO: Explore different options here!
        num_packets = int(np.ceil(frame_size / packet_size))

        first_packet_timestamp = frame_sequence.timestamps[frame_idx]
        packet_interval = ut.timestamp(time_per_frame / num_packets)
                
        
        # only necessary on frames after the I frame
        if not overlap_packets_of_diff_frames and frame_idx != I_frame_idx:
            if first_packet_timestamp < packet_timestamps[-1]:
                # check if the previous frame was at the end of the GoP
                frame_idx_in_gop = frame_idxs.index(frame_idx)
                at_the_end_of_GoP = frame_idx < frame_idxs[frame_idx_in_gop-1]
                
                # even if it was at the end of the GoP, it may overlap with
                # packets from the beginning. If it does (the last packet is
                # already on the beginning) then the transformation should
                # happen anyway
                already_transitioned = packet_timestamps[-1] < \
                    frame_sequence.timestamps[frame_idx-1]
                
                if at_the_end_of_GoP and not already_transitioned:
                    pass
                else: 
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
    def __init__(self, parent_packet_sequence, packet_delay_threshold):
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
        
        # Period Index - used to know how many periods have passed
        self.periods_past = 0
        
        # Empty buffer variable, to signal when the buffer has no packets
        self.is_empty = True
        
        # self.num_packets_in_buffer = 0
    
        
        # Shortcut for the number of packets the buffer can hold
        self.buffer_size = parent_packet_sequence.num_packets
        
        # Packet size tracker - tracks the bits left to send in each packet
        self.bits_left = np.zeros(self.buffer_size, dtype=int)
        
        # Bits in each packet
        self.default_packet_size = int(parent_packet_sequence.packet_size)
        
        # To know which packet sequence the buffer will be updated with.
        self.parent_packet_seq = parent_packet_sequence

        # Create all the information objects for each frame in the 1st GoP
        self.frame_infos = [[Frame_Info()
                             for i in range(self.parent_packet_seq.GoP)]]
        
        # TODO: Number of packets containing I-frames in current buffer 
        self.num_I_packets = 0
                
    
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
        
        
    def update_number_I_packets(self):
        """
        TODO: Look at first xxx packets in the buffer and returns the current 
        number of packets containing an I-frame
        -> The exact value of how many packets are looked into might have to 
        be determined dynamically, with inputs s.a. packet size (from sim_par),
        maximum achievable throughput (or even better the average achieved TP)
        per scheduling time slot and sub-band...
        For 100 MHz and 0.25ms slots, lets put the range as xxxMbit/packet size 
        -> For now with 480Mbps and 1.5kB packets this gives max. 10 packets 
        (-> 3000kB - 5 packets; 7500kB - 2 packets)
        """         
        self.num_I_packets = 0
        
        # Loop over all/the first (x) packets in buffer and check the frametype
        # of the parent frame for every packet
        # only check for packets in the buffer that have bits left to send!
        if self.cursor_a_idx < self.cursor_b_idx:
            for i in range (self.cursor_a_idx, self.cursor_a_idx + 10):
                if i < self.buffer_size and self.bits_left[i] != 0 and \
                    self.parent_packet_seq.packet_type[i] == 'I':
                        self.num_I_packets += 1
        else:           
            for i in range(self.cursor_b_idx, self.buffer_size):
                if self.bits_left[i] != 0 and \
                self.parent_packet_seq.packet_type[i] == 'I':
                    self.num_I_packets += 1
            for i in range(0, self.cursor_a_idx):
                if self.bits_left[i] != 0 and \
                self.parent_packet_seq.packet_type[i] == 'I':
                    self.num_I_packets += 1
        
    
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
        """
        
        # If it is called, there must be packets to pop
        self.bits_left[self.cursor_a_idx] = 0    
        
        self.increment_dropped_packet_stats(self.cursor_a_idx)
        
        self.increment_cursor_a()
    
    
    def remove_bits(self, n_bits, start_idx=-1):
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
        # TODO: Check !!!
        self.add_new_packets(tti)
        self.update_head_of_queue_delay(tti)
        self.discard_late_packets(tti)    
        self.update_number_I_packets()

        
    def count_non_empty_packets(self):
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



def gen_transport_blocks(buffer, bits_left_to_put_into_TBs, tb_size, tti):
    """
    Returns the transport blocks list tupples with a given size and index.
    The size may be smaller than {tb_size} (namely the last transport block
    or in case the buffer has less bits in it than the transport blocks can 
    carry.
    The index is the packet start index of that transport block.
    
    """
    
    list_of_transport_blocks = []
    
    buffer_cursor = buffer.cursor_a_idx
    is_empty = buffer.is_empty
    bits_left_in_packet = buffer.bits_left[buffer_cursor]
    
    while bits_left_to_put_into_TBs != 0 and not is_empty:
        
        # Go to next non-empty packet
        while bits_left_in_packet == 0:
            buffer_cursor += 1
            if buffer_cursor == buffer.cursor_b_idx:
                is_empty = True
                break
            if buffer_cursor == buffer.buffer_size:
                buffer_cursor = 0 
            bits_left_in_packet = buffer.bits_left[buffer_cursor]
        start_packet_idx = buffer_cursor 
        
        curr_tb_size = 0
        
        # Add packets until the TB is full or the buffer is empty
        while curr_tb_size < tb_size:
            if bits_left_to_put_into_TBs == 0 or is_empty:
                break
            if bits_left_in_packet == 0:
                buffer_cursor += 1
                if buffer_cursor == buffer.cursor_b_idx:
                    is_empty = True
                    break
                if buffer_cursor == buffer.buffer_size:
                    buffer_cursor = 0
                    if buffer_cursor == buffer.cursor_b_idx:
                        is_empty = True
                        break
                bits_left_in_packet = buffer.bits_left[buffer_cursor]
            
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
    
    
    


        



