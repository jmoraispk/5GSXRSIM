# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:40:30 2021

@author: duzhe
"""

import numpy as np
import pandas as pd
import operator
import time 
import os


"""
TODOs:
    - Correct choice/tuning of parameters, especially for random variables 
    
    - Performance optimization (15s, 5 queues, 50 kbps => 15 seconds)
    - Implementation: 
          - "BUG" - IF PACKETS HAVE THE SAME ARRIVAL TIME, THE ORDER IS THE REVERSE 
              OF THE INDICES 
              - Default interpacket time? 
              - Second sort command to sort by packet index!      
    - "Last event time" not needed - Last departure time for each queue enough!
    - Nr. of queues: Do research on realistic number of hops -> traceroute
    
"""



class Event: 
    def __init__(self, event_time, action, queue, packet):
        
        # Time at which an event occurs
        self.time = event_time
        # Type of action: Packet arrival vs departure  
        self.action = action
        # Location of event (queue i)  
        self.queue = queue
        # Packet involved in the event
        self.packet = packet
        self.packet_type = packet.packet_type


class Packet:
    def __init__(self, packet_id, packet_size, queue, # arr_time, dep_time, 
                 packet_type):
        
        # ID of packet - null/-1 if background packet, otherwise VR idx 
        self.id = packet_id
        # Size of packet
        self.size = packet_size
        # Current location / queue of packet
        self.queue = queue
        # Time of entry into whole queueing system
        # self.arrival = arr_time
        # Time of departure from last queue (and into BS buffer)
        # self.departure = dep_time
        # Background traffic vs. VR packet - 'BG' & 'VR'
        self.packet_type = packet_type 
        
        
class Queue: 
    def __init__(self, serving_bitrate):
        
        # (Fixed) Serving bitrate of queue
        self.bitrate = serving_bitrate
        # List of current packets in queue
        self.packet_list = []
        
        
def initialise_event_calendar(vr_timestamps, vr_sizes, queues, max_packet_size, 
                              max_time, exp_size, exp_time, sim_time, debug): 

    # Initialize event calendar to track all packets etc.
    event_calendar = []
    max_size = max_packet_size
    min_size = np.floor(max_size / 100)
    # Generate all background packet arrivals in each queue
    for q in range(queues):
        
        curr_time = 0.0000
        bg_count = 0
        
        while curr_time < sim_time: 
            
            if debug[0]:
                if bg_count > debug[1]:
                    break
            
            np.random.seed(1)
            
            # Exponentially distributed packet size (max size = XXX)
            new_size = int(np.random.exponential(exp_size))                    
            # Exponentially distributed inter-packet arrival times
            inter_arr_time = np.random.exponential(exp_time)    
            curr_time += inter_arr_time 
            
            if new_size > max_size:
                new_size = max_size
                
            elif new_size < min_size:
                new_size = min_size
            
            if curr_time < sim_time:
                bg_packet = Packet(packet_id=-1, packet_size=new_size, queue=q, 
                                   # arr_time=curr_time, dep_time=None, 
                                   packet_type='BG')
                if debug[0]:
                    event_calendar.append(Event(0, 'packet_arrival', q, 
                                            bg_packet))
                else:
                    event_calendar.append(Event(curr_time, 'packet_arrival', q, 
                                            bg_packet))
                    
            bg_count += 1
            
        print(f"Queue: {q} - BG packets: {bg_count}")
                
        
    # Generate all packet arrivals for VR packets at the first queue
    curr_time = 0.0000
    vr_packet_counter = 0
    total_packets = len(vr_timestamps)
    
    while curr_time <= sim_time and vr_packet_counter < total_packets:
        
        curr_time = vr_timestamps[vr_packet_counter]
        new_size = vr_sizes[vr_packet_counter]
        
        if curr_time < sim_time:
            
            vr_packet = Packet(packet_id=vr_packet_counter, packet_size=new_size,
                               queue=0, packet_type = 'VR')
                               # arr_time=curr_time, dep_time=None, 
                               
            event_calendar.append(Event(curr_time, 'packet_arrival', 0, 
                                        vr_packet))
        vr_packet_counter += 1    
        
    print("VR packets:", vr_packet_counter)
        
    return event_calendar, vr_packet_counter



def main(serving_bitrate, n_queues, max_packet_size, max_time, exp_size, 
         exp_time, sim_time, debug):
    
    tic = time.perf_counter()      
    
    # Folder with packet traces 
    file_folder = r"C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\Trace" + '\\'
    file_to_simulate = file_folder + "trace_APP10.csv"
    
    # Load into dataframe
    sim_data = pd.read_csv(file_to_simulate, encoding='utf-16-LE')  
    # Packet timestamp count starts at zero 
    sim_data['time'] = sim_data['time'].apply(lambda x: x - sim_data['time'][0])
    
    # Adjust timestamps to match frame generation time for simulation
    # TODO: Is this needed???
    # fps = int(np.ceil(sim_data["frame"].iloc[-1] / sim_data["time"].iloc[-1]))
    # frame_time = 1 / fps    
    # sim_data['time'] = sim_data['frame'] * frame_time 
    
    vr_timestamps = sim_data['time'].values
    vr_sizes = sim_data['size'].values 
    
    if debug[0]:
        test_number = debug[1]
        
        vr_timestamps = sim_data['time'][0:test_number].values
        print("vr_timestamps", vr_timestamps)
        
        vr_sizes = sim_data['size'][0:test_number].values 
        print("vr_sizes", vr_sizes)
        
    if n_queues < 1: 
        print('Warning: Number of queues cannot be less than 1!' + \
              '\nPlease change the number of queues in the main() arguments!')
        raise SystemExit
    queues = n_queues # [Queue(serving_bitrate) for i in range(n_queues)]    
    
    toc = time.perf_counter()
    print(f"Initializing Video File: {toc-tic:0.4f} seconds")
    
    tic = time.perf_counter()    
    event_calendar, total_vr_packets = \
        initialise_event_calendar(vr_timestamps, vr_sizes, queues, 
                                  max_packet_size, max_time, exp_size, 
                                  exp_time, sim_time, debug)
        
    toc = time.perf_counter()        
    print(f"Initializing Event Calendar: {toc-tic:0.4f} seconds")
    print(f"Total start events: {len(event_calendar)}")
    
    # raise SystemExit
    
    ##### Start of event simulation #####
    print("Starting Event Simulation...")

    tic = time.perf_counter()    
    curr_time = 0.000
    
    # For performance and debugging
    counter = 0
    vr_packet_counter = 0
    vr_timestamps_end = np.zeros(total_vr_packets)
    
    
    # last_event_time = np.zeros(n_queues)
    last_departure_time = np.zeros(n_queues)

    while event_calendar != []: # and curr_time < sim_time and :
    
        if debug[0]:
            print(f"\nIterations: {counter} - Time: {curr_time}")
    
        if counter % 1000 == 0:
            print(f"\nIterations: {counter} - Time: {curr_time}")
        
        # Always first sort the event calendar by time 
        event_calendar.sort(key = operator.attrgetter('time'), reverse = True)
        next_event = event_calendar.pop()
        
        # Keep track of location of events for proper timestamping
        curr_queue = next_event.queue        
        # Keep track of current simulation time
        # curr_time = max(next_event.time, last_event_time[curr_queue])
        curr_time = next_event.time
        
        if debug[0]:
            print(f"Current time: {curr_time}") # " - Last event time: {last_event_time}")
            print(f"Event: {next_event.packet.packet_type} - {next_event.action}" + \
                  f" - {next_event.packet.size}")
        
        # Simulate arrival of packet in queue
        if next_event.action == 'packet_arrival':
            
            # Calculate departure time for packet
            serving_time = next_event.packet.size / serving_bitrate      
            
            if curr_time >= last_departure_time[curr_queue]:     
                new_departure_time = curr_time + serving_time 
            else: 
                new_departure_time = serving_time + last_departure_time[
                                                        curr_queue] 
                
            # Update last departure time for respective queue
            last_departure_time[curr_queue] = new_departure_time
            
            if debug[0]:
                print("New departure time:", new_departure_time, 
                      next_event.packet_type)                                 
            # Create new event for packet departure into new queue
            event_calendar.append(Event(new_departure_time, 'packet_departure', 
                                        next_event.queue, next_event.packet))   
            
            # last_event_time[curr_queue] = curr_time
            # print("Arrival - new last event time:", last_event_time)
            # arrival_counter += 1
            
            
        elif next_event.action == 'packet_departure': 
                        
            if debug[0]:
                print(next_event.packet.packet_type)
                
            # VR packets are send to next queue or if at last queue, to the BS
            if next_event.packet.packet_type == 'VR':            
                
                # last_event_time[curr_queue] += curr_time
                # print("Departure - new last event time:", last_event_time)
                next_queue = next_event.queue + 1
                
                # Departure from last queue - send to BS
                if next_queue >= n_queues:
                    # Save time for correct packet ID
                    vr_timestamps_end[next_event.packet.id] = curr_time                   
                    vr_packet_counter += 1
                    
                # Otherwise send to next queue in simulation 
                else: 
                    # Create new arrival event with new parameters
                    # Departure time is new arrival time 
                    # (assuming instant travel between queues)
                    new_arr_time = curr_time 
                    packet = next_event.packet
                    
                    event_calendar.append(Event(new_arr_time, 'packet_arrival', 
                                                next_queue, packet))          
            else:
                pass
                # BG packet departs from queue, update last event time
                # last_event_time[curr_queue] += curr_time
                # print("Departure - new last event time:", last_event_time)
        # else: 
        #     print('Warning: Unknown event in the calendar!' + \
        #           '\nPlease check the proper initialization of events!')
        #     raise SystemExit
        
        counter += 1
        
    output_save_path = file_folder # + file_name + "\\"
    output_file_name = f'{file_to_simulate.strip(".csv")}_burst.csv'

    full_file_name = os.path.join(output_save_path, output_file_name)
    
    np.savetxt(full_file_name, vr_timestamps_end, delimiter=",")# , fmt='%s')
            
    if debug[0]:
        print("Final:", vr_timestamps_end[0:debug[1]])
    
    toc = time.perf_counter()    
    print(f"Finished Event Simulation: {toc-tic:0.4f} seconds")

    print("Final VR packets:", vr_packet_counter)


if __name__ == "__main__":

    main(serving_bitrate=100000000, n_queues=5, max_packet_size=1500, max_time=0.02,
         exp_size=500, exp_time=0.005, sim_time=10, debug=[False,3])





