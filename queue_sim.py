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
        - Default interpacket time? -> Prevent overtaking of packets! (needed?)
          (Due to service time = bitrate/bits!!!)
              
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


class Packet:
    def __init__(self, packet_id, packet_size, queue, arr_time, dep_time, 
                 packet_type):
        
        # ID of packet - null/-1 if background packet, otherwise VR idx 
        self.id = packet_id
        # Size of packet
        self.size = packet_size
        # Current location / queue of packet
        self.queue = queue
        # Time of entry into whole queueing system
        self.arrival = arr_time
        # Time of departure from last queue (and into BS buffer)
        self.departure = dep_time
        # Background traffic vs. VR packet - 'BG' & 'VR'
        self.packet_type = packet_type 
        
        
# class Queue: 
#     def __init__(self, serving_bitrate):
        
#         # (Fixed) Serving bitrate of queue
#         self.bitrate = serving_bitrate
#         # List of current packets in queue
#         self.packet_list = []
        
        
def initialise_event_calendar(vr_timestamps, vr_sizes, queues, max_packet_size, 
                              max_time, exp_size, exp_time, sim_time): 

    # Initialize event calendar to track all packets etc.
    event_calendar = []
    max_size = max_packet_size
    min_size = np.floor(max_size / 100)
    # Generate all background packet arrivals in each queue
    for q in range(queues):
        curr_time = 0.0000
        while curr_time < sim_time:
            
            # Exponentially distributed inter-packet arrival times
            inter_arr_time = np.random.exponential(exp_time)            
            curr_time += inter_arr_time 
            
            # Exponentially distributed packet size (max size = XXX)
            new_size = int(np.random.exponential(exp_size))
            
            if new_size > max_size:
                new_size = max_size
                
            elif new_size < min_size:
                new_size = min_size
            
            if curr_time < sim_time:
                bg_packet = Packet(packet_id=-1, packet_size=new_size, queue=q, 
                                   arr_time=curr_time, dep_time=0, packet_type='BG')
                event_calendar.append(Event(curr_time, 'packet_arrival', q, 
                                            bg_packet))

    # Generate all packet arrivals of VR session at first queue
    curr_time = 0.0000
    packet_counter = 0
    total_packets = len(vr_timestamps)
    
    while curr_time <= sim_time and packet_counter <= total_packets - 1:
        
        curr_time = vr_timestamps[packet_counter]
        new_size = vr_sizes[packet_counter]
        
        if curr_time < sim_time:
            
            vr_packet = Packet(packet_id=packet_counter, packet_size=new_size,
                               queue=0, arr_time=curr_time, dep_time=0, 
                               packet_type = 'VR')
            event_calendar.append(Event(curr_time, 'packet_arrival', 0, 
                                        vr_packet))
        packet_counter += 1    
        
    return event_calendar


def main(serving_bitrate, n_queues, max_packet_size, max_time, exp_size, 
         exp_time, sim_time):
    
    # return serving_bitrate
    
    tic = time.perf_counter()      
    
    # Folder with traces 
    file_folder = r"C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\Trace" + '\\'
    file_to_simulate = file_folder + "trace_APP10.csv"
    
    # Load video trace into dataframe
    sim_data = pd.read_csv(file_to_simulate, encoding='utf-16-LE')  
    
    sim_data['time'] = sim_data['time'].apply(lambda x: x - sim_data['time'][0])
    
    fps = int(np.ceil(sim_data["frame"].iloc[-1] / sim_data["time"].iloc[-1]))
    frame_time = 1 / fps
    
    # Adjust timestamps to match frame generation time for simulation
    sim_data['time'] = sim_data['frame'] * frame_time 
    
    vr_timestamps = sim_data['time'].values
    vr_sizes = sim_data['size'].values
    # vr_sizes *= 8 # Convert to bits
    
    if n_queues < 1: 
        print('Warning: Number of queues cannot be less than 1!' + \
              '\nPlease change the number of queues in the main() arguments!')
        raise SystemExit
    queues = n_queues # [Queue(serving_bitrate) for i in range(n_queues)]    
    
    toc = time.perf_counter()
    print(f"Initializing Video File: {toc-tic:0.4f} seconds")
    
    tic = time.perf_counter()    
    event_calendar = initialise_event_calendar(vr_timestamps, vr_sizes, queues, 
                                               max_packet_size, max_time, 
                                               exp_size, exp_time, sim_time)
    toc = time.perf_counter()        
    print(f"Initializing Event Calendar: {toc-tic:0.4f} seconds")
    print(len(event_calendar))
    
    # raise SystemExit
    
    ##### Start of event simulation #####
    print("Starting Event Simulation...")

    tic = time.perf_counter()    

    curr_time = 0.000
    
    # For performance and debugging
    counter = 0
    vr_packet_counter = 0
    vr_timestamps_burst = np.zeros(len(vr_timestamps))
    
    arrival_counter = 0
    
    while curr_time < sim_time and event_calendar != []:
    
        if counter % 1000 == 0:
            print(f"Events: {counter} - Time: {curr_time}")
        
        # if counter == 10000:
        #     raise SystemExit
        
        # Always first sort the event calendar by time 
        event_calendar.sort(key = operator.attrgetter('time'), reverse = True)
        next_event = event_calendar.pop()

        curr_time = next_event.time
        
        # filter_arr = filter(lambda x: x.action == 'packet_arrival', event_calendar)
        # filter_dep = filter(lambda x: x.action == 'packet_departure', event_calendar)

        # print(next(filter_arr).action)
        # print(next(filter_dep).action, None)
        
        # Simulate arrival of packet in queue
        if next_event.action == 'packet_arrival':
            
            # Calculate departure time for packet
            serving_time = next_event.packet.size / serving_bitrate
            
            new_departure_time = curr_time + serving_time
            
            # Create new event for packet departure into new queue
            event_calendar.append(Event(new_departure_time, 'packet_departure', 
                                        next_event.queue, next_event.packet))            
            
            arrival_counter += 1
            
            if counter % 1000 == 0:
                print(len(event_calendar))
            
        elif next_event.action == 'packet_departure': 
            
            # Background packets are discarded from queues
            # if next_event.packet.packet_type == 'BG':
            #     pass
            
            # VR packets are send to next queue or if at last queue, to the BS
            if next_event.packet.packet_type == 'VR':              
                next_queue = next_event.queue + 1
                
                # Departure from last queue - send to BS
                if next_queue >= n_queues:
                    # Save time for correct packet ID
                    vr_timestamps_burst[next_event.packet.id] = curr_time
                    vr_packet_counter += 1
                    
                # Otherwise send to next queue in simulation 
                else: 
                    # Create new arrival event with new parameters
                    # Departure time is new arrival time 
                    new_arr_time = next_event.time
                    packet = next_event.packet
                    event_calendar.append(Event(new_arr_time, 'packet_arrival', 
                                                next_queue, packet))          
        
        else: 
            print('Warning: Unknown event in the calendar!' + \
                  '\nPlease check the proper initialization of events!')
            raise SystemExit
        
        counter += 1
        
    output_save_path = file_folder # + file_name + "\\"
    output_file_name = f'{file_to_simulate.strip(".csv")}_burst.csv'

    full_file_name = os.path.join(output_save_path, output_file_name)
    
    np.savetxt(full_file_name, vr_timestamps_burst, delimiter=",")# , fmt='%s')
        
    print(vr_timestamps_burst[0:50])
    
    tic = time.perf_counter()    
    print(f"Finished Event Simulation: {toc-tic:0.4f} seconds")

    print("VR packets:", vr_packet_counter)
    print("Arrivals:", arrival_counter)


if __name__ == "__main__":

    main(serving_bitrate=50000, n_queues=5, max_packet_size=1500, max_time=0.01,
         exp_size=1000, exp_time=0.005, sim_time=10)





