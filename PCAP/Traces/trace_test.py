# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:55:42 2021

@author: duzhe
"""
import numpy as np
import pandas as pd
import csv


# test parsing csv file
class PCAP_File:
    def __init__(self, csv_trace_file):
        # Create Object for the PCAP file which stores information for the 
        # whole pcap trace, e.g. packet sizes, timestamps etc. 
        
        # Create list, with each entry containing stats of one packet
        # Create packet sequences with information from accessing this 
        # object here
        self.packet_index = []
        self.packet_time = []
        self.packet_size = []
        self.packet_rtp = []
        self.packet_frame = []
        self.packet_type = []
        
        self.parse_file(csv_trace_file) 
        
        self.adjust_packet_time()
        
    def parse_file(self, csv_trace_file):
        
        with open(csv_trace_file, "r", encoding='utf-16-le') as file: 
            lines = file.readlines()
            for i, line in enumerate(lines):
                if i == 0: continue
                index = i - 1 
                element = line.split(',')
                time = float(element[0].strip('"')) # in seconds
                size = int(element[1]) # in bytes
                rtp_time = int(element[2]) # integer
                frame = int(element[3]) # Index of frame
                frametype = element[4].strip('"\n') # I or P frame
                # is every information needed for a new packet??                
                self.packet_index.append(index)
                self.packet_time.append(time)
                self.packet_size.append(size)
                self.packet_rtp.append(rtp_time)
                self.packet_frame.append(frame)
                self.packet_type.append(frametype)                
    
    def adjust_packet_time(self):
        # Adjust each packet's timestamp, so it starts at 0 seconds
        start_time = self.packet_time[0]
        for i in range(len(self.packet_index)):            
            self.packet_time[i] = self.packet_time[i] - start_time

input_file = 'trace_FHD.csv'                
new_pcap_file = PCAP_File(input_file)

total_size = 0
packet_sizes = []
frames = []
for i in range(100):
    #total_size = total_size + float(new_pcap_file.packet_stats[i][1])

    if new_pcap_file.packet_frame[i] != new_pcap_file.packet_frame[i + 1]:
        frames.append(new_pcap_file.packet_frame[i])
        #packet_sizes.append([float(new_pcap_file.packet_stats[i][1])])
        #print(f'Index {new_pcap_file.packet_index[i]}: ' + \
        #      f'Packet Time: {new_pcap_file.packet_time[i]} ')    
        
    print(new_pcap_file.packet_index[i])
