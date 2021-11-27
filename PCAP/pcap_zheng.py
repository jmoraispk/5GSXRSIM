# -*- coding: utf-8 -*-
"""PCAP reader for traces that contain RTP streams."""

from typing import Generator

import csv_script

import numpy as np

import os
import sys

# import dpkt
from dpkt import pcap

from packet import Packet

# TODO: GoP Size
KEY_FRAME_INTERVAL = 10  # Note: Make sure this matches the RTP settings!!



def _read_packets(pcap_input_file: pcap.Reader) -> Generator[Packet, None, None]:
    for timestamp, buffer in pcap_input_file:
        packet = Packet(timestamp, buffer)
        yield packet


def _log_header() -> None:
    print("time,size,rtptime,frame,frametype")


def _log_packet(packet: Packet, frame_nr: int) -> None:
    # Determine frame type
    frame_type = True if frame_nr % KEY_FRAME_INTERVAL == 0 else False 
    
    # For now just return the packet size
    return packet.size
    # print(f"{packet.timestamp:.5f},{packet.size},{packet.rtp_timestamp},{frame_nr},{frame_type}")
    # header = ['Timestamp', 'Size', 'RTP Timestamp', 'Frame', 'Frametype']
    # data = [packet.timestamp, packet.size, frame_nr, frame_type]
    # with open('output.csv', 'w', encoding='UTF8', newline='') as f:
        
    #     writer = csv.writer(f)
    
    #     # write the header
    #     writer.writerow(header)
    
    #     # write the data
    #     writer.writerow(data)
    

def main(drop_rate_file, filename, offset): # drop_rate, 
    """Read and process the PCAP file.

    Parameters
    ----------
    args : Namespace
        Command line arguments
    """
    # Set input file name
    input_pcap_name='input_FHD.pcap'  
    
    pcap_input_file = open(input_pcap_name, 'rb')
    pcap_reader = pcap.Reader(pcap_input_file)
    
    if offset:
        output_save_path = 'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\PCAP_output\Offset'
        
    else: 
        output_save_path = 'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\PCAP_output'
    
    output_file_name = f'output_{filename.strip(".csv")}.pcap'
    output_file_name = os.path.join(output_save_path, output_file_name)    
        
    # drop_rate = drop_rate # in percent 
    # output_pcap_file = open(f'output_{drop_rate_file.strip(".csv")}.pcap', 'wb')
    output_pcap_file = open(output_file_name, 'wb')
    output_pcap_writer = pcap.Writer(output_pcap_file)
    
    # Keep track of frames
    frame_nr: int = -1
    frame_rtp_timestamp: int = None
    
    # _log_header()
    
    # List of dropped and successful packets
    dropped = np.genfromtxt(drop_rate_file,delimiter=',',dtype=int)
    
    total_packets = 0
    succ_packets = 0
    dropped_packets = 0
    
    total_bits = 0
    succ_bits = 0       
    dropped_bits = 0       
       
    for packet in _read_packets(pcap_reader):
        # Increase frame number
        if frame_rtp_timestamp != packet.rtp_timestamp:
            frame_nr += 1
            frame_rtp_timestamp = packet.rtp_timestamp
        """
        TODO
        Add code here to do something
        E.g. decide whether to keep this packet or discard it...
        """
        # Successful packets
        if dropped[total_packets] != 0: 
            output_pcap_writer.writepkt(packet.buffer, packet.timestamp)
            succ_packets +=1
            succ_bits += _log_packet(packet, frame_nr)
        
        # Dropped packets
        else:    
            dropped_packets += 1    
            dropped_bits += _log_packet(packet, frame_nr)
            
        # Counting total packets    
        total_packets += 1
        total_bits += _log_packet(packet, frame_nr)
        
    """    
    # Print PDR stats    
    print(f'Total: {total_packets} packets, {round(total_bits/1000,2)} kByte.' +
          f'\nDropped: {dropped_packets} packets, {round(dropped_bits/1000,2)} kByte.' + 
          f'\nSuccessful: {succ_packets} packets, {round(succ_bits/1000,2)} kByte.')
    print(f'Packet drop rate: {round(100*dropped_packets/total_packets, 2)}%' + 
          f'\nBit drop rate: {round(100*dropped_bits / total_bits, 2)}%')
    """
    print('Output File:', output_pcap_file.name)
    # print(drop_rate_file)
    
    pcap_input_file.close()
    output_pcap_file.close()
    
    return

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#pdr = [0.1,0.5,1,2,3,4,5]

# Set if PDR should have a slight offset!!!
offset = True

if offset:
    input_path = 'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\PDR_output\Offset'
else:
    input_path = 'C:\Zheng Data\TU Delft\Thesis\Thesis Work\GitHub\SXRSIMv3\PCAP\PDR_output'
    
for i in range(len(csv_script.pdr)):  
    
    input_drop_file = f'PDR_{csv_script.pdr[i]}%_offset.csv'
    input_drop_file_name = os.path.join(input_path, input_drop_file)

    main(input_drop_file_name, input_drop_file, offset)

print('done')




