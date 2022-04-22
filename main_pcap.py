"""PCAP reader for traces that contain RTP streams."""

from argparse import Namespace
from typing import Generator
import random
import os
import pandas as pd
import numpy as np

from dpkt import pcap

from packet import Packet

KEY_FRAME_INTERVAL = 10  # Note: Make sure this matches the RTP settings!!


def _read_packets(pcap_reader: pcap.Reader) -> Generator[Packet, None, None]:
    for timestamp, buffer in pcap_reader:
        packet = Packet(timestamp, buffer)
        yield packet


def _log_header() -> None:
    print("time,size,rtptime,frame,frametype")


def _log_packet(packet: Packet, frame_nr: int) -> None:
    # Determine frame type
    frame_type = True if frame_nr % KEY_FRAME_INTERVAL == 0 else False

    print(f"{packet.timestamp:.5f},{packet.size},{packet.rtp_timestamp},{frame_nr},{frame_type}")


def main(args: Namespace, ue, n_ues, seed) -> None:
    """Read and process the PCAP file.

    Parameters
    ----------
    args : Namespace
        Command line arguments
    """
    
    # E.g.: 'BW-150_E2E-LAT-50_LEN-16.0s_EDD_Offset-1.0_UE4'
    sim_params = args.params        
    # Up to 20 Seeds
    seed = seed    
    # E.g. '10Q - 70.0%'
    queues = f"SEED1 - {args.queue} Load" # = args.queue    
    # E2E Lat: 25/50/100 ms
    e2e_lat = args.e2e / 1000    
    # Bitrate: E.g. 100  
    bitrate = args.bitrate    
    #Burstiness: E.g. 0.6
    burst = args.burst
    
    og_pcap = os.getcwd() + f"\\PSNR\\PCAP_FILES\\input_APP{bitrate}_16s.pcap"
    pcap_file = open(og_pcap, 'rb') 
    # pcap_file = open(args.pcap, 'rb')
    pcap_reader = pcap.Reader(pcap_file)
    
        
    stats_path = os.getcwd() + "\\Stats\\New_Offset\\New Sensitivity\\PCAP\\"
    if n_ues <= 4:
        stats_folder = stats_path + sim_params + f"\\SEED{seed}_omni\\" + \
                       queues + f"\\trace_APP{bitrate}_{burst}\\"
    else: 
        stats_folder = stats_path + sim_params + f"\\SEED{seed}_omni_8\\" + \
                       queues + f"\\trace_APP{bitrate}_{burst}\\"
                       
                       
    trace_file = stats_folder + f"trace_APP{bitrate}_{burst} - 16.0s_UE{ue}.csv"
    output_trace = pd.read_csv(trace_file, encoding='utf-8', index_col=0)
    
    # Account for UE Offset
    offset = float(sim_params.split("_")[4].strip("Offset-"))    
    ue_offset = ue * (10 / (30 * n_ues)) * offset
    
    # Take all packets that arrived within E2E Latency    
    success_E2E_idx = output_trace[(output_trace["arr_time"] < (
        output_trace["frame"]*(1/30) + e2e_lat + ue_offset))]
    
    # Only take indices of packets also sent within RAN latency
    success_total = success_E2E_idx[
        success_E2E_idx["arr_time"] > 0.0]["index"].to_numpy() 
    
    pdr_idx = np.zeros(success_total[-1] + 1)
    
    for i, idx in enumerate(success_total):
        pdr_idx[idx] = 1
    
    pdr_file = round(100*len(pdr_idx[pdr_idx == 0])/len(pdr_idx), 4)
    
    # print("PDR:", pdr_file, "%")    

    output_pcap_file = open(args.output, 'wb')
    output_pcap_writer = pcap.Writer(output_pcap_file)

    # Keep track of frames
    frame_nr: int = -1
    frame_rtp_timestamp: int = None

    # _log_header()
    packet_idx = 0
    for packet in _read_packets(pcap_reader):
    # Increase frame number
        if frame_rtp_timestamp != packet.rtp_timestamp:
            frame_nr += 1
            frame_rtp_timestamp = packet.rtp_timestamp        
        
        # Check for last package    
        if packet_idx < len(pdr_idx):    
            if pdr_idx[packet_idx] == 1: 
            # Save successful packets within E2E latency
                output_pcap_writer.writepkt(packet.buffer, packet.timestamp)
            
        packet_idx += 1
                
        
    pcap_file.close()
    output_pcap_file.close()
    
    return pdr_file
    # raise SystemExit()
