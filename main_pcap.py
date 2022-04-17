"""PCAP reader for traces that contain RTP streams."""

from argparse import Namespace
from typing import Generator
import random
import os

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


def main(args: Namespace, ue, seed) -> None:
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
    queue_params = args.queue
    
    # E2E Lat: 25/50/100 ms
    e2e_lat = args.e2e
    
    # Bitrate: E.g. 100  
    bitrate = args.bitrate
    
    #Burstiness: E.g. 0.6
    burst = args.burst
    
    og_pcap = os.getcwd() + f"\\PSNR\\PCAP_FILES\\input_APP{bitrate}_16s.pcap"
    # pcap_file = open(og_pcap, 'rb') 
    # pcap_file = open(args.pcap, 'rb')
    # pcap_reader = pcap.Reader(pcap_file)
    
    print(sim_params, "\n", seed, "\n", queue_params, "\n", e2e_lat, "\n", 
          bitrate, "\n", burst)
    queues = f"SEED1 - {queue_params} Load"
        
    stats_path = os.getcwd() + "\\Stats\\New_Offset\\New Sensitivity\\PCAP\\"
    if ue <= 4:
        stats_folder = stats_path + sim_params + f"\\SEED{seed}_omni\\" + \
                       queues + f"\\trace_name}\\"
    else: 
        stats_folder = stats_path + sim_parameters + f"\\SEED{seed}_omni_8\\" + \
                       queues + f"\\{trace_name}\\"
                       
    # sim_output_trace = 
        
    raise SystemExit()
    

    output_pcap_file = open(args.output, 'wb')
    output_pcap_writer = pcap.Writer(output_pcap_file)

    # Keep track of frames
    frame_nr: int = -1
    frame_rtp_timestamp: int = None

    # _log_header()
    for packet in _read_packets(pcap_reader):
        # Increase frame number
        if frame_rtp_timestamp != packet.rtp_timestamp:
            frame_nr += 1
            frame_rtp_timestamp = packet.rtp_timestamp
            
            
        
        # if frame_nr < 480: # Save only 60 seconds of video (for 30 FPS)
        #     _log_packet(packet, frame_nr)
        
        # if packet is not dropped: keep packets!!!
        
        
        
        
        
        
            output_pcap_writer.writepkt(packet.buffer, packet.timestamp)

            # TODO: Add code here to decide whether to keep this packet or discard it..
            # For now, only use to convert pcap trace to csv
            # drop = random.random()
            # print(drop)
            #if drop > 0.05: 
  
    pcap_file.close()
    output_pcap_file.close()
