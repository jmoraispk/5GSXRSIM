"""PCAP reader for traces that contain RTP streams."""

from argparse import Namespace
from typing import Generator
import random

from dpkt import pcap

from pcap_traces.packet import Packet

KEY_FRAME_INTERVAL = 10  # Note: Make sure this matches the RTP settings!!


def _read_packets(pcap_reader: pcap.Reader) -> Generator[Packet, None, None]:
    for timestamp, buffer in pcap_reader:
        packet = Packet(timestamp, buffer)
        yield packet


def _log_header() -> None:
    print("time,size,rtptime,frame,frametype")


def _log_packet(packet: Packet, frame_nr: int) -> None:
    # Determine frame type
    frame_type = 'I' if frame_nr % KEY_FRAME_INTERVAL == 0 else 'P'

    print(f"{packet.timestamp:.5f},{packet.size},{packet.rtp_timestamp},{frame_nr},{frame_type}")


def main(args: Namespace) -> None:
    """Read and process the PCAP file.

    Parameters
    ----------
    args : Namespace
        Command line arguments
    """

    pcap_file = open(args.pcap, 'rb')
    pcap_reader = pcap.Reader(pcap_file)

    output_pcap_file = open(args.output, 'wb')
    output_pcap_writer = pcap.Writer(output_pcap_file)

    # Keep track of frames
    frame_nr: int = -1
    frame_rtp_timestamp: int = None

    x = 0
    
    if(args.verbose):
        _log_header()
    for packet in _read_packets(pcap_reader):
        # Increase frame number
        if frame_rtp_timestamp != packet.rtp_timestamp:
            frame_nr += 1
            frame_rtp_timestamp = packet.rtp_timestamp
        if(args.verbose):
            _log_packet(packet, frame_nr)

        # Add code here to decide whether to keep this packet or discard it..
        drop = random.random()
        # print(drop)
        if drop > 0.005:
            output_pcap_writer.writepkt(packet.buffer, packet.timestamp)

    pcap_file.close()
    output_pcap_file.close()
