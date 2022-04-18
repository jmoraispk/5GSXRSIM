pcap_traces - main: 
input arguments: FOR PROCESS_VIDEO.PY


(base folder with results: Stats/New_Offset/New Sensitivity/PCAP/


add: (Var to create path to sim - output

BW: int
LAT: RAN or E2E
lat: int
Scheduler: EDD/EDD-Frametype/M-LWDF/M-LWDF-Frametype
Offset: 1.0
UEs: int (1-8)

SEED (max. up to): int (e.g.20) => change omni to omni_8 if UEs >4

SEED1: - 

#Queues: 5/10/15
Load: 50/70/85

Bitrate/Burstiness: trace_APP{bitrate}_{burstiness}

=> For ue in range #ues: trace_APP{bitrate}_{burstiness} - 16.0s_UE{ue}.csv

=>> Input trace: input_APP{bitrate}_16s.pcap!!!

=>> Reference video: input_APP{bitrate}_16s.mp4!!!

REMOVE OUTPUT FROM PCAP-TRACE MAIN (NOT NEEDED ANYMORE)


