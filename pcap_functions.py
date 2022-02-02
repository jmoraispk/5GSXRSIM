# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 15:24:25 2022

@author: duzhe
"""

import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import time

pd.options.mode.chained_assignment = None  # default='warn'

"""
Scripts to use with pcap traces from output/intput of simulator

- Functions using pcap traces (.csv) 
  - (Adjust) Packet drop rates -> by APP in UE (packets exceeding frame time)
  - Packet dispersions after going through BS scheduling
  - Plots, e.g. PDR, per frame, I- vs. P-frame, Packet Latencies, ...
  - ...
  
- Modifying video traces with simulation data (.pcap) 
  - Apply pdr statistics (either from RAN or APP on UE) on og.pcap traces
  - Use new trace to create video file (pcap -> YUV -> mp4 => PSNR)

- Using tools, e.g. ffmpeg, for analyzing simulations (e.g. PSNR, etc...) 
  (Do here or in separate script ran using cmd?)


"""
# Files and folders of input data

queue_parameters = "SEED1 - 10Q - 70.0% Load" 

trace_path = os.getcwd() + "\\PCAP\\Traces\\" # OG traces
trace_name = "trace_APP100_0.6"
trace_folder = trace_path + f'\\{trace_name}\\'
input_trace = trace_folder + queue_parameters + "\\" + \
              f"{trace_name}_0.0-16.0s.csv"
              
sim_trace = pd.read_csv(input_trace, encoding='utf-16-LE') 

# Files and folders of simulation output
stats_path = os.getcwd() + "\\Stats\\Queue_Sim\\PCAP\\"

sim_parameters = 'SEED1_omni_BW-150_RAN-LAT-60_LEN-16.0s_M-LWDF_PCAP-True_Offset-1.0'
stats_folder = stats_path + sim_parameters + "\\" + queue_parameters + "\\"


n_ues = 4
lat_E2E = 100 / 1000 # ms

output_trace = [0] * n_ues 
for ue in range(n_ues):
    output_trace_ue = stats_folder + f"{trace_name} - 16.0s_UE{ue}.csv"
    output_trace[ue] = pd.read_csv(output_trace_ue, encoding='utf-16-LE',
                                   index_col=0)

# print("Queue:", queue_parameters)
print("SXR:", sim_parameters)

pdr_RAN = []
pdr_E2E = []
pdr_Total = []

for ue in range (n_ues):
    
    ue_offset = ue * (1 / (30 * 4))
    # Do not forget to take into account the offset for different users!!! 
    dropped_RAN = len(output_trace[ue][(output_trace[ue]["arr_time"] == 0)])

    dropped_E2E = len(output_trace[ue][(
        output_trace[ue]["arr_time"] > (output_trace[ue]["frame"]*(1/30) + \
                                        lat_E2E + ue_offset))]) 

    dropped_total = dropped_RAN + dropped_E2E
    pdr_RAN.append(round(dropped_RAN / len(output_trace[ue]) * 100, 3))  
    pdr_E2E.append(round(dropped_E2E / len(output_trace[ue]) * 100, 3))  
    pdr_Total.append(round(dropped_total / len(output_trace[ue]) * 100, 3))  

# pdr_ues = [100 * i for i in pdr_ues]

print(f"\nE2E latency budget: {int(lat_E2E*1000)}ms")
print(f"Mean RAN PDR: {round(np.mean(pdr_RAN), 2)}%")
print(f"Mean E2E PDR: {round(np.mean(pdr_E2E), 2)}%")
print(f"Total PDR per UE: {pdr_Total}%")    
print(f"Total Mean PDR: {round(np.mean(pdr_Total), 2)}")

    


# %% Adjust output files 
# Mimicking APP on UE -> packets arriving too late will be dropped 
# (count as dropped even though successfully scheduled and transmitted by RAN)




# %% Plot functions for output pcap traces 