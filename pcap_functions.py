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
tic = time.perf_counter()

queue_parameters = "SEED1 - 10Q - 70.0% Load" 

trace_path = os.getcwd() + "\\PCAP\\Traces\\" # OG traces
trace_name = "trace_APP100_0.6"
trace_folder = trace_path + f'\\{trace_name}\\'
input_trace = trace_folder + queue_parameters + "\\" + \
              f"{trace_name}_0.0-16.0s.csv"
              
sim_trace = pd.read_csv(input_trace, encoding='utf-16-LE') 

# Files and folders of simulation output
stats_path = os.getcwd() + "\\Stats\\Queue_Sim\\PCAP\\"

sim_parameters = 'BW-200_RAN-LAT-50_LEN-16.0s_M-LWDF_Offset-1.0'

seeds = range(1,21)
n_ues = 4 

print("SXR:", sim_parameters)
print("Seeds:", seeds)

lat_E2E = 100 / 1000 # ms # DO NOT FORGET TO CHECK THIS VALUE!!!!!
print(f"E2E latency budget: {int(lat_E2E*1000)}ms\n")

    
mean_pdr_RAN = []
mean_pdr_E2E = []
mean_pdr_Total = []

for seed in seeds:

    stats_folder = stats_path + sim_parameters + f"\\SEED{seed}_omni\\" + \
                   queue_parameters + f"\\{trace_name}\\"
    sim_duration = sim_parameters.split("LEN-")[1].split('s')[0] 
    
    output_trace = [0] * n_ues 
    for ue in range(n_ues):
        output_trace_ue = stats_folder + f"{trace_name} - {sim_duration}s_UE{ue}.csv"
        output_trace[ue] = pd.read_csv(output_trace_ue, 
                                       encoding='utf-8', 
                                       # encoding='utf-16-LE',
                                       index_col=0)
    
    # print("Queue:", queue_parameters)
    
    pdr_RAN = []
    pdr_E2E = []
    pdr_Total = []
    offset = True # sim_parameters
    for ue in range (n_ues):
        # Cut off packets until end of simulation time 
        # output_trace[ue] = output_trace[ue][output_trace[ue]["arr_time"] <= 16]
        
        # Do not forget to take into account the offset for different users!!! 
        if offset: 
            ue_offset = ue * (1 / (30 * 4))
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO!!! - Sort PDR by I-frames and P-frames!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
        dropped_RAN = len(output_trace[ue][(output_trace[ue]["arr_time"] == 0)])
        
        # output_trace[ue] = output_trace[ue][output_trace[ue]["success"] == True]
        # E2E_delay = output_trace[ue]["arr_time"] - ue_offset - \
        #                 output_trace[ue]["frame"]*(1/30)
        # dropped_E2E = E2E_delay
        # if ue >= 3: raise SystemExit
        
        dropped_E2E = len(output_trace[ue][(
            output_trace[ue]["arr_time"] > (output_trace[ue]["frame"]*(1/30) + \
                                            lat_E2E + ue_offset))]) 
        dropped_total = dropped_RAN + dropped_E2E
        
        pdr_RAN.append(round(dropped_RAN / len(output_trace[ue]) * 100, 3))  
        pdr_E2E.append(round(dropped_E2E / len(output_trace[ue]) * 100, 3))  
        pdr_Total.append(round(dropped_total / len(output_trace[ue]) * 100, 3))  
                
    # pdr_ues = [100 * i for i in pdr_ues]
    
    mean_seed_pdr_RAN = round(np.mean(pdr_RAN), 3)
    mean_seed_pdr_E2E = round(np.mean(pdr_E2E), 3)
    mean_seed_pdr_Total = round(np.mean(pdr_Total), 3)
    mean_pdr_RAN.append(mean_seed_pdr_RAN)
    mean_pdr_E2E.append(mean_seed_pdr_E2E)
    mean_pdr_Total.append(mean_seed_pdr_Total)
    
    
    # print(f"Total RAN PDR per UE: {pdr_RAN}%")    
    # print(f"Total E2E PDR per UE: {pdr_Total}%")    
    
    # print(f"Mean RAN PDR:    {mean_seed_pdr_RAN}%")
    # print(f"Mean E2E PDR:    {mean_seed_pdr_E2E}%")
    # print(f"Mean Total PDR:  {mean_seed_pdr_Total}")

    
# Calculate confidence intervals! 
z_value = 2.093 # 1.96 # 95% Confidence interval

n_size = len(seeds)
total_mean_RAN = round(np.mean(mean_pdr_RAN), 4)
total_mean_E2E = round(np.mean(mean_pdr_E2E), 4)
total_mean_Total = round(np.mean(mean_pdr_Total), 4)

total_std_RAN = np.std(mean_pdr_RAN)
total_std_E2E = np.std(mean_pdr_E2E)
total_std_Total = np.std(mean_pdr_Total)

deviation_RAN = round(z_value * (total_std_RAN / n_size), 4) 
deviation_E2E = round(z_value * (total_std_E2E / n_size), 4) 
deviation_Total = round(z_value * (total_std_Total / n_size), 4) 

conf_int_RAN = [total_mean_RAN, deviation_RAN, 
                round(100 * deviation_RAN / total_mean_RAN, 4)]

conf_int_E2E = [total_mean_E2E, deviation_E2E, 
                round(100 * deviation_E2E / total_mean_E2E, 4)]

conf_int_Total = [total_mean_Total, deviation_Total, 
                  round(100 * deviation_Total / total_mean_Total, 4)]

print("95% Confidence Intervals for PDR [%]")
print(f"RAN:   {conf_int_RAN}")
print(f"E2E:   {conf_int_E2E}")
print(f"Total: {conf_int_Total}")
toc = time.perf_counter()
print(f"\nTime: {toc-tic:0.4f} seconds.")
# %% Adjust output files 
# Mimicking APP on UE -> packets arriving too late will be dropped 
# (count as dropped even though successfully scheduled and transmitted by RAN)




# %% Plot functions for output pcap traces 