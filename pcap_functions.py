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
import warnings

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")
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
# trace_path = os.getcwd() + "\\PCAP\\Traces\\" # OG traces
# trace_folder = trace_path + f'\\{trace_name}\\'
# input_trace = trace_folder + queue_parameters + "\\" + \
#               f"{trace_name}_0.0-16.0s.csv"              
# sim_trace = pd.read_csv(input_trace, encoding='utf-16-LE') 

# Files and folders of input data
tic = time.perf_counter()

# Files and folders of simulation output
stats_path = os.getcwd() + "\\Stats\\Queue_Sim\\PCAP\\"
stats_path = os.getcwd() + "\\Stats\\Queue_Sim\\Tune-RAN\\PCAP\\"
output_path = os.getcwd() + "\\PDR\\Tune-RAN\\" 


seeds = range(1,21)
n_ues = 4 
print("Seeds:", seeds)

lat_E2E = 100 / 1000 # ms # DO NOT FORGET TO CHECK THIS VALUE!!!!!
print(f"E2E latency budget: {int(lat_E2E*1000)}ms\n")

queue_parameters = "SEED1 - 10Q - 70.0% Load" 
trace_name = "trace_APP100_0.6"

# TODO: parameters
sim_parameters = 'BW-125_RAN-LAT-70_LEN-16.0s_Frametype-M-LWDF-4.0_Offset-1.0'
print("SXR:", sim_parameters)
    
# Sim parameters to use for naming output stat file
E2E_budget = f"E2E-{int(lat_E2E * 1000)}ms" 
bw = sim_parameters.split("_")[0]
lat = sim_parameters.split("_")[1]
scheduler = sim_parameters.split("_")[3]
sync_offset = sim_parameters.split("_")[4]

mean_pdr_RAN, mean_pdr_RAN_I, mean_pdr_RAN_P = [[] for i in range(3)]
mean_pdr_E2E, mean_pdr_E2E_I, mean_pdr_E2E_P = [[] for i in range(3)]
mean_pdr_Total, mean_pdr_Total_I, mean_pdr_Total_P = [[] for i in range(3)]

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
    
    pdr_RAN, pdr_RAN_I, pdr_RAN_P = [[] for i in range(3)]
    pdr_E2E, pdr_E2E_I, pdr_E2E_P = [[] for i in range(3)]
    pdr_Total, pdr_Total_I, pdr_Total_P = [[] for i in range(3)]
    
    offset = float(sim_parameters.split("-")[-1]) # sim_parameters
    
    for ue in range (n_ues):
        # Cut off packets until end of simulation time 
        # output_trace[ue] = output_trace[ue][output_trace[ue]["arr_time"] <= 16]
        
        # Do not forget to take into account the offset for different users!!! 
        ue_offset = ue * (1 / (30 * 4)) * offset
        
############################################################################### 
        # Sort PDR by I-frames and P-frames       
        output_trace_I = output_trace[ue][output_trace[ue]["frametype"] == True]
        output_trace_P = output_trace[ue][output_trace[ue]["frametype"] == False] 
                        
        dropped_RAN = len(output_trace[ue][(output_trace[ue]["arr_time"] == 0)])
        dropped_RAN_I = len(output_trace_I[(output_trace_I["arr_time"] == 0)])
        dropped_RAN_P = len(output_trace_P[(output_trace_P["arr_time"] == 0)])

        # output_trace[ue] = output_trace[ue][output_trace[ue]["success"] == True]
        # E2E_delay = output_trace[ue]["arr_time"] - ue_offset - \
        #                 output_trace[ue]["frame"]*(1/30)
        # dropped_E2E = E2E_delay
        # if ue >= 3: raise SystemExit
        
        dropped_E2E = len(output_trace[ue][(
            output_trace[ue]["arr_time"] > (output_trace[ue]["frame"]*(1/30) + \
                                            lat_E2E + ue_offset))]) 
        dropped_E2E_I = len(output_trace_I[(
            output_trace_I["arr_time"] > (output_trace_I["frame"]*(1/30) + \
                                          lat_E2E + ue_offset))])
        dropped_E2E_P = len(output_trace_P[(
            output_trace_P["arr_time"] > (output_trace_P["frame"]*(1/30) + \
                                          lat_E2E + ue_offset))])  
             
        dropped_total = dropped_RAN + dropped_E2E
        dropped_total_I = dropped_RAN_I + dropped_E2E_I
        dropped_total_P = dropped_RAN_P + dropped_E2E_P      
###############################################################################

###############################################################################        
        pdr_RAN.append(round(dropped_RAN / len(output_trace[ue]) * 100, 3))
        pdr_RAN_I.append(round(dropped_RAN_I / len(output_trace_I) * 100, 3))  
        pdr_RAN_P.append(round(dropped_RAN_P / len(output_trace_P) * 100, 3))  

        pdr_E2E.append(round(dropped_E2E / len(output_trace[ue]) * 100, 3))
        pdr_E2E_I.append(round(dropped_E2E_I / len(output_trace_I) * 100, 3))
        pdr_E2E_P.append(round(dropped_E2E_P / len(output_trace_P) * 100, 3))

        pdr_Total.append(round(dropped_total / len(output_trace[ue]) * 100, 3)) 
        pdr_Total_I.append(round(dropped_total_I / len(output_trace_I) * 100, 3))
        pdr_Total_P.append(round(dropped_total_P / len(output_trace_P) * 100, 3))
###############################################################################

###############################################################################    
    mean_seed_pdr_RAN = round(np.mean(pdr_RAN), 3)
    mean_seed_pdr_RAN_I = round(np.mean(pdr_RAN_I), 3)
    mean_seed_pdr_RAN_P = round(np.mean(pdr_RAN_P), 3)

    mean_seed_pdr_E2E = round(np.mean(pdr_E2E), 3)
    mean_seed_pdr_E2E_I = round(np.mean(pdr_E2E_I), 3)
    mean_seed_pdr_E2E_P = round(np.mean(pdr_E2E_P), 3)

    mean_seed_pdr_Total = round(np.mean(pdr_Total), 3)
    mean_seed_pdr_Total_I = round(np.mean(pdr_Total_I), 3)
    mean_seed_pdr_Total_P = round(np.mean(pdr_Total_P), 3)
###############################################################################

###############################################################################
    mean_pdr_RAN.append(mean_seed_pdr_RAN)
    mean_pdr_RAN_I.append(mean_seed_pdr_RAN_I)
    mean_pdr_RAN_P.append(mean_seed_pdr_RAN_P)

    mean_pdr_E2E.append(mean_seed_pdr_E2E)
    mean_pdr_E2E_I.append(mean_seed_pdr_E2E_I)
    mean_pdr_E2E_P.append(mean_seed_pdr_E2E_P)

    mean_pdr_Total.append(mean_seed_pdr_Total)
    mean_pdr_Total_I.append(mean_seed_pdr_Total_I)
    mean_pdr_Total_P.append(mean_seed_pdr_Total_P)
###############################################################################
    
    # print(f"Total RAN PDR per UE: {pdr_RAN}%")    
    # print(f"Total E2E PDR per UE: {pdr_Total}%")    
    
    # print(f"Mean RAN PDR:    {mean_seed_pdr_RAN}%")
    # print(f"Mean E2E PDR:    {mean_seed_pdr_E2E}%")
    # print(f"Mean Total PDR:  {mean_seed_pdr_Total}")
    
# Calculate confidence intervals! 
z_value = 2.093 # 1.96 # 95% Confidence interval

n_size = len(seeds)
###############################################################################
total_mean_RAN = round(np.mean(mean_pdr_RAN), 3)
total_mean_RAN_I = round(np.mean(mean_pdr_RAN_I), 3)
total_mean_RAN_P = round(np.mean(mean_pdr_RAN_P), 3)

total_mean_E2E = round(np.mean(mean_pdr_E2E), 3)
total_mean_E2E_I = round(np.mean(mean_pdr_E2E_I), 3)
total_mean_E2E_P = round(np.mean(mean_pdr_E2E_P), 3)

total_mean_Total = round(np.mean(mean_pdr_Total), 3)
total_mean_Total_I = round(np.mean(mean_pdr_Total_I), 3)
total_mean_Total_P = round(np.mean(mean_pdr_Total_P), 3)
###############################################################################

###############################################################################
total_std_RAN = np.std(mean_pdr_RAN)
total_std_RAN_I = np.std(mean_pdr_RAN_I)
total_std_RAN_P = np.std(mean_pdr_RAN_P)

total_std_E2E = np.std(mean_pdr_E2E)
total_std_E2E_I = np.std(mean_pdr_E2E_I)
total_std_E2E_P = np.std(mean_pdr_E2E_P)

total_std_Total = np.std(mean_pdr_Total)
total_std_Total_I = np.std(mean_pdr_Total_I)
total_std_Total_P = np.std(mean_pdr_Total_P)
###############################################################################

###############################################################################
deviation_RAN = round(z_value * (total_std_RAN / n_size), 3) 
deviation_RAN_I = round(z_value * (total_std_RAN_I / n_size), 3) 
deviation_RAN_P = round(z_value * (total_std_RAN_P / n_size), 3) 

deviation_E2E = round(z_value * (total_std_E2E / n_size), 3) 
deviation_E2E_I = round(z_value * (total_std_E2E_I / n_size), 3) 
deviation_E2E_P = round(z_value * (total_std_E2E_P / n_size), 3) 

deviation_Total = round(z_value * (total_std_Total / n_size), 3) 
deviation_Total_I = round(z_value * (total_std_Total_I / n_size), 3) 
deviation_Total_P = round(z_value * (total_std_Total_P / n_size), 3) 
###############################################################################

###############################################################################
conf_int_RAN = [total_mean_RAN, deviation_RAN, 
                round(100 * deviation_RAN / total_mean_RAN, 3)]
conf_int_RAN_I = [total_mean_RAN_I, deviation_RAN_I, 
                round(100 * deviation_RAN_I / total_mean_RAN_I, 3)]
conf_int_RAN_P = [total_mean_RAN_P, deviation_RAN_P, 
                round(100 * deviation_RAN_P / total_mean_RAN_P, 3)]

conf_int_E2E = [total_mean_E2E, deviation_E2E, 
                round(100 * deviation_E2E / total_mean_E2E, 3)]
conf_int_E2E_I = [total_mean_E2E_I, deviation_E2E_I, 
                round(100 * deviation_E2E_I / total_mean_E2E_I, 3)]
conf_int_E2E_P = [total_mean_E2E_P, deviation_E2E_P, 
                round(100 * deviation_E2E_P / total_mean_E2E_P, 3)]

conf_int_Total = [total_mean_Total, deviation_Total, 
                  round(100 * deviation_Total / total_mean_Total, 3)]
conf_int_Total_I = [total_mean_Total_I, deviation_Total_I, 
                  round(100 * deviation_Total_I / total_mean_Total_I, 3)]
conf_int_Total_P = [total_mean_Total_P, deviation_Total_P, 
                  round(100 * deviation_Total_P / total_mean_Total_P, 3)]
###############################################################################

###############################################################################

# print("95% Confidence Intervals for PDR [%]\n")

print(f"RAN:   {conf_int_RAN}")
# print(f"I:     {conf_int_RAN_I}")
# print(f"P:     {conf_int_RAN_P}\n")

print(f"E2E:   {conf_int_E2E}")
# print(f"I:     {conf_int_E2E_I}")
# print(f"P:     {conf_int_E2E_P}\n")

print(f"Total: {conf_int_Total}")
# print(f"I:     {conf_int_Total_I}")
# print(f"P:     {conf_int_Total_P}\n")

###############################################################################
toc = time.perf_counter()
print(f"\nTime: {toc-tic:0.4f} seconds.")

save_stats = 1

if save_stats: 
    
    # pdr_type = ["RAN-All", "RAN-I", "RAN-P", 
    #             "E2E-All", "E2E-I", "E2E-P",
    #             "Total-All", "Total-I", "Total-P"] 
    # pdr_stats = [conf_int_RAN[0], conf_int_RAN_I[0], conf_int_RAN_P[0],
    #              conf_int_E2E[0], conf_int_E2E_I[0], conf_int_E2E_P[0],
    #              conf_int_Total[0], conf_int_Total_I[0], conf_int_Total_P[0]] 
    # conf_int = [conf_int_RAN[1], conf_int_RAN_I[1], conf_int_RAN_P[1],
    #             conf_int_E2E[1], conf_int_E2E_I[1], conf_int_E2E_P[1],
    #             conf_int_Total[1], conf_int_Total_I[1], conf_int_Total_P[1]] 
         
    # # dictionary of lists  
    # dict_1 = {'PDR-Type': pdr_type, 'PDR': pdr_stats, 'Conf-Int': conf_int}  
    
    dict_2 = {'RAN-All': conf_int_RAN, 
              'RAN-I': conf_int_RAN_I, 
              'RAN-P': conf_int_RAN_P, 
              'E2E-All': conf_int_E2E, 
              'E2E-I': conf_int_E2E_I, 
              'E2E-P': conf_int_E2E_P, 
              'Total-All': conf_int_Total, 
              'Total-I': conf_int_Total_I, 
              'Total-P': conf_int_Total_P}
       
    # pdr_df = pd.DataFrame(dict_1)      
    pdr_df = pd.DataFrame(dict_2) 
    
    output_folder = output_path + f"{E2E_budget}" + "\\" + \
            f"{trace_name.strip('trace_')}_{bw}_{lat}_{sync_offset}" + "\\" + \
            f"{queue_parameters.split('-')[1].strip()} - " + \
            f"{queue_parameters.split('-')[2].strip()}" 
    output_file = f"{scheduler}.csv"
      
    os.makedirs(output_folder, exist_ok=True)
    output_full_name = os.path.join(output_folder, output_file)     
    # saving the dataframe 
    pdr_df.to_csv(output_full_name, encoding='utf-8', index=False) 
    # print(f"Saved:\n{trace_name}\n{queue_parameters}\n{sim_parameters}")    
    print(f"Saved:\n{output_folder.strip(output_path)}")    


# Save all parameters in file name
# Save all PDR metrics
# Use one line per type 
# RAN: (PDR) total - I-frame - P-frame -- (error bar) total - I-frame - P-frame 
# E2E: (PDR) total - I-frame - P-frame -- (error bar) total - I-frame - P-frame   
# Total: (PDR) total - I-frame - P-frame -- (error bar) total - I-frame - P-frame   
  


# %% Adjust output files 
# Mimicking APP on UE -> packets arriving too late will be dropped 
# (count as dropped even though successfully scheduled and transmitted by RAN)




# %% Plot functions for output pcap traces 