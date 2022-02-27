# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:43:05 2022

@author: duzhe
"""

import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import time
from itertools import combinations

pd.options.mode.chained_assignment = None  # default='warn'


"""
Scripts to use with statistics of pcap traces from output of simulator

- Use script to automate plots with PDR statistics saved as csv/txt/pd as input

"""

# %% Parameters - Utils
stats_path = os.getcwd() + "\\PDR\\" 

save_plots_path = os.getcwd() + "\\Zheng - Plots\\NEW\\" 

# Parameters for plots: 
# First entry [0] is always the default value 
    
ues = [4, 1, 2, 4, 6, 8]

e2e_lat = [100, 25, 50, 100]# (25) - 50 - 100 [ms] 

bitrate = [100, 50, 100, 150, 200] # 50 - 100 - 150 - 200 [Mbps]

dispersion = [0.6, 0.99] # 0.6 - 0.99 -> Percent of Interframe time 

BW = [125, 100, 125, 150, 200] # (75) - 100 - 125 - 150 - 200 [MHz]

# IF RAN-Scheduler: 10 - 20 - 30 - 40 - 50 - 60 - 70 - 80 - 90 - 100 [ms]
# IF E2E-Scheduler: SAME AS E2E_budget!
ran_lat = [70, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 

offset = [1.0, 0.0] # 0.0 - 1.0 (sync vs max. async)

queues = [10, 5, 10, 15] # 5 - 10 - 15

bg_load = [70.0, 50.0, 70.0, 85.0] # 50.0 - 70.0 - 85.0 [%]

# Schedulers 

ran_schedulers = ["EDD", "M-LWDF", "Frametype-EDD", "Frametype-M-LWDF"]   
ran_schedulers = ["EDD", "M-LWDF"] 

ran_labels = ["EDD (RAN)", "M-LWDF (RAN)", "Frametype-EDD (RAN)", 
              "Frametype-M-LWDF (RAN)"]   

e2e_schedulers = ["EDD", "M-LWDF", "Frametype-EDD", "Frametype-M-LWDF"]   
e2e_schedulers = ["EDD", "M-LWDF"] 

e2e_labels = ["EDD (E2E)", "M-LWDF (E2E)", "Frametype-EDD (E2E)", 
              "Frametype-M-LWDF (E2E)"]   

# 7 colors x 2 (All and I-frames)
colors_schedulers_all = [
        'maroon', 'maroon', 'forestgreen', 'forestgreen', 
        'goldenrod', 'goldenrod', 'steelblue', 'steelblue', 
        'midnightblue', 'midnightblue', 'orchid', 'orchid', 'teal', 'teal']
colors_schedulers_ran = [
        'maroon', 'maroon', 'forestgreen', 'forestgreen', 
        'goldenrod', 'goldenrod', 'steelblue', 'steelblue']
colors_schedulers_e2e = [
        'midnightblue', 'midnightblue', 'orchid', 'orchid', 'teal', 'teal']

schedulers_all = ["PF", "PF I-F", 
                  "(RAN) M-LWDF", "(RAN) M-LWDF I-F", 
                  "(RAN) Frametype", "(RAN) Frametype I-F",
                  "(RAN) EDD", "(RAN) EDD I-F",
                  "(E2E) M-LWDF", "(E2E) M-LWDF I-F", 
                  "(E2E) Frametype", "(E2E) Frametype I-F",
                  "(E2E) EDD", "(E2E) EDD I-F",]

schedulers_ran = ["PF", "PF I-F", 
                  "(RAN) M-LWDF", "(RAN) M-LWDF I-F", 
                  "(RAN) Frametype", "(RAN) Frametype I-F",
                  "(RAN) EDD", "(RAN) EDD I-F"]

schedulers_e2e = ["(E2E) M-LWDF", "(E2E) M-LWDF I-F", 
                  "(E2E) Frametype", "(E2E) Frametype I-F",
                  "(E2E) EDD", "(E2E) EDD I-F",]

markers = ["^", "<", ">", "P", "X", "*", "1", "2", "3"]


# %% Plot 0: Vary Bandwidth - Rest Default parameters - All schedulers

# Four Bandwidths
n_bw = len(BW[1:])
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_bw * n_schedulers)

stats_list = [[[0] for i in range(n_schedulers)] for j in range(n_bw)]
# ALL FRAMES
pdr = np.zeros((n_bw, n_schedulers), dtype='float') # [[0] * 7] * n_params
pdr_err = np.zeros((n_bw, n_schedulers), dtype='float')

pdr_ran = np.zeros((n_bw, n_schedulers), dtype='float')
pdr_ran_err = np.zeros((n_bw, n_schedulers), dtype='float')

pdr_e2e = np.zeros((n_bw, n_schedulers), dtype='float')
pdr_e2e_err = np.zeros((n_bw, n_schedulers), dtype='float')

# I-FRAMES
pdr_I = np.zeros((n_bw, n_schedulers), dtype='float')
pdr_I_err = np.zeros((n_bw, n_schedulers), dtype='float')

pdr_I_ran = np.zeros((n_bw, n_schedulers), dtype='float')
pdr_I_ran_err = np.zeros((n_bw, n_schedulers), dtype='float')

pdr_I_e2e = np.zeros((n_bw, n_schedulers), dtype='float')
pdr_I_e2e_err = np.zeros((n_bw, n_schedulers), dtype='float')


# Loop over all BWs
for i, bw in enumerate(BW[1:]):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{bw}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[i*n_schedulers+j] = round(temp_df.drop([2]), 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{bw}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[i*n_schedulers+4+k] = round(temp_df.drop([2]), 1)         
        
for bw in range(n_bw):     
    for s in range(n_schedulers):  
        
        pdr[bw][s] = stats_list[bw*n_schedulers+s]["Total-All"][0]        
        pdr_ran[bw][s] = stats_list[bw*n_schedulers+s]["RAN-All"][0]
        pdr_e2e[bw][s] = stats_list[bw*n_schedulers+s]["E2E-All"][0]
        pdr_err[bw][s] = stats_list[bw*n_schedulers+s]["Total-All"][1]        
        pdr_ran_err[bw][s] = stats_list[bw*n_schedulers+s]["RAN-All"][1]
        pdr_e2e_err[bw][s] = stats_list[bw*n_schedulers+s]["E2E-All"][1]
                
        pdr_I[bw][s] = stats_list[bw*n_schedulers+s]["Total-I"][0]        
        pdr_I_ran[bw][s] = stats_list[bw*n_schedulers+s]["RAN-I"][0]
        pdr_I_e2e[bw][s] = stats_list[bw*n_schedulers+s]["E2E-I"][0]
        pdr_I_err[bw][s] = stats_list[bw*n_schedulers+s]["Total-I"][1]        
        pdr_I_ran_err[bw][s] = stats_list[bw*n_schedulers+s]["RAN-I"][1]
        pdr_I_e2e_err[bw][s] = stats_list[bw*n_schedulers+s]["E2E-I"][1]
        
###############################################################################
#Plot##########################################################################
###############################################################################
bar_width = 0.35

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 20),
                       constrained_layout=True)

fig.suptitle("PDR - All Schedulers - Varying Bandwidth\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]}_" +
             f"RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load\\", 
             fontsize=20, fontweight='bold') 

plt.setp(ax, ylim=(0, np.max(pdr_I) + 2.5))

x_axis_labels = ran_labels + e2e_labels
x_axis = np.arange(len(ran_schedulers + e2e_schedulers))

y_axis = pdr
y_axis_ran = pdr_ran
y_axis_e2e = pdr_e2e

y_axis_I = pdr_I
y_axis_I_ran = pdr_I_ran
y_axis_I_e2e = pdr_I_e2e

# TOTAL PDR
pdr_100_ran = ax[0][0].bar(x_axis - bar_width/2, y_axis_ran[0], bar_width, 
                           yerr=pdr_err[0], 
                           label='Dropped by BS (All Frames')

pdr_100_e2e = ax[0][0].bar(x_axis - bar_width/2, y_axis_e2e[0], bar_width, 
                           bottom=y_axis_ran[0], yerr=pdr_err[0], 
                           label='Dropped by UE (All Frames)')

pdr_125 = ax[0][1].bar(x_axis - bar_width/2, y_axis[1], bar_width, 
                       yerr=pdr_err[1], label='All Frames')

pdr_150 = ax[1][0].bar(x_axis - bar_width/2, y_axis[2], bar_width, 
                       yerr=pdr_err[2], label='All Frames')
    
pdr_200 = ax[1][1].bar(x_axis - bar_width/2, y_axis[3], bar_width, 
                       yerr=pdr_err[3], label='All Frames')

# I-FRAME PDR
# pdr_I_100 = ax[0][0].bar(x_axis + bar_width/2, y_axis_I[0], bar_width, 
#                        yerr=pdr_err[0])

pdr_I_100_ran = ax[0][0].bar(x_axis + bar_width/2, y_axis_I_ran[0], bar_width, 
                             yerr=pdr_err[0], 
                             label='Dropped by BS (I-Frames)')

pdr_I_100_e2e = ax[0][0].bar(x_axis + bar_width/2, y_axis_I_e2e[0], bar_width, 
                             bottom=y_axis_I_ran[0], yerr=pdr_err[0], 
                             label='Dropped by UE (I-Frames)')


pdr_I_125 = ax[0][1].bar(x_axis + bar_width/2, y_axis_I[1], bar_width, 
                         yerr=pdr_err[1], label='I-Frames')

pdr_I_150 = ax[1][0].bar(x_axis + bar_width/2, y_axis_I[2], bar_width, 
                         yerr=pdr_err[2], label='I-Frames')
    
pdr_I_200 = ax[1][1].bar(x_axis + bar_width/2, y_axis_I[3], bar_width, 
                         yerr=pdr_err[3], label='I-Frames')
    
# AXIS FUNCTIONS
ax[0][0].set_title('Bandwidth 100 MHz', fontsize=16)

ax[0][0].set_xticks(x_axis)
ax[0][0].set_xticklabels(x_axis_labels)
# ax[0][0].bar_label(pdr_100, fontsize=16, padding=3)
ax[0][0].bar_label(pdr_100_ran, fontsize=14, label_type='center')
ax[0][0].bar_label(pdr_100_e2e, fontsize=14, label_type='center')
ax[0][0].bar_label(pdr_100_e2e, fontsize=16, padding=3,)
# ax[0][0].bar_label(pdr_I_100, fontsize=16, padding=3)

ax[0][0].bar_label(pdr_I_100_ran, fontsize=14, label_type='center')
ax[0][0].bar_label(pdr_I_100_e2e, fontsize=14, label_type='center')
ax[0][0].bar_label(pdr_I_100_e2e, fontsize=16, padding=3,)
ax[0][0].set_ylabel('E2E PDR [%]', fontsize=20)

ax[0][0].legend(prop={'size': 15})
# ax[0][0].legend([pdr_100_ran,pdr_I_100],['All Frames','I-Frames'], 
#                 prop={'size': 15}, loc='upper center')
# ax[0][0].add_artist(leg1)

ax[0][1].set_title('Bandwidth 125 MHz', fontsize=16)
ax[0][1].set_xticks(x_axis)
ax[0][1].set_xticklabels(x_axis_labels)
ax[0][1].bar_label(pdr_125, fontsize=16, padding=3)
ax[0][1].bar_label(pdr_I_125, fontsize=16, padding=3)
ax[0][1].legend(prop={'size': 15})

ax[1][0].set_title('Bandwidth 150 MHz', fontsize=16)
ax[1][0].set_xticks(x_axis)
ax[1][0].set_xticklabels(x_axis_labels)
ax[1][0].bar_label(pdr_150, fontsize=16, padding=3)
ax[1][0].bar_label(pdr_I_150, fontsize=16, padding=3)
ax[1][0].set_ylabel('E2E PDR [%]', fontsize=20)
ax[1][0].legend(prop={'size': 15})


ax[1][1].set_title('Bandwidth 200 MHz', fontsize=16)
ax[1][1].set_xticks(x_axis)
ax[1][1].set_xticklabels(x_axis_labels)
ax[1][1].bar_label(pdr_200, fontsize=16, padding=3)
ax[1][1].bar_label(pdr_I_200, fontsize=16, padding=3)
ax[1][1].legend(prop={'size': 15})

# fig.constrained_layout()
plt.show()

save_name = "Vary Bandwidth - ALL Schedulers.png"
fig.savefig(save_plots_path + save_name, dpi=200)
print(f'Figure saved: {save_name}')
# Create the legend
# fig.legend([l1, l2, l3, l4],     # The line objects
#            labels=line_labels,   # The labels for each line
#            loc="center right",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            title="Legend Title"  # Title for the legend
#            )

# %% Plot 0a: Vary Bandwidth - Rest Default parameters - All schedulers
# Variation 

# Four Bandwidths
n_bw = len(BW[1:])
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_bw * n_schedulers)

stats_list = [[0]] * n_params

# Arrays with corresponding values to plot

# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_bw, n_schedulers), dtype='float')
pdr_I = np.zeros((2, n_bw, n_schedulers), dtype='float')

pdr_ran = np.zeros((2, n_bw, n_schedulers), dtype='float')
pdr_ran_I = np.zeros((2, n_bw, n_schedulers), dtype='float')

pdr_e2e = np.zeros((2, n_bw, n_schedulers), dtype='float')
pdr_e2e_I = np.zeros((2, n_bw, n_schedulers), dtype='float')


# Loop over all BWs
for i, bw in enumerate(BW[1:]):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{bw}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[i*n_schedulers+j] = round(temp_df.drop([2]), 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{bw}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[i*n_schedulers+4+k] = round(temp_df.drop([2]), 1)         
        

for s in range(n_schedulers):  
    for bw in range(n_bw):  

        pdr[0][bw][s] = stats_list[bw*n_schedulers+s]["Total-All"][0]        
        pdr[1][bw][s] = stats_list[bw*n_schedulers+s]["Total-All"][1]       
        pdr_I[0][bw][s] = stats_list[bw*n_schedulers+s]["Total-I"][0]        
        pdr_I[1][bw][s] = stats_list[bw*n_schedulers+s]["Total-I"][1] 
        
        pdr_ran[0][bw][s] = stats_list[bw*n_schedulers+s]["RAN-All"][0]        
        pdr_ran[1][bw][s] = stats_list[bw*n_schedulers+s]["RAN-All"][1] 
        pdr_ran_I[0][bw][s] = stats_list[bw*n_schedulers+s]["RAN-I"][0]        
        pdr_ran_I[1][bw][s] = stats_list[bw*n_schedulers+s]["RAN-I"][1] 
        
        pdr_e2e[0][bw][s] = stats_list[bw*n_schedulers+s]["E2E-All"][0]        
        pdr_e2e[1][bw][s] = stats_list[bw*n_schedulers+s]["E2E-All"][1] 
        pdr_e2e_I[0][bw][s] = stats_list[bw*n_schedulers+s]["E2E-I"][0]        
        pdr_e2e_I[1][bw][s] = stats_list[bw*n_schedulers+s]["E2E-I"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################
bar_width = 0.2

fig, ax = plt.subplots(figsize=(30, 15), constrained_layout=True)

fig.suptitle("PDR - All Schedulers - Varying Bandwidth (No PF)\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]}_" +
             f"RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load\\", 
             fontsize=20, fontweight='bold') 

x_axis_labels = ran_labels[:] + e2e_labels
x_axis = np.arange(len(ran_schedulers[:] + e2e_schedulers))


ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)

p100_ran = ax.bar(x_axis - bar_width*1.5, pdr_ran[0][0], bar_width, 
              yerr=pdr[1][0], label='BW 100 MHz - Dropped at BS')
p100_e2e = ax.bar(x_axis - bar_width*1.5, pdr_e2e[0][0], bar_width, 
              bottom=pdr_ran[0][0],  yerr=pdr[1][0], 
              label='BW 100 MHz - Dropped at UE')

# p125 = ax.bar(x_axis - bar_width*1.5 + bar_width, pdr[0][1][1:], bar_width, 
#             yerr=pdr[1][1][1:], label='BW 125 MHz')

p125_ran = ax.bar(x_axis - bar_width*1.5 + bar_width, pdr_ran[0][1], bar_width, 
              yerr=pdr[1][1], label='BW 125 MHz - Dropped at BS')
p125_e2e = ax.bar(x_axis - bar_width*1.5 + bar_width, pdr_e2e[0][1], bar_width, 
              bottom=pdr_ran[0][1],  yerr=pdr[1][1], 
              label='BW 125 MHz - Dropped at UE')

# p150 = ax.bar(x_axis - bar_width*1.5 + bar_width*2, pdr[0][2][1:], bar_width, 
#             yerr=pdr[1][2][1:], label='BW 150 MHz')
p150_ran = ax.bar(x_axis - bar_width*1.5 + bar_width*2, pdr_ran[0][2], bar_width, 
              yerr=pdr[1][2], label='BW 150 MHz - Dropped at BS')
p150_e2e = ax.bar(x_axis - bar_width*1.5 + bar_width*2, pdr_e2e[0][2], bar_width, 
              bottom=pdr_ran[0][2],  yerr=pdr[1][2], 
              label='BW 150 MHz - Dropped at UE')

# p200 = ax.bar(x_axis - bar_width*1.5 + bar_width*3, pdr[0][3][1:], bar_width, 
#             yerr=pdr[1][3][1:], label='BW 200 MHz')
p200_ran = ax.bar(x_axis - bar_width*1.5 + bar_width*3, pdr_ran[0][3], bar_width, 
              yerr=pdr[1][3], label='BW 200 MHz - Dropped at BS')
p200_e2e = ax.bar(x_axis - bar_width*1.5 + bar_width*3, pdr_e2e[0][3], bar_width, 
              bottom=pdr_ran[0][3],  yerr=pdr[1][2], 
              label='BW 200 MHz - Dropped at UE')


ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel('Schedulers', fontsize=20)

# # ax.bar_label(p100, fontsize=16, padding=3)
# ax.bar_label(p100_ran, fontsize=14, label_type='center')
# ax.bar_label(p100_e2e, fontsize=14, label_type='center')
# ax.bar_label(p100_e2e, fontsize=16, padding=3,)

# # ax.bar_label(p125, fontsize=16, padding=3)
# ax.bar_label(p125_ran, fontsize=14, label_type='center')
# ax.bar_label(p125_e2e, fontsize=14, label_type='center')
# ax.bar_label(p125_e2e, fontsize=16, padding=3,)

# # ax.bar_label(p150, fontsize=16, padding=3)
# ax.bar_label(p150_ran, fontsize=14, label_type='center')
# ax.bar_label(p150_e2e, fontsize=14, label_type='center')
# ax.bar_label(p150_e2e, fontsize=16, padding=3,)

# # ax.bar_label(p200, fontsize=16, padding=3)
# ax.bar_label(p200_ran, fontsize=14, label_type='center')
# ax.bar_label(p200_e2e, fontsize=14, label_type='center')
# ax.bar_label(p200_e2e, fontsize=16, padding=3,)

ax.legend(prop={'size': 15})

save_name = "Vary Bandwidth - ALL Schedulers - V2.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')

# %% Plot 0b: Vary Bandwidth - Rest Default parameters - All schedulers
# Variation: Every grouped bar one BW - Every bar in group different scheduler

# Four Bandwidths
n_bw = len(BW[1:])
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_bw * n_schedulers)

stats_list = [[[0] for i in range(n_bw)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_bw), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_bw), dtype='float')
pdr_ran = np.zeros((2, n_schedulers, n_bw), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_bw), dtype='float')
pdr_e2e = np.zeros((2, n_schedulers, n_bw), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_bw), dtype='float')


# Loop over all BWs
for i, bw in enumerate(BW[1:]):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{bw}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df.drop([2]), 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{bw}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df.drop([2]), 1)         
        

for s in range(n_schedulers):  
    for bw in range(n_bw):  

        pdr[0][s][bw] = stats_list[s][bw]["Total-All"][0]        
        pdr[1][s][bw] = stats_list[s][bw]["Total-All"][1]       
        pdr_I[0][s][bw] = stats_list[s][bw]["Total-I"][0]        
        pdr_I[1][s][bw] = stats_list[s][bw]["Total-I"][1] 
        
        pdr_ran[0][s][bw] = stats_list[s][bw]["RAN-All"][0]        
        pdr_ran[1][s][bw] = stats_list[s][bw]["RAN-All"][1] 
        pdr_ran_I[0][s][bw] = stats_list[s][bw]["RAN-I"][0]        
        pdr_ran_I[1][s][bw] = stats_list[s][bw]["RAN-I"][1] 
        
        pdr_e2e[0][s][bw] = stats_list[s][bw]["E2E-All"][0]        
        pdr_e2e[1][s][bw] = stats_list[s][bw]["E2E-All"][1] 
        pdr_e2e_I[0][s][bw] = stats_list[s][bw]["E2E-I"][0]        
        pdr_e2e_I[1][s][bw] = stats_list[s][bw]["E2E-I"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################
fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle("PDR - All Schedulers - Varying Bandwidth (No PF)\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]}_" +
             f"RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load\\", 
             fontsize=20, fontweight='bold') 

x_axis_labels = ['100 MHz', '125 MHz', '150 MHz', '200 MHz'] 
bar_labels = ran_labels + e2e_labels
x_axis = np.arange(len(BW[1:])) # ran_schedulers[:] + e2e_schedulers))
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11


plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    print(i)
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr_ran[0][i], bar_width, 
                          yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
                          yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel('Bandwidths', fontsize=20)
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()


save_name = "Vary Bandwidth - ALL Schedulers - V3.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')

# %% Plot 1: Vary RAN budget - Rest Default parameters

# All lats 
par_to_vary = "RAN Latencies"
plot_param = ran_lat[1:] # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all RAN Lat Budgets, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{par}_Offset-{offset[0]}_UE{ues[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{BW[0]}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]} - " +
             f"BW{BW[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load - {ues[0]}UEs", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['10ms', '20ms', '30ms', '40ms', '50ms', '60ms', '70ms', '80ms', 
                 '90ms', '100ms'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')

# %% Plot 2: Vary Bandwidth - Rest Default parameters

# All Bandwidths
par_to_vary = "Channel Bandwidths"
plot_param = BW[1:] # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all BWs, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{par}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{par}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]} - " +
             f"RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load - {ues[0]}UEs", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['100MHz', '125MHz', '150MHz', '200MHz'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')

# %% Plot 3: Vary APP Bitrate - Rest Default parameters

# All Bitrates 
par_to_vary = "Application Bitrate"
plot_param = bitrate[1:] # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all Bitrates, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{par}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{par}_" + \
           f"{dispersion[0]}_BW-{BW[0]}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - Bandwidth {BW[0]} - " +
             f"RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load - {ues[0]}UEs", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['50Mbps', '100Mbps', '150Mbps', '200Mbps'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')


# %% Plot 4: Vary Queues - Rest Default parameters

# All Queues 
par_to_vary = "Network Hops"
plot_param = queues[1:] # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all Queues, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
            f"{par}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{BW[0]}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
           f"{par}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]} - " +
             f"BW{BW[0]}MHz - RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{bg_load[0]}% Load - {ues[0]}UEs", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['5 Hops (National)', '10 Hops (Continental)', 
                 '15 Hops (Intercontinental)'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')


# %% Plot 5: Vary Network Load - Rest Default parameters

# All Loads 
par_to_vary = "Network Hops - Background Traffic Load"
plot_param = bg_load[1:] # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all Loads, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
            f"{queues[0]}Q - {par}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{BW[0]}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
           f"{queues[0]}Q - {par}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]} - " +
             f"BW{BW[0]}MHz - RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {ues[0]}UEs", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['50%', '70%', '85%'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')


# %% Plot 6: Vary E2E LAT Budget - Rest Default parameters

# All lats 
par_to_vary = "E2E Latency Budgets"
plot_param = e2e_lat[1:] # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all E2E Lat Budgets, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{par}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}_UE{ues[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{par}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{BW[0]}_E2E-LAT-{par}_Offset-{offset[0]}_UE{ues[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"APP-Bitrate{bitrate[0]} - BW{BW[0]}MHz - " +
             f"RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load - {ues[0]}UEs", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['25ms', '50ms', '100ms'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')


# %% Plot 7: Vary #UEs - Rest Default parameters

# All lats 
par_to_vary = "#UEs"
plot_param = ues[1:] # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all UEs, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{ran_lat[0]}_Offset-{offset[0]}_UE{par}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{BW[0]}_E2E-LAT-{e2e_lat[0]}_Offset-{offset[0]}_UE{par}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]} - " +
             f"BW {BW[0]}MHz - RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['1UE', '2UEs', '4UEs', '6UEs', '8UEs'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')


# %% Plot 8: Vary UE Offset - Rest Default parameters

# All lats 
par_to_vary = "UE Offset"
plot_param = offset # TODO: Vary per plot!!!
n_plot_param = len(plot_param)
n_schedulers = len(ran_schedulers + e2e_schedulers)
n_params = int(n_plot_param * n_schedulers)


stats_list = [[[0] for i in range(n_plot_param)] for j in range(n_schedulers)]

# Arrays with corresponding values to plot
# ALL FRAMES (PDR & ERR)
pdr = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_ran = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_ran_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')

pdr_e2e = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_I = np.zeros((2, n_schedulers, n_plot_param), dtype='float')
pdr_e2e_P = np.zeros((2, n_schedulers, n_plot_param), dtype='float')


# Loop over all Offsets, rest default!!!!!
for i, par in enumerate(plot_param):
    # Loop for all RAN-based schedulers
    for j, scheduler in enumerate(ran_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{par}_Offset-{par}_UE{ues[0]}\\" + \
            f"{queues[0]}Q - {bg_load[0]}% Load\\"    
        stats_folder = stats_path + parameters    
        temp_df = pd.read_csv(stats_folder + scheduler +".csv")
        stats_list[j][i]= round(temp_df, 1) 
                
    # Loop for all E2E-based schedulers
    for k, scheduler in enumerate(e2e_schedulers):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
           f"{dispersion[0]}_BW-{BW[0]}_E2E-LAT-{e2e_lat[0]}_Offset-{par}_UE{ues[0]}\\" + \
           f"{queues[0]}Q - {bg_load[0]}% Load\\"
        stats_folder = stats_path + parameters
        temp_df = pd.read_csv(stats_folder + scheduler + ".csv")
        stats_list[k+len(ran_schedulers)][i] = round(temp_df, 1)         
        
for s in range(n_schedulers):  
    for par in range(plot_param):  

        pdr[0][s][par] = stats_list[s][par]["Total-All"][0]        
        pdr[1][s][par] = stats_list[s][par]["Total-All"][1]       
        pdr_I[0][s][par] = stats_list[s][par]["Total-I"][0]        
        pdr_I[1][s][par] = stats_list[s][par]["Total-I"][1] 
        pdr_P[0][s][par] = stats_list[s][par]["Total-P"][0]        
        pdr_P[1][s][par] = stats_list[s][par]["Total-P"][1] 
        
        pdr_ran[0][s][par] = stats_list[s][par]["RAN-All"][0]        
        pdr_ran[1][s][par] = stats_list[s][par]["RAN-All"][1] 
        pdr_ran_I[0][s][par] = stats_list[s][par]["RAN-I"][0]        
        pdr_ran_I[1][s][par] = stats_list[s][par]["RAN-I"][1] 
        pdr_ran_P[0][s][par] = stats_list[s][par]["RAN-P"][0]        
        pdr_ran_P[1][s][par] = stats_list[s][par]["RAN-P"][1] 
        
        pdr_e2e[0][s][par] = stats_list[s][par]["E2E-All"][0]        
        pdr_e2e[1][s][par] = stats_list[s][par]["E2E-All"][1] 
        pdr_e2e_I[0][s][par] = stats_list[s][par]["E2E-I"][0]        
        pdr_e2e_I[1][s][par] = stats_list[s][par]["E2E-I"][1]  
        pdr_e2e_P[0][s][par] = stats_list[s][par]["E2E-P"][0]        
        pdr_e2e_P[1][s][par] = stats_list[s][par]["E2E-P"][1] 

###############################################################################
#Plot##########################################################################
###############################################################################

fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
fig.suptitle(f"PDR - All Schedulers - Varying {par_to_vary}\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]} - " +
             f"BW {BW[0]}MHz - RAN-LAT{ran_lat[0]} - Offset-{offset[0]} - " + 
             f"{queues[0]}Q - {bg_load[0]}% Load - {ues[0]}UEs", 
             fontsize=20, fontweight='bold') 
# TODO: Par to vary: LAT!!!!!
x_axis_labels = ['Maximally Asynchronized', 'Perfectly Synchronized'] 

bar_labels = ran_labels + e2e_labels
x_axis = np.arange(n_plot_param) 
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_labels, fontsize=16)
bar_width = 0.11

plots_tot = [[0] for i in range(n_schedulers)]
plots_ran = [[0] for i in range(n_schedulers)]
plots_e2e = [[0] for i in range(n_schedulers)]

for i in range(n_schedulers):
    plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
                          pdr[0][i], bar_width, yerr=pdr[1][i], 
                          label=f'{bar_labels[i]}')
    
    # plots_ran[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_ran[0][i], bar_width, 
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]}- Dropped at BS')
    # plots_e2e[i] = ax.bar(x_axis - bar_width * 3 + bar_width * i, 
    #                       pdr_e2e[0][i], bar_width, bottom=pdr_ran[0][i],  
    #                       yerr=pdr[1][i], label=f'{bar_labels[i]} - Dropped at UE')

ax.set_ylabel('E2E PDR [%]', fontsize=20)
ax.set_xlabel(f"{par_to_vary}", fontsize=20) # TODO: Paratermeter to vary!!!!!
ax.legend(bar_labels, title="Network Hops & Load", title_fontsize=16, 
          loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, 
          fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()

save_name = f"Vary {par_to_vary} - ALL Schedulers.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')
