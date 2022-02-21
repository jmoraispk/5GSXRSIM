# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:14:05 2022

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
ran_schedulers = ["PF", "M-LWDF", "EDD", "Frametype"]   
# ran_schedulers = ["M-LWDF", "EDD", "Frametype"]   
ran_labels = ["PF", "M-LWDF (RAN)", "EDD (RAN)", "Frametype (RAN)"]   

e2e_schedulers = ["M-LWDF", "EDD", "Frametype"] 
e2e_labels = ["M-LWDF (E2E)", "EDD (E2E)", "Frametype (E2E)"] 


queue_params = [ "5Q - 50.0% Load",  "5Q - 70.0% Load",  # "5Q - 85.0% Load",
                "10Q - 70.0% Load"] # "10Q - 70.0% Load", "10Q - 85.0% Load",
                # "15Q - 50.0% Load", "15Q - 70.0% Load", "15Q - 85.0% Load"]


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



def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='center')
        

# %% Plot 1: Tune Default Parameters - E2E-100 - All Queues and RAN LAT
# Only NON-CL schedulers (PF - RAN-M-LWDF - RAN-EDD)

n_ran = len(ran_lat[1:])
n_queue_params = len(queue_params)
n_params = int(n_ran * n_queue_params)

stats_list_mlwdf = [[[0] for i in range(n_ran)] for j in range(n_queue_params)]
stats_list_edd = [[[0] for i in range(n_ran)] for j in range(n_queue_params)]

# Fill up arrays with corresponding values to plot
total_pdr_pf = np.zeros((n_queue_params, n_ran), dtype='float') 
total_pdr_pf_err = np.zeros((n_queue_params, n_ran), dtype='float') 

total_pdr_mlwdf = np.zeros((n_queue_params, n_ran), dtype='float')
total_pdr_mlwdf_err = np.zeros((n_queue_params, n_ran), dtype='float')

total_pdr_edd = np.zeros((n_queue_params, n_ran), dtype='float')
total_pdr_edd_err = np.zeros((n_queue_params, n_ran), dtype='float')

# Loop for all RAN-based schedulers for 9 queue params and 10 RAN-LATs
for i, q in enumerate(queue_params):
    for j, lat in enumerate(ran_lat[1:]):
        parameters = f"E2E-{e2e_lat[0]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{lat}_" + \
            f"Offset-{offset[0]}\\{q}\\"  
        stats_folder = stats_path + parameters         
                
        # temp_df = pd.read_csv(stats_folder + ran_schedulers[1] +".csv")
        # stats_list_mlwdf[i*10+j] = round(temp_df.drop([2]), 1)       
        temp_df = pd.read_csv(stats_folder + ran_schedulers[2] +".csv")
        stats_list_edd[i][j] = round(temp_df.drop([2]), 1)
                
    
# HARDCODE - ITS ONLY 4/8 VALUES ANYWAY
total_pdr_pf[0] = 10.1   # 5Q - 50%
total_pdr_pf[1] = 15.1  # 5Q - 70%
total_pdr_pf[2] = 20.1  # 5Q - 85%
# total_pdr_pf[3] = 25.1   # 10Q - 50%
# total_pdr_pf[4] = 30.1    # 10Q - 70%
# total_pdr_pf[5] = 35.1  # 10Q - 85%
# total_pdr_pf[6] = 40.1  # 15Q - 50%
# total_pdr_pf[7] = 45.1  # 15Q - 70%
# total_pdr_pf[8] = 50.1   # 15Q - 85%

for q in range(len(queue_params)):
    for lat in range(len(ran_lat[1:])):
        total_pdr_mlwdf[q][lat]     = q * 10 + lat # stats_list_mlwdf[q][lat]["Total-All"][0]
        total_pdr_edd[q][lat]     = stats_list_edd[q][lat]["Total-All"][0]
        
        # total_pdr_mlwdf_err[q][lat] = q * 10 + lat # stats_list_mlwdf[q][lat]["Total-All"][1]
        # total_pdr_edd_err[q][lat] = q * 10 + lat # stats_list_edd[q][lat]["Total-All"][1]


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

fig.suptitle("Total PDR - Non Cross-layer Schedulers\n" + 
             f"E2E-Budget {e2e_lat[0]}ms - APP-Bitrate{bitrate[0]} - " +
             f"Offset-MAX - {ues[0]} UEs", 
             fontsize=20, fontweight='bold') 
plt.setp(ax, ylim=(0, 20))

x_axis = ran_lat[1:]

ax[0].set_ylabel('E2E PDR [%]', fontsize=20)

ax[0].set_title('RAN - EDD Scheduler', fontsize=16)
ax[0].set_xticks(x_axis)
ax[0].set_xlabel('RAN Latency Budget [ms]', fontsize=12)
for i in range(n_queue_params):
    ax[0].errorbar(x_axis, total_pdr_edd[i], # yerr = total_pdr_edd_err[0],
                   marker=markers[i], markersize=12)    
    
ax[0].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=3, fancybox=True, shadow=True, prop={'size': 15})


ax[1].set_title('RAN - M-LWDF Scheduler', fontsize=16)
ax[1].set_xticks(x_axis)
ax[1].set_xlabel('RAN Latency Budget [ms]', fontsize=12)
for i in range(n_queue_params):
    ax[1].errorbar(x_axis, total_pdr_mlwdf[i], # yerr = total_pdr_mlwdf_err[0],
                   marker=markers[i], markersize=12)    
ax[1].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper left', prop={'size': 15})
    
ax[1].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=3, fancybox=True, shadow=True, prop={'size': 15})

ax[2].set_title('RAN - PF Scheduler', fontsize=16)
ax[2].set_xticks(x_axis)
ax[2].set_xlabel('RAN Latency Budget [ms]', fontsize=12)
for i in range(n_queue_params):
    ax[2].errorbar(x_axis, total_pdr_pf[i], # yerr = total_pdr_mlwdf_err[0],
                   marker=markers[i], markersize=12)    
ax[2].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper left', prop={'size': 15})
    
ax[2].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=3, fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()
# plt.show()

save_name = f"Tuning Default Values - All Queues - {ues[0]} UEs.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')


# %% Plot 2: Tune Default Parameters - E2E-50 - All Queues and RAN LAT
# Only NON-CL schedulers (PF - RAN-M-LWDF - RAN-EDD)

n_ran = len(ran_lat[1:6])
n_queue_params = len(queue_params)
n_params = int(n_ran * n_queue_params)

stats_list_mlwdf = [[[0] for i in range(n_ran)] for j in range(n_queue_params)]
stats_list_edd = [[[0] for i in range(n_ran)] for j in range(n_queue_params)]

# Fill up arrays with corresponding values to plot
total_pdr_pf = np.zeros((n_queue_params, n_ran), dtype='float') 
total_pdr_pf_err = np.zeros((n_queue_params, n_ran), dtype='float') 

total_pdr_mlwdf = np.zeros((n_queue_params, n_ran), dtype='float')
total_pdr_mlwdf_err = np.zeros((n_queue_params, n_ran), dtype='float')

total_pdr_edd = np.zeros((n_queue_params, n_ran), dtype='float')
total_pdr_edd_err = np.zeros((n_queue_params, n_ran), dtype='float')

# Loop for all RAN-based schedulers for 9 queue params and 10 RAN-LATs
for i, q in enumerate(queue_params):
    for j, lat in enumerate(ran_lat[1:6]):
        parameters = f"E2E-{e2e_lat[2]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{lat}_" + \
            f"Offset-{offset[0]}\\{q}\\"  
        stats_folder = stats_path + parameters         
                
        # temp_df = pd.read_csv(stats_folder + ran_schedulers[1] +".csv")
        # stats_list_mlwdf[i*10+j] = round(temp_df.drop([2]), 1)       
        temp_df = pd.read_csv(stats_folder + ran_schedulers[2] +".csv")
        stats_list_edd[i][j] = round(temp_df.drop([2]), 1)
                
    
# HARDCODE - ITS ONLY 4/8 VALUES ANYWAY
total_pdr_pf[0] = 10.1   # 5Q - 50%
total_pdr_pf[1] = 15.1  # 5Q - 70%
total_pdr_pf[2] = 20.1  # 5Q - 85%
# total_pdr_pf[3] = 25.1   # 10Q - 50%
# total_pdr_pf[4] = 30.1    # 10Q - 70%
# total_pdr_pf[5] = 35.1  # 10Q - 85%
# total_pdr_pf[6] = 40.1  # 15Q - 50%
# total_pdr_pf[7] = 45.1  # 15Q - 70%
# total_pdr_pf[8] = 50.1   # 15Q - 85%

for q in range(len(queue_params)):
    for lat in range(len(ran_lat[1:6])):
        total_pdr_mlwdf[q][lat]     = q * 10 + lat # stats_list_mlwdf[q][lat]["Total-All"][0]
        total_pdr_edd[q][lat]     = stats_list_edd[q][lat]["Total-All"][0]
        
        # total_pdr_mlwdf_err[q][lat] = q * 10 + lat # stats_list_mlwdf[q][lat]["Total-All"][1]
        # total_pdr_edd_err[q][lat] = q * 10 + lat # stats_list_edd[q][lat]["Total-All"][1]


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

fig.suptitle("Total PDR - Non Cross-layer Schedulers\n" + 
             f"E2E-Budget {e2e_lat[2]}ms - APP-Bitrate{bitrate[0]} - " +
             f"Offset-MAX - {ues[0]} UEs", 
             fontsize=20, fontweight='bold') 
plt.setp(ax, ylim=(0, 20))

x_axis = ran_lat[1:6]

ax[0].set_ylabel('E2E PDR [%]', fontsize=20)

ax[0].set_title('RAN - EDD Scheduler', fontsize=16)
ax[0].set_xticks(x_axis)
ax[0].set_xlabel('RAN Latency Budget [ms]', fontsize=12)    
for i in range(n_queue_params):
    ax[0].errorbar(x_axis, total_pdr_edd[i], # yerr = total_pdr_edd_err[0],
                   marker=markers[i], markersize=12)    

ax[1].set_title('RAN - M-LWDF Scheduler', fontsize=16)
ax[1].set_xticks(x_axis)
ax[1].set_xlabel('RAN Latency Budget [ms]', fontsize=12)
for i in range(n_queue_params):
    ax[1].errorbar(x_axis, total_pdr_mlwdf[i], # yerr = total_pdr_mlwdf_err[0],
                   marker=markers[i], markersize=12)    
    
ax[2].set_title('RAN - PF Scheduler', fontsize=16)
ax[2].set_xticks(x_axis)
ax[2].set_xlabel('RAN Latency Budget [ms]', fontsize=12)
for i in range(n_queue_params):
    ax[2].errorbar(x_axis, total_pdr_pf[i], # yerr = total_pdr_mlwdf_err[0],
                   marker=markers[i], markersize=12)    
    
for i in range(3):
    ax[i].legend(queue_params, title="Network Hops & Load", title_fontsize=16, 
                 loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, 
                 fancybox=True, shadow=True, prop={'size': 15})

fig.tight_layout()
# plt.show()

save_name = f"Tuning Default Values - All Queues - {ues[0]} UEs.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')


# %% Plot 3: Tune Default Parameters - E2E-25 - All Queues and RAN LAT
# Only NON-CL schedulers (PF - RAN-M-LWDF - RAN-EDD)

n_ran = len(ran_lat[1:4])
n_queue_params = len(queue_params)
n_params = int(n_ran * n_queue_params)

stats_list_mlwdf = [[[0] for i in range(n_ran)] for j in range(n_queue_params)]
stats_list_edd = [[[0] for i in range(n_ran)] for j in range(n_queue_params)]

# Fill up arrays with corresponding values to plot
total_pdr_pf = np.zeros((n_queue_params, n_ran), dtype='float') 
total_pdr_pf_err = np.zeros((n_queue_params, n_ran), dtype='float') 

total_pdr_mlwdf = np.zeros((n_queue_params, n_ran), dtype='float')
total_pdr_mlwdf_err = np.zeros((n_queue_params, n_ran), dtype='float')

total_pdr_edd = np.zeros((n_queue_params, n_ran), dtype='float')
total_pdr_edd_err = np.zeros((n_queue_params, n_ran), dtype='float')

# Loop for all RAN-based schedulers for 9 queue params and 10 RAN-LATs
for i, q in enumerate(queue_params):
    for j, lat in enumerate(ran_lat[1:4]):
        parameters = f"E2E-{e2e_lat[1]}ms\\APP{bitrate[0]}_" + \
            f"{dispersion[0]}_BW-{BW[0]}_RAN-LAT-{lat}_" + \
            f"Offset-{offset[0]}\\{q}\\"  
        stats_folder = stats_path + parameters         
                
        # temp_df = pd.read_csv(stats_folder + ran_schedulers[1] +".csv")
        # stats_list_mlwdf[i*10+j] = round(temp_df.drop([2]), 1)       
        temp_df = pd.read_csv(stats_folder + ran_schedulers[2] +".csv")
        stats_list_edd[i][j] = round(temp_df.drop([2]), 1)
                
    
# HARDCODE - ITS ONLY 4/8 VALUES ANYWAY
total_pdr_pf[0] = 10.1   # 5Q - 50%
total_pdr_pf[1] = 15.1  # 5Q - 70%
total_pdr_pf[2] = 20.1  # 5Q - 85%
# total_pdr_pf[3] = 25.1   # 10Q - 50%
# total_pdr_pf[4] = 30.1    # 10Q - 70%
# total_pdr_pf[5] = 35.1  # 10Q - 85%
# total_pdr_pf[6] = 40.1  # 15Q - 50%
# total_pdr_pf[7] = 45.1  # 15Q - 70%
# total_pdr_pf[8] = 50.1   # 15Q - 85%

for q in range(len(queue_params)):
    for lat in range(len(ran_lat[1:4])):
        total_pdr_mlwdf[q][lat]     = q * 10 + lat # stats_list_mlwdf[q][lat]["Total-All"][0]
        total_pdr_edd[q][lat]     = stats_list_edd[q][lat]["Total-All"][0]
        
        # total_pdr_mlwdf_err[q][lat] = q * 10 + lat # stats_list_mlwdf[q][lat]["Total-All"][1]
        # total_pdr_edd_err[q][lat] = q * 10 + lat # stats_list_edd[q][lat]["Total-All"][1]


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

fig.suptitle("Total PDR - Non Cross-layer Schedulers\n" + 
             f"E2E-Budget {e2e_lat[1]}ms - APP-Bitrate{bitrate[0]} - " +
             f"Offset-MAX - {ues[0]} UEs", 
             fontsize=20, fontweight='bold') 
plt.setp(ax, ylim=(0, 20))

x_axis = ran_lat[1:4]

ax[0].set_ylabel('E2E PDR [%]', fontsize=20)

ax[0].set_title('RAN - EDD Scheduler', fontsize=16)
ax[0].set_xticks(x_axis)
ax[0].set_xlabel('RAN Latency Budget [ms]', fontsize=12)
for i in range(n_queue_params):
    ax[0].errorbar(x_axis, total_pdr_edd[i], # yerr = total_pdr_edd_err[0],
                   marker=markers[i], markersize=12)    
    
ax[0].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=3, fancybox=True, shadow=True, prop={'size': 15})


ax[1].set_title('RAN - M-LWDF Scheduler', fontsize=16)
ax[1].set_xticks(x_axis)
ax[1].set_xlabel('RAN Latency Budget [ms]', fontsize=12)
for i in range(n_queue_params):
    ax[1].errorbar(x_axis, total_pdr_mlwdf[i], # yerr = total_pdr_mlwdf_err[0],
                   marker=markers[i], markersize=12)    
    
ax[1].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=3, fancybox=True, shadow=True, prop={'size': 15})

ax[2].set_title('RAN - PF Scheduler', fontsize=16)
ax[2].set_xticks(x_axis)
ax[2].set_xlabel('RAN Latency Budget [ms]', fontsize=12)    
ax[2].legend(queue_params, title="Network Hops & Load",
             title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=3, fancybox=True, shadow=True, prop={'size': 15})
for i in range(n_queue_params):
    ax[2].errorbar(x_axis, total_pdr_pf[i], # yerr = total_pdr_mlwdf_err[0],
                   marker=markers[i], markersize=12)    
    
fig.tight_layout()
# plt.show()

save_name = f"Tuning Default Values - All Queues - {ues[0]} UEs.png"
# fig.savefig(save_plots_path + save_name, dpi=200)
# print(f'Figure saved: {save_name}')